from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .policy_value_model import PolicyValueModel


Move = int


@dataclass
class MCTSNode:
    """A node in the search tree.

    Q is stored from the viewpoint of the player to move at this node.
    """

    prior: float
    parent: MCTSNode | None = None
    children: dict[Move, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def is_leaf(self) -> bool:
        return not self.children

    def expand(self, action_priors: Iterable[tuple[Move, float]]) -> None:
        for move, prior in action_priors:
            if move not in self.children:
                self.children[move] = MCTSNode(prior=float(prior), parent=self)

    def select(self, c_puct: float) -> tuple[Move, MCTSNode]:
        return max(
            self.children.items(),
            key=lambda item: item[1].ucb_score(c_puct),
        )

    def ucb_score(self, c_puct: float) -> float:
        if self.parent is None:
            return self.q

        exploration = (
            c_puct
            * self.prior
            * sqrt(self.parent.visits)
            / (1 + self.visits)
        )
        # Child q is from the child player's viewpoint, so the parent sees -q.
        return -self.q + exploration

    def update(self, leaf_value: float) -> None:
        self.visits += 1
        self.value_sum += leaf_value

    def update_recursive(self, leaf_value: float) -> None:
        if self.parent is not None:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS:
    """Monte Carlo Tree Search for Gomoku guided by PolicyValueModel.

    Board format:
        A 2D array with shape (height, width). Empty cells are 0. Stones can be
        encoded as either {1, -1} or {1, 2}; pass the side to move as
        current_player using the same encoding.

    Move format:
        An integer action index in row-major order: row * width + col.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        board_height: int = 15,
        board_width: int = 15,
        n_playout: int = 500,
        c_puct: float = 5.0,
        player_values: Sequence[int] = (1, -1),
        device: torch.device | str | None = None,
        rules: GomokuRules | None = None,
    ) -> None:
        self.model = model
        self.rules = rules or GomokuRules(
            board_height=board_height,
            board_width=board_width,
            player_values=player_values,
        )
        self.board_height = self.rules.board_height
        self.board_width = self.rules.board_width
        self.board_size = self.rules.board_size
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.device = torch.device(device) if device is not None else self._model_device()
        self.root = MCTSNode(prior=1.0)

    def select(self, node: MCTSNode, board: np.ndarray, current_player: int) -> tuple[MCTSNode, np.ndarray, int]:
        """Select a leaf node by repeatedly maximizing PUCT."""
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            board = self.rules.next_board(board, move, current_player)
            current_player = self.rules.next_player(current_player)
        return node, board, current_player

    def expand(self, node: MCTSNode, board: np.ndarray, current_player: int) -> float:
        """Expand a leaf using model policy and return its value estimate."""
        ended, winner = self.rules.game_end(board)
        if ended:
            if winner == 0:
                return 0.0
            return 1.0 if winner == current_player else -1.0

        action_priors, value = self._policy_value(board, current_player)
        node.expand(action_priors)
        return value

    def simulate(self, board: np.ndarray, current_player: int) -> None:
        """Run one model-guided playout from root and backpropagate the value."""
        node, leaf_board, leaf_player = self.select(self.root, board, current_player)
        leaf_value = self.expand(node, leaf_board, leaf_player)
        node.update_recursive(leaf_value)

    def get_action_probs(
        self,
        board: np.ndarray,
        current_player: int,
        temp: float = 1e-3,
    ) -> tuple[list[Move], np.ndarray]:
        """Run MCTS and return legal moves with their visit-count probabilities."""
        board = self.rules.as_board(board)
        for _ in range(self.n_playout):
            self.simulate(board, current_player)

        if not self.root.children:
            return [], np.array([], dtype=np.float32)

        moves, visits = zip(*((move, child.visits) for move, child in self.root.children.items()))
        visits = np.asarray(visits, dtype=np.float64)

        if temp <= 1e-3:
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[int(np.argmax(visits))] = 1.0
        else:
            adjusted = visits ** (1.0 / temp)
            probs = (adjusted / adjusted.sum()).astype(np.float32)

        return list(moves), probs

    def get_action(
        self,
        board: np.ndarray,
        current_player: int,
        temp: float = 1e-3,
        return_probs: bool = False,
    ) -> Move | tuple[Move, np.ndarray]:
        """Return the selected move, optionally with a full-board probability vector."""
        moves, move_probs = self.get_action_probs(board, current_player, temp)
        if not moves:
            raise ValueError("No legal moves available.")

        move = int(np.random.choice(moves, p=move_probs))

        if not return_probs:
            return move

        full_probs = np.zeros(self.board_size, dtype=np.float32)
        full_probs[np.asarray(moves, dtype=np.int64)] = move_probs
        return move, full_probs

    def update_with_move(self, last_move: Move | None) -> None:
        """Reuse the subtree after a real move; pass None to reset the tree."""
        if last_move is not None and last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTSNode(prior=1.0)

    def reset(self) -> None:
        self.root = MCTSNode(prior=1.0)

    def _policy_value(self, board: np.ndarray, current_player: int) -> tuple[list[tuple[Move, float]], float]:
        legal_moves = self.rules.legal_moves(board)
        if len(legal_moves) == 0:
            return [], 0.0

        state = self._encode_state(board, current_player)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()

        legal_probs = policy[legal_moves]
        prob_sum = float(legal_probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            legal_probs = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        else:
            legal_probs = legal_probs / prob_sum

        # Model value is a win rate in [0, 1]; MCTS backs up values in [-1, 1].
        leaf_value = float(value.item()) * 2.0 - 1.0
        return list(zip(legal_moves.tolist(), legal_probs.tolist())), leaf_value

    def _encode_state(self, board: np.ndarray, current_player: int) -> torch.Tensor:
        state = self.rules.encode_state(board, current_player)
        return torch.from_numpy(state).unsqueeze(0).to(self.device)

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
