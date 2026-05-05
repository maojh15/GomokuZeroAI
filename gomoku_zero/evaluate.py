from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .mcts import MCTS
from .policy_value_model import PolicyValueModel


@dataclass(frozen=True)
class EvaluationResult:
    current_wins: int
    previous_wins: int
    draws: int
    games: int

    @property
    def current_score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.current_wins + 0.5 * self.draws) / self.games


def evaluate_models(
    current_model: PolicyValueModel,
    previous_model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
) -> EvaluationResult:
    current_model.eval()
    previous_model.eval()
    current_wins = 0
    previous_wins = 0
    draws = 0

    for game_index in range(games):
        current_is_first = game_index % 2 == 0
        winner_owner = _play_eval_game(
            current_model=current_model,
            previous_model=previous_model,
            current_is_first=current_is_first,
            rules=rules,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
        )
        if winner_owner == "current":
            current_wins += 1
        elif winner_owner == "previous":
            previous_wins += 1
        else:
            draws += 1

    return EvaluationResult(
        current_wins=current_wins,
        previous_wins=previous_wins,
        draws=draws,
        games=games,
    )


def _play_eval_game(
    current_model: PolicyValueModel,
    previous_model: PolicyValueModel,
    current_is_first: bool,
    rules: GomokuRules,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
) -> str:
    first_player, second_player = rules.player_values
    current_player_value = first_player if current_is_first else second_player
    previous_player_value = second_player if current_is_first else first_player
    model_by_player = {
        current_player_value: current_model,
        previous_player_value: previous_model,
    }
    mcts_by_player = {
        current_player_value: MCTS(
            model=current_model,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            rules=rules,
        ),
        previous_player_value: MCTS(
            model=previous_model,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            rules=rules,
        ),
    }
    for mcts in mcts_by_player.values():
        mcts.reset()

    board = np.zeros((rules.board_height, rules.board_width), dtype=np.int8)
    player_to_move = first_player

    while True:
        mcts = mcts_by_player[player_to_move]
        model_by_player[player_to_move].eval()
        move = mcts.get_action(board, player_to_move, temp=temp, return_probs=False)
        board = rules.next_board(board, move, player_to_move)

        for player_mcts in mcts_by_player.values():
            player_mcts.update_with_move(move)

        ended, winner = rules.game_end(board)
        if ended:
            if winner == 0:
                return "draw"
            return "current" if winner == current_player_value else "previous"

        player_to_move = rules.next_player(player_to_move)
