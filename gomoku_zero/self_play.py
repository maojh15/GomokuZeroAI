from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .mcts import MCTS
from .policy_value_model import PolicyValueModel
from .replay_buffer import TrainingSample


@dataclass(frozen=True)
class SelfPlayStats:
    winner: int
    moves: int


@dataclass(frozen=True)
class SelfPlayResult:
    samples: list[TrainingSample]
    stats: SelfPlayStats


@dataclass(frozen=True)
class PendingSample:
    state: np.ndarray
    policy: np.ndarray
    player: int


def generate_self_play_game(
    model: PolicyValueModel,
    rules: GomokuRules,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    temp_threshold: int,
) -> SelfPlayResult:
    model.eval()
    board = np.zeros((rules.board_height, rules.board_width), dtype=np.int8)
    current_player = rules.player_values[0]
    mcts = MCTS(
        model=model,
        n_playout=n_playout,
        c_puct=c_puct,
        device=device,
        rules=rules,
    )
    mcts.reset()

    pending: list[PendingSample] = []
    moves = 0
    while True:
        move_temp = temp if moves < temp_threshold else 1e-3
        state = rules.encode_state(board, current_player)
        move, policy = mcts.get_action(
            board,
            current_player,
            temp=move_temp,
            return_probs=True,
        )
        pending.append(PendingSample(state=state, policy=policy, player=current_player))

        board = rules.next_board(board, move, current_player)
        moves += 1
        ended, winner = rules.game_end(board)
        mcts.update_with_move(move)

        if ended:
            samples = [
                TrainingSample(
                    state=item.state,
                    policy=item.policy,
                    value=_value_target(winner, item.player),
                )
                for item in pending
            ]
            return SelfPlayResult(samples=samples, stats=SelfPlayStats(winner=winner, moves=moves))

        current_player = rules.next_player(current_player)


def generate_self_play_games(
    model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    temp_threshold: int,
) -> tuple[list[TrainingSample], list[SelfPlayStats]]:
    all_samples: list[TrainingSample] = []
    stats: list[SelfPlayStats] = []
    for _ in range(games):
        result = generate_self_play_game(
            model=model,
            rules=rules,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            temp_threshold=temp_threshold,
        )
        all_samples.extend(result.samples)
        stats.append(result.stats)
    return all_samples, stats


def _value_target(winner: int, player: int) -> float:
    if winner == 0:
        return 0.5
    return 1.0 if winner == player else 0.0
