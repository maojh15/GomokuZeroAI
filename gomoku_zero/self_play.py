from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .mcts import MCTS
from .policy_value_model import PolicyValueModel
from .replay_buffer import TrainingSample


ModelState = dict[str, torch.Tensor]


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
    candidate_distance: int | None = None,
    tactical_shortcuts: bool = True,
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
        candidate_distance=candidate_distance,
        tactical_shortcuts=tactical_shortcuts,
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
        ended, winner = rules.game_end_after_move(board, move, current_player)
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
    candidate_distance: int | None = None,
    tactical_shortcuts: bool = True,
    workers: int = 1,
    seed: int | None = None,
) -> tuple[list[TrainingSample], list[SelfPlayStats]]:
    if workers <= 1 or games <= 1:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        return _generate_self_play_games_serial(
            model=model,
            rules=rules,
            games=games,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            temp_threshold=temp_threshold,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        )

    worker_count = min(int(workers), int(games))
    model_state = _state_dict_to_cpu(model)
    game_counts = _split_games(games, worker_count)
    worker_args = [
        (
            model_state,
            rules.board_height,
            rules.board_width,
            rules.player_values,
            _model_channels(model),
            count,
            n_playout,
            c_puct,
            str(device),
            temp,
            temp_threshold,
            candidate_distance,
            tactical_shortcuts,
            None if seed is None else seed + worker_id,
        )
        for worker_id, count in enumerate(game_counts)
        if count > 0
    ]

    all_samples: list[TrainingSample] = []
    stats: list[SelfPlayStats] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_self_play_worker, args) for args in worker_args]
        for future in as_completed(futures):
            worker_samples, worker_stats = future.result()
            all_samples.extend(worker_samples)
            stats.extend(worker_stats)
    return all_samples, stats


def _generate_self_play_games_serial(
    model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    temp_threshold: int,
    candidate_distance: int | None = None,
    tactical_shortcuts: bool = True,
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
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        )
        all_samples.extend(result.samples)
        stats.append(result.stats)
    return all_samples, stats


def _self_play_worker(args: tuple) -> tuple[list[TrainingSample], list[SelfPlayStats]]:
    (
        model_state,
        board_height,
        board_width,
        player_values,
        channels,
        games,
        n_playout,
        c_puct,
        device,
        temp,
        temp_threshold,
        candidate_distance,
        tactical_shortcuts,
        seed,
    ) = args
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

    device = torch.device(device)
    rules = GomokuRules(
        board_height=board_height,
        board_width=board_width,
        player_values=player_values,
    )
    model = PolicyValueModel(
        in_channels=2,
        channels=channels,
        board_height=board_height,
        board_width=board_width,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return _generate_self_play_games_serial(
        model=model,
        rules=rules,
        games=games,
        n_playout=n_playout,
        c_puct=c_puct,
        device=device,
        temp=temp,
        temp_threshold=temp_threshold,
        candidate_distance=candidate_distance,
        tactical_shortcuts=tactical_shortcuts,
    )


def _state_dict_to_cpu(model: PolicyValueModel) -> ModelState:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _model_channels(model: PolicyValueModel) -> int:
    first_conv = model.backend.layers[0][0]
    return int(first_conv.out_channels)


def _split_games(games: int, workers: int) -> list[int]:
    base = games // workers
    remainder = games % workers
    return [base + (1 if worker_id < remainder else 0) for worker_id in range(workers)]


def _value_target(winner: int, player: int) -> float:
    if winner == 0:
        return 0.5
    return 1.0 if winner == player else 0.0
