from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .mcts import MCTS
from .policy_value_model import PolicyValueModel


ModelState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class EvaluationResult:
    current_wins: int
    previous_wins: int
    draws: int
    games: int
    total_moves: int = 0
    max_moves: int = 0

    @property
    def current_score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.current_wins + 0.5 * self.draws) / self.games

    @property
    def average_moves(self) -> float:
        if self.games == 0:
            return 0.0
        return self.total_moves / self.games


def evaluate_models(
    current_model: PolicyValueModel,
    previous_model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    explore_temp: float | None = None,
    temp_threshold: int = 0,
    candidate_distance: int | None = None,
    tactical_shortcuts: bool = True,
    workers: int = 1,
    seed: int | None = None,
    backend: str = "python",
    eval_batch_size: int = 128,
) -> EvaluationResult:
    current_model.eval()
    previous_model.eval()
    explore_temp = temp if explore_temp is None else explore_temp

    if backend == "cpp":
        from .cpp_mcts import evaluate_models_cpp

        return evaluate_models_cpp(
            current_model=current_model,
            previous_model=previous_model,
            rules=rules,
            games=games,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            explore_temp=explore_temp,
            temp_threshold=temp_threshold,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
            eval_batch_size=eval_batch_size,
            seed=seed,
        )
    if backend != "python":
        raise ValueError(f"Unknown MCTS backend: {backend}")

    if workers > 1 and games > 1:
        return _evaluate_models_parallel(
            current_model=current_model,
            previous_model=previous_model,
            rules=rules,
            games=games,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            explore_temp=explore_temp,
            temp_threshold=temp_threshold,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
            workers=workers,
            seed=seed,
        )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    current_wins = 0
    previous_wins = 0
    draws = 0
    total_moves = 0
    max_moves = 0

    for game_index in range(games):
        current_is_first = game_index % 2 == 0
        winner_owner, moves = _play_eval_game(
            current_model=current_model,
            previous_model=previous_model,
            current_is_first=current_is_first,
            rules=rules,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            explore_temp=explore_temp,
            temp_threshold=temp_threshold,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        )
        if winner_owner == "current":
            current_wins += 1
        elif winner_owner == "previous":
            previous_wins += 1
        else:
            draws += 1
        total_moves += moves
        max_moves = max(max_moves, moves)

    return EvaluationResult(
        current_wins=current_wins,
        previous_wins=previous_wins,
        draws=draws,
        games=games,
        total_moves=total_moves,
        max_moves=max_moves,
    )


def _evaluate_models_parallel(
    current_model: PolicyValueModel,
    previous_model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    explore_temp: float,
    temp_threshold: int,
    candidate_distance: int | None,
    tactical_shortcuts: bool,
    workers: int,
    seed: int | None,
) -> EvaluationResult:
    worker_count = min(int(workers), int(games))
    game_indices = _split_game_indices(games, worker_count)
    current_state = _state_dict_to_cpu(current_model)
    previous_state = _state_dict_to_cpu(previous_model)
    channels = _model_channels(current_model)
    worker_args = [
        (
            current_state,
            previous_state,
            rules.board_height,
            rules.board_width,
            rules.player_values,
            channels,
            indices,
            n_playout,
            c_puct,
            str(device),
            temp,
            explore_temp,
            temp_threshold,
            candidate_distance,
            tactical_shortcuts,
            None if seed is None else seed + worker_id,
        )
        for worker_id, indices in enumerate(game_indices)
        if indices
    ]

    result = EvaluationResult(current_wins=0, previous_wins=0, draws=0, games=0)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_evaluation_worker, args) for args in worker_args]
        for future in as_completed(futures):
            result = _merge_results(result, future.result())
    return result


def _evaluation_worker(args: tuple) -> EvaluationResult:
    (
        current_state,
        previous_state,
        board_height,
        board_width,
        player_values,
        channels,
        game_indices,
        n_playout,
        c_puct,
        device,
        temp,
        explore_temp,
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
    current_model = PolicyValueModel(
        in_channels=2,
        channels=channels,
        board_height=board_height,
        board_width=board_width,
    ).to(device)
    previous_model = PolicyValueModel(
        in_channels=2,
        channels=channels,
        board_height=board_height,
        board_width=board_width,
    ).to(device)
    current_model.load_state_dict(current_state)
    previous_model.load_state_dict(previous_state)
    current_model.eval()
    previous_model.eval()

    current_wins = 0
    previous_wins = 0
    draws = 0
    total_moves = 0
    max_moves = 0
    for game_index in game_indices:
        winner_owner, moves = _play_eval_game(
            current_model=current_model,
            previous_model=previous_model,
            current_is_first=game_index % 2 == 0,
            rules=rules,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            temp=temp,
            explore_temp=explore_temp,
            temp_threshold=temp_threshold,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        )
        if winner_owner == "current":
            current_wins += 1
        elif winner_owner == "previous":
            previous_wins += 1
        else:
            draws += 1
        total_moves += moves
        max_moves = max(max_moves, moves)
    return EvaluationResult(
        current_wins=current_wins,
        previous_wins=previous_wins,
        draws=draws,
        games=len(game_indices),
        total_moves=total_moves,
        max_moves=max_moves,
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
    explore_temp: float,
    temp_threshold: int,
    candidate_distance: int | None = None,
    tactical_shortcuts: bool = True,
) -> tuple[str, int]:
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
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        ),
        previous_player_value: MCTS(
            model=previous_model,
            n_playout=n_playout,
            c_puct=c_puct,
            device=device,
            rules=rules,
            candidate_distance=candidate_distance,
            tactical_shortcuts=tactical_shortcuts,
        ),
    }
    for mcts in mcts_by_player.values():
        mcts.reset()

    board = np.zeros((rules.board_height, rules.board_width), dtype=np.int8)
    player_to_move = first_player
    moves = 0

    while True:
        mcts = mcts_by_player[player_to_move]
        model_by_player[player_to_move].eval()
        move_temp = explore_temp if moves < temp_threshold else temp
        move = mcts.get_action(board, player_to_move, temp=move_temp, return_probs=False)
        board = rules.next_board(board, move, player_to_move)
        moves += 1

        for player_mcts in mcts_by_player.values():
            player_mcts.update_with_move(move)

        ended, winner = rules.game_end_after_move(board, move, player_to_move)
        if ended:
            if winner == 0:
                return "draw", moves
            return ("current" if winner == current_player_value else "previous"), moves

        player_to_move = rules.next_player(player_to_move)


def _state_dict_to_cpu(model: PolicyValueModel) -> ModelState:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _model_channels(model: PolicyValueModel) -> int:
    first_conv = model.backend.layers[0][0]
    return int(first_conv.out_channels)


def _split_game_indices(games: int, workers: int) -> list[list[int]]:
    chunks: list[list[int]] = [[] for _ in range(workers)]
    for game_index in range(games):
        chunks[game_index % workers].append(game_index)
    return chunks


def _merge_results(left: EvaluationResult, right: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(
        current_wins=left.current_wins + right.current_wins,
        previous_wins=left.previous_wins + right.previous_wins,
        draws=left.draws + right.draws,
        games=left.games + right.games,
        total_moves=left.total_moves + right.total_moves,
        max_moves=max(left.max_moves, right.max_moves),
    )
