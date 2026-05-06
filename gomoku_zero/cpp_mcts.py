from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gomoku_rules import GomokuRules
from .policy_value_model import PolicyValueModel
from .replay_buffer import TrainingSample


def require_cpp_mcts():
    try:
        from . import _mcts_cpp
    except ImportError as exc:
        raise RuntimeError(
            "C++ MCTS backend is not built. Run: python setup.py build_ext --inplace"
        ) from exc
    return _mcts_cpp


@dataclass
class CppSelfPlayGameState:
    game: object
    pending: list[tuple[np.ndarray, np.ndarray, int]]
    moves: int = 0
    ended: bool = False
    winner: int = 0


@dataclass(frozen=True)
class CppSelfPlayStats:
    winner: int
    moves: int


@dataclass
class CppEvalGameState:
    current_tree: object
    previous_tree: object
    current_player_value: int
    previous_player_value: int
    player_to_move: int
    moves: int = 0
    ended: bool = False
    winner_owner: str = "draw"


def generate_self_play_games_cpp(
    model: PolicyValueModel,
    rules: GomokuRules,
    games: int,
    n_playout: int,
    c_puct: float,
    device: torch.device | str,
    temp: float,
    temp_threshold: int,
    candidate_distance: int | None,
    tactical_shortcuts: bool,
    eval_batch_size: int,
    seed: int | None = None,
) -> tuple[list[TrainingSample], list[CppSelfPlayStats]]:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    cpp = require_cpp_mcts()
    model.eval()
    device = torch.device(device)
    distance = -1 if candidate_distance is None else int(candidate_distance)
    states = [
        CppSelfPlayGameState(
            game=cpp.CppMCTSGame(
                rules.board_height,
                rules.board_width,
                rules.player_values[0],
                rules.player_values[1],
                n_playout,
                c_puct,
                distance,
                tactical_shortcuts,
            ),
            pending=[],
        )
        for _ in range(games)
    ]

    while any(not state.ended for state in states):
        active = [state for state in states if not state.ended]
        _run_search_batch(
            search_items=[(state.game, model) for state in active],
            device=device,
            board_shape=(rules.board_height, rules.board_width),
            batch_size=eval_batch_size,
        )
        for state in active:
            _play_cpp_self_play_move(state, rules, temp, temp_threshold)

    samples: list[TrainingSample] = []
    stats: list[CppSelfPlayStats] = []
    for state in states:
        for encoded_state, policy, player in state.pending:
            samples.append(
                TrainingSample(
                    state=encoded_state,
                    policy=policy,
                    value=_value_target(state.winner, player),
                )
            )
        stats.append(CppSelfPlayStats(winner=state.winner, moves=state.moves))
    return samples, stats


def evaluate_models_cpp(
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
    eval_batch_size: int,
    seed: int | None = None,
):
    from .evaluate import EvaluationResult

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    cpp = require_cpp_mcts()
    current_model.eval()
    previous_model.eval()
    device = torch.device(device)
    first_player, second_player = rules.player_values
    distance = -1 if candidate_distance is None else int(candidate_distance)
    states: list[CppEvalGameState] = []
    for game_index in range(games):
        current_is_first = game_index % 2 == 0
        current_player_value = first_player if current_is_first else second_player
        previous_player_value = second_player if current_is_first else first_player
        states.append(
            CppEvalGameState(
                current_tree=cpp.CppMCTSGame(
                    rules.board_height,
                    rules.board_width,
                    first_player,
                    second_player,
                    n_playout,
                    c_puct,
                    distance,
                    tactical_shortcuts,
                ),
                previous_tree=cpp.CppMCTSGame(
                    rules.board_height,
                    rules.board_width,
                    first_player,
                    second_player,
                    n_playout,
                    c_puct,
                    distance,
                    tactical_shortcuts,
                ),
                current_player_value=current_player_value,
                previous_player_value=previous_player_value,
                player_to_move=first_player,
            )
        )

    while any(not state.ended for state in states):
        search_items = []
        owners = []
        for state in states:
            if state.ended:
                continue
            if state.player_to_move == state.current_player_value:
                search_items.append((state.current_tree, current_model))
                owners.append((state, "current"))
            else:
                search_items.append((state.previous_tree, previous_model))
                owners.append((state, "previous"))
        _run_search_batch(
            search_items=search_items,
            device=device,
            board_shape=(rules.board_height, rules.board_width),
            batch_size=eval_batch_size,
        )
        for state, owner in owners:
            tree = state.current_tree if owner == "current" else state.previous_tree
            move_temp = explore_temp if state.moves < temp_threshold else temp
            move = _choose_cpp_move(tree, move_temp)
            state.current_tree.apply_move(move)
            state.previous_tree.apply_move(move)
            state.moves += 1
            ended, winner = tree.game_end_after_move(move, state.player_to_move)
            if ended:
                state.ended = True
                state.winner_owner = _winner_owner(int(winner), state)
            else:
                state.player_to_move = rules.opponent_of(state.player_to_move)

    current_wins = sum(1 for state in states if state.winner_owner == "current")
    previous_wins = sum(1 for state in states if state.winner_owner == "previous")
    draws = sum(1 for state in states if state.winner_owner == "draw")
    return EvaluationResult(current_wins=current_wins, previous_wins=previous_wins, draws=draws, games=games)


def _run_search_batch(
    search_items: list[tuple[object, PolicyValueModel]],
    device: torch.device,
    board_shape: tuple[int, int],
    batch_size: int,
) -> None:
    active = [item for item in search_items if item[0].tactical_move() < 0]
    while active:
        pending: list[tuple[object, PolicyValueModel, int, np.ndarray]] = []
        made_progress = False
        for game, model in active:
            if game.root_visits() >= game.n_playout():
                continue
            request = game.request_leaf()
            status = request["status"]
            if status == "leaf":
                state = np.asarray(request["state"], dtype=np.float32).reshape(2, *board_shape)
                pending.append((game, model, int(request["request_id"]), state))
                made_progress = True
            if len(pending) >= batch_size:
                _flush_pending(pending, device)
                pending.clear()
        if pending:
            _flush_pending(pending, device)
        if not made_progress and all(game.root_visits() >= game.n_playout() for game, _model in active):
            break
        active = [item for item in active if item[0].root_visits() < item[0].n_playout()]


def _flush_pending(pending: list[tuple[object, PolicyValueModel, int, np.ndarray]], device: torch.device) -> None:
    by_model: dict[int, list[tuple[object, PolicyValueModel, int, np.ndarray]]] = {}
    for item in pending:
        by_model.setdefault(id(item[1]), []).append(item)
    for items in by_model.values():
        model = items[0][1]
        states = torch.from_numpy(np.stack([item[3] for item in items])).to(device)
        with torch.no_grad():
            logits, values = model(states)
            policies = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            values_np = values.squeeze(1).detach().cpu().numpy()
        for (game, _model, request_id, _state), policy, value in zip(items, policies, values_np):
            game.apply_evaluation(request_id, policy.tolist(), float(value))


def _play_cpp_self_play_move(state: CppSelfPlayGameState, rules: GomokuRules, temp: float, temp_threshold: int) -> None:
    player = int(state.game.current_player())
    board = np.asarray(state.game.board(), dtype=np.int8).reshape(rules.board_height, rules.board_width)
    encoded_state = rules.encode_state(board, player)
    move_temp = temp if state.moves < temp_threshold else 1e-3
    moves, probs = state.game.action_probs(move_temp)
    if not moves:
        tactical = state.game.tactical_move()
        if tactical < 0:
            raise ValueError("C++ MCTS returned no legal action.")
        moves, probs = [int(tactical)], [1.0]
    move_probs = _normalize_probs(probs)
    move = int(np.random.choice(np.asarray(moves, dtype=np.int64), p=move_probs))
    full_policy = np.zeros(rules.board_size, dtype=np.float32)
    full_policy[np.asarray(moves, dtype=np.int64)] = move_probs.astype(np.float32)
    state.pending.append((encoded_state, full_policy, player))
    state.game.apply_move(move)
    state.moves += 1
    ended, winner = state.game.game_end_after_move(move, player)
    if ended:
        state.ended = True
        state.winner = int(winner)


def _choose_cpp_move(game: object, temp: float) -> int:
    tactical = game.tactical_move()
    if tactical >= 0:
        return int(tactical)
    moves, probs = game.action_probs(temp)
    if not moves:
        raise ValueError("C++ MCTS returned no legal action.")
    return int(np.random.choice(np.asarray(moves, dtype=np.int64), p=_normalize_probs(probs)))


def _winner_owner(winner: int, state: CppEvalGameState) -> str:
    if winner == 0:
        return "draw"
    return "current" if winner == state.current_player_value else "previous"


def _value_target(winner: int, player: int) -> float:
    if winner == 0:
        return 0.5
    return 1.0 if winner == player else 0.0


def _normalize_probs(probs: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("probabilities must be a non-empty 1D array.")
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    values = values / total
    # np.random.choice is strict; this final adjustment removes tiny roundoff drift.
    values[-1] = 1.0 - float(values[:-1].sum())
    if values[-1] < 0.0:
        values = np.maximum(values, 0.0)
        values = values / float(values.sum())
    return values
