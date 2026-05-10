from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .gomoku_rules import GomokuRules
from .replay_buffer import TrainingSample


def append_human_replay_records(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    rules: GomokuRules,
) -> int:
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with replay_path.open("a", encoding="utf-8") as f:
        for record in records:
            compact = {
                "p": int(record["player"]),
                "s": _sparse_board(record["board"], rules),
                "pi": _sparse_visits(record["visits"], rules),
                "z": float(record["value"]),
            }
            f.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_human_replay_samples(path: str | Path, rules: GomokuRules) -> list[TrainingSample]:
    replay_path = Path(path)
    if not replay_path.exists():
        return []

    samples: list[TrainingSample] = []
    with replay_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sample = _record_to_sample(record, rules)
            except Exception as exc:
                raise ValueError(f"Invalid human replay record at {replay_path}:{line_no}: {exc}") from exc
            samples.append(sample)
    return samples


def _record_to_sample(record: dict[str, Any], rules: GomokuRules) -> TrainingSample:
    expected_keys = {"p", "s", "pi", "z"}
    if set(record) != expected_keys:
        raise ValueError(f"record keys must be exactly {sorted(expected_keys)}")

    board = np.zeros((rules.board_height, rules.board_width), dtype=np.int8)
    board_moves: set[int] = set()
    for move, stone in record["s"]:
        move_index = _json_int(move, "board move")
        if not 0 <= move_index < rules.board_size:
            raise ValueError(f"board move out of range: {move_index}")
        if move_index in board_moves:
            raise ValueError(f"duplicate board move: {move_index}")
        board_moves.add(move_index)
        player = _json_int(stone, "stone value")
        rules.validate_player(player)
        row, col = divmod(move_index, rules.board_width)
        board[row, col] = player

    current_player = _json_int(record["p"], "player")
    rules.validate_player(current_player)
    state = rules.encode_state(board, current_player)

    policy = np.zeros(rules.board_size, dtype=np.float32)
    policy_moves: set[int] = set()
    for move, visits in record["pi"]:
        move_index = _json_int(move, "policy move")
        if not 0 <= move_index < rules.board_size:
            raise ValueError(f"policy move out of range: {move_index}")
        if move_index in policy_moves:
            raise ValueError(f"duplicate policy move: {move_index}")
        policy_moves.add(move_index)
        visit_count = _json_int(visits, "visit count")
        if visit_count <= 0:
            raise ValueError(f"invalid visit count: {visit_count}")
        policy[move_index] = visit_count
    policy_sum = float(policy.sum())
    if policy_sum <= 0.0 or not np.isfinite(policy_sum):
        raise ValueError("policy visits must have a positive finite sum")
    policy = policy / policy_sum

    value = float(record["z"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")

    return TrainingSample(
        state=np.ascontiguousarray(state, dtype=np.float32),
        policy=np.ascontiguousarray(policy, dtype=np.float32),
        value=value,
    )


def _json_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


def _sparse_board(board: Any, rules: GomokuRules) -> list[list[int]]:
    board_array = rules.as_board(np.asarray(board, dtype=np.int8))
    sparse: list[list[int]] = []
    for move, stone in enumerate(board_array.reshape(-1)):
        if int(stone) != 0:
            sparse.append([int(move), int(stone)])
    return sparse


def _sparse_visits(visits: Any, rules: GomokuRules) -> list[list[int]]:
    visit_array = np.asarray(visits, dtype=np.int64).reshape(-1)
    if visit_array.shape != (rules.board_size,):
        raise ValueError(f"visits shape {visit_array.shape} does not match ({rules.board_size},)")
    sparse: list[list[int]] = []
    for move, visit_count in enumerate(visit_array):
        if int(visit_count) > 0:
            sparse.append([int(move), int(visit_count)])
    if not sparse:
        raise ValueError("visits must contain at least one positive entry")
    return sparse
