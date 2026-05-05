from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    board_height: int = 15
    board_width: int = 15
    player_values: tuple[int, int] = (1, -1)
    in_channels: int = 2
    channels: int = 128

    device: str | None = None
    seed: int = 0
    checkpoint_dir: str = "checkpoints"

    num_iterations: int = 10
    self_play_games_per_iteration: int = 10
    mcts_playouts: int = 500
    c_puct: float = 5.0
    self_play_temp: float = 1.0
    self_play_temp_threshold: int = 12
    eval_temp: float = 1e-3

    replay_buffer_size: int = 10000
    augment_symmetry: bool = True
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    eval_games: int = 10

    log_interval: int = 10

    @property
    def board_size(self) -> int:
        return self.board_height * self.board_width

    @property
    def board_suffix(self) -> str:
        return f"{self.board_height}x{self.board_width}"


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_mapping(config_path)
    valid_keys = {field.name for field in fields(TrainConfig)}
    unknown_keys = sorted(set(raw) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    if "player_values" in raw:
        raw["player_values"] = tuple(int(value) for value in raw["player_values"])
    return TrainConfig(**raw)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at the top level.")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line {line_no}: {line}")
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")
