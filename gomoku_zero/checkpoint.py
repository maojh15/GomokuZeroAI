from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig
from .policy_value_model import PolicyValueModel


def checkpoint_path(config: TrainConfig, iteration: int | None) -> Path:
    checkpoint_dir = Path(config.checkpoint_dir)
    if iteration is None:
        name = f"initial_{config.board_suffix}.pt"
    else:
        name = f"iter_{iteration:04d}_{config.board_suffix}.pt"
    return checkpoint_dir / name


def find_latest_iteration_checkpoint(config: TrainConfig) -> tuple[int, Path] | None:
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    latest: tuple[int, Path] | None = None
    pattern = f"iter_*_{config.board_suffix}.pt"
    for path in checkpoint_dir.glob(pattern):
        iteration = _iteration_from_path(path, config)
        if iteration is None:
            continue
        if latest is None or iteration > latest[0]:
            latest = (iteration, path)
    return latest


def save_checkpoint(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
    iteration: int | None,
) -> Path:
    path = checkpoint_path(config, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "iteration": iteration,
        "config": config.__dict__,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> PolicyValueModel:
    checkpoint = torch.load(path, map_location=device)
    config = _config_from_checkpoint(checkpoint)
    model = PolicyValueModel(
        in_channels=config.in_channels,
        channels=config.channels,
        board_height=config.board_height,
        board_width=config.board_width,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_training_checkpoint(
    path: str | Path,
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int | None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("iteration")


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> TrainConfig:
    raw_config = checkpoint.get("config")
    if raw_config is None:
        raise KeyError("Checkpoint does not contain a saved config.")
    if isinstance(raw_config, TrainConfig):
        return raw_config
    if not isinstance(raw_config, dict):
        raise TypeError("Checkpoint config must be a TrainConfig or dict.")

    config_data = dict(raw_config)
    if "player_values" in config_data:
        config_data["player_values"] = tuple(int(value) for value in config_data["player_values"])
    return TrainConfig(**config_data)


def _iteration_from_path(path: Path, config: TrainConfig) -> int | None:
    prefix = "iter_"
    suffix = f"_{config.board_suffix}.pt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw_iteration = name[len(prefix) : -len(suffix)]
    try:
        return int(raw_iteration)
    except ValueError:
        return None
