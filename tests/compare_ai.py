from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku_zero.config import TrainConfig
from gomoku_zero.gomoku_rules import GomokuRules


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    config: TrainConfig
    iteration: int | None


def main() -> None:
    args = parse_args()
    import_torch()
    from gomoku_zero.checkpoint import load_model_checkpoint
    from gomoku_zero.evaluate import evaluate_models

    checkpoint_a = read_checkpoint_info(args.checkpoint_a)
    checkpoint_b = read_checkpoint_info(args.checkpoint_b)
    ensure_compatible(checkpoint_a, checkpoint_b)

    device = resolve_device(args.device)
    model_a = load_model_checkpoint(checkpoint_a.path, device=device)
    model_b = load_model_checkpoint(checkpoint_b.path, device=device)
    rules = GomokuRules(
        board_height=checkpoint_a.config.board_height,
        board_width=checkpoint_a.config.board_width,
        player_values=checkpoint_a.config.player_values,
    )

    result = evaluate_models(
        current_model=model_a,
        previous_model=model_b,
        rules=rules,
        games=args.games,
        n_playout=args.playouts,
        c_puct=args.c_puct,
        device=device,
        temp=args.temp,
        explore_temp=args.explore_temp,
        temp_threshold=args.temp_threshold,
        workers=args.workers,
        seed=args.seed,
        backend="cpp"
    )

    wins_a = result.current_wins
    wins_b = result.previous_wins
    draws = result.draws
    games = result.games
    score_a = result.current_score_rate
    score_b = 1.0 - score_a if games else 0.0

    print("Checkpoint A:", checkpoint_a.path)
    print("Checkpoint B:", checkpoint_b.path)
    print(
        f"Board: {checkpoint_a.config.board_height}x{checkpoint_a.config.board_width}, "
        f"channels: {checkpoint_a.config.channels}, playouts: {args.playouts}, games: {games}"
    )
    print()
    print(f"A wins: {wins_a:4d} ({rate(wins_a, games)})")
    print(f"B wins: {wins_b:4d} ({rate(wins_b, games)})")
    print(f"Draws : {draws:4d} ({rate(draws, games)})")
    print()
    print(f"A score rate: {score_a:.2%}  (win + 0.5 * draw)")
    print(f"B score rate: {score_b:.2%}  (win + 0.5 * draw)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load two GomokuZero checkpoints and compare them with balanced model-vs-model games.",
    )
    parser.add_argument("checkpoint_a", type=Path, help="Path to the first checkpoint.")
    parser.add_argument("checkpoint_b", type=Path, help="Path to the second checkpoint.")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play.")
    parser.add_argument("--playouts", type=int, default=500, help="MCTS playouts per move.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel evaluation workers.")
    parser.add_argument("--c-puct", type=float, default=5.0, help="PUCT exploration constant.")
    parser.add_argument("--temp", type=float, default=1e-3, help="Move temperature after the opening.")
    parser.add_argument(
        "--explore-temp",
        type=float,
        default=1.0,
        help="Move temperature for the first --temp-threshold moves.",
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=0,
        help="Opening moves that use --explore-temp instead of --temp.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible matches.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    return parser.parse_args()


def read_checkpoint_info(path: Path) -> CheckpointInfo:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    torch = import_torch()
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")
    config = config_from_checkpoint(checkpoint, path)
    return CheckpointInfo(
        path=path,
        config=config,
        iteration=checkpoint.get("iteration"),
    )


def config_from_checkpoint(checkpoint: dict[str, Any], path: Path) -> TrainConfig:
    from gomoku_zero.checkpoint import config_from_checkpoint_payload

    return config_from_checkpoint_payload(checkpoint, path)


def ensure_compatible(left: CheckpointInfo, right: CheckpointInfo) -> None:
    comparable_fields = ("board_height", "board_width", "player_values", "in_channels", "channels")
    mismatches = [
        field
        for field in comparable_fields
        if getattr(left.config, field) != getattr(right.config, field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: {getattr(left.config, field)!r} != {getattr(right.config, field)!r}"
            for field in mismatches
        )
        raise ValueError(f"Checkpoints are not compatible for direct evaluation ({details}).")


def resolve_device(device: str | None) -> str:
    if device:
        return device
    torch = import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def rate(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total:.2%}"


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to load checkpoints. Activate the project's Python environment "
            "or install torch, then run this script again."
        ) from exc
    return torch


if __name__ == "__main__":
    main()
