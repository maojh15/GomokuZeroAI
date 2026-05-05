from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from .checkpoint import load_model_checkpoint, save_checkpoint
from .config import TrainConfig, load_config
from .evaluate import evaluate_models
from .gomoku_rules import GomokuRules
from .policy_value_model import PolicyValueModel
from .replay_buffer import ReplayBuffer
from .self_play import generate_self_play_games
from .trainer import train_one_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Gomoku policy-value model with self-play.")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to the YAML training config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


def run_training(config: TrainConfig) -> None:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    print(f"Using device: {device}")

    rules = GomokuRules(
        board_height=config.board_height,
        board_width=config.board_width,
        player_values=config.player_values,
    )
    model = _build_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        board_height=config.board_height,
        board_width=config.board_width,
        augment_symmetry=config.augment_symmetry,
    )

    previous_checkpoint = save_checkpoint(model, optimizer, config, iteration=None)
    print(f"Saved initial checkpoint: {previous_checkpoint}")

    for iteration in range(1, config.num_iterations + 1):
        print(f"\n=== iteration {iteration}/{config.num_iterations} ===")
        samples, game_stats = generate_self_play_games(
            model=model,
            rules=rules,
            games=config.self_play_games_per_iteration,
            n_playout=config.mcts_playouts,
            c_puct=config.c_puct,
            device=device,
            temp=config.self_play_temp,
            temp_threshold=config.self_play_temp_threshold,
        )
        replay_buffer.add_many(samples)
        print(
            f"self-play games={len(game_stats)} raw_samples={len(samples)} "
            f"buffer_size={len(replay_buffer)}"
        )
        _print_self_play_stats(game_stats)

        metrics = train_one_iteration(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            batch_size=config.batch_size,
            epochs=config.epochs,
            device=device,
            log_interval=config.log_interval,
        )
        print(
            f"train avg policy_loss={metrics.policy_loss:.4f} "
            f"value_loss={metrics.value_loss:.4f} total_loss={metrics.total_loss:.4f} "
            f"steps={metrics.steps}"
        )

        current_checkpoint = save_checkpoint(model, optimizer, config, iteration=iteration)
        print(f"Saved checkpoint: {current_checkpoint}")

        if config.eval_games > 0:
            previous_model = load_model_checkpoint(previous_checkpoint, device=device)
            result = evaluate_models(
                current_model=model,
                previous_model=previous_model,
                rules=rules,
                games=config.eval_games,
                n_playout=config.mcts_playouts,
                c_puct=config.c_puct,
                device=device,
                temp=config.eval_temp,
            )
            print(
                "eval vs previous: "
                f"current_wins={result.current_wins} "
                f"previous_wins={result.previous_wins} draws={result.draws} "
                f"score_rate={result.current_score_rate:.3f}"
            )

        previous_checkpoint = current_checkpoint


def _build_model(config: TrainConfig, device: torch.device) -> PolicyValueModel:
    model = PolicyValueModel(
        in_channels=config.in_channels,
        channels=config.channels,
        board_height=config.board_height,
        board_width=config.board_width,
    )
    return model.to(device)


def _resolve_device(config_device: str | None) -> torch.device:
    if config_device:
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_self_play_stats(game_stats: list) -> None:
    if not game_stats:
        return
    winners = [stats.winner for stats in game_stats]
    moves = [stats.moves for stats in game_stats]
    winner_counts = {winner: winners.count(winner) for winner in sorted(set(winners))}
    print(
        f"self-play winners={winner_counts} "
        f"avg_moves={float(np.mean(moves)):.1f} max_moves={max(moves)}"
    )


if __name__ == "__main__":
    main()
