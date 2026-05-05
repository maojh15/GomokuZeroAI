# GomokuZeroAI

AlphaZero-style Gomoku training playground using a policy-value network and MCTS self-play.

## Overview

This project trains a Gomoku policy-value model with the following loop:

1. Save the initial network checkpoint.
2. Generate self-play games with the current model and MCTS.
3. Store training samples in a replay buffer.
4. Train the policy-value network from replay buffer batches.
5. Save a new checkpoint.
6. Evaluate the new checkpoint against the previous checkpoint.

The board state is encoded as two channels:

- channel 0: stones belonging to the player to move
- channel 1: stones belonging to the opponent

The value target is from the stored state's player-to-move perspective:

- win: `1.0`
- draw: `0.5`
- loss: `0.0`

## Layout

- `train.py`: root training entry point.
- `train_config.yaml`: default training configuration.
- `gomoku_zero/`: core package.
- `gomoku_zero/gomoku_rules.py`: board utilities, legal moves, player switching, and game-end detection.
- `gomoku_zero/policy_value_model.py`: convolutional policy-value network.
- `gomoku_zero/mcts.py`: PUCT-based MCTS guided by the policy-value model.
- `gomoku_zero/config.py`: training config dataclass and YAML loading.
- `gomoku_zero/self_play.py`: self-play data generation.
- `gomoku_zero/replay_buffer.py`: replay buffer and optional board-symmetry augmentation.
- `gomoku_zero/trainer.py`: one training iteration over replay buffer batches.
- `gomoku_zero/checkpoint.py`: checkpoint save/load helpers.
- `gomoku_zero/evaluate.py`: model-vs-model evaluation with balanced first player.
- `tests/`: smoke tests for the training pipeline.

## Environment

Use a Python environment with PyTorch installed. Required packages:

- `numpy`
- `torch`
- optional: `pyyaml`

`pyyaml` is optional because `config.py` includes a small fallback parser for the simple `train_config.yaml` format.

## Run Tests

```bash
python -m unittest -v
```

## Train

Start training with the default config:

```bash
python train.py --config train_config.yaml
```

Checkpoints are written to `checkpoints/` by default:

- `initial_15x15.pt`
- `iter_0001_15x15.pt`
- `iter_0002_15x15.pt`

## Quick Smoke Config

For a very fast sanity check, temporarily reduce these values in `train_config.yaml`:

```yaml
num_iterations: 1
self_play_games_per_iteration: 1
mcts_playouts: 2
epochs: 1
eval_games: 2
channels: 16
```

This is not useful for strength, but it verifies that self-play, training, checkpointing, and evaluation all run end to end.

## Important Config Fields

- `board_height`, `board_width`: board dimensions; checkpoint names include this suffix.
- `player_values`: stone encodings, default `[1, -1]`.
- `channels`: model width.
- `num_iterations`: number of train/eval cycles.
- `self_play_games_per_iteration`: self-play games generated per iteration.
- `mcts_playouts`: MCTS simulations per action.
- `c_puct`: exploration constant for PUCT.
- `self_play_temp`: sampling temperature used early in self-play.
- `self_play_temp_threshold`: number of opening moves using `self_play_temp`; later moves use low temperature.
- `replay_buffer_size`: maximum stored training samples.
- `augment_symmetry`: whether to add 8 board symmetry variants per sample.
- `batch_size`, `epochs`, `learning_rate`: training hyperparameters.
- `eval_games`: games against the previous checkpoint after each iteration.

## Notes

MCTS stores values in `[-1, 1]` internally, from each node's player-to-move perspective. The model value head outputs a win-rate estimate in `[0, 1]`, so `gomoku_zero/mcts.py` converts model output with:

```python
leaf_value = value * 2.0 - 1.0
```

Training targets stay in `[0, 1]` because the value head uses `Sigmoid()`.
