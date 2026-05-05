from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from gomoku_zero.checkpoint import checkpoint_path, load_model_checkpoint, save_checkpoint
from gomoku_zero.config import TrainConfig
from gomoku_zero.evaluate import evaluate_models
from gomoku_zero.gomoku_rules import GomokuRules
from gomoku_zero.policy_value_model import PolicyValueModel
from gomoku_zero.replay_buffer import ReplayBuffer, TrainingSample
from gomoku_zero.self_play import generate_self_play_game
from gomoku_zero.trainer import train_one_iteration


class TrainingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.config = TrainConfig(
            board_height=5,
            board_width=5,
            channels=8,
            mcts_playouts=2,
            self_play_temp_threshold=2,
            replay_buffer_size=64,
            batch_size=4,
            epochs=1,
            eval_games=2,
            augment_symmetry=False,
            device="cpu",
        )
        self.rules = GomokuRules(
            board_height=self.config.board_height,
            board_width=self.config.board_width,
            player_values=self.config.player_values,
        )

    def test_self_play_sample_shapes(self) -> None:
        model = self._model()
        result = generate_self_play_game(
            model=model,
            rules=self.rules,
            n_playout=self.config.mcts_playouts,
            c_puct=self.config.c_puct,
            device="cpu",
            temp=self.config.self_play_temp,
            temp_threshold=self.config.self_play_temp_threshold,
        )
        sample = result.samples[0]
        self.assertEqual(sample.state.shape, (2, 5, 5))
        self.assertEqual(sample.policy.shape, (25,))
        self.assertAlmostEqual(float(sample.policy.sum()), 1.0, places=5)
        self.assertGreaterEqual(sample.value, 0.0)
        self.assertLessEqual(sample.value, 1.0)

    def test_replay_buffer_capacity_and_training_step(self) -> None:
        buffer = ReplayBuffer(capacity=4, board_height=5, board_width=5, augment_symmetry=False)
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 1.0
        samples = [
            TrainingSample(
                state=np.random.rand(2, 5, 5).astype(np.float32),
                policy=policy,
                value=1.0,
            )
            for _ in range(6)
        ]
        buffer.add_many(samples)
        self.assertEqual(len(buffer), 4)

        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = next(model.parameters()).detach().clone()
        metrics = train_one_iteration(
            model=model,
            optimizer=optimizer,
            replay_buffer=buffer,
            batch_size=2,
            epochs=1,
            device="cpu",
            log_interval=0,
        )
        after = next(model.parameters()).detach()
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertFalse(torch.equal(before, after))

    def test_checkpoint_name_and_eval(self) -> None:
        tmpdir = Path(".test_checkpoints")
        tmpdir.mkdir(exist_ok=True)
        try:
            config = TrainConfig(
                board_height=5,
                board_width=5,
                channels=8,
                checkpoint_dir=str(tmpdir),
                eval_games=2,
                mcts_playouts=1,
                device="cpu",
            )
            model = self._model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            path = save_checkpoint(model, optimizer, config, iteration=None)
            self.assertEqual(path, checkpoint_path(config, iteration=None))
            self.assertTrue(path.name.endswith("5x5.pt"))
            loaded_model = load_model_checkpoint(path, device="cpu")
            loaded_policy, loaded_value = loaded_model(torch.zeros(1, 2, 5, 5))
            self.assertEqual(loaded_policy.shape, (1, 25))
            self.assertEqual(loaded_value.shape, (1, 1))

            previous_model = self._model()
            result = evaluate_models(
                current_model=model,
                previous_model=previous_model,
                rules=self.rules,
                games=2,
                n_playout=1,
                c_puct=5.0,
                device="cpu",
                temp=1e-3,
            )
            self.assertEqual(result.games, 2)
            self.assertEqual(result.current_wins + result.previous_wins + result.draws, 2)
        finally:
            for path in tmpdir.glob("*"):
                path.unlink()
            tmpdir.rmdir()

    def _model(self) -> PolicyValueModel:
        return PolicyValueModel(
            in_channels=2,
            channels=8,
            board_height=5,
            board_width=5,
        )


if __name__ == "__main__":
    unittest.main()
