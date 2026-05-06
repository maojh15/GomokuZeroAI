from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from gomoku_zero.checkpoint import (
    checkpoint_path,
    find_latest_iteration_checkpoint,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from gomoku_zero.config import TrainConfig
from gomoku_zero.evaluate import evaluate_models
from gomoku_zero.gomoku_rules import GomokuRules
from gomoku_zero.mcts import MCTS
from gomoku_zero.policy_value_model import PolicyValueModel
from gomoku_zero.replay_buffer import ReplayBuffer, TrainingSample
from gomoku_zero.self_play import generate_self_play_game, generate_self_play_games
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

    def test_parallel_self_play_generates_games(self) -> None:
        model = self._model()
        samples, stats = generate_self_play_games(
            model=model,
            rules=self.rules,
            games=2,
            n_playout=1,
            c_puct=self.config.c_puct,
            device="cpu",
            temp=self.config.self_play_temp,
            temp_threshold=self.config.self_play_temp_threshold,
            candidate_distance=1,
            tactical_shortcuts=True,
            workers=2,
            seed=123,
        )
        self.assertEqual(len(stats), 2)
        self.assertGreater(len(samples), 0)

    def test_game_end_after_move_matches_full_scan(self) -> None:
        board = np.zeros((5, 5), dtype=np.int8)
        board[2, 0:5] = 1
        self.assertEqual(self.rules.game_end_after_move(board, 2 * 5 + 4, 1), (True, 1))
        self.assertEqual(self.rules.game_end_after_move(board, 2 * 5 + 4, 1), self.rules.game_end(board))

    def test_candidate_moves_and_tactical_shortcut(self) -> None:
        board = np.zeros((5, 5), dtype=np.int8)
        board[2, 0:4] = 1
        candidates = self.rules.candidate_moves(board, distance=1)
        self.assertIn(2 * 5 + 4, candidates.tolist())

        mcts = MCTS(
            model=self._model(),
            n_playout=50,
            rules=self.rules,
            candidate_distance=1,
            tactical_shortcuts=True,
        )
        move = mcts.get_action(board, current_player=1, temp=1e-3, return_probs=False)
        self.assertEqual(move, 2 * 5 + 4)

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

    def test_value_head_keeps_gradient_when_conv_outputs_are_negative(self) -> None:
        model = self._model()
        value_conv = model.value_head.net[0]
        value_conv.weight.data.fill_(-1.0)
        value_conv.bias.data.fill_(-1.0)

        inputs = torch.rand(2, 2, 5, 5)
        _, values = model(inputs)
        loss = values.sum()
        loss.backward()

        self.assertGreater(float(value_conv.weight.grad.abs().sum()), 0.0)
        self.assertEqual(values.shape, (2, 1))

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
                explore_temp=1.0,
                temp_threshold=2,
                candidate_distance=1,
                tactical_shortcuts=True,
            )
            self.assertEqual(result.games, 2)
            self.assertEqual(result.current_wins + result.previous_wins + result.draws, 2)

            parallel_result = evaluate_models(
                current_model=model,
                previous_model=previous_model,
                rules=self.rules,
                games=2,
                n_playout=1,
                c_puct=5.0,
                device="cpu",
                temp=1e-3,
                explore_temp=1.0,
                temp_threshold=2,
                candidate_distance=1,
                tactical_shortcuts=True,
                workers=2,
                seed=456,
            )
            self.assertEqual(parallel_result.games, 2)
            self.assertEqual(
                parallel_result.current_wins + parallel_result.previous_wins + parallel_result.draws,
                2,
            )
        finally:
            for path in tmpdir.glob("*"):
                path.unlink()
            tmpdir.rmdir()

    def test_find_latest_and_load_training_checkpoint(self) -> None:
        tmpdir = Path(".test_resume_checkpoints")
        tmpdir.mkdir(exist_ok=True)
        try:
            config = TrainConfig(
                board_height=5,
                board_width=5,
                channels=8,
                checkpoint_dir=str(tmpdir),
                device="cpu",
            )
            model = self._model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            save_checkpoint(model, optimizer, config, iteration=None)
            first_path = save_checkpoint(model, optimizer, config, iteration=1)
            second_path = save_checkpoint(model, optimizer, config, iteration=2)

            latest = find_latest_iteration_checkpoint(config)
            self.assertEqual(latest, (2, second_path))

            resumed_model = self._model()
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
            iteration = load_training_checkpoint(
                first_path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                device="cpu",
            )
            self.assertEqual(iteration, 1)
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
