from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrainingSample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        board_height: int,
        board_width: int,
        augment_symmetry: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.board_height = int(board_height)
        self.board_width = int(board_width)
        self.augment_symmetry = bool(augment_symmetry)
        self._samples: deque[TrainingSample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def samples_by_indices(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = [self._samples[int(index)] for index in indices]
        if not batch:
            raise ValueError("Cannot build an empty replay batch.")
        states = np.stack([sample.state for sample in batch]).astype(np.float32)
        policies = np.stack([sample.policy for sample in batch]).astype(np.float32)
        values = np.asarray([sample.value for sample in batch], dtype=np.float32)
        return states, policies, values

    def add_many(self, samples: Iterable[TrainingSample]) -> None:
        for sample in samples:
            if self.augment_symmetry:
                self._samples.extend(self._augment(sample))
            else:
                self._samples.append(self._normalize(sample))

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._samples) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        size = min(int(batch_size), len(self._samples))
        indices = np.random.choice(len(self._samples), size=size, replace=False)
        return self.samples_by_indices(indices)

    def _normalize(self, sample: TrainingSample) -> TrainingSample:
        return TrainingSample(
            state=np.ascontiguousarray(sample.state, dtype=np.float32),
            policy=np.ascontiguousarray(sample.policy, dtype=np.float32),
            value=float(sample.value),
        )

    def _augment(self, sample: TrainingSample) -> list[TrainingSample]:
        state = np.asarray(sample.state, dtype=np.float32)
        policy_board = np.asarray(sample.policy, dtype=np.float32).reshape(
            self.board_height,
            self.board_width,
        )

        augmented: list[TrainingSample] = []
        for k in range(4):
            rotated_state = np.rot90(state, k=k, axes=(1, 2))
            rotated_policy = np.rot90(policy_board, k=k)
            augmented.append(self._make_sample(rotated_state, rotated_policy, sample.value))

            flipped_state = np.flip(rotated_state, axis=2)
            flipped_policy = np.fliplr(rotated_policy)
            augmented.append(self._make_sample(flipped_state, flipped_policy, sample.value))
        return augmented

    @staticmethod
    def _make_sample(state: np.ndarray, policy_board: np.ndarray, value: float) -> TrainingSample:
        return TrainingSample(
            state=np.ascontiguousarray(state, dtype=np.float32),
            policy=np.ascontiguousarray(policy_board.reshape(-1), dtype=np.float32),
            value=float(value),
        )
