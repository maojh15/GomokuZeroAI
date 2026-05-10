from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .policy_value_model import PolicyValueModel
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class TrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    steps: int


def train_one_iteration(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    epochs: int,
    device: torch.device | str,
    log_interval: int = 10,
    human_replay_buffer: ReplayBuffer | None = None,
) -> TrainMetrics:
    if len(replay_buffer) == 0:
        raise ValueError("Replay buffer is empty; generate self-play data before training.")

    device = torch.device(device)
    model.train()
    policy_losses: list[float] = []
    value_losses: list[float] = []
    total_losses: list[float] = []
    step = 0

    for epoch in range(epochs):
        for states, policy_targets, value_targets in _iter_training_batches(
            replay_buffer=replay_buffer,
            human_replay_buffer=human_replay_buffer,
            batch_size=batch_size,
        ):
            states_tensor = torch.from_numpy(states).to(device)
            policy_targets_tensor = torch.from_numpy(policy_targets).to(device)
            value_targets_tensor = torch.from_numpy(value_targets).to(device)

            policy_logits, values = model(states_tensor)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets_tensor * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values.squeeze(1), value_targets_tensor)
            loss = policy_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            total_losses.append(float(loss.item()))

            if log_interval > 0 and step % log_interval == 0:
                print(
                    f"train step={step} epoch={epoch + 1}/{epochs} "
                    f"policy_loss={policy_losses[-1]:.4f} "
                    f"value_loss={value_losses[-1]:.4f} total_loss={total_losses[-1]:.4f}"
                )

    return TrainMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        total_loss=float(np.mean(total_losses)),
        steps=step,
    )


def _iter_training_batches(
    replay_buffer: ReplayBuffer,
    human_replay_buffer: ReplayBuffer | None,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    replay_size = len(replay_buffer)
    human_size = 0 if human_replay_buffer is None else len(human_replay_buffer)
    total_size = replay_size + human_size
    order = np.random.permutation(total_size)

    for start in range(0, total_size, batch_size):
        batch_indices = order[start : start + batch_size]
        replay_indices = [int(index) for index in batch_indices if int(index) < replay_size]
        human_indices = [int(index) - replay_size for index in batch_indices if int(index) >= replay_size]

        batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if replay_indices:
            batches.append(replay_buffer.samples_by_indices(replay_indices))
        if human_indices and human_replay_buffer is not None:
            batches.append(human_replay_buffer.samples_by_indices(human_indices))

        if len(batches) == 1:
            yield batches[0]
            continue

        states = np.concatenate([batch[0] for batch in batches], axis=0)
        policies = np.concatenate([batch[1] for batch in batches], axis=0)
        values = np.concatenate([batch[2] for batch in batches], axis=0)
        shuffle = np.random.permutation(len(values))
        yield states[shuffle], policies[shuffle], values[shuffle]
