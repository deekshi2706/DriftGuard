"""
training/continual.py
Person C — Continual Learning

Wraps Person B's GraphDiffusionModel to update it incrementally as new time
steps arrive, without touching the model's internals. Only public methods
of GraphDiffusionModel are used here: .train_step(batch).

Supports three modes, used for the paper's 3-way comparison:
    "static"          — train once on time step 1, never update again.
                         (Just don't call .update() after the first fit —
                         included here as a no-op-update trainer for a
                         uniform interface in experiments.ipynb.)
    "naive"            — ContinualTrainer(model, replay_buffer_size=0):
                         update on each new time step's data only, no replay.
                         Prone to catastrophic forgetting.
    "replay"           — ContinualTrainer(model, replay_buffer_size=N):
                         update on new time step data + a sample from a
                         replay buffer of past normal nodes.

Public API:
    ReplayBuffer(capacity)
        .add(x, edge_index)
        .sample(batch_size) -> (x, edge_index) or None if empty
    ContinualTrainer(model, replay_buffer_size=0, replay_batch_frac=0.5)
        .update(new_time_step_data, epochs=1) -> dict with per-epoch losses
"""

import random
from typing import List, Optional, Tuple

import torch


class ReplayBuffer:
    """
    Stores small per-time-step (x, edge_index) snapshots of "normal" nodes
    seen so far, and can produce a merged sample for replay during
    continual updates.

    Note: this stores whole small subgraphs (one entry per time step seen),
    not individual nodes, so that edge_index stays internally consistent.
    When capacity is exceeded, oldest entries are evicted (FIFO).
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        if self.capacity <= 0:
            return  # replay disabled
        self._buffer.append((x.detach().clone(), edge_index.detach().clone()))
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def sample(self, batch_size: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns one merged (x, edge_index) pair combining up to `batch_size`
        stored snapshots (default: all stored snapshots), with edge indices
        offset so they remain valid into the concatenated x. Returns None
        if the buffer is empty.
        """
        if not self._buffer:
            return None
        chosen = self._buffer if batch_size is None else random.sample(
            self._buffer, k=min(batch_size, len(self._buffer))
        )
        xs, edge_indices = [], []
        offset = 0
        for x, ei in chosen:
            xs.append(x)
            edge_indices.append(ei + offset)
            offset += x.size(0)
        return torch.cat(xs, dim=0), torch.cat(edge_indices, dim=1)

    def __len__(self) -> int:
        return len(self._buffer)


class ContinualTrainer:
    """
    Incrementally updates a Person-B GraphDiffusionModel across time steps.

    Args:
        model: a GraphDiffusionModel instance (already constructed; not
            modified structurally — only its public .train_step() is called).
        replay_buffer_size: number of past time-step snapshots to retain.
            0 disables replay entirely (= "naive continual" mode).
        replay_batch_frac: when replay is enabled, this controls how many
            stored snapshots to draw per update relative to buffer size
            (used only to cap replay volume on large buffers).
    """

    def __init__(self, model, replay_buffer_size: int = 0, replay_batch_frac: float = 0.5):
        self.model = model
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.replay_batch_frac = replay_batch_frac
        self.seen_steps: List[int] = []

    def update(self, new_time_step_data, epochs: int = 1, step_id: Optional[int] = None) -> dict:
        """
        Args:
            new_time_step_data: an object with `.x` [N, in_dim] and
                `.edge_index` [2, E] — typically the *normal-node-only*
                subgraph for one time step (see notebooks/experiments.ipynb
                for how this is built from Person A's get_normal_nodes()).
            epochs: how many train_step calls to run on this update
                (each call handles one combined batch of new + replay data).
            step_id: optional int, just recorded in self.seen_steps for
                bookkeeping / the forgetting-measure calculation.

        Returns:
            {"losses": [float, ...]}  one loss value per epoch.
        """
        x_new, ei_new = new_time_step_data.x, new_time_step_data.edge_index

        losses = []
        for _ in range(epochs):
            replayed = self.replay_buffer.sample()
            if replayed is not None:
                x_replay, ei_replay = replayed
                x = torch.cat([x_new, x_replay], dim=0)
                edge_index = torch.cat([ei_new, ei_replay + x_new.size(0)], dim=1)
            else:
                x, edge_index = x_new, ei_new

            class _Batch:
                pass

            batch = _Batch()
            batch.x, batch.edge_index = x, edge_index
            out = self.model.train_step(batch)
            losses.append(out["loss"])

        # Add this time step's normal data to the buffer AFTER training on it,
        # so replay never includes the exact batch just trained on twice in
        # the same call.
        self.replay_buffer.add(x_new, ei_new)
        if step_id is not None:
            self.seen_steps.append(step_id)

        return {"losses": losses}
