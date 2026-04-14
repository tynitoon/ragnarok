"""World model (RSSM) training loop."""

import numpy as np
import torch
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE


class WorldModelTrainer:
    """Trains the RSSM world model on experience from the replay buffer."""

    def __init__(self, rssm: RSSM, replay_buffer: ReplayBuffer,
                 lr: float = 3e-4, grad_clip: float = 100.0,
                 kl_weight: float = 0.1, free_nats: float = 1.0,
                 batch_size: int = 50, seq_length: int = 50,
                 shuffle_transitions: bool = False):
        self.rssm = rssm
        self.buffer = replay_buffer
        self.kl_weight = kl_weight
        self.free_nats = free_nats
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.grad_clip = grad_clip
        # A9 mechanism-isolation ablation (preregistration §5 ablations).
        # When True, shuffles `obs[:, t]` for t >= 1 across the batch dim
        # with an independent permutation per t. Breaks the dynamics
        # (s_{t-1}, a_{t-1}) → s_t while preserving marginals, so a WM
        # trained with shuffle cannot have learned transition structure.
        # Transfer using such a WM isolates the architectural contribution
        # from the learned-dynamics contribution.
        self.shuffle_transitions = shuffle_transitions

        self.optimizer = torch.optim.Adam(rssm.parameters(), lr=lr, eps=1e-5)

    def _shuffle_next_state_targets(self, obs: np.ndarray) -> np.ndarray:
        """Cross-trajectory shuffle of next-state targets (A9 ablation).

        For each timestep t >= 1, apply an independent random permutation
        over the batch dim to `obs[:, t]`. The first timestep (t=0) stays
        unshuffled so initial-state encoding still matches the first
        action. Downstream, each (obs[:, t-1], action[:, t-1]) → obs[:, t]
        mapping is broken — the RSSM cannot learn real dynamics.

        Rewards and dones stay paired with their original trajectories
        (prereg thresholds.json: "cross-trajectory shuffle of next-state
        targets") — only the obs reconstruction target is shuffled.
        """
        B, T = obs.shape[0], obs.shape[1]
        shuffled = obs.copy()
        for t in range(1, T):
            perm = np.random.permutation(B)
            shuffled[:, t] = obs[perm, t]
        return shuffled

    def train_step(self) -> dict[str, float]:
        """Single training step: sample batch, compute loss, update weights."""
        if self.buffer.num_episodes == 0:
            return {}

        # Sample sequences from replay buffer
        obs, actions, rewards, dones = self.buffer.sample_sequences(
            self.batch_size, self.seq_length
        )

        if self.shuffle_transitions:
            obs = self._shuffle_next_state_targets(obs)

        # Convert to tensors
        obs_t = torch.tensor(obs, device=DEVICE)
        act_t = torch.tensor(actions, device=DEVICE)
        rew_t = torch.tensor(rewards, device=DEVICE)
        done_t = torch.tensor(dones, device=DEVICE)

        # Compute loss
        losses = self.rssm.loss(obs_t, act_t, rew_t, done_t,
                                self.kl_weight, self.free_nats)

        # Backprop
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), self.grad_clip)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self, steps: int) -> dict[str, float]:
        """Train for multiple steps, return average metrics."""
        if self.buffer.num_episodes == 0:
            return {}

        totals: dict[str, float] = {}
        count = 0

        for _ in range(steps):
            metrics = self.train_step()
            if metrics:
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                count += 1

        if count == 0:
            return {}
        return {k: v / count for k, v in totals.items()}
