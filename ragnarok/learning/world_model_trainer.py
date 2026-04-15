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

        # Split the optimizer into two param groups so a cross-dim transfer
        # can scale the LR on the env-agnostic subset (core.gru / prior /
        # posterior) independently of the per-env IO (encoder, pre_gru,
        # decoder, reward/continue predictors). After a transfer, the
        # transferable subset is warm-started from source weights we want
        # to preserve; the per-env IO is fresh-random and needs full LR
        # to catch up. A flat Adam would burn through the transferred
        # priors in ~hundreds of steps (Bug E smoke observation).
        self._base_lr = lr
        self._transferable_lr_scale = 1.0
        self._warmup_episodes_remaining = 0
        self.optimizer = torch.optim.Adam([
            {"params": list(rssm.transferable_params()),
             "lr": lr, "name": "transferable"},
            {"params": list(rssm.non_transferable_params()),
             "lr": lr, "name": "io"},
        ], eps=1e-5)

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

    # ── Transferable-subset LR scaling (Bug E Phase 5 fix) ────────────
    #
    # When ``RagnarokAgent.try_transfer`` performs a cross-dim load it
    # warm-starts the transferable RSSM subset from a source skill but
    # leaves the per-env IO fresh-random. The IO needs full LR to learn
    # the new obs/action layout, but the transferable subset's source
    # weights would be wiped out in a few hundred Adam steps if it ran
    # at the same rate. We therefore drop the transferable group's LR
    # for `warmup_episodes` and let the IO catch up first.

    def set_transferable_lr_scale(self, scale: float, warmup_episodes: int):
        """Scale the LR on the env-agnostic RSSM param group.

        Called by RagnarokAgent immediately after a successful cross-dim
        transfer. The scale applies for ``warmup_episodes`` calls to
        ``step_episode()``, then snaps back to 1.0.

        Args:
            scale: multiplier applied to the base LR for the
                transferable group. Typically 0.1.
            warmup_episodes: number of episodes during which the scale
                stays in effect. After the counter expires the LR
                snaps back to ``self._base_lr``.
        """
        self._transferable_lr_scale = scale
        self._warmup_episodes_remaining = warmup_episodes
        for g in self.optimizer.param_groups:
            if g["name"] == "transferable":
                g["lr"] = self._base_lr * scale

    def step_episode(self):
        """Decrement the warmup counter and restore LR when it expires.

        Call exactly once per episode end. RagnarokAgent does this
        unconditionally — when no warmup is active the call is a no-op.
        """
        if self._warmup_episodes_remaining > 0:
            self._warmup_episodes_remaining -= 1
            if self._warmup_episodes_remaining == 0:
                self._transferable_lr_scale = 1.0
                for g in self.optimizer.param_groups:
                    if g["name"] == "transferable":
                        g["lr"] = self._base_lr

    def get_transferable_lr(self) -> float:
        """Current LR on the transferable param group (for tests + logging)."""
        for g in self.optimizer.param_groups:
            if g["name"] == "transferable":
                return float(g["lr"])
        raise RuntimeError("transferable param group missing")
