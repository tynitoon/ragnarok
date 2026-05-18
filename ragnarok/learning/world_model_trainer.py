"""World model (RSSM) training loop."""

import numpy as np
import torch
import torch.nn.functional as F
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE, mark_step


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
        mark_step()  # XLA: materialize the lazy graph (no-op on CUDA/CPU)

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

    def train_world_model_on_rollout(self, batch, epochs: int = 5,
                                     n_minibatches: int = 8,
                                     seq_chunk: int = 32) -> dict[str, float]:
        """Train the RSSM directly on a fixed-shape on-device RolloutBatch.

        The device-path counterpart of ``train()`` / ``train_step()``:
        instead of sampling padded subsequences from the host
        ``ReplayBuffer`` and transferring them, it consumes a
        ``RolloutBatch`` (N env rows x T steps, all device tensors) from
        ``ragnarok.learning.rollout.collect_rollout`` — no host sampling,
        no host->device transfer.

        On-policy world-model training: one rollout at large N is hundreds
        of thousands of transitions from N independent envs at random
        episode phases — already far more decorrelated than the gym
        ``ReplayBuffer``, and the WM's reconstruction/dynamics loss does
        not drift with the policy the way a value target does. ``epochs``
        passes with reshuffled env-row minibatches supply the gradient
        diversity a replay buffer otherwise would.

        Truncated unroll depth (TPU-divergence fix). The GRU is unrolled
        over at most ``seq_chunk`` steps, not the full rollout length T:
        each row is split along T into consecutive chunks, every chunk is
        an independent ``RSSM.loss`` call (fresh zero state), and the
        per-chunk losses are length-weighted so the aggregate is the mean
        over all T steps. Backprop-through-time therefore never reaches
        deeper than ``seq_chunk``. Training the WM on the full 128-step
        unroll converges on a CUDA GPU but DIVERGES on the TPU (KL
        1.3 -> 70) — even at fp32-faithful matmul precision a 128-step
        recurrent backward is a marginal instability the TPU tips over
        around rollout ~12. A TPU diagnostic swept the unroll depth on
        CartPole: 32 steps converged cleanly, 64 was marginal, 128 blew
        up — so ``seq_chunk`` defaults to 32 (the calibrated gym world
        model used 50-step subsequences, but the TPU needs shorter). A
        ``seq_chunk >= T`` collapses to a single full-length unroll (the
        pre-fix behaviour).

        A rollout row spans several auto-reset episodes, so ``RSSM.loss``
        runs with ``full_sequence_valid=True`` (every step real, no
        padding) and ``observe`` resets the GRU at each episode seam.
        Minibatching is along the env dimension N; the chunk split is
        along T. A chunk boundary mid-episode restarts the GRU from zero
        state — the standard truncated-BPTT approximation, and how the
        gym path's sampled subsequences already behaved.

        Discrete actions arrive as (N, T) indices and are one-hot encoded;
        continuous actions (Stage 4) arrive as (N, T, action_dim) already.
        """
        if batch.actions.dim() == 2:
            actions = F.one_hot(batch.actions.long(),
                                self.rssm.action_dim).float()
        else:
            actions = batch.actions

        N, T = batch.obs.shape[0], batch.obs.shape[1]
        assert N % n_minibatches == 0, (
            f"rollout env count {N} must be divisible by n_minibatches "
            f"{n_minibatches}")
        mb = N // n_minibatches
        chunk = seq_chunk

        last: dict[str, torch.Tensor] = {}
        for _ in range(epochs):
            # Shuffle env rows. rand().argsort(), NOT randperm — randperm's
            # int64 RNG emits an s64 HLO the TPU's X64 pass cannot lower.
            perm = torch.rand(N, device=DEVICE).argsort()
            for k in range(n_minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                o, a = batch.obs[idx], actions[idx]
                r, d = batch.rewards[idx], batch.dones[idx]
                self.optimizer.zero_grad()
                # Accumulate the gradient over consecutive T-axis chunks.
                # Each chunk is an independent unroll of at most `chunk`
                # steps, so BPTT never reaches deeper than that.
                agg: dict[str, torch.Tensor] = {}
                for s in range(0, T, chunk):
                    e = min(s + chunk, T)
                    losses = self.rssm.loss(
                        o[:, s:e], a[:, s:e], r[:, s:e], d[:, s:e],
                        self.kl_weight, self.free_nats,
                        full_sequence_valid=True)
                    w = (e - s) / T          # length-weight -> mean over T
                    (losses["total_loss"] * w).backward()
                    for name, val in losses.items():
                        agg[name] = agg.get(name, 0.0) + val.detach() * w
                torch.nn.utils.clip_grad_norm_(self.rssm.parameters(),
                                               self.grad_clip)
                self.optimizer.step()
                mark_step()  # XLA: one graph per minibatch update
                last = agg

        return {f"wm/{k}": v.item() for k, v in last.items()}

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

    def reset_transferable_optimizer_state(self):
        """Drop Adam moment estimates for the transferable param group.

        Architecture-review concern (Bug E v2, 2026-04-15): after a
        cross-dim ``load_transferable_state_dict``, the Adam moments
        ``exp_avg`` / ``exp_avg_sq`` for those params still reflect the
        gradient statistics of whatever target-env training had happened
        before the load (typically very little — most transfers happen
        early). More importantly, the loaded weights are now from the
        SOURCE task; their gradients on the target task have a different
        scale, so any pre-load second-moment estimate is stale.

        Without a reset, Adam's first few updates after the load are
        either (a) under-damped (if pre-load v_t was tiny → effective
        step size is much larger than the nominal `lr * scale`) or
        (b) over-damped (if pre-load v_t was huge from random-init
        gradients → updates barely move). Either way, the LR-warmup
        knob's nominal 0.1× scale is a misleading lower bound on the
        actual update magnitude.

        Resetting clears `exp_avg`, `exp_avg_sq`, and `step` for every
        param in the ``transferable`` group. The next backward pass
        re-initializes them via Adam's normal first-step path. The IO
        group is left untouched — it never had its weights swapped.
        """
        for g in self.optimizer.param_groups:
            if g.get("name") != "transferable":
                continue
            for p in g["params"]:
                if p in self.optimizer.state:
                    # Drop the entry entirely; Adam will re-create it on
                    # the next ``step()`` with step=0, exp_avg=zeros,
                    # exp_avg_sq=zeros — exactly the fresh-start state.
                    del self.optimizer.state[p]

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
