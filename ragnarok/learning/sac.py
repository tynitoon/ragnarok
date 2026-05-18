"""Soft Actor-Critic (SAC) for continuous control.

Off-policy algorithm with:
- Twin Q-networks (reduces overestimation bias)
- Automatic entropy coefficient tuning
- Replay buffer for sample efficiency
- Tanh-squashed Gaussian policy

Much more effective than PPO/A2C for continuous envs like Pendulum.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE, mark_step


class QNetwork(nn.Module):
    """Q-function: maps (obs, action) -> Q-value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SACPolicy(nn.Module):
    """Squashed Gaussian policy for SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        super().__init__()
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)

        if action_low is not None and action_high is not None:
            self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        else:
            self.register_buffer("action_low", -torch.ones(action_dim))
            self.register_buffer("action_high", torch.ones(action_dim))

        nn.init.constant_(self.logstd_head.bias, -1.0)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features).clamp(-5.0, 2.0)
        return mean, logstd

    def sample(self, obs: torch.Tensor):
        """Sample action with log_prob (reparameterized, tanh-squashed).

        Returns (action_env_scale, log_prob).
        """
        mean, logstd = self.forward(obs)
        std = logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()

        # Tanh squash + log_prob correction
        squashed = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

        # Rescale to env bounds
        action = self._rescale(squashed)
        return action, log_prob

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select action for environment interaction."""
        with torch.no_grad():
            mean, logstd = self.forward(obs)
            if deterministic:
                action = self._rescale(torch.tanh(mean))
            else:
                action, _ = self.sample(obs)
        return action.squeeze(0).cpu().numpy()

    def _rescale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (tanh_action + 1.0) * 0.5 * (self.action_high - self.action_low)


class SACReplayBuffer:
    """Host-resident ring buffer for off-policy SAC.

    The five ring arrays live in numpy on the host; sample() draws a batch
    and moves only that fixed-size (batch_size × dim) batch to DEVICE.

    A device-resident buffer (the prior design) is faster on CUDA but fatal
    on XLA/TPU: add() writes at an ever-changing ring index (self._obs[i]=)
    and sample() draws torch.randint(0, self.size, ...) with an ever-growing
    bound — both bake a value that changes every step into the XLA graph, so
    it recompiles on every single step until the executable cache exhausts
    TPU memory (the SAC-phase OOM the smoke hit). Keeping the buffer on the
    host means every device-side tensor sample() produces is a fixed
    (batch_size, dim) shape — compiled once, reused. On CUDA the per-sample
    H2D copy of a small classic-control batch is negligible.

    API compatibility: the constructor signature is preserved (capacity
    only) so existing callers and tests (test_sac.py) keep working;
    dims are inferred from the first add() call.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        # numpy arrays allocated lazily on first add().
        self._obs: np.ndarray | None = None
        self._act: np.ndarray | None = None
        self._rew: np.ndarray | None = None
        self._next_obs: np.ndarray | None = None
        self._done: np.ndarray | None = None

    def _ensure_allocated(self, obs, action):
        if self._obs is not None:
            return
        obs_dim = np.asarray(obs, dtype=np.float32).reshape(-1).shape[0]
        act_dim = np.asarray(action, dtype=np.float32).reshape(-1).shape[0]
        self._obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self._rew = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self._ensure_allocated(obs, action)
        i = self.ptr
        # Pure host-side writes — no XLA op, so the ever-changing index i
        # never enters a graph.
        self._obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._act[i] = np.asarray(action, dtype=np.float32).reshape(-1)
        self._rew[i] = float(reward)
        self._next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self._done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Uniform sample (with replacement) from the live portion.

        Indices are drawn on the host; only the gathered fixed-size batch
        crosses to DEVICE, so the XLA graph sees one constant shape.
        """
        n = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=n)
        return (
            torch.tensor(self._obs[idx], device=DEVICE),
            torch.tensor(self._act[idx], device=DEVICE),
            torch.tensor(self._rew[idx], device=DEVICE),
            torch.tensor(self._next_obs[idx], device=DEVICE),
            torch.tensor(self._done[idx], device=DEVICE),
        )

    def __len__(self):
        return self.size


class DeviceSACBuffer:
    """Device-resident replay buffer for the accelerator-resident SAC path.

    SAC is off-policy and cannot train on just the latest rollout — it needs
    a buffer of past transitions. The device path keeps that buffer ON the
    accelerator: fixed-capacity device tensors fed directly from a
    RolloutBatch (collect_rollout output), with no per-rollout device->host
    transfer of the experience itself.

    XLA-safety — the trap SACReplayBuffer's docstring describes is a device
    buffer whose write index and sample bound get baked into the graph (they
    change every step -> recompile -> executable-cache OOM). This buffer
    sidesteps it the way SACReplayBuffer does for sampling, and extends it to
    writes: the write positions and the sample indices are generated on the
    HOST as small int arrays (numpy — cheap, no graph) and handed to the
    device as plain index tensors. Those are graph INPUTS, not baked
    constants, so index_copy / index_select compile once and are reused.
    Only the tiny index arrays cross the host boundary; the experience
    tensors never leave the device.

    sample() returns the same (obs, act, rew, next_obs, done) tuple as
    SACReplayBuffer, so SACTrainer._update consumes it unchanged.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._ptr = 0       # host int — never enters a graph
        self._size = 0      # host int
        self._obs = self._act = self._rew = None
        self._next_obs = self._done = None

    def _ensure_allocated(self, obs_dim: int, act_dim: int):
        if self._obs is not None:
            return
        c = self.capacity
        self._obs = torch.zeros(c, obs_dim, device=DEVICE)
        self._act = torch.zeros(c, act_dim, device=DEVICE)
        self._rew = torch.zeros(c, device=DEVICE)
        self._next_obs = torch.zeros(c, obs_dim, device=DEVICE)
        self._done = torch.zeros(c, device=DEVICE)

    def add_rollout(self, batch) -> None:
        """Fold a RolloutBatch's N*T transitions into the buffer.

        next_obs is the rollout shifted one step in time (last_obs closes the
        final step). After an auto-reset done the shifted next_obs is a
        fresh-episode state, but SAC's q_target multiplies the bootstrap by
        (1 - done), so a done transition's next_obs is never read — correct.
        """
        obs = batch.obs                                  # (N, T, obs_dim)
        act = batch.actions                              # (N, T, act_dim)
        if act.dim() == 2:                               # discrete index form
            act = act.unsqueeze(-1).float()
        next_obs = torch.cat(
            [obs[:, 1:], batch.last_obs.unsqueeze(1)], dim=1)
        n, t, obs_dim = obs.shape
        act_dim = act.shape[-1]
        self._ensure_allocated(obs_dim, act_dim)

        m = n * t
        assert m <= self.capacity, (
            f"rollout has {m} transitions > buffer capacity {self.capacity}")

        # Write positions: a contiguous ring span, host-computed — the
        # ever-changing _ptr never enters the XLA graph.
        pos_np = (self._ptr + np.arange(m)) % self.capacity
        pos = torch.tensor(pos_np, dtype=torch.long, device=DEVICE)
        self._obs = self._obs.index_copy(0, pos, obs.reshape(m, obs_dim))
        self._act = self._act.index_copy(0, pos, act.reshape(m, act_dim))
        self._rew = self._rew.index_copy(0, pos, batch.rewards.reshape(m))
        self._next_obs = self._next_obs.index_copy(
            0, pos, next_obs.reshape(m, obs_dim))
        self._done = self._done.index_copy(0, pos, batch.dones.reshape(m))

        self._ptr = (self._ptr + m) % self.capacity
        self._size = min(self._size + m, self.capacity)

    def sample(self, batch_size: int):
        """Uniform sample (with replacement). Indices are host-generated; only
        the fixed-shape gathered batch is a device tensor."""
        idx_np = np.random.randint(0, self._size, size=batch_size)
        idx = torch.tensor(idx_np, dtype=torch.long, device=DEVICE)
        return (
            self._obs.index_select(0, idx),
            self._act.index_select(0, idx),
            self._rew.index_select(0, idx),
            self._next_obs.index_select(0, idx),
            self._done.index_select(0, idx),
        )

    def __len__(self) -> int:
        return self._size


class SACTrainer:
    """Soft Actor-Critic trainer for continuous control.

    Manages policy, twin Q-networks, target networks, and auto-entropy.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None,
                 hidden: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 lr: float = 3e-4,
                 buffer_capacity: int = 100_000,
                 buffer=None,
                 batch_size: int = 256,
                 warmup_steps: int = 1000,
                 reward_shaper=None,
                 curiosity=None,
                 latent_curiosity=None):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_dim = action_dim
        self.reward_shaper = reward_shaper
        self.curiosity = curiosity  # ForwardPredictor (fallback)
        self.latent_curiosity = latent_curiosity  # LatentCuriosityModule or None

        # Policy
        self.policy = SACPolicy(
            obs_dim, action_dim, hidden, action_low, action_high
        ).to(DEVICE)

        # Twin Q-networks
        self.q1 = QNetwork(obs_dim, action_dim, hidden).to(DEVICE)
        self.q2 = QNetwork(obs_dim, action_dim, hidden).to(DEVICE)

        # Target Q-networks (Polyak-averaged)
        self.q1_target = QNetwork(obs_dim, action_dim, hidden).to(DEVICE)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic: -dim(A)
        self.log_alpha = torch.tensor(
            np.log(0.2), dtype=torch.float32, device=DEVICE, requires_grad=True
        )

        # Optimizers
        # Twin Q-networks share a single optimizer — q1 and q2 don't share
        # parameters, so this is mathematically identical to two separate
        # Adam optimizers, but cuts backward-pass launches in half on the
        # hot path. (Phase 2.3b throughput optim.)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Cached target-param lists for fused Polyak averaging — avoids
        # re-materializing the lists every _update() (called N×per step).
        self._q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self._q_target_params = (
            list(self.q1_target.parameters()) + list(self.q2_target.parameters())
        )

        # Replay buffer. Default: the host-resident SACReplayBuffer (gym
        # path). The device-SAC path passes a DeviceSACBuffer so the
        # experience stays on the accelerator.
        self.replay = (buffer if buffer is not None
                       else SACReplayBuffer(buffer_capacity))
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def collect_and_train(self, env, updates_per_step: int = 1):
        """Collect one episode, train after each step.

        Returns (episode_reward, metrics, episode_data).
        """
        # Track episodes for latent curiosity readiness
        if self.latent_curiosity is not None:
            self.latent_curiosity._episodes_seen += 1

        obs = env.reset()
        done = False
        episode_reward = 0.0
        observations, actions, rewards, next_observations = [], [], [], []
        metrics = {}

        while not done:
            # Select action
            if self.total_steps < self.warmup_steps:
                action = env.sample_random_action()
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = self.policy.act(obs_t, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Apply reward shaping using raw observations
            shaped_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                shaped_reward = self.reward_shaper(obs, reward, raw_obs)

            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(reward)
            next_observations.append(next_obs.copy())

            # Curiosity augments the shaped reward for replay buffer
            buffer_reward = shaped_reward
            if self.latent_curiosity is not None and self.latent_curiosity.rssm_ready:
                intrinsic = self.latent_curiosity.compute_batch_kl(
                    obs.reshape(1, -1), action.reshape(1, -1)
                )
                buffer_reward += intrinsic[0]
            elif self.curiosity is not None:
                intrinsic = self.curiosity.compute_intrinsic_rewards(
                    obs.reshape(1, -1), action.reshape(1, -1), next_obs.reshape(1, -1)
                )
                buffer_reward += intrinsic[0]

            self.replay.add(obs, action, buffer_reward, next_obs, float(done))
            episode_reward += reward
            self.total_steps += 1
            obs = next_obs

            # Train after warmup
            if self.total_steps >= self.warmup_steps and len(self.replay) >= self.batch_size:
                for _ in range(updates_per_step):
                    metrics = self._update()

        # Train curiosity predictor on the episode
        if self.curiosity is not None and len(observations) > 1:
            obs_arr = np.array(observations)
            act_arr = np.array(actions)
            next_arr = np.array(next_observations)
            metrics["curiosity_loss"] = self.curiosity.train_on_transitions(
                obs_arr, act_arr, next_arr
            )

        # Build episode data for RSSM replay buffer
        dones = [0.0] * len(rewards)
        dones[-1] = 1.0
        episode_data = (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
        return episode_reward, metrics, episode_data

    def _update(self) -> dict:
        """Single SAC update step."""
        obs, action, reward, next_obs, done = self.replay.sample(self.batch_size)

        # --- Q-function update ---
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + self.gamma * (1 - done) * q_next

        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        q_loss = q1_loss + q2_loss  # Single backward for both twin heads

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy update ---
        new_action, log_prob = self.policy.sample(obs)
        q1_new = self.q1(obs, new_action)
        q2_new = self.q2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- Entropy coefficient update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target networks (fused foreach ops) ---
        # Equivalent to: p_target = (1 - tau) * p_target + tau * p, for every
        # param of q1 and q2. _foreach_mul_ + _foreach_add_ fuses the
        # element-wise kernels across all tensors — two kernel launches for
        # ~20 parameter tensors instead of ~40 in the prior Python loop.
        with torch.no_grad():
            torch._foreach_mul_(self._q_target_params, 1 - self.tau)
            torch._foreach_add_(self._q_target_params, self._q_params, alpha=self.tau)

        mark_step()  # XLA: materialize the lazy graph (no-op on CUDA/CPU)

        # Batch .item() conversions in a single CUDA sync at the end of the
        # update step, instead of 5 separate syncs interleaved with ops.
        entropy = -log_prob.mean()
        return {
            "sac/q1_loss": q1_loss.item(),
            "sac/q2_loss": q2_loss.item(),
            "sac/policy_loss": policy_loss.item(),
            "sac/alpha": self.alpha.item(),
            "sac/entropy": entropy.item(),
        }

    # ── Device-resident path (Phase 2 TPU re-architecture) ────────────
    #
    # collect_and_train above runs ONE gym episode in a host loop with a
    # SAC update interleaved per step. The two methods below instead drive
    # SAC from a RolloutBatch — N envs x T steps collected batched on the
    # device by collect_rollout — feeding a DeviceSACBuffer that keeps the
    # experience on the accelerator.

    @torch.no_grad()
    def device_policy_fn(self, obs: torch.Tensor):
        """``policy_fn`` for ``collect_rollout`` — batched SAC action sampling.

        Before warmup, uniform-random actions across the env bounds (SAC's
        standard exploration warmup); after, the squashed-Gaussian policy.
        ``logp``/``value`` are returned as zeros — the SAC update recomputes
        everything from the replay buffer, so ``collect_rollout`` only needs
        the action itself.
        """
        n = obs.shape[0]
        if self.total_steps < self.warmup_steps:
            u = torch.rand(n, self.action_dim, device=obs.device)
            action = self.policy.action_low + u * (
                self.policy.action_high - self.policy.action_low)
        else:
            action, _ = self.policy.sample(obs)
        zeros = torch.zeros(n, device=obs.device)
        return action, zeros, zeros

    def train_on_rollout(self, batch, n_updates: int) -> dict:
        """SAC update from a fixed-shape on-device ``RolloutBatch``.

        Folds the rollout's N*T transitions into the (device-resident)
        replay buffer, then runs ``n_updates`` SAC updates sampling
        minibatches from it. Unlike the PPO / world-model on-rollout
        trainers (on-policy, fixed epochs over the rollout), SAC is
        off-policy: ``n_updates`` is the update-to-collected-step ratio, a
        tuning knob the caller sets — large-N parallel collection does not
        change how many gradient steps SAC needs to converge.

        Returns {} until warmup is satisfied (total_steps >= warmup_steps
        and the buffer holds >= batch_size transitions).
        """
        self.replay.add_rollout(batch)
        mark_step()  # XLA: flush the buffer-write graph here — during warmup
                     # no _update follows to flush it, so without this the
                     # index_copy ops would ride into the next collect graph.
        self.total_steps += batch.total_steps
        if (self.total_steps < self.warmup_steps
                or len(self.replay) < self.batch_size):
            return {}
        metrics: dict = {}
        for _ in range(n_updates):
            metrics = self._update()
        return metrics

    def evaluate(self, env, episodes: int = 5) -> float:
        """Evaluate deterministic policy."""
        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = self.policy.act(obs_t, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        return float(np.mean(rewards))
