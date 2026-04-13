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
from collections import deque
import random

from ragnarok.infrastructure.device import DEVICE


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
    """Simple replay buffer for off-policy SAC."""

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(act), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(rew), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(done), dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


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
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay = SACReplayBuffer(buffer_capacity)
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def collect_and_train(self, env, updates_per_step: int = 1):
        """Collect one episode, train after each step.

        Returns (episode_reward, metrics, episode_data).
        """
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

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

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

        # --- Soft update target networks ---
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(p.data * self.tau)
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return {
            "sac/q1_loss": q1_loss.item(),
            "sac/q2_loss": q2_loss.item(),
            "sac/policy_loss": policy_loss.item(),
            "sac/alpha": self.alpha.item(),
            "sac/entropy": -log_prob.mean().item(),
        }

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
