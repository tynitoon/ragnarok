"""Policy gradient training from real environment episodes.

Supports both discrete (Categorical A2C) and continuous (Gaussian A2C)
action spaces. The direct policy operates on raw observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ragnarok.infrastructure.device import DEVICE


class DirectPolicyNet(nn.Module):
    """Simple actor-critic for discrete actions on raw observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """Returns (action_logits, value)."""
        features = self.shared(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """Select action from observation."""
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1).item()
        return torch.distributions.Categorical(logits=logits).sample().item()


class ContinuousPolicyNet(nn.Module):
    """Actor-critic for continuous actions with tanh-squashed Gaussian.

    Outputs mean and log_std for each action dimension. Actions are
    squashed through tanh and rescaled to [action_low, action_high].
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        super().__init__()
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

        # Action rescaling: tanh outputs [-1, 1], rescale to [low, high]
        if action_low is not None and action_high is not None:
            self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        else:
            self.register_buffer("action_low", -torch.ones(action_dim))
            self.register_buffer("action_high", torch.ones(action_dim))

        # Initialize log_std to a reasonable value
        nn.init.constant_(self.logstd_head.bias, -0.5)

    def forward(self, obs: torch.Tensor):
        """Returns (mean, log_std, value)."""
        features = self.shared(obs)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features).clamp(-5.0, 2.0)
        value = self.critic_head(features).squeeze(-1)
        return mean, logstd, value

    def get_dist(self, obs: torch.Tensor):
        """Get action distribution and value."""
        mean, logstd, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, logstd.exp())
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select action, returns numpy array rescaled to action bounds."""
        mean, logstd, _ = self.forward(obs)
        if deterministic:
            raw = mean
        else:
            dist = torch.distributions.Normal(mean, logstd.exp())
            raw = dist.sample()
        # Squash through tanh and rescale
        squashed = torch.tanh(raw)
        action = self._rescale(squashed)
        return action.squeeze(0).detach().cpu().numpy()

    def _rescale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        """Rescale tanh output [-1, 1] to [action_low, action_high]."""
        return self.action_low + (tanh_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def log_prob(self, obs: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of pre-tanh action."""
        mean, logstd, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, logstd.exp())
        lp = dist.log_prob(raw_action).sum(dim=-1)
        # Tanh correction: log|det(d tanh/d raw)|
        lp -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)
        return lp

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Approximate entropy (Gaussian entropy, ignoring tanh)."""
        _, logstd, _ = self.forward(obs)
        return (0.5 + 0.5 * np.log(2 * np.pi) + logstd).sum(dim=-1)


class RealExperienceTrainer:
    """A2C trainer from real environment episodes.

    Automatically handles discrete or continuous action spaces.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 discrete: bool = True,
                 gamma: float = 0.99, entropy_coeff: float = 0.01,
                 lr: float = 3e-4, grad_clip: float = 0.5,
                 reward_shaper=None,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.action_dim = action_dim
        self.discrete = discrete
        self.reward_shaper = reward_shaper

        if discrete:
            self.policy = DirectPolicyNet(obs_dim, action_dim).to(DEVICE)
        else:
            self.policy = ContinuousPolicyNet(
                obs_dim, action_dim, action_low=action_low, action_high=action_high
            ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_and_train(self, env, deterministic: bool = False):
        """Collect one episode and train on it.

        Returns (total_reward, metrics, episode_data).
        """
        if self.discrete:
            return self._collect_discrete(env, deterministic)
        else:
            return self._collect_continuous(env, deterministic)

    def _collect_discrete(self, env, deterministic: bool = False):
        """Discrete action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            action_idx = action.item()
            action_onehot = env.action_to_onehot(action_idx)

            observations.append(obs.copy())
            actions.append(action_onehot)

            next_obs, reward, terminated, truncated, _ = env.step(action_onehot)

            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        return total_reward, self._train_a2c(log_probs, values, entropies, rewards), episode_data

    def _collect_continuous(self, env, deterministic: bool = False):
        """Continuous action collection + training."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        raw_actions = []  # Pre-tanh actions for log_prob computation
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, logstd, value = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, logstd.exp())

            if deterministic:
                raw_action = mean
            else:
                raw_action = dist.rsample()  # Reparameterized

            lp = dist.log_prob(raw_action).sum(dim=-1)
            # Tanh correction
            squashed = torch.tanh(raw_action)
            lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

            log_probs.append(lp)
            values.append(value)
            entropies.append(self.policy.entropy(obs_t))
            raw_actions.append(raw_action)

            # Rescale to environment bounds and step
            env_action = self.policy._rescale(squashed).squeeze(0).detach().cpu().numpy()

            observations.append(obs.copy())
            actions.append(env_action.copy())

            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)

        if deterministic or len(rewards) < 2:
            return total_reward, {}, episode_data

        return total_reward, self._train_a2c(log_probs, values, entropies, rewards), episode_data

    def _build_episode_data(self, observations, actions, rewards):
        """Build episode data tuple for replay buffer."""
        dones = [0.0] * len(rewards)
        dones[-1] = 1.0
        return (
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def _train_a2c(self, log_probs, values, entropies, rewards) -> dict:
        """Shared A2C training on collected episode data."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)
        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "real/actor_loss": actor_loss.item(),
            "real/critic_loss": critic_loss.item(),
            "real/entropy": entropies.mean().item(),
            "real/mean_return": returns.mean().item(),
        }

    def collect_batch_and_train(self, env, batch_episodes: int = 4):
        """Collect multiple episodes, then train on all data at once.

        Much lower variance than single-episode training for continuous envs.
        Returns (mean_reward, metrics, last_episode_data).
        """
        all_log_probs, all_values, all_entropies, all_rewards = [], [], [], []
        total_rewards = []
        last_episode_data = None

        for _ in range(batch_episodes):
            if self.discrete:
                result = self._collect_discrete_data(env)
            else:
                result = self._collect_continuous_data(env)

            ep_reward, log_probs, values, entropies, rewards, episode_data = result
            total_rewards.append(ep_reward)
            all_log_probs.extend(log_probs)
            all_values.extend(values)
            all_entropies.extend(entropies)
            all_rewards.extend(rewards)
            last_episode_data = episode_data

        if len(all_rewards) < 2:
            return float(np.mean(total_rewards)), {}, last_episode_data

        metrics = self._train_a2c(all_log_probs, all_values, all_entropies, all_rewards)
        return float(np.mean(total_rewards)), metrics, last_episode_data

    def _collect_continuous_data(self, env):
        """Collect one continuous episode without training. Returns components."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, logstd, value = self.policy(obs_t)
            dist = torch.distributions.Normal(mean, logstd.exp())
            raw_action = dist.rsample()

            lp = dist.log_prob(raw_action).sum(dim=-1)
            squashed = torch.tanh(raw_action)
            lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)

            log_probs.append(lp)
            values.append(value)
            entropies.append(self.policy.entropy(obs_t))

            env_action = self.policy._rescale(squashed).squeeze(0).detach().cpu().numpy()
            observations.append(obs.copy())
            actions.append(env_action.copy())

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, log_probs, values, entropies, rewards, episode_data

    def _collect_discrete_data(self, env):
        """Collect one discrete episode without training. Returns components."""
        obs = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        observations, actions = [], []
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, value = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            action_idx = action.item()
            action_onehot = env.action_to_onehot(action_idx)
            observations.append(obs.copy())
            actions.append(action_onehot)

            next_obs, reward, terminated, truncated, _ = env.step(action_onehot)
            train_reward = reward
            if self.reward_shaper is not None:
                raw_obs = getattr(env, 'last_raw_obs', next_obs)
                train_reward = self.reward_shaper(obs, reward, raw_obs)

            rewards.append(train_reward)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        episode_data = self._build_episode_data(observations, actions, rewards)
        return total_reward, log_probs, values, entropies, rewards, episode_data

    def evaluate(self, env, episodes: int = 5) -> float:
        """Evaluate policy without training."""
        rewards = []
        for _ in range(episodes):
            r, _, _ = self.collect_and_train(env, deterministic=True)
            rewards.append(r)
        return float(np.mean(rewards))

    def get_action_onehot(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action as one-hot from raw observation (discrete only)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_idx = self.policy.act(obs_t, deterministic)
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action_idx] = 1.0
        return onehot
