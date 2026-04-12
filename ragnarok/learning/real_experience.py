"""Policy gradient training from real environment episodes.

Two modes:
1. Latent mode: uses RSSM-encoded states (when world model is trained)
2. Direct mode: uses raw observations (always works, baseline approach)

The direct mode provides the "reality check" that ensures the agent
actually learns from real environment outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ragnarok.infrastructure.device import DEVICE


class DirectPolicyNet(nn.Module):
    """Simple actor-critic that works directly on raw observations.

    This is the "fallback brain" that ensures the agent can always learn,
    even when the world model isn't good enough for dream training.
    """

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


class RealExperienceTrainer:
    """A2C trainer from real environment episodes.

    Uses a direct policy network on raw observations for stable learning.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 gamma: float = 0.99, entropy_coeff: float = 0.01,
                 lr: float = 3e-4, grad_clip: float = 0.5):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.action_dim = action_dim

        self.policy = DirectPolicyNet(obs_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_and_train(self, env, deterministic: bool = False) -> tuple[float, dict[str, float]]:
        """Collect one episode and train on it. Returns (total_reward, metrics)."""
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

            obs, reward, terminated, truncated, _ = env.step(action_onehot)
            rewards.append(reward)
            done = terminated or truncated
            total_reward += reward

        if deterministic or len(rewards) < 2:
            return total_reward, {}

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)
        advantages = returns - values.detach()

        # Combined loss
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return total_reward, {
            "real/actor_loss": actor_loss.item(),
            "real/critic_loss": critic_loss.item(),
            "real/entropy": entropies.mean().item(),
            "real/mean_return": returns.mean().item(),
        }

    def evaluate(self, env, episodes: int = 5) -> float:
        """Evaluate policy without training."""
        rewards = []
        for _ in range(episodes):
            r, _ = self.collect_and_train(env, deterministic=True)
            rewards.append(r)
        return float(np.mean(rewards))

    def get_action_onehot(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action as one-hot from raw observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action_idx = self.policy.act(obs_t, deterministic)
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action_idx] = 1.0
        return onehot
