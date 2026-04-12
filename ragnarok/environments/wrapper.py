"""Unified environment wrapper for Gymnasium environments."""

import numpy as np
import gymnasium as gym
from ragnarok.core.normalizer import RunningNormalizer


class RagnarokEnv:
    """Wraps a Gymnasium environment with observation normalization
    and action space abstraction."""

    def __init__(self, env_name: str, normalizer: RunningNormalizer | None = None,
                 seed: int | None = None):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.seed = seed

        # Determine dimensions and type
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        if self.is_discrete:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = int(np.prod(self.env.action_space.shape))

        # Normalizer
        self.normalizer = normalizer or RunningNormalizer(
            shape=(self.obs_dim,)
        )

    def reset(self) -> np.ndarray:
        """Reset environment and return normalized observation."""
        obs, _ = self.env.reset(seed=self.seed)
        obs = obs.flatten().astype(np.float32)
        self.last_raw_obs = obs.copy()
        self.normalizer.update(obs)
        return self.normalizer.normalize(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info).

        Action should be one-hot for discrete environments.
        """
        if self.is_discrete:
            # Convert one-hot to int
            env_action = int(np.argmax(action))
        else:
            env_action = action

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = obs.flatten().astype(np.float32)
        self.last_raw_obs = obs.copy()
        self.normalizer.update(obs)
        return self.normalizer.normalize(obs), float(reward), terminated, truncated, info

    def action_to_onehot(self, action_idx: int) -> np.ndarray:
        """Convert integer action to one-hot vector."""
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[action_idx] = 1.0
        return onehot

    def sample_random_action(self) -> np.ndarray:
        """Sample a random action (returned as one-hot for discrete)."""
        if self.is_discrete:
            return self.action_to_onehot(self.env.action_space.sample())
        else:
            return self.env.action_space.sample().astype(np.float32)

    def close(self):
        self.env.close()
