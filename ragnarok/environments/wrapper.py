"""Unified environment wrapper for Gymnasium environments."""

import numpy as np
import gymnasium as gym
from ragnarok.core.normalizer import RunningNormalizer

# Image size for pixel observations (32x32 for fast training)
PIXEL_SIZE = 32


class RagnarokEnv:
    """Wraps a Gymnasium environment with observation normalization
    and action space abstraction.

    Supports both vector and pixel observation modes.
    """

    def __init__(self, env_name: str, normalizer: RunningNormalizer | None = None,
                 seed: int | None = None, pixel_obs: bool = False,
                 normalize: bool = True):
        self.pixel_obs = pixel_obs
        self.seed = seed
        self.normalize = normalize

        if pixel_obs:
            self.env = gym.make(env_name, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name)
        self.env_name = env_name

        # Action space
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.action_dim = self.env.action_space.n
            self.action_low = None
            self.action_high = None
        else:
            self.action_dim = int(np.prod(self.env.action_space.shape))
            self.action_low = self.env.action_space.low.flatten().astype(np.float32)
            self.action_high = self.env.action_space.high.flatten().astype(np.float32)

        # Observation dimensions
        if pixel_obs:
            self.n_channels = 1  # Grayscale frame
            self.obs_dim = self.n_channels * PIXEL_SIZE * PIXEL_SIZE
            self.vector_obs_dim = int(np.prod(self.env.observation_space.shape))
        else:
            self.obs_dim = int(np.prod(self.env.observation_space.shape))

        # Normalizer (for vector obs; pixel obs uses /255 scaling)
        self.normalizer = normalizer or RunningNormalizer(
            shape=(self.obs_dim,) if not pixel_obs else (self.obs_dim,)
        )

        # Fixed normalization based on observation space bounds
        # (for off-policy methods like SAC: deterministic, no drift)
        if not pixel_obs and not self.is_discrete:
            obs_low = self.env.observation_space.low.flatten().astype(np.float32)
            obs_high = self.env.observation_space.high.flatten().astype(np.float32)
            # Clip infinite bounds to reasonable range
            obs_low = np.clip(obs_low, -10.0, 10.0)
            obs_high = np.clip(obs_high, -10.0, 10.0)
            self._obs_center = (obs_high + obs_low) / 2.0
            self._obs_scale = np.maximum((obs_high - obs_low) / 2.0, 1e-6)
        else:
            self._obs_center = None
            self._obs_scale = None

    def _render_single_frame(self) -> np.ndarray:
        """Render current frame as 64x64 grayscale, CHW float32 / 255."""
        frame = self.env.render()  # (H, W, 3) uint8
        from PIL import Image
        img = Image.fromarray(frame).resize(
            (PIXEL_SIZE, PIXEL_SIZE), Image.BILINEAR
        ).convert('L')  # Grayscale
        pixels = np.array(img, dtype=np.float32) / 255.0  # (64, 64)
        return pixels.reshape(1, PIXEL_SIZE, PIXEL_SIZE)  # (1, 64, 64)

    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        obs, _ = self.env.reset(seed=self.seed)
        obs = obs.flatten().astype(np.float32)
        self.last_raw_obs = obs.copy()

        if self.pixel_obs:
            return self._render_single_frame().flatten()

        self.normalizer.update(obs)
        if self.normalize:
            return self.normalizer.normalize(obs)
        return self._fixed_normalize(obs)

    def _fixed_normalize(self, obs: np.ndarray) -> np.ndarray:
        """Fixed normalization from observation space bounds.

        Used for off-policy methods (SAC) to avoid distribution shift.
        Maps obs to approximately [-1, 1] using known bounds.
        """
        if self._obs_center is not None:
            return ((obs - self._obs_center) / self._obs_scale).astype(np.float32)
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info).

        Action should be one-hot for discrete environments.
        """
        if self.is_discrete:
            env_action = int(np.argmax(action))
        else:
            env_action = action

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = obs.flatten().astype(np.float32)
        self.last_raw_obs = obs.copy()

        if self.pixel_obs:
            return self._render_single_frame().flatten(), float(reward), terminated, truncated, info

        self.normalizer.update(obs)
        if self.normalize:
            return self.normalizer.normalize(obs), float(reward), terminated, truncated, info
        return self._fixed_normalize(obs), float(reward), terminated, truncated, info

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
