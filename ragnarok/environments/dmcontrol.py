"""DeepMind Control Suite integration.

Wraps DMC environments into the RagnarokEnv-compatible interface.
DMC tasks return dict observations — this wrapper flattens them into
a single float32 vector, matching the Gymnasium-based wrapper's API.

Requires: pip install dm_control shimmy[dm_control]
Falls back gracefully if dm_control is not installed.
"""

import numpy as np
from ragnarok.infrastructure.device import mark_step

DMC_AVAILABLE = False
try:
    from dm_control import suite
    DMC_AVAILABLE = True
except ImportError:
    pass

from ragnarok.core.normalizer import RunningNormalizer


# ── Canonical DMC tasks ──────────────────────────────────────────────

DMC_TASKS = {
    "walker-walk": ("walker", "walk"),
    "cheetah-run": ("cheetah", "run"),
    "cartpole-swingup": ("cartpole", "swingup"),
    "hopper-stand": ("hopper", "stand"),
    "finger-spin": ("finger", "spin"),
}


def get_dmc_obs_dim(domain: str, task: str) -> int:
    """Get flattened observation dimension for a DMC task."""
    if not DMC_AVAILABLE:
        # Hardcoded dims for when dm_control isn't installed (for registry)
        dims = {
            ("walker", "walk"): 24,
            ("cheetah", "run"): 17,
            ("cartpole", "swingup"): 5,
            ("hopper", "stand"): 15,
            ("finger", "spin"): 9,
        }
        return dims.get((domain, task), 0)

    env = suite.load(domain, task)
    spec = env.observation_spec()
    total = sum(np.prod(v.shape) for v in spec.values())
    env.close()
    return int(total)


def get_dmc_action_dim(domain: str, task: str) -> int:
    """Get action dimension for a DMC task."""
    if not DMC_AVAILABLE:
        dims = {
            ("walker", "walk"): 6,
            ("cheetah", "run"): 6,
            ("cartpole", "swingup"): 1,
            ("hopper", "stand"): 4,
            ("finger", "spin"): 2,
        }
        return dims.get((domain, task), 0)

    env = suite.load(domain, task)
    dim = int(np.prod(env.action_spec().shape))
    env.close()
    return dim


class DMControlEnv:
    """Wraps a DeepMind Control Suite environment.

    Provides the same interface as RagnarokEnv:
      - reset() -> obs (flat float32 vector)
      - step(action) -> (obs, reward, terminated, truncated, info)
      - Observation normalization via RunningNormalizer
      - Action scaling to [-1, 1] range
    """

    def __init__(self, domain: str, task: str,
                 normalizer: RunningNormalizer | None = None,
                 seed: int | None = None, normalize: bool = True,
                 max_episode_steps: int = 1000):
        if not DMC_AVAILABLE:
            raise ImportError(
                "dm_control is not installed. "
                "Install with: pip install dm_control shimmy[dm_control]"
            )

        self.domain = domain
        self.task = task
        self.seed = seed
        self.normalize = normalize
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Create DMC environment
        task_kwargs = {"random": seed} if seed is not None else {}
        self.env = suite.load(domain, task, task_kwargs=task_kwargs)

        self.env_name = f"dmc:{domain}-{task}"
        self.is_discrete = False
        self.pixel_obs = False

        # Action space (always continuous, bounded [-1, 1] in DMC)
        action_spec = self.env.action_spec()
        self.action_dim = int(np.prod(action_spec.shape))
        self.action_low = action_spec.minimum.flatten().astype(np.float32)
        self.action_high = action_spec.maximum.flatten().astype(np.float32)

        # Observation space (flattened dict)
        obs_spec = self.env.observation_spec()
        self.obs_dim = sum(int(np.prod(v.shape)) for v in obs_spec.values())

        # Normalizer
        self.normalizer = normalizer or RunningNormalizer(shape=(self.obs_dim,))

        # Fixed normalization bounds for SAC (continuous envs)
        # DMC obs are generally well-scaled but we still center them
        self._obs_center = np.zeros(self.obs_dim, dtype=np.float32)
        self._obs_scale = np.ones(self.obs_dim, dtype=np.float32) * 5.0

    def _flatten_obs(self, time_step) -> np.ndarray:
        """Flatten DMC's dict observation into a single vector."""
        parts = []
        for key in sorted(time_step.observation.keys()):
            val = np.asarray(time_step.observation[key], dtype=np.float32)
            parts.append(val.flatten())
        return np.concatenate(parts)

    def _fixed_normalize(self, obs: np.ndarray) -> np.ndarray:
        """Fixed normalization for off-policy methods."""
        return ((obs - self._obs_center) / self._obs_scale).astype(np.float32)

    def reset(self) -> np.ndarray:
        """Reset environment and return flat observation."""
        self._step_count = 0
        time_step = self.env.reset()
        obs = self._flatten_obs(time_step)
        self.last_raw_obs = obs.copy()

        self.normalizer.update(obs)
        if self.normalize:
            return self.normalizer.normalize(obs)
        return self._fixed_normalize(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info).

        Action is expected as a flat numpy array. DMC actions are bounded
        by action_spec, so we clip to be safe.
        """
        # XLA: close the lazy graph from the policy forward pass (see
        # RagnarokEnv.step for the full rationale). No-op on CUDA/CPU.
        mark_step()

        action = np.clip(action.flatten(), self.action_low, self.action_high)
        time_step = self.env.step(action)
        self._step_count += 1

        obs = self._flatten_obs(time_step)
        self.last_raw_obs = obs.copy()
        reward = float(time_step.reward or 0.0)

        # DMC uses time_step.last() for episode end
        terminated = time_step.last()
        truncated = self._step_count >= self.max_episode_steps

        self.normalizer.update(obs)
        if self.normalize:
            obs = self.normalizer.normalize(obs)
        else:
            obs = self._fixed_normalize(obs)

        return obs, reward, terminated, truncated, {}

    def action_to_onehot(self, action_idx: int) -> np.ndarray:
        """Not applicable for continuous — returns zeros."""
        return np.zeros(self.action_dim, dtype=np.float32)

    def sample_random_action(self) -> np.ndarray:
        """Sample a random action from action space."""
        return np.random.uniform(
            self.action_low, self.action_high
        ).astype(np.float32)

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
