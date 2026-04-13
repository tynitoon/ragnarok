"""Vectorized environment wrapper for parallel data collection.

Runs N copies of a RagnarokEnv synchronously, enabling batched GPU
inference during episode collection. Auto-resets envs on termination.
"""

import numpy as np
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.core.normalizer import RunningNormalizer


class VecRagnarokEnv:
    """Synchronous vectorized wrapper over N RagnarokEnv instances.

    All envs share a single normalizer so observation statistics
    remain consistent. Each env gets a distinct seed.
    """

    def __init__(self, env_name: str, num_envs: int, seed: int = 42,
                 pixel_obs: bool = False, normalize: bool = True,
                 normalizer: RunningNormalizer | None = None):
        self.num_envs = num_envs
        self.env_name = env_name

        # Shared normalizer across all envs
        self._shared_normalizer = normalizer

        # Create envs with distinct seeds
        self.envs: list[RagnarokEnv] = []
        for i in range(num_envs):
            env = RagnarokEnv(
                env_name, seed=seed + i, pixel_obs=pixel_obs,
                normalize=normalize, normalizer=self._shared_normalizer,
            )
            if self._shared_normalizer is None:
                # Use first env's normalizer as shared
                self._shared_normalizer = env.normalizer
            else:
                # Point all envs to shared normalizer
                env.normalizer = self._shared_normalizer
            self.envs.append(env)

        # Expose properties from first env
        self.is_discrete = self.envs[0].is_discrete
        self.action_dim = self.envs[0].action_dim
        self.obs_dim = self.envs[0].obs_dim
        self.action_low = self.envs[0].action_low
        self.action_high = self.envs[0].action_high
        self.normalizer = self._shared_normalizer
        self.normalize = normalize

    def reset(self) -> np.ndarray:
        """Reset all envs. Returns (num_envs, obs_dim)."""
        obs = np.stack([env.reset() for env in self.envs])
        return obs

    def reset_single(self, idx: int) -> np.ndarray:
        """Reset a single env by index."""
        return self.envs[idx].reset()

    def step(self, actions: np.ndarray):
        """Step all envs with a batch of actions.

        Args:
            actions: (num_envs, action_dim) array

        Returns:
            obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            terminated: (num_envs,) bool
            truncated: (num_envs,) bool
            infos: list of dicts
        """
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []

        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def action_to_onehot(self, action_idx: int) -> np.ndarray:
        return self.envs[0].action_to_onehot(action_idx)

    def sample_random_actions(self) -> np.ndarray:
        """Sample random actions for all envs."""
        return np.stack([env.sample_random_action() for env in self.envs])

    def close(self):
        for env in self.envs:
            env.close()
