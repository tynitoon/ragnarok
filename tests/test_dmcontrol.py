"""Tests for DMControl Suite integration.

Tests are split into:
- Unit tests (always run) — registry, factory, dimension constants
- Integration tests (skip if dm_control not installed) — actual env interaction
"""

import numpy as np
import pytest

from ragnarok.environments.dmcontrol import DMC_AVAILABLE, DMC_TASKS
from ragnarok.environments.registry import get_env_spec, make_env, REGISTRY


class TestDMCRegistry:
    """Test that DMC environments are registered correctly."""

    def test_dmc_envs_in_registry(self):
        """All 5 DMC tasks should be in the registry."""
        for name in DMC_TASKS:
            assert name in REGISTRY, f"{name} not in REGISTRY"

    def test_dmc_specs_continuous(self):
        """All DMC envs should be continuous (not discrete)."""
        for name in DMC_TASKS:
            spec = get_env_spec(name)
            assert not spec.is_discrete
            assert not spec.pixel_obs

    def test_dmc_gym_name_prefix(self):
        """DMC gym_names should start with 'dmc:'."""
        for name in DMC_TASKS:
            spec = get_env_spec(name)
            assert spec.gym_name.startswith("dmc:")

    def test_dmc_obs_dims_positive(self):
        """Observation dims should be positive integers."""
        for name in DMC_TASKS:
            spec = get_env_spec(name)
            assert spec.obs_dim > 0

    def test_dmc_action_dims_positive(self):
        """Action dims should be positive integers."""
        for name in DMC_TASKS:
            spec = get_env_spec(name)
            assert spec.action_dim > 0

    def test_known_dims(self):
        """Verify known obs/action dims for key envs."""
        # These are the hardcoded fallback dims
        assert get_env_spec("walker-walk").obs_dim == 24
        assert get_env_spec("walker-walk").action_dim == 6
        assert get_env_spec("cheetah-run").obs_dim == 17
        assert get_env_spec("cheetah-run").action_dim == 6
        assert get_env_spec("cartpole-swingup").obs_dim == 5
        assert get_env_spec("cartpole-swingup").action_dim == 1

    def test_make_env_gym_still_works(self):
        """make_env should still create Gymnasium envs correctly."""
        env = make_env("cartpole", seed=42)
        assert env.is_discrete
        assert env.obs_dim == 4
        obs = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_make_env_continuous_gym(self):
        """make_env for Pendulum should work."""
        env = make_env("pendulum", seed=42)
        assert not env.is_discrete
        assert env.obs_dim == 3
        obs = env.reset()
        assert obs.shape == (3,)
        env.close()


@pytest.mark.skipif(not DMC_AVAILABLE, reason="dm_control not installed")
class TestDMCEnvInteraction:
    """Integration tests — require dm_control."""

    @pytest.mark.parametrize("name", list(DMC_TASKS.keys()))
    def test_reset_returns_correct_shape(self, name):
        """Reset should return flat obs of correct dimension."""
        env = make_env(name, seed=42)
        spec = get_env_spec(name)
        obs = env.reset()
        assert obs.shape == (spec.obs_dim,)
        assert obs.dtype == np.float32
        env.close()

    @pytest.mark.parametrize("name", list(DMC_TASKS.keys()))
    def test_step_returns_correct_shape(self, name):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = make_env(name, seed=42)
        spec = get_env_spec(name)
        env.reset()
        action = env.sample_random_action()
        assert action.shape == (spec.action_dim,)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (spec.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_walker_walk_episode(self):
        """Run a short episode on walker-walk."""
        env = make_env("walker-walk", seed=42)
        obs = env.reset()
        total_reward = 0.0
        for _ in range(50):
            action = env.sample_random_action()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()
        # Random policy should get some reward (DMC rewards are 0-1)
        assert total_reward >= 0.0

    def test_make_env_dmc(self):
        """make_env should create DMControlEnv for DMC tasks."""
        from ragnarok.environments.dmcontrol import DMControlEnv
        env = make_env("cheetah-run", seed=42)
        assert isinstance(env, DMControlEnv)
        assert env.env_name == "dmc:cheetah-run"
        env.close()


@pytest.mark.skipif(not DMC_AVAILABLE, reason="dm_control not installed")
class TestDMCAgentIntegration:
    """Test that RagnarokAgent works with DMC environments."""

    def test_agent_creates_sac_for_dmc(self):
        """DMC envs are continuous — agent should use SAC."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole-swingup")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = make_env("cartpole-swingup", seed=42)
        agent = RagnarokAgent(config, env)

        assert agent.sac_trainer is not None
        env.close()

    def test_agent_trains_one_episode_dmc(self):
        """Agent should be able to train one episode on DMC."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole-swingup")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = make_env("cartpole-swingup", seed=42)
        agent = RagnarokAgent(config, env)

        reward, metrics = agent.train_policy_real()
        assert isinstance(reward, float)
        assert agent.total_episodes >= 1
        env.close()
