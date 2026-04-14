"""Benchmark-hygiene pinning tests (preregistration §6.1 fix #3).

The env-name branching paths (reward shaper, curiosity beta overrides,
training-hparam overrides) contaminate cross-method comparisons. The
preregistration commits to reporting H1 numbers on *untuned* defaults,
so those paths must be gated behind opt-in config flags that default off.

A silent regression here — e.g. someone re-enabling the default shaping
path — would make every primary-endpoint RMST comparison compromised.
"""

import pytest

from ragnarok.infrastructure.config import (
    RagnarokConfig,
    RewardShapingConfig,
    EnvOverridesConfig,
)
from ragnarok.environments.registry import get_env_spec
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.core.agent import RagnarokAgent


def _make_agent(env_name: str = "cartpole") -> tuple[RagnarokAgent, RagnarokEnv]:
    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=0)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim
    config.curiosity.enabled = False
    env = RagnarokEnv(spec.gym_name, seed=0)
    return RagnarokAgent(config, env), env


class TestDefaultsAreClean:
    """Default config must not apply any env-name branching."""

    def test_reward_shaping_disabled_by_default(self):
        cfg = RagnarokConfig()
        assert cfg.reward_shaping.enabled is False

    def test_env_overrides_disabled_by_default(self):
        cfg = RagnarokConfig()
        assert cfg.env_overrides.enabled is False

    @pytest.mark.parametrize("env_name", ["MountainCar-v0", "Acrobot-v1",
                                          "CartPole-v1", "Pendulum-v1"])
    def test_reward_shaper_returns_none_by_default(self, env_name):
        agent, env = _make_agent()
        try:
            assert agent._get_reward_shaper(env_name) is None
        finally:
            env.close()

    @pytest.mark.parametrize("env_name,beta_arg", [
        ("MountainCar-v0", 0.07),
        ("Acrobot-v1", 0.11),
        ("CartPole-v1", 0.13),
        ("Pendulum-v1", 0.17),
        ("SomeOtherEnv", 0.19),
    ])
    def test_curiosity_beta_returns_arg_by_default(self, env_name, beta_arg):
        """With overrides off, _get_curiosity_beta returns the caller's
        default verbatim regardless of env_name."""
        agent, env = _make_agent()
        try:
            assert agent._get_curiosity_beta(env_name, beta_arg) == beta_arg
        finally:
            env.close()

    @pytest.mark.parametrize("env_name", ["MountainCar-v0", "Acrobot-v1",
                                          "CartPole-v1", "Pendulum-v1",
                                          "MountainCarContinuous-v0"])
    def test_training_hparams_generic_by_default(self, env_name):
        """With overrides off, _get_training_hparams returns the generic
        (0.01, 3e-4) defaults regardless of env_name."""
        agent, env = _make_agent()
        try:
            entropy, lr = agent._get_training_hparams(env_name)
            assert entropy == 0.01
            assert lr == 3e-4
        finally:
            env.close()


class TestOptInRestoresLegacy:
    """When flags are explicitly enabled, the legacy tuning kicks in.

    These pin the *existence* of the opt-in path so the tuned numbers
    reported in internal experiments remain reproducible when someone
    flips the flag.
    """

    def test_opt_in_reward_shaper_returns_callable(self):
        cfg = RagnarokConfig(seed=0)
        cfg.reward_shaping.enabled = True
        cfg.curiosity.enabled = False
        spec = get_env_spec("cartpole")
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            shaper = agent._get_reward_shaper("MountainCar-v0")
            assert callable(shaper)
            # Spot-check: shaping adds a positive bonus for non-goal states
            shaped = shaper([0.0, 0.0], 0.0, [0.0, 0.0])
            assert shaped > 0.0
        finally:
            env.close()

    def test_opt_in_training_hparams_uses_env_tuning(self):
        cfg = RagnarokConfig(seed=0)
        cfg.env_overrides.enabled = True
        cfg.curiosity.enabled = False
        spec = get_env_spec("cartpole")
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            # MountainCar tuning: (0.02, 1e-3), not the generic (0.01, 3e-4)
            entropy, lr = agent._get_training_hparams("MountainCar-v0")
            assert entropy == 0.02
            assert lr == 1e-3
        finally:
            env.close()

    def test_opt_in_curiosity_beta_uses_env_tuning(self):
        cfg = RagnarokConfig(seed=0)
        cfg.env_overrides.enabled = True
        cfg.curiosity.enabled = False
        spec = get_env_spec("cartpole")
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            # With overrides on and a *different* default from the tuning
            # table, the table should win for known envs:
            assert agent._get_curiosity_beta("MountainCar-v0", 0.999) == 0.3
            assert agent._get_curiosity_beta("CartPole-v1", 0.999) == 0.01
            # Unknown envs still fall through to the caller's default:
            assert agent._get_curiosity_beta("UnseenEnv", 0.999) == 0.999
        finally:
            env.close()


class TestAgentWiringHonorsFlags:
    """End-to-end: an agent built with default config must have no shaper
    wired into the real_trainer, regardless of which env it's targeting.
    """

    def test_default_agent_has_no_reward_shaper_wired(self):
        spec = get_env_spec("mountaincar")
        cfg = RagnarokConfig(seed=0)
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        cfg.curiosity.enabled = False
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            # The shaper stored on the trainer should be None on defaults,
            # even for MountainCar which previously triggered the hardcoded
            # height-bonus shaper.
            assert agent.real_trainer.reward_shaper is None
        finally:
            env.close()

    def test_default_sac_trainer_has_no_reward_shaper(self):
        """H1-primary endpoint (CartPole -> MountainCarContinuous) goes
        through SAC. If defaults secretly wired a shaper into sac_trainer,
        the primary claim would be a shaped-reward comparison. This test
        pins the contract for the continuous path.
        """
        spec = get_env_spec("mountaincar-continuous")
        cfg = RagnarokConfig(seed=0)
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        cfg.curiosity.enabled = False
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            assert agent.sac_trainer is not None, (
                "Continuous env should instantiate sac_trainer"
            )
            assert agent.sac_trainer.reward_shaper is None, (
                "sac_trainer.reward_shaper must default to None — shaping "
                "must not contaminate the H1 primary endpoint"
            )
        finally:
            env.close()

    def test_pendulum_sac_trainer_has_no_reward_shaper(self):
        """Second continuous target — same contract."""
        spec = get_env_spec("pendulum")
        cfg = RagnarokConfig(seed=0)
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        cfg.curiosity.enabled = False
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            assert agent.sac_trainer is not None
            assert agent.sac_trainer.reward_shaper is None
        finally:
            env.close()
