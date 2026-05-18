"""Tests for the device-path agent orchestrator (Phase 2 Stage 5.4)."""

import torch

from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)


class TestDeviceAgent:

    def test_cartpole_train_iteration(self):
        """A discrete (PPO) DeviceAgent trains policy + WM + latent each iter."""
        agent = DeviceAgent(DeviceVecCartPole, num_envs=16, horizon=16)
        m = agent.train_iteration()
        assert any(k.startswith("real/") for k in m)
        assert any(k.startswith("wm/") for k in m)
        assert any(k.startswith("latent/") for k in m)

    def test_mcc_train_iteration(self):
        """A continuous (SAC) DeviceAgent runs — curiosity folds into the SAC
        reward; WM and latent policy train every iteration."""
        agent = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=16,
                            horizon=16, sac_updates=4)
        assert agent.curiosity is not None
        m = agent.train_iteration()
        assert any(k.startswith("wm/") for k in m)
        assert any(k.startswith("latent/") for k in m)

    def test_evaluate_returns_float(self):
        agent = DeviceAgent(DeviceVecCartPole, num_envs=16, horizon=16)
        score = agent.evaluate(steps=64, n_envs=16)
        assert isinstance(score, float)

    def test_snapshot_load_cross_dim(self):
        """Snapshot a CartPole agent (obs 4 / act 2), load into an MCC agent
        (obs 2 / act 1). The transferable subset is env-agnostic, so the
        cross-dim load must succeed and actually change the target weights."""
        src = DeviceAgent(DeviceVecCartPole, num_envs=16, horizon=16)
        snap = src.snapshot()
        assert snap["rssm_core"] and snap["latent_trunk"]

        dst = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=16,
                          horizon=16, sac_updates=4)
        dst.load_snapshot(snap)   # must not raise — env-agnostic subset
        for k, v in snap["rssm_core"].items():
            assert torch.equal(dst.rssm.state_dict()[k].detach().cpu(), v)
        for k, v in snap["latent_trunk"].items():
            assert torch.equal(
                dst.latent.policy.state_dict()[k].detach().cpu(), v)
        assert dst.acting_mode == "latent"   # transfer flips the acting mode

    def test_latent_acting_mode_train_iteration(self):
        """After load_snapshot the agent acts via the latent policy:
        train_iteration collects latent-mode, trains latent + WM, and does
        NOT train the real (SAC) policy — it is no longer the actor."""
        src = DeviceAgent(DeviceVecCartPole, num_envs=16, horizon=16)
        dst = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=16,
                          horizon=16, sac_updates=4)
        dst.load_snapshot(src.snapshot())
        m = dst.train_iteration()
        assert any(k.startswith("wm/") for k in m)
        assert any(k.startswith("latent/") for k in m)
        assert not any(k.startswith("sac/") for k in m)
        assert dst.total_env_steps == 16 * 16

    def test_latent_mode_evaluate(self):
        """evaluate() in latent mode runs the RSSM-state-threaded eval."""
        src = DeviceAgent(DeviceVecCartPole, num_envs=16, horizon=16)
        dst = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=16,
                          horizon=16, sac_updates=4)
        dst.load_snapshot(src.snapshot())
        assert isinstance(dst.evaluate(steps=64, n_envs=16), float)
