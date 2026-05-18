"""Tests for the intrinsic curiosity module."""

import numpy as np
import torch
import pytest

from ragnarok.learning.curiosity import CuriosityModule, DeviceLatentCuriosity
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)
from ragnarok.learning.rollout import collect_rollout
from ragnarok.core.rssm import RSSM
from ragnarok.infrastructure.device import DEVICE


class TestCuriosityModule:
    """Test forward prediction curiosity."""

    def test_intrinsic_rewards_shape(self):
        cm = CuriosityModule(obs_dim=4, action_dim=2)
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        next_obs = obs + 0.1 * np.random.randn(10, 4).astype(np.float32)
        rewards = cm.compute_intrinsic_rewards(obs, act, next_obs)
        assert rewards.shape == (10,)
        assert rewards.dtype == np.float64 or rewards.dtype == np.float32

    def test_intrinsic_rewards_nonnegative(self):
        cm = CuriosityModule(obs_dim=4, action_dim=2, beta=0.5)
        obs = np.random.randn(20, 4).astype(np.float32)
        act = np.random.randn(20, 2).astype(np.float32)
        next_obs = obs + 0.1 * np.random.randn(20, 4).astype(np.float32)
        rewards = cm.compute_intrinsic_rewards(obs, act, next_obs)
        assert np.all(rewards >= 0), "Intrinsic rewards should be non-negative"

    def test_predictor_loss_decreases(self):
        cm = CuriosityModule(obs_dim=4, action_dim=2, lr=1e-3)
        obs = np.random.randn(50, 4).astype(np.float32)
        act = np.random.randn(50, 2).astype(np.float32)
        next_obs = obs + 0.1 * np.random.randn(50, 4).astype(np.float32)

        loss1 = cm.train_on_transitions(obs, act, next_obs)
        for _ in range(20):
            cm.train_on_transitions(obs, act, next_obs)
        loss2 = cm.train_on_transitions(obs, act, next_obs)
        assert loss2 < loss1, f"Loss should decrease: {loss1:.4f} -> {loss2:.4f}"

    def test_novel_states_higher_reward(self):
        cm = CuriosityModule(obs_dim=2, action_dim=1, beta=1.0, lr=1e-2)

        # Train on familiar region
        familiar_obs = np.zeros((50, 2), dtype=np.float32)
        familiar_act = np.zeros((50, 1), dtype=np.float32)
        familiar_next = np.ones((50, 2), dtype=np.float32) * 0.1
        for _ in range(50):
            cm.train_on_transitions(familiar_obs, familiar_act, familiar_next)

        # Novel region should have higher intrinsic reward
        novel_obs = np.ones((10, 2), dtype=np.float32) * 5.0
        novel_act = np.ones((10, 1), dtype=np.float32)
        novel_next = np.ones((10, 2), dtype=np.float32) * 5.5

        familiar_rewards = cm.compute_intrinsic_rewards(
            familiar_obs[:10], familiar_act[:10], familiar_next[:10]
        )
        novel_rewards = cm.compute_intrinsic_rewards(novel_obs, novel_act, novel_next)
        assert novel_rewards.mean() > familiar_rewards.mean(), \
            "Novel states should have higher intrinsic reward"

    def test_state_dict_roundtrip(self):
        cm1 = CuriosityModule(obs_dim=4, action_dim=2)
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        next_obs = obs + np.random.randn(10, 4).astype(np.float32) * 0.1
        cm1.train_on_transitions(obs, act, next_obs)
        r1 = cm1.compute_intrinsic_rewards(obs, act, next_obs)

        sd = cm1.state_dict()
        cm2 = CuriosityModule(obs_dim=4, action_dim=2)
        cm2.load_state_dict(sd)
        r2 = cm2.compute_intrinsic_rewards(obs, act, next_obs)

        np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_beta_scaling(self):
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        next_obs = obs + np.random.randn(10, 4).astype(np.float32)

        cm_low = CuriosityModule(obs_dim=4, action_dim=2, beta=0.1)
        cm_high = CuriosityModule(obs_dim=4, action_dim=2, beta=1.0)
        # Same predictor weights
        cm_high.predictor.load_state_dict(cm_low.predictor.state_dict())
        cm_high._reward_mean = cm_low._reward_mean
        cm_high._reward_var = cm_low._reward_var
        cm_high._reward_count = cm_low._reward_count

        r_low = cm_low.compute_intrinsic_rewards(obs, act, next_obs)
        r_high = cm_high.compute_intrinsic_rewards(obs, act, next_obs)

        # High beta should give higher rewards (approximately 10x)
        # Not exact due to running stats update, but should be clearly larger
        assert r_high.sum() > r_low.sum(), "Higher beta should give higher rewards"


class TestDeviceLatentCuriosity:
    """Device-resident RSSM Bayesian-surprise curiosity from a RolloutBatch."""

    @staticmethod
    def _discrete_rollout(n=8, t=16):
        def pf(obs):
            b = obs.shape[0]
            dist = torch.distributions.Categorical(
                logits=torch.zeros(b, 2, device=obs.device))
            a = dist.sample()
            return a, dist.log_prob(a), torch.zeros(b, device=obs.device)
        return collect_rollout(DeviceVecCartPole(n), pf, t)

    @staticmethod
    def _continuous_rollout(n=8, t=16):
        def pf(obs):
            b = obs.shape[0]
            a = torch.rand(b, 1, device=obs.device) * 2 - 1
            return (a, torch.zeros(b, device=obs.device),
                    torch.zeros(b, device=obs.device))
        return collect_rollout(DeviceVecMountainCarContinuous(n), pf, t)

    @staticmethod
    def _rssm(obs_dim, action_dim):
        return RSSM(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32,
                    stoch_dim=8, encoder_hidden=32).to(DEVICE)

    def test_discrete_rollout_reward_shape_and_range(self):
        """Discrete (one-hot encoded) rollout: reward is (N,T), ReLU'd, capped
        at beta*clip."""
        cur = DeviceLatentCuriosity(self._rssm(4, 2), beta=0.1, clip=5.0)
        r = cur.intrinsic_reward(self._discrete_rollout(n=8, t=16))
        assert r.shape == (8, 16)
        assert (r >= 0).all()                      # ReLU
        assert (r <= 0.1 * 5.0 + 1e-5).all()       # beta * clip ceiling

    def test_continuous_rollout(self):
        cur = DeviceLatentCuriosity(self._rssm(2, 1), beta=0.1)
        r = cur.intrinsic_reward(self._continuous_rollout(n=8, t=16))
        assert r.shape == (8, 16)
        assert (r >= 0).all()
        assert r.device.type == DEVICE.type

    def test_stats_accumulate_across_rollouts(self):
        """Each intrinsic_reward call folds N*T KL values into the stats."""
        cur = DeviceLatentCuriosity(self._rssm(4, 2))
        assert float(cur._count) == 0.0
        cur.intrinsic_reward(self._discrete_rollout(n=8, t=16))
        assert float(cur._count) == 8 * 16
        cur.intrinsic_reward(self._discrete_rollout(n=8, t=16))
        assert float(cur._count) == 2 * 8 * 16

    def test_warmup_gates_curiosity(self):
        """During the warmup window the curiosity is gated to zero — the
        RSSM's KL is a meaningless signal until the world model is trained.
        Stats are still folded so the normalizer is warm when the gate opens.
        """
        cur = DeviceLatentCuriosity(self._rssm(4, 2), beta=0.1, warmup=2)
        for call in range(1, 4):
            r = cur.intrinsic_reward(self._discrete_rollout(n=8, t=16))
            if call <= 2:
                assert float(r.abs().max()) == 0.0, f"call {call} not gated"
        assert cur._calls == 3
        assert float(cur._count) == 3 * 8 * 16   # stats folded even while gated
