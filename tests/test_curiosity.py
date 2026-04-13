"""Tests for the intrinsic curiosity module."""

import numpy as np
import torch
import pytest

from ragnarok.learning.curiosity import CuriosityModule


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
