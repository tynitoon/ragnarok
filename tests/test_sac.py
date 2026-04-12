"""Tests for SAC (Soft Actor-Critic)."""

import torch
import numpy as np
import pytest

from ragnarok.learning.sac import QNetwork, SACPolicy, SACReplayBuffer, SACTrainer


class TestQNetwork:
    def test_output_shape(self):
        q = QNetwork(obs_dim=3, action_dim=1, hidden=64)
        obs = torch.randn(8, 3)
        act = torch.randn(8, 1)
        out = q(obs, act)
        assert out.shape == (8,)

    def test_gradients_flow(self):
        q = QNetwork(obs_dim=3, action_dim=1, hidden=64)
        obs = torch.randn(4, 3)
        act = torch.randn(4, 1, requires_grad=True)
        out = q(obs, act)
        out.sum().backward()
        assert act.grad is not None


class TestSACPolicy:
    def test_sample_shapes(self):
        policy = SACPolicy(obs_dim=3, action_dim=1, hidden=64)
        obs = torch.randn(8, 3)
        action, log_prob = policy.sample(obs)
        assert action.shape == (8, 1)
        assert log_prob.shape == (8,)

    def test_act_returns_numpy(self):
        policy = SACPolicy(obs_dim=3, action_dim=1, hidden=64)
        obs = torch.randn(1, 3)
        action = policy.act(obs, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_action_within_bounds(self):
        low = np.array([-2.0])
        high = np.array([2.0])
        policy = SACPolicy(obs_dim=3, action_dim=1, hidden=64,
                           action_low=low, action_high=high)
        obs = torch.randn(100, 3)
        for _ in range(5):
            action, _ = policy.sample(obs)
            assert (action >= -2.0 - 1e-5).all()
            assert (action <= 2.0 + 1e-5).all()


class TestSACReplayBuffer:
    def test_add_and_sample(self):
        buf = SACReplayBuffer(capacity=100)
        for i in range(20):
            buf.add(
                np.zeros(3), np.zeros(1), float(i),
                np.ones(3), float(i == 19)
            )
        assert len(buf) == 20
        obs, act, rew, next_obs, done = buf.sample(8)
        assert obs.shape == (8, 3)
        assert act.shape == (8, 1)
        assert rew.shape == (8,)


class TestSACTrainer:
    def test_construction(self):
        trainer = SACTrainer(
            obs_dim=3, action_dim=1,
            action_low=np.array([-2.0]),
            action_high=np.array([2.0]),
            hidden=64, warmup_steps=10, batch_size=8,
        )
        assert trainer.policy is not None
        assert trainer.q1 is not None
        assert trainer.q2 is not None

    def test_alpha_auto_tune(self):
        trainer = SACTrainer(
            obs_dim=3, action_dim=1, hidden=64,
            warmup_steps=5, batch_size=4,
        )
        # Fill replay with random data
        for _ in range(20):
            trainer.replay.add(
                np.random.randn(3).astype(np.float32),
                np.random.randn(1).astype(np.float32),
                np.random.randn(),
                np.random.randn(3).astype(np.float32),
                0.0,
            )
        metrics = trainer._update()
        assert "sac/alpha" in metrics
        assert metrics["sac/alpha"] > 0
