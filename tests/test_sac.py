"""Tests for SAC (Soft Actor-Critic)."""

import torch
import numpy as np
import pytest

from ragnarok.learning.sac import (
    QNetwork, SACPolicy, SACReplayBuffer, SACTrainer, DeviceSACBuffer)
from ragnarok.environments.device_env import DeviceVecMountainCarContinuous
from ragnarok.learning.rollout import collect_rollout
from ragnarok.infrastructure.device import DEVICE


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


class TestDeviceSACBuffer:
    """Device-resident SAC replay buffer fed from a RolloutBatch."""

    @staticmethod
    def _rollout(n=8, t=16):
        def policy_fn(obs):
            b = obs.shape[0]
            action = torch.rand(b, 1, device=obs.device) * 2 - 1
            return (action, torch.zeros(b, device=obs.device),
                    torch.zeros(b, device=obs.device))
        return collect_rollout(DeviceVecMountainCarContinuous(n), policy_fn, t)

    def test_add_rollout_len(self):
        buf = DeviceSACBuffer(capacity=10_000)
        buf.add_rollout(self._rollout(n=8, t=16))
        assert len(buf) == 8 * 16

    def test_sample_shapes(self):
        buf = DeviceSACBuffer(capacity=10_000)
        buf.add_rollout(self._rollout(n=8, t=16))
        obs, act, rew, next_obs, done = buf.sample(32)
        assert obs.shape == (32, 2)        # MCC obs_dim
        assert act.shape == (32, 1)        # MCC action_dim
        assert rew.shape == (32,)
        assert next_obs.shape == (32, 2)
        assert done.shape == (32,)
        assert obs.device.type == DEVICE.type   # on the accelerator

    def test_capacity_caps_len(self):
        """Adding past capacity wraps the ring — len saturates at capacity."""
        cap = 8 * 16 * 2          # exactly two rollouts
        buf = DeviceSACBuffer(capacity=cap)
        for _ in range(5):
            buf.add_rollout(self._rollout(n=8, t=16))
        assert len(buf) == cap

    def test_next_obs_is_obs_shifted(self):
        """next_obs[t] == obs[t+1] within a row; last_obs closes the row."""
        batch = self._rollout(n=4, t=8)
        buf = DeviceSACBuffer(capacity=1000)
        buf.add_rollout(batch)
        # Flatten is row-major: buffer index k -> row k//T, step k%T.
        torch.testing.assert_close(buf._next_obs[0], batch.obs[0, 1])
        torch.testing.assert_close(buf._next_obs[7], batch.last_obs[0])

    def test_train_on_rollout_smoke(self):
        """End-to-end device-SAC step: RolloutBatch -> buffer -> SAC updates."""
        trainer = SACTrainer(
            obs_dim=2, action_dim=1, hidden=64,
            warmup_steps=0, batch_size=64,
            buffer=DeviceSACBuffer(capacity=10_000))
        metrics = trainer.train_on_rollout(self._rollout(n=16, t=16),
                                           n_updates=3)
        assert "sac/q1_loss" in metrics
        assert "sac/policy_loss" in metrics

    def test_train_on_rollout_warmup_returns_empty(self):
        """Before warmup is satisfied, train_on_rollout is a no-op collector."""
        trainer = SACTrainer(
            obs_dim=2, action_dim=1, hidden=64,
            warmup_steps=10_000, batch_size=64,
            buffer=DeviceSACBuffer(capacity=10_000))
        metrics = trainer.train_on_rollout(self._rollout(n=16, t=16),
                                           n_updates=3)
        assert metrics == {}
        assert len(trainer.replay) == 16 * 16   # data still buffered
