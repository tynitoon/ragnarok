"""Tests for the replay buffer."""

import numpy as np
import pytest
from ragnarok.memory.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_episode(self):
        buf = ReplayBuffer(capacity=1000)
        obs = np.random.randn(10, 4).astype(np.float32)
        acts = np.random.randn(10, 2).astype(np.float32)
        rews = np.ones(10, dtype=np.float32)
        dones = np.zeros(10, dtype=np.float32)
        dones[-1] = 1.0

        buf.add_episode(obs, acts, rews, dones)
        assert buf.num_episodes == 1
        assert len(buf) == 10

    def test_sample_sequences(self):
        buf = ReplayBuffer(capacity=1000)
        for _ in range(5):
            T = np.random.randint(10, 30)
            obs = np.random.randn(T, 4).astype(np.float32)
            acts = np.random.randn(T, 2).astype(np.float32)
            rews = np.ones(T, dtype=np.float32)
            dones = np.zeros(T, dtype=np.float32)
            dones[-1] = 1.0
            buf.add_episode(obs, acts, rews, dones)

        obs, acts, rews, dones = buf.sample_sequences(3, 8)
        assert obs.shape == (3, 8, 4)
        assert acts.shape == (3, 8, 2)
        assert rews.shape == (3, 8)

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=50)
        for _ in range(10):
            obs = np.random.randn(20, 4).astype(np.float32)
            acts = np.random.randn(20, 2).astype(np.float32)
            rews = np.ones(20, dtype=np.float32)
            dones = np.zeros(20, dtype=np.float32)
            buf.add_episode(obs, acts, rews, dones)

        assert len(buf) <= 50

    def test_short_episodes_padded_to_fixed_length(self):
        """sample_sequences always returns exactly seq_length steps.

        Episodes shorter than seq_length are zero-padded (dones padded with
        1.0). A FIXED output shape is required for PyTorch/XLA — a varying
        length recompiles the XLA graph on every training step. The padded
        steps are masked out of the RSSM loss (see RSSM.loss), so the world
        model never trains on padding.
        """
        buf = ReplayBuffer(capacity=1000)
        # Add short episodes (5 steps each)
        for _ in range(5):
            obs = np.random.randn(5, 4).astype(np.float32)
            acts = np.random.randn(5, 2).astype(np.float32)
            rews = np.ones(5, dtype=np.float32)
            dones = np.zeros(5, dtype=np.float32)
            buf.add_episode(obs, acts, rews, dones)

        # Request seq_length=50; episodes are only 5 steps -> padded to 50,
        # NOT capped to 5 (the pre-XLA-fix behavior).
        obs, acts, rews, dones = buf.sample_sequences(2, 50)
        assert obs.shape == (2, 50, 4)
        assert acts.shape == (2, 50, 2)
        assert rews.shape == (2, 50)
        assert dones.shape == (2, 50)
        # Padded steps (indices 5:) are zero for obs, and dones==1.0 so the
        # RSSM loss mask (cumsum-of-done) excludes them.
        assert np.all(obs[:, 5:] == 0.0)
        assert np.all(dones[:, 5:] == 1.0)
