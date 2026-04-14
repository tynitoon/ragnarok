"""Tests for ObsEncoder integration and unified policy architecture."""

import numpy as np
import torch
import pytest

from ragnarok.core.obs_encoder import MLPObsEncoder, CNNObsEncoder, create_obs_encoder
from ragnarok.learning.real_experience import DirectPolicyNet, ContinuousPolicyNet
from ragnarok.learning.dream_augmenter import DreamAugmenter
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE


class TestObsEncoder:

    def test_mlp_cnn_same_output_dim(self):
        """MLP and CNN encoders produce the same embedding dimension."""
        embed_dim = 128
        mlp = MLPObsEncoder(obs_dim=4, embed_dim=embed_dim)
        cnn = CNNObsEncoder(channels=2, embed_dim=embed_dim)
        assert mlp.embed_dim == cnn.embed_dim == embed_dim

        # MLP: vector obs
        vec_obs = torch.randn(3, 4)
        vec_out = mlp(vec_obs)
        assert vec_out.shape == (3, embed_dim)

        # CNN: pixel obs (2 channels, 64x64)
        pix_obs = torch.randn(3, 2, 64, 64)
        pix_out = cnn(pix_obs)
        assert pix_out.shape == (3, embed_dim)

    def test_cnn_accepts_flattened(self):
        """CNN encoder handles flattened pixel observations."""
        cnn = CNNObsEncoder(channels=2, embed_dim=64)
        flat_obs = torch.randn(2, 2 * 64 * 64)
        out = cnn(flat_obs)
        assert out.shape == (2, 64)

    def test_factory_function(self):
        """create_obs_encoder returns correct type."""
        mlp = create_obs_encoder(obs_dim=8, embed_dim=64, pixel=False)
        cnn = create_obs_encoder(obs_dim=0, embed_dim=64, pixel=True, channels=3)
        assert isinstance(mlp, MLPObsEncoder)
        assert isinstance(cnn, CNNObsEncoder)


class TestPolicyWithEncoder:

    def test_discrete_policy_with_encoder(self):
        """DirectPolicyNet works with ObsEncoder."""
        encoder = MLPObsEncoder(obs_dim=4, embed_dim=32).to(DEVICE)
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32,
                                 obs_encoder=encoder).to(DEVICE)
        obs = torch.randn(5, 4, device=DEVICE)
        logits, value = policy(obs)
        assert logits.shape == (5, 2)
        assert value.shape == (5,)

    def test_discrete_policy_without_encoder(self):
        """DirectPolicyNet works without ObsEncoder (backward compat)."""
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32).to(DEVICE)
        obs = torch.randn(5, 4, device=DEVICE)
        logits, value = policy(obs)
        assert logits.shape == (5, 2)
        assert value.shape == (5,)

    def test_continuous_policy_with_encoder(self):
        """ContinuousPolicyNet works with ObsEncoder."""
        encoder = MLPObsEncoder(obs_dim=3, embed_dim=32).to(DEVICE)
        policy = ContinuousPolicyNet(obs_dim=3, action_dim=1, hidden=32,
                                     obs_encoder=encoder).to(DEVICE)
        obs = torch.randn(5, 3, device=DEVICE)
        mean, logstd, value = policy(obs)
        assert mean.shape == (5, 1)
        assert logstd.shape == (5, 1)
        assert value.shape == (5,)

    def test_encoder_gradients_flow(self):
        """Gradients flow through encoder -> policy during training."""
        encoder = MLPObsEncoder(obs_dim=4, embed_dim=32).to(DEVICE)
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32,
                                 obs_encoder=encoder).to(DEVICE)
        obs = torch.randn(3, 4, device=DEVICE)
        logits, value = policy(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        # Encoder parameters should have gradients
        for p in encoder.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


class TestSingleOptimizer:

    def test_dream_augmenter_shared_optimizer(self):
        """DreamAugmenter uses shared optimizer when provided."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=32,
                     stoch_dim=8, encoder_hidden=32).to(DEVICE)
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32).to(DEVICE)
        buffer = ReplayBuffer(capacity=1000)
        shared_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

        augmenter = DreamAugmenter(
            rssm=rssm, policy=policy, replay_buffer=buffer,
            optimizer=shared_opt, dream_grad_scale=0.3,
        )
        assert augmenter.optimizer is shared_opt
        assert augmenter.dream_grad_scale == 0.3

    def test_dream_augmenter_fallback_optimizer(self):
        """DreamAugmenter creates own optimizer when none provided."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=32,
                     stoch_dim=8, encoder_hidden=32).to(DEVICE)
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32).to(DEVICE)
        buffer = ReplayBuffer(capacity=1000)

        augmenter = DreamAugmenter(
            rssm=rssm, policy=policy, replay_buffer=buffer, lr=1e-4,
        )
        assert augmenter.dream_grad_scale == 1.0
        # Has its own optimizer (not None)
        assert augmenter.optimizer is not None

    def test_dream_no_regression_cartpole(self):
        """Dream training on CartPole-like data doesn't crash or NaN."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=32,
                     stoch_dim=8, encoder_hidden=32).to(DEVICE)
        policy = DirectPolicyNet(obs_dim=4, action_dim=2, hidden=32).to(DEVICE)
        buffer = ReplayBuffer(capacity=1000)
        shared_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

        augmenter = DreamAugmenter(
            rssm=rssm, policy=policy, replay_buffer=buffer,
            horizon=5, dream_batch=4,
            optimizer=shared_opt, dream_grad_scale=0.3,
        )

        # Fill buffer with synthetic episodes
        for _ in range(10):
            T = np.random.randint(10, 30)
            obs = np.random.randn(T, 4).astype(np.float32)
            acts = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, T)]
            rews = np.random.randn(T).astype(np.float32)
            dones = np.zeros(T, dtype=np.float32)
            dones[-1] = 1.0
            buffer.add_episode(obs, acts, rews, dones)

        # Train dream augmenter
        metrics = augmenter.dream_and_train()
        assert len(metrics) > 0
        for v in metrics.values():
            assert not np.isnan(v), f"NaN in dream metrics: {metrics}"
