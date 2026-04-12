"""Tests for CNN encoder/decoder and pixel-based RSSM."""

import torch
import pytest
from ragnarok.core.cnn import CNNEncoder, CNNDecoder
from ragnarok.core.rssm import RSSM


class TestCNNEncoder:
    def test_output_shape(self):
        enc = CNNEncoder(channels=3, feature_dim=128, depth=32)
        # Flattened 64x64x3 input
        x = torch.randn(4, 3 * 64 * 64)
        out = enc(x)
        assert out.shape == (4, 128)

    def test_4d_input(self):
        enc = CNNEncoder(channels=3, feature_dim=128, depth=32)
        # Direct NCHW input
        x = torch.randn(4, 3, 64, 64)
        out = enc(x)
        assert out.shape == (4, 128)

    def test_gradients_flow(self):
        enc = CNNEncoder(channels=3, feature_dim=128, depth=32)
        x = torch.randn(2, 3 * 64 * 64, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestCNNDecoder:
    def test_output_shape(self):
        dec = CNNDecoder(latent_dim=160, channels=3, depth=32)
        z = torch.randn(4, 160)
        out = dec(z)
        assert out.shape == (4, 3 * 64 * 64)

    def test_gradients_flow(self):
        dec = CNNDecoder(latent_dim=160, channels=3, depth=32)
        z = torch.randn(2, 160, requires_grad=True)
        out = dec(z)
        out.sum().backward()
        assert z.grad is not None


class TestPixelRSSM:
    @pytest.fixture
    def pixel_rssm(self):
        enc = CNNEncoder(channels=3, feature_dim=128, depth=16)
        dec = CNNDecoder(latent_dim=32 + 8, channels=3, depth=16)
        return RSSM(
            obs_dim=3 * 64 * 64, action_dim=2,
            hidden_dim=32, stoch_dim=8, encoder_hidden=128,
            encoder=enc, decoder=dec,
        )

    def test_observe_shapes(self, pixel_rssm):
        batch, time = 2, 5
        obs = torch.randn(batch, time, 3 * 64 * 64)
        actions = torch.randn(batch, time, 2)
        outputs = pixel_rssm.observe(obs, actions)
        assert outputs["h"].shape == (batch, time, 32)
        assert outputs["z"].shape == (batch, time, 8)
        assert outputs["recon_obs"].shape == (batch, time, 3 * 64 * 64)

    def test_loss_computes(self, pixel_rssm):
        batch, time = 2, 3
        obs = torch.randn(batch, time, 3 * 64 * 64)
        actions = torch.randn(batch, time, 2)
        rewards = torch.ones(batch, time)
        dones = torch.zeros(batch, time)
        losses = pixel_rssm.loss(obs, actions, rewards, dones)
        assert losses["total_loss"].shape == ()
        assert losses["total_loss"].requires_grad

    def test_encode_single_observation(self, pixel_rssm):
        h, z = pixel_rssm.initial_state(1, torch.device("cpu"))
        obs = torch.randn(1, 3 * 64 * 64)
        action = torch.randn(1, 2)
        h_new, z_new = pixel_rssm.encode_observation(obs, h, z, action)
        assert h_new.shape == (1, 32)
        assert z_new.shape == (1, 8)
