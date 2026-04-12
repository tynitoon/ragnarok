"""Tests for the RSSM world model."""

import torch
import pytest
from ragnarok.core.rssm import RSSM, RSSMEncoder, RSSMCore


@pytest.fixture
def rssm():
    return RSSM(obs_dim=4, action_dim=2, hidden_dim=32, stoch_dim=8, encoder_hidden=32)


class TestRSSMShapes:
    def test_encoder_output_shape(self):
        enc = RSSMEncoder(obs_dim=4, hidden=32)
        obs = torch.randn(5, 4)
        out = enc(obs)
        assert out.shape == (5, 32)

    def test_initial_state(self, rssm):
        h, z = rssm.initial_state(3, torch.device("cpu"))
        assert h.shape == (3, 32)
        assert z.shape == (3, 8)

    def test_observe_output_shapes(self, rssm):
        batch, time = 4, 10
        obs = torch.randn(batch, time, 4)
        actions = torch.randn(batch, time, 2)
        outputs = rssm.observe(obs, actions)

        assert outputs["h"].shape == (batch, time, 32)
        assert outputs["z"].shape == (batch, time, 8)
        assert outputs["prior_mean"].shape == (batch, time, 8)
        assert outputs["recon_obs"].shape == (batch, time, 4)
        assert outputs["reward_pred"].shape == (batch, time)
        assert outputs["continue_pred"].shape == (batch, time)

    def test_imagine_output_shapes(self, rssm):
        batch, horizon = 4, 5
        h0, z0 = rssm.initial_state(batch, torch.device("cpu"))

        def dummy_policy(h, z):
            return torch.randn(h.shape[0], 2)

        outputs = rssm.imagine(h0, z0, dummy_policy, horizon)
        assert outputs["h"].shape == (batch, horizon + 1, 32)
        assert outputs["z"].shape == (batch, horizon + 1, 8)
        assert outputs["action"].shape == (batch, horizon, 2)
        assert outputs["reward_pred"].shape == (batch, horizon)

    def test_loss_produces_scalar(self, rssm):
        batch, time = 2, 5
        obs = torch.randn(batch, time, 4)
        actions = torch.randn(batch, time, 2)
        rewards = torch.ones(batch, time)
        dones = torch.zeros(batch, time)
        dones[:, -1] = 1.0

        losses = rssm.loss(obs, actions, rewards, dones)
        assert losses["total_loss"].shape == ()
        assert losses["total_loss"].requires_grad

    def test_encode_observation(self, rssm):
        h, z = rssm.initial_state(1, torch.device("cpu"))
        obs = torch.randn(1, 4)
        action = torch.randn(1, 2)
        h_new, z_new = rssm.encode_observation(obs, h, z, action)
        assert h_new.shape == (1, 32)
        assert z_new.shape == (1, 8)

    def test_state_dim(self, rssm):
        assert rssm.state_dim == 32 + 8
