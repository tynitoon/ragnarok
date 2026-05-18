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


class TestObserveEpisodeReset:
    """observe(done_seq=...) zeroes the GRU state at episode boundaries so a
    multi-episode rollout row never leaks recurrent state across a seam."""

    def test_state_resets_after_done(self, rssm):
        """Two rollouts identical from the boundary onward but different
        before it must produce identical post-reset states."""
        batch, time, d = 3, 8, 3            # episode boundary at step d
        obs_a = torch.randn(batch, time, 4)
        act_a = torch.randn(batch, time, 2)
        obs_b = obs_a.clone()
        act_b = act_a.clone()
        obs_b[:, :d] = torch.randn(batch, d, 4)   # differ strictly before d
        act_b[:, :d] = torch.randn(batch, d, 2)
        dones = torch.zeros(batch, time)
        dones[:, d] = 1.0

        torch.manual_seed(0)
        out_a = rssm.observe(obs_a, act_a, done_seq=dones)
        torch.manual_seed(0)
        out_b = rssm.observe(obs_b, act_b, done_seq=dones)
        # The reset at step d+1 wipes every trace of the pre-d divergence.
        torch.testing.assert_close(out_a["h"][:, d + 1:], out_b["h"][:, d + 1:])
        torch.testing.assert_close(out_a["z"][:, d + 1:], out_b["z"][:, d + 1:])

    def test_no_reset_without_done_seq(self, rssm):
        """Without done_seq the recurrence runs unbroken — a pre-boundary
        difference DOES propagate (guards against an accidental no-op)."""
        batch, time, d = 3, 8, 3
        obs_a = torch.randn(batch, time, 4)
        act_a = torch.randn(batch, time, 2)
        obs_b = obs_a.clone()
        act_b = act_a.clone()
        obs_b[:, :d] = torch.randn(batch, d, 4)
        act_b[:, :d] = torch.randn(batch, d, 2)

        torch.manual_seed(0)
        out_a = rssm.observe(obs_a, act_a)
        torch.manual_seed(0)
        out_b = rssm.observe(obs_b, act_b)
        assert not torch.allclose(out_a["h"][:, d + 1:], out_b["h"][:, d + 1:])


class TestLossFullSequenceValid:
    """loss(full_sequence_valid=True) counts every step — needed for device
    rollout rows, which span multiple auto-reset episodes with no padding."""

    def test_changes_loss_for_multi_episode_sequence(self, rssm):
        """The default cumsum mask drops every step after the second done;
        full_sequence_valid keeps them, so the loss must differ."""
        batch, time = 2, 10
        obs = torch.randn(batch, time, 4)
        actions = torch.randn(batch, time, 2)
        rewards = torch.ones(batch, time)
        dones = torch.zeros(batch, time)
        dones[:, 3] = 1.0
        dones[:, 7] = 1.0   # two episode boundaries within the sequence
        torch.manual_seed(0)
        masked = rssm.loss(obs, actions, rewards, dones)
        torch.manual_seed(0)
        full = rssm.loss(obs, actions, rewards, dones, full_sequence_valid=True)
        assert not torch.allclose(masked["total_loss"], full["total_loss"])

    def test_noop_when_no_done(self, rssm):
        """With no done the cumsum mask is already all-ones — the flag must
        be a no-op (guards against it changing the gym path spuriously)."""
        batch, time = 2, 8
        obs = torch.randn(batch, time, 4)
        actions = torch.randn(batch, time, 2)
        rewards = torch.ones(batch, time)
        dones = torch.zeros(batch, time)
        torch.manual_seed(0)
        masked = rssm.loss(obs, actions, rewards, dones)
        torch.manual_seed(0)
        full = rssm.loss(obs, actions, rewards, dones, full_sequence_valid=True)
        torch.testing.assert_close(masked["total_loss"], full["total_loss"])
