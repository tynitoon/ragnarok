"""Tests for latent curiosity (KL from RSSM) and adaptive horizon."""

import numpy as np
import torch
import pytest

from ragnarok.core.rssm import RSSM
from ragnarok.learning.curiosity import LatentCuriosityModule
from ragnarok.infrastructure.device import DEVICE


@pytest.fixture
def rssm():
    """Create a small RSSM for testing."""
    return RSSM(obs_dim=4, action_dim=2, hidden_dim=32, stoch_dim=8,
                encoder_hidden=32).to(DEVICE)


@pytest.fixture
def latent_curiosity(rssm):
    """Create LatentCuriosityModule with low min_episodes for testing."""
    return LatentCuriosityModule(rssm=rssm, beta=0.5, min_rssm_episodes=2)


class TestLatentCuriosityModule:

    def test_kl_shape(self, latent_curiosity):
        """KL output shape should be (T,)."""
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        # Needs to be "ready" first
        latent_curiosity._episodes_seen = 5
        rewards = latent_curiosity.compute_batch_kl(obs, act)
        assert rewards.shape == (10,)
        assert rewards.dtype == np.float32

    def test_kl_nonnegative(self, latent_curiosity):
        """Intrinsic rewards from KL should be non-negative (ReLU)."""
        latent_curiosity._episodes_seen = 5
        obs = np.random.randn(20, 4).astype(np.float32)
        act = np.random.randn(20, 2).astype(np.float32)
        rewards = latent_curiosity.compute_batch_kl(obs, act)
        assert np.all(rewards >= 0), "KL rewards should be non-negative"

    def test_rssm_ready_flag(self, latent_curiosity):
        """RSSM should only be ready after min_rssm_episodes."""
        assert not latent_curiosity.rssm_ready
        latent_curiosity.reset_episode(action_dim=2)
        assert not latent_curiosity.rssm_ready
        latent_curiosity.reset_episode(action_dim=2)
        assert latent_curiosity.rssm_ready  # min_rssm_episodes=2

    def test_step_kl_updates_state(self, latent_curiosity):
        """Per-step KL should update RSSM internal state."""
        latent_curiosity._episodes_seen = 5
        latent_curiosity.reset_episode(action_dim=2)
        h_before = latent_curiosity._h.clone()

        obs = np.random.randn(4).astype(np.float32)
        act = np.random.randn(2).astype(np.float32)
        kl = latent_curiosity.compute_step_kl(obs, act)

        assert isinstance(kl, float)
        assert kl >= 0.0
        # State should have changed
        assert not torch.equal(h_before, latent_curiosity._h)

    def test_novel_higher_kl(self, rssm, latent_curiosity):
        """States the RSSM hasn't been trained on should yield higher KL."""
        latent_curiosity._episodes_seen = 5

        # Train RSSM briefly on a specific region
        familiar_obs = torch.randn(4, 10, 4, device=DEVICE) * 0.1
        familiar_act = torch.randn(4, 10, 2, device=DEVICE) * 0.1
        familiar_rew = torch.zeros(4, 10, device=DEVICE)
        familiar_done = torch.zeros(4, 10, device=DEVICE)
        optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)

        for _ in range(30):
            losses = rssm.loss(familiar_obs, familiar_act, familiar_rew, familiar_done)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

        # Familiar region
        fam_obs_np = np.random.randn(10, 4).astype(np.float32) * 0.1
        fam_act_np = np.random.randn(10, 2).astype(np.float32) * 0.1
        fam_kl = latent_curiosity.compute_batch_kl(fam_obs_np, fam_act_np)

        # Novel region (very different scale)
        nov_obs_np = np.random.randn(10, 4).astype(np.float32) * 10.0
        nov_act_np = np.random.randn(10, 2).astype(np.float32) * 10.0
        nov_kl = latent_curiosity.compute_batch_kl(nov_obs_np, nov_act_np)

        # Novel region should have higher mean KL
        assert nov_kl.mean() >= fam_kl.mean() * 0.5, \
            f"Novel KL ({nov_kl.mean():.4f}) should be higher than familiar ({fam_kl.mean():.4f})"

    def test_state_dict_roundtrip(self, rssm):
        """State dict save/load should preserve running stats."""
        lc1 = LatentCuriosityModule(rssm=rssm, beta=0.3, min_rssm_episodes=1)
        lc1._episodes_seen = 10
        # Push some stats through
        obs = np.random.randn(5, 4).astype(np.float32)
        act = np.random.randn(5, 2).astype(np.float32)
        lc1.compute_batch_kl(obs, act)

        sd = lc1.state_dict()
        lc2 = LatentCuriosityModule(rssm=rssm, beta=0.3, min_rssm_episodes=1)
        lc2.load_state_dict(sd)

        assert lc2._reward_mean == lc1._reward_mean
        assert lc2._reward_var == lc1._reward_var
        assert lc2._reward_count == lc1._reward_count
        assert lc2._episodes_seen == lc1._episodes_seen

    def test_zero_params(self, latent_curiosity):
        """Latent curiosity should have zero extra parameters."""
        assert latent_curiosity.params_count == 0

    def test_short_sequence(self, latent_curiosity):
        """Sequences < 2 should return zeros."""
        latent_curiosity._episodes_seen = 5
        obs = np.random.randn(1, 4).astype(np.float32)
        act = np.random.randn(1, 2).astype(np.float32)
        rewards = latent_curiosity.compute_batch_kl(obs, act)
        assert rewards.shape == (1,)
        assert rewards[0] == 0.0


class TestAdaptiveHorizon:

    def test_horizon_adapts_with_episode_length(self):
        """Horizon should change based on average episode length."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.policy.horizon_update_interval = 5
        config.policy.max_horizon = 50
        config.policy.horizon_ratio = 0.33

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        initial_horizon = agent.dream_trainer.horizon

        # Simulate episode lengths
        for length in [200, 180, 190, 210, 195]:
            agent.episode_lengths.append(length)
        agent.total_episodes = 5
        agent._update_adaptive_horizon()

        # avg ~195, * 0.33 = ~64 -> clamped to max_horizon=50
        assert agent.dream_trainer.horizon == 50

        # Now short episodes
        agent.episode_lengths.clear()
        for length in [20, 25, 22, 18, 21]:
            agent.episode_lengths.append(length)
        agent.total_episodes = 10
        agent._update_adaptive_horizon()

        # avg ~21, * 0.33 = ~7
        new_horizon = agent.dream_trainer.horizon
        assert 5 <= new_horizon <= 10, f"Expected horizon 5-10, got {new_horizon}"

        env.close()

    def test_horizon_minimum_5(self):
        """Horizon should never go below 5."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.policy.horizon_update_interval = 5
        config.policy.horizon_ratio = 0.33

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # Very short episodes
        for length in [3, 4, 5, 2, 3]:
            agent.episode_lengths.append(length)
        agent.total_episodes = 5
        agent._update_adaptive_horizon()

        assert agent.dream_trainer.horizon >= 5
        env.close()
