"""Tests for LatentPolicyHead — cross-dim transfer via shared trunk."""

import torch
import numpy as np
import pytest

from ragnarok.learning.latent_policy import LatentPolicyHead
from ragnarok.infrastructure.device import DEVICE


class TestLatentPolicyForward:
    """Basic forward pass tests."""

    def test_discrete_forward_shapes(self):
        """Discrete policy returns (logits, value) with correct shapes."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        latent = torch.randn(4, 160, device=DEVICE)
        logits, value = policy(latent)
        assert logits.shape == (4, 3)
        assert value.shape == (4,)

    def test_continuous_forward_shapes(self):
        """Continuous policy returns (mean, logstd, value) with correct shapes."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False).to(DEVICE)
        latent = torch.randn(4, 160, device=DEVICE)
        mean, logstd, value = policy(latent)
        assert mean.shape == (4, 2)
        assert logstd.shape == (4, 2)
        assert value.shape == (4,)

    def test_logstd_clamped(self):
        """Logstd should be clamped to [-5.0, 2.0]."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False).to(DEVICE)
        latent = torch.randn(32, 160, device=DEVICE) * 100  # Large input
        _, logstd, _ = policy(latent)
        assert logstd.min() >= -5.0
        assert logstd.max() <= 2.0


class TestTrunkTransfer:
    """Test trunk save/load for cross-env transfer."""

    def test_trunk_state_dict_keys(self):
        """Trunk state dict should contain only shared + critic keys."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        trunk_sd = policy.get_trunk_state_dict()
        for key in trunk_sd:
            assert key.startswith("shared.") or key.startswith("critic_head.")
        # Should NOT contain actor keys
        assert not any(k.startswith("actor_head.") for k in trunk_sd)

    def test_trunk_roundtrip(self):
        """Save trunk -> load into fresh policy -> shared weights match."""
        src = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=5, discrete=True).to(DEVICE)
        dst.load_trunk_state_dict(trunk_sd)

        for key in trunk_sd:
            assert torch.equal(dst.state_dict()[key], trunk_sd[key])

    def test_cross_dim_transfer(self):
        """Transfer trunk between policies with different action dims."""
        # CartPole-like (2 actions) -> Acrobot-like (3 actions)
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)

        # Train src briefly to get non-init weights
        optimizer = torch.optim.Adam(src.parameters(), lr=1e-3)
        for _ in range(10):
            latent = torch.randn(8, 160, device=DEVICE)
            logits, value = src(latent)
            loss = -value.mean() + logits.sum() * 0.01
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        dst_before = {k: v.clone() for k, v in dst.state_dict().items()
                      if k.startswith("shared.")}
        dst.load_trunk_state_dict(trunk_sd)
        dst_after = dst.state_dict()

        # Shared weights should have changed
        changed = any(not torch.equal(dst_before[k], dst_after[k])
                      for k in dst_before)
        assert changed, "Shared weights should change after trunk transfer"

        # Actor head should be untouched (different dim)
        assert dst_after["actor_head.weight"].shape == (3, 128)

    def test_discrete_to_continuous_trunk(self):
        """Trunk transfer works across discrete/continuous boundaries."""
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=1, discrete=False).to(DEVICE)
        dst.load_trunk_state_dict(trunk_sd)

        # Shared + critic should match
        for key in trunk_sd:
            assert torch.equal(dst.state_dict()[key], trunk_sd[key])

        # Continuous heads should still work
        latent = torch.randn(4, 160, device=DEVICE)
        mean, logstd, value = dst(latent)
        assert mean.shape == (4, 1)

    def test_mismatched_latent_dim_ignored(self):
        """If latent_dim differs, mismatched keys are skipped (no crash)."""
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        # Different latent_dim -> first linear layer won't match
        dst = LatentPolicyHead(latent_dim=256, action_dim=3, discrete=True).to(DEVICE)
        dst_before = {k: v.clone() for k, v in dst.state_dict().items()}

        dst.load_trunk_state_dict(trunk_sd)  # Should not crash

        # Keys with matching shapes should transfer, others stay
        dst_after = dst.state_dict()
        # shared.0.weight has shape (128, latent_dim), so it won't match
        assert dst_after["shared.0.weight"].shape == (128, 256)


class TestAgentIntegration:
    """Test latent policy integration with agent."""

    def test_agent_has_latent_policy(self):
        """Agent should create a latent policy on init."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        assert hasattr(agent, "latent_policy")
        assert isinstance(agent.latent_policy, LatentPolicyHead)
        expected_dim = config.world_model.hidden_dim + config.world_model.stoch_dim
        assert agent.latent_policy.latent_dim == expected_dim
        env.close()

    def test_crystallization_saves_trunk(self):
        """Skill crystallization should include latent trunk state dict."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent
        from ragnarok.skills.skill import Skill
        import tempfile

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        with tempfile.TemporaryDirectory() as tmpdir:
            config.skill.skills_dir = tmpdir
            env = RagnarokEnv(spec.gym_name, seed=42)
            agent = RagnarokAgent(config, env)

            # Force crystallization
            agent._crystallization_rewards = [500.0] * 20
            agent.total_episodes = 100
            skill = agent.check_crystallization()

            if skill is not None:
                assert hasattr(skill, "latent_trunk_state_dict")
                assert len(skill.latent_trunk_state_dict) > 0
            env.close()
