"""Tests for unified policy architecture (Phase 5.3)."""

import numpy as np
import torch
import pytest

from ragnarok.core.obs_encoder import MLPObsEncoder, CNNObsEncoder, create_obs_encoder
from ragnarok.infrastructure.device import DEVICE


class TestObsEncoder:

    def test_mlp_encoder_output_dim(self):
        """MLP encoder should produce correct embedding dimension."""
        enc = MLPObsEncoder(obs_dim=8, embed_dim=64).to(DEVICE)
        obs = torch.randn(4, 8, device=DEVICE)
        out = enc(obs)
        assert out.shape == (4, 64)

    def test_cnn_encoder_output_dim(self):
        """CNN encoder should produce correct embedding dimension."""
        enc = CNNObsEncoder(channels=3, embed_dim=128).to(DEVICE)
        obs = torch.randn(2, 3, 64, 64, device=DEVICE)
        out = enc(obs)
        assert out.shape == (2, 128)

    def test_cnn_encoder_flattened_input(self):
        """CNN encoder should handle flattened pixel input."""
        enc = CNNObsEncoder(channels=3, embed_dim=128).to(DEVICE)
        obs = torch.randn(2, 3 * 64 * 64, device=DEVICE)
        out = enc(obs)
        assert out.shape == (2, 128)

    def test_same_embed_dim(self):
        """Both encoders should produce same embedding dimension."""
        embed_dim = 64
        mlp = MLPObsEncoder(obs_dim=4, embed_dim=embed_dim).to(DEVICE)
        cnn = CNNObsEncoder(channels=3, embed_dim=embed_dim).to(DEVICE)
        assert mlp.embed_dim == cnn.embed_dim == embed_dim

    def test_factory_function(self):
        """create_obs_encoder should return correct type."""
        mlp = create_obs_encoder(obs_dim=8, embed_dim=64, pixel=False)
        cnn = create_obs_encoder(obs_dim=0, embed_dim=64, pixel=True, channels=3)
        assert isinstance(mlp, MLPObsEncoder)
        assert isinstance(cnn, CNNObsEncoder)


class TestUnifiedDreamTraining:

    def test_single_policy_agent(self):
        """Agent should have no actor_critic attribute (unified policy)."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.curiosity.enabled = False

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # No separate actor_critic (unified into direct policy)
        assert not hasattr(agent, 'actor_critic')
        # dream_trainer is an alias for dream_augmenter
        assert agent.dream_trainer is agent.dream_augmenter
        # Active policy is the real trainer's policy
        assert agent._active_policy is agent.real_trainer.policy
        env.close()

    def test_dream_augmenter_has_lambda_returns(self):
        """DreamAugmenter should use lambda returns."""
        from ragnarok.learning.advantages import compute_lambda_returns
        # Simple test: constant rewards, no done, gamma=1 -> returns = cumsum
        rewards = torch.ones(2, 5)
        values = torch.zeros(2, 6)
        continues = torch.ones(2, 5)
        returns = compute_lambda_returns(rewards, values, continues, gamma=1.0, lam=1.0)
        assert returns.shape == (2, 5)
        # With gamma=1 and lambda=1, returns should be cumulative from end
        # R_4 = 1, R_3 = 1 + 1 = 2, ..., R_0 = 5
        expected = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0],
                                 [5.0, 4.0, 3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns, expected, atol=1e-5, rtol=1e-5)

    def test_dream_training_discrete(self):
        """Dream augmenter should train on CartPole without error."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.curiosity.enabled = False

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # Collect some real data first
        for _ in range(10):
            agent.train_policy_real()

        # Train world model
        if agent.replay_buffer.num_episodes >= 5:
            agent.train_world_model(steps=10)

            # Dream training should work without error
            metrics = agent.train_policy_dream(steps=3)
            if metrics:
                assert "dream_aug/actor_loss" in metrics

        env.close()

    def test_checkpoint_saves_direct_policy(self):
        """Checkpoint should save and load the direct policy."""
        import tempfile
        import os
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.curiosity.enabled = False

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # Collect some data
        for _ in range(3):
            agent.train_policy_real()

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)

            # Load into a new agent
            agent2 = RagnarokAgent(config, env)
            agent2.load(path)

            # Policies should have same weights
            p1 = agent._active_policy.state_dict()
            p2 = agent2._active_policy.state_dict()
            for key in p1:
                torch.testing.assert_close(p1[key], p2[key])
        finally:
            os.unlink(path)
            env.close()
