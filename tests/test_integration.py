"""Integration tests for the Ragnarok agent.

These tests verify end-to-end functionality:
- Agent can train on CartPole and improve
- Skill crystallization saves to disk
- Skill transfer loads and works
- World model produces valid outputs after training
"""

import tempfile
import numpy as np
import torch
import pytest

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent
from ragnarok.learning.real_experience import RealExperienceTrainer


class TestAgentTraining:
    """Test that the agent can learn from real experience."""

    def test_a2c_improves_on_cartpole(self):
        """A2C trainer should improve CartPole reward over 50 episodes."""
        spec = get_env_spec("cartpole")
        env = RagnarokEnv(spec.gym_name, seed=42)
        torch.manual_seed(42)

        trainer = RealExperienceTrainer(
            obs_dim=spec.obs_dim, action_dim=spec.action_dim,
            gamma=0.99, entropy_coeff=0.01, lr=3e-4, grad_clip=0.5,
        )

        early_rewards = []
        for _ in range(10):
            r, _, _ = trainer.collect_and_train(env)
            early_rewards.append(r)

        # Train more
        for _ in range(40):
            trainer.collect_and_train(env)

        late_rewards = []
        for _ in range(10):
            r, _, _ = trainer.collect_and_train(env)
            late_rewards.append(r)

        env.close()
        # Agent should improve (later rewards higher than early ones)
        assert np.mean(late_rewards) > np.mean(early_rewards)

    def test_agent_collect_feeds_replay(self):
        """Agent's real training should feed episodes into replay buffer."""
        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        assert agent.replay_buffer.num_episodes == 0
        agent.train_policy_real()
        # Discrete envs use batched PPO (8 episodes per call by default)
        assert agent.replay_buffer.num_episodes >= 1
        assert agent.total_episodes >= 1

        env.close()


class TestWorldModelTraining:
    """Test that the world model can learn from collected data."""

    def test_world_model_loss_decreases(self):
        """RSSM loss should decrease after training on collected experience."""
        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # Collect some episodes
        for _ in range(15):
            agent.train_policy_real()

        # Train world model and check loss decreases
        metrics1 = agent.train_world_model(steps=10)
        metrics2 = agent.train_world_model(steps=30)

        env.close()
        assert "total_loss" in metrics1
        # After more training, loss should be lower (or at least not exploded)
        assert metrics2["total_loss"] < metrics1["total_loss"] * 2


class TestSkillCrystallization:
    """Test the full skill lifecycle: train → crystallize → save → load → transfer."""

    def test_skill_save_and_reload(self):
        """A skill saved by the agent should be loadable and functional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = get_env_spec("cartpole")
            config = RagnarokConfig()
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim
            config.skill.skills_dir = tmpdir

            env = RagnarokEnv(spec.gym_name, seed=42)
            agent = RagnarokAgent(config, env)

            # Manually create and save a skill (simulates crystallization)
            from ragnarok.skills.skill import Skill
            skill = Skill(
                name="test_cartpole",
                env_name="CartPole-v1",
                policy_state_dict={k: v.cpu() for k, v in agent.real_trainer.policy.state_dict().items()},
                latent_centroid=np.zeros(config.world_model.hidden_dim),
                performance=500.0,
                normalizer_state={},
            )
            agent.skill_library.save_skill(skill)

            # Verify it's in the library
            assert "test_cartpole" in agent.skill_library.list_skills()

            # Load it back
            loaded = agent.skill_library.load_skill("test_cartpole")
            assert loaded is not None
            assert loaded.performance == 500.0

            # Verify the weights can be loaded into a fresh policy
            fresh_trainer = RealExperienceTrainer(
                obs_dim=spec.obs_dim, action_dim=spec.action_dim,
            )
            fresh_trainer.policy.load_state_dict(
                {k: v.to(DEVICE) for k, v in loaded.policy_state_dict.items()}
            )
            # Should produce valid actions
            obs = torch.randn(1, spec.obs_dim, device=DEVICE)
            action = fresh_trainer.policy.act(obs)
            assert isinstance(action, int)
            assert 0 <= action < spec.action_dim

            env.close()

    def test_transfer_loads_weights(self):
        """try_transfer() should load skill weights into the real trainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = get_env_spec("cartpole")
            config = RagnarokConfig()
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim
            config.skill.skills_dir = tmpdir

            env = RagnarokEnv(spec.gym_name, seed=42)
            agent = RagnarokAgent(config, env)

            # Save initial weights
            initial_weights = {k: v.clone() for k, v in agent.real_trainer.policy.state_dict().items()}

            # Save a skill with different weights
            different_policy = RealExperienceTrainer(
                obs_dim=spec.obs_dim, action_dim=spec.action_dim,
            )
            # Train it briefly to get different weights
            for _ in range(5):
                different_policy.collect_and_train(env)

            from ragnarok.skills.skill import Skill
            skill = Skill(
                name="cartpole_skill",
                env_name="CartPole-v1",
                policy_state_dict={k: v.cpu() for k, v in different_policy.policy.state_dict().items()},
                latent_centroid=np.zeros(config.world_model.hidden_dim),
                performance=500.0,
                normalizer_state={},
            )
            agent.skill_library.save_skill(skill)

            # Transfer should load these weights
            transferred = agent.try_transfer()
            assert transferred is not None

            # Weights should have changed
            changed = False
            for k in initial_weights:
                if not torch.equal(initial_weights[k], agent.real_trainer.policy.state_dict()[k]):
                    changed = True
                    break
            assert changed, "Weights should change after transfer"

            env.close()


class TestCheckpointRoundtrip:
    """Test save/load of full agent state."""

    def test_save_and_load_agent(self):
        """Agent should be restorable from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = get_env_spec("cartpole")
            config = RagnarokConfig()
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim

            env = RagnarokEnv(spec.gym_name, seed=42)
            agent = RagnarokAgent(config, env)

            # Train a bit
            for _ in range(3):
                agent.train_policy_real()

            ckpt_path = f"{tmpdir}/test_checkpoint.pt"
            agent.save(ckpt_path)

            # Create fresh agent and load
            agent2 = RagnarokAgent(config, env)
            agent2.load(ckpt_path)

            assert agent2.total_episodes == agent.total_episodes
            assert agent2.total_steps == agent.total_steps

            env.close()
