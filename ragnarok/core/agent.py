"""Ragnarok Agent: the main orchestrator.

Ties together the world model (RSSM), policy (actor-critic),
episodic memory, skill library, and training loops into a single
coherent agent that can:
    1. Interact with environments
    2. Learn from experience (world model + dream training)
    3. Crystallize skills when proficient
    4. Transfer skills to new tasks
"""

import numpy as np
import torch
from collections import deque

from ragnarok.core.rssm import RSSM
from ragnarok.core.policy import ActorCritic
from ragnarok.core.normalizer import RunningNormalizer
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.memory.episodic import EpisodicMemory
from ragnarok.skills.skill import Skill
from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.selector import SkillSelector
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.dreamer import DreamTrainer
from ragnarok.learning.real_experience import RealExperienceTrainer
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE, to_numpy
from ragnarok.infrastructure.checkpoint import save_checkpoint, load_checkpoint


class RagnarokAgent:
    """The self-learning agent."""

    def __init__(self, config: RagnarokConfig, env: RagnarokEnv):
        self.config = config
        self.env = env

        # World Model
        self.rssm = RSSM(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dim=config.world_model.hidden_dim,
            stoch_dim=config.world_model.stoch_dim,
            encoder_hidden=config.world_model.encoder_hidden,
        ).to(DEVICE)

        # Policy — input = h + z (no memory context initially)
        state_dim = self.rssm.state_dim
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=env.action_dim,
            hidden=config.policy.hidden_dim,
            mid=config.policy.mid_dim,
            discrete=env.is_discrete,
        ).to(DEVICE)

        # Memory
        self.replay_buffer = ReplayBuffer(capacity=config.memory.replay_capacity)
        self.episodic_memory = EpisodicMemory(
            state_dim=config.world_model.hidden_dim,
            action_dim=env.action_dim,
            capacity=config.memory.episodic_capacity,
        )

        # Skills
        self.skill_library = SkillLibrary(skills_dir=config.skill.skills_dir)
        self.skill_selector = SkillSelector(self.rssm, self.skill_library)

        # Trainers
        self.wm_trainer = WorldModelTrainer(
            rssm=self.rssm,
            replay_buffer=self.replay_buffer,
            lr=config.world_model.lr,
            grad_clip=config.world_model.grad_clip,
            kl_weight=config.world_model.kl_weight,
            free_nats=config.world_model.free_nats,
            batch_size=config.world_model.batch_size,
            seq_length=config.world_model.sequence_length,
        )
        self.dream_trainer = DreamTrainer(
            rssm=self.rssm,
            actor_critic=self.actor_critic,
            replay_buffer=self.replay_buffer,
            imagination_horizon=config.policy.imagination_horizon,
            imagination_batch=config.policy.imagination_batch,
            gamma=config.policy.gamma,
            gae_lambda=config.policy.gae_lambda,
            entropy_bonus=config.policy.entropy_bonus,
            actor_lr=config.policy.actor_lr,
            critic_lr=config.policy.critic_lr,
            grad_clip=config.policy.grad_clip,
        )

        # Real experience trainer (direct A2C on raw observations)
        reward_shaper = self._get_reward_shaper(env.env_name)
        self.real_trainer = RealExperienceTrainer(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            gamma=config.policy.gamma,
            entropy_coeff=0.01,
            lr=3e-4,
            grad_clip=0.5,
            reward_shaper=reward_shaper,
        )

        # Tracking
        self.episode_rewards: deque[float] = deque(maxlen=config.skill.crystallization_window)
        self.recent_episodes: deque = deque(maxlen=10)  # Recent episodes for real-experience training
        self.total_episodes = 0
        self.total_steps = 0
        self.h_accum: list[np.ndarray] = []  # For latent centroid computation

    @staticmethod
    def _get_reward_shaper(env_name: str):
        """Get environment-specific reward shaping function."""
        if "MountainCar" in env_name:
            # MountainCar: encourage reaching higher positions
            # obs[0] = position (range: -1.2 to 0.6), obs[1] = velocity
            # Goal at position >= 0.5
            def shaper(obs, reward, next_obs):
                # Potential-based shaping (position + abs(velocity))
                height_bonus = (next_obs[0] + 1.2) / 1.8  # Normalize to [0, 1]
                velocity_bonus = abs(next_obs[1]) * 10  # Encourage movement
                return reward + 0.1 * height_bonus + 0.05 * velocity_bonus
            return shaper
        return None  # No shaping for other environments

    def collect_episode(self, explore_ratio: float = 0.1) -> float:
        """Run one episode, collecting data into buffers.

        Args:
            explore_ratio: probability of random action (epsilon-greedy)

        Returns:
            Total episode reward
        """
        obs = self.env.reset()
        h, z = self.rssm.initial_state(1, DEVICE)
        prev_action = torch.zeros(1, self.env.action_dim, device=DEVICE)

        observations, actions, rewards, dones = [], [], [], []
        total_reward = 0.0

        done = False
        while not done:
            obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                # Encode current observation
                h, z = self.rssm.encode_observation(obs_t, h, z, prev_action)

                # Epsilon-greedy exploration
                if np.random.random() < explore_ratio:
                    action_np = self.env.sample_random_action()
                else:
                    action_t = self.actor_critic.act(h, z, deterministic=True)
                    action_np = to_numpy(action_t.squeeze(0))

            # Store h_t for centroid computation
            self.h_accum.append(to_numpy(h.squeeze(0)))

            # Store in episodic memory
            h_np = to_numpy(h.squeeze(0))
            self.episodic_memory.add(h_np, action_np, 0.0)  # reward updated later

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            total_reward += reward

            observations.append(obs)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(float(done))

            obs = next_obs
            prev_action = torch.tensor(action_np, device=DEVICE).unsqueeze(0)

            self.total_steps += 1

        # Store episode in replay buffer
        obs_arr = np.array(observations)
        act_arr = np.array(actions)
        rew_arr = np.array(rewards)
        done_arr = np.array(dones)

        self.replay_buffer.add_episode(obs_arr, act_arr, rew_arr, done_arr)
        self.recent_episodes.append((obs_arr, act_arr, rew_arr, done_arr))

        self.episode_rewards.append(total_reward)
        self.total_episodes += 1

        return total_reward

    def train_world_model(self, steps: int | None = None) -> dict[str, float]:
        """Train the RSSM world model."""
        steps = steps or self.config.world_model.train_steps
        return self.wm_trainer.train(steps)

    def train_policy(self, steps: int | None = None) -> dict[str, float]:
        """Train the policy via dream training."""
        steps = steps or self.config.policy.train_steps
        return self.dream_trainer.train(steps)

    def train_policy_real(self) -> tuple[float, dict[str, float]]:
        """Collect and train from real experience (A2C on raw observations).

        Also feeds episode data into the replay buffer for world model training.
        Returns (episode_reward, metrics).
        """
        reward, metrics, episode_data = self.real_trainer.collect_and_train(self.env)

        # Feed into replay buffer for world model training
        if episode_data is not None:
            obs, acts, rews, dones = episode_data
            self.replay_buffer.add_episode(obs, acts, rews, dones)
            self.recent_episodes.append(episode_data)
            self.total_steps += len(rews)

        self.episode_rewards.append(reward)
        self.total_episodes += 1
        return reward, metrics

    def check_crystallization(self, eval_episodes: int = 5) -> Skill | None:
        """Check if the current policy should be crystallized into a skill.

        Runs evaluation episodes (deterministic) to get true performance.
        Returns the new Skill if crystallized, None otherwise.
        """
        if self.total_episodes < self.config.skill.min_episodes:
            return None

        # Run actual evaluation (deterministic policy)
        eval_reward = self.real_trainer.evaluate(self.env, episodes=eval_episodes)
        threshold = self.config.skill.thresholds.get(self.env.env_name, float("inf"))

        if eval_reward >= threshold:
            # Compute latent centroid from world model encoding
            centroid = np.zeros(self.config.world_model.hidden_dim)
            if self.replay_buffer.num_episodes > 0:
                try:
                    obs, acts, _, _ = self.replay_buffer.sample_sequences(10, 10)
                    obs_t = torch.tensor(obs, device=DEVICE)
                    act_t = torch.tensor(acts, device=DEVICE)
                    with torch.no_grad():
                        outputs = self.rssm.observe(obs_t, act_t)
                        centroid = to_numpy(outputs["h"].mean(dim=(0, 1)))
                except Exception:
                    pass

            skill = Skill(
                name=f"{self.env.env_name}_{self.total_episodes}ep",
                env_name=self.env.env_name,
                policy_state_dict={k: v.cpu() for k, v in self.real_trainer.policy.state_dict().items()},
                latent_centroid=centroid,
                performance=eval_reward,
                normalizer_state=self.env.normalizer.state_dict(),
                episodes_trained=self.total_episodes,
            )
            self.skill_library.save_skill(skill)
            return skill

        return None

    def try_transfer(self) -> Skill | None:
        """Try to find and load a relevant skill for the current environment."""
        skill = self.skill_selector.select(self.env)
        if skill is not None:
            self.actor_critic.load_state_dict(
                {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
            )
        return skill

    def save(self, path: str):
        """Save full agent state to checkpoint."""
        save_checkpoint(
            path,
            rssm=self.rssm.state_dict(),
            actor_critic=self.actor_critic.state_dict(),
            normalizer=self.env.normalizer.state_dict(),
            episodic_memory=self.episodic_memory.state_dict(),
            total_episodes=self.total_episodes,
            total_steps=self.total_steps,
            episode_rewards=list(self.episode_rewards),
        )

    def load(self, path: str):
        """Load agent state from checkpoint."""
        ckpt = load_checkpoint(path, device=DEVICE)
        self.rssm.load_state_dict(ckpt["rssm"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.env.normalizer = RunningNormalizer.from_state_dict(ckpt["normalizer"])
        self.episodic_memory = EpisodicMemory.from_state_dict(ckpt["episodic_memory"])
        self.total_episodes = ckpt["total_episodes"]
        self.total_steps = ckpt["total_steps"]
        self.episode_rewards = deque(ckpt["episode_rewards"],
                                     maxlen=self.config.skill.crystallization_window)
