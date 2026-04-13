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
from ragnarok.core.cnn import CNNEncoder, CNNDecoder
from ragnarok.core.normalizer import RunningNormalizer
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.memory.episodic import EpisodicMemory
from ragnarok.skills.skill import Skill
from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.selector import SkillSelector
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.real_experience import (
    RealExperienceTrainer, PixelPPOTrainer,
)
from ragnarok.learning.sac import SACTrainer
from ragnarok.learning.dream_augmenter import DreamAugmenter
from ragnarok.learning.curiosity import CuriosityModule, LatentCuriosityModule
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE, to_numpy
from ragnarok.infrastructure.checkpoint import save_checkpoint, load_checkpoint


class RagnarokAgent:
    """The self-learning agent."""

    def __init__(self, config: RagnarokConfig, env: RagnarokEnv):
        self.config = config
        self.env = env

        # World Model — use CNN encoder/decoder for pixel observations
        encoder = None
        decoder = None
        if getattr(env, 'pixel_obs', False):
            n_channels = getattr(env, 'n_channels', 12)
            encoder = CNNEncoder(
                channels=n_channels, feature_dim=config.world_model.encoder_hidden,
            )
            decoder = CNNDecoder(
                latent_dim=config.world_model.hidden_dim + config.world_model.stoch_dim,
                channels=3,
            )

        self.rssm = RSSM(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dim=config.world_model.hidden_dim,
            stoch_dim=config.world_model.stoch_dim,
            encoder_hidden=config.world_model.encoder_hidden,
            encoder=encoder,
            decoder=decoder,
            ensemble_cores=config.transfer.ensemble_cores,
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

        # Trainers — use smaller batches for pixel observations
        is_pixel = getattr(env, 'pixel_obs', False)
        wm_batch = config.world_model.pixel_batch_size if is_pixel else config.world_model.batch_size
        wm_seq = config.world_model.pixel_sequence_length if is_pixel else config.world_model.sequence_length
        self.wm_trainer = WorldModelTrainer(
            rssm=self.rssm,
            replay_buffer=self.replay_buffer,
            lr=config.world_model.lr,
            grad_clip=config.world_model.grad_clip,
            kl_weight=config.world_model.kl_weight,
            free_nats=config.world_model.free_nats,
            batch_size=wm_batch,
            seq_length=wm_seq,
        )
        dream_batch = config.policy.pixel_dream_batch if is_pixel else config.policy.imagination_batch

        # Real experience trainer
        reward_shaper = self._get_reward_shaper(env.env_name)
        entropy_coeff, lr = self._get_training_hparams(env.env_name)

        # Intrinsic curiosity: forward prediction (fallback) + latent KL (primary)
        self.curiosity: CuriosityModule | None = None
        self.latent_curiosity: LatentCuriosityModule | None = None
        if config.curiosity.enabled and not getattr(env, 'pixel_obs', False):
            beta = self._get_curiosity_beta(env.env_name, config.curiosity.beta)
            # Forward predictor as baseline / fallback
            self.curiosity = CuriosityModule(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                hidden=config.curiosity.hidden_dim,
                lr=config.curiosity.lr,
                beta=beta,
                grad_clip=config.curiosity.grad_clip,
            )
            # Latent curiosity: KL(posterior||prior) from RSSM
            if config.curiosity.use_latent:
                self.latent_curiosity = LatentCuriosityModule(
                    rssm=self.rssm,
                    beta=beta,
                    min_rssm_episodes=config.curiosity.min_rssm_episodes,
                )

        # Use SAC for continuous envs, A2C/PPO for discrete
        self.sac_trainer: SACTrainer | None = None
        if not env.is_discrete:
            # Disable observation normalization for off-policy SAC.
            # RunningNormalizer's changing stats create distribution
            # shift in the replay buffer. SAC's 256-hidden network
            # handles raw observation scales without normalization.
            env.normalize = False
            self.sac_trainer = SACTrainer(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                action_low=env.action_low,
                action_high=env.action_high,
                gamma=config.policy.gamma,
                reward_shaper=reward_shaper,
                curiosity=self.curiosity,
                latent_curiosity=self.latent_curiosity,
            )

        self.real_trainer = RealExperienceTrainer(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            discrete=env.is_discrete,
            gamma=config.policy.gamma,
            entropy_coeff=entropy_coeff,
            lr=lr,
            grad_clip=0.5,
            reward_shaper=reward_shaper,
            curiosity=self.curiosity,
            latent_curiosity=self.latent_curiosity,
            action_low=env.action_low,
            action_high=env.action_high,
        )

        # Pixel policy — PPO with CNN (on-policy, no Q-divergence)
        self.pixel_ppo: PixelPPOTrainer | None = None
        if getattr(env, 'pixel_obs', False):
            n_channels = getattr(env, 'n_channels', 3)
            self.pixel_ppo = PixelPPOTrainer(
                action_dim=env.action_dim,
                channels=n_channels,
                state_dim=env.vector_obs_dim,
                aux_weight=2.0,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                entropy_coeff=0.01,
                value_coeff=0.5,
                lr=2.5e-4,
                grad_clip=0.5,
                ppo_epochs=4,
                minibatch_size=64,
            )

        # Dream augmenter (unified dream training: single policy, decoded obs)
        policy_for_dream = self.sac_trainer.policy if self.sac_trainer else self.real_trainer.policy
        self.dream_augmenter = DreamAugmenter(
            rssm=self.rssm,
            policy=policy_for_dream,
            replay_buffer=self.replay_buffer,
            horizon=config.policy.imagination_horizon,
            dream_batch=dream_batch,
            gamma=config.policy.gamma,
            gae_lambda=config.policy.gae_lambda,
            entropy_coeff=entropy_coeff,
            lr=lr * config.policy.dream_lr_ratio,
            disagreement_weight=config.transfer.disagreement_weight,
        )
        # Alias for adaptive horizon (single dream training path)
        self.dream_trainer = self.dream_augmenter

        # Tracking
        self.episode_rewards: deque[float] = deque(maxlen=config.skill.crystallization_window)
        self.episode_lengths: deque[int] = deque(maxlen=50)  # For adaptive horizon
        self.recent_episodes: deque = deque(maxlen=10)  # Recent episodes for real-experience training
        self.total_episodes = 0
        self.total_steps = 0
        self.h_accum: list[np.ndarray] = []  # For latent centroid computation

        # Trust region for transfer (initialized when try_transfer succeeds)
        self._transfer_ref_policy = None  # Frozen copy of transferred policy
        self._transfer_episode_start = None  # Episode when transfer happened

    @property
    def _active_policy(self):
        """The policy used for acting (SAC for continuous, A2C for discrete)."""
        if self.sac_trainer is not None:
            return self.sac_trainer.policy
        return self.real_trainer.policy

    @staticmethod
    def _get_training_hparams(env_name: str) -> tuple[float, float]:
        """Environment-specific hyperparameters (entropy_coeff, lr)."""
        if "MountainCar" in env_name:
            return 0.02, 1e-3  # More exploration, faster learning
        if "Acrobot" in env_name:
            return 0.05, 1e-3  # High exploration for swing-up discovery
        if "Pendulum" in env_name:
            return 0.02, 3e-4  # Continuous: moderate entropy, stable LR
        if "MountainCarContinuous" in env_name:
            return 0.01, 1e-3
        return 0.01, 3e-4  # Default (works for CartPole)

    @staticmethod
    def _get_curiosity_beta(env_name: str, default: float) -> float:
        """Environment-specific curiosity strength.

        Exploration-hard envs (sparse/negative reward) need strong curiosity.
        Dense-reward envs need minimal curiosity to avoid signal pollution.
        """
        if "MountainCar" in env_name and "Continuous" not in env_name:
            return 1.0   # Very strong: -1 reward every step, needs big bonus
        if "Acrobot" in env_name:
            return 0.3   # Moderate: sparse but less extreme
        if "CartPole" in env_name:
            return 0.01  # Minimal: dense +1 reward, curiosity noise hurts
        if "Pendulum" in env_name:
            return 0.05  # Low: continuous cost signal, don't overwhelm
        return default

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
        if "Acrobot" in env_name:
            # Acrobot: encourage the tip to reach higher
            # Raw obs = [cos(t1), sin(t1), cos(t2), sin(t2), thetaDot1, thetaDot2]
            # Tip height = -cos(t1) - cos(t1 + t2), range [-2, 2], goal > 1
            # Note: next_obs here is raw (via env.last_raw_obs)
            def shaper(obs, reward, next_obs):
                cos1, sin1 = next_obs[0], next_obs[1]
                cos2, sin2 = next_obs[2], next_obs[3]
                cos12 = cos1 * cos2 - sin1 * sin2
                tip_height = -cos1 - cos12
                height_bonus = (tip_height + 2) / 4  # Normalize to [0, 1]
                angular_velocity = abs(next_obs[4]) + abs(next_obs[5])
                return reward + 0.5 * height_bonus + 0.1 * angular_velocity
            return shaper
        return None  # No shaping for other environments

    def collect_episode(self, explore_ratio: float | None = None) -> float:
        """Run one episode, collecting data into buffers.

        Args:
            explore_ratio: probability of random action (epsilon-greedy)

        Returns:
            Total episode reward
        """
        if explore_ratio is None:
            explore_ratio = self.config.policy.explore_ratio

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

                # Epsilon-greedy exploration using the direct policy
                if np.random.random() < explore_ratio:
                    action_np = self.env.sample_random_action()
                elif self.env.is_discrete:
                    action_idx = self.real_trainer.policy.act(obs_t, deterministic=True)
                    action_np = self.env.action_to_onehot(action_idx)
                elif self.sac_trainer is not None:
                    action_np = self.sac_trainer.policy.act(obs_t, deterministic=True)
                else:
                    action_np = self.real_trainer.policy.act(obs_t, deterministic=True)

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
        """Train the policy via dream training (imagined decoded observations)."""
        steps = steps or self.config.policy.train_steps
        return self.dream_augmenter.train(steps)

    def train_policy_dream(self, steps: int = 10) -> dict[str, float]:
        """Train the direct policy on imagined experience."""
        return self.dream_augmenter.train(steps)

    def _update_adaptive_horizon(self):
        """Adapt imagination horizon based on average episode length."""
        cfg = self.config.policy
        if (self.total_episodes % cfg.horizon_update_interval != 0
                or len(self.episode_lengths) < 5):
            return
        avg_len = float(np.mean(list(self.episode_lengths)))
        new_horizon = min(cfg.max_horizon,
                          max(5, int(avg_len * cfg.horizon_ratio)))
        if new_horizon != self.dream_augmenter.horizon:
            self.dream_augmenter.horizon = new_horizon

    def train_policy_real(self) -> tuple[float, dict[str, float]]:
        """Collect and train from real experience.

        Uses SAC for continuous envs, A2C for discrete.
        For pixel envs, uses RSSM encoder + latent policy (Dreamer-style).
        Returns (episode_reward, metrics).
        """
        # Update trust region in trainers
        alpha = self._trust_region_alpha()
        self.real_trainer.trust_region_alpha = alpha
        self.real_trainer.trust_region_ref = self._transfer_ref_policy

        if getattr(self.env, 'pixel_obs', False):
            return self._train_pixel()

        if self.sac_trainer is not None:
            return self._train_sac()

        reward, metrics, episode_data = self.real_trainer.collect_and_train(self.env)

        if episode_data is not None:
            obs, acts, rews, dones = episode_data
            self.replay_buffer.add_episode(obs, acts, rews, dones)
            self.recent_episodes.append(episode_data)
            self.episode_lengths.append(len(rews))
            self.total_steps += len(rews)

        self.episode_rewards.append(reward)
        self.total_episodes += 1
        self._update_adaptive_horizon()
        return reward, metrics

    def _train_pixel(self) -> tuple[float, dict[str, float]]:
        """Pixel training: PPO with CNN and auxiliary state prediction.

        Collects a rollout of 512 steps, then runs PPO updates.
        On-policy approach avoids DQN's Q-value divergence.
        """
        ppo = self.pixel_ppo

        # Collect rollout (may span multiple episodes)
        rollout = ppo.collect_rollout(self.env, n_steps=512)

        # Train PPO on collected rollout
        train_metrics = ppo.train_on_rollout(rollout)

        self.total_steps = ppo.total_steps
        n_eps = train_metrics["n_episodes"]
        self.total_episodes += max(n_eps, 1)
        ep_reward = train_metrics["mean_reward"]
        self.episode_rewards.append(ep_reward)

        metrics = {
            "ppo/pg_loss": train_metrics["pg_loss"],
            "ppo/vf_loss": train_metrics["vf_loss"],
            "ppo/entropy": train_metrics["entropy"],
            "ppo/aux_loss": train_metrics["aux_loss"],
            "ppo/n_episodes": n_eps,
        }

        return ep_reward, metrics

    def _train_sac(self) -> tuple[float, dict[str, float]]:
        """SAC training: collect episode with off-policy updates."""
        reward, metrics, episode_data = self.sac_trainer.collect_and_train(self.env)

        if episode_data is not None:
            obs, acts, rews, dones = episode_data
            self.replay_buffer.add_episode(obs, acts, rews, dones)
            self.recent_episodes.append(episode_data)
            self.episode_lengths.append(len(rews))
            self.total_steps += len(rews)

        self.episode_rewards.append(reward)
        self.total_episodes += 1
        self._update_adaptive_horizon()
        return reward, metrics

    def check_crystallization(self, eval_episodes: int = 5) -> Skill | None:
        """Check if the current policy should be crystallized into a skill.

        Runs evaluation episodes (deterministic) to get true performance.
        Only crystallizes once per environment.
        Returns the new Skill if crystallized, None otherwise.
        """
        if self.total_episodes < self.config.skill.min_episodes:
            return None

        # Don't re-crystallize if we already have a skill for this env+mode
        is_pixel = getattr(self.env, 'pixel_obs', False)
        skill_env_name = f"{self.env.env_name}_pixels" if is_pixel else self.env.env_name
        existing = self.skill_library.list_skills()
        for name in existing:
            skill = self.skill_library.load_skill(name)
            if skill and skill.env_name == skill_env_name:
                return None

        # Run actual evaluation (deterministic policy)
        if getattr(self.env, 'pixel_obs', False):
            eval_reward = self._evaluate_pixel(episodes=eval_episodes)
        elif self.sac_trainer is not None:
            eval_reward = self.sac_trainer.evaluate(self.env, episodes=eval_episodes)
        else:
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
                except Exception as e:
                    print(f"Warning: centroid extraction failed: {e}")

            # Use PPO/DQN network state dict for pixel envs
            if self.pixel_ppo is not None:
                policy_sd = {k: v.cpu() for k, v in self.pixel_ppo.net.state_dict().items()}
            else:
                policy_sd = {k: v.cpu() for k, v in self._active_policy.state_dict().items()}

            skill = Skill(
                name=f"{skill_env_name}_{self.total_episodes}ep",
                env_name=skill_env_name,
                policy_state_dict=policy_sd,
                latent_centroid=centroid,
                performance=eval_reward,
                normalizer_state=self.env.normalizer.state_dict(),
                episodes_trained=self.total_episodes,
            )
            self.skill_library.save_skill(skill)
            return skill

        return None

    def _evaluate_pixel(self, episodes: int = 5) -> float:
        """Evaluate pixel policy (greedy, no exploration)."""
        return self.pixel_ppo.evaluate(self.env, episodes=episodes)

    def try_transfer(self) -> Skill | None:
        """Try to find and load a relevant skill for the current environment.

        Strategy:
        1. First try exact env_name match (guaranteed dimension compatibility)
        2. Fall back to latent-space nearest-neighbor (cross-task transfer)

        After successful transfer, activates trust region (KL penalty) to
        prevent catastrophic forgetting of the transferred knowledge.
        """
        # Pixel envs use different architecture from vector skills
        if self.pixel_ppo is not None:
            return None

        loaded_skill = None

        # 1. Exact env_name match — most reliable transfer
        env_name = self.env.env_name
        for skill in self.skill_library._cache.values():
            if skill.env_name == env_name:
                try:
                    self._active_policy.load_state_dict(
                        {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
                    )
                    if skill.normalizer_state:
                        self.env.normalizer = RunningNormalizer.from_state_dict(
                            skill.normalizer_state
                        )
                    loaded_skill = skill
                    break
                except RuntimeError:
                    continue

        # 2. Latent-space nearest neighbor (cross-task)
        if loaded_skill is None:
            skill = self.skill_selector.select(self.env)
            if skill is not None:
                try:
                    self._active_policy.load_state_dict(
                        {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
                    )
                    if skill.normalizer_state:
                        self.env.normalizer = RunningNormalizer.from_state_dict(
                            skill.normalizer_state
                        )
                    loaded_skill = skill
                except RuntimeError:
                    return None

        # Activate trust region: save frozen reference policy
        if loaded_skill is not None:
            import copy
            self._transfer_ref_policy = copy.deepcopy(self._active_policy)
            self._transfer_ref_policy.eval()
            for p in self._transfer_ref_policy.parameters():
                p.requires_grad_(False)
            self._transfer_episode_start = self.total_episodes

        return loaded_skill

    def _trust_region_alpha(self) -> float:
        """Compute current trust region KL penalty weight (linear decay)."""
        if self._transfer_ref_policy is None:
            return 0.0
        episodes_since = self.total_episodes - self._transfer_episode_start
        max_episodes = self.config.transfer.trust_region_episodes
        if episodes_since >= max_episodes:
            self._transfer_ref_policy = None  # Free memory
            return 0.0
        alpha = self.config.transfer.trust_region_alpha
        return alpha * (1.0 - episodes_since / max_episodes)

    def save(self, path: str):
        """Save full agent state to checkpoint."""
        save_checkpoint(
            path,
            rssm=self.rssm.state_dict(),
            policy=self._active_policy.state_dict(),
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
        # Load direct policy (backward compat: try 'policy' then 'actor_critic')
        policy_sd = ckpt.get("policy", ckpt.get("actor_critic"))
        if policy_sd is not None:
            try:
                self._active_policy.load_state_dict(policy_sd)
            except RuntimeError:
                pass  # Architecture mismatch (old checkpoint), skip
        self.env.normalizer = RunningNormalizer.from_state_dict(ckpt["normalizer"])
        self.episodic_memory = EpisodicMemory.from_state_dict(ckpt["episodic_memory"])
        self.total_episodes = ckpt["total_episodes"]
        self.total_steps = ckpt["total_steps"]
        self.episode_rewards = deque(ckpt["episode_rewards"],
                                     maxlen=self.config.skill.crystallization_window)
