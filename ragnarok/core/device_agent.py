"""Accelerator-resident agent — Phase 2 Stage 5.4.

DeviceAgent bundles the device-path components for one task and orchestrates
training: collect a fixed-shape rollout, then train the real policy
(PPO / SAC), the RSSM world model, and the latent policy on it — all batched
on the accelerator, with no host env-loop and no host replay sampling.

Built additively: DeviceAgent does NOT touch RagnarokAgent (the calibrated
gym-path agent) — the two share no code path. snapshot()/load_snapshot()
carry the env-agnostic transferable subset (the RSSM core + the latent-policy
trunk) for cross-task transfer.
"""

import dataclasses

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.rollout import collect_rollout, device_evaluate
from ragnarok.learning.real_experience import RealExperienceTrainer
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.latent_policy import LatentPolicyTrainer
from ragnarok.learning.curiosity import DeviceLatentCuriosity
from ragnarok.environments.device_env import DeviceRunningNormalizer


class DeviceAgent:
    """Full device-path agent for one task — discrete (PPO) or continuous (SAC).

    Args:
        env_cls: a DeviceVec* class (DeviceVecCartPole / ...MountainCar...).
        num_envs, horizon: rollout dimensions — Python constants, so the whole
            collect+train graph is fixed-shape (XLA-clean).
        sac_updates: SAC updates per rollout (continuous tasks only).
    """

    def __init__(self, env_cls, num_envs: int = 256, horizon: int = 128,
                 sac_updates: int = 1024):
        self.env_cls = env_cls
        self.num_envs = num_envs
        self.horizon = horizon
        self.sac_updates = sac_updates
        self.obs_dim = env_cls.obs_dim
        self.action_dim = env_cls.action_dim
        self.discrete = env_cls.is_discrete

        self.env = env_cls(num_envs)
        self.normalizer = DeviceRunningNormalizer(self.obs_dim)

        # World model — trained every iteration, shared with the latent policy
        # (latents) and, for continuous tasks, with curiosity.
        self.rssm = RSSM(self.obs_dim, self.action_dim).to(DEVICE)
        self.wm = WorldModelTrainer(self.rssm, ReplayBuffer())

        # Real policy: PPO for discrete, SAC for continuous.
        bounds = None
        if not self.discrete:
            bounds = (np.full(self.action_dim, -1.0, dtype=np.float32),
                      np.full(self.action_dim, 1.0, dtype=np.float32))
        if self.discrete:
            self.real = RealExperienceTrainer(
                self.obs_dim, self.action_dim, discrete=True)
            self.curiosity = None
        else:
            self.real = SACTrainer(
                self.obs_dim, self.action_dim,
                action_low=bounds[0], action_high=bounds[1],
                warmup_steps=num_envs * horizon,
                buffer=DeviceSACBuffer(capacity=200_000))
            # MountainCar's reward is too sparse for bare SAC — curiosity
            # supplies the exploration drive (Stage 5.1).
            self.curiosity = DeviceLatentCuriosity(self.rssm)

        # Latent policy — the cross-task transfer vehicle. Its shared trunk +
        # critic are env-agnostic and travel in snapshot().
        self.latent = LatentPolicyTrainer(
            self.rssm.state_dim, self.action_dim, discrete=self.discrete,
            action_low=None if bounds is None else bounds[0],
            action_high=None if bounds is None else bounds[1])

    def train_iteration(self) -> dict:
        """One collect-and-train cycle over every device component."""
        batch = collect_rollout(self.env, self.real.device_policy_fn,
                                self.horizon, normalizer=self.normalizer)
        if self.discrete:
            metrics = self.real.train_on_rollout(batch)
        else:
            # Curiosity augments the SAC reward only (it drives MountainCar
            # exploration); the world model and latent policy keep training
            # on the raw env reward.
            intrinsic = self.curiosity.intrinsic_reward(batch)
            sac_batch = dataclasses.replace(
                batch, rewards=batch.rewards + intrinsic)
            metrics = self.real.train_on_rollout(
                sac_batch, n_updates=self.sac_updates)
        metrics.update(self.wm.train_world_model_on_rollout(batch))
        metrics.update(self.latent.train_on_rollout(batch, self.rssm))
        self.normalizer.update(batch.raw_obs.reshape(-1, self.obs_dim))
        return metrics

    def _greedy_act_fn(self):
        """Deterministic action function over the real policy, for evaluation."""
        policy = self.real.policy
        if self.discrete:
            def act_fn(obs):
                logits, _ = policy(obs)
                return logits.argmax(dim=-1)
        else:
            def act_fn(obs):
                mean, _ = policy.forward(obs)
                return policy._rescale(torch.tanh(mean))
        return act_fn

    @torch.no_grad()
    def evaluate(self, steps: int = 500, n_envs: int = 256) -> float:
        """Greedy mean completed-episode return on a fresh eval env."""
        eval_env = self.env_cls(n_envs)
        return device_evaluate(eval_env, self._greedy_act_fn(), steps,
                               normalizer=self.normalizer)

    def snapshot(self) -> dict:
        """Env-agnostic transferable subset — RSSM core + latent-policy trunk.

        Returns CPU tensors, ready to load into a different-dimensional
        target task (the cross-dim transfer payload).
        """
        return {
            "rssm_core": {k: v.detach().cpu() for k, v in
                          self.rssm.transferable_state_dict().items()},
            "latent_trunk": {k: v.detach().cpu() for k, v in
                             self.latent.policy.get_trunk_state_dict().items()},
        }

    def load_snapshot(self, snap: dict) -> None:
        """Load a snapshot from another (possibly different-dim) task."""
        self.rssm.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in snap["rssm_core"].items()})
        self.latent.policy.load_trunk_state_dict(
            {k: v.to(DEVICE) for k, v in snap["latent_trunk"].items()})
