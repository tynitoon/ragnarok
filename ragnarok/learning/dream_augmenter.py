"""Dream augmenter: generate synthetic experience for the direct policy.

Bridges the gap between the world model (latent space) and the direct
policy (raw observations). Uses the RSSM to:
1. Sample starting states from real experience
2. Imagine trajectories using the direct policy (via decoded observations)
3. Decode imagined states back to observation space
4. Train the direct policy on this synthetic data

This is the unified dream training path - replaces the separate latent-space
DreamTrainer by operating through decoded observations. The policy trained
here is the same one used for real environment interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ragnarok.core.rssm import RSSM
from ragnarok.learning.real_experience import DirectPolicyNet, ContinuousPolicyNet
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE


from ragnarok.learning.advantages import compute_lambda_returns


class DreamAugmenter:
    """Generates synthetic training data by dreaming in the world model.

    Trains the same DirectPolicyNet/ContinuousPolicyNet/SACPolicy used for
    real experience, but on imagined data decoded from RSSM latent space.
    Uses lambda-returns for more accurate value estimation.
    """

    def __init__(self, rssm: RSSM, policy, replay_buffer: ReplayBuffer,
                 horizon: int = 15, dream_batch: int = 64,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 entropy_coeff: float = 0.01,
                 lr: float = 1e-4, grad_clip: float = 0.5,
                 disagreement_weight: float = 0.1,
                 optimizer=None, dream_grad_scale: float = 0.3):
        self.rssm = rssm
        self.policy = policy
        self.buffer = replay_buffer
        self.horizon = horizon
        self.dream_batch = dream_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.discrete = isinstance(policy, DirectPolicyNet)
        self.disagreement_weight = disagreement_weight

        # Single optimizer: share the real trainer's optimizer for unified
        # Adam moments. Dream gradients are scaled down to avoid overwhelming
        # real experience signal.
        if optimizer is not None:
            self.optimizer = optimizer
            self.dream_grad_scale = dream_grad_scale
        else:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
            self.dream_grad_scale = 1.0

    @torch.no_grad()
    def _get_start_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample initial latent states by encoding real observations."""
        seq_len = min(10, self.buffer.max_episode_length)
        seq_len = max(seq_len, 2)
        obs, actions, _, _ = self.buffer.sample_sequences(self.dream_batch, seq_len)
        obs_t = torch.tensor(obs, device=DEVICE)
        act_t = torch.tensor(actions, device=DEVICE)

        outputs = self.rssm.observe(obs_t, act_t)
        batch_size, time_steps = outputs["h"].shape[:2]
        # Pick a random timestep per batch element
        t_idx = torch.randint(0, time_steps, (batch_size,), device=DEVICE)
        batch_idx = torch.arange(batch_size, device=DEVICE)
        h0 = outputs["h"][batch_idx, t_idx]
        z0 = outputs["z"][batch_idx, t_idx]
        return h0, z0

    def dream_and_train(self) -> dict[str, float]:
        """Generate one batch of synthetic experience and train on it.

        Returns training metrics.
        """
        if self.buffer.num_episodes < 5:
            return {}

        # 1. Get starting states from real experience
        h, z = self._get_start_states()

        # Reset ensemble states for this dream batch
        if self.rssm.ensemble is not None:
            self._ensemble_states = self.rssm.ensemble.initial_state(
                h.shape[0], h.device)

        # 2. Imagine trajectory, using decoded observations for the direct policy
        log_probs_list = []
        values_list = []
        entropies_list = []
        rewards_list = []
        continues_list = []

        for step in range(self.horizon):
            # Decode latent state to observation space
            with torch.no_grad():
                decoded_obs = self.rssm.decoder(torch.cat([h, z], dim=-1))

            if self.discrete:
                # Discrete: Categorical policy
                logits, value = self.policy(decoded_obs)
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                log_probs_list.append(dist.log_prob(action_idx))
                values_list.append(value)
                entropies_list.append(dist.entropy())
                # Convert to one-hot for RSSM
                action_for_rssm = F.one_hot(action_idx, num_classes=self.rssm.action_dim).float()
            else:
                # Continuous: Gaussian policy
                mean, logstd, value = self.policy(decoded_obs)
                dist = torch.distributions.Normal(mean, logstd.exp())
                raw_action = dist.rsample()
                lp = dist.log_prob(raw_action).sum(dim=-1)
                squashed = torch.tanh(raw_action)
                lp -= torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)
                log_probs_list.append(lp)
                values_list.append(value)
                entropies_list.append(self.policy.entropy(decoded_obs))
                # Rescale for RSSM (use raw env-scale action)
                action_for_rssm = self.policy._rescale(squashed)

            # Step world model forward
            with torch.no_grad():
                h = self.rssm.core.step(h, z, action_for_rssm)
                prior_mean, prior_logstd = self.rssm.core.forward_prior(h)
                z = self.rssm.core.sample(prior_mean, prior_logstd)

                reward = self.rssm.reward_predictor(h, z)
                continue_logit = self.rssm.continue_predictor(h, z)
                continue_prob = torch.sigmoid(continue_logit)

                # Ensemble disagreement penalty: penalize rewards in uncertain regions
                if self.rssm.ensemble is not None and self.disagreement_weight > 0:
                    # Step ensemble cores
                    ens_hs = self.rssm.ensemble.step_all(
                        self._ensemble_states, action_for_rssm)
                    disagr = self.rssm.ensemble.disagreement(ens_hs)
                    reward = reward - self.disagreement_weight * disagr
                    # Update ensemble states
                    self._ensemble_states = [(eh, z) for eh in ens_hs]

            rewards_list.append(reward)
            continues_list.append(continue_prob)

        # Get bootstrap value for last state
        with torch.no_grad():
            last_decoded = self.rssm.decoder(torch.cat([h, z], dim=-1))
            if self.discrete:
                _, last_value = self.policy(last_decoded)
            else:
                _, _, last_value = self.policy(last_decoded)

        # 3. Lambda returns (better quality than simple discounted returns)
        rewards = torch.stack(rewards_list, dim=1)  # (batch, horizon)
        continues = torch.stack(continues_list, dim=1)  # (batch, horizon)
        values_t = torch.stack(values_list, dim=1)  # (batch, horizon)

        # Build full value sequence for lambda returns: (batch, horizon+1)
        all_values = torch.cat([values_t, last_value.unsqueeze(1)], dim=1)
        returns = compute_lambda_returns(
            rewards, all_values.detach(), continues,
            self.gamma, self.gae_lambda,
        )

        # 4. Policy update with clipped advantages
        log_probs = torch.stack(log_probs_list, dim=1).reshape(-1)
        values_flat = values_t.reshape(-1)
        entropies = torch.stack(entropies_list, dim=1).reshape(-1)
        returns_flat = returns.reshape(-1)
        advantages = (returns_flat - values_flat.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values_flat, returns_flat.detach())
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss
        loss = loss * self.dream_grad_scale  # Scale dream gradients down

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "dream_aug/actor_loss": actor_loss.item(),
            "dream_aug/critic_loss": critic_loss.item(),
            "dream_aug/entropy": entropies.mean().item(),
            "dream_aug/mean_reward": rewards.mean().item(),
            "dream_aug/mean_continue": continues.mean().item(),
        }

    def train(self, steps: int) -> dict[str, float]:
        """Run multiple dream augmentation steps."""
        if self.buffer.num_episodes < 5:
            return {}

        totals: dict[str, float] = {}
        count = 0

        for _ in range(steps):
            metrics = self.dream_and_train()
            if metrics:
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                count += 1

        if count == 0:
            return {}
        return {k: v / count for k, v in totals.items()}
