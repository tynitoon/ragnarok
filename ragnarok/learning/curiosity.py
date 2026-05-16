"""Intrinsic Curiosity Module: exploration via forward prediction error.

The agent is curious about states it cannot predict well. A small
forward model learns to predict next observations from (obs, action).
Prediction error = intrinsic reward, encouraging exploration of novel
states.

As the predictor improves on visited states, curiosity naturally
shifts to unvisited regions. This solves exploration-hard environments
like MountainCar where extrinsic reward is uniformly negative.

Reference: Pathak et al., "Curiosity-driven Exploration by
Self-Supervised Prediction" (ICM, 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE, mark_step


class ForwardPredictor(nn.Module):
    """Predicts next observation from (obs, action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class CuriosityModule:
    """Forward-prediction curiosity for exploration.

    Computes intrinsic_reward = beta * normalized_prediction_error
    where the predictor learns to predict next_obs from (obs, action).

    Novel states -> high prediction error -> high intrinsic reward.
    Familiar states -> low error -> exploration moves elsewhere.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden: int = 64, lr: float = 1e-3,
                 beta: float = 0.1, grad_clip: float = 1.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.beta = beta
        self.grad_clip = grad_clip

        self.predictor = ForwardPredictor(obs_dim, action_dim, hidden).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running normalization for intrinsic rewards (Welford's online algorithm)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def compute_intrinsic_rewards(self, obs_seq: np.ndarray,
                                  action_seq: np.ndarray,
                                  next_obs_seq: np.ndarray) -> np.ndarray:
        """Compute intrinsic rewards for a sequence of transitions.

        Args:
            obs_seq: (T, obs_dim)
            action_seq: (T, action_dim)
            next_obs_seq: (T, obs_dim)

        Returns:
            intrinsic_rewards: (T,) - beta-scaled normalized prediction errors
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE)
            act_t = torch.tensor(action_seq, dtype=torch.float32, device=DEVICE)
            next_t = torch.tensor(next_obs_seq, dtype=torch.float32, device=DEVICE)

            pred = self.predictor(obs_t, act_t)
            # Per-step MSE (not reduced across batch)
            errors = (pred - next_t).pow(2).mean(dim=-1).cpu().numpy()

        # Update running stats and normalize
        for e in errors:
            self._update_stats(float(e))

        std = max(self._reward_var ** 0.5, 1e-8)
        normalized = (errors - self._reward_mean) / std
        # Clip to avoid extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        # ReLU: only positive curiosity (novel = bonus, familiar = no penalty)
        normalized = np.maximum(normalized, 0.0)

        return self.beta * normalized

    def train_on_transitions(self, obs_seq: np.ndarray,
                             action_seq: np.ndarray,
                             next_obs_seq: np.ndarray) -> float:
        """Update predictor on observed transitions.

        Returns mean prediction loss.
        """
        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE)
        act_t = torch.tensor(action_seq, dtype=torch.float32, device=DEVICE)
        next_t = torch.tensor(next_obs_seq, dtype=torch.float32, device=DEVICE)

        pred = self.predictor(obs_t, act_t)
        loss = F.mse_loss(pred, next_t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def _update_stats(self, value: float):
        """Online mean/variance (Welford's algorithm)."""
        self._reward_count += 1
        delta = value - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = value - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count

    @property
    def params_count(self) -> int:
        return sum(p.numel() for p in self.predictor.parameters())

    def state_dict(self) -> dict:
        return {
            "predictor": self.predictor.state_dict(),
            "reward_mean": self._reward_mean,
            "reward_var": self._reward_var,
            "reward_count": self._reward_count,
        }

    def load_state_dict(self, state: dict):
        self.predictor.load_state_dict(state["predictor"])
        self._reward_mean = state["reward_mean"]
        self._reward_var = state["reward_var"]
        self._reward_count = state["reward_count"]


class LatentCuriosityModule:
    """Latent-space curiosity via RSSM Bayesian surprise.

    Uses KL(posterior || prior) from the RSSM as intrinsic reward.
    Novel states -> high KL (posterior deviates from prior prediction).
    Familiar states -> low KL (world model predicts accurately).

    Zero extra parameters: leverages the already-trained RSSM.
    Falls back to ForwardPredictor when RSSM is undertrained.
    """

    def __init__(self, rssm, beta: float = 0.1,
                 min_rssm_episodes: int = 20):
        self.rssm = rssm
        self.beta = beta
        self.min_rssm_episodes = min_rssm_episodes
        self._episodes_seen = 0

        # RSSM state tracking (reset each episode)
        self._h = None
        self._z = None
        self._prev_action = None

        # Running normalization for KL rewards (Welford's)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    @property
    def rssm_ready(self) -> bool:
        """Whether RSSM has seen enough data for meaningful KL."""
        return self._episodes_seen >= self.min_rssm_episodes

    def reset_episode(self, action_dim: int):
        """Reset RSSM state for a new episode."""
        self._h, self._z = self.rssm.initial_state(1, DEVICE)
        self._prev_action = torch.zeros(1, action_dim, device=DEVICE)
        self._episodes_seen += 1

    def compute_step_kl(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute KL surprise for a single step, updating RSSM state.

        Args:
            obs: current observation (obs_dim,)
            action: action taken (action_dim,)

        Returns:
            Normalized, beta-scaled KL divergence (intrinsic reward).
        """
        if self._h is None:
            return 0.0

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Step the GRU with previous state
            h_new = self.rssm.core.step(self._h, self._z, self._prev_action)

            # Prior: what the model predicts
            prior_mean, prior_logstd = self.rssm.core.forward_prior(h_new)

            # Posterior: what actually happened (with observation)
            features = self.rssm.encoder(obs_t)
            post_mean, post_logstd = self.rssm.core.forward_posterior(h_new, features)

            # KL(posterior || prior) per dimension, summed
            prior = torch.distributions.Normal(prior_mean, prior_logstd.exp())
            posterior = torch.distributions.Normal(post_mean, post_logstd.exp())
            kl = torch.distributions.kl_divergence(posterior, prior)
            kl_sum = kl.sum(dim=-1).item()

            # Sample from posterior and update state
            self._z = self.rssm.core.sample(post_mean, post_logstd)
            self._h = h_new
            self._prev_action = action_t

        # Normalize
        self._update_stats(kl_sum)
        std = max(self._reward_var ** 0.5, 1e-8)
        normalized = (kl_sum - self._reward_mean) / std
        normalized = max(min(normalized, 5.0), 0.0)  # ReLU + clip

        return self.beta * normalized

    def compute_batch_kl(self, obs_seq: np.ndarray,
                         action_seq: np.ndarray) -> np.ndarray:
        """Compute KL surprise for a full episode (batch mode).

        Args:
            obs_seq: (T, obs_dim)
            action_seq: (T, action_dim)

        Returns:
            intrinsic_rewards: (T,) beta-scaled normalized KL values

        XLA: RSSM.observe() unrolls its GRU per timestep, so calling it on a
        whole variable-length episode recompiles the graph for every distinct
        episode length (the bug that saturated the TPU host CPU). The episode
        is instead processed in fixed-length chunks — observe() then sees one
        shape, compiled once. The (h, z) state and the boundary action are
        threaded across chunks, so the per-step KL is identical to a single
        full-sequence observe(). The last chunk is zero-padded and the padding
        sliced back off. Harmless on CUDA/CPU.
        """
        T = obs_seq.shape[0]
        if T < 2:
            return np.zeros(T, dtype=np.float32)

        CHUNK = 64
        with torch.no_grad():
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE)
            act_t = torch.tensor(action_seq, dtype=torch.float32, device=DEVICE)

            kl_parts: list[np.ndarray] = []
            state = None        # threaded (h, z) across chunk boundaries
            prev_action = None  # action preceding the chunk's first obs
            for start in range(0, T, CHUNK):
                n = min(CHUNK, T - start)
                obs_c = obs_t[start:start + n]
                act_c = act_t[start:start + n]
                if n < CHUNK:
                    pad = CHUNK - n
                    obs_c = torch.cat(
                        [obs_c, obs_c.new_zeros((pad,) + obs_c.shape[1:])], 0)
                    act_c = torch.cat(
                        [act_c, act_c.new_zeros((pad,) + act_c.shape[1:])], 0)

                outputs = self.rssm.observe(
                    obs_c.unsqueeze(0), act_c.unsqueeze(0),
                    init_state=state, init_action=prev_action)

                prior = torch.distributions.Normal(
                    outputs["prior_mean"], outputs["prior_logstd"].exp())
                posterior = torch.distributions.Normal(
                    outputs["post_mean"], outputs["post_logstd"].exp())
                kl = torch.distributions.kl_divergence(posterior, prior)
                kl_c = kl.sum(dim=-1).squeeze(0)              # (CHUNK,)
                kl_parts.append(kl_c[:n].cpu().numpy())       # drop padding

                # Thread GRU state + boundary action into the next chunk.
                state = (outputs["h"][:, n - 1], outputs["z"][:, n - 1])
                prev_action = act_c[n - 1].unsqueeze(0)
                mark_step()  # XLA: cut the graph at the chunk boundary

            kl_per_step = np.concatenate(kl_parts)  # (T,)

        # Normalize with running stats
        for k in kl_per_step:
            self._update_stats(float(k))

        std = max(self._reward_var ** 0.5, 1e-8)
        normalized = (kl_per_step - self._reward_mean) / std
        normalized = np.clip(normalized, 0.0, 5.0)  # ReLU + clip

        return (self.beta * normalized).astype(np.float32)

    def _update_stats(self, value: float):
        """Welford's online mean/variance."""
        self._reward_count += 1
        delta = value - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = value - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count

    @property
    def params_count(self) -> int:
        return 0  # No extra parameters

    def state_dict(self) -> dict:
        return {
            "reward_mean": self._reward_mean,
            "reward_var": self._reward_var,
            "reward_count": self._reward_count,
            "episodes_seen": self._episodes_seen,
        }

    def load_state_dict(self, state: dict):
        self._reward_mean = state["reward_mean"]
        self._reward_var = state["reward_var"]
        self._reward_count = state["reward_count"]
        self._episodes_seen = state.get("episodes_seen", 0)
