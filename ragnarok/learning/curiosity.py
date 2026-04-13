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

from ragnarok.infrastructure.device import DEVICE


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
