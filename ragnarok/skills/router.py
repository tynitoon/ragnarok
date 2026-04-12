"""Skill router: dynamically selects which skill to execute.

Provides two routing strategies:
1. Centroid-based: L2 distance between RSSM latent state and skill centroids
2. Learned: small MLP trained to map observations to skill selection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE


class CentroidRouter:
    """Routes to the nearest skill based on latent centroid distance.

    Uses the RSSM to encode observations into latent space, then
    finds the skill whose centroid is closest.
    """

    def __init__(self, skill_centroids: dict[str, np.ndarray],
                 temperature: float = 1.0):
        """
        Args:
            skill_centroids: {skill_name: centroid_vector}
            temperature: softmax temperature (lower = more decisive)
        """
        self.skill_names = list(skill_centroids.keys())
        self.centroids = np.stack([skill_centroids[n] for n in self.skill_names])
        self.temperature = temperature

    def select(self, latent_state: np.ndarray) -> str:
        """Select the best skill given a latent state vector."""
        distances = np.sqrt(np.sum((self.centroids - latent_state) ** 2, axis=-1))
        return self.skill_names[np.argmin(distances)]

    def select_soft(self, latent_state: np.ndarray) -> dict[str, float]:
        """Return a probability distribution over skills."""
        distances = np.sqrt(np.sum((self.centroids - latent_state) ** 2, axis=-1))
        neg_dist = -distances / self.temperature
        exp_d = np.exp(neg_dist - neg_dist.max())
        probs = exp_d / exp_d.sum()
        return {name: float(p) for name, p in zip(self.skill_names, probs)}


class LearnedRouter(nn.Module):
    """Small MLP that learns to route observations to skills.

    Trained via supervision (which skill performed best) or
    reinforcement (which skill got highest reward).
    """

    def __init__(self, obs_dim: int, num_skills: int, hidden: int = 64):
        super().__init__()
        self.num_skills = num_skills
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_skills),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits over skills."""
        return self.net(obs)

    def select(self, obs: torch.Tensor) -> int:
        """Select best skill index."""
        with torch.no_grad():
            logits = self.forward(obs)
            return logits.argmax(dim=-1).item()

    def train_step(self, obs: torch.Tensor, best_skill_idx: torch.Tensor) -> float:
        """Train the router with supervised signal (which skill performed best).

        Args:
            obs: (batch, obs_dim)
            best_skill_idx: (batch,) long tensor — index of best skill
        Returns:
            loss value
        """
        logits = self.forward(obs)
        loss = F.cross_entropy(logits, best_skill_idx)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
