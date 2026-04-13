"""Observation encoder protocol and implementations.

Provides a unified interface for encoding observations (vector or pixel)
into fixed-size embeddings that can be shared by policies and world model.
This enables transfer between pixel and vector observation spaces.
"""

import torch
import torch.nn as nn


class MLPObsEncoder(nn.Module):
    """Encodes vector observations into embeddings."""

    def __init__(self, obs_dim: int, embed_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CNNObsEncoder(nn.Module):
    """Encodes pixel observations (64x64) into embeddings."""

    def __init__(self, channels: int = 3, embed_dim: int = 128, depth: int = 32):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv2d(channels, depth, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(depth, depth * 2, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(depth * 4, depth * 8, 3, stride=2, padding=1),
            nn.ELU(),
        )
        # 64x64 -> 4x4 with depth*8 channels
        self.fc = nn.Linear(depth * 8 * 4 * 4, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.view(-1, self.channels, 64, 64)
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_obs_encoder(obs_dim: int, embed_dim: int = 128,
                       pixel: bool = False, channels: int = 3) -> nn.Module:
    """Factory: create appropriate encoder for the observation type."""
    if pixel:
        return CNNObsEncoder(channels=channels, embed_dim=embed_dim)
    return MLPObsEncoder(obs_dim=obs_dim, embed_dim=embed_dim)
