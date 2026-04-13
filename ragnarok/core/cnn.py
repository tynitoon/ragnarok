"""CNN encoder/decoder for pixel-based observations.

Converts 64x64x3 RGB images to/from 128-dim feature vectors,
compatible with the RSSM encoder interface.

Architecture follows DreamerV2 conventions but smaller:
    Encoder: 4 conv layers (32->64->128->256) stride 2, then linear
    Decoder: linear, then 4 transposed conv layers back to 64x64x3
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Encodes 64x64 RGB images into feature vectors."""

    def __init__(self, channels: int = 3, feature_dim: int = 128,
                 depth: int = 32):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim

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
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc = nn.Linear(depth * 8 * 4 * 4, feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode image observation.

        Args:
            obs: (batch, channels*height*width) flattened or (batch, C, H, W)
        Returns:
            features: (batch, feature_dim)
        """
        if obs.dim() == 2:
            # Flattened: reshape to (batch, C, H, W)
            obs = obs.view(-1, self.channels, 64, 64)
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNDecoder(nn.Module):
    """Decodes latent state back to 64x64 RGB images."""

    def __init__(self, latent_dim: int = 160, channels: int = 3,
                 depth: int = 32):
        super().__init__()
        self.depth = depth

        self.fc = nn.Linear(latent_dim, depth * 8 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(depth, channels, 3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to flattened image.

        Args:
            latent: (batch, latent_dim) — typically concat(h, z)
        Returns:
            obs: (batch, channels*height*width) — flattened 64x64 RGB
        """
        x = self.fc(latent)
        x = x.view(-1, self.depth * 8, 4, 4)
        x = self.deconv(x)  # (batch, 3, 64, 64)
        return x.view(x.size(0), -1)  # Flatten to match obs format
