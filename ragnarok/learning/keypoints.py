"""Unsupervised object KEYPOINTS via conditional cross-frame reconstruction (percept v0.5).

The root cause of v0.1-v0.4's ball failure was object IDENTITY: permutation-free slots partition by
region, so a roaming object has no stable slot. Keypoints fix this STRUCTURALLY: a CNN emits K
heatmaps, one per FIXED output channel; position = spatial soft-argmax. Keypoint k is the same channel
every frame -> stable identity by construction, feed-forward (fast), no recurrence.

Trained unsupervised (Jakab 2018 / Transporter, Kulkarni 2019): reconstruct frame x' from the
APPEARANCE of a different frame x plus the GEOMETRY (keypoints) of x'. Because appearance comes from x
(where the object is elsewhere), the decoder must read the moving object's location from x''s keypoints
-> keypoints lock onto what MOVES (ball, paddles). No labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_softargmax(heat):
    """heat (N,K,h,w) -> positions (N,K,2) in [0,1] via spatial softmax expectation."""
    N, K, h, w = heat.shape
    flat = heat.reshape(N, K, h * w)
    soft = flat.softmax(-1).reshape(N, K, h, w)
    xs = torch.linspace(0, 1, w, device=heat.device)
    ys = torch.linspace(0, 1, h, device=heat.device)
    kx = (soft.sum(2) * xs).sum(-1)                         # (N,K)
    ky = (soft.sum(3) * ys).sum(-1)
    return torch.stack([kx, ky], -1), soft


def gaussian_maps(pos, h, w, sigma=0.08):
    """pos (N,K,2) -> Gaussian heatmaps (N,K,h,w) centred at each keypoint."""
    N, K, _ = pos.shape
    xs = torch.linspace(0, 1, w, device=pos.device)
    ys = torch.linspace(0, 1, h, device=pos.device)
    gx = torch.exp(-((xs[None, None] - pos[..., 0:1]) ** 2) / (2 * sigma ** 2))   # (N,K,w)
    gy = torch.exp(-((ys[None, None] - pos[..., 1:2]) ** 2) / (2 * sigma ** 2))   # (N,K,h)
    return gy[..., None] * gx[:, :, None, :]               # (N,K,h,w)


class KeypointNet(nn.Module):
    def __init__(self, K=4, img=48, feat=64, sigma=0.08):
        super().__init__()
        self.K, self.img, self.sigma = K, img, sigma
        self.appear = nn.Sequential(                        # appearance encoder Phi(x)
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),           # 48->24
            nn.Conv2d(32, feat, 5, 2, 2), nn.ReLU())        # 24->12
        self.keynet = nn.Sequential(                        # keypoint encoder Psi(x')
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),           # 48->24
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),          # 24->12
            nn.Conv2d(64, K, 1))                            # K heatmaps (one per channel)
        self.decode = nn.Sequential(                        # decoder(Phi(x) + geometry(x')) -> x'
            nn.Conv2d(feat + K, 64, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 12->24
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # 24->48
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1))

    def keypoints(self, x):
        """x (N,3,H,W) -> keypoint positions (N,K,2) in [0,1]."""
        heat = self.keynet(x)
        pos, _ = spatial_softargmax(heat)
        return pos

    def forward(self, x_src, x_tgt):
        """Reconstruct x_tgt from appearance(x_src) + geometry(keypoints of x_tgt)."""
        phi = self.appear(x_src)                            # (N,feat,12,12)
        heat = self.keynet(x_tgt)
        pos, _ = spatial_softargmax(heat)                   # (N,K,2)
        h, w = phi.shape[-2:]
        geom = gaussian_maps(pos, h, w, self.sigma)         # (N,K,12,12)
        recon = self.decode(torch.cat([phi, geom], 1)).sigmoid()
        return recon, pos
