"""From-scratch OBJECT PERCEPTION core (the first lock of the developmental model).

A slot-attention autoencoder: the frame is explained by K SLOTS that COMPETE to bind to
regions (objects), trained ONLY by reconstruction (no labels). The inductive bias (slots
compete via attention; each decodes an image+mask; masks compose the frame) makes slots
bind to objects. This is our own clean implementation; slot-attention is the established
way to do unsupervised object discovery, used here as the perception brick of the larger
from-scratch agent. Validated separately: do slots track the real objects (ball/paddle)?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden=128, eps=1e-8):
        super().__init__()
        self.num_slots, self.iters, self.eps, self.scale = num_slots, iters, eps, dim ** -0.5
        self.mu = nn.Parameter(torch.randn(1, 1, dim))
        self.logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.norm_in = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs):                                   # (N, P, D)
        N, P, D = inputs.shape
        slots = self.mu + self.logsigma.exp() * torch.randn(N, self.num_slots, D, device=inputs.device)
        inputs = self.norm_in(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        attn_vis = None
        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))                # (N,S,D)
            dots = torch.einsum("nsd,npd->nsp", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps                # COMPETE over slots
            attn_vis = attn
            w = attn / attn.sum(dim=-1, keepdim=True)            # weighted mean over pixels
            updates = torch.einsum("nsp,npd->nsd", w, v)         # (N,S,D)
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(N, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn_vis                                  # attn_vis (N,S,P)


class Encoder(nn.Module):
    def __init__(self, dim, img):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.ReLU(),               # img -> img/2
            nn.Conv2d(dim, dim, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 2), nn.ReLU())
        self.fh = img // 2
        self.pos = nn.Parameter(torch.randn(1, self.fh * self.fh, dim) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        f = self.conv(x)                                        # (N,D,fh,fh)
        N, D, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(N, h * w, D) + self.pos
        return self.mlp(self.norm(f))


class Decoder(nn.Module):
    """Spatial-broadcast decoder: each slot -> (image, mask) over the full frame; masks
    compose. Returns recon + per-slot masks."""
    def __init__(self, dim, img):
        super().__init__()
        self.img = img
        self.pos = nn.Parameter(torch.randn(1, dim, img, img) * 0.02)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(dim, dim, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(dim, 4, 3, 1, 1))

    def forward(self, slots):                                   # (N,S,D)
        N, S, D = slots.shape
        x = slots.reshape(N * S, D, 1, 1).expand(-1, -1, self.img, self.img) + self.pos
        x = self.conv(x).reshape(N, S, 4, self.img, self.img)
        rgb, alpha = x[:, :, :3], x[:, :, 3:4]
        masks = alpha.softmax(dim=1)                            # (N,S,1,H,W) compete over slots
        recon = (rgb * masks).sum(1)                            # (N,3,H,W)
        return recon, masks


class SlotAE(nn.Module):
    def __init__(self, num_slots=5, dim=48, img=48, iters=3):
        super().__init__()
        self.enc = Encoder(dim, img)
        self.slot = SlotAttention(num_slots, dim, iters)
        self.dec = Decoder(dim, img)
        self.num_slots, self.img = num_slots, img

    def forward(self, x):
        slots, attn = self.slot(self.enc(x))
        recon, masks = self.dec(slots)
        return recon, masks, slots, attn

    @torch.no_grad()
    def mask_centroids(self, masks):
        """(N,S,1,H,W) -> per-slot centroid (N,S,2) in [0,1] (x,y), + mass (N,S)."""
        N, S, _, H, W = masks.shape
        m = masks.squeeze(2)                                    # (N,S,H,W)
        xs = torch.arange(W, device=DEVICE).float()
        ys = torch.arange(H, device=DEVICE).float()
        mass = m.sum((-1, -2)).clamp(min=1e-6)
        cx = (m.sum(-2) * xs).sum(-1) / mass / (W - 1)
        cy = (m.sum(-1) * ys).sum(-1) / mass / (H - 1)
        return torch.stack([cx, cy], -1), mass
