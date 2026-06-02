"""Temporal object-permanence tracker (percept v0.4) — the decisive perception attempt.

Diagnosis from v0.1-v0.3: feed-forward reconstruction binds POSITION-STABLE objects (paddles) but
hands the small fast ROAMING ball between slots frame-to-frame (slots partition by region; no object
identity/permanence). Fix: process VIDEO and CARRY slot state across frames, so the slot that binds
an object keeps tracking it. Three ingredients, each of which a prior attempt had only in part:
  - SlotAttention (iterative competition)  -> object IDENTITY (slots specialise, don't all grab one)
  - temporal carry of slot state           -> object PERMANENCE (the ball-slot persists & tracks)
  - explicit sprite renderer (pos bottleneck) -> object LOCALISATION (commit a slot to one place)
Still LEARNED, no labels: trained self-supervised by reconstructing each video frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.learning.slots import Encoder, SlotAttention


class RecurrentSpriteTracker(nn.Module):
    def __init__(self, num_slots=4, dim=64, img=48, patch=10, iters_first=3, iters_step=2):
        super().__init__()
        self.K, self.dim, self.img, self.patch = num_slots, dim, img, patch
        self.scale = patch / img
        self.iters_first, self.iters_step = iters_first, iters_step
        self.enc = Encoder(dim, img)                              # frame -> (N, P, dim) tokens
        self.slot = SlotAttention(num_slots, dim, iters=iters_step)
        self.predict = nn.GRUCell(dim, dim)                      # transition prior slots_{t}|slots_{t-1}
        self.pos = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.pres = nn.Linear(dim, 1)
        self.app = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3 * patch * patch))

    def render(self, slots):
        """slots (N,K,dim) -> recon (N,3,H,W), positions (N,K,2), presence (N,K)."""
        N, K, D = slots.shape
        pos = self.pos(slots).sigmoid()                         # (N,K,2) in [0,1]
        pres = self.pres(slots).squeeze(-1).sigmoid()          # (N,K)
        app = self.app(slots).reshape(N * K, 3, self.patch, self.patch).sigmoid()
        cx = (2 * pos[..., 0] - 1).reshape(-1)
        cy = (2 * pos[..., 1] - 1).reshape(-1)
        s = self.scale
        z = torch.zeros_like(cx)
        theta = torch.stack([torch.stack([torch.full_like(cx, 1 / s), z, -cx / s], -1),
                             torch.stack([z, torch.full_like(cx, 1 / s), -cy / s], -1)], 1)
        grid = F.affine_grid(theta, (N * K, 3, self.img, self.img), align_corners=False)
        placed = F.grid_sample(app, grid, align_corners=False, padding_mode="zeros")
        placed = placed.reshape(N, K, 3, self.img, self.img)
        recon = (placed * pres[:, :, None, None, None]).sum(1).clamp(0, 1)
        return recon, pos, pres

    def forward(self, frames):
        """frames (T,N,3,H,W) -> recon (T,N,3,H,W), positions (T,N,K,2), presence (T,N,K).

        Slots are carried across time: at t=0 sampled from the prior; at t>0 the previous slots are
        passed through a learned transition (GRU) then refined by attention on the current frame.
        """
        T, N = frames.shape[0], frames.shape[1]
        recons, positions, presences = [], [], []
        slots = None
        for t in range(T):
            tok = self.enc(frames[t])                          # (N,P,dim)
            if slots is None:
                slots, _ = self.slot(tok)                      # init from prior, iters_first
            else:
                prior = self.predict(slots.reshape(-1, self.dim)).reshape(N, self.K, self.dim)
                slots, _ = self.slot(tok, init_slots=prior)    # carry -> refine on frame t
            recon, pos, pres = self.render(slots)
            recons.append(recon); positions.append(pos); presences.append(pres)
        return (torch.stack(recons), torch.stack(positions), torch.stack(presences))
