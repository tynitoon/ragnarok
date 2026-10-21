"""The live policy: what to press, from an egocentric view of what is not background.

Input (agent-side only): a W x W window centred on me of the foreground mask (pixels that differ
from the agent's own background estimate) for the current and the previous step, downsampled by 2,
plus my normalised position. No colours, no sprite identities, no RAM, no score value.
Trained only by self-imitation: (i) the explorer's own demonstration, (ii) the policy's own
rollouts that produced a score event (hindsight on achieved events, commanded goal = score-up).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 64


def ego_view(obs, bg, me, prev_fg=None):
    """prev_fg: (prev ego crop, prev global view) or None. Returns (img[4,32,32], pos, (crop, g))."""
    fg = (obs != bg).any(-1).astype(np.float32)
    H, W = fg.shape
    x, y = (W // 2, H // 2) if me is None else (int(me[0]), int(me[1]))
    x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    pad = WIN // 2
    big = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
    big[pad:pad + H, pad:pad + W] = fg
    crop = big[y:y + WIN, x:x + WIN]
    crop = crop.reshape(WIN // 2, 2, WIN // 2, 2).max((1, 3))
    pos = np.array([x / W, y / H], np.float32)
    # global coarse view: the whole screen's foreground pooled to the same 32x32 grid
    gh, gw = WIN // 2, WIN // 2
    g = fg[: H - H % gh, : W - W % gw].reshape(gh, H // gh, gw, W // gw).max((1, 3))
    pc, pg = prev_fg if prev_fg is not None else (crop, g)
    return np.stack([crop, pc, g, pg]), pos, (crop, g)


class PolicyNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 * 4 * 4 + 2, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, img, pos):
        z = self.conv(img).flatten(1)
        return self.head(torch.cat([z, pos], 1))


class Policy:
    def __init__(self, n_actions, device="cuda" if torch.cuda.is_available() else "cpu", lr=3e-4):
        self.net = PolicyNet(n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device
        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, img, pos, rng, temp=1.0, eps=0.02, mask=None):
        ok = np.ones(self.n_actions, bool) if mask is None else np.asarray(mask, bool)
        if rng.random() < eps:
            return int(rng.choice(np.nonzero(ok)[0]))
        lo = self.net(torch.from_numpy(img)[None].to(self.device), torch.from_numpy(pos)[None].to(self.device))
        lo = lo[0].masked_fill(torch.from_numpy(~ok).to(self.device), -1e9)
        p = F.softmax(lo / temp, -1).cpu().numpy()
        return int(rng.choice(self.n_actions, p=p / p.sum()))

    def fit(self, imgs, poss, acts, steps=300, bs=256, mask=None, weights=None):
        """Cross-entropy imitation. `mask` (available outputs) is applied to the logits exactly as when
        acting (diagnosis D1: an unmasked fit on a game where one output exists saturates the head and
        collapses any later use of it); with fewer than 2 available outputs nothing is learned.
        `weights` gives per-row importance (e.g. demonstration rows down-weighted)."""
        if mask is not None and int(np.sum(mask)) < 2:
            return float("nan")
        I = torch.from_numpy(np.asarray(imgs, np.uint8)).to(self.device)
        P = torch.from_numpy(np.asarray(poss)).to(self.device)
        A = torch.from_numpy(np.asarray(acts, np.int64)).to(self.device)
        W = None if weights is None else torch.from_numpy(np.asarray(weights, np.float32)).to(self.device)
        M = None if mask is None else torch.from_numpy(~np.asarray(mask, bool)).to(self.device)
        n = len(A)
        loss = None
        for _ in range(steps):
            idx = torch.randint(0, n, (min(bs, n),), device=self.device)
            lo = self.net(I[idx].float(), P[idx])
            if M is not None:
                lo = lo.masked_fill(M, -1e9)
            ce = F.cross_entropy(lo, A[idx], reduction="none")
            loss = ce.mean() if W is None else (ce * W[idx]).sum() / W[idx].sum().clamp(min=1e-6)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.detach())


class EffectAdapter:
    """Per-game table effect -> buttons, rebuilt by the probes at game entry (grant G5). The policy
    acts over the canonical EFFECTS, never over buttons, so its weights can cross games."""

    def __init__(self, table, effects):
        self.effects = list(effects)
        self.buttons = {e: [a for a, row in enumerate(table) if row["effect"] == e] for e in self.effects}
        if not self.buttons.get("none"):
            self.buttons["none"] = [0]                     # NOOP is action 0 in ALE minimal sets
        self.available = np.array([bool(self.buttons.get(e)) for e in self.effects])
        self.effect_of = {a: self.effects.index(row["effect"]) for a, row in enumerate(table)}

    def to_button(self, e, rng):
        opts = self.buttons.get(self.effects[e]) or self.buttons["none"]
        return int(opts[int(rng.integers(len(opts)))])

    def to_effect(self, a):
        return self.effect_of.get(int(a), 0)
