"""The NOTION GRAPH — our original architecture (see NOTION_GRAPH_DESIGN.md).

A growing, self-compressing library of local dynamics NOTIONS. Each notion is a
small predictor (local spatio-temporal context -> next patch) PLUS a context
DETECTOR (a centroid in context space). Per patch, learning binds to the
lowest-error notion (EM/VQ specialisation); FORWARD prediction (for control)
binds to the nearest-context notion, with NO future target needed. Unexplained
surprise -> MINT; redundancy/disuse -> PRUNE. The agent represents and ACTS on the
world ONLY by composing notions (reuse is forced, not optional).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE


def patchify(img, p):
    """(N,C,H,W) -> patches (N, gh*gw, C*p*p), grid (gh,gw). Non-overlapping p x p."""
    N, C, H, W = img.shape
    x = img.unfold(2, p, p).unfold(3, p, p)              # (N,C,gh,gw,p,p)
    gh, gw = x.shape[2], x.shape[3]
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, gh * gw, C * p * p)
    return x, gh, gw


def unpatchify(patches, gh, gw, p, C=3):
    """Inverse of patchify: (N, gh*gw, C*p*p) -> (N, C, gh*p, gw*p)."""
    N = patches.shape[0]
    x = patches.reshape(N, gh, gw, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(N, C, gh * p, gw * p)


class NotionLibrary(nn.Module):
    """K notions as a batched 2-layer MLP (ctx -> next patch) + per-notion context
    centroids (the detector). Learning binds by lowest error; forward prediction
    binds by nearest context centroid. Grows on surprise, shrinks on disuse."""

    def __init__(self, ctx_dim, patch_dim, hidden=64, k_init=1, k_max=32, lr=1e-3):
        super().__init__()
        self.ctx_dim, self.patch_dim, self.hidden = ctx_dim, patch_dim, hidden
        self.k_max, self.lr = k_max, lr
        self.K = k_init
        self.W1 = nn.Parameter(self._w(k_init, ctx_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(k_init, hidden, device=DEVICE))
        self.W2 = nn.Parameter(self._w(k_init, hidden, patch_dim))
        self.b2 = nn.Parameter(torch.zeros(k_init, patch_dim, device=DEVICE))
        self.usage = torch.zeros(k_init, device=DEVICE)         # EMA assignment share
        self.centroid = torch.zeros(k_init, ctx_dim, device=DEVICE)   # context detector
        self._seen = torch.zeros(k_init, device=DEVICE)         # has centroid been set
        self._mk_opt()

    def _w(self, k, din, dout):
        w = torch.empty(k, din, dout, device=DEVICE)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        return w

    def _mk_opt(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_all(self, ctx):
        """ctx (P, ctx_dim) -> (P, K, patch_dim)."""
        h = F.relu(torch.einsum("pd,kdh->pkh", ctx, self.W1) + self.b1)
        return torch.einsum("pkh,khd->pkd", h, self.W2) + self.b2

    @torch.no_grad()
    def predict(self, ctx):
        """FORWARD prediction (for control): bind each patch to the nearest-context
        notion (NO target needed), return that notion's prediction. -> (pred, assign)."""
        d = torch.cdist(ctx, self.centroid)                      # (P,K)
        d = torch.where(self._seen.unsqueeze(0) > 0, d, torch.full_like(d, 1e9))
        assign = d.argmin(1)
        preds = self.predict_all(ctx)
        pred = preds.gather(1, assign.view(-1, 1, 1).expand(-1, 1, self.patch_dim)).squeeze(1)
        return pred, assign

    def learn(self, ctx, target, fg=0.05):
        """One gradient step: each patch trained on its BEST (lowest-error) notion.
        Loss is FOREGROUND-WEIGHTED (bright/moving pixels weighted more) so notions
        are forced to render the sparse-but-crucial objects, not just the background.
        Also EMA-updates that notion's context centroid. Returns (loss, min_err, assign)."""
        preds = self.predict_all(ctx)                            # (P,K,patch_dim)
        w = target + fg                                          # (P, patch_dim): bright matters
        sq = (preds - target.unsqueeze(1)) ** 2                  # (P,K,patch_dim)
        err = (sq * w.unsqueeze(1)).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1e-6)  # (P,K)
        min_err, assign = err.min(1)                             # (P,)
        loss = min_err.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            # usage weighted by FOREGROUND MASS: a notion explaining the sparse-but-
            # crucial bright objects (paddle/fruit) counts as much as it matters, so the
            # compression/prune drive does NOT kill it for explaining few patches.
            pw = target.clamp(min=0).sum(-1) + 0.02
            wcnt = torch.zeros(self.K, device=DEVICE).index_add_(0, assign, pw)
            self.usage.mul_(0.98).add_(0.02 * (wcnt / wcnt.sum().clamp(min=1e-6)))
            cnt = torch.zeros(self.K, device=DEVICE).index_add_(0, assign,
                                                               torch.ones_like(pw))
            csum = torch.zeros(self.K, self.ctx_dim, device=DEVICE).index_add_(0, assign, ctx)
            cmean = csum / cnt.clamp(min=1).unsqueeze(-1)
            upd = cnt > 0
            # first time a notion is used, set centroid; else EMA
            fresh = upd & (self._seen == 0)
            self.centroid[fresh] = cmean[fresh]
            ema = upd & (self._seen > 0)
            self.centroid[ema] = 0.95 * self.centroid[ema] + 0.05 * cmean[ema]
            self._seen[upd] = 1.0
        return float(loss.item()), min_err.detach(), assign.detach()

    @torch.no_grad()
    def mint(self, ctx_seed=None, target_seed=None):
        """Add one notion. Initialise its predictor bias to the mean surprise target
        and its detector centroid to the mean surprise context."""
        if self.K >= self.k_max:
            return False
        w1 = self._w(1, self.ctx_dim, self.hidden)[0]
        b1 = torch.zeros(self.hidden, device=DEVICE)
        w2 = self._w(1, self.hidden, self.patch_dim)[0]
        b2 = (target_seed.mean(0) if target_seed is not None and target_seed.numel()
              else torch.zeros(self.patch_dim, device=DEVICE))
        cen = (ctx_seed.mean(0) if ctx_seed is not None and ctx_seed.numel()
               else torch.zeros(self.ctx_dim, device=DEVICE))
        self.W1 = nn.Parameter(torch.cat([self.W1.data, w1[None]], 0))
        self.b1 = nn.Parameter(torch.cat([self.b1.data, b1[None]], 0))
        self.W2 = nn.Parameter(torch.cat([self.W2.data, w2[None]], 0))
        self.b2 = nn.Parameter(torch.cat([self.b2.data, b2[None]], 0))
        self.usage = torch.cat([self.usage, torch.full((1,), 1.0 / self.K, device=DEVICE)])
        self.centroid = torch.cat([self.centroid, cen[None]], 0)
        self._seen = torch.cat([self._seen, torch.ones(1, device=DEVICE)])
        self.K += 1
        self._mk_opt()
        return True

    @torch.no_grad()
    def prune(self, min_usage=1e-3):
        """Remove notions with negligible usage (keep at least 1)."""
        if self.K <= 1:
            return 0
        keep = self.usage >= min_usage
        if keep.all() or int(keep.sum()) == 0:
            return 0
        idx = keep.nonzero(as_tuple=True)[0]
        self.W1 = nn.Parameter(self.W1.data[idx])
        self.b1 = nn.Parameter(self.b1.data[idx])
        self.W2 = nn.Parameter(self.W2.data[idx])
        self.b2 = nn.Parameter(self.b2.data[idx])
        removed = self.K - len(idx)
        self.usage = self.usage[idx]
        self.centroid = self.centroid[idx]
        self._seen = self._seen[idx]
        self.K = len(idx)
        self._mk_opt()
        return removed
