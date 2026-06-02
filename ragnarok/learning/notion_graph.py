"""The NOTION GRAPH — our original architecture (see NOTION_GRAPH_DESIGN.md).

A growing, self-compressing library of local dynamics NOTIONS. Each notion is a
small predictor (local spatio-temporal context -> next patch). Per patch, the
best-predicting notion is BOUND (hard selection); the prediction is its output.
Unexplained surprise -> MINT a new notion; redundancy/disuse -> PRUNE. The agent
represents pixel dynamics ONLY as a composition of notions, so prior notions are
always in the loop (reuse is forced, not optional).

This is the v0.1 core: perception+prediction. Control (acting through notions),
cross-world forced-reuse, and consolidation come in later versions.
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


class NotionLibrary(nn.Module):
    """K notions as a batched 2-layer MLP (ctx -> next patch). Hard per-patch binding
    to the lowest-error notion; that notion is the one trained on the patch (VQ/EM-like
    specialisation). The library GROWS on surprise and SHRINKS on disuse."""

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

    def learn(self, ctx, target):
        """One gradient step: each patch trained on its BEST notion. Returns
        (loss, per-patch min-error, assignment)."""
        preds = self.predict_all(ctx)                            # (P,K,patch_dim)
        err = ((preds - target.unsqueeze(1)) ** 2).mean(-1)      # (P,K)
        min_err, assign = err.min(1)                             # (P,)
        loss = min_err.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            share = torch.zeros(self.K, device=DEVICE)
            share.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float))
            share /= share.sum().clamp(min=1)
            self.usage.mul_(0.98).add_(0.02 * share)
        return float(loss.item()), min_err.detach(), assign.detach()

    @torch.no_grad()
    def mint(self, ctx_seed=None, target_seed=None):
        """Add one notion. If seed patches are given, initialise it to fit their
        MEAN target (a useful starting bias for the surprising dynamics)."""
        if self.K >= self.k_max:
            return False
        w1 = self._w(1, self.ctx_dim, self.hidden)[0]
        b1 = torch.zeros(self.hidden, device=DEVICE)
        w2 = self._w(1, self.hidden, self.patch_dim)[0]
        b2 = torch.zeros(self.patch_dim, device=DEVICE)
        if target_seed is not None and target_seed.numel():
            b2 = target_seed.mean(0)                             # predict the mean surprise
        self.W1 = nn.Parameter(torch.cat([self.W1.data, w1[None]], 0))
        self.b1 = nn.Parameter(torch.cat([self.b1.data, b1[None]], 0))
        self.W2 = nn.Parameter(torch.cat([self.W2.data, w2[None]], 0))
        self.b2 = nn.Parameter(torch.cat([self.b2.data, b2[None]], 0))
        self.usage = torch.cat([self.usage, torch.full((1,), 1.0 / self.K, device=DEVICE)])
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
        self.K = len(idx)
        self._mk_opt()
        return removed
