"""v21 — INTEGRATION M1: concept LIBRARY + RECOGNISER + REUSE.

The bottom rung of the grand integration ("drop it on a new task -> recognise
which known concept applies -> reuse it"). Combines the v5 relevance-gate idea
with v19 concept-models.

Setup: K distinct 'physics' for where a shape lands on a terrain:
  rule 0 (gravity):      landing = min over the piece's columns of (surface-bp)
  rule 1 (edge-collide): landing = the first column only
  rule 2 (soft):         landing = mean over the columns
For each rule we train a model on BROAD shape variety (the v19 recipe, so each
generalises to unseen shapes). The LIBRARY = the K models. Faced with a NEW task
(a batch of observations from a HIDDEN rule, using HELD-OUT shapes), the
RECOGNISER picks the library model with the lowest error, and we REUSE it.

Decisive: recognition accuracy ~ perfect, and the reused model's error is low on
ALL rules, while a single FIXED model fails on the non-matching rules.

Usage: python -m scripts.integrate_v21
"""

import argparse
import itertools
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE

W, H, NCOL = 8, 14, 5
ALL_SHAPES = torch.tensor(list(itertools.product(range(4), repeat=4)),
                          dtype=torch.float32, device=DEVICE)        # (256,4)
RULES = ["gravity(min)", "edge(first)", "soft(mean)"]


def landing(surface, bp, rule):
    out = torch.empty(surface.shape[0], NCOL, device=DEVICE)
    for col in range(NCOL):
        cand = surface[:, col:col + 4] - 1 - bp                      # (n,4)
        if rule == 0:
            out[:, col] = cand.min(dim=1).values
        elif rule == 1:
            out[:, col] = cand[:, 0]
        else:
            out[:, col] = cand.mean(dim=1)
    return out


def gen_batch(n, pool, rule):
    h = torch.randint(0, H - 2, (n, W), device=DEVICE).float()
    surface = H - h
    bp = pool[torch.randint(0, pool.shape[0], (n,), device=DEVICE)]
    return surface, bp, landing(surface, bp, rule)


class RuleNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W + 4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, NCOL))

    def forward(self, surface, bp):
        return self.net(torch.cat([surface / H, bp / 3.0], -1)) * H


def train_one(rule, train_shapes, iters=3000, batch=2048):
    m = RuleNet().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(iters):
        s, bp, land = gen_batch(batch, train_shapes, rule)
        loss = (m(s, bp) - land).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def model_mae(m, s, bp, land):
    return float((m(s, bp) - land).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train-shapes", type=int, default=200)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--n-tasks", type=int, default=300)
    p.add_argument("--obs", type=int, default=64)        # observations per new task
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.n_tasks = 300, 50

    perm = torch.randperm(256, device=DEVICE)
    train_shapes = ALL_SHAPES[perm[:args.n_train_shapes]]
    test_shapes = ALL_SHAPES[perm[args.n_train_shapes:]]              # HELD-OUT
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v21] device={DEVICE} | INTEGRATION M1 | library of {len(RULES)} "
          f"concept-models {RULES} | recognise+reuse on new tasks (unseen shapes)",
          flush=True)
    t0 = time.perf_counter()
    library = [train_one(r, train_shapes, args.iters) for r in range(len(RULES))]
    print(f"  trained library of {len(library)} concept-models | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    # each new task = a batch of observations from a HIDDEN rule, UNSEEN shapes
    correct, reuse_err, fixed_err, oracle_err = 0, [], [], []
    for _ in range(args.n_tasks):
        rule = int(torch.randint(0, len(RULES), (1,)))
        s, bp, land = gen_batch(args.obs, test_shapes, rule)
        maes = [model_mae(m, s, bp, land) for m in library]
        rec = int(torch.tensor(maes).argmin())
        correct += int(rec == rule)
        reuse_err.append(maes[rec])          # reuse the RECOGNISED model
        fixed_err.append(maes[0])            # always model 0 (no recognition)
        oracle_err.append(maes[rule])        # the true model (upper bound)

    acc = correct / args.n_tasks
    re_, fx, orc = (sum(reuse_err) / len(reuse_err), sum(fixed_err) / len(fixed_err),
                    sum(oracle_err) / len(oracle_err))
    ok = acc >= 0.9 and re_ <= fx * 0.6
    verdict = (f"RECOGNISE-AND-REUSE WORKS — recognition accuracy {acc:.0%} on "
               f"new tasks (unseen shapes, hidden rule); reused-model error "
               f"{re_:.3f} ~ oracle {orc:.3f}, vs a fixed single model {fx:.3f}. "
               f"The agent picks the right learned concept for a new situation "
               f"and reuses it — the core of the integration (v5 gate x v19 "
               f"concepts)."
               if ok else
               f"PARTIAL — recog acc {acc:.0%}, reuse err {re_:.3f} vs fixed "
               f"{fx:.3f} (oracle {orc:.3f}).")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v21_integration_m1.json"), "w") as f:
        json.dump(dict(rules=RULES, recog_acc=acc, reuse_err=re_, fixed_err=fx,
                       oracle_err=orc, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
