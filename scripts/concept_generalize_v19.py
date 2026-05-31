"""v19 — can a net learn the GRAVITY RULE and GENERALISE to unseen shapes?

v18 failed (a landing model trained on 5 tetrominoes memorised them, 7.9 lines
zero-shot on 2 unseen). Hypothesis: it memorised because there were too few
shapes. Isolated, rigorous test (no game, no noise): the pure "where does a
shape land on a terrain" rule.

  terrain: a height per column.
  shape:   a bottom-profile bp (the lowest cell row-offset in each of its 4
           columns) — there are 4^4 = 256 such shapes.
  landing (the gravity+collision RULE): for a placement at column `col`,
           landing = min over the shape's columns i of (surface[col+i] - 1 - bp[i]).

Train a net on MANY shapes, test on HELD-OUT shapes. If held-out error ~ train
error, the net learned the general RULE (not memorised) -> a genuinely
reusable concept is learnable once it can't just memorise. If held-out error is
much worse, even this simple rule needs a structured inductive bias.

Usage: python -m scripts.concept_generalize_v19 [--n-train-shapes 200]
"""

import argparse
import itertools
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE

W, H = 8, 14
ALL_SHAPES = torch.tensor(list(itertools.product(range(4), repeat=4)),
                          dtype=torch.float32, device=DEVICE)   # (256, 4) bottom-profiles
NCOL = W - 3                                                    # valid placement columns


def gen_batch(n, shape_pool):
    """Random terrains + random shapes from shape_pool; true landings per col."""
    h = torch.randint(0, H - 2, (n, W), device=DEVICE).float()  # column heights
    surface = H - h                                             # row of top filled cell
    sidx = torch.randint(0, shape_pool.shape[0], (n,), device=DEVICE)
    bp = shape_pool[sidx]                                       # (n,4)
    land = torch.full((n, NCOL), 1e9, device=DEVICE)
    for col in range(NCOL):
        # landing = min_i (surface[col+i] - 1 - bp[i])
        cand = surface[:, col:col + 4] - 1 - bp                 # (n,4)
        land[:, col] = cand.min(dim=1).values
    return surface, bp, land


class RuleNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W + 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NCOL))

    def forward(self, surface, bp):
        return self.net(torch.cat([surface / H, bp / 3.0], -1)) * H


@torch.no_grad()
def mae(model, shape_pool, n=8192):
    surface, bp, land = gen_batch(n, shape_pool)
    return float((model(surface, bp) - land).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train-shapes", type=int, default=200)   # of 256
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters = 200

    perm = torch.randperm(256, device=DEVICE)
    train_shapes = ALL_SHAPES[perm[:args.n_train_shapes]]
    test_shapes = ALL_SHAPES[perm[args.n_train_shapes:]]        # HELD-OUT shapes
    os.makedirs(args.out_dir, exist_ok=True)
    model = RuleNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[v19] device={DEVICE} | learn the GRAVITY RULE | train on "
          f"{args.n_train_shapes} shapes, test on {256 - args.n_train_shapes} "
          f"HELD-OUT shapes", flush=True)

    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        surface, bp, land = gen_batch(args.batch, train_shapes)
        loss = (model(surface, bp) - land).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, args.iters // 8) == 0:
            print(f"  it {it:>5} | train MAE {mae(model, train_shapes):.3f} | "
                  f"HELD-OUT MAE {mae(model, test_shapes):.3f} rows | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    tr, te = mae(model, train_shapes), mae(model, test_shapes)
    # baseline: predicting the mean landing (no rule) — MAE of a constant
    surf, bp, land = gen_batch(8192, test_shapes)
    base = float((land - land.mean()).abs().mean())
    ok = te <= tr * 1.5 and te <= 0.5            # held-out ~ train AND accurate (<0.5 row)
    verdict = (f"GRAVITY RULE GENERALISES — trained on {args.n_train_shapes} "
               f"shapes, the net predicts landings on UNSEEN shapes to {te:.3f} "
               f"rows (train {tr:.3f}; naive baseline {base:.2f}). It learned the "
               f"RULE, not memorised shapes -> a genuinely reusable concept is "
               f"learnable once it can't just memorise (v18 failed for lack of "
               f"shape variety)."
               if ok else
               f"DOES NOT FULLY GENERALISE — held-out MAE {te:.3f} vs train "
               f"{tr:.3f} (baseline {base:.2f}). Even this simple rule needs more "
               f"shapes or a structured inductive bias.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v19_rule_generalize.json"), "w") as f:
        json.dump(dict(n_train_shapes=args.n_train_shapes, train_mae=tr,
                       heldout_mae=te, baseline_mae=base, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
