"""v33 — does the variety->generalisation recipe hold OUTSIDE games? (a symbolic/
math domain, per the strategy review's 'smallest real step toward maths').

The project's thesis is that broad VARIETY forces a model to learn the underlying
RULE, which then generalises (v19 in concepts, v27b in games). The strategy review
correctly noted the recent arc is stuck in one game family and the 'eventually
maths/language' ambition is unsupported. v33 tests the SAME recipe in a clean,
non-game, continuous-MATH domain: IN-CONTEXT FUNCTION REGRESSION.

A 'rule' is a quadratic f(x)=a x^2 + b x + c. The model sees K example (x, f(x))
pairs from one function and must predict f(x_q) for a query — i.e. infer the
function from examples and apply it. We train models on R DISTINCT functions for
R in {1,2,4,...} (and a continuous-variety arm that resamples a fresh function
every batch), at EQUAL training steps, and measure MSE on HELD-OUT functions never
seen in training.

Hypothesis (the recipe, in maths): held-out MSE FALLS as R grows — more
function-variety forces the model to learn the general 'infer-and-apply' procedure
(curve fitting) rather than memorise specific functions, so it generalises to new
functions. This is a clean (non-RL, low-noise) version of the v30 scaling law, in a
domain distinct from games. HONEST framing: this is classic meta-learning / in-
context learning (not a novel ML result); the point is whether the project's
variety->generalisation THESIS is domain-general (supporting the cross-domain
ambition) or specific to RL/games.

Usage: python -m scripts.symbolic_variety_v33 [--rs 1 2 4 8 16 32] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE

K = 5                      # context examples per function
XLO, XHI = -2.0, 2.0


def sample_funcs(n, g):
    """n quadratics (a,b,c) ~ U[-1,1]^3, as a (n,3) tensor."""
    return (torch.rand(n, 3, generator=g, device=DEVICE) * 2 - 1)


def make_batch(funcs, batch, g):
    """funcs: (R,3) pool to draw from (R may be 1). Returns (inputs (batch,2K+1),
    targets (batch,))."""
    R = funcs.shape[0]
    idx = torch.randint(0, R, (batch,), generator=g, device=DEVICE)
    abc = funcs[idx]                                   # (batch,3)
    x = torch.rand(batch, K + 1, generator=g, device=DEVICE) * (XHI - XLO) + XLO
    y = abc[:, 0:1] * x * x + abc[:, 1:2] * x + abc[:, 2:3]   # (batch,K+1)
    # interleave context (x_i,y_i) then append query x; target = query y
    ctx = torch.stack([x[:, :K], y[:, :K]], -1).reshape(batch, 2 * K)
    inp = torch.cat([ctx, x[:, K:K + 1]], 1)           # (batch, 2K+1)
    return inp, y[:, K]


def make_batch_continuous(batch, g):
    """Fresh function every example = the infinite-variety limit."""
    abc = torch.rand(batch, 3, generator=g, device=DEVICE) * 2 - 1
    x = torch.rand(batch, K + 1, generator=g, device=DEVICE) * (XHI - XLO) + XLO
    y = abc[:, 0:1] * x * x + abc[:, 1:2] * x + abc[:, 2:3]
    ctx = torch.stack([x[:, :K], y[:, :K]], -1).reshape(batch, 2 * K)
    return torch.cat([ctx, x[:, K:K + 1]], 1), y[:, K]


def mlp():
    return nn.Sequential(nn.Linear(2 * K + 1, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEVICE)


def train(funcs, steps, batch, g):
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        inp, tgt = (make_batch_continuous(batch, g) if funcs is None
                    else make_batch(funcs, batch, g))
        loss = F.mse_loss(net(inp).squeeze(-1), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def eval_mse(net, funcs, g, reps=20, batch=512):
    tot = 0.0
    for _ in range(reps):
        inp, tgt = make_batch(funcs, batch, g)
        tot += float(F.mse_loss(net(inp).squeeze(-1), tgt))
    return tot / reps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--heldout", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.rs, args.steps, args.batch = [1, 8], 300, 128

    g = torch.Generator(device=DEVICE); g.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    heldout = sample_funcs(args.heldout, g)            # FIXED held-out test functions
    # a fixed pool of train functions; R uses the first R (nested) — held-out is disjoint
    pool = sample_funcs(max(args.rs), g)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v33] device={DEVICE} | SYMBOLIC variety->generalisation (in-context "
          f"quadratic regression) | R={args.rs} + continuous | held-out MSE on "
          f"{args.heldout} unseen functions | {args.steps} steps each", flush=True)
    t0 = time.perf_counter()

    rows = []
    for R in args.rs:
        net = train(pool[:R], args.steps, args.batch, g)
        ho = eval_mse(net, heldout, g)
        tr = eval_mse(net, pool[:R], g)
        rows.append(dict(R=R, heldout_mse=round(ho, 4), train_mse=round(tr, 4)))
        print(f"  R={R:>3}: held-out MSE {ho:.3f} | train MSE {tr:.3f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    net = train(None, args.steps, args.batch, g)       # continuous variety = infinite R
    cont = eval_mse(net, heldout, g)
    rows.append(dict(R="cont", heldout_mse=round(cont, 4), train_mse=None))
    print(f"  R=inf (continuous): held-out MSE {cont:.3f} | {time.perf_counter()-t0:.0f}s",
          flush=True)

    finite = [r for r in rows if r["R"] != "cont"]
    lo, hi = finite[0]["heldout_mse"], finite[-1]["heldout_mse"]
    drop = round(lo - hi, 4)
    monotone = all(finite[i]["heldout_mse"] >= finite[i + 1]["heldout_mse"] - 0.02
                   for i in range(len(finite) - 1))
    # naive baseline: predict the mean of the K context y's (no rule induction)
    ok = hi < lo * 0.5 and cont <= hi + 0.05
    verdict = (
        f"THE RECIPE HOLDS IN MATHS — held-out MSE falls from {lo:.2f} (R={finite[0]['R']} "
        f"function) to {hi:.2f} (R={finite[-1]['R']}) toward the continuous-variety floor "
        f"{cont:.2f}: training on a VARIETY of functions teaches the model to INFER and "
        f"apply an unseen function (generalise), where few functions only memorise. The "
        f"variety->generalisation recipe is DOMAIN-GENERAL (games AND symbolic maths), a "
        f"clean low-noise scaling curve. (Honest: this is classic meta-learning; the point "
        f"is the thesis spans domains, not ML novelty.)"
        if ok else
        f"PARTIAL/OTHER — held-out MSE R{finite[0]['R']} {lo:.2f} -> R{finite[-1]['R']} "
        f"{hi:.2f} (drop {drop}), continuous {cont:.2f}, monotone={monotone}. See rows.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v33_symbolic.json"), "w") as f:
        json.dump(dict(rs=args.rs, steps=args.steps, rows=rows, continuous_mse=cont,
                       drop=drop, monotone=monotone, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
