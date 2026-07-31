"""v37 — MASTER a notion via SEVERAL different exercises -> it becomes RELIABLY
REUSABLE on a NEW exercise. (The user's pedagogical insight: like a human, you
master a notion by practising it across varied exercises, not one.)

Notion = GRAVITY. Inputs = a launch (x0, y0, vx, vy); a fixed g determines the
whole parabola. Four DIFFERENT exercises all depend on gravity:
  E1 landing-x, E2 peak height, E3 flight time, E4 impact speed.
We pretrain a shared BODY (MLP) on K of these exercises (K = 0..4, EQUAL data),
each with its own head, then FREEZE the body and measure how well its
representation transfers to a NEW, held-out exercise — E5: height at a given x —
with only N=20 examples (a ridge linear probe). More exercise-variety should give
a more gravity-general representation -> lower E5 error (better reuse).

Hypothesis: E5 transfer error DECREASES as K (number of varied exercises) grows.
A clean, reliable (supervised) test of 'varied practice -> a reusable notion'.

Usage: python -m scripts.concept_mastery_v37 [--ks 0 1 2 3 4] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE

G = 0.5


def sample_s(n, gen):
    x0 = torch.rand(n, generator=gen, device=DEVICE) * 0.3
    y0 = torch.rand(n, generator=gen, device=DEVICE) * 0.6 + 0.4
    vx = torch.rand(n, generator=gen, device=DEVICE) * 0.45 + 0.35
    vy = torch.rand(n, generator=gen, device=DEVICE) * 0.9 - 0.3
    return torch.stack([x0, y0, vx, vy], -1)


def quantities(s):
    x0, y0, vx, vy = s.unbind(-1)
    t_land = (vy + (vy * vy + 2 * G * y0).clamp(min=1e-6).sqrt()) / G
    e1 = x0 + vx * t_land                                   # landing x
    e2 = y0 + vy.clamp(min=0) ** 2 / (2 * G)                # peak height
    e3 = t_land                                             # flight time
    e4 = (vx * vx + (vy - G * t_land) ** 2).sqrt()          # impact speed
    tX = ((0.6 - x0) / vx).clamp(min=0)                     # held-out: height at x=0.6
    e5 = y0 + vy * tX - 0.5 * G * tX * tX
    return torch.stack([e1, e2, e3, e4], -1), e5


def make_standardizer(gen):
    s = sample_s(20000, gen)
    tr, e5 = quantities(s)
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    mu5, sd5 = e5.mean(), e5.std() + 1e-6
    return mu, sd, mu5, sd5


def body_net():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()).to(DEVICE)


def pretrain(K, steps, batch, gen, mu, sd):
    body = body_net()
    heads = [nn.Linear(64, 1).to(DEVICE) for _ in range(max(K, 1))]
    params = list(body.parameters())
    for h in heads:
        params += list(h.parameters())
    opt = torch.optim.Adam(params, 1e-3)
    if K == 0:
        return body                                        # untrained body (baseline)
    for _ in range(steps):
        s = sample_s(batch, gen)
        feats = body(s)
        tgt = (quantities(s)[0] - mu) / sd                 # standardized targets
        loss = sum(F.mse_loss(heads[k](feats).squeeze(-1), tgt[:, k]) for k in range(K))
        opt.zero_grad(); loss.backward(); opt.step()
    return body


@torch.no_grad()
def probe_e5(body, n_train, gen, mu5, sd5, lam=1e-2):
    """Freeze body; fit a ridge linear head on n_train E5 examples; test MSE."""
    s_tr = sample_s(n_train, gen); s_te = sample_s(4000, gen)
    ftr = body(s_tr); fte = body(s_te)
    ytr = ((quantities(s_tr)[1] - mu5) / sd5)
    yte = ((quantities(s_te)[1] - mu5) / sd5)
    X = torch.cat([ftr, torch.ones(n_train, 1, device=DEVICE)], 1)     # (N,65)
    A = X.t() @ X + lam * torch.eye(X.shape[1], device=DEVICE)
    w = torch.linalg.solve(A, X.t() @ ytr.unsqueeze(-1))               # (65,1)
    Xte = torch.cat([fte, torch.ones(4000, 1, device=DEVICE)], 1)
    pred = (Xte @ w).squeeze(-1)
    return float(F.mse_loss(pred, yte))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--probe-n", type=int, default=20)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.reps = 400, 2

    gen = torch.Generator(device=DEVICE); gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    mu, sd, mu5, sd5 = make_standardizer(gen)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v37] device={DEVICE} | MASTER gravity via K varied exercises -> reuse on "
          f"a NEW exercise (height@x, N={args.probe_n} examples) | K={args.ks}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for K in args.ks:
        errs = []
        for r in range(args.reps):
            body = pretrain(K, args.steps, args.batch, gen, mu, sd)
            errs.append(probe_e5(body, args.probe_n, gen, mu5, sd5))
        m = sum(errs) / len(errs)
        rows.append(dict(K=K, e5_transfer_mse=round(m, 4)))
        print(f"  K={K} exercises: E5 transfer MSE {m:.3f} (over {args.reps} reps) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    by_k = {r["K"]: r["e5_transfer_mse"] for r in rows}
    lo_k, hi_k = min(by_k), max(by_k)
    drop = round(by_k[lo_k] - by_k[hi_k], 4)
    monotone = all(rows[i]["e5_transfer_mse"] >= rows[i + 1]["e5_transfer_mse"] - 0.03
                   for i in range(len(rows) - 1))
    ok = by_k[hi_k] < by_k[lo_k] * 0.6 and monotone
    verdict = (
        f"VARIED PRACTICE -> A REUSABLE NOTION — transferring the gravity "
        f"representation to a NEW exercise (height@x, {args.probe_n} examples) gets "
        f"steadily EASIER with more varied exercises: E5 MSE {by_k[lo_k]:.2f} (K={lo_k}) "
        f"-> {by_k[hi_k]:.2f} (K={hi_k}). Practising the SAME notion across several "
        f"different exercises makes it a genuinely reusable abstraction — the user's "
        f"insight, confirmed cleanly. (Combined with v36: this reusable notion pays "
        f"off in contexts where re-deriving it is hard.)"
        if ok else
        f"PARTIAL — E5 MSE by K {by_k}; drop {drop}, monotone={monotone}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v37_concept_mastery.json"), "w") as f:
        json.dump(dict(ks=args.ks, rows=rows, drop=drop, monotone=monotone,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
