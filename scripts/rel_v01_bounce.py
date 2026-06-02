"""rel v0.1 — REUSE a SHARED PHYSICAL RULE across games (not appearance).

The lesson from NG (option 1): patch-prediction reuse was APPEARANCE similarity (the ball
sprite), a warm-start that faded. Here we target a true shared INVARIANT: the bounce rule
"the velocity component perpendicular to a surface flips near that surface" — identical in
Pong/Breakout/Catcher (same env physics), independent of appearance/geometry.

Model = ONE shared bounce module h(v, d_lo, d_hi) -> Delta v, applied to BOTH axes:
  vx_next = vx + h(vx, x, 1-x);  vy_next = vy + h(vy, y, 1-y).
This bakes in the relational/shared structure. Baseline = a param-matched MONOLITH MLP
(vx,vy,x,y)->(vx_next,vy_next) with NO shared structure. Reuse test: train on Pong, transfer
to Breakout; measure next-velocity error WARM (reuse) vs SCRATCH, at the ASYMPTOTE (not a head-
start), for both model classes. If the shared-bounce module reuses DURABLY across games where the
monolith does not, a genuine shared-dynamics invariant transfers (the thing NG could not show).

Usage: python -m scripts.rel_v01_bounce [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout


def perceive_ball(frame, img):
    """White ball centroid (min over channels = white; paddles are single-colour)."""
    xs = torch.arange(img, device=DEVICE).float()
    ys = torch.arange(img, device=DEVICE).float()
    white = frame.min(1).values                                   # (N,img,img)
    m = white.sum((-1, -2)).clamp(min=1e-3)
    bx = (white.sum(-2) * xs).sum(-1) / m / (img - 1)
    by = (white.sum(-1) * ys).sum(-1) / m / (img - 1)
    return bx, by


class SharedBounce(nn.Module):
    """One shared module h(v, d_lo, d_hi)->Delta v, applied to both axes."""
    def __init__(self, hidden=32):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, vx, vy, x, y):
        dvx = self.h(torch.stack([vx, x, 1 - x], -1)).squeeze(-1)
        dvy = self.h(torch.stack([vy, y, 1 - y], -1)).squeeze(-1)
        return vx + dvx, vy + dvy


class Monolith(nn.Module):
    """Param-matched MLP (vx,vy,x,y)->(vx_next,vy_next), no shared structure."""
    def __init__(self, hidden=46):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, vx, vy, x, y):
        o = self.f(torch.stack([vx, vy, x, y], -1))
        return vx + o[..., 0], vy + o[..., 1]


def collect(env, img, steps, N):
    """Roll env; return tensors (vx,vy,x,y, tvx,tvy) of bounce-relevant transitions."""
    def ball():
        return perceive_ball(env.state.view(N, 3, img, img), img)
    bx0, by0 = ball()
    env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
    bx1, by1 = ball()
    X = []
    for _ in range(steps):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        bx2, by2 = ball()
        vx, vy = bx1 - bx0, by1 - by0
        tvx, tvy = bx2 - bx1, by2 - by1
        X.append(torch.stack([vx, vy, bx1, by1, tvx, tvy], -1))
        bx0, by0, bx1, by1 = bx1, by1, bx2, by2
    return torch.cat(X, 0)                                         # (steps*N, 6)


def train_eval(model, data_train, data_test, iters, lr=1e-3, bs=4096):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    def loss_on(d):
        vx, vy, x, y, tvx, tvy = d.unbind(-1)
        px, py = model(vx, vy, x, y)
        # weight bounce steps (where velocity changed) — the meaningful signal
        w = (((tvx - vx) ** 2 + (tvy - vy) ** 2) > 1e-6).float() + 0.05
        return (((px - tvx) ** 2 + (py - tvy) ** 2) * w).sum() / w.sum()
    def bounce_err(d):                              # error ONLY on bounce steps (the shared rule)
        vx, vy, x, y, tvx, tvy = d.unbind(-1)
        px, py = model(vx, vy, x, y)
        b = ((tvx - vx) ** 2 + (tvy - vy) ** 2) > 1e-6
        if int(b.sum()) == 0:
            return 0.0
        e = (px - tvx) ** 2 + (py - tvy) ** 2
        return float(e[b].mean())
    curve = []
    n = data_train.shape[0]
    for it in range(1, iters + 1):
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        loss = loss_on(data_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 10) == 0:
            with torch.no_grad():
                curve.append(round(bounce_err(data_test), 6))
    return curve


def run_arm(ModelCls, hidden, pong_tr, bk_tr, bk_te, iters, seed):
    torch.manual_seed(seed)
    # WARM: pretrain on Pong, then continue on Breakout
    warm = ModelCls(hidden).to(DEVICE)
    train_eval(warm, pong_tr, pong_tr[:1], iters)                 # pretrain on Pong
    warm_curve = train_eval(warm, bk_tr, bk_te, iters)            # adapt on Breakout
    # SCRATCH: Breakout only
    torch.manual_seed(seed + 1)
    scr = ModelCls(hidden).to(DEVICE)
    scr_curve = train_eval(scr, bk_tr, bk_te, iters)
    return warm_curve, scr_curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--collect-steps", type=int, default=400)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.num_envs, args.collect_steps, args.iters = [0], 64, 200, 800

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[rel v0.1] device={DEVICE} | REUSE a SHARED BOUNCE RULE Pong->Breakout (not appearance) "
          f"| shared-module vs param-matched monolith | WARM vs SCRATCH, asymptotic | seeds {args.seeds}",
          flush=True)
    t0 = time.perf_counter()
    rows = []
    for s in args.seeds:
        torch.manual_seed(s)
        pong = DeviceVecPong(args.num_envs, img=args.img, seed=s)
        bk = DeviceVecBreakout(args.num_envs, img=args.img, seed=s)
        bk_te_env = DeviceVecBreakout(args.num_envs, img=args.img, seed=s + 99)
        pong_tr = collect(pong, args.img, args.collect_steps, args.num_envs)
        bk_tr = collect(bk, args.img, args.collect_steps, args.num_envs)
        bk_te = collect(bk_te_env, args.img, args.collect_steps // 2, args.num_envs)

        sw, ss = run_arm(SharedBounce, 32, pong_tr, bk_tr, bk_te, args.iters, s)
        mw, ms = run_arm(Monolith, 46, pong_tr, bk_tr, bk_te, args.iters, s)
        # asymptotic (last-3 mean) warm vs scratch
        def asy(c):
            return sum(c[-3:]) / len(c[-3:])
        sh_warm, sh_scr = asy(sw), asy(ss)
        mo_warm, mo_scr = asy(mw), asy(ms)
        sh_gain = (sh_scr - sh_warm) / max(sh_scr, 1e-9)
        mo_gain = (mo_scr - mo_warm) / max(mo_scr, 1e-9)
        rows.append(dict(seed=s, shared_warm=round(sh_warm, 6), shared_scratch=round(sh_scr, 6),
                         mono_warm=round(mo_warm, 6), mono_scratch=round(mo_scr, 6),
                         shared_gain=round(sh_gain, 3), mono_gain=round(mo_gain, 3)))
        print(f"  seed {s}: SHARED warm {sh_warm:.5f} vs scratch {sh_scr:.5f} (gain {sh_gain*100:+.0f}%) "
              f"| MONOLITH warm {mo_warm:.5f} vs scratch {mo_scr:.5f} (gain {mo_gain*100:+.0f}%) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    sh_gains = [r["shared_gain"] for r in rows]
    mo_gains = [r["mono_gain"] for r in rows]
    durable = all(r["shared_warm"] <= r["shared_scratch"] for r in rows)        # warm not worse at asymptote
    structure = all(r["shared_gain"] > r["mono_gain"] + 0.05 for r in rows)
    positive = len(rows) >= 3 and durable and structure and all(g > 0.1 for g in sh_gains)
    verdict = (
        f"SHARED-RULE REUSE IS DURABLE — the shared bounce module reuses Pong->Breakout at the "
        f"ASYMPTOTE (warm <= scratch, gains {[f'{g*100:+.0f}%' for g in sh_gains]}) and beats the "
        f"param-matched monolith ({[f'{g*100:+.0f}%' for g in mo_gains]}) every seed -> a genuine "
        f"shared-DYNAMICS invariant transfers (not appearance, unlike NG). REVIEW before reporting."
        if positive else
        f"PARTIAL/CHECK — shared gains {sh_gains} vs mono {mo_gains}, durable={durable}, structure={structure}. "
        f"If shared~mono or warm>scratch, the shared structure gives no durable reuse here.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "rel_v01.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, rows=rows, positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
