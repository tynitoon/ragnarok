"""percept v0.3 — MOTION-driven object perception (the moving thing is the object).

percept v0.2 (single-frame sprite AE) bound the constant-x PADDLES but FAILED on the 2D-moving BALL
(err ~0.21 ~ random on 2/3 seeds): the ball is ~1-2 px and single-frame reconstruction is dominated
by the larger paddles, so nothing pressures a slot onto the tiny ball. The ball's defining feature is
that it MOVES every frame. percept v0.3 keeps the EXACT same SpriteAE + eval but weights the
reconstruction loss by MOTION |frame_t - frame_{t-1}|, so moving pixels (the ball) carry strong
gradient. ONE change vs v0.2 (motion weighting); everything else frozen, for a clean attribution.

Usage: python -m scripts.percept_v03_motion [--steps 6000] [--seed 0] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from scripts.percept_v02_sprites import (SpriteAE, ball_diag, fair_random_baseline,
                                         multi_object_diag)


@torch.no_grad()
def collect_pairs(env, steps, N, img):
    """Return temporally-consecutive (prev, cur) frame pairs + cur object positions.

    prev/cur are consecutive timesteps per env, so motion = |cur - prev| isolates moving objects.
    """
    prevs, curs, objs = [], [], []
    prev = env.state.view(N, 3, img, img).clone()
    for _ in range(steps):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        cur = env.state.view(N, 3, img, img).clone()
        xa = torch.full_like(env.bx, env.x_a)
        xo = torch.full_like(env.bx, env.x_o)
        ball = torch.stack([env.bx, env.by], -1)
        padl = torch.stack([xa, env.pad_a], -1)
        padr = torch.stack([xo, env.pad_o], -1)
        prevs.append(prev); curs.append(cur)
        objs.append(torch.stack([ball, padl, padr], 1))
        prev = cur
    return torch.cat(prevs, 0), torch.cat(curs, 0), torch.cat(objs, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--collect-steps", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--motion-w", type=float, default=3.0)     # how strongly moving pixels are up-weighted
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.collect_steps = 2000, 64, 40

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tr = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    te = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed + 7)
    prev, cur, objs = collect_pairs(tr, args.collect_steps, args.num_envs, args.img)
    tprev, tcur, teo = collect_pairs(te, max(8, args.collect_steps // 3), args.num_envs, args.img)
    teb = teo[:, 0]
    M = cur.shape[0]
    model = SpriteAE(K=args.slots, img=args.img, patch=args.patch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd = fair_random_baseline(teb, args.slots)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.3] device={DEVICE} | MOTION-driven SPRITE perception on Pong ({M} frames, "
          f"{args.slots} slots, motion-w {args.motion_w}) | reconstruction weighted by motion | "
          f"FAIR min-over-{args.slots} random baseline {rnd:.3f} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, M, (args.bs,), device=DEVICE)
        tgt, pre = cur[idx], prev[idx]
        recon, _, pres = model(tgt)
        motion = (tgt - pre).abs().amax(1, keepdim=True)              # moving pixels (the ball!)
        w = tgt.amax(1, keepdim=True) + args.motion_w * motion + 0.02  # bright OR moving -> weighted
        loss = (((recon - tgt) ** 2) * w).sum() / w.sum() + 0.001 * pres.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps:
            dg = ball_diag(model, tcur, teb, args.slots)
            curve.append(dict(step=step, recon=round(float(loss), 5), **dg))
            print(f"  step {step:>5} | recon {float(loss):.5f} | BALL err {dg['fixed_err']:.4f} "
                  f"(slot {dg['fixed_slot']}, stable {dg['stability']:.0%}) | per-frame {dg['perframe_err']:.4f} "
                  f"(fair-random {rnd:.3f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    decomp = multi_object_diag(model, tcur, teo, ["ball", "padL", "padR"])
    distinct = len({d["slot"] for d in decomp}) == len(decomp)
    print("  scene decomposition: " + " | ".join(
        f"{d['name']}->slot{d['slot']} err{d['fixed_err']:.3f}" for d in decomp) +
        f" | distinct-slots={distinct}", flush=True)

    final = curve[-1]
    # PRIMARY (headline, load-bearing): the 2D-moving BALL binds — fixed-slot err < 0.06 AND < 0.5xfair.
    ok = final["fixed_err"] < 0.06 and final["fixed_err"] < 0.5 * rnd
    verdict = (
        f"MOTION-DRIVEN PERCEPTION BINDS THE BALL (percept v0.3) — fixed slot {final['fixed_slot']} "
        f"tracks the 2D-moving ball at {final['fixed_err']:.3f} (fair-random {rnd:.3f}), stability "
        f"{final['stability']:.0%}. Motion weighting cracked what single-frame recon could not. "
        f"Confirm >=3 seeds + adversarial review before reporting."
        if ok else
        f"PARTIAL/CHECK — ball err {final['fixed_err']:.3f} (fair-random {rnd:.3f}), stability "
        f"{final['stability']:.0%}, per-slot {final['per_slot']}. Motion weighting insufficient.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"percept_v03_s{args.seed}.json"), "w") as f:
        json.dump(dict(fair_random=rnd, curve=curve, ok=ok, verdict=verdict,
                       decomp=decomp), f, indent=2)


if __name__ == "__main__":
    main()
