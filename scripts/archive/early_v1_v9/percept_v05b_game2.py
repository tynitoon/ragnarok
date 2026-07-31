"""percept v0.5b — GENERALITY: do unsupervised keypoints bind the moving ball on a DIFFERENT game?

Same KeypointNet + motion-weighted cross-frame recon as v0.5, but on Breakout (2D-moving ball, but a
different layout: bricks, walls, different paddle) and optionally Flappy (1D bird under gravity). If a
fixed keypoint channel tracks the mover on a game it was never tuned for, the perception is GENERAL,
not Pong-specific. Validation tracks only the ball/mover (the load-bearing object).

Usage: python -m scripts.percept_v05b_game2 --env breakout [--seed 0] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.flappy import DeviceVecFlappy
from ragnarok.learning.keypoints import KeypointNet
from scripts.percept_v04_temporal import fair_random_baseline


@torch.no_grad()
def collect_mover(env, L, N, img, mover_x=None):
    """Roll env; return ordered frames (L,N,3,H,W) + mover (ball/bird) position (L*N,2)."""
    frames, mov = [], []
    for _ in range(L):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        frames.append(env.state.view(N, 3, img, img).clone())
        bx = env.bx if hasattr(env, "bx") else torch.full_like(env.by, mover_x)
        mov.append(torch.stack([bx, env.by], -1))
    return torch.stack(frames), torch.cat(mov, 0)


@torch.no_grad()
def mover_diag(model, frames, mover, K, bs=256):
    L, N = frames.shape[0], frames.shape[1]
    fr = frames.reshape(L * N, *frames.shape[2:])
    pos = []
    for i in range(0, fr.shape[0], bs):
        pos.append(model.keypoints(fr[i:i + bs]))
    pos = torch.cat(pos, 0)                                  # (F,K,2)
    d = (pos - mover.unsqueeze(1)).norm(dim=-1)              # (F,K)
    per = d.mean(0)
    fixed = int(per.argmin())
    return dict(per_kp=[round(float(x), 4) for x in per], fixed_slot=fixed,
                fixed_err=round(float(per[fixed]), 4),
                perframe_err=round(float(d.min(1).values.mean()), 4),
                stability=round(float((d.argmin(1) == fixed).float().mean()), 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["breakout", "flappy"], default="breakout")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--motion-w", type=float, default=4.0)
    p.add_argument("--collect-steps", type=int, default=80)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.collect_steps = 2500, 64, 60

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Env = {"breakout": DeviceVecBreakout, "flappy": DeviceVecFlappy}[args.env]
    mx = 0.3 if args.env == "flappy" else None                  # flappy bird x is fixed
    tr = Env(args.num_envs, img=args.img, seed=args.seed)
    te = Env(args.num_envs, img=args.img, seed=args.seed + 7)
    frames, _ = collect_mover(tr, args.collect_steps, args.num_envs, args.img, mx)
    tef, tem = collect_mover(te, args.collect_steps, args.num_envs, args.img, mx)
    L, N = frames.shape[0], frames.shape[1]
    model = KeypointNet(K=args.slots, img=args.img, sigma=args.sigma).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd = fair_random_baseline(tem, args.slots)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.5b] device={DEVICE} | GENERALITY: keypoints on {args.env.upper()} "
          f"({L}x{N}, {args.slots} kp, gap {args.gap}) | FAIR baseline {rnd:.3f} | steps {args.steps}",
          flush=True)
    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        s0 = int(torch.randint(0, L - args.gap, (1,)))
        idx = torch.randint(0, N, (args.bs,), device=DEVICE)
        x_src, x_tgt = frames[s0][idx], frames[s0 + args.gap][idx]
        recon, _ = model(x_src, x_tgt)
        motion = (x_tgt - x_src).abs().amax(1, keepdim=True)
        w = x_tgt.amax(1, keepdim=True) + args.motion_w * motion + 0.02
        loss = (((recon - x_tgt) ** 2) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            dg = mover_diag(model, tef, tem, args.slots)
            curve.append(dict(step=step, recon=round(float(loss), 5), **dg))
            print(f"  step {step:>5} | recon {float(loss):.5f} | MOVER err {dg['fixed_err']:.4f} "
                  f"(kp {dg['fixed_slot']}, stable {dg['stability']:.0%}) | per-frame {dg['perframe_err']:.4f} "
                  f"(fair {rnd:.3f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]
    ok = final["fixed_err"] < 0.06 and final["fixed_err"] < 0.5 * rnd and final["stability"] > 0.8
    verdict = (f"GENERAL ({args.env}) — keypoint {final['fixed_slot']} tracks the mover at "
               f"{final['fixed_err']:.3f} (fair {rnd:.3f}), stable {final['stability']:.0%}."
               if ok else
               f"PARTIAL ({args.env}) — mover err {final['fixed_err']:.3f} (fair {rnd:.3f}), stable "
               f"{final['stability']:.0%}, per-kp {final['per_kp']}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"percept_v05b_{args.env}_s{args.seed}.json"), "w") as f:
        json.dump(dict(env=args.env, fair_random=rnd, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
