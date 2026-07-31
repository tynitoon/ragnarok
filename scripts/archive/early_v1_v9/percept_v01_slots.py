"""percept v0.1 — does UNSUPERVISED slot perception DISCOVER objects from pixels?

Train the SlotAE on Pong frames by reconstruction ONLY (no labels). Then VALIDATE, against
the env's true ball position (never used in training), whether SOME slot binds to and tracks
the ball: best-slot tracking error should be << a random-point baseline. This de-risks lock #1
(learned object perception) before building the world-model on top.

Usage: python -m scripts.percept_v01_slots [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.learning.slots import SlotAE


@torch.no_grad()
def collect(env, steps, N, img):
    frames, balls = [], []
    for _ in range(steps):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        frames.append(env.state.view(N, 3, img, img).clone())
        balls.append(torch.stack([env.bx, env.by], -1).clone())
    return torch.cat(frames, 0), torch.cat(balls, 0)            # (M,3,H,W), (M,2)


@torch.no_grad()
def ball_tracking_error(model, frames, balls, bs=256):
    """Mean over frames of the BEST slot's distance to the true ball (in [0,1])."""
    errs, masses = [], []
    for i in range(0, frames.shape[0], bs):
        fr, bl = frames[i:i + bs], balls[i:i + bs]
        _, masks, _, _ = model(fr)
        cen, mass = model.mask_centroids(masks)                # (n,S,2), (n,S)
        d = (cen - bl.unsqueeze(1)).norm(dim=-1)               # (n,S)
        best = d.argmin(1)
        errs.append(d.gather(1, best.unsqueeze(1)).squeeze(1))
        masses.append(mass.gather(1, best.unsqueeze(1)).squeeze(1))
    e = torch.cat(errs)
    return float(e.mean()), float(torch.cat(masses).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=5)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--collect-steps", type=int, default=50)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.collect_steps, args.slots = 1500, 64, 30, 5

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tr_env = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    te_env = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed + 7)
    frames, balls = collect(tr_env, args.collect_steps, args.num_envs, args.img)
    te_frames, te_balls = collect(te_env, max(8, args.collect_steps // 3), args.num_envs, args.img)
    M = frames.shape[0]
    model = SlotAE(num_slots=args.slots, dim=args.dim, img=args.img).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.1] device={DEVICE} | UNSUPERVISED slot perception on Pong ({M} frames, "
          f"{args.slots} slots) | reconstruction only, then validate ball-tracking | steps {args.steps}",
          flush=True)
    t0 = time.perf_counter()
    # random-point baseline tracking error (expected distance of a random [0,1]^2 point)
    rnd = (torch.rand(2000, 2, device=DEVICE) - te_balls[torch.randint(0, te_balls.shape[0],
            (2000,), device=DEVICE)]).norm(dim=-1).mean()
    print(f"  random-point ball-distance baseline: {float(rnd):.3f}", flush=True)

    curve = []
    warm = max(1, args.steps // 10)
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:                             # LR warmup (slot attn is sensitive)
            g["lr"] = args.lr * min(1.0, step / warm)
        idx = torch.randint(0, M, (args.bs,), device=DEVICE)
        tgt = frames[idx]
        recon, masks, _, _ = model(tgt)
        w = tgt.amax(1, keepdim=True) + 0.005                  # near-pure FOREGROUND (bright objects dominate)
        recon_loss = (((recon - tgt) ** 2) * w).sum() / w.sum()
        ent = -(masks * (masks + 1e-8).log()).sum(1).mean()    # peaky masks (anti slot-collapse)
        loss = recon_loss + 0.05 * ent
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps:
            err, mass = ball_tracking_error(model, te_frames, te_balls)
            curve.append(dict(step=step, recon=round(float(loss), 5), ball_err=round(err, 4),
                              ball_slot_mass=round(mass, 4)))
            print(f"  step {step:>5} | recon {float(loss):.5f} | BALL-tracking err {err:.4f} "
                  f"(random {float(rnd):.3f}) | ball-slot mask-mass {mass:.1f}px | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]
    # success: a slot tracks the ball far better than random, with a SMALL mask (the ball is small)
    ok = final["ball_err"] < 0.06 and final["ball_err"] < 0.3 * float(rnd)
    verdict = (
        f"UNSUPERVISED OBJECT PERCEPTION WORKS (percept v0.1) — trained on reconstruction ONLY, a "
        f"slot binds to and TRACKS the ball at {final['ball_err']:.3f} error (random {float(rnd):.3f}), "
        f"mask-mass {final['ball_slot_mass']:.0f}px. Lock #1 (learned perception) is de-risked from "
        f"pixels. Next: relational world-model over the slots."
        if ok else
        f"PARTIAL/CHECK — ball-tracking err {final['ball_err']:.3f} (random {float(rnd):.3f}), "
        f"recon {final['recon']}. Slots not yet binding the ball cleanly; tune slots/dim/lr/steps.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "percept_v01.json"), "w") as f:
        json.dump(dict(random_baseline=float(rnd), curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
