"""percept v0.5 — unsupervised object KEYPOINTS via cross-frame reconstruction.

Channel-indexed keypoints (identity stable by construction) trained to reconstruct frame x' from
appearance(x) + geometry(keypoints of x'). Validation: a FIXED keypoint channel tracks the 2D-moving
ball with low error and (by construction) high stability; scene decomposes into distinct keypoints for
ball + paddles. This is the structural fix for the identity/permanence failure of v0.1-v0.4.

Usage: python -m scripts.percept_v05_keypoints [--steps 6000] [--seed 0] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.learning.keypoints import KeypointNet
from scripts.percept_v04_temporal import collect_seq, fair_random_baseline


@torch.no_grad()
def kp_diag(model, frames, objs, K, bs=256):
    """Channel-indexed keypoint binding diagnostics over all frames."""
    L, N = frames.shape[0], frames.shape[1]
    fr = frames.reshape(L * N, *frames.shape[2:])
    ob = objs.reshape(L * N, objs.shape[-2], 2)
    pos = []
    for i in range(0, fr.shape[0], bs):
        pos.append(model.keypoints(fr[i:i + bs]))
    pos = torch.cat(pos, 0)                                  # (F,K,2)
    ball = ob[:, 0]
    d = (pos - ball.unsqueeze(1)).norm(dim=-1)               # (F,K)
    per_slot = d.mean(0)
    fixed = int(per_slot.argmin())
    me = torch.stack([(pos - ob[:, o:o + 1]).norm(dim=-1).mean(0) for o in range(3)])
    taken, decomp = set(), []
    for o in sorted(range(3), key=lambda o: float(me[o].min())):
        row = me[o].clone()
        for k in taken:
            row[k] = 1e9
        k = int(row.argmin()); taken.add(k)
        decomp.append(dict(name=["ball", "padL", "padR"][o], slot=k,
                           err=round(float((pos[:, k] - ob[:, o]).norm(dim=-1).mean()), 4)))
    decomp.sort(key=lambda d: d["name"])
    return dict(per_slot=[round(float(x), 4) for x in per_slot], fixed_slot=fixed,
                fixed_err=round(float(per_slot[fixed]), 4),
                perframe_err=round(float(d.min(1).values.mean()), 4),
                stability=round(float((d.argmin(1) == fixed).float().mean()), 3), decomp=decomp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--gap", type=int, default=2)            # frame gap between x and x' (ball motion)
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--motion-w", type=float, default=4.0)   # up-weight moving pixels (the tiny ball)
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
    tr = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    te = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed + 7)
    frames, _ = collect_seq(tr, args.collect_steps, args.num_envs, args.img)
    tef, teo = collect_seq(te, args.collect_steps, args.num_envs, args.img)
    teb = teo[:, :, 0].reshape(-1, 2)
    L, N = frames.shape[0], frames.shape[1]
    model = KeypointNet(K=args.slots, img=args.img, sigma=args.sigma).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd = fair_random_baseline(teb, args.slots)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.5] device={DEVICE} | unsupervised KEYPOINTS (cross-frame recon, gap {args.gap}) "
          f"on Pong ({L}x{N}, {args.slots} kp) | FAIR baseline {rnd:.3f} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        s0 = int(torch.randint(0, L - args.gap, (1,)))
        env_idx = torch.randint(0, N, (args.bs,), device=DEVICE)
        x_src = frames[s0][env_idx]
        x_tgt = frames[s0 + args.gap][env_idx]
        recon, _ = model(x_src, x_tgt)
        motion = (x_tgt - x_src).abs().amax(1, keepdim=True)        # moving pixels (dominated by the ball)
        w = x_tgt.amax(1, keepdim=True) + args.motion_w * motion + 0.02
        loss = (((recon - x_tgt) ** 2) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps:
            dg = kp_diag(model, tef, teo, args.slots)
            curve.append(dict(step=step, recon=round(float(loss), 5),
                              **{k: dg[k] for k in ("fixed_slot", "fixed_err", "perframe_err", "stability")}))
            print(f"  step {step:>5} | recon {float(loss):.5f} | BALL err {dg['fixed_err']:.4f} "
                  f"(kp {dg['fixed_slot']}, stable {dg['stability']:.0%}) | per-frame {dg['perframe_err']:.4f} "
                  f"(fair {rnd:.3f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    dg = kp_diag(model, tef, teo, args.slots)
    print("  scene decomposition: " + " | ".join(
        f"{d['name']}->kp{d['slot']} err{d['err']:.3f}" for d in dg["decomp"]) +
        f" | distinct={len({d['slot'] for d in dg['decomp']})==3}", flush=True)
    ok = dg["fixed_err"] < 0.06 and dg["fixed_err"] < 0.5 * rnd and dg["stability"] > 0.8
    verdict = (
        f"UNSUPERVISED KEYPOINTS BIND THE BALL (percept v0.5) — fixed keypoint {dg['fixed_slot']} tracks "
        f"the 2D-moving ball at {dg['fixed_err']:.3f} (fair {rnd:.3f}), stability {dg['stability']:.0%}. "
        f"Channel-indexed identity cracked the permanence problem. Confirm >=3 seeds + 2nd game + review."
        if ok else
        f"PARTIAL/CHECK — ball err {dg['fixed_err']:.3f} (fair {rnd:.3f}), stability {dg['stability']:.0%}, "
        f"per-kp {dg['per_slot']}. Keypoints not binding the ball on this config.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"percept_v05_s{args.seed}.json"), "w") as f:
        json.dump(dict(fair_random=rnd, curve=curve, ok=ok, verdict=verdict, final=dg), f, indent=2)


if __name__ == "__main__":
    main()
