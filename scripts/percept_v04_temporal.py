"""percept v0.4 — TEMPORAL object permanence: does a carried slot stably TRACK the moving ball?

v0.1-v0.3 showed feed-forward perception binds position-stable objects (paddles) but hands the
roaming ball between slots (no permanence). v0.4 processes VIDEO with a recurrent slot tracker
(carry slot state across frames). Trained self-supervised by reconstructing each frame. Validation:
the SAME fixed slot should track the 2D-moving ball across the whole sequence (stability high, err low).

Usage: python -m scripts.percept_v04_temporal [--steps 4000] [--seed 0] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.learning.recurrent_slots import RecurrentSpriteTracker


@torch.no_grad()
def collect_seq(env, L, N, img):
    """Roll env L steps; return ordered frames (L,N,3,H,W) + object positions (L,N,3,2)."""
    frames, objs = [], []
    for _ in range(L):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        frames.append(env.state.view(N, 3, img, img).clone())
        xa = torch.full_like(env.bx, env.x_a); xo = torch.full_like(env.bx, env.x_o)
        ball = torch.stack([env.bx, env.by], -1)
        padl = torch.stack([xa, env.pad_a], -1); padr = torch.stack([xo, env.pad_o], -1)
        objs.append(torch.stack([ball, padl, padr], 1))
    return torch.stack(frames), torch.stack(objs)


@torch.no_grad()
def fair_random_baseline(balls, K, trials=4000):
    tgt = balls[torch.randint(0, balls.shape[0], (trials,), device=DEVICE)]
    pts = torch.rand(trials, K, 2, device=DEVICE)
    return float((pts - tgt.unsqueeze(1)).norm(dim=-1).min(1).values.mean())


@torch.no_grad()
def eval_track(model, frames, objs, T):
    """Run the tracker over non-overlapping windows; report ball binding over all (t,n) frames."""
    L, N = frames.shape[0], frames.shape[1]
    allpos, allobj = [], []
    for t0 in range(0, L - T + 1, T):
        _, pos, _ = model(frames[t0:t0 + T])                   # (T,N,K,2)
        allpos.append(pos.reshape(-1, pos.shape[-2], 2))       # (T*N,K,2)
        allobj.append(objs[t0:t0 + T].reshape(-1, objs.shape[-2], 2))
    pos = torch.cat(allpos, 0)                                 # (F,K,2)
    obj = torch.cat(allobj, 0)                                 # (F,3,2)
    ball = obj[:, 0]
    d = (pos - ball.unsqueeze(1)).norm(dim=-1)                 # (F,K)
    per_slot = d.mean(0)
    fixed = int(per_slot.argmin())
    # distinct-slot decomposition for ball + 2 paddles
    me = torch.stack([(pos - obj[:, o:o + 1]).norm(dim=-1).mean(0) for o in range(3)])  # (3,K)
    taken, decomp = set(), []
    for o in sorted(range(3), key=lambda o: float(me[o].min())):
        row = me[o].clone()
        for k in taken:
            row[k] = 1e9
        k = int(row.argmin()); taken.add(k)
        decomp.append(dict(name=["ball", "padL", "padR"][o], slot=k,
                           err=round(float((pos[:, k] - obj[:, o]).norm(dim=-1).mean()), 4)))
    decomp.sort(key=lambda d: d["name"])
    return dict(per_slot=[round(float(x), 4) for x in per_slot], fixed_slot=fixed,
                fixed_err=round(float(per_slot[fixed]), 4),
                perframe_err=round(float(d.min(1).values.mean()), 4),
                stability=round(float((d.argmin(1) == fixed).float().mean()), 3),
                decomp=decomp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--patch", type=int, default=10)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--collect-steps", type=int, default=64)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--motion-w", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.collect_steps = 1500, 64, 48

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tr = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    te = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed + 7)
    frames, _ = collect_seq(tr, args.collect_steps, args.num_envs, args.img)
    tef, teo = collect_seq(te, args.collect_steps, args.num_envs, args.img)
    teb = teo[:, :, 0].reshape(-1, 2)                          # all ball positions
    L, N = frames.shape[0], frames.shape[1]
    T = args.seq
    model = RecurrentSpriteTracker(num_slots=args.slots, dim=args.dim, img=args.img,
                                   patch=args.patch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd = fair_random_baseline(teb, args.slots)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.4] device={DEVICE} | TEMPORAL permanence tracker on Pong video "
          f"(L={L} x N={N}, T={T}, {args.slots} slots) | self-supervised recon | FAIR baseline "
          f"{rnd:.3f} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        s0 = int(torch.randint(0, L - T + 1, (1,)))
        env_idx = torch.randint(0, N, (args.bs,), device=DEVICE)
        seq = frames[s0:s0 + T][:, env_idx]                    # (T,bs,3,H,W)
        recon, _, pres = model(seq)
        motion = (seq[1:] - seq[:-1]).abs().amax(2, keepdim=True)            # (T-1,bs,1,H,W)
        motion = torch.cat([torch.zeros_like(motion[:1]), motion], 0)        # pad t=0
        w = seq.amax(2, keepdim=True) + args.motion_w * motion + 0.02
        loss = (((recon - seq) ** 2) * w).sum() / w.sum() + 0.001 * pres.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            dg = eval_track(model, tef, teo, T)
            curve.append(dict(step=step, recon=round(float(loss), 5), **{k: dg[k] for k in
                         ("fixed_slot", "fixed_err", "perframe_err", "stability")}))
            print(f"  step {step:>5} | recon {float(loss):.5f} | BALL err {dg['fixed_err']:.4f} "
                  f"(slot {dg['fixed_slot']}, stable {dg['stability']:.0%}) | per-frame "
                  f"{dg['perframe_err']:.4f} (fair {rnd:.3f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    dg = eval_track(model, tef, teo, T)
    print("  scene decomposition: " + " | ".join(
        f"{d['name']}->slot{d['slot']} err{d['err']:.3f}" for d in dg["decomp"]) +
        f" | distinct={len({d['slot'] for d in dg['decomp']})==3}", flush=True)
    ok = dg["fixed_err"] < 0.06 and dg["fixed_err"] < 0.5 * rnd and dg["stability"] > 0.8
    verdict = (
        f"TEMPORAL PERMANENCE BINDS THE BALL (percept v0.4) — a carried slot ({dg['fixed_slot']}) "
        f"tracks the 2D-moving ball at {dg['fixed_err']:.3f} (fair {rnd:.3f}), stability "
        f"{dg['stability']:.0%}. Permanence cracked what feed-forward could not. Confirm >=3 seeds + review."
        if ok else
        f"PARTIAL/CHECK — ball err {dg['fixed_err']:.3f} (fair {rnd:.3f}), stability {dg['stability']:.0%}, "
        f"per-slot {dg['per_slot']}. Temporal carry insufficient on this config.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"percept_v04_s{args.seed}.json"), "w") as f:
        json.dump(dict(fair_random=rnd, curve=curve, ok=ok, verdict=verdict, final=dg), f, indent=2)


if __name__ == "__main__":
    main()
