"""percept v0.2 — object perception via an EXPLICIT SPRITE autoencoder (more robust than slots).

Slot-attention (v0.1) did not bind objects on our sparse sprite scenes. Here each of K slots is
EXPLICITLY an object = (position x,y ; presence ; a small appearance patch), and a differentiable
spatial-transformer renderer pastes each patch at its position onto the canvas; the composite must
reconstruct the frame. The position BOTTLENECK forces each slot to localise to one object. Trained
by reconstruction ONLY (no labels). Validation (unsupervised): a slot's predicted POSITION tracks
the true ball far better than random -> learned object perception, de-risking lock #1.

Usage: python -m scripts.percept_v02_sprites [--steps 6000] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong


class SpriteAE(nn.Module):
    def __init__(self, K=4, img=48, patch=14, dim=256):
        super().__init__()
        self.K, self.img, self.patch = K, img, patch
        self.scale = patch / img
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),               # 48->24
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),             # 24->12
            nn.Conv2d(64, 128, 5, 2, 2), nn.ReLU())            # 12->6
        self.head = nn.Sequential(nn.Linear(128 * 6 * 6, dim), nn.ReLU())
        self.pos = nn.Linear(dim, K * 2)
        self.pres = nn.Linear(dim, K)
        self.app = nn.Linear(dim, K * 3 * patch * patch)

    def forward(self, x):
        N = x.shape[0]
        h = self.head(self.enc(x).reshape(N, -1))
        pos = self.pos(h).reshape(N, self.K, 2).sigmoid()       # (N,K,2) in [0,1]
        pres = self.pres(h).reshape(N, self.K).sigmoid()        # (N,K)
        app = self.app(h).reshape(N * self.K, 3, self.patch, self.patch).sigmoid()
        # render: place each patch at its position via a write spatial transformer
        cx = (2 * pos[..., 0] - 1).reshape(-1)                  # (N*K,)
        cy = (2 * pos[..., 1] - 1).reshape(-1)
        s = self.scale
        z = torch.zeros_like(cx)
        theta = torch.stack([torch.stack([torch.full_like(cx, 1 / s), z, -cx / s], -1),
                             torch.stack([z, torch.full_like(cx, 1 / s), -cy / s], -1)], 1)
        grid = F.affine_grid(theta, (N * self.K, 3, self.img, self.img), align_corners=False)
        placed = F.grid_sample(app, grid, align_corners=False, padding_mode="zeros")
        placed = placed.reshape(N, self.K, 3, self.img, self.img)
        recon = (placed * pres[:, :, None, None, None]).sum(1).clamp(0, 1)
        return recon, pos, pres


@torch.no_grad()
def collect(env, steps, N, img):
    """Return frames + true object positions (ball, left paddle, right paddle) -> (M,3,2)."""
    frames, objs = [], []
    for _ in range(steps):
        env.step(torch.randint(0, env.action_dim, (N,), device=DEVICE))
        frames.append(env.state.view(N, 3, img, img).clone())
        xa = torch.full_like(env.bx, env.x_a)
        xo = torch.full_like(env.bx, env.x_o)
        ball = torch.stack([env.bx, env.by], -1)
        padl = torch.stack([xa, env.pad_a], -1)
        padr = torch.stack([xo, env.pad_o], -1)
        objs.append(torch.stack([ball, padl, padr], 1))        # (N,3,2)
    return torch.cat(frames, 0), torch.cat(objs, 0)


@torch.no_grad()
def ball_diag(model, frames, balls, K, bs=256):
    """Rigorous binding diagnostics (guards against the min-over-K trap + unstable binding).

    Returns dict with:
      per_slot[k]   = mean over frames of |pos_k - ball|         (which fixed slot owns the ball?)
      fixed_err     = min_k per_slot[k]  -> the BEST FIXED slot's mean error (STABLE binding)
      perframe_err  = mean_t min_k |pos_kt - ball|               (per-frame best, optimistic)
      stability     = frac of frames whose per-frame argmin == the fixed best slot
    """
    allpos = []
    for i in range(0, frames.shape[0], bs):
        _, pos, _ = model(frames[i:i + bs])
        allpos.append(pos)
    pos = torch.cat(allpos, 0)                                   # (M,K,2)
    d = (pos - balls.unsqueeze(1)).norm(dim=-1)                  # (M,K)
    per_slot = d.mean(0)                                         # (K,)
    fixed_slot = int(per_slot.argmin())
    fixed_err = float(per_slot[fixed_slot])
    perframe_err = float(d.min(1).values.mean())
    stability = float((d.argmin(1) == fixed_slot).float().mean())
    return dict(per_slot=[round(float(x), 4) for x in per_slot], fixed_slot=fixed_slot,
                fixed_err=round(fixed_err, 4), perframe_err=round(perframe_err, 4),
                stability=round(stability, 3))


@torch.no_grad()
def fair_random_baseline(balls, K, trials=4000):
    """Expected MIN-over-K-random-points distance to the ball (matches the min-over-K metric)."""
    tgt = balls[torch.randint(0, balls.shape[0], (trials,), device=DEVICE)]   # (T,2)
    pts = torch.rand(trials, K, 2, device=DEVICE)                              # (T,K,2)
    return float((pts - tgt.unsqueeze(1)).norm(dim=-1).min(1).values.mean())


@torch.no_grad()
def multi_object_diag(model, frames, objs, names, bs=256):
    """Greedy DISTINCT-slot assignment: does the scene decompose into separate slots per object?

    objs: (M,O,2) true positions. Assign each object to a distinct slot (greedy on mean error).
    Returns per-object (slot, fixed_err, stability) — distinct slots => real scene decomposition.
    """
    allpos = []
    for i in range(0, frames.shape[0], bs):
        _, pos, _ = model(frames[i:i + bs])
        allpos.append(pos)
    pos = torch.cat(allpos, 0)                                   # (M,K,2)
    O, K = objs.shape[1], pos.shape[1]
    # mean error of every (object, slot) pair
    me = torch.stack([(pos - objs[:, o:o + 1]).norm(dim=-1).mean(0) for o in range(O)])  # (O,K)
    taken, out = set(), []
    order = sorted(range(O), key=lambda o: float(me[o].min()))   # assign easiest objects first
    for o in order:
        row = me[o].clone()
        for k in taken:
            row[k] = 1e9
        k = int(row.argmin()); taken.add(k)
        d = (pos[:, k] - objs[:, o]).norm(dim=-1)                # per-frame err for this slot
        out.append((o, dict(name=names[o], slot=k, fixed_err=round(float(d.mean()), 4))))
    out.sort()
    return [v for _, v in out]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--collect-steps", type=int, default=60)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs, args.collect_steps = 2000, 64, 30

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tr = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    te = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed + 7)
    frames, objs = collect(tr, args.collect_steps, args.num_envs, args.img)
    tef, teo = collect(te, max(8, args.collect_steps // 3), args.num_envs, args.img)
    teb = teo[:, 0]                                             # ball positions (primary)
    M = frames.shape[0]
    model = SpriteAE(K=args.slots, img=args.img, patch=args.patch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rnd = fair_random_baseline(teb, args.slots)                 # FAIR: min over K random points

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[percept v0.2] device={DEVICE} | EXPLICIT SPRITE perception on Pong ({M} frames, "
          f"{args.slots} slots) | reconstruction only, validate ball-tracking via slot POSITION | "
          f"FAIR min-over-{args.slots} random baseline {rnd:.3f} | steps {args.steps}", flush=True)
    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, M, (args.bs,), device=DEVICE)
        tgt = frames[idx]
        recon, _, pres = model(tgt)
        w = tgt.amax(1, keepdim=True) + 0.02
        loss = (((recon - tgt) ** 2) * w).sum() / w.sum() + 0.001 * pres.mean()  # mild presence sparsity
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps:
            dg = ball_diag(model, tef, teb, args.slots)
            curve.append(dict(step=step, recon=round(float(loss), 5), **dg))
            print(f"  step {step:>5} | recon {float(loss):.5f} | FIXED-slot err {dg['fixed_err']:.4f} "
                  f"(slot {dg['fixed_slot']}, stable {dg['stability']:.0%}) | per-frame {dg['perframe_err']:.4f} "
                  f"(fair-random {rnd:.3f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    # SECONDARY: does the scene decompose into distinct slots for ball + the two paddles?
    decomp = multi_object_diag(model, tef, teo, ["ball", "padL", "padR"])
    distinct = len({d["slot"] for d in decomp}) == len(decomp)
    print("  scene decomposition (distinct-slot assignment): " +
          " | ".join(f"{d['name']}->slot{d['slot']} err{d['fixed_err']:.3f}" for d in decomp) +
          f" | distinct-slots={distinct}", flush=True)

    final = curve[-1]
    # success demands a STABLE binding: one FIXED slot tracks the ball, well below the FAIR baseline,
    # and is the per-frame best slot most of the time (a real object representation, not a min-trick).
    ok = (final["fixed_err"] < 0.5 * rnd and final["fixed_err"] < 0.06 and final["stability"] > 0.8)
    verdict = (
        f"SPRITE OBJECT PERCEPTION WORKS (percept v0.2) — reconstruction ONLY, FIXED slot {final['fixed_slot']} "
        f"tracks the ball at {final['fixed_err']:.3f} (fair-random {rnd:.3f}), stable {final['stability']:.0%} "
        f"of frames. A STABLE learned object representation from pixels; lock #1 de-risked. Next: relational "
        f"world-model over the slots. REVIEW before reporting."
        if ok else
        f"PARTIAL/CHECK — fixed-slot err {final['fixed_err']:.3f} (fair-random {rnd:.3f}), stability "
        f"{final['stability']:.0%}, per-frame {final['perframe_err']:.3f}, per-slot {final['per_slot']}. "
        f"If fixed_err>>per_frame or stability low, binding is not stable; tune slots/patch/scale/lr.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "percept_v02.json"), "w") as f:
        json.dump(dict(fair_random=rnd, curve=curve, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
