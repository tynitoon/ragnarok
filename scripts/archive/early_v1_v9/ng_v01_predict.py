"""v0.1 of the NOTION GRAPH (NOTION_GRAPH_DESIGN.md): can the agent learn to PREDICT
next-frame pixels ONLY by composing a SMALL, growing-then-plateauing library of local
dynamics notions? On Pong (clear local dynamics: ball moves/bounces, paddles slide).

Core success signs (v0.1): prediction error DROPS (competence) while the notion count
GROWS then PLATEAUS at a small number (a few notions compose to explain the world) ->
the compositional-compression core works on pixels. (Forced cross-world reuse = v0.2.)

Usage: python -m scripts.ng_v01_predict [--steps 4000] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.learning.notion_graph import NotionLibrary, patchify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--k-max", type=int, default=24)
    p.add_argument("--mint-every", type=int, default=40)
    p.add_argument("--mint-tol", type=float, default=0.010)   # top-decile patch MSE to keep minting
    p.add_argument("--prune-every", type=int, default=200)
    p.add_argument("--warmup", type=int, default=100)         # no mint/prune before this
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.num_envs = 600, 64

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = DeviceVecPong(args.num_envs, img=args.img, seed=args.seed)
    A = env.action_dim
    P_dim = 3 * args.patch * args.patch
    ctx_dim = 2 * P_dim + A
    lib = NotionLibrary(ctx_dim, P_dim, hidden=args.hidden, k_init=1, k_max=args.k_max)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[NG v0.1] device={DEVICE} | Pong {args.img}px patch {args.patch} | "
          f"ctx_dim {ctx_dim} patch_dim {P_dim} | learn-by-composing-notions | "
          f"steps {args.steps}", flush=True)
    t0 = time.perf_counter()

    def frame():
        return env.state.view(args.num_envs, 3, args.img, args.img)

    prev = frame()
    env.step(torch.randint(0, A, (args.num_envs,), device=DEVICE))
    cur = frame()
    topema = 1.0
    curve = []
    for step in range(1, args.steps + 1):
        act = torch.randint(0, A, (args.num_envs,), device=DEVICE)
        env.step(act)
        nxt = frame()

        pv, gh, gw = patchify(prev, args.patch)
        cv, _, _ = patchify(cur, args.patch)
        nx, _, _ = patchify(nxt, args.patch)
        G = gh * gw
        act_oh = torch.zeros(args.num_envs, A, device=DEVICE)
        act_oh[torch.arange(args.num_envs, device=DEVICE), act] = 1.0
        act_b = act_oh.unsqueeze(1).expand(-1, G, -1)
        ctx = torch.cat([pv, cv, act_b], -1).reshape(-1, ctx_dim)
        target = nx.reshape(-1, P_dim)

        loss, min_err, assign = lib.learn(ctx, target)

        # dynamic MSE: error on patches that actually CHANGED (the meaningful signal)
        with torch.no_grad():
            changed = ((nx - cv) ** 2).mean(-1).reshape(-1) > 1e-5
            dyn_mse = float(min_err[changed].mean()) if bool(changed.any()) else 0.0
            topk = max(1, min_err.numel() // 10)
            top_err = float(min_err.topk(topk).values.mean())
            topema = 0.9 * topema + 0.1 * top_err

        # MINT on persistent surprise; PRUNE disused notions (compression)
        if step > args.warmup and step % args.mint_every == 0 and topema > args.mint_tol:
            with torch.no_grad():
                ti = min_err.topk(topk).indices
                lib.mint(target_seed=target[ti])
        if step > args.warmup and step % args.prune_every == 0:
            lib.prune(min_usage=1e-3)

        prev, cur = cur, nxt
        if step % args.log_every == 0 or step == args.steps:
            row = dict(step=step, mse=round(loss, 5), dyn_mse=round(dyn_mse, 5),
                       top_mse=round(topema, 5), K=lib.K)
            curve.append(row)
            print(f"  step {step:>5} | mse {loss:.5f} | dyn_mse {dyn_mse:.5f} | "
                  f"top10% {topema:.5f} | notions K={lib.K} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    first = curve[0] if curve else dict(dyn_mse=1.0)
    last = curve[-1]
    drop = (first["dyn_mse"] - last["dyn_mse"]) / max(first["dyn_mse"], 1e-6)
    # plateau check: K over the last third should be roughly stable + small
    ks = [r["K"] for r in curve]
    plateaued = len(ks) >= 3 and (max(ks[-3:]) - min(ks[-3:])) <= 2 and ks[-1] <= args.k_max - 2
    ok = drop > 0.4 and plateaued and last["K"] >= 2
    verdict = (
        f"NG CORE WORKS (v0.1) — dynamic prediction error fell {drop*100:.0f}% while the "
        f"notion library grew then PLATEAUED at K={last['K']} (a few reused notions compose "
        f"to explain Pong's dynamics). The compositional-compression core runs on pixels. "
        f"Next: v0.2 forced cross-world reuse." if ok else
        f"PARTIAL/CHECK — dyn_mse drop {drop*100:.0f}%, K={last['K']} plateaued={plateaued}. "
        f"Tune minting/pruning drives.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ng_v01.json"), "w") as f:
        json.dump(dict(curve=curve, drop=drop, plateaued=plateaued, ok=ok,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
