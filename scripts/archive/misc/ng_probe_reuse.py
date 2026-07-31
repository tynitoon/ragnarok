"""ADVERSARIAL RE-RUN of NG v0.2 forced-reuse, with a SWAPPABLE second world and
extra controls, to test whether 'cross-world reuse' survives DISSIMILARITY and is
not an artifact of the mint/prune schedule.

Mirrors scripts/ng_v02_reuse.py's loop EXACTLY (same mint_tol, mint_every, prune,
dyn metric, reuse counting) but:
  * --world2 {breakout, snake}: the second world. Snake = dissimilar (grid food,
    no smooth bouncing ball, no paddle). Breakout = the original (near-identical).
  * action one-hot padded to width AMAX=4 so a Pong (A=3) library transfers to
    Snake (A=4) with matching ctx_dim (clean, no leakage; the extra action bit is
    just always-0 for Pong).
  * Extra arm 'shuffled': warm library after Pong, but every notion's predictor
    weights are RANDOM-REINITIALISED (K kept). Tests attack #4: if 'shuffled' (same
    K, same schedule head-start, NO real Pong knowledge) reuses ~as well as 'warm',
    then the gain is mechanical (head-start in K), not transferred knowledge.

Usage: python -m scripts.ng_probe_reuse --world2 snake --seeds 0 1 2
       python -m scripts.ng_probe_reuse --world2 breakout --seeds 0 1 2   # replicate
"""
import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake
from ragnarok.learning.notion_graph import NotionLibrary, patchify

AMAX = 4  # padded action one-hot width (Pong/Breakout=3 used, Snake=4 used)


def make_world(name, n, img, seed):
    if name == "pong":
        return DeviceVecPong(n, img=img, seed=seed), 3
    if name == "breakout":
        return DeviceVecBreakout(n, img=img, seed=seed), 3
    if name == "snake":
        return DeviceVecSnake(n, img=img, seed=seed), 4
    raise ValueError(name)


def run_world(lib, env, A, steps, cfg, allow_mint, allow_prune, base_K, log_every=200):
    """IDENTICAL loop to ng_v02_reuse.run_world, but action one-hot padded to AMAX."""
    N, img, P_dim, patch = cfg["num_envs"], cfg["img"], cfg["P_dim"], cfg["patch"]
    ctx_dim = lib.ctx_dim
    ar = torch.arange(N, device=DEVICE)

    def frame():
        return env.state.view(N, 3, img, img)

    prev = frame()
    env.step(torch.randint(0, A, (N,), device=DEVICE))
    cur = frame()
    topema, minted, r_hit, r_tot, curve = 0.02, 0, 0, 0, []
    for step in range(1, steps + 1):
        act = torch.randint(0, A, (N,), device=DEVICE)
        env.step(act)
        nxt = frame()
        pv, gh, gw = patchify(prev, patch)
        cv, _, _ = patchify(cur, patch)
        nx, _, _ = patchify(nxt, patch)
        G = gh * gw
        aoh = torch.zeros(N, AMAX, device=DEVICE)
        aoh[ar, act] = 1.0
        ctx = torch.cat([pv, cv, aoh.unsqueeze(1).expand(-1, G, -1)], -1).reshape(-1, ctx_dim)
        target = nx.reshape(-1, P_dim)
        loss, min_err, assign = lib.learn(ctx, target)
        with torch.no_grad():
            changed = ((nx - cv) ** 2).mean(-1).reshape(-1) > 1e-5
            dyn = float(min_err[changed].mean()) if bool(changed.any()) else 0.0
            topk = max(1, min_err.numel() // 10)
            topema = 0.9 * topema + 0.1 * float(min_err.topk(topk).values.mean())
            if bool(changed.any()):
                r_hit += int((assign[changed] < base_K).sum())
                r_tot += int(changed.sum())
        if allow_mint and step % cfg["mint_every"] == 0 and topema > cfg["mint_tol"] \
                and lib.K < lib.k_max:
            ti = min_err.topk(topk).indices
            if lib.mint(target_seed=target[ti]):
                minted += 1
        if allow_prune and step % cfg["prune_every"] == 0:
            lib.prune(1e-3)
        prev, cur = cur, nxt
        if step % log_every == 0 or step == steps:
            curve.append(dict(step=step, dyn=round(dyn, 5), K=lib.K))
    return curve, minted, (r_hit / max(1, r_tot))


def new_lib(cfg, k_max, hidden):
    # ctx_dim uses AMAX (padded action), not 3
    return NotionLibrary(cfg["P_dim"] * 2 + AMAX, cfg["P_dim"], hidden=hidden,
                         k_init=1, k_max=k_max)


@torch.no_grad()
def reinit_predictors(lib):
    """Randomly re-initialise EVERY notion's weights (keep K, usage). Destroys the
    Pong-learned function but preserves the library SIZE & mint head-start."""
    lib.W1.data = lib._w(lib.K, lib.ctx_dim, lib.hidden)
    lib.b1.data = torch.zeros(lib.K, lib.hidden, device=DEVICE)
    lib.W2.data = lib._w(lib.K, lib.hidden, lib.patch_dim)
    lib.b2.data = torch.zeros(lib.K, lib.patch_dim, device=DEVICE)
    lib._mk_opt()


def arm(cfg, seed, k_max, hidden, label, world2, shuffled=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    lib = new_lib(cfg, k_max, hidden)
    pong, Ap = make_world("pong", cfg["num_envs"], cfg["img"], seed)
    run_world(lib, pong, Ap, cfg["pong_steps"], cfg, allow_mint=k_max > 1,
              allow_prune=k_max > 1, base_K=lib.K)
    k_after_pong = lib.K
    if shuffled:
        reinit_predictors(lib)        # keep K, destroy knowledge
    w2env, A2 = make_world(world2, cfg["num_envs"], cfg["img"], seed)
    warm_curve, warm_mint, warm_reuse = run_world(
        lib, w2env, A2, cfg["w2_steps"], cfg, allow_mint=k_max > 1,
        allow_prune=False, base_K=k_after_pong)
    # SCRATCH
    torch.manual_seed(seed + 1)
    lib2 = new_lib(cfg, k_max, hidden)
    w2b, A2b = make_world(world2, cfg["num_envs"], cfg["img"], seed + 1)
    scr_curve, scr_mint, _ = run_world(lib2, w2b, A2b, cfg["w2_steps"], cfg,
                                       allow_mint=k_max > 1, allow_prune=False,
                                       base_K=lib2.K)
    return dict(label=label, k_after_pong=k_after_pong, warm_curve=warm_curve,
                scratch_curve=scr_curve, warm_mint=warm_mint, scratch_mint=scr_mint,
                warm_reuse=round(warm_reuse, 3),
                warm_final=warm_curve[-1]["dyn"], scratch_final=scr_curve[-1]["dyn"])


def auc(curve):
    return sum(r["dyn"] for r in curve) / max(1, len(curve))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--world2", default="snake", choices=["breakout", "snake"])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--pong-steps", type=int, default=3000)
    p.add_argument("--w2-steps", type=int, default=2500)
    p.add_argument("--mint-every", type=int, default=40)
    p.add_argument("--mint-tol", type=float, default=0.010)
    p.add_argument("--prune-every", type=int, default=200)
    p.add_argument("--lib-hidden", type=int, default=64)
    p.add_argument("--lib-kmax", type=int, default=16)
    p.add_argument("--mono-hidden", type=int, default=1024)
    p.add_argument("--out-dir", default="craft_v6_out")
    args = p.parse_args()

    P_dim = 3 * args.patch * args.patch
    cfg = dict(num_envs=args.num_envs, img=args.img, patch=args.patch, P_dim=P_dim,
               pong_steps=args.pong_steps, w2_steps=args.w2_steps,
               mint_every=args.mint_every, mint_tol=args.mint_tol,
               prune_every=args.prune_every)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[NG probe-reuse] Pong->{args.world2.upper()} | lib(K<={args.lib_kmax},h{args.lib_hidden}) "
          f"vs monolith(h{args.mono_hidden}) | +shuffled control | seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()
    rows = []
    for s in args.seeds:
        lib_r = arm(cfg, s, args.lib_kmax, args.lib_hidden, "library", args.world2)
        mono_r = arm(cfg, s, 1, args.mono_hidden, "monolith", args.world2)
        shuf_r = arm(cfg, s, args.lib_kmax, args.lib_hidden, "shuffled", args.world2,
                     shuffled=True)
        lib_gain = (auc(lib_r["scratch_curve"]) - auc(lib_r["warm_curve"])) / \
            max(auc(lib_r["scratch_curve"]), 1e-6)
        mono_gain = (auc(mono_r["scratch_curve"]) - auc(mono_r["warm_curve"])) / \
            max(auc(mono_r["scratch_curve"]), 1e-6)
        shuf_gain = (auc(shuf_r["scratch_curve"]) - auc(shuf_r["warm_curve"])) / \
            max(auc(shuf_r["scratch_curve"]), 1e-6)
        rows.append(dict(seed=s, lib=lib_r, mono=mono_r, shuf=shuf_r,
                         lib_gain=round(lib_gain, 3), mono_gain=round(mono_gain, 3),
                         shuf_gain=round(shuf_gain, 3)))
        print(f"  seed {s}: LIB gain {lib_gain*100:+.0f}% (mint {lib_r['warm_mint']}v{lib_r['scratch_mint']}, "
              f"reuse {lib_r['warm_reuse']}) | MONO {mono_gain*100:+.0f}% | "
              f"SHUFFLED {shuf_gain*100:+.0f}% (mint {shuf_r['warm_mint']}v{shuf_r['scratch_mint']}) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    lib_gains = [r["lib_gain"] for r in rows]
    mono_gains = [r["mono_gain"] for r in rows]
    shuf_gains = [r["shuf_gain"] for r in rows]
    import statistics as st
    def ms(xs):
        return (round(st.mean(xs), 3), round(st.pstdev(xs), 3))
    summary = dict(world2=args.world2,
                   lib_gain_mean_std=ms(lib_gains), mono_gain_mean_std=ms(mono_gains),
                   shuf_gain_mean_std=ms(shuf_gains),
                   lib_gains=lib_gains, mono_gains=mono_gains, shuf_gains=shuf_gains)
    print(f"\n  SUMMARY {args.world2}: lib {ms(lib_gains)}  mono {ms(mono_gains)}  "
          f"shuffled {ms(shuf_gains)}", flush=True)
    print(f"  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"ng_probe_reuse_{args.world2}.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, world2=args.world2, summary=summary,
                       rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
