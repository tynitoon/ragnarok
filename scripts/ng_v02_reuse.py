"""v0.2 of the NOTION GRAPH — FORCED CROSS-WORLD REUSE (the decisive representation test).

Build the notion library on Pong, then drop it on Breakout (shares ball+paddle+bounce;
new: bricks + brick-collisions). WARM (reuse the Pong library) vs SCRATCH (fresh library):
does WARM reach low Breakout prediction error FASTER and MINT FEWER new notions?

FAIR control (the v45/v48 lesson): a param-matched MONOLITH (single big predictor, no
notion structure) run WARM vs SCRATCH too. If the library's WARM-vs-SCRATCH gain EXCEEDS
the monolith's, then the compositional STRUCTURE is what reuses — not just transferred weights.

Usage: python -m scripts.ng_v02_reuse [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.learning.notion_graph import NotionLibrary, patchify


def run_world(lib, env, steps, cfg, allow_mint, allow_prune, base_K, log_every=200):
    """Predict-learn-(mint/prune) loop on `env`. Returns (curve, minted, reuse_rate).
    reuse_rate = fraction of CHANGED patches bound to notions that existed at phase
    start (index < base_K) -> reuse of prior notions on the new world."""
    N, img, P_dim, patch, A = (cfg["num_envs"], cfg["img"], cfg["P_dim"],
                               cfg["patch"], 3)
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
        aoh = torch.zeros(N, A, device=DEVICE)
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
    return NotionLibrary(cfg["ctx_dim"], cfg["P_dim"], hidden=hidden, k_init=1,
                         k_max=k_max)


def arm(cfg, seed, k_max, hidden, label):
    """WARM (Pong->Breakout) and SCRATCH (Breakout) for one model class."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # WARM: learn Pong, then Breakout (reuse)
    lib = new_lib(cfg, k_max, hidden)
    pong = DeviceVecPong(cfg["num_envs"], img=cfg["img"], seed=seed)
    run_world(lib, pong, cfg["pong_steps"], cfg, allow_mint=k_max > 1,
              allow_prune=k_max > 1, base_K=lib.K)
    k_after_pong = lib.K
    bk = DeviceVecBreakout(cfg["num_envs"], img=cfg["img"], seed=seed)
    warm_curve, warm_mint, warm_reuse = run_world(
        lib, bk, cfg["bk_steps"], cfg, allow_mint=k_max > 1, allow_prune=False,
        base_K=k_after_pong)
    # SCRATCH: Breakout only
    torch.manual_seed(seed + 1)
    lib2 = new_lib(cfg, k_max, hidden)
    bk2 = DeviceVecBreakout(cfg["num_envs"], img=cfg["img"], seed=seed + 1)
    scr_curve, scr_mint, _ = run_world(lib2, bk2, cfg["bk_steps"], cfg,
                                       allow_mint=k_max > 1, allow_prune=False,
                                       base_K=lib2.K)
    return dict(label=label, k_after_pong=k_after_pong, warm_curve=warm_curve,
                scratch_curve=scr_curve, warm_mint=warm_mint, scratch_mint=scr_mint,
                warm_reuse=round(warm_reuse, 3),
                warm_final=warm_curve[-1]["dyn"], scratch_final=scr_curve[-1]["dyn"])


def auc(curve):
    return sum(r["dyn"] for r in curve) / max(1, len(curve))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--pong-steps", type=int, default=3000)
    p.add_argument("--bk-steps", type=int, default=2500)
    p.add_argument("--mint-every", type=int, default=40)
    p.add_argument("--mint-tol", type=float, default=0.010)
    p.add_argument("--prune-every", type=int, default=200)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lib-hidden", type=int, default=64)
    p.add_argument("--lib-kmax", type=int, default=16)
    p.add_argument("--mono-hidden", type=int, default=1024)  # >= 16-notion library params (conservative: favours monolith)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.num_envs = [0], 64
        args.pong_steps, args.bk_steps, args.mono_hidden = 800, 700, 256

    P_dim = 3 * args.patch * args.patch
    cfg = dict(num_envs=args.num_envs, img=args.img, patch=args.patch, P_dim=P_dim,
               ctx_dim=2 * P_dim + 3, pong_steps=args.pong_steps, bk_steps=args.bk_steps,
               mint_every=args.mint_every, mint_tol=args.mint_tol,
               prune_every=args.prune_every, warmup=args.warmup)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[NG v0.2] device={DEVICE} | FORCED CROSS-WORLD REUSE Pong->Breakout | "
          f"library(K<= {args.lib_kmax}, h{args.lib_hidden}) vs param-matched monolith"
          f"(h{args.mono_hidden}) | WARM vs SCRATCH | seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        lib_r = arm(cfg, s, args.lib_kmax, args.lib_hidden, "library")
        mono_r = arm(cfg, s, 1, args.mono_hidden, "monolith")
        # reuse gain = how much WARM beats SCRATCH on Breakout (lower AUC = faster)
        lib_gain = (auc(lib_r["scratch_curve"]) - auc(lib_r["warm_curve"])) / \
            max(auc(lib_r["scratch_curve"]), 1e-6)
        mono_gain = (auc(mono_r["scratch_curve"]) - auc(mono_r["warm_curve"])) / \
            max(auc(mono_r["scratch_curve"]), 1e-6)
        rows.append(dict(seed=s, lib=lib_r, mono=mono_r,
                         lib_gain=round(lib_gain, 3), mono_gain=round(mono_gain, 3)))
        print(f"  seed {s}: LIBRARY warm-vs-scratch AUC gain {lib_gain*100:+.0f}% | "
              f"warm-mint {lib_r['warm_mint']} vs scratch-mint {lib_r['scratch_mint']} | "
              f"reuse {lib_r['warm_reuse']} (Pong notions used on Breakout) | "
              f"MONOLITH gain {mono_gain*100:+.0f}% | {time.perf_counter()-t0:.0f}s",
              flush=True)

    lib_gains = [r["lib_gain"] for r in rows]
    mono_gains = [r["mono_gain"] for r in rows]
    fewer_mint = all(r["lib"]["warm_mint"] < r["lib"]["scratch_mint"] for r in rows)
    struct = all(r["lib_gain"] > r["mono_gain"] for r in rows)
    positive = (len(rows) >= 3 and all(g > 0.1 for g in lib_gains) and fewer_mint
                and struct)
    verdict = (
        f"FORCED CROSS-WORLD REUSE WORKS — the notion library learns Breakout FASTER warm "
        f"than scratch ({[f'{g*100:+.0f}%' for g in lib_gains]} AUC), MINTS FEWER new notions "
        f"warm, and the gain EXCEEDS the param-matched monolith ({[f'{g*100:+.0f}%' for g in mono_gains]}) "
        f"every seed -> the compositional STRUCTURE reuses, from pixels. REVIEW before reporting."
        if positive else
        f"PARTIAL/CHECK — lib gains {lib_gains}, mono gains {mono_gains}, fewer_mint={fewer_mint}, "
        f"struct>{mono_gains}={struct}. Honest: read curves, tune, or bound the claim.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "ng_v02_reuse.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, cfg={k: cfg[k] for k in
                       ("num_envs", "patch", "pong_steps", "bk_steps")},
                       lib_hidden=args.lib_hidden, mono_hidden=args.mono_hidden,
                       rows=rows, positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
