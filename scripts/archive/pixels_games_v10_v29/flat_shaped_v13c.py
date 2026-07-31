"""v13c — FAIR shaped-reward FLAT baseline for v13b (per phase-gate review).

Reviewer A's key critique: v13b's FLAT arm failed under a SPARSE goal-only
reward, so its failure could be a sparse-reward-exploration artifact, not
evidence that depth is unlearnable without reuse. The fair test: give a FLAT
agent (no reuse, no granted prerequisites) a DENSE, curriculum-free learning
signal and a generous budget, and see how DEEP it can climb the tech-tree
from pixels.

Setup: NON-goal mode (the env gives +1 for EACH first-time achievement — a
dense shaped signal that rewards climbing), grant=None (must derive everything
itself), pixels, generous iters/steps. Measure the achievement profile (the
fraction of envs reaching each of the 9 achievements).

Decisive: if the shaped-flat agent still fails to reach the DEEP targets
(make_stone_pickaxe d4, make_iron_pickaxe d6: <=0.2) while v13b's reuse arm
reaches 1.0, then reuse's advantage is REAL, not a sparse-reward artifact
(matching symbolic M2, where shaped/curiosity-flat stalled at depth <=4). If
shaped-flat DOES reach the deep targets, the v13b claim must be weakened.

Usage: python -m scripts.flat_shaped_v13c [--seeds 3] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, N_ACH, ACH_DEPTH, A_SPICK, A_IPICK)


@torch.no_grad()
def _profile(ppo, cfg, n=128):
    """Deterministic rollout in NON-goal pixel env; fraction reaching each ach."""
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], n_resource=cfg["n_resource"],
                              pixel=True, tile=cfg["tile"])
    ever = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= env.unlocked
    return ever.float().mean(0).cpu()


def train_flat_shaped(cfg, iters, eval_every, seed):
    torch.manual_seed(seed)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], n_resource=cfg["n_resource"],
                              pixel=True, tile=cfg["tile"], seed=seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=cfg["hidden"])
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=cfg["entropy"], net=net)
    best = torch.zeros(N_ACH)
    for it in range(1, iters + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % eval_every == 0:
            prof = _profile(ppo, cfg)
            best = torch.maximum(best, prof)
            depth = max((i for i in range(N_ACH) if prof[i] >= 0.5), default=-1)
            print(f"    [seed {seed}] it {it:>3} | deepest>=.5 d"
                  f"{ACH_DEPTH.get(depth, -1) if depth >= 0 else -1} "
                  f"({ACH_NAMES[depth] if depth >= 0 else 'none'}) | "
                  f"wood {prof[0]:.2f} spick {prof[A_SPICK]:.2f} "
                  f"ipick {prof[A_IPICK]:.2f}", flush=True)
    return torch.maximum(best, _profile(ppo, cfg))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=300)   # generous (>> reuse's ~10-30)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.iters, args.eval_every, args.num_envs = 1, 12, 4, 64

    cfg = {k: getattr(args, k) for k in
           ("grid", "view", "tile", "max_steps", "rollout", "hidden",
            "entropy", "num_envs", "n_resource")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v13c] device={DEVICE} | FAIR shaped-reward FLAT baseline from "
          f"pixels (per-achievement reward, no reuse) | {args.seeds} seeds x "
          f"{args.iters} iters", flush=True)
    t0 = time.perf_counter()
    profs = []
    for seed in range(args.seeds):
        print(f"\n  ##### seed {seed} #####", flush=True)
        profs.append(train_flat_shaped(cfg, args.iters, args.eval_every, seed))
        with open(os.path.join(args.out_dir, "v13c_partial.json"), "w") as f:
            json.dump(dict(done=seed + 1, profiles=[x.tolist() for x in profs]), f)

    mean = torch.stack(profs).mean(0)
    print(f"\n  {'achievement':20s} {'depth':>5} {'flat-shaped':>12} {'v13b-reuse':>11}")
    reuse_ref = {A_SPICK: 1.00, A_IPICK: 1.00}   # from v13b.json (deep targets)
    for i, nm in enumerate(ACH_NAMES):
        ref = f"{reuse_ref[i]:.2f}" if i in reuse_ref else "-"
        print(f"  {nm:20s} {ACH_DEPTH[i]:>5} {mean[i]:>12.2f} {ref:>11}", flush=True)
    deepest = max((i for i in range(N_ACH) if mean[i] >= 0.5), default=-1)
    spick, ipick = float(mean[A_SPICK]), float(mean[A_IPICK])
    reuse_wins = spick <= 0.2 and ipick <= 0.2
    verdict = (f"REUSE ADVANTAGE IS REAL (not a sparse-reward artifact) — even a "
               f"FAIR shaped-reward flat agent, no reuse, generous budget, stalls "
               f"at depth {ACH_DEPTH.get(deepest, -1) if deepest>=0 else -1} "
               f"(spick {spick:.2f}, ipick {ipick:.2f}) where v13b's reuse arm "
               f"reaches 1.00. Reuse, not reward shaping, unlocks the deep skills "
               f"from pixels."
               if reuse_wins else
               f"CLAIM WEAKENED — shaped-flat reaches deep targets (spick "
               f"{spick:.2f}, ipick {ipick:.2f}); reward shaping (not reuse) was "
               f"a key factor. v13b's reuse>flat contrast must be re-scoped.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v13c.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, mean_profile=mean.tolist(),
                       profiles=[x.tolist() for x in profs], deepest=deepest,
                       spick=spick, ipick=ipick, verdict=verdict,
                       ach_names=ACH_NAMES), f, indent=2)


if __name__ == "__main__":
    main()
