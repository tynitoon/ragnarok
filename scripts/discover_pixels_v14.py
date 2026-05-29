"""v14 — AUTONOMOUS DISCOVERY FROM PIXELS (capstone).

Unifies the validated pieces of the project on the hard substrate:
  - v7.0  self-directed discovery (frontier item-novelty + reuse, no goals),
  - v12-A perception (skills learned from raw pixels), and
  - v13b  compounding (reused prerequisites make deep skills learnable).

The agent observes ONLY pixels. No goals, recipes, or order are given. Each
round it explores from its "all-mastered-items" state, detects any NEW item it
can reach, and learns a goal-conditioned CNN skill FROM PIXELS to obtain it
(prerequisites granted = "already mastered"); it repeats until nothing new
appears. Because an item is only reachable once its prerequisites are mastered,
the discovery ORDER reconstructs the dependency DAG bottom-up — from pixels.

Honest caveat: the novelty/reachability signal reads the inventory
(proprioception, as in symbolic v7); PERCEPTION, NAVIGATION and skill learning
are all from pixels.

Usage: python -m scripts.discover_pixels_v14 [--seeds 3] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.craft_world import DeviceVecCraftWorld, IPICK
from scripts.discover_v7 import (
    _discover, ITEM_TO_ACH, ITEM_NAME, ALL_ITEMS, _grant_vec)


@torch.no_grad()
def _skill_success(ppo, goal, grant, cfg, n=128):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant,
                              n_resource=cfg["n_resource"], pixel=True,
                              tile=cfg["tile"])
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def _learn_item_pixels(item, mastered, cfg):
    """Train a goal-conditioned CNN skill FROM PIXELS to obtain `item`, with
    mastered items granted. Returns (env_steps, best_success)."""
    goal = ITEM_TO_ACH[item]
    grant = _grant_vec(mastered)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant,
                              n_resource=cfg["n_resource"], pixel=True,
                              tile=cfg["tile"])
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=cfg["hidden"])
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=cfg["entropy"], net=net)
    best = 0.0
    for it in range(1, cfg["skill_cap"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % cfg["eval_every"] == 0:
            best = max(best, _skill_success(ppo, goal, grant, cfg))
            if best >= cfg["mastery"]:
                break
    return ppo.total_steps, max(best, _skill_success(ppo, goal, grant, cfg))


def _run_once(cfg, args, seed):
    mastered, order, costs = set(), [], {}
    for rnd in range(1, args.max_rounds + 1):
        frac = _discover(mastered, cfg)
        cand = sorted([i for i, f in frac.items() if f >= args.disc_thresh],
                      key=lambda i: -frac[i])
        print(f"[s{seed} round {rnd}] mastered={[ITEM_NAME[i] for i in mastered]} "
              f"| reachable={[ITEM_NAME[i] for i in cand]}", flush=True)
        if not cand:
            break
        for item in cand:
            if item in mastered:
                continue
            steps, succ = _learn_item_pixels(item, mastered, cfg)
            if succ >= args.mastery:
                mastered.add(item); order.append(ITEM_NAME[item]); costs[ITEM_NAME[item]] = steps
                print(f"    +mastered {ITEM_NAME[item]} FROM PIXELS "
                      f"({succ:.2f}, {steps:,} steps)", flush=True)
            else:
                print(f"    x {ITEM_NAME[item]} only {succ:.2f}", flush=True)
        if len(mastered) == len(ALL_ITEMS):
            break

    pos = {name: k for k, name in enumerate(order)}
    bf = lambda a, b: pos.get(a, 1e9) < pos.get(b, -1)
    dag_ok = (bf("wood", "table") and bf("table", "wood_pickaxe")
              and bf("wood_pickaxe", "stone") and bf("stone", "stone_pickaxe")
              and bf("stone_pickaxe", "iron") and bf("iron", "iron_pickaxe")
              and bf("furnace", "iron_pickaxe"))
    res = dict(mastered_count=len(mastered), order=order, costs=costs,
               dag_ok=bool(dag_ok), reached_ipick=IPICK in mastered)
    print(f"  [s{seed}] mastered {res['mastered_count']}/{len(ALL_ITEMS)} FROM "
          f"PIXELS | dag_ok={dag_ok} | ipick={res['reached_ipick']} | {order}",
          flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--skill-cap", type=int, default=70)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--disc-envs", type=int, default=512)
    p.add_argument("--disc-steps", type=int, default=120)
    p.add_argument("--disc-thresh", type=float, default=0.01)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.num_envs, args.skill_cap = 1, 64, 16
        args.disc_envs, args.disc_steps, args.max_rounds = 128, 60, 4

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "tile", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap", "eval_every", "mastery", "disc_envs",
            "disc_steps", "disc_thresh", "n_resource")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v14] device={DEVICE} | AUTONOMOUS DISCOVERY FROM PIXELS | "
          f"no goals given | seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()
    runs = []
    for seed in range(args.seeds):
        import numpy as _np
        torch.manual_seed(seed); _np.random.seed(seed)
        print(f"\n########## SEED {seed} ##########", flush=True)
        runs.append(_run_once(cfg, args, seed))
        with open(os.path.join(args.out_dir, "v14_partial.json"), "w") as f:
            json.dump(dict(done_seeds=seed + 1, runs=runs), f)

    n_full = sum(1 for r in runs if r["mastered_count"] == len(ALL_ITEMS))
    n_dag = sum(1 for r in runs if r["dag_ok"])
    n_ipick = sum(1 for r in runs if r["reached_ipick"])
    print(f"\n{'=' * 74}\n  v14 AUTONOMOUS DISCOVERY FROM PIXELS — N={args.seeds}")
    print(f"{'=' * 74}")
    print(f"  full tree (9/9) from pixels: {n_full}/{args.seeds}")
    print(f"  DAG-valid discovery order:   {n_dag}/{args.seeds}")
    print(f"  reached iron_pickaxe:        {n_ipick}/{args.seeds}", flush=True)
    ok = n_full >= (args.seeds + 1) // 2 and n_dag >= (args.seeds + 1) // 2 \
        and n_ipick >= (args.seeds + 1) // 2
    verdict = ("AUTONOMOUS DISCOVERY FROM PIXELS WORKS — given only pixels and "
               "no goals, the agent discovers its own curriculum and masters the "
               "full tech-tree bottom-up (DAG-valid, reaching iron_pickaxe). The "
               "whole developmental vision, on the raw-pixel substrate."
               if ok else
               f"PARTIAL/NEGATIVE — full {n_full}/{args.seeds}, dag {n_dag}/"
               f"{args.seeds}, ipick {n_ipick}/{args.seeds}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v14.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, runs=runs, n_full=n_full, n_dag=n_dag,
                       n_ipick=n_ipick, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
