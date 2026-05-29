"""v7.0 — AUTONOMOUS DISCOVERY: the agent generates its own sub-goal
curriculum. No given achievement list / order / task reward — only
item-novelty. It discovers what to learn next by FRONTIER EXPANSION:

  repeat:
    DISCOVER: from the state of "all mastered items available" (granted —
      the agent can produce them via its skills), run a short RANDOM
      exploration; any NEW item type that appears is a frontier sub-goal.
    LEARN:    train a goal-conditioned skill to obtain each new item
      (grant = mastered items; M3-style — only the new step is learned).
      Add mastered items to the library.
  until no new item is discovered.

Because an item is only discoverable once its prerequisites are mastered
(and thus granted), the discovery ORDER reconstructs the dependency DAG
bottom-up — without ever being told it.

Baseline (curiosity-flat) = flat PPO with per-item-novelty reward; that is
exactly the M2 flat agent (iron_pickaxe 0.11, reliable depth <=4), cited.

Usage: python -m scripts.discover_v7 [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, N_ITEMS, N_ACH, ACH_DEPTH,
    WOOD, STONE_I, COAL_I, IRON_I, WPICK, SPICK, IPICK, TABLE, FURNACE,
    A_WOOD, A_TABLE, A_WPICK, A_STONE, A_COAL, A_SPICK, A_FURNACE, A_IRON, A_IPICK)

# item index -> the achievement (goal) of obtaining it, and a readable name
ITEM_TO_ACH = {WOOD: A_WOOD, TABLE: A_TABLE, WPICK: A_WPICK, STONE_I: A_STONE,
               COAL_I: A_COAL, SPICK: A_SPICK, FURNACE: A_FURNACE,
               IRON_I: A_IRON, IPICK: A_IPICK}
ITEM_NAME = {WOOD: "wood", STONE_I: "stone", COAL_I: "coal", IRON_I: "iron",
             WPICK: "wood_pickaxe", SPICK: "stone_pickaxe", IPICK: "iron_pickaxe",
             TABLE: "table", FURNACE: "furnace"}
ALL_ITEMS = list(ITEM_TO_ACH.keys())


def _grant_vec(mastered):
    g = [0] * N_ITEMS
    for i in mastered:
        g[i] = 5                       # generous stock so random crafts have inputs
    return g


@torch.no_grad()
def _discover(mastered, cfg):
    """From the granted mastered-item state, random-explore; return the
    fraction of envs that obtain each NOT-yet-mastered item."""
    env = DeviceVecCraftWorld(cfg["disc_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["disc_steps"] + 1, grant=_grant_vec(mastered))
    seen = torch.zeros(cfg["disc_envs"], N_ITEMS, dtype=torch.bool, device=DEVICE)
    start = env.inv.clone()
    for _ in range(cfg["disc_steps"]):
        a = torch.randint(0, 10, (cfg["disc_envs"],), device=DEVICE)
        env.step(a)
        # an item is "obtained" if its count rose above the granted baseline
        seen |= env.inv > start
    frac = seen.float().mean(0)
    return {i: float(frac[i]) for i in ALL_ITEMS if i not in mastered}


@torch.no_grad()
def _skill_success(ppo, goal, grant, cfg, n=256):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def _learn_item(item, mastered, cfg):
    """Train a skill to obtain `item` given mastered items granted."""
    goal = ITEM_TO_ACH[item]
    grant = _grant_vec(mastered)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    succ = 0.0
    for it in range(1, cfg["skill_cap"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % cfg["eval_every"] == 0:
            succ = _skill_success(ppo, goal, grant, cfg)
            if succ >= cfg["mastery"]:
                break
    return ppo.total_steps, max(succ, _skill_success(ppo, goal, grant, cfg))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--skill-cap", type=int, default=70)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--disc-envs", type=int, default=512)
    p.add_argument("--disc-steps", type=int, default=120)
    p.add_argument("--disc-thresh", type=float, default=0.01)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.skill_cap, args.disc_envs = 64, 12, 128
        args.disc_steps, args.max_rounds, args.eval_every = 60, 4, 4

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap", "eval_every", "mastery", "disc_envs",
            "disc_steps", "disc_thresh")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[discover-v7] device={DEVICE} | NO given goals — novelty only | "
          f"seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    runs = []
    for seed in range(args.seeds):
        import numpy as _np
        torch.manual_seed(seed); _np.random.seed(seed)
        print(f"\n########## SEED {seed} ##########", flush=True)
        runs.append(_run_once(cfg, args, seed))

    n_full = sum(1 for r in runs if r["mastered_count"] == len(ALL_ITEMS))
    n_dag = sum(1 for r in runs if r["dag_ok"])
    n_ipick = sum(1 for r in runs if r["reached_ipick"])
    print(f"\n{'=' * 74}\n  v7.0 AUTONOMOUS DISCOVERY — N={args.seeds}")
    print(f"{'=' * 74}")
    print(f"  reached full tree (9/9):     {n_full}/{args.seeds}")
    print(f"  DAG-valid discovery order:   {n_dag}/{args.seeds}")
    print(f"  reached iron_pickaxe:        {n_ipick}/{args.seeds}")
    print(f"  baseline curiosity-flat (M2): iron_pickaxe 0.11, reliable depth <=4")
    robust = n_full == args.seeds and n_dag == args.seeds
    verdict = ("AUTONOMOUS DISCOVERY WORKS (robust) — every seed discovers its "
               "own curriculum and masters the FULL tree via a DAG-valid order, "
               "no goals given; far past curiosity-flat."
               if robust else
               f"PARTIAL — full {n_full}/{args.seeds}, dag {n_dag}/{args.seeds}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v7.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, runs=runs, n_full=n_full, n_dag=n_dag,
                       n_ipick=n_ipick, verdict=verdict), f, indent=2)


def _run_once(cfg, args, seed):
    """One full autonomous-discovery run; returns a result dict."""
    mastered = set()
    order = []
    for rnd in range(1, args.max_rounds + 1):
        frac = _discover(mastered, cfg)
        cand = sorted([i for i, f in frac.items() if f >= args.disc_thresh],
                      key=lambda i: -frac[i])
        print(f"[s{seed} round {rnd}] mastered={[ITEM_NAME[i] for i in mastered]} "
              f"| reachable={[(ITEM_NAME[i], round(frac[i], 2)) for i in cand]}",
              flush=True)
        if not cand:
            break
        for item in cand:
            if item in mastered:
                continue
            steps, succ = _learn_item(item, mastered, cfg)
            if succ >= args.mastery:
                mastered.add(item); order.append(ITEM_NAME[item])
                print(f"    +mastered {ITEM_NAME[item]} ({succ:.2f})", flush=True)
        if len(mastered) == len(ALL_ITEMS):
            break

    pos = {name: k for k, name in enumerate(order)}

    def before(a, b):
        return pos.get(a, 1e9) < pos.get(b, -1)
    dag_ok = (before("wood", "table") and before("table", "wood_pickaxe")
              and before("wood_pickaxe", "stone") and before("stone", "stone_pickaxe")
              and before("stone_pickaxe", "iron") and before("iron", "iron_pickaxe")
              and before("furnace", "iron_pickaxe"))
    res = dict(mastered_count=len(mastered), order=order, dag_ok=bool(dag_ok),
               reached_ipick=IPICK in mastered)
    print(f"  [s{seed}] mastered {res['mastered_count']}/{len(ALL_ITEMS)} | "
          f"dag_ok={dag_ok} | ipick={res['reached_ipick']} | order={order}",
          flush=True)
    return res


if __name__ == "__main__":
    main()
