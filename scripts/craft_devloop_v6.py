"""v6.0 M3 — the conclusive learning-to-learn curve on a deep tech tree.

For each achievement node g (depth 0..6), measure the env-steps to LEARN it:
  - DEVELOPMENTAL: given its prerequisites are available (env `grant` =
    prerequisite outputs, i.e. the agent already has the prerequisite
    skills) -> only the NEW step is learned.
  - NO-REUSE: from the BASE state (no grant), goal-only sparse reward ->
    the whole chain must be discovered from one terminal reward.

The developmental signature ("de plus en plus vite"): DEVELOPMENTAL
marginal cost stays ~flat in depth and masters EVERY node incl. depth-6
make_iron_pickaxe; NO-REUSE cost explodes with depth and fails the deep
nodes (matching the flat-PPO gap from M2, where iron_pickaxe = 0.11). Reuse
of learned basics is the lever.

Usage: python -m scripts.craft_devloop_v6 [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, ACH_DEPTH, N_ACH, N_ITEMS,
    WOOD, STONE_I, COAL_I, IRON_I, WPICK, SPICK, TABLE, FURNACE,
    A_WOOD, A_TABLE, A_WPICK, A_STONE, A_COAL, A_SPICK, A_FURNACE, A_IRON, A_IPICK)

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _grant(*idxs):
    g = [0] * N_ITEMS
    for i in idxs:
        g[i] = 1
    return g


# prerequisite-output inventory granted to the DEVELOPMENTAL learner per node
GRANTS = {
    A_WOOD: _grant(),
    A_TABLE: _grant(WOOD),
    A_WPICK: _grant(WOOD, TABLE),
    A_STONE: _grant(WPICK),
    A_COAL: _grant(WPICK),
    A_SPICK: _grant(WOOD, STONE_I, TABLE),
    A_FURNACE: _grant(STONE_I, TABLE),
    A_IRON: _grant(SPICK),
    A_IPICK: _grant(WOOD, COAL_I, IRON_I, TABLE, FURNACE),
}
NODES = list(range(N_ACH))


@torch.no_grad()
def _success(ppo, goal, grant, cfg, n=256):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        a = ppo.act(obs, deterministic=True)
        obs, _, term, _, _ = env.step(a)
        ever |= term
    return float(ever.float().mean().item())


def _train_node(goal, grant, cfg, cap):
    """Train a goal-conditioned skill; return (env_steps_to_master, mastered,
    final_success)."""
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    succ = 0.0
    for it in range(1, cap + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % cfg["eval_every"] == 0:
            succ = _success(ppo, goal, grant, cfg)
            if succ >= cfg["mastery"]:
                return ppo.total_steps, True, succ
    succ = _success(ppo, goal, grant, cfg)
    return ppo.total_steps, succ >= cfg["mastery"], succ


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--dev-cap", type=int, default=80)
    p.add_argument("--noreuse-cap", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.num_envs = 1, 64
        args.dev_cap, args.noreuse_cap, args.eval_every = 8, 8, 4
        args.max_steps = 60

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "eval_every", "mastery")}
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "m3.json")
    print(f"[craft-devloop-v6] device={DEVICE} | seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = json.load(f).get("seeds", {})
        print(f"[resume] seeds done: {list(done)}", flush=True)

    for seed in range(args.seeds):
        if str(seed) in done:
            print(f"[seed {seed}] cached", flush=True); continue
        print(f"\n[seed {seed}]", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        rows = []
        for g in NODES:
            d = ACH_DEPTH[g]
            cd, md, sd = _train_node(g, GRANTS[g], cfg, args.dev_cap)
            cn, mn, sn = _train_node(g, None, cfg, args.noreuse_cap)
            rows.append(dict(node=ACH_NAMES[g], depth=d,
                             dev_cost=cd, dev_master=md, dev_succ=sd,
                             nore_cost=cn, nore_master=mn, nore_succ=sn))
            print(f"  {ACH_NAMES[g]:20s} d{d} | DEV {cd:>8,} m={md} ({sd:.2f}) "
                  f"| NO-REUSE {cn:>9,} m={mn} ({sn:.2f})", flush=True)
        done[str(seed)] = rows
        with open(results_path, "w") as f:
            json.dump({"seeds": done}, f, indent=2)
        dev_master = sum(r["dev_master"] for r in rows)
        nore_master = sum(r["nore_master"] for r in rows)
        print(f"  [seed {seed}] DEV mastered {dev_master}/{N_ACH} | "
              f"NO-REUSE mastered {nore_master}/{N_ACH}", flush=True)

    # ---- aggregate ----
    seeds = [done[str(s)] for s in range(args.seeds) if str(s) in done]
    if not seeds:
        return
    N = len(seeds)
    print(f"\n{'=' * 78}\n  v6.0 M3 — learning-to-learn on a deep tech tree | N={N}")
    print(f"{'=' * 78}")
    print(f"  {'node':20s} {'depth':>5} {'DEV steps':>12} {'DEVm':>5} "
          f"{'NO-REUSE steps':>15} {'NRm':>5}")
    dev_deep = []      # dev mastered depth>=5 nodes?
    nore_deep = []
    for i, g in enumerate(NODES):
        dcs = [s[i]["dev_cost"] for s in seeds]
        ncs = [s[i]["nore_cost"] for s in seeds]
        dm = np.mean([s[i]["dev_master"] for s in seeds])
        nm = np.mean([s[i]["nore_master"] for s in seeds])
        print(f"  {ACH_NAMES[g]:20s} {ACH_DEPTH[g]:>5} {int(np.mean(dcs)):>12,} "
              f"{dm:>5.2f} {int(np.mean(ncs)):>15,} {nm:>5.2f}")
        if ACH_DEPTH[g] >= 5:
            dev_deep.append(dm); nore_deep.append(nm)
    dev_tot = [sum(r["dev_cost"] for r in s) for s in seeds]
    nore_tot = [sum(r["nore_cost"] for r in s) for s in seeds]
    dev_master_all = [sum(r["dev_master"] for r in s) for s in seeds]
    nore_master_all = [sum(r["nore_master"] for r in s) for s in seeds]
    print(f"\n  DEV mastered/seed: {dev_master_all} of {N_ACH} | total learn-steps "
          f"{int(np.mean(dev_tot)):,}")
    print(f"  NO-REUSE mastered/seed: {nore_master_all} of {N_ACH} | total "
          f"{int(np.mean(nore_tot)):,}")
    print(f"  deep nodes (depth>=5) mastery: DEV {np.mean(dev_deep):.2f} vs "
          f"NO-REUSE {np.mean(nore_deep):.2f}")

    decisive = (np.mean(dev_master_all) >= N_ACH - 0.5
                and np.mean(dev_deep) >= 0.8 and np.mean(nore_deep) < 0.5)
    verdict = ("DEVELOPMENTAL LEARNING WORKS — reusing learned basics lets the "
               "agent master the FULL deep tree (incl. depth-6 iron_pickaxe) at "
               "bounded per-node cost, where learning from scratch (no reuse) "
               "explodes with depth and fails the deep nodes. 'De plus en plus "
               "vite' on a real tech tree."
               if decisive else
               "CHECK — developmental advantage not clearly decisive; see table.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(results_path, "w") as f:
        json.dump({"seeds": done, "verdict": verdict,
                   "dev_total_mean": float(np.mean(dev_tot)),
                   "nore_total_mean": float(np.mean(nore_tot))}, f, indent=2)
    print(f"  results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
