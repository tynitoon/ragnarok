"""v6.0 M2 — (a) a goal-conditioned SKILL is learnable on CraftWorld, and
(b) how DEEP a FLAT PPO agent gets on the sparse tech tree in a fixed budget
(the gap the developmental agent must close).

Skills are goal-conditioned PPO policies "achieve node g GIVEN its
prerequisites" (env grant = prerequisite outputs pre-stocked). The
substantive, learnable skills are the COLLECT (navigation) skills; craft
skills are near-trivial once materials are present.

Usage: python -m scripts.craft_skill_v6 [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, ACH_DEPTH, N_ACH, N_ITEMS,
    WPICK, SPICK, TABLE, FURNACE, A_WOOD, A_STONE, A_IRON)


def _grant(*item_idxs):
    g = [0] * N_ITEMS
    for i in item_idxs:
        g[i] = 1                       # tools/prereq outputs as flags
    return g


# representative skills: a shallow navigation skill and a DEEP one (given prereqs)
SKILLS = [
    ("collect_wood", A_WOOD, _grant()),
    ("collect_stone", A_STONE, _grant(WPICK, TABLE)),
    ("collect_iron", A_IRON, _grant(WPICK, SPICK, TABLE)),
]


@torch.no_grad()
def _skill_success(ppo, goal, grant, cfg, n=256):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        a = ppo.act(obs, deterministic=True)
        obs, _, term, _, _ = env.step(a)
        ever |= term
    return float(ever.float().mean().item())


@torch.no_grad()
def _ach_profile(ppo, cfg, n=256):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"])
    unlocked = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        a = ppo.act(obs, deterministic=True)
        obs, _, _, _, _ = env.step(a)
        unlocked |= env.unlocked
    return unlocked.float().mean(0).cpu()


def _train_skill(name, goal, grant, cfg):
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    succ = 0.0
    for it in range(1, cfg["skill_iters"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % cfg["eval_every"] == 0:
            succ = _skill_success(ppo, goal, grant, cfg)
            print(f"    [skill {name}] it {it:>3} | success {succ:.2f} | "
                  f"steps {ppo.total_steps:,}", flush=True)
            if succ >= 0.95:
                break
    return succ, ppo.total_steps


def _train_flat(cfg):
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"])
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    for it in range(1, cfg["flat_iters"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % cfg["flat_eval_every"] == 0:
            prof = _ach_profile(ppo, cfg)
            deepest = max((ACH_DEPTH[i] for i in range(N_ACH) if prof[i] >= 0.5),
                          default=-1)
            print(f"    [flat] it {it:>3} | steps {ppo.total_steps:,} | "
                  f"deepest(>=0.5) depth {deepest} | iron_pick {prof[-1]:.2f}",
                  flush=True)
    return _ach_profile(ppo, cfg), ppo.total_steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--skill-iters", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--flat-iters", type=int, default=300)
    p.add_argument("--flat-eval-every", type=int, default=30)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.num_envs, args.skill_iters, args.flat_iters = 64, 12, 12
        args.eval_every, args.flat_eval_every, args.max_steps = 4, 4, 60

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_iters", "eval_every", "flat_iters",
            "flat_eval_every")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[craft-skill-v6] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    print("\n[skills] each skill learnable given its prerequisites?", flush=True)
    skill_res = {}
    for name, goal, grant in SKILLS:
        s, steps = _train_skill(name, goal, grant, cfg)
        skill_res[name] = dict(success=s, steps=steps)
        print(f"  {name:16s} -> success {s:.2f} ({steps:,} env-steps)", flush=True)

    print("\n[flat] flat PPO on the full sparse tech tree (depth reached?)",
          flush=True)
    prof, flat_steps = _train_flat(cfg)
    print(f"\n  flat achievement profile ({flat_steps:,} env-steps):", flush=True)
    for i, nm in enumerate(ACH_NAMES):
        print(f"    {nm:20s} depth {ACH_DEPTH[i]} : {prof[i]:.2f}", flush=True)

    skills_ok = all(v["success"] >= 0.8 for v in skill_res.values())
    flat_deep = float(prof[-1])               # iron_pickaxe
    flat_deepest = max((ACH_DEPTH[i] for i in range(N_ACH) if prof[i] >= 0.5),
                       default=-1)
    print(f"\n  skills learnable (all >=0.8): {skills_ok}")
    print(f"  flat deepest achievement (>=0.5): depth {flat_deepest} | "
          f"iron_pickaxe {flat_deep:.2f}")
    verdict = ("M2 OK: skills learnable; flat PPO stalls shallow (deep nodes "
               "remain the gap for the developmental agent)"
               if skills_ok and flat_deep < 0.2 else
               "CHECK: see numbers")
    print(f"  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "m2.json"), "w") as f:
        json.dump(dict(skills=skill_res, flat_profile=prof.tolist(),
                       flat_steps=flat_steps, flat_deepest_depth=flat_deepest,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
