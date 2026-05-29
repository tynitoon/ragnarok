"""v6.0 M4 — end-to-end: the agent COMPLETES make_iron_pickaxe by composing
its LEARNED skills, from the base state (no grants), under partial
(egocentric) observation.

The learned, reusable part = 4 goal-conditioned COLLECT skills (navigate to
a resource and harvest), each trained once. The composition = a
resource-aware high-level controller that follows the tech-tree DAG: gather
the needed quantity of each resource by running the matching learned skill,
and emit a craft action when materials are present. Crafts are trivial
given materials, so the substance is the learned navigation skills.

Decisive demonstration: completion rate of make_iron_pickaxe is HIGH (the
library composes end-to-end), vs flat PPO ~0.11 (M2) and a no-skills
(random-collect) control. This is "watch her actually build it".

Usage: python -m scripts.craft_endtoend_v6 [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ITEMS, N_ACH, TREE, STONE, COAL, IRON,
    WOOD, STONE_I, COAL_I, IRON_I, WPICK, SPICK, IPICK, TABLE, FURNACE,
    A_WOOD, A_STONE, A_COAL, A_IRON)

# learned collect skills: resource cell type -> (achievement goal, grant)
COLLECT = {
    TREE:  (A_WOOD,  []),
    STONE: (A_STONE, [WPICK, TABLE]),
    COAL:  (A_COAL,  [WPICK, TABLE]),
    IRON:  (A_IRON,  [SPICK, WPICK, TABLE]),
}


def _grant(idxs):
    g = [0] * N_ITEMS
    for i in idxs:
        g[i] = 1
    return g


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


def _train_collect(cell_type, cfg):
    goal, grant_idx = COLLECT[cell_type]
    grant = _grant(grant_idx)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    for it in range(1, cfg["skill_cap"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % 10 == 0 and _skill_success(ppo, goal, grant, cfg) >= 0.9:
            break
    return ppo, _skill_success(ppo, goal, grant, cfg)


def _goal_onehot(goal_idx, n):
    g = torch.zeros(n, N_ACH, device=DEVICE)
    g[:, goal_idx] = 1.0
    return g


def _choose_target(inv):
    """Per-env: which resource to gather next (resource-aware, DAG-ordered).
    Ensures enough wood/stone for the downstream crafts."""
    N = inv.shape[0]
    tgt = torch.full((N,), TREE, dtype=torch.long, device=DEVICE)   # default wood
    has_w = inv[:, WPICK] >= 1
    has_s = inv[:, SPICK] >= 1
    # before wood pickaxe: gather wood (>=2 for table+wpick)
    # after wpick, before spick: need stone>=2 (spick+furnace) and wood>=1
    m = has_w & (~has_s) & (inv[:, STONE_I] < 2)
    tgt[m] = STONE
    # after spick: need iron, coal, and wood for iron_pickaxe
    tgt[has_s & (inv[:, IRON_I] < 1)] = IRON
    tgt[has_s & (inv[:, IRON_I] >= 1) & (inv[:, COAL_I] < 1)] = COAL
    return tgt


def _hl_action(env, skills):
    """Resource-aware controller over LEARNED collect skills + scripted crafts."""
    inv = env.inv
    N = env.num_envs
    a = torch.full((N,), -1, dtype=torch.long, device=DEVICE)

    def setif(mask, act):
        m = mask & (a < 0)
        a[m] = act

    # craft when materials present (deepest first), if not already owned
    setif((inv[:, WOOD] >= 1) & (inv[:, COAL_I] >= 1) & (inv[:, IRON_I] >= 1)
          & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] >= 1) & (inv[:, IPICK] == 0), 9)
    setif((inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] == 0), 8)
    setif((inv[:, WOOD] >= 1) & (inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1)
          & (inv[:, SPICK] == 0), 7)
    setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, WPICK] == 0), 6)
    setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] == 0), 5)

    # the rest: gather a resource using its LEARNED skill
    target = _choose_target(inv)
    base = env.state                              # (N, base_obs) — non-goal env
    for cell, (goal_idx, _) in COLLECT.items():
        mask = (a < 0) & (target == cell)
        if bool(mask.any()):
            obs_g = torch.cat([base[mask], _goal_onehot(goal_idx, int(mask.sum()))], -1)
            a[mask] = skills[cell].act(obs_g, deterministic=True)
    a[a < 0] = 4                                  # fallback: collect
    return a


@torch.no_grad()
def _rollout_complete(controller, cfg, n=512, budget=400):
    """Run the controller in the FULL env (no grant); fraction completing
    each achievement, esp. make_iron_pickaxe."""
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=budget + 1)
    unlocked = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
    for _ in range(budget):
        env.step(controller(env))
        unlocked |= env.unlocked
    return unlocked.float().mean(0).cpu()


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
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.skill_cap, args.budget, args.max_steps = 64, 10, 200, 60

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[craft-e2e-v6] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    print("\n[skills] training the 4 reusable collect skills...", flush=True)
    skills, succ = {}, {}
    names = {TREE: "wood", STONE: "stone", COAL: "coal", IRON: "iron"}
    for cell in (TREE, STONE, COAL, IRON):
        ppo, s = _train_collect(cell, cfg)
        skills[cell] = ppo; succ[names[cell]] = s
        print(f"  collect_{names[cell]:6s} skill success {s:.2f}", flush=True)

    print("\n[compose] running the resource-aware controller over learned "
          "skills (full env, no grants)...", flush=True)
    prof = _rollout_complete(lambda e: _hl_action(e, skills), cfg, n=512,
                             budget=args.budget)

    # no-skills control: same controller logic but collect = random move/collect
    def _random_collect(env):
        inv = env.inv; N = env.num_envs
        a = torch.full((N,), -1, dtype=torch.long, device=DEVICE)

        def setif(mask, act):
            m = mask & (a < 0); a[m] = act
        setif((inv[:, WOOD] >= 1) & (inv[:, COAL_I] >= 1) & (inv[:, IRON_I] >= 1)
              & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] >= 1) & (inv[:, IPICK] == 0), 9)
        setif((inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] == 0), 8)
        setif((inv[:, WOOD] >= 1) & (inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1)
              & (inv[:, SPICK] == 0), 7)
        setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, WPICK] == 0), 6)
        setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] == 0), 5)
        rnd = torch.randint(0, 5, (N,), device=DEVICE)     # move/collect
        a[a < 0] = rnd[a < 0]
        return a
    prof_ctrl = _rollout_complete(_random_collect, cfg, n=512, budget=args.budget)

    from ragnarok.environments.craft_world import ACH_NAMES, A_IPICK
    print(f"\n  {'achievement':20s} {'learned-skills':>15} {'random-collect':>15}")
    for i, nm in enumerate(ACH_NAMES):
        print(f"  {nm:20s} {prof[i]:>15.2f} {prof_ctrl[i]:>15.2f}", flush=True)
    e2e = float(prof[A_IPICK])
    ctrl = float(prof_ctrl[A_IPICK])
    print(f"\n  END-TO-END make_iron_pickaxe: learned-skill agent {e2e:.2f} | "
          f"random-collect controller {ctrl:.2f} | flat PPO (M2) 0.11")
    ok = e2e >= 0.8
    verdict = ("AGENT COMPLETES THE TREE — composing its learned skills, the "
               "agent builds the depth-6 iron_pickaxe end-to-end from the base "
               "state under partial observation, far beyond flat PPO (0.11)."
               if ok else "CHECK — end-to-end completion below target; see table.")
    print(f"  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "m4.json"), "w") as f:
        json.dump(dict(skill_success=succ, profile_learned=prof.tolist(),
                       profile_random=prof_ctrl.tolist(), iron_pickaxe=e2e,
                       iron_pickaxe_random=ctrl, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
