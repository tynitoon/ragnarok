"""v9.0 — MODEL-BASED: learn the world's rules, then PLAN to any goal.

The agent LEARNS each item's operator (precondition = which items must be
present to obtain it) by a causal leave-one-out necessity test, recovering
the recipe DAG from interaction (not given). It then BFS-PLANS the sub-goal
order to reach ANY target item zero-shot, and EXECUTES the plan with the
reused collect skills + craft actions. "Understand the world's rules -> any
goal is a planning problem." Connects to v4 Phase 1; subsumes recipe-discovery.

Usage: python -m scripts.model_based_v9 [--smoke]
"""

import argparse
import json
import os
import time
from collections import deque

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ITEMS, WOOD, STONE_I, COAL_I, IRON_I,
    WPICK, SPICK, IPICK, TABLE, FURNACE, A_WOOD, A_STONE, A_COAL, A_IRON)
from scripts.craft_endtoend_v6 import _train_collect, _goal_onehot

# item -> (achievement goal, is_collect, craft_action_or_None, readable name)
ITEM_INFO = {
    WOOD:    (A_WOOD,  True,  None, "wood"),
    STONE_I: (A_STONE, True,  None, "stone"),
    COAL_I:  (A_COAL,  True,  None, "coal"),
    IRON_I:  (A_IRON,  True,  None, "iron"),
    TABLE:   (None,    False, 5,    "table"),
    WPICK:   (None,    False, 6,    "wood_pickaxe"),
    SPICK:   (None,    False, 7,    "stone_pickaxe"),
    FURNACE: (None,    False, 8,    "furnace"),
    IPICK:   (None,    False, 9,    "iron_pickaxe"),
}
# achievement index for craft items (obtaining them = the craft achievement)
from ragnarok.environments.craft_world import (A_TABLE, A_WPICK, A_SPICK,
                                               A_FURNACE, A_IPICK)
CRAFT_ACH = {TABLE: A_TABLE, WPICK: A_WPICK, SPICK: A_SPICK,
             FURNACE: A_FURNACE, IPICK: A_IPICK}
ALL = list(ITEM_INFO.keys())
NAME = {i: ITEM_INFO[i][3] for i in ALL}
COLLECT_CELL = {WOOD: 1, STONE_I: 2, COAL_I: 3, IRON_I: 4}   # cell type per resource

# ground-truth direct item-preconditions (for scoring recovery only)
TRUE_PRE = {
    WOOD: set(), TABLE: {WOOD}, WPICK: {WOOD, TABLE}, STONE_I: {WPICK},
    COAL_I: {WPICK}, SPICK: {WOOD, STONE_I, TABLE}, FURNACE: {STONE_I, TABLE},
    IRON_I: {SPICK}, IPICK: {WOOD, COAL_I, IRON_I, TABLE, FURNACE},
}


def _grant_vec(items):
    g = [0] * N_ITEMS
    for i in items:
        g[i] = 3
    return g


def _goal_of(item):
    ach, is_collect, _, _ = ITEM_INFO[item]
    return ach if is_collect else CRAFT_ACH[item]


@torch.no_grad()
def _attempt(item, granted, skills, cfg, n=128):
    """Success fraction of obtaining `item` given `granted` items stocked."""
    goal = _goal_of(item)
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal,
                              grant=_grant_vec(granted))
    _, is_collect, craft_a, _ = ITEM_INFO[item]
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state                       # goal-conditioned env -> already 168-d
    for _ in range(cfg["max_steps"]):
        if is_collect:
            a = skills[item].act(obs, deterministic=True)
        else:
            a = torch.full((n,), craft_a, dtype=torch.long, device=DEVICE)
        obs, _, term, _, _ = env.step(a)
        ever |= term
    return float(ever.float().mean().item())


def _learn_preconditions(skills, cfg):
    """Leave-one-out necessity test per item -> learned precondition sets."""
    learned = {}
    others_all = set(ALL)
    for I in ALL:
        cand = [x for x in ALL if x != I]
        base = _attempt(I, set(cand), skills, cfg)          # all others granted
        pre = set()
        detail = {}
        for X in cand:
            s = _attempt(I, set(cand) - {X}, skills, cfg)    # leave X out
            detail[NAME[X]] = round(s, 2)
            if base > 0.5 and s < 0.15:                      # X is necessary
                pre.add(X)
        learned[I] = pre
        print(f"  [{NAME[I]:14s}] base {base:.2f} | precond "
              f"{sorted(NAME[x] for x in pre)}", flush=True)
    return learned


def _plan(target, ops):
    """BFS over learned operators: op applicable when precond ⊆ have."""
    start = frozenset()
    if target in start:
        return []
    seen = {start}
    q = deque([(start, [])])
    while q:
        have, path = q.popleft()
        for it in ALL:
            if it in have or not ops[it] <= have:
                continue
            nh = have | {it}
            npath = path + [it]
            if it == target:
                return npath
            if nh not in seen:
                seen.add(nh); q.append((nh, npath))
    return None


@torch.no_grad()
def _execute_plan(plan, skills, cfg, n=256):
    """Follow the planned item order in the FULL env (no grant); quantity-
    aware collects; return fraction obtaining the final target."""
    target = plan[-1]
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=10 ** 9)
    for item in plan:
        _, is_collect, craft_a, goal = (*ITEM_INFO[item][:3], _goal_of(item))
        for _ in range(cfg["exec_per_item"]):
            obs = env.state                          # non-goal env -> 159-d
            if is_collect:                           # append goal onehot -> 168
                a = skills[item].act(torch.cat([obs, _goal_onehot(goal, n)], -1),
                                     deterministic=True)
            else:
                a = torch.full((n,), craft_a, dtype=torch.long, device=DEVICE)
            env.step(a)
    return float((env.inv[:, target] > 0).float().mean().item())


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
    p.add_argument("--exec-per-item", type=int, default=40)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.skill_cap, args.max_steps = 64, 10, 60
        args.exec_per_item = 20

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap", "exec_per_item")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[model-based-v9] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    print("\n[skills] training the 4 reusable collect skills...", flush=True)
    skills = {}
    for item in (WOOD, STONE_I, COAL_I, IRON_I):
        ppo, s = _train_collect(COLLECT_CELL[item], cfg)
        skills[item] = ppo
        print(f"  collect_{NAME[item]:6s} skill {s:.2f}", flush=True)

    print("\n[rules] learning operator preconditions (leave-one-out)...", flush=True)
    learned = _learn_preconditions(skills, cfg)

    # rule recovery vs ground truth
    tp = fp = fn = 0
    for I in ALL:
        L, T = learned[I], TRUE_PRE[I]
        tp += len(L & T); fp += len(L - T); fn += len(T - L)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    exact = sum(1 for I in ALL if learned[I] == TRUE_PRE[I])

    print("\n[plan] BFS to every target from the LEARNED model:", flush=True)
    plans = {}
    for I in ALL:
        pl = _plan(I, learned)
        plans[I] = [NAME[x] for x in pl] if pl is not None else None
        print(f"  -> {NAME[I]:14s}: {plans[I]}", flush=True)
    all_planned = all(plans[I] is not None for I in ALL)

    # execute the deepest plan (iron_pickaxe)
    print("\n[exec] executing the learned plan for iron_pickaxe...", flush=True)
    ipick_plan = _plan(IPICK, learned)
    exec_succ = _execute_plan(ipick_plan, skills, cfg) if ipick_plan else 0.0

    print(f"\n{'=' * 74}\n  v9.0 MODEL-BASED (learn rules -> plan)")
    print(f"{'=' * 74}")
    print(f"  rule recovery vs ground truth: precision {prec:.2f} recall {rec:.2f}"
          f" | exact {exact}/{len(ALL)} items")
    print(f"  planner solved ALL {len(ALL)} targets: {all_planned}")
    print(f"  execute iron_pickaxe plan: {exec_succ:.2f} | flat PPO 0.11")
    ok = prec >= 0.95 and rec >= 0.95 and all_planned and exec_succ >= 0.6
    verdict = ("MODEL-BASED WORKS — the agent LEARNED the recipe DAG from "
               "interaction (precision/recall ~1.0), PLANS to any target "
               "zero-shot incl. iron_pickaxe, and executes far past flat. "
               "Understand the world -> any goal is a planning problem."
               if ok else
               f"PARTIAL/CHECK — prec {prec:.2f} rec {rec:.2f} planned "
               f"{all_planned} exec {exec_succ:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v9.json"), "w") as f:
        json.dump(dict(learned={NAME[I]: sorted(NAME[x] for x in learned[I])
                                 for I in ALL},
                       precision=prec, recall=rec, exact=exact,
                       plans=plans, all_planned=all_planned,
                       exec_iron_pickaxe=exec_succ, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
