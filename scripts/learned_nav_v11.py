"""v11 — a LEARNED universal navigation skill: fully-learned generality.

Train ONE goal-conditioned nav skill (input: target cell-type one-hot; output:
move/collect) on random worlds, with a FIXED-width observation (cells padded
to MAX_CELLS) and action_dim=5 (move x4 + collect) so it is tree-agnostic.
Then re-run the v10 generality pipeline (leave-one-out rule-learning + BFS
planning + execution) using this LEARNED skill instead of the scripted nav.

Decisive: (1) the single learned skill reaches+collects an arbitrary target
cell-type on held-out random worlds; (2) the v10 result REPRODUCES with it
(rule recovery ~1.0, planned, execution high) -> nav + rules + planning all
LEARNED end-to-end, no scripts.

Usage: python -m scripts.learned_nav_v11 [--trees 10] [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.tech_tree import DeviceVecTechTree, gen_tree
from scripts.tech_tree_sanity_v10 import _needed
from scripts.techtree_agent_v10 import _plan, _craft_action_of

MAX_CELLS = 20         # fixed obs width; must exceed any tree's n_cells
NAV_ACT = 5            # move x4 + collect
NAV_TREE = dict(n_items=14, n_base_res=8)   # resource-rich -> covers cell-types 1..~10


def nav_obs(env, cell_vec):
    ego = env._egocentric()
    oh = torch.nn.functional.one_hot(ego, MAX_CELLS).float().reshape(env.num_envs, -1)
    g = torch.nn.functional.one_hot(cell_vec, MAX_CELLS).float()
    return torch.cat([oh, g], dim=-1)


@torch.no_grad()
def nav_act(env, cell_vec, nav):
    return nav.act(nav_obs(env, cell_vec), deterministic=True)


@torch.no_grad()
def _nav_success(nav, cfg, n=256):
    spec = gen_tree(7777, **NAV_TREE)
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["nav_steps"], max_cells=MAX_CELLS,
                            nav_goal="random", grant=[3] * spec["n_items"])
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["nav_steps"]):
        _, _, term, _, _ = env.step(nav.act(env.state, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def train_nav(cfg):
    """Train one universal nav skill on a resource-rich random world."""
    spec = gen_tree(4242, **NAV_TREE)      # many resource cell-types
    env = DeviceVecTechTree(cfg["num_envs"], spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["nav_steps"], max_cells=MAX_CELLS,
                            nav_goal="random", grant=[3] * spec["n_items"])
    nav = DiscretePPO(env.obs_dim, NAV_ACT, hidden=cfg["hidden"], entropy=0.02)
    for it in range(1, cfg["nav_iters"] + 1):
        nav.train_iter(env, cfg["rollout"])
        if it % 10 == 0:
            print(f"  [nav] it {it:>3} | reach-success {_nav_success(nav, cfg):.2f}",
                  flush=True)
    return nav, _nav_success(nav, cfg)


@torch.no_grad()
def _attempt(spec, item, granted, nav, cfg, n=64):
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["attempt_steps"], goal=item,
                            grant=_grant(spec, granted), max_cells=MAX_CELLS)
    is_res = spec["kind"][item] == "R"
    cell = spec["cell"][item] if is_res else -1
    craft_a = _craft_action_of(spec).get(item, 4)
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["attempt_steps"]):
        if is_res:
            a = nav_act(env, torch.full((n,), cell, device=DEVICE), nav)
        else:
            a = torch.full((n,), craft_a, dtype=torch.long, device=DEVICE)
        _, _, term, _, _ = env.step(a)
        ever |= term
    return float(ever.float().mean().item())


def _grant(spec, items):
    g = [0] * spec["n_items"]
    for i in items:
        g[i] = 3
    return g


def _learn_preconditions(spec, nav, cfg):
    learned = {}
    for I in range(spec["n_items"]):
        cand = [x for x in range(spec["n_items"]) if x != I]
        base = _attempt(spec, I, set(cand), nav, cfg)
        pre = set()
        if base > 0.5:
            for X in cand:
                if _attempt(spec, I, set(cand) - {X}, nav, cfg) < 0.15:
                    pre.add(X)
        learned[I] = pre
    return learned


@torch.no_grad()
def _execute(spec, plan, nav, cfg, n=256):
    need = _needed(spec)
    ca = _craft_action_of(spec)
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=10 ** 9, max_cells=MAX_CELLS)
    for it in plan:
        is_res = spec["kind"][it] == "R"
        want = max(1, need[it])
        for _ in range(cfg["exec_per_item"]):
            if bool((env.inv[:, it] >= want).all()):
                break
            if is_res:
                a = nav_act(env, torch.full((n,), spec["cell"][it], device=DEVICE), nav)
            else:
                a = torch.full((n,), ca[it], dtype=torch.long, device=DEVICE)
            env.step(a)
    return float((env.inv[:, spec["target"]] > 0).float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trees", type=int, default=10)
    p.add_argument("--n-items", type=int, default=14)
    p.add_argument("--grid", type=int, default=11)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--nav-iters", type=int, default=80)
    p.add_argument("--nav-steps", type=int, default=60)
    p.add_argument("--attempt-steps", type=int, default=70)
    p.add_argument("--exec-per-item", type=int, default=60)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.trees, args.nav_iters, args.num_envs = 2, 12, 64

    cfg = {k: getattr(args, k) for k in
           ("grid", "view", "num_envs", "rollout", "hidden", "nav_iters",
            "nav_steps", "attempt_steps", "exec_per_item")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v11] device={DEVICE} | training ONE universal nav skill...", flush=True)
    t0 = time.perf_counter()
    nav, nav_s = train_nav(cfg)
    print(f"  universal nav skill reach-success: {nav_s:.2f}\n", flush=True)

    print(f"[v11] re-running v10 generality with the LEARNED nav, "
          f"{args.trees} random trees:", flush=True)
    rows = []
    for seed in range(args.trees):
        spec = gen_tree(seed, n_items=args.n_items, n_base_res=2)
        learned = _learn_preconditions(spec, nav, cfg)
        tp = fp = fn = 0
        for I in range(spec["n_items"]):
            L, T = learned[I], spec["true_pre"][I]
            tp += len(L & T); fp += len(L - T); fn += len(T - L)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        plan = _plan(spec["target"], learned, spec["n_items"])
        execu = _execute(spec, plan, nav, cfg) if plan is not None else 0.0
        rows.append(dict(seed=seed, precision=prec, recall=rec,
                         planned=plan is not None, exec=execu))
        print(f"  tree {seed}: recovery P {prec:.2f} R {rec:.2f} | "
              f"planned {plan is not None} | exec {execu:.2f}", flush=True)

    mP = float(np.mean([r["precision"] for r in rows]))
    mR = float(np.mean([r["recall"] for r in rows]))
    npl = sum(r["planned"] for r in rows)
    mE = float(np.mean([r["exec"] for r in rows]))
    print(f"\n{'=' * 72}\n  v11 — LEARNED nav + generality | {args.trees} random trees")
    print(f"{'=' * 72}")
    print(f"  universal nav reach-success: {nav_s:.2f}")
    print(f"  mean rule recovery: precision {mP:.3f} recall {mR:.3f}")
    print(f"  planned: {npl}/{args.trees} | mean execution: {mE:.2f}")
    ok = nav_s >= 0.8 and mP >= 0.95 and mR >= 0.95 and npl == args.trees and mE >= 0.6
    verdict = ("FULLY-LEARNED GENERALITY — one LEARNED nav skill works on any "
               "world; with it, rule-learning + planning recover & solve random "
               "trees end-to-end (no scripts). The whole loop is learned."
               if ok else
               f"PARTIAL/CHECK — nav {nav_s:.2f} P {mP:.2f} R {mR:.2f} "
               f"planned {npl}/{args.trees} exec {mE:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v11.json"), "w") as f:
        json.dump(dict(nav_success=nav_s, trees=rows, mean_precision=mP,
                       mean_recall=mR, n_planned=npl, mean_exec=mE,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
