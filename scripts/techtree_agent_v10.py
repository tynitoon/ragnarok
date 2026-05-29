"""v10 — GENERALITY: the model-based agent learns ARBITRARY random tech-tree
DAGs from interaction and plans to their target, on worlds nobody hand-built.

For each random tree: (1) LEARN each item's precondition by leave-one-out
necessity probing (recover the hidden DAG), (2) BFS-PLAN to the deepest item
from the learned model, (3) EXECUTE the plan. Navigation uses a tree-agnostic
scripted primitive (go to nearest cell of a type + collect; learned versions
shown in v6/v7/v9) so the SAME agent runs on every tree with no retraining.

Decisive over K random unseen trees: rule recovery precision/recall ~1.0,
valid plan to the target on every tree, execution >> random.

Usage: python -m scripts.techtree_agent_v10 [--trees 10] [--smoke]
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import DeviceVecTechTree, gen_tree
from scripts.tech_tree_sanity_v10 import _nearest_move, _needed, _bfs_order


def _grant_vec(spec, items):
    g = [0] * spec["n_items"]
    for i in items:
        g[i] = 3
    return g


def _craft_action_of(spec):
    return {it: 5 + k for k, it in enumerate(spec["craft_actions"])}


@torch.no_grad()
def _attempt(spec, item, granted, cfg, n=64):
    """Success fraction of obtaining `item` given `granted` items, with
    scripted navigation (resource) or the craft action (craft)."""
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["attempt_steps"], goal=item,
                            grant=_grant_vec(spec, granted))
    is_res = spec["kind"][item] == "R"
    cell = spec["cell"][item] if is_res else -1
    craft_a = _craft_action_of(spec).get(item, 4)
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["attempt_steps"]):
        if is_res:
            a = _nearest_move(env, torch.full((n,), cell, device=DEVICE))
        else:
            a = torch.full((n,), craft_a, dtype=torch.long, device=DEVICE)
        _, _, term, _, _ = env.step(a)
        ever |= term
    return float(ever.float().mean().item())


def _learn_preconditions(spec, cfg):
    """Leave-one-out necessity test per item -> learned precondition sets."""
    learned = {}
    for I in range(spec["n_items"]):
        cand = [x for x in range(spec["n_items"]) if x != I]
        base = _attempt(spec, I, set(cand), cfg)
        pre = set()
        if base > 0.5:
            for X in cand:
                if _attempt(spec, I, set(cand) - {X}, cfg) < 0.15:
                    pre.add(X)
        learned[I] = pre
    return learned


def _plan(target, ops, n_items):
    start = frozenset()
    seen = {start}
    q = deque([(start, [])])
    while q:
        have, path = q.popleft()
        for it in range(n_items):
            if it in have or not ops[it] <= have:
                continue
            nh = have | {it}
            if it == target:
                return path + [it]
            if nh not in seen:
                seen.add(nh); q.append((nh, path + [it]))
    return None


@torch.no_grad()
def _execute(spec, plan, cfg, n=256):
    """Follow the plan (scripted nav + crafts), quantity-aware; return the
    fraction obtaining the target."""
    need = _needed(spec)
    target = spec["target"]
    ca = _craft_action_of(spec)
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=10 ** 9)
    for it in plan:
        is_res = spec["kind"][it] == "R"
        want = max(1, need[it])
        for _ in range(cfg["exec_per_item"]):
            if bool((env.inv[:, it] >= want).all()):
                break
            if is_res:
                a = _nearest_move(env, torch.full((n,), spec["cell"][it], device=DEVICE))
            else:
                a = torch.full((n,), ca[it], dtype=torch.long, device=DEVICE)
            env.step(a)
    return float((env.inv[:, target] > 0).float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trees", type=int, default=10)
    p.add_argument("--n-items", type=int, default=14)
    p.add_argument("--grid", type=int, default=11)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--attempt-steps", type=int, default=70)
    p.add_argument("--exec-per-item", type=int, default=60)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.trees, args.attempt_steps = 2, 50

    cfg = {k: getattr(args, k) for k in ("grid", "view", "attempt_steps",
                                         "exec_per_item")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[techtree-v10] device={DEVICE} | {args.trees} random trees", flush=True)
    t0 = time.perf_counter()

    rows = []
    for seed in range(args.trees):
        spec = gen_tree(seed, n_items=args.n_items, n_base_res=2)
        learned = _learn_preconditions(spec, cfg)
        tp = fp = fn = 0
        for I in range(spec["n_items"]):
            L, T = learned[I], spec["true_pre"][I]
            tp += len(L & T); fp += len(L - T); fn += len(T - L)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        plan = _plan(spec["target"], learned, spec["n_items"])
        planned = plan is not None
        execu = _execute(spec, plan, cfg) if planned else 0.0
        depth = spec["depth"][spec["target"]]
        rows.append(dict(seed=seed, n_items=spec["n_items"], target_depth=depth,
                         precision=prec, recall=rec, planned=planned, exec=execu))
        print(f"  tree {seed}: items {spec['n_items']} target-depth {depth} | "
              f"recovery P {prec:.2f} R {rec:.2f} | planned {planned} | "
              f"exec {execu:.2f}", flush=True)

    mP = float(np.mean([r["precision"] for r in rows]))
    mR = float(np.mean([r["recall"] for r in rows]))
    n_planned = sum(r["planned"] for r in rows)
    mE = float(np.mean([r["exec"] for r in rows]))
    print(f"\n{'=' * 72}\n  v10 GENERALITY over {args.trees} RANDOM unseen trees")
    print(f"{'=' * 72}")
    print(f"  mean rule recovery: precision {mP:.3f} recall {mR:.3f}")
    print(f"  planned to target: {n_planned}/{args.trees}")
    print(f"  mean execution (reach target): {mE:.2f}")
    ok = mP >= 0.95 and mR >= 0.95 and n_planned == args.trees and mE >= 0.6
    verdict = ("GENERALITY HOLDS — the model-based agent learns ARBITRARY "
               "random tech-tree DAGs from interaction (precision/recall ~1.0), "
               "plans to every target, and executes — on worlds nobody "
               "hand-built. It develops in any tech-tree world, not just one."
               if ok else
               f"PARTIAL/CHECK — P {mP:.2f} R {mR:.2f} planned {n_planned}/"
               f"{args.trees} exec {mE:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v10.json"), "w") as f:
        json.dump(dict(trees=rows, mean_precision=mP, mean_recall=mR,
                       n_planned=n_planned, mean_exec=mE, verdict=verdict), f,
                  indent=2)


if __name__ == "__main__":
    main()
