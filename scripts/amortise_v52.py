"""v52 — the FAIR amortisation (reviews' #1 ask): does CHILDHOOD (train the skill ONCE on a
distribution) beat training the skill FRESH per task, when composition is the SAME fixed greedy planner
in BOTH arms? This isolates the ONLY honest reuse quantity left: skill-training amortisation. No
overclaim — composition is a fixed (trivially-correct on this substrate) planner; the planner is a
constant in both arms, so the comparison is purely skill cost.

WARM(k)    = C_library (8-tree childhood, paid ONCE) + 0 adulthood   [reused skill + greedy]
SCRATCH(k) = sum over k of (C_skill_fresh + 0 adulthood)             [fresh 1-tree skill + greedy]
Both must MASTER the held-out tree (else the cost is moot). Report master rates + costs + break-even.

Usage: python -m scripts.amortise_v52 [--n-heldout 6]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.depth_scaling_v49 import N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood
from scripts.meta_manager_v51 import greedy_master_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--n-train-trees", type=int, default=8)
    p.add_argument("--n-heldout", type=int, default=6)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--childhood-iters", type=int, default=400)
    p.add_argument("--scratch-iters", type=int, default=300)   # fresh per-tree skill
    p.add_argument("--option-timeout", type=int, default=40)
    p.add_argument("--macro-budget", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    args = p.parse_args()

    base = dict(num_envs=args.num_envs, grid=args.grid, view=args.view, n_resource=args.n_resource,
                rollout=args.rollout, entropy=args.entropy, nav_max_steps=args.nav_max_steps,
                option_timeout=args.option_timeout, macro_budget=args.macro_budget,
                skill_stochastic=True)
    cfg_child = {**base, "skill_iters": args.childhood_iters}
    cfg_scr = {**base, "skill_iters": args.scratch_iters}
    os.makedirs(args.out_dir, exist_ok=True)
    ni = N_ITEMS_FOR_DEPTH[args.depth]
    train_specs = [gen_tree(1000 + i, n_items=ni) for i in range(args.n_train_trees)]
    heldout = [gen_tree(5000 + i, n_items=ni) for i in range(args.n_heldout)]

    print(f"[v52 amortise] device={DEVICE} | depth~{args.depth} | childhood {args.n_train_trees} trees "
          f"vs fresh-per-tree | composition = SAME fixed greedy planner both arms", flush=True)
    t0 = time.perf_counter()
    skill, c_lib = train_childhood(train_specs, cfg_child, args.seed)
    print(f"  childhood skill cost C_lib = {c_lib/1e6:.2f}M | {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for i, spec in enumerate(heldout):
        sd = args.seed + 1 + i
        warm = greedy_master_rate(spec, skill, cfg_child, sd)                # reuse childhood skill
        fresh, c_skill = train_childhood([spec], cfg_scr, sd)                # fresh 1-tree skill
        scr = greedy_master_rate(spec, fresh, cfg_scr, sd)
        rows.append(dict(tree=i, depth=int(spec["depth"][spec["target"]]),
                         warm_master=round(warm, 3), scratch_master=round(scr, 3),
                         c_skill_fresh=c_skill))
        print(f"    tree {i} (d{rows[-1]['depth']}): WARM-master {warm:.2f} | SCRATCH-master {scr:.2f} "
              f"(fresh skill {c_skill/1e6:.2f}M) | {time.perf_counter()-t0:.0f}s", flush=True)

    # both arms have ZERO adulthood learning cost (greedy is free) -> cost is purely skill training
    mean_cskill = sum(r["c_skill_fresh"] for r in rows) / len(rows)
    warm_master = sum(r["warm_master"] for r in rows) / len(rows)
    scr_master = sum(r["scratch_master"] for r in rows) / len(rows)
    breakeven = c_lib / mean_cskill if mean_cskill > 0 else None   # k where C_lib < k*C_skill
    verdict = (
        f"FAIR SKILL-AMORTISATION — both arms MASTER held-out trees (warm {warm_master:.2f}, scratch "
        f"{scr_master:.2f}) via the same greedy planner; childhood pays C_lib={c_lib/1e6:.2f}M ONCE vs "
        f"fresh {mean_cskill/1e6:.2f}M PER tree -> warm cumulative < scratch after ~{breakeven:.1f} trees. "
        f"This is REAL but MODEST reuse (only the one-time skill cost is amortised; composition is a "
        f"fixed trivially-correct planner, not learned/reused)."
        if warm_master >= 0.8 and scr_master >= 0.8 else
        f"CHECK — warm-master {warm_master:.2f}, scratch-master {scr_master:.2f}. If scratch<warm, the "
        f"fresh 1-tree skill is too weak -> childhood ENABLES mastery scratch can't (stronger claim); "
        f"if both high, it's pure skill-cost amortisation (modest).")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"v52_amortise_s{args.seed}.json"), "w") as f:
        json.dump(dict(depth=args.depth, c_lib=c_lib, rows=rows, warm_master=warm_master,
                       scratch_master=scr_master, mean_c_skill=mean_cskill, breakeven_trees=breakeven,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
