"""v50 — CHILDHOOD AMORTISATION: does a library learned ONCE make a STREAM of novel tasks cheaper?

The developmental thesis, made measurable. CHILDHOOD: train ONE tree-agnostic nav-collect skill on a
DISTRIBUTION of procedural trees (so it generalises). ADULTHOOD: a stream of HELD-OUT trees (unseen
tasks). For each, WARM = a manager that REUSES the childhood skill masters the tree's deep target;
FLAT = PPO from scratch masters it. Cost = primitive env-steps to reach the success threshold.
AMORTISATION: cumulative warm (C_library_once + sum of cheap managers) vs cumulative flat (sum of full
from-scratch). Childhood pays off if warm_cum crosses BELOW flat_cum after a small break-even number of
tasks. This is the regime where the field's FAIR reuse wins actually live (amortise over a distribution).

Key enabler: the nav skill is fully transferable — obs padded to MAX_CELLS (constant dim) and its action
space restricted to move+collect (5, constant across trees), so ONE skill applies to ANY tree.

Usage: python -m scripts.childhood_v50 --check        # validate nav transfers to held-out trees
       python -m scripts.childhood_v50 [full amortisation run]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.tech_tree import gen_tree, DeviceVecTechTree
from scripts.depth_scaling_v49 import (TechTreeConvNet, MAX_CELLS, N_ITEMS_FOR_DEPTH,
                                       TreeManagerEnv, eval_target)

NAV_ACTIONS = 5            # move(0-3)+collect(4) only -> constant action space across all trees


def manager_cost(spec, skill, cfg, seed, budget):
    """Train a manager (reusing `skill`) until it MASTERS the tree's target (success>=thresh) or budget.
    Returns primitive-step cost-to-master (or budget if never), mastered flag, final success."""
    torch.manual_seed(seed + 100)
    env = TreeManagerEnv(cfg["num_envs"], spec, skill, cfg, seed=seed + 100)
    mgr = DiscretePPO(env.obs_dim, env.action_dim, hidden=cfg["mgr_hidden"],
                      entropy=cfg["mgr_entropy"], gamma=0.99, lam=0.95)
    env._prim = 0
    final = 0.0
    while env._prim < budget:
        for _ in range(cfg["check_every"]):
            mgr.train_iter(env, cfg["macro_budget"])
        final = eval_target(mgr, spec, cfg, seed, "compose", skill=skill, n=256)
        if final >= cfg["thresh"]:
            return dict(cost=env._prim, mastered=True, final=round(final, 3))
    return dict(cost=env._prim, mastered=False, final=round(final, 3))


def fresh_skill(spec, cfg, seed):
    """Train a 1-tree nav-collect skill FROM SCRATCH (no childhood). Returns (skill, primitive steps)."""
    return train_childhood([spec], cfg, seed, log=False)


def flat_cost(spec, cfg, seed, budget):
    """Train end-to-end flat PPO (per-achievement novelty + entropy) until it MASTERS the target or budget."""
    torch.manual_seed(seed + 200)
    env = DeviceVecTechTree(cfg["num_envs"], spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["flat_max_steps"], n_resource=cfg["n_resource"],
                            max_cells=MAX_CELLS, seed=seed + 200)
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, spec["n_items"], env.action_dim)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, net=net, entropy=cfg["flat_entropy"],
                      gamma=0.99, lam=0.95)
    steps = 0
    final = 0.0
    chunk = cfg["check_every"] * 4
    while steps < budget:
        for _ in range(chunk):
            ppo.train_iter(env, cfg["rollout"]); steps += cfg["num_envs"] * cfg["rollout"]
        final = eval_target(ppo, spec, cfg, seed, "flat", n=256)
        if final >= cfg["thresh"]:
            return dict(cost=steps, mastered=True, final=round(final, 3))
    return dict(cost=steps, mastered=False, final=round(final, 3))


def nav_env(spec, cfg, seed, n):
    return DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"], max_steps=cfg["nav_max_steps"],
                             n_resource=cfg["n_resource"], nav_goal="random", max_cells=MAX_CELLS,
                             grant=[1] * spec["n_items"], seed=seed)


def train_childhood(train_specs, cfg, seed, log=False):
    """ONE tree-agnostic nav-collect skill trained on a DISTRIBUTION of trees (rotated). 5 actions."""
    torch.manual_seed(seed)
    envs = [nav_env(s, cfg, seed + 1 + i, cfg["num_envs"]) for i, s in enumerate(train_specs)]
    obs_dim = envs[0].obs_dim
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(obs_dim, NAV_ACTIONS, net=net, entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    steps = 0
    for it in range(1, cfg["skill_iters"] + 1):
        env = envs[it % len(envs)]
        ppo.train_iter(env, cfg["rollout"])
        steps += cfg["num_envs"] * cfg["rollout"]
        if log and (it % max(1, cfg["skill_iters"] // 8) == 0 or it == cfg["skill_iters"]):
            tr = sum(nav_success_on(ppo, s, cfg, seed) for s in train_specs) / len(train_specs)
            print(f"    [childhood] it {it:>4} | train-tree nav {tr:.2f} | {steps/1e6:.2f}M", flush=True)
    return ppo, steps


@torch.no_grad()
def nav_success_on(ppo, spec, cfg, seed, n=256):
    """Fraction collecting the (random) target cell-type on THIS tree within nav_max_steps."""
    env = nav_env(spec, cfg, seed + 777, n)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["nav_max_steps"]):
        obs, r, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        got |= term
    return float(got.float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=7)        # depth level (n_items) of the trees
    p.add_argument("--n-train-trees", type=int, default=4)
    p.add_argument("--n-heldout", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--skill-iters", type=int, default=300)         # childhood (on the train trees)
    p.add_argument("--scratch-skill-iters", type=int, default=180)  # per-tree fresh skill (no childhood)
    p.add_argument("--mgr-hidden", type=int, default=128)
    p.add_argument("--mgr-entropy", type=float, default=0.03)
    p.add_argument("--macro-budget", type=int, default=20)
    p.add_argument("--option-timeout", type=int, default=14)
    p.add_argument("--thresh", type=float, default=0.8)            # "mastered" = target success >= thresh
    p.add_argument("--check-every", type=int, default=10)          # manager iters between success checks
    p.add_argument("--budget-mgr", type=float, default=8e6)        # cap on manager cost-to-master
    p.add_argument("--budget-flat", type=float, default=8e6)       # cap on flat cost-to-master
    p.add_argument("--flat-max-steps", type=int, default=200)
    p.add_argument("--flat-eval-steps", type=int, default=200)
    p.add_argument("--flat-entropy", type=float, default=0.03)
    p.add_argument("--no-flat", action="store_true", help="skip the (slow) flat baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--check", action="store_true", help="only validate nav transfer to held-out trees")
    args = p.parse_args()

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "n_resource", "rollout", "entropy", "nav_max_steps", "skill_iters",
            "mgr_hidden", "mgr_entropy", "macro_budget", "option_timeout", "thresh", "check_every",
            "flat_max_steps", "flat_eval_steps", "flat_entropy")}
    os.makedirs(args.out_dir, exist_ok=True)
    n_items = N_ITEMS_FOR_DEPTH[args.depth]
    # disjoint train / held-out tree seeds
    train_specs = [gen_tree(1000 + i, n_items=n_items) for i in range(args.n_train_trees)]
    heldout_specs = [gen_tree(5000 + i, n_items=n_items) for i in range(args.n_heldout)]

    print(f"[v50 childhood] device={DEVICE} | depth~{args.depth} (n_items {n_items}) | "
          f"{args.n_train_trees} train trees, {args.n_heldout} held-out | skill-iters {args.skill_iters}",
          flush=True)
    t0 = time.perf_counter()
    skill, c_lib = train_childhood(train_specs, cfg, args.seed, log=True)

    # KEY ASSUMPTION CHECK: does the childhood skill navigate on HELD-OUT (unseen) trees zero-shot?
    tr_nav = [round(nav_success_on(skill, s, cfg, args.seed), 3) for s in train_specs]
    ho_nav = [round(nav_success_on(skill, s, cfg, args.seed), 3) for s in heldout_specs]
    print(f"\n  nav on TRAIN trees:    {tr_nav} (mean {sum(tr_nav)/len(tr_nav):.2f})", flush=True)
    print(f"  nav on HELD-OUT trees: {ho_nav} (mean {sum(ho_nav)/len(ho_nav):.2f})", flush=True)
    transfers = sum(ho_nav) / len(ho_nav) >= 0.8
    print(f"  -> childhood skill {'GENERALISES' if transfers else 'does NOT generalise'} to held-out "
          f"trees (lib cost {c_lib/1e6:.2f}M steps) | {time.perf_counter()-t0:.0f}s", flush=True)
    if args.check:
        with open(os.path.join(args.out_dir, "v50_navcheck.json"), "w") as f:
            json.dump(dict(depth=args.depth, c_lib=c_lib, train_nav=tr_nav, heldout_nav=ho_nav,
                           transfers=transfers), f, indent=2)
        return

    # ADULTHOOD: a stream of HELD-OUT trees. WARM reuses the childhood skill; SCRATCH relearns the skill
    # per tree (isolates childhood's value); FLAT is end-to-end. Cost = primitive steps to master target.
    print(f"\n  [adulthood] {len(heldout_specs)} held-out trees | thresh {args.thresh} | "
          f"c_library {c_lib/1e6:.2f}M (paid once)", flush=True)
    cfg_scr = {**cfg, "skill_iters": args.scratch_skill_iters}
    rows = []
    for i, spec in enumerate(heldout_specs):
        sd = args.seed + 1 + i
        warm = manager_cost(spec, skill, cfg, sd, args.budget_mgr)
        fresh, c_skill = fresh_skill(spec, cfg_scr, sd)
        scr = manager_cost(spec, fresh, cfg, sd, args.budget_mgr)
        scratch_cost = c_skill + scr["cost"]
        flat = (dict(cost=0, mastered=None, final=None) if args.no_flat
                else flat_cost(spec, cfg, sd, args.budget_flat))
        rows.append(dict(tree=i, true_depth=int(spec["depth"][spec["target"]]),
                         warm_cost=warm["cost"], warm_master=warm["mastered"], warm_final=warm["final"],
                         scratch_cost=scratch_cost, scratch_skill=c_skill, scratch_master=scr["mastered"],
                         flat_cost=flat["cost"], flat_master=flat["mastered"], flat_final=flat["final"]))
        print(f"    tree {i} (d{rows[-1]['true_depth']}): WARM {warm['cost']/1e6:.2f}M "
              f"(master {warm['mastered']}, {warm['final']:.2f}) | SCRATCH {scratch_cost/1e6:.2f}M "
              f"(skill {c_skill/1e6:.2f}M) | FLAT {flat['cost']/1e6:.2f}M (master {flat['mastered']}) "
              f"| {time.perf_counter()-t0:.0f}s", flush=True)

    # cumulative amortisation: childhood pays C_lib once, then cheap managers
    warm_cum = c_lib; scr_cum = 0.0; flat_cum = 0.0; breakeven = None
    for k, r in enumerate(rows, 1):
        warm_cum += r["warm_cost"]; scr_cum += r["scratch_cost"]; flat_cum += r["flat_cost"]
        if breakeven is None and warm_cum < scr_cum:
            breakeven = k
    warm_master_rate = sum(r["warm_master"] for r in rows) / len(rows)
    flat_master_rate = (None if args.no_flat else sum(bool(r["flat_master"]) for r in rows) / len(rows))
    # POSITIVE: warm masters most held-out trees AND cumulative warm crosses below scratch within the
    # stream (childhood amortises) AND warm is cheaper PER tree on average (the per-task saving is real).
    mean_warm = sum(r["warm_cost"] for r in rows) / len(rows)
    mean_scr = sum(r["scratch_cost"] for r in rows) / len(rows)
    positive = (warm_master_rate >= 0.8 and breakeven is not None and breakeven < len(rows)
                and mean_warm < mean_scr)
    verdict = (
        f"CHILDHOOD AMORTISES (v50) — one reusable skill (cost {c_lib/1e6:.2f}M, learned on OTHER trees) "
        f"lets a manager master {warm_master_rate:.0%} of HELD-OUT trees; cumulative WARM crosses BELOW "
        f"SCRATCH after {breakeven} trees (warm {warm_cum/1e6:.1f}M vs scratch {scr_cum/1e6:.1f}M at "
        f"{len(rows)} trees" + (f", vs flat {flat_cum/1e6:.1f}M, flat-master {flat_master_rate:.0%}"
        if not args.no_flat else "") + "). Reuse over a distribution makes new tasks cheaper. REVIEW."
        if positive else
        f"PARTIAL/CHECK — warm-master {warm_master_rate:.0%}, break-even {breakeven}, "
        f"warm_cum {warm_cum/1e6:.1f}M vs scratch_cum {scr_cum/1e6:.1f}M"
        + (f" vs flat_cum {flat_cum/1e6:.1f}M" if not args.no_flat else "") + ".")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"v50_amortise_s{args.seed}.json"), "w") as f:
        json.dump(dict(depth=args.depth, c_lib=c_lib, heldout_nav=ho_nav, rows=rows,
                       warm_cum=warm_cum, scratch_cum=scr_cum, flat_cum=flat_cum, breakeven=breakeven,
                       warm_master_rate=warm_master_rate, positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
