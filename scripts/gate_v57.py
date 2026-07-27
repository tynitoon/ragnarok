"""v57 GATE — ~1 GPU-hour that decides whether the remaining ~18 are spent at all.

Pre-committed question: with credit given ONLY to the commanded goal, is a knowledge-free control even
CONSTRUCTIBLE in this codebase, or does the fix starve the mechanism outright?

The leak meter already established the defect (99.69% of gradient steps at depth were about something
other than the commanded goal). The arithmetic of the fix is brutal: at depth only ~5-23 of 1024
env-episodes reach the goal, so commanded-only credit yields ~20-80 samples/round against v55's 8k-33k.

KILL-1 (pre-committed, fires HERE): under the fixed rule, the accumulating arm A masters NONE of the
shallow goals it mastered in v55. Meaning: the credit fix breaks the agent rather than the control, a
knowledge-free control is not constructible here, and the line ENDS having spent ~1 GPU-hour.
KILL-2 (pre-committed, fires HERE): the fixed rule yields < 20 samples/round at depth for BOTH arms, i.e.
there is no channel through which goal-conditioned knowledge could be learned at depth. The line ENDS.

Neither outcome licenses "try a bigger world / a longer chain / a new substrate" — that response is
pre-forbidden in writing, because it is exactly the move that produced v49 through v55.

Usage: python -m scripts.gate_v57 [--seed 0] [--n-goals 5]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.hidden_recipe_v55 import permute_spec, admitted_goals, Composer, Buffer
from scripts.credit_fix_v57 import run_goal_commanded

WORLDS = {0: 3002, 1: 3003, 2: 3016}


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-goals", type=int, default=5, help="prefix of the ascending stream to gate on")
    p.add_argument("--r-max", type=int, default=10)
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = dict(num_envs=256, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02, nav_max_steps=40,
               skill_iters=400, option_timeout=16, macro_budget=26, episodes_per_round=4,
               train_steps_per_round=300, max_samples_per_ep=8192, epsilon=0.05, temp=1.0,
               thresh=0.6, r_max=a.r_max, skill_stochastic=True, mgr_entropy=0.03, router_iters=0)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()

    w = WORLDS[a.seed]
    spec = permute_spec(gen_tree(w, n_items=14), w)
    adm = admitted_goals(spec)
    pc = {g: c for g, c, _ in adm}
    goals = [g for g, _, _ in adm][:a.n_goals]
    deep = [g for g, c, _ in adm if c >= 10][:2]                 # 2 deep goals for the KILL-2 check
    skill = load_skill(cfg, a.seed, a.out_dir)
    v55 = json.load(open(os.path.join(a.out_dir, f"v55_s{a.seed}.json")))
    was = {r["goal"]: r for r in v55["arms"]["A"]["rows"]}

    print("=" * 100)
    print(f"v57 GATE | world {w} | credit ONLY the commanded goal | {a.r_max} rounds/goal")
    print(f"stream {goals} (pc {[pc[g] for g in goals]}) + deep probes {deep} (pc {[pc[g] for g in deep]})")
    print("=" * 100, flush=True)

    res = dict(seed=a.seed, world=w, goals=goals, deep=deep, A=[], K=[])

    # ---- arm A: accumulating, fixed credit rule ---------------------------------------------------
    print("\n  ARM A (accumulating, commanded-goal credit only)", flush=True)
    cA, bA = Composer("memory"), Buffer()
    for g in goals + deep:
        r = run_goal_commanded(spec, skill, cA, bA, cfg, a.seed + 11 * goals.index(g) + 1 if g in goals
                               else a.seed + 900 + g, g)
        r["v55_rounds"] = was.get(g, {}).get("rounds"); r["v55_mastered"] = was.get(g, {}).get("mastered")
        res["A"].append(r)
        print(f"    goal {g:>2} (pc {pc[g]:>2}): master {r['master']:.2f} in {r['rounds']:>2}r "
              f"({'M' if r['mastered'] else 'x'}) | demos/r {r['demos_per_round']} | "
              f"samples/r {r['samples_per_round']} | v55 was "
              f"{'M' if r['v55_mastered'] else 'x'}{r['v55_rounds']}r | {time.perf_counter()-t0:.0f}s",
              flush=True)

    # ---- arm K: knowledge-free per goal, fixed credit rule ----------------------------------------
    print("\n  ARM K (fresh composer+buffer per goal — the first genuinely knowledge-free control)",
          flush=True)
    for g in goals + deep:
        r = run_goal_commanded(spec, skill, Composer("memory"), Buffer(cap=400_000), cfg,
                               a.seed + 11 * goals.index(g) + 1 if g in goals else a.seed + 900 + g, g)
        res["K"].append(r)
        print(f"    goal {g:>2} (pc {pc[g]:>2}): master {r['master']:.2f} in {r['rounds']:>2}r "
              f"({'M' if r['mastered'] else 'x'}) | demos/r {r['demos_per_round']} | "
              f"samples/r {r['samples_per_round']} | {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- gate ------------------------------------------------------------------------------------
    shallow_A = [r for r in res["A"] if r["goal"] in goals]
    a_master = sum(r["mastered"] for r in shallow_A)
    a_was = sum(1 for r in shallow_A if r["v55_mastered"])
    deep_samp = [min(r["samples_per_round"]) if r["samples_per_round"] else 0
                 for r in res["A"] + res["K"] if r["goal"] in deep]
    kill1 = a_master == 0 and a_was > 0
    kill2 = bool(deep_samp) and max(deep_samp) < 20
    res.update(a_master=a_master, a_was=a_was, deep_min_samples=deep_samp,
               kill1=kill1, kill2=kill2, proceed=not (kill1 or kill2))

    print(f"\n{'='*100}")
    print(f"  arm A shallow mastery under the fixed rule: {a_master}/{len(shallow_A)}   "
          f"(v55, leaky rule: {a_was}/{len(shallow_A)})")
    print(f"  min samples/round on the deep probes: {deep_samp}  (v55 collected 8k-33k per round)")
    if kill1:
        print("\n!! KILL-1 FIRES — the credit fix breaks the AGENT, not the control. A knowledge-free")
        print("   control is not constructible in this codebase. THE LINE ENDS HERE (~1 GPU-hour spent).")
    elif kill2:
        print("\n!! KILL-2 FIRES — under honest credit there is no channel at depth at all (<20 samples/")
        print("   round). Goal-conditioned knowledge cannot be learned there. THE LINE ENDS HERE.")
    else:
        print("\n  GATE PASSED — a knowledge-free control exists and there is signal at depth.")
        print("  Proceed to freeze the v57 prereg and the scorer BEFORE any confirmatory arm.")
    print("=" * 100, flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"v57_gate_s{a.seed}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
