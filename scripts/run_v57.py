"""v57 driver — HONEST CREDIT. Prereg FROZEN (preregistration.md) and scored by scripts/score_v57.py,
both committed BEFORE this ran.

Arms, all sharing the per-seed frozen childhood skill reloaded from v55 (never retrained):
  A  ACCUMULATING   — one composer + one buffer across the whole ascending goal stream
  K  KNOWLEDGE-FREE — fresh composer + buffer at every goal, identical budget, identical observation,
                      identical credit rule. The first genuinely knowledge-free control in this project.
  U  BUDGET CONTROL — fresh, R_max=60 (6x K), on the ONE deepest goal that A mastered and K was starved on

Usage: python -m scripts.run_v57 --seed 0 [--resume]
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
DEEP_PC = 10
STARVED_DEMOS = 50
U_ROUNDS = 60


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
    p.add_argument("--r-max", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=7.0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
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
    goals = [g for g, _, _ in adm]
    DEEP = [g for g in goals if pc[g] >= DEEP_PC]
    skill = load_skill(cfg, a.seed, a.out_dir)
    jp = os.path.join(a.out_dir, f"v57_s{a.seed}.json")

    res = dict(seed=a.seed, world=w, admitted=[(g, c, round(b, 2)) for g, c, b in adm],
               deep=DEEP, arms={})
    if a.resume and os.path.exists(jp):
        res = json.load(open(jp)); res["resumed"] = True
    done = lambda t: bool(res["arms"].get(t, {}).get("complete"))            # noqa: E731
    save = lambda: json.dump(res, open(jp, "w"), indent=2)                   # noqa: E731
    gseed = lambda g: a.seed + 11 * goals.index(g) + 1                       # noqa: E731

    print("=" * 100)
    print(f"v57 | world {w} | credit ONLY the commanded goal | {len(goals)} goals, "
          f"{len(DEEP)} deep (pc>={DEEP_PC})")
    print(f"  pc {[pc[g] for g in goals]}")
    print("=" * 100, flush=True)

    def arm(tag, accumulate):
        comp, buf = Composer("memory"), Buffer()
        rows = []
        for g in goals:
            if not accumulate:
                comp, buf = Composer("memory"), Buffer(cap=400_000)
            r = run_goal_commanded(spec, skill, comp, buf, cfg, gseed(g), g)
            rows.append(r)
            res["arms"][tag] = dict(rows=rows, complete=(g == goals[-1]))
            save()
            print(f"    [{tag}] goal {g:>2} (pc {pc[g]:>2}{', DEEP' if g in DEEP else '    '}): "
                  f"master {r['master']:.2f} in {r['rounds']:>2}r ({'M' if r['mastered'] else 'x'}) | "
                  f"demos {sum(r['demos_per_round'])} | buf {buf.n} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            if time.perf_counter() - t0 > a.max_hours * 3600:
                print(f"    !! {a.max_hours}h cap — aborting cleanly with state saved", flush=True)
                raise SystemExit(3)
        return rows

    if not done("A"):
        print("\n  ARM A (accumulating)", flush=True)
        arm("A", True)
    if not done("K"):
        print("\n  ARM K (knowledge-free: fresh composer+buffer per goal)", flush=True)
        arm("K", False)

    # ---- U: is the gap merely budget? ---------------------------------------------------------
    if "U" not in res["arms"]:
        rA = {r["goal"]: r for r in res["arms"]["A"]["rows"]}
        rK = {r["goal"]: r for r in res["arms"]["K"]["rows"]}
        cand = [g for g in DEEP if rA.get(g, {}).get("mastered")
                and sum(rK.get(g, {}).get("demos_per_round", []) or [0]) <= STARVED_DEMOS]
        res["arms"]["U"] = []
        if cand:
            g = max(cand, key=lambda x: pc[x])       # the deepest such goal
            print(f"\n  ARM U (fresh, {U_ROUNDS} rounds = 6x K, on goal {g} pc {pc[g]})", flush=True)
            r = run_goal_commanded(spec, skill, Composer("memory"), Buffer(), cfg, gseed(g), g,
                                   r_max=U_ROUNDS)
            r["r_max"] = U_ROUNDS
            res["arms"]["U"] = [r]
            print(f"    [U] goal {g}: master {r['master']:.2f} in {r['rounds']}r "
                  f"({'M' if r['mastered'] else 'x'}) | demos {sum(r['demos_per_round'])}", flush=True)
        else:
            print("\n  ARM U skipped: no deep goal where A mastered AND K was starved", flush=True)
        save()

    res["elapsed_s"] = round(time.perf_counter() - t0)
    save()
    print(f"\n  done in {res['elapsed_s']}s -> {jp}  (score with: python -m scripts.score_v57)",
          flush=True)


if __name__ == "__main__":
    main()
