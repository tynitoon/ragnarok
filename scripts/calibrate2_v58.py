"""ARC 2 — CALIBRATION v2. Re-measures the null under the REPAIRED design (ARC2_PLAN sections 8-9).

WHY v1 WAS VOID (verification audit, all confirmed by re-derivation):
  - the P1 threshold was fitted on the SINGLE-world null (min 0.750) but applied to a 3-world POOLED
    ratio, whose null is materially tighter — so a genuine 20% saving would have published as REFUTED;
  - the raw-attempts metric carried a fatal CENSORING ASYMMETRY: one goal a frozen-weight arm missed
    charged r_max full rounds, exceeding that arm's entire above-floor allowance ("M must be flawless
    on 27/27");
  - 26/54 goal-runs were mastered ON ARRIVAL yet still charged a full round — dead weight;
  - P2's statistic had zero resolution (54/54 values = 48) because with 256 parallel envs some env
    always reaches the goal in the first episode.

THE REPAIRED DESIGN THIS SCRIPT MEASURES (frozen in ARC2_PLAN section 9 BEFORE it ran):
  - num_envs 256 -> 64. The root cause of both the power problem and the P2 degeneracy was that 256
    parallel trials make discovery nearly free; 64 quarters the data per round so a fresh agent must
    actually work, opening the headroom that transfer needs to be visible in.
  - SCORED cost (run_goal_v58 `cost`): 0 on arrival-mastery, rounds*192 on mastery, censor_cap(3)*192
    on censoring — for EVERY arm. Spending/collection unchanged (>= 1 round always).
  - POOLED-unit null: for each of the 3 calibration worlds pick an ordered pair of DISTINCT inits,
    pool numerators and denominators across worlds -> the distribution of pooled F/F ratios IS the
    null of the confirmatory primary, on the same unit the scorer uses.
  - P2 candidate, CONTINUOUS: mean paired difference of `discovery.frac` (fraction of envs reaching
    the goal at first exposure; resolution 1/64). Its null is measured the same pooled way.
    `min_step` is banned (saturates low); win rates are dead (26% ties at ceiling in v1).
  - MARGIN GUARD, explicit: the fitted P1 threshold must leave room for at least TWO censored goals
    on the treatment side (2 * censor_cap * 192 / pooled F cost), else the design is still too tight
    and the confirmatory run must not start.
  - K2 unchanged: between-world median SCORED cost spread > 2x kills.

Usage: python -m scripts.calibrate2_v58 [--resume]
"""

import argparse
import itertools
import json
import os
import statistics
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.hidden_recipe_v55 import admitted_goals, nav_gate
from scripts.evidence_net_v58 import (ComposerV58, BufferV58, permute_spec_v58, cfg_v58,
                                      make_world_env, run_goal_v58)

CAL_POOL = [5000, 5001, 5002, 5003, 5004, 5005]      # first 3 passing the nav gate are used
N_WORLDS = 3
INITS = [0, 1, 2]
NUM_ENVS = 64
CENSOR_CAP = 3


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def run_F(spec, skill, cfg, goals, init_seed):
    torch.manual_seed(init_seed * 7919 + 13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed * 7919 + 13)
    comp, buf = ComposerV58(), BufferV58(cap=1_200_000)
    env = make_world_env(spec, skill, cfg, seed=init_seed * 101 + 7, goal=goals[0])
    return [run_goal_v58(env, spec, skill, comp, buf, cfg, 1000 + 11 * goals.index(g), g)
            for g in goals]


def scored(rows):
    return sum(r["cost"] for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r-max", type=int, default=8)
    p.add_argument("--skill-seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=NUM_ENVS, r_max=a.r_max, censor_cap=CENSOR_CAP)
    t0 = time.perf_counter()
    jp = os.path.join(a.out_dir, "v58_calibration2.json")
    res = (json.load(open(jp)) if (a.resume and os.path.exists(jp))
           else dict(design=dict(num_envs=NUM_ENVS, censor_cap=CENSOR_CAP, r_max=a.r_max),
                     worlds=[], runs={}))
    skill = load_skill(cfg, a.skill_seed, a.out_dir)

    print("=" * 100)
    print(f"ARC 2 CALIBRATION v2 — repaired design: num_envs {NUM_ENVS}, SCORED cost "
          f"(0 on arrival / censor cap {CENSOR_CAP}), pooled-unit null")
    print("=" * 100, flush=True)

    # ---- pick the 3 calibration worlds mechanically (nav gate, ascending) --------------------------
    if not res["worlds"]:
        for w in CAL_POOL:
            spec = permute_spec_v58(gen_tree(w, n_items=14), w)
            nav = nav_gate(skill, spec, cfg, a.skill_seed)
            ok = min(nav.values()) >= 0.85
            print(f"  nav gate world {w}: min {min(nav.values()):.3f} "
                  f"{'PASS' if ok else 'FAIL -> next'}", flush=True)
            if ok:
                res["worlds"].append(w)
            if len(res["worlds"]) == N_WORLDS:
                break
        json.dump(res, open(jp, "w"), indent=2)
    worlds = res["worlds"]
    print(f"  calibration worlds: {worlds}", flush=True)

    # ---- runs -------------------------------------------------------------------------------------
    for w in worlds:
        spec = permute_spec_v58(gen_tree(w, n_items=14), w)
        goals = [g for g, _, _ in admitted_goals(spec)]
        for s in INITS:
            key = f"{w}_{s}"
            if key in res["runs"]:
                continue
            rows = run_F(spec, skill, cfg, goals, s)
            res["runs"][key] = dict(world=w, init=s, goals=goals, rows=rows)
            json.dump(res, open(jp, "w"), indent=2)
            print(f"  world {w} init {s}: scored cost {scored(rows):>5} | "
                  f"mastered {sum(r['mastered'] for r in rows)}/{len(rows)} "
                  f"(on-arrival {sum(r['mastered_on_arrival'] for r in rows)}) | censored "
                  f"{sum(1 for r in rows if not r['mastered'])} | {time.perf_counter()-t0:.0f}s",
                  flush=True)
            if time.perf_counter() - t0 > a.max_hours * 3600:
                print(f"  !! {a.max_hours}h cap — state saved, relaunch with --resume", flush=True)
                return

    # ---- pooled null ------------------------------------------------------------------------------
    per_round = cfg["episodes_per_round"] * cfg["macro_budget"]
    R = {w: [res["runs"][f"{w}_{s}"]["rows"] for s in INITS] for w in worlds}
    null_ratios, null_dfrac = [], []
    pairs_per_world = list(itertools.permutations(range(len(INITS)), 2))
    for combo in itertools.product(pairs_per_world, repeat=len(worlds)):
        num = sum(scored(R[w][combo[i][0]]) for i, w in enumerate(worlds))
        den = sum(scored(R[w][combo[i][1]]) for i, w in enumerate(worlds))
        null_ratios.append(num / max(1, den))
        df, n = 0.0, 0
        for i, w in enumerate(worlds):
            for ra, rb in zip(R[w][combo[i][0]], R[w][combo[i][1]]):
                fa = (ra.get("discovery") or {}).get("frac")
                fb = (rb.get("discovery") or {}).get("frac")
                if fa is not None and fb is not None:
                    df += fa - fb; n += 1
        if n:
            null_dfrac.append(df / n)

    meds = {w: statistics.median(scored(r) for r in R[w]) for w in worlds}
    between = max(meds.values()) / max(1e-9, min(meds.values()))
    F_pooled = statistics.median(
        sum(scored(R[w][s]) for w in worlds) for s in range(len(INITS)))
    margin_needed = 2 * CENSOR_CAP * per_round / max(1, F_pooled)
    p1_thr = round(min(null_ratios) * 0.95, 3)
    guard_ok = p1_thr >= margin_needed
    k2 = between > 2.0

    res.update(
        per_world_median=meds, between_world_ratio=round(between, 3), k2_fires=bool(k2),
        pooled_F_median=F_pooled,
        null_ratio=dict(n=len(null_ratios), min=round(min(null_ratios), 3),
                        p05=round(sorted(null_ratios)[max(0, len(null_ratios)//20)], 3),
                        median=round(statistics.median(null_ratios), 3),
                        max=round(max(null_ratios), 3)),
        null_dfrac=dict(n=len(null_dfrac), min=round(min(null_dfrac), 4),
                        median=round(statistics.median(null_dfrac), 4),
                        max=round(max(null_dfrac), 4)) if null_dfrac else None,
        margin=dict(needed=round(margin_needed, 3), fitted_p1=p1_thr, ok=bool(guard_ok)),
        fitted=dict(P1_MAX_RATIO=p1_thr,
                    REFUTE_RATIO=round(min(null_ratios), 3),
                    P2_MIN_DFRAC=(round(max(null_dfrac) + 0.02, 4) if null_dfrac else None)),
        elapsed_s=round(time.perf_counter() - t0))
    json.dump(res, open(jp, "w"), indent=2)

    print(f"\n{'='*100}\nPOOLED NULL ({len(null_ratios)} combos over {len(worlds)} worlds x "
          f"{len(INITS)} inits)")
    for w in worlds:
        print(f"  world {w}: scored costs {[scored(r) for r in R[w]]} | median {meds[w]:.0f}")
    print(f"  between-world spread {between:.2f} (K2 kills at > 2.00)")
    print(f"  null pooled F/F cost ratio : min {min(null_ratios):.3f} | "
          f"median {statistics.median(null_ratios):.3f} | max {max(null_ratios):.3f}")
    if null_dfrac:
        print(f"  null pooled dFRAC          : min {min(null_dfrac):+.4f} | "
              f"median {statistics.median(null_dfrac):+.4f} | max {max(null_dfrac):+.4f}")
    print(f"\n  MARGIN GUARD: fitted P1 {p1_thr} must be >= 2 censored goals' worth "
          f"{margin_needed:.3f} -> {'OK' if guard_ok else 'FAILS — design still too tight, DO NOT RUN'}")
    if k2:
        print("\n!! K2 FIRES — between-world spread > 2x; the pooled metric cannot carry a verdict.")
    elif guard_ok:
        print(f"\nFITTED (to be frozen verbatim in score_v58.py + the amended prereg, same commit):")
        print(f"   P1_MAX_RATIO = {p1_thr} | REFUTE_RATIO = {round(min(null_ratios), 3)}"
              + (f" | P2_MIN_DFRAC = {res['fitted']['P2_MIN_DFRAC']}" if null_dfrac else ""))
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
