"""ARC 2, Step 4 — CALIBRATION. Measure the noise floor, then let it set the thresholds.

THE DEFECT THIS REPAIRS. Three consecutive ARC-1 verdicts (v54, v55, v57) were decided by thresholds and
strata chosen from a pencil rather than from data, and all three mis-calibrated: v57's DEEP=pc>=10 stratum
put starvation at pc 7 in one world and missed it at pc 10/14/16 in others. ARC2_PLAN section 0 makes the
repair non-negotiable: thresholds are FIT ONCE here, from measured variability, then hard-frozen into a
committed scorer BEFORE any confirmatory arm.

HOW. Run ONLY arm F (fresh, in-world) on 2 calibration worlds x several independent init seeds. Two
independent F runs differ by nothing but initialisation and sampling noise, so the distribution of their
pairwise ratios IS the null distribution of the primary statistic — what "no transfer whatsoever" looks
like on this metric. A threshold placed at the tail of that null is a threshold the null cannot reach by
chance, which is exactly what the previous three failed to be.

  P1 (stream cost ratio M/F) threshold  <- lower tail of the null F/F stream-cost ratio distribution
  P2 (paired first-demo win rate)       <- upper tail of the null F-vs-F win-rate distribution
  K2 KILL: if F's stream cost varies more than 2x BETWEEN the two calibration worlds, the metric is too
     unstable to carry a pooled verdict -> stop and redesign the metric; do not proceed on pencil numbers.

Nothing here touches held-out test worlds (6000-6002, 7000-7001) or the pretrain pool used by M.

Usage: python -m scripts.calibrate_v58 [--init-seeds 0 1 2] [--resume]
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

CAL_WORLDS = [5000, 5001]


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def run_F(spec, skill, cfg, goals, init_seed):
    """One independent arm-F run over a world's goal stream. Fresh net, fresh buffer, fresh store."""
    torch.manual_seed(init_seed * 7919 + 13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed * 7919 + 13)
    comp, buf = ComposerV58(), BufferV58(cap=1_200_000)
    env = make_world_env(spec, skill, cfg, seed=init_seed * 101 + 7, goal=goals[0])
    rows = []
    for g in goals:
        rows.append(run_goal_v58(env, spec, skill, comp, buf, cfg,
                                 1000 + 11 * goals.index(g), g))
    return rows


def stream_cost(rows):
    """Frozen primary statistic: total macro-attempts per env over the goal stream. Unmastered goals are
    naturally censored at the full per-goal budget (run_goal spends it all before returning)."""
    return sum(r["attempts"] for r in rows)


def win_rate(rows_a, rows_b):
    """Paired attempts-to-first-demo: fraction of NON-TIED goals where a beat b, and the win/loss ratio.
    A goal where neither side ever reached the goal is a tie and is excluded."""
    wins = losses = 0
    for ra, rb in zip(rows_a, rows_b):
        fa, fb = ra["first_demo_attempt"], rb["first_demo_attempt"]
        if fa is None and fb is None:
            continue
        if fb is None:
            wins += 1
        elif fa is None:
            losses += 1
        elif fa < fb:
            wins += 1
        elif fb < fa:
            losses += 1
    n = wins + losses
    return (wins / n if n else 0.5), wins, losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--r-max", type=int, default=8)
    p.add_argument("--skill-seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=256, r_max=a.r_max)
    t0 = time.perf_counter()
    jp = os.path.join(a.out_dir, "v58_calibration.json")
    res = json.load(open(jp)) if (a.resume and os.path.exists(jp)) else dict(runs={}, cfg_r_max=a.r_max)
    skill = load_skill(cfg, a.skill_seed, a.out_dir)

    print("=" * 100)
    print("ARC 2 CALIBRATION — arm F only, measuring the noise floor that will set the thresholds")
    print(f"  worlds {CAL_WORLDS} x init seeds {a.init_seeds} | r_max {a.r_max}")
    print("=" * 100, flush=True)

    for w in CAL_WORLDS:
        spec = permute_spec_v58(gen_tree(w, n_items=14), w)
        nav = nav_gate(skill, spec, cfg, a.skill_seed)
        adm = admitted_goals(spec)
        goals = [g for g, _, _ in adm]
        print(f"\nWORLD {w} | nav min {min(nav.values()):.3f} | {len(goals)} goals "
              f"(pc {[c for _, c, _ in adm]})", flush=True)
        if min(nav.values()) < 0.85:
            print("  NAV GATE FAILS — abort calibration (do not tune the skill)."); return
        for s in a.init_seeds:
            key = f"{w}_{s}"
            if key in res["runs"]:
                print(f"  init {s}: cached ({stream_cost(res['runs'][key]['rows'])} attempts)", flush=True)
                continue
            rows = run_F(spec, skill, cfg, goals, s)
            res["runs"][key] = dict(world=w, init=s, goals=goals, rows=rows)
            json.dump(res, open(jp, "w"), indent=2)
            m = sum(r["mastered"] for r in rows)
            print(f"  init {s}: stream cost {stream_cost(rows):>5} attempts | mastered {m}/{len(rows)}"
                  f" | {time.perf_counter()-t0:.0f}s", flush=True)
            if time.perf_counter() - t0 > a.max_hours * 3600:
                print(f"  !! {a.max_hours}h cap reached — stopping with state saved"); json.dump(
                    res, open(jp, "w"), indent=2); return

    # ---- fit the thresholds from the measured null ------------------------------------------------
    per_world = {w: [res["runs"][f"{w}_{s}"] for s in a.init_seeds if f"{w}_{s}" in res["runs"]]
                 for w in CAL_WORLDS}
    costs = {w: [stream_cost(r["rows"]) for r in per_world[w]] for w in CAL_WORLDS}
    null_ratios, null_wins = [], []
    for w in CAL_WORLDS:
        for x, y in itertools.permutations(range(len(per_world[w])), 2):
            null_ratios.append(stream_cost(per_world[w][x]["rows"]) /
                               max(1, stream_cost(per_world[w][y]["rows"])))
            null_wins.append(win_rate(per_world[w][x]["rows"], per_world[w][y]["rows"])[0])

    med = {w: statistics.median(costs[w]) for w in CAL_WORLDS}
    between = max(med.values()) / max(1e-9, min(med.values()))
    k2 = between > 2.0

    # HEADROOM (D2 guard). One round is the smallest cost a goal can take, so n_goals * attempts_per_
    # round is the metric's hard FLOOR. If F already sits near it, no treatment can beat the P1
    # threshold and the statistic is saturated — the same defect class that made v53's cost-to-master
    # unmeasurable. Measured here, before anything is frozen.
    per_round = cfg["episodes_per_round"] * cfg["macro_budget"]
    head = {}
    for w in CAL_WORLDS:
        floor = len(per_world[w][0]["goals"]) * per_round
        head[w] = dict(floor=floor, median_cost=med[w], ratio=round(med[w] / floor, 3),
                       best_possible_M_over_F=round(floor / med[w], 3))

    print(f"\n{'='*100}\nNOISE FLOOR (independent F runs — this is what NO transfer looks like)")
    for w in CAL_WORLDS:
        print(f"  world {w}: stream costs {costs[w]} | median {med[w]:.0f}")
    print(f"  between-world median ratio {between:.2f} (K2 kills at > 2.00)")
    print(f"  null F/F stream-cost ratios : min {min(null_ratios):.3f} | "
          f"median {statistics.median(null_ratios):.3f} | max {max(null_ratios):.3f}")
    print(f"  null F/F first-demo win rate: min {min(null_wins):.3f} | "
          f"median {statistics.median(null_wins):.3f} | max {max(null_wins):.3f}")

    print(f"\nHEADROOM (D2 saturating-metric guard):")
    for w in CAL_WORLDS:
        h = head[w]
        print(f"  world {w}: floor {h['floor']} | F median {h['median_cost']:.0f} "
              f"({h['ratio']:.2f}x floor) | best M/F physically reachable {h['best_possible_M_over_F']:.3f}")
    best_reachable = max(h["best_possible_M_over_F"] for h in head.values())

    p1_thr = round(min(null_ratios) * 0.95, 3)      # strictly below anything the null produced
    p2_thr = round(min(0.90, max(0.60, max(null_wins) + 0.05)), 3)
    saturated = p1_thr < best_reachable
    res.update(headroom=head, best_reachable_ratio=best_reachable, p1_saturated=bool(saturated),
               costs=costs, between_world_ratio=round(between, 3), k2_fires=bool(k2),
               null_ratio=dict(min=round(min(null_ratios), 3),
                               median=round(statistics.median(null_ratios), 3),
                               max=round(max(null_ratios), 3)),
               null_winrate=dict(min=round(min(null_wins), 3),
                                 median=round(statistics.median(null_wins), 3),
                                 max=round(max(null_wins), 3)),
               fitted=dict(P1_MAX_RATIO=p1_thr, P2_MIN_WINRATE=p2_thr),
               elapsed_s=round(time.perf_counter() - t0))
    json.dump(res, open(jp, "w"), indent=2)

    print(f"\n{'='*100}")
    if k2:
        print("!! KILL K2 FIRES: F's stream cost varies more than 2x between calibration worlds.")
        print("   The pooled metric cannot carry a verdict. STOP and redesign the metric — do NOT")
        print("   proceed on pencil numbers. (ARC2_PLAN section 5.)")
    else:
        print("K2 does not fire. FITTED THRESHOLDS, to be hard-coded verbatim into scripts/score_v58.py")
        print("and committed BEFORE any confirmatory arm runs:")
        print(f"   P1_MAX_RATIO  = {p1_thr}   (pooled M/F stream-cost ratio must be <= this;")
        print(f"                              the null F/F distribution never went below "
              f"{min(null_ratios):.3f})")
        print(f"   P2_MIN_WINRATE = {p2_thr}  (paired first-demo win rate must be >= this;")
        print(f"                              the null reached at most {max(null_wins):.3f})")
    if saturated:
        print(f"\n!! P1 IS SATURATED: the fitted threshold {p1_thr} is BELOW the best ratio physically")
        print(f"   reachable ({best_reachable:.3f}) — arm F already sits near the metric's floor, so no")
        print(f"   treatment could pass P1 however good it was. This is the D2 defect class. P1 must be")
        print(f"   repaired (finer cost resolution, or harder worlds) BEFORE the prereg is frozen; it")
        print(f"   may NOT be carried into a confirmatory run as-is. P2 (first-demo, macro-step")
        print(f"   resolution) is unaffected and remains usable.")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
