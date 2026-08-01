"""ARC 2 — GATE KA, the pilot that decides whether the confirmatory happens at all.

THE NORTH STAR: show the agent learns and uses what it learned to LEARN FASTER. So both arms LEARN in a
world neither has seen, and we compare their learning curves.

    M    weights carried from a previous world, now learning here
    Fa   fresh weights, learning here
    Fb   a SECOND independent fresh run — Fa vs Fb IS the null, measured in the same rows

PRIMARY (design v3, monotone by construction):
    Ā_a(g) = mean over b of m_a(g,b), the area under goal g's learning curve, b = 0..B_max
    Δ      = mean over paired goals of [ Ā_M(g) − ½(Ā_Fa(g) + Ā_Fb(g)) ]
    se_null = sd_g[ Ā_Fa(g) − Ā_Fb(g) ] / sqrt(2N)      <- a FORMULA over contemporaneous rows, never a
                                                            number transplanted from another calibration
Fixed budget, no early break: every arm runs exactly B_max rounds on every goal, so store SIZE is
bit-matched at each goal's entry and only its content differs.

GATE KA, committed before this runs:
    Δ_pilot <= 0     the confirmatory does NOT run; ARC 2 closes on what is already established
    0 < Δ < 0.10     only the minimal one-test-world confirmatory
    Δ >= 0.10        the full design
THE PILOT IS A POWER MEASUREMENT, NEVER EVIDENCE. It runs on world 6100, which is already burned by
publication and may never be a confirmatory test world.

Usage: python -m scripts.pilot_v60
"""

import argparse
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

HARD = dict(n_items=20, p_resource=0.15)
PRETRAIN_WORLD = 6101      # where M gains its prior experience
PILOT_WORLD = 6100         # burned by publication — pilot only, never a confirmatory test world
B_MAX = 3


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def area(row):
    m = row["master_per_round"]
    return sum(m) / len(m)


def run_arm(tag, spec, skill, cfg, goals, composer, seed_base, t0):
    """One arm over the goal stream at a FIXED budget. Fresh store and buffer; weights are the treatment."""
    env = make_world_env(spec, skill, cfg, seed=seed_base, goal=goals[0])
    buf = BufferV58(cap=600_000)
    rows = []
    for i, g in enumerate(goals):
        r = run_goal_v58(env, spec, skill, composer, buf, cfg, PILOT_WORLD + 11 * i, g,
                         r_max=B_MAX, train=True, fixed_budget=True)
        rows.append(r)
        print(f"    [{tag}] goal {g:>2}: curve {r['master_per_round']} | area {area(r):.3f} | "
              f"first mastered at {r['first_mastered_at']} | {time.perf_counter()-t0:.0f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-goals", type=int, default=6)
    p.add_argument("--n-pretrain-goals", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=64, r_max=B_MAX)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()
    skill = load_skill(cfg, a.seed, a.out_dir)

    print("=" * 100)
    print(f"ARC 2 GATE KA — pilot: does prior experience make LEARNING FASTER in an unseen world?")
    print(f"  M gains experience on {PRETRAIN_WORLD}, then all arms LEARN on {PILOT_WORLD} "
          f"| fixed budget {B_MAX} rounds/goal, no early break")
    print("=" * 100, flush=True)

    # ---- M gains prior experience -------------------------------------------------------------
    sP = permute_spec_v58(gen_tree(PRETRAIN_WORLD, **HARD), PRETRAIN_WORLD)
    gP = [g for g, _, _ in admitted_goals(sP)][:a.n_pretrain_goals]
    envP = make_world_env(sP, skill, cfg, seed=PRETRAIN_WORLD + 7, goal=gP[0])
    cM, bufP = ComposerV58(), BufferV58(cap=800_000)
    print(f"\n  PRETRAIN M on world {PRETRAIN_WORLD} ({len(gP)} goals)", flush=True)
    for i, g in enumerate(gP):
        r = run_goal_v58(envP, sP, skill, cM, bufP, cfg, PRETRAIN_WORLD + 11 * i, g,
                         r_max=B_MAX, train=True, fixed_budget=True)
        print(f"    [pretrain] goal {g:>2}: curve {r['master_per_round']} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    torch.save(cM.net.state_dict(), os.path.join(a.out_dir, "v60_M_pilot.pt"))

    # ---- the pilot world: all three arms LEARN ------------------------------------------------
    sT = permute_spec_v58(gen_tree(PILOT_WORLD, **HARD), PILOT_WORLD)
    nav = nav_gate(skill, sT, cfg, a.seed)
    adm = admitted_goals(sT)
    goals = [g for g, _, _ in adm][:a.n_goals]
    print(f"\n  PILOT WORLD {PILOT_WORLD} | nav min {min(nav.values()):.3f} | goals {goals} "
          f"(pc {[pc for g, pc, _ in adm if g in goals]})", flush=True)
    if min(nav.values()) < 0.85:
        print("  NAV GATE FAILS — abort"); return

    torch.manual_seed(a.seed + 1)
    rowsFa = run_arm("Fa", sT, skill, cfg, goals, ComposerV58(), PILOT_WORLD + 7, t0)
    torch.manual_seed(a.seed + 2)
    rowsFb = run_arm("Fb", sT, skill, cfg, goals, ComposerV58(), PILOT_WORLD + 7, t0)
    rowsM = run_arm("M ", sT, skill, cfg, goals, cM, PILOT_WORLD + 7, t0)

    # ---- the primary --------------------------------------------------------------------------
    dM = [area(m) - 0.5 * (area(fa) + area(fb)) for m, fa, fb in zip(rowsM, rowsFa, rowsFb)]
    dN = [area(fa) - area(fb) for fa, fb in zip(rowsFa, rowsFb)]
    N = len(dM)
    delta = sum(dM) / N
    se = (statistics.stdev(dN) / (2 * N) ** 0.5) if N > 1 else float("nan")

    def count(rows, B):
        return sum(1 for r in rows if max(r["master_per_round"][:B + 1]) >= cfg["thresh"])

    print(f"\n{'='*100}")
    print(f"  learning curve (goals mastered within B rounds of practice, cumulative-OR):")
    for tag, rows in (("M ", rowsM), ("Fa", rowsFa), ("Fb", rowsFb)):
        print(f"    {tag}: " + "  ".join(f"B={B}:{count(rows, B)}/{N}" for B in range(B_MAX + 1)))
    print(f"\n  Delta (paired curve-area advantage of M over fresh) = {delta:+.4f}")
    print(f"  se_null (from Fa vs Fb, the contemporaneous null)     = {se:.4f}")
    print(f"  per-goal deltas: {[round(x, 3) for x in dM]}")

    gate = ("FULL" if delta >= 0.10 else "MINIMAL" if delta > 0 else "STOP")
    json.dump(dict(world=PILOT_WORLD, pretrain=PRETRAIN_WORLD, goals=goals, B_max=B_MAX,
                   M=[r["master_per_round"] for r in rowsM],
                   Fa=[r["master_per_round"] for r in rowsFa],
                   Fb=[r["master_per_round"] for r in rowsFb],
                   delta=round(delta, 4), se_null=round(se, 4), per_goal=dM, gate=gate,
                   counts={t: [count(r, B) for B in range(B_MAX + 1)]
                           for t, r in (("M", rowsM), ("Fa", rowsFa), ("Fb", rowsFb))}),
              open(os.path.join(a.out_dir, "v60_pilot.json"), "w"), indent=2)

    print(f"\n  GATE KA -> {gate}")
    if gate == "STOP":
        print("     Delta <= 0: prior experience does NOT speed up learning here. The confirmatory does")
        print("     NOT run. ARC 2 closes on what is already honestly established.")
    elif gate == "MINIMAL":
        print("     0 < Delta < 0.10: run only the minimal one-test-world confirmatory, and pre-register")
        print("     INCONCLUSIVE as the most likely outcome.")
    else:
        print("     Delta >= 0.10: the full design is worth its compute.")
    print("  (A pilot is a POWER MEASUREMENT, never evidence. World 6100 is burned and can never be a")
    print("   confirmatory test world.)")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
