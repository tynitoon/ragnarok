"""ARC 2, Step 3 — GATE K1 (<=1 GPU-hour). Does the architecture work at all?

Two questions, both pre-committed in ARC2_PLAN.md section 5, answered before ANY confirmatory spend:
  Q1 CAN IT LEARN?   a fresh EvidenceNet (arm F) trained in-world under commanded-only credit must
                     reach MemoryNet-level mastery on the goal stream. Operationalised as >= 0.70 of
                     the attempted goals mastered (v57's arm A mastered 10/11, 8/9, 10/11 on its
                     worlds, and 100% of shallow goals under the honest credit rule).
  Q2 DOES IT READ THE STORE?  re-evaluating the SAME trained weights with the evidence half zeroed
                     (arm Z's mechanism) must cost >= 30% of the mastery. If the net solves the world
                     while ignoring its own world knowledge, the portability claim is empty before it
                     starts — there would be nothing for the store to carry between worlds.

FAIL on either -> one shot for the pre-registered fallback (single attention round over slots), one more
1 GPU-h gate; if that also fails, ARC 2 ends here, before any confirmatory GPU is spent.

Runs on world 4000 (pretrain pool — never a held-out test world, so nothing is contaminated).

Usage: python -m scripts.gate_k1_v58 [--n-goals 6] [--r-max 5]
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
from scripts.hidden_recipe_v55 import admitted_goals, nav_gate
from scripts.evidence_net_v58 import (ComposerV58, BufferV58, permute_spec_v58, cfg_v58,
                                      make_world_env, run_goal_v58, eval_goal_v58)

GATE_WORLD = 4000
PASS_MASTERY = 0.70
PASS_STORE_DROP = 0.30


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
    p.add_argument("--n-goals", type=int, default=6)
    p.add_argument("--r-max", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=1.4)
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=256, r_max=a.r_max)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()

    spec = permute_spec_v58(gen_tree(GATE_WORLD, n_items=14), GATE_WORLD)
    adm = admitted_goals(spec)
    pc = {g: c for g, c, _ in adm}
    goals = [g for g, _, _ in adm][:a.n_goals]
    skill = load_skill(cfg, a.seed, a.out_dir)
    nav = nav_gate(skill, spec, cfg, a.seed)

    print("=" * 96)
    print(f"ARC 2 GATE K1 | world {GATE_WORLD} | EvidenceNet, commanded-only credit, macro_budget "
          f"{cfg['macro_budget']}")
    print(f"  nav gate min {min(nav.values()):.3f} {nav}")
    print(f"  goals {goals} (pc {[pc[g] for g in goals]}) | r_max {a.r_max}")
    print("=" * 96, flush=True)
    if min(nav.values()) < 0.85:
        print("  NAV GATE FAILS on the gate world — abort (do not tune the skill)."); return

    comp, buf = ComposerV58(), BufferV58(cap=800_000)
    env = make_world_env(spec, skill, cfg, seed=a.seed + 101, goal=goals[0])
    rows = []
    for g in goals:
        r = run_goal_v58(env, spec, skill, comp, buf, cfg, a.seed + 11 * goals.index(g) + 1, g)
        rows.append(r)
        print(f"    goal {g:>2} (pc {pc[g]:>2}): master {r['master']:.2f} in {r['rounds']}r "
              f"({'M' if r['mastered'] else 'x'}) | demos {sum(r['demos_per_round'])} | "
              f"attempts {r['attempts']} | buf {buf.n} | {time.perf_counter()-t0:.0f}s", flush=True)
        if time.perf_counter() - t0 > a.max_hours * 3600:
            print(f"    !! {a.max_hours}h cap reached — stopping the gate here", flush=True)
            break

    # ---- Q2: does the trained policy actually READ the store? --------------------------------------
    print("\n  store-read probe (same weights, evidence half zeroed = arm Z's mechanism):", flush=True)
    probe = []
    for r in rows:
        g = r["goal"]
        with_s, _ = eval_goal_v58(spec, skill, comp, cfg, a.seed + 11 * goals.index(g) + 1, g,
                                  env.store.state_dict())
        zero_s, _ = eval_goal_v58(spec, skill, comp, cfg, a.seed + 11 * goals.index(g) + 1, g,
                                  env.store.state_dict(), zero_store=True)
        probe.append(dict(goal=g, with_store=round(with_s, 3), zero_store=round(zero_s, 3)))
        print(f"    goal {g:>2}: with store {with_s:.3f} | zeroed {zero_s:.3f} | "
              f"drop {with_s - zero_s:+.3f}", flush=True)

    mastery = sum(r["mastered"] for r in rows) / max(1, len(rows))
    live = [x for x in probe if x["with_store"] >= 0.10]
    rel = (sum((x["with_store"] - x["zero_store"]) / x["with_store"] for x in live) / len(live)
           if live else 0.0)
    q1, q2 = mastery >= PASS_MASTERY, rel >= PASS_STORE_DROP
    res = dict(world=GATE_WORLD, nav=nav, goals=goals, rows=rows, probe=probe,
               mastery=round(mastery, 3), store_drop_rel=round(rel, 3),
               q1_can_learn=q1, q2_reads_store=q2, passed=bool(q1 and q2),
               elapsed_s=round(time.perf_counter() - t0))
    json.dump(res, open(os.path.join(a.out_dir, "v58_gate_k1.json"), "w"), indent=2)

    print(f"\n{'='*96}")
    print(f"  Q1 can it learn      : mastery {mastery:.2f} (need >= {PASS_MASTERY}) -> {q1}")
    print(f"  Q2 does it read store: relative drop when zeroed {rel:.2f} "
          f"(need >= {PASS_STORE_DROP}) -> {q2}")
    if q1 and q2:
        print("\n  GATE K1 PASSED — proceed to Step 4 (calibration), then freeze the prereg + scorer.")
    else:
        print("\n  GATE K1 FAILED — one shot for the pre-registered attention fallback, then ARC 2 ends.")
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
