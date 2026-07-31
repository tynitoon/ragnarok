"""ARC 2 — the linchpin probe for design v3. Does the STORE do everything, or do the WEIGHTS matter?

Two measurement designs were voided because the primary was a cost ratio against a fresh agent that was
trained IN-WORLD and already mastered everything — leaving ~0.68 rounds/goal of headroom and producing a
boundary artifact every time it was patched (ARC2_PLAN sections 8 and 11).

DESIGN v3 changes the question rather than the difficulty knob alone. The comparison becomes:
    M  meta-trained weights, FROZEN, with its store      (the treatment)
    R  RANDOM weights,       FROZEN, with the same store (the control that has real dynamic range)
    G  the hand-coded evidence policy, no weights at all (what the store supports with zero learning)
    F  fresh weights trained in-world                    (a CEILING reference, no longer the primary)
and the primary statistic becomes MASTERY COUNT, not a cost ratio — monotone in goodness, so failing can
never score better than succeeding, which is precisely how the last design broke.

THIS PROBE TESTS THE ONE ASSUMPTION THE WHOLE DESIGN RESTS ON: that R is genuinely bad. If R matches G
and F, then the portable knowledge lives in the hand-designed store rather than in any learned weights —
and that is the honest finding, to be published as-is instead of designed around.

Worlds are HARDER along the axis the envelope survey showed is safe: n_items 20 with p_resource 0.15
deepens chains (median pc 7.7 -> 9.8) while keeping the max cell ID at 9, inside the frozen nav skill's
measured range of 1..9. Raising n_items instead pushes cell IDs to 14-16 and collapses navigation for
every arm — the defect that killed v54.

Usage: python -m scripts.probe_v59 [--worlds 6100 6101] [--n-goals 6]
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
from scripts.evidence_store_v58 import evidence_policy, StoreEnv
from scripts.evidence_net_v58 import (ComposerV58, BufferV58, StoreEnvV58, permute_spec_v58,
                                      cfg_v58, make_world_env, run_goal_v58)

HARD = dict(n_items=20, p_resource=0.15)


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


@torch.no_grad()
def run_hand_coded(spec, skill, cfg, goals, seed):
    """Arm G: the store with no learned policy at all — the representation's own ceiling."""
    out = []
    for g in goals:
        env = StoreEnvV58(cfg["num_envs"], spec, skill, cfg, seed=seed + 9, goal=g, hidden=True)
        got = torch.zeros(cfg["num_envs"], dtype=torch.bool, device=DEVICE)
        for _ in range(cfg["r_max"]):
            env.reset(); env.set_goal(g)
            for _ in range(cfg["macro_budget"]):
                env.step(evidence_policy(env, g))
                got |= env.post_unlocked[:, g]
        out.append(float(got.float().mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, nargs="+", default=[6100, 6101])
    p.add_argument("--n-goals", type=int, default=6)
    p.add_argument("--r-max", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=64, r_max=a.r_max)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()
    skill = load_skill(cfg, a.seed, a.out_dir)
    res = dict(hard=HARD, r_max=a.r_max, worlds={})

    print("=" * 96)
    print("ARC 2 PROBE v3 — do the WEIGHTS matter, or does the STORE do everything?")
    print(f"  harder worlds: n_items {HARD['n_items']}, p_resource {HARD['p_resource']} "
          f"(deeper chains, cell IDs stay <= 9)")
    print("=" * 96, flush=True)

    for w in a.worlds:
        spec = permute_spec_v58(gen_tree(w, **HARD), w)
        nav = nav_gate(skill, spec, cfg, a.seed)
        adm = admitted_goals(spec)
        goals = [g for g, _, _ in adm][:a.n_goals]
        pcs = {g: pc for g, pc, _ in adm}
        cells = sorted({spec["cell"][i] for i in range(spec["n_items"])
                        if spec["kind"][i] == "R"})
        print(f"\nWORLD {w} | nav min {min(nav.values()):.3f} | cells {cells} | "
              f"{len(adm)} goals, probing {goals} (pc {[pcs[g] for g in goals]})", flush=True)
        if min(nav.values()) < 0.85:
            print("  NAV GATE FAILS — this difficulty setting is not usable"); continue

        # ---- R: random frozen weights + store. No training whatsoever. -------------------------
        envR = make_world_env(spec, skill, cfg, seed=w + 7, goal=goals[0])
        cR, bR = ComposerV58(), BufferV58(cap=200_000)
        rowsR = [run_goal_v58(envR, spec, skill, cR, bR, cfg, w + 11 * i, g, train=False)
                 for i, g in enumerate(goals)]
        print(f"  R (random frozen + store): master {[r['master'] for r in rowsR]} | "
              f"mastered {sum(r['mastered'] for r in rowsR)}/{len(goals)} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

        # ---- G: the hand-coded policy over the same store ---------------------------------------
        mG = run_hand_coded(spec, skill, cfg, goals, w)
        print(f"  G (hand-coded over store): master {[round(x,3) for x in mG]} | "
              f"mastered {sum(x >= cfg['thresh'] for x in mG)}/{len(goals)} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

        # ---- F: fresh weights TRAINED in-world (the ceiling reference) --------------------------
        envF = make_world_env(spec, skill, cfg, seed=w + 7, goal=goals[0])
        cF, bF = ComposerV58(), BufferV58(cap=600_000)
        rowsF = [run_goal_v58(envF, spec, skill, cF, bF, cfg, w + 11 * i, g, train=True)
                 for i, g in enumerate(goals)]
        print(f"  F (fresh, trained here)  : master {[r['master'] for r in rowsF]} | "
              f"mastered {sum(r['mastered'] for r in rowsF)}/{len(goals)} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

        res["worlds"][w] = dict(nav=nav, goals=goals, pcs={int(k): v for k, v in pcs.items()},
                                R=[r["master"] for r in rowsR], G=[round(x, 3) for x in mG],
                                F=[r["master"] for r in rowsF],
                                R_mastered=sum(r["mastered"] for r in rowsR),
                                G_mastered=sum(x >= cfg["thresh"] for x in mG),
                                F_mastered=sum(r["mastered"] for r in rowsF), n=len(goals))
        json.dump(res, open(os.path.join(a.out_dir, "v59_probe.json"), "w"), indent=2)

    tot = {k: sum(v[f"{k}_mastered"] for v in res["worlds"].values()) for k in ("R", "G", "F")}
    n = sum(v["n"] for v in res["worlds"].values())
    print(f"\n{'='*96}")
    print(f"  R (random frozen + store) {tot['R']}/{n} | G (hand-coded) {tot['G']}/{n} | "
          f"F (trained in-world) {tot['F']}/{n}")
    if n and tot["R"] / n < 0.35 and tot["F"] / n > 0.6:
        print("  => R is genuinely weak while the world is solvable: M-vs-R has real dynamic range.")
        print("     Design v3 (mastery count, frozen M vs frozen R) is measurable. Proceed to prereg.")
    elif n and tot["R"] / n >= 0.6:
        print("  => R already masters most goals: the portable knowledge lives in the hand-designed")
        print("     STORE, not in learned weights. That is the finding — publish it, do not design")
        print("     around it. No confirmatory run is warranted.")
    else:
        print("  => Ambiguous or the world is too hard for every arm (the v57 wall). Report and stop.")
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
