"""ARC 3 — pretraining a LINEAGE (ARC3_PLAN.md 4.2): four admitted worlds in order, a 4-goal chain-blind
stream per world (tier 2, 3, and the two lowest-index admitted tier-4 goals), B rounds per goal, per-goal
store, buffers and optimizer rebuilt at world entry, weights carried. Checkpoints M1..M4.

Recorded for free: per-world learning-curve areas (world k given k-1 prior worlds) with L beside them as
difficulty context, and the lineage's EXERCISED bond cells (element pairs it actually proposed) after each
world — the coverage of 3.2 measured, not assumed.

Usage: python -m scripts.pretrain_v61 --lineage 0 [--b-max 3] [--resume] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import sample_laws, make_world, admitted_worlds, T_MAX
from scripts.hidden_recipe_v55 import nav_gate
from scripts.pair_store_v61 import PAIR_I, PAIR_J
from scripts.pair_net_v61 import (ComposerV61, cfg_v61, load_skill, eval_goal_v61, run_unit_v61, area,
                                  init_seed)
from scripts.hand_v61 import LawKnower

LAWS_SEED = 31337


def stream4(spec):
    return [spec["admitted"][2][0], spec["admitted"][3][0]] + spec["admitted"][4][:2]


def exercised_cells(store_state, spec):
    """Element pairs (a<b) this store has an outcome for, in ANY env: the cells the weights were trained on."""
    tried = (store_state["pair_succ"] + store_state["pair_inert"] + store_state["pair_fail"]) > 0     # (N,378)
    anyp = tried.any(0).nonzero(as_tuple=True)[0].cpu().tolist()
    el, n, tier, decoy = spec["el"], spec["n_items"], spec["tier"], spec["decoy"]
    cells = set()
    for p in anyp:
        i, j = int(PAIR_I[p]), int(PAIR_J[p])
        # only equal-tier real pairs below T_MAX are Bond cells; an unequal-tier botch teaches nothing about Bond
        if i < n and j < n and tier[i] == tier[j] < T_MAX and not decoy[i] and not decoy[j]:
            cells.add((min(el[i], el[j]), max(el[i], el[j])))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", type=int, required=True)
    ap.add_argument("--b-max", type=int, default=3)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--n-worlds", type=int, default=4)
    ap.add_argument("--out-dir", default="craft_v6_out")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    s = a.lineage
    cfg = cfg_v61(num_envs=a.num_envs, r_max=a.b_max)
    laws = sample_laws(LAWS_SEED)
    skill = load_skill(cfg, a.seed, a.out_dir)
    tag = f"v61_pre_s{s}{sfx}"
    logf = open(os.path.join(a.out_dir, tag + ".log"), "a")

    def L_(x):
        print(x, flush=True); logf.write(x + "\n"); logf.flush()

    t0 = time.perf_counter()
    window = range(8200 + 10 * s, 8200 + 10 * s + 10)
    cands = admitted_worlds(laws, window)
    L_(f"LINEAGE {s} | candidates {cands} | B {a.b_max} | {a.num_envs} envs | {time.strftime('%Y-%m-%d %H:%M')}")
    jpath = os.path.join(a.out_dir, tag + ".json")
    res = json.load(open(jpath)) if (a.resume and os.path.exists(jpath)) else dict(lineage=s, b_max=a.b_max, worlds=[])
    done = {w["world"] for w in res["worlds"]}
    comp = ComposerV61(init_seed=init_seed(s, cands[0], 4))
    if done:
        last = res["worlds"][-1]["k"]
        comp.net.load_state_dict(torch.load(os.path.join(a.out_dir, f"{tag}_M{last}.pt"), map_location=DEVICE))
        L_(f"  [resume] loaded M{last}")
    exercised = set(tuple(c) for c in res.get("exercised", []))
    k = len(res["worlds"])
    for w in cands:
        if k >= a.n_worlds:
            break
        if w in done:
            continue
        spec = make_world(w, laws)
        nav = nav_gate(skill, spec, cfg, a.seed)
        if min(nav.values()) < 0.85:
            L_(f"  world {w}: nav gate {min(nav.values()):.3f} < 0.85 — skipped (logged)"); continue
        goals = stream4(spec)
        tiers = [spec["tier"][g] for g in goals]
        L_(f"\nWORLD {w} (k={k+1}) | stream {goals} tiers {tiers} | nav min {min(nav.values()):.3f}")
        Lk = LawKnower(spec)
        mL = [eval_goal_v61(spec, skill, Lk, cfg, w + 11 * p, g) for p, g in enumerate(goals)]
        L_(f"  L: {[round(x, 3) for x in mL]}")
        torch.manual_seed(init_seed(s, w, 4) + 500)
        cells_world = set()

        def on_goal(r):
            L_(f"  [M{k}->] goal {r['goal']} (tier {r['tier']}): curve {r['master_per_round']} rows {r['samples_per_round']} "
               f"demos {r['demos_per_round']} | {time.perf_counter()-t0:.0f}s")

        # run_unit resets the store per goal; to record exercised cells we wrap the log to read the store
        # at each goal's end is not possible from outside -> record via env hook: run goals manually
        from scripts.pair_store_v61 import PairEnv, PairStore
        from scripts.pair_net_v61 import BufferV61, run_goal_v61
        env = PairEnv(cfg["num_envs"], spec, skill, cfg, seed=w + 7, goal=goals[0])
        buf = BufferV61(cap=cfg["buf_cap"])
        comp.reset_optimizer()
        rows = []
        for p, g in enumerate(goals):
            env.store = PairStore(env.num_envs, env.n_items)
            r = run_goal_v61(env, spec, skill, comp, buf, cfg, w + 11 * p, g, a.b_max, train=True)
            r["tier"] = tiers[p]; r["pos"] = p
            cells_world |= exercised_cells(env.store.state_dict(), spec)
            on_goal(r)
            rows.append(r)
        k += 1
        exercised |= cells_world
        torch.save(comp.net.state_dict(), os.path.join(a.out_dir, f"{tag}_M{k}.pt"))
        n_true = int(laws["n_true"])
        cov = len({c for c in exercised if laws["Bond"][c[0], c[1]]}) / n_true
        res["worlds"].append(dict(world=w, k=k, goals=goals, tiers=tiers, L=mL,
                                  rows=[dict(goal=r["goal"], tier=r["tier"], curve=r["master_per_round"],
                                             rows=r["samples_per_round"], demos=r["demos_per_round"]) for r in rows],
                                  area=sum(area(r, 1, a.b_max) for r in rows) / len(rows),
                                  area_L=sum(mL) / len(mL), cells_world=len(cells_world),
                                  coverage_true=round(cov, 3), secs=round(time.perf_counter() - t0)))
        res["exercised"] = sorted(exercised)
        json.dump(res, open(jpath, "w"), indent=1)
        L_(f"  M{k} saved | world area {res['worlds'][-1]['area']:.3f} (L {res['worlds'][-1]['area_L']:.3f}) | "
           f"cells exercised here {len(cells_world)} | lineage coverage of true cells {cov:.3f} | {time.perf_counter()-t0:.0f}s")
    L_(f"lineage {s} done: areas {[round(x['area'],3) for x in res['worlds']]} coverage "
       f"{[x['coverage_true'] for x in res['worlds']]} | total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
