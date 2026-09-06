"""ARC 3 — the CONFIRMATORY (ARC3_PLAN.md 4.4). Per lineage s: the first N/3 admitted seeds of the test
window 8300+10s..+9 (nav gate at entry, skipped seeds logged), arms M4 (lineage weights), Fa, Fb (fresh,
per-unit init seeds), L (one eval per goal, flat), G' (lineage-0 units only unless --g-all). Weights only at
entry; per-goal store; identical grid/eval seeds across arms. Per-(unit, arm) JSON checkpoints; --resume.
Writes craft_v6_out/v61_confirm.json in the scorer's format. Never inspects a test world before running it.

Usage: python -m scripts.confirm_v61 --lineage 0 --n-units 4 --b-max 3 [--resume] [--smoke]
       (run once per lineage; then python -m scripts.score_v61)
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import sample_laws, make_world, goal_stream, admitted_worlds
from scripts.hidden_recipe_v55 import nav_gate
from scripts.pair_net_v61 import ComposerV61, cfg_v61, load_skill, eval_goal_v61, run_unit_v61, init_seed
from scripts.hand_v61 import LawKnower, StoreSweeper

LAWS_SEED = 31337


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", type=int, required=True)
    ap.add_argument("--n-units", type=int, default=4)
    ap.add_argument("--b-max", type=int, default=3)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--g-all", action="store_true", help="run G' on every lineage's units (budget)")
    ap.add_argument("--out-dir", default="craft_v6_out")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-persists", action="store_true", help="the notch: store persists across the stream")
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    s, B = a.lineage, a.b_max
    cfg = cfg_v61(num_envs=a.num_envs, r_max=B)
    cfg["store_persists"] = bool(a.store_persists)
    laws = sample_laws(LAWS_SEED)
    skill = load_skill(cfg, a.seed, a.out_dir)
    ck = os.path.join(a.out_dir, f"v61_confirm_ckpt{sfx}"); os.makedirs(ck, exist_ok=True)
    logf = open(os.path.join(a.out_dir, f"v61_confirm_s{s}{sfx}.log"), "a")

    def L_(x):
        print(x, flush=True); logf.write(x + "\n"); logf.flush()

    def load_or(name, fn):
        p = os.path.join(ck, name + ".json")
        if a.resume and os.path.exists(p):
            L_(f"  [resume] {name}"); return json.load(open(p))
        r = fn(); json.dump(r, open(p, "w"), indent=1); return r

    t0 = time.perf_counter()
    pre = f"v61_pre_s{s}{sfx}"
    prej = json.load(open(os.path.join(a.out_dir, pre + ".json")))
    kM = len(prej["worlds"])
    Mpath = os.path.join(a.out_dir, f"{pre}_M{kM}.pt")
    window = range(8300 + 10 * s, 8300 + 10 * s + 10)
    L_(f"CONFIRMATORY lineage {s} | M{kM} from {Mpath} | window {list(window)} | N/3 {a.n_units} | B {B} | "
       f"{a.num_envs} envs | {time.strftime('%Y-%m-%d %H:%M')}")
    units = []
    for w in admitted_worlds(laws, window):
        if len(units) >= a.n_units:
            break
        spec = make_world(w, laws)
        nav = nav_gate(skill, spec, cfg, a.seed)
        if min(nav.values()) < 0.85:
            L_(f"  world {w}: nav gate {min(nav.values()):.3f} < 0.85 — skipped (logged)"); continue
        goals = goal_stream(spec); tiers = [spec["tier"][g] for g in goals]
        L_(f"\nUNIT s{s}_{w} | stream {goals} tiers {tiers} | nav min {min(nav.values()):.3f}")
        arms = {}

        def rows_of(rows):
            return [dict(goal=r["goal"], tier=r["tier"], curve=r["master_per_round"], rows=r.get("samples_per_round"),
                         demos=r.get("demos_per_round")) for r in rows]

        def learned(name, comp_fn):
            def fn():
                comp = comp_fn()
                rows, _ = run_unit_v61(spec, skill, comp, cfg, w, goals, r_max=B, train=True,
                                       log=lambda r: L_(f"  [{name}] goal {r['goal']} (tier {r['tier']}): curve "
                                                        f"{r['master_per_round']} rows {r['samples_per_round']} | "
                                                        f"{time.perf_counter()-t0:.0f}s"))
                return rows_of(rows)
            return load_or(f"s{s}_{w}_{name}", fn)

        def m4():
            c = ComposerV61(init_seed=0)
            c.net.load_state_dict(torch.load(Mpath, map_location=DEVICE))
            torch.manual_seed(init_seed(s, w, 4) + 500)
            return c

        def fresh(arm):
            def f():
                seed = init_seed(s, w, arm)
                c = ComposerV61(init_seed=seed); torch.manual_seed(seed + 500); return c
            return f
        arms["M4"] = learned("M4", m4)
        arms["Fa"] = learned("Fa", fresh(1))
        arms["Fb"] = learned("Fb", fresh(2))

        def l_rows():
            Lk = LawKnower(spec)
            out = []
            for p, g in enumerate(goals):
                m = eval_goal_v61(spec, skill, Lk, cfg, w + 11 * p, g)
                out.append(dict(goal=g, tier=tiers[p], curve=[m] * (B + 1)))
            L_(f"  [L] {[round(r['curve'][0], 3) for r in out]}")
            return out
        arms["L"] = load_or(f"s{s}_{w}_L", l_rows)
        if s == 0 or a.g_all:
            def g_rows():
                rows, _ = run_unit_v61(spec, skill, StoreSweeper(spec, seed=w), cfg, w, goals, r_max=B, train=False,
                                       log=lambda r: L_(f"  [G'] goal {r['goal']} (tier {r['tier']}): curve "
                                                        f"{r['master_per_round']} | {time.perf_counter()-t0:.0f}s"))
                return rows_of(rows)
            arms["G"] = load_or(f"s{s}_{w}_G", g_rows)
        else:
            arms["G"] = None
        units.append(dict(id=f"s{s}_{w}", lineage=s, world=w, goals=goals, tiers=tiers, arms=arms))
        json.dump(dict(lineage=s, b_max=B, units=units), open(os.path.join(a.out_dir, f"v61_confirm_s{s}{sfx}.json"), "w"), indent=1)
    # merge every lineage file present into the scorer's input
    merged = dict(b_max=B, units=[])
    for ss in range(3):
        p = os.path.join(a.out_dir, f"v61_confirm_s{ss}{sfx}.json")
        if os.path.exists(p):
            merged["units"] += json.load(open(p))["units"]
    json.dump(merged, open(os.path.join(a.out_dir, f"v61_confirm{sfx}.json"), "w"), indent=1)
    L_(f"lineage {s} done: {len(units)} units | merged {len(merged['units'])} | total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
