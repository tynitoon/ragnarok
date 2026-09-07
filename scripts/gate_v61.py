"""ARC 3 — THE CHEAP GATE (ARC3_PLAN.md 4.1). Runs first; decides whether the arc spends its budget.

    K0  hand-coded, no learning: nav gate, L (law-knower) on every goal, G' (store sweeper) from an empty
        store per goal at the gate budget.   REACHABLE / CONSISTENT
    K1  learned: three fresh arms Fa, Fb, Fc (per-unit init seeds) on the same units at the gate budget.
        FRESH-LEARNS / ROOM, sd_init on the confirmatory's unit, H per tier, outcome-head AUC per round.

Gate worlds = the first two admitted seeds >= 8100 (burned). Every K0/K1 number is printed and written to
craft_v6_out/v61_gate.json; per-(world, arm) checkpoints allow --resume. --smoke suffixes every output and
is excluded from everything downstream.

Usage: python -m scripts.gate_v61 --k0 --k1 [--b-max 4] [--num-envs 64] [--n-conf 12] [--smoke] [--resume]
"""

import argparse
import json
import os
import statistics as st
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import sample_laws, make_world, goal_stream, admitted_worlds, T_MAX
from scripts.hidden_recipe_v55 import nav_gate
from scripts.pair_store_v61 import PairEnv, PairStore, MAX_ITEMS, PAIR_INDEX, N_SLOT, GOAL_COL
from scripts.pair_net_v61 import (ComposerV61, cfg_v61, load_skill, eval_goal_v61, run_unit_v61, area,
                                  init_seed)
from scripts.hand_v61 import LawKnower, StoreSweeper

LAWS_SEED = 31337
GATE_LINEAGE = 9


def log_line(f, s):
    print(s, flush=True)
    f.write(s + "\n"); f.flush()


@torch.no_grad()
def law_auc(composer, spec, skill, cfg, laws):
    """Outcome-head AUC of P(lawful) against the true Bond table, over every equal-tier pair of REAL items
    of this world (tier < T_MAX), each judged from a fresh env holding exactly that pair, EMPTY store —
    i.e. what the WEIGHTS know. Returns (auc, n_pairs)."""
    n, el, tier, decoy = spec["n_items"], spec["el"], spec["tier"], spec["decoy"]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if not decoy[i] and not decoy[j] and tier[i] == tier[j] < T_MAX]
    env = PairEnv(len(pairs), spec, skill, cfg, seed=1, goal=spec["target"])
    env.reset()
    env.base.inv[:] = 0
    for k, (i, j) in enumerate(pairs):
        env.base.inv[k, i] = 1; env.base.inv[k, j] = 1
    env.base._set_state()
    obs = env.obs()
    # the outcome head is trained on GOAL-FREE rows (collect_episode zeroes the goal column before storing);
    # measure it in the same distribution (review finding)
    obs.view(len(pairs), -1)[:, :MAX_ITEMS * N_SLOT].view(len(pairs), MAX_ITEMS, N_SLOT)[..., GOAL_COL] = 0.0
    _, o3, _ = composer.net(obs)
    idx = torch.tensor([int(PAIR_INDEX[i, j]) for (i, j) in pairs], device=DEVICE)
    p = torch.softmax(o3[torch.arange(len(pairs), device=DEVICE), idx], -1)
    p_law = (1.0 - p[:, 0]).cpu().numpy()
    y = np.array([laws["Bond"][el[i], el[j]] for (i, j) in pairs], dtype=bool)
    if y.all() or (~y).all():
        return float("nan"), len(pairs)
    pos, neg = p_law[y], p_law[~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return auc, len(pairs)


def b_star(row, thresh):
    for b, m in enumerate(row["master_per_round"]):
        if b >= 1 and m >= thresh:
            return b
    return float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k0", action="store_true"); ap.add_argument("--k1", action="store_true")
    ap.add_argument("--b-max", type=int, default=4)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--n-conf", type=int, default=12, help="N units of the confirmatory, for se_proj")
    ap.add_argument("--worlds", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default="craft_v6_out")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-persists", action="store_true",
                    help="THE NOTCH (ARC3_PLAN 4.1, not FRESH-LEARNS): the store persists across the stream")
    ap.add_argument("--tag", default="", help="output suffix (e.g. notch) so an earlier gate run is preserved")
    ap.add_argument("--rule", default="v61", choices=["v61", "v62"],
                    help="v61: median b*(tier2)<=2 and median b*(tier3)<=3 on both units; "
                         "v62 (ARC3_PLAN 12): all 3 fresh arms b*(tier2)<=2 AND >=2 of 3 arms b*(tier3)<=B on both units")
    ap.add_argument("--b-test", type=int, default=None, help="v62: B_max(test) fixed at the freeze (default = --b-max)")
    ap.add_argument("--n-test", type=int, default=None, help="v62: N fixed at the freeze (default 9)")
    a = ap.parse_args()
    sfx = ("_smoke" if a.smoke else "") + (f"_{a.tag}" if a.tag else "")
    ck = os.path.join(a.out_dir, f"v61_gate_ckpt{sfx}"); os.makedirs(ck, exist_ok=True)
    logf = open(os.path.join(a.out_dir, f"v61_gate{sfx}.log"), "a")
    L_ = lambda s: log_line(logf, s)                                                      # noqa: E731
    cfg = cfg_v61(num_envs=a.num_envs, r_max=a.b_max)
    cfg["store_persists"] = bool(a.store_persists)
    B = a.b_max
    torch.manual_seed(a.seed)
    laws = sample_laws(LAWS_SEED)
    skill = load_skill(cfg, a.seed, a.out_dir)
    worlds = a.worlds or admitted_worlds(laws, range(8100, 8110), need=2)
    t0 = time.perf_counter()
    L_("=" * 100)
    L_(f"ARC 3 GATE (v61) | worlds {worlds} | B_max(gate) {B} | {a.num_envs} envs | N_conf {a.n_conf} | "
       f"{'store PERSISTS across the stream (NOTCH)' if a.store_persists else 'per-goal store'} | "
       f"{'SMOKE ' if a.smoke else ''}{time.strftime('%Y-%m-%d %H:%M')}")
    L_("=" * 100)

    def ckpt(name):
        return os.path.join(ck, name + ".json")

    def load_or(name, fn):
        p = ckpt(name)
        if a.resume and os.path.exists(p):
            L_(f"  [resume] {name}")
            return json.load(open(p))
        r = fn()
        json.dump(r, open(p, "w"), indent=1)
        return r

    res = dict(worlds=worlds, b_max=B, num_envs=a.num_envs, n_conf=a.n_conf, store_persists=bool(a.store_persists),
               units={})
    for w in worlds:
        spec = make_world(w, laws)
        goals = goal_stream(spec)
        tiers = [spec["tier"][g] for g in goals]
        nav = nav_gate(skill, spec, cfg, a.seed)
        u = dict(goals=goals, tiers=tiers, nav=nav, nav_min=min(nav.values()),
                 plan_steps=[spec["plans"][g]["steps"] for g in goals])
        L_(f"\nWORLD {w} | stream {goals} tiers {tiers} plan steps {u['plan_steps']} | nav min {u['nav_min']:.3f}")
        if u["nav_min"] < 0.85:
            # an instrument failure on a gate world is a REACHABLE failure of the whole gate, never a
            # silent drop to one unit (review finding, critical)
            L_("  NAV GATE FAILS — instrument; REACHABLE = False for the gate (logged)")
            u["nav_fail"] = True; res["units"][w] = u; continue

        if a.k0:
            def k0():
                out = dict(L={}, G={}, G_first={})
                Lk = LawKnower(spec)
                for p, g in enumerate(goals):
                    out["L"][str(g)] = eval_goal_v61(spec, skill, Lk, cfg, w + 11 * p, g)
                    L_(f"  [K0] L goal {g} (tier {tiers[p]}): {out['L'][str(g)]:.3f} | {time.perf_counter()-t0:.0f}s")
                rows, _ = run_unit_v61(spec, skill, StoreSweeper(spec), cfg, w, goals, r_max=B, train=False,
                                       log=lambda r: L_(f"  [K0] G' goal {r['goal']} (tier {r['tier']}): curve "
                                                        f"{r['master_per_round']} first-demo attempt {r.get('first_demo_attempt')} "
                                                        f"| {time.perf_counter()-t0:.0f}s"))
                for r in rows:
                    out["G"][str(r["goal"])] = r["master_per_round"]
                    out["G_first"][str(r["goal"])] = r.get("first_demo_attempt")
                return out
            u["k0"] = load_or(f"{w}_k0", k0)

        if a.k1:
            u["k1"] = {}
            for arm in (1, 2, 3):
                def k1(arm=arm):
                    seed = init_seed(GATE_LINEAGE, w, arm)
                    comp = ComposerV61(init_seed=seed)
                    torch.manual_seed(seed + 500)
                    aucs = []

                    def on_goal(r):
                        auc, npairs = law_auc(comp, spec, skill, cfg, laws)
                        aucs.append(auc)
                        L_(f"  [K1] F{arm} goal {r['goal']} (tier {r['tier']}): curve {r['master_per_round']} "
                           f"rows {r['samples_per_round']} demos {r['demos_per_round']} | AUC {auc:.3f} ({npairs} cells) "
                           f"| {time.perf_counter()-t0:.0f}s")
                    rows, _ = run_unit_v61(spec, skill, comp, cfg, w, goals, r_max=B, train=True, log=on_goal)
                    return dict(init_seed=seed, rows=[dict(goal=r["goal"], tier=r["tier"], curve=r["master_per_round"],
                                                            rows=r["samples_per_round"], demos=r["demos_per_round"])
                                                       for r in rows], auc=aucs,
                                secs=round(time.perf_counter() - t0))
                u["k1"][f"F{arm}"] = load_or(f"{w}_F{arm}", k1)
        res["units"][w] = u
        json.dump(res, open(os.path.join(a.out_dir, f"v61_gate{sfx}.json"), "w"), indent=1)

    # ---- the decisions --------------------------------------------------------------------------
    units = [w for w in worlds if "k1" in res["units"].get(w, {}) or "k0" in res["units"].get(w, {})]
    nav_failed = [w for w in worlds if res["units"].get(w, {}).get("nav_fail")]
    L_(f"\n{'='*100}\nGATE SUMMARY | units {units} | nav-failed {nav_failed} | B {B} | thresh {cfg['thresh']}")
    thresh = cfg["thresh"]
    reachable = (not nav_failed and len(units) == len(worlds) and
                 all(res["units"][w].get("k0", {}).get("L", {}) and
                     min(res["units"][w]["k0"]["L"].values()) >= 0.85 for w in units)) if a.k0 else None
    if len(units) < 2:
        L_("  fewer than 2 units: no decision is computed (REACHABLE = False)")
        res["summary"] = dict(reachable=False, proceed=False, reason="fewer than 2 gate units")
        json.dump(res, open(os.path.join(a.out_dir, f"v61_gate{sfx}.json"), "w"), indent=1)
        return
    if a.k0:
        for w in units:
            k0 = res["units"][w]["k0"]
            L_(f"  {w}: L {[round(v,3) for v in k0['L'].values()]} | G' curves {list(k0['G'].values())} | "
               f"G' first-demo attempts {list(k0['G_first'].values())}")
        L_(f"  REACHABLE = {reachable}  (nav >= 0.85 and L >= 0.85 on every goal)")
    if a.k1:
        A = {}          # A[w][arm] per-unit area over b = 1..B (mean over goals)
        Ag = {}         # per-goal areas
        bstar = {}
        for w in units:
            k1 = res["units"][w]["k1"]
            A[w], Ag[w], bstar[w] = {}, {}, {}
            for arm, d in k1.items():
                A[w][arm] = st.mean(area(dict(master_per_round=r["curve"]), 1, B) for r in d["rows"])
                Ag[w][arm] = [area(dict(master_per_round=r["curve"]), 1, B) for r in d["rows"]]
                bstar[w][arm] = [b_star(dict(master_per_round=r["curve"]), thresh) for r in d["rows"]]
            if a.k0:
                k0 = res["units"][w]["k0"]
                A[w]["L"] = st.mean(k0["L"].values())
                Ag[w]["L"] = list(k0["L"].values())
                A[w]["G"] = st.mean(area(dict(master_per_round=c), 1, B) for c in k0["G"].values())
                Ag[w]["G"] = [area(dict(master_per_round=c), 1, B) for c in k0["G"].values()]
            L_(f"  {w}: A(u) " + " ".join(f"{k}={v:.3f}" for k, v in A[w].items()) +
               f" | b* per goal " + " ".join(f"{k}={v}" for k, v in bstar[w].items()))
        arms = ["F1", "F2", "F3"]
        pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
        d_unit = [A[w][x] - A[w][y] for w in units for (x, y) in pairs]
        d_goal = [Ag[w][x][p] - Ag[w][y][p] for w in units for (x, y) in pairs for p in range(3)]
        sd_init = st.pstdev(d_unit) * (len(d_unit) / (len(d_unit) - 1)) ** 0.5 if len(d_unit) > 1 else float("nan")
        sd_goal = st.pstdev(d_goal) * (len(d_goal) / (len(d_goal) - 1)) ** 0.5 if len(d_goal) > 1 else float("nan")
        med_b = {}
        for p in range(3):
            med_b[p] = [st.median(bstar[w][arm][p] for arm in arms) for w in units]
        n_reach = {p: [sum(1 for arm in arms if bstar[w][arm][p] <= (2 if p == 0 else B)) for w in units]
                   for p in range(3)}                    # arms reaching 0.6 within the budget, per unit
        if a.rule == "v61":
            fresh_learns = all(b <= 2 for b in med_b[0]) and all(b <= 3 for b in med_b[1])   # ARC3_PLAN 4.1
            # B_max(test) and N follow from K1 (ARC3_PLAN 4.4): clamp(b*(tier-4 median) + 1, 3, 4); N 12 at B 3, 9 at B 4
            b4 = st.median(med_b[2]) if all(b < float("inf") for b in med_b[2]) else float("inf")
            B_test = 4 if b4 == float("inf") else int(min(4, max(3, b4 + 1)))
            N = 12 if B_test == 3 else 9
        else:
            # ARC3_PLAN 12 (v62): a FRACTION over the three fresh arms, not a 3-arm median
            fresh_learns = all(k == 3 for k in n_reach[0]) and all(k >= 2 for k in n_reach[1])
            B_test = a.b_test or B
            N = a.n_test or 9
        L_(f"  fresh arms reaching 0.6 within budget, per unit: tier2 {n_reach[0]} (of 3, within 2 rounds) "
           f"tier3 {n_reach[1]} tier4 {n_reach[2]} (within {B})")
        se_res = 0.0625 * (2 / (3 * B_test * N)) ** 0.5
        se_proj = max(sd_init / (2 * N) ** 0.5, se_res)
        se_proj_diag = max(sd_init / (2 * a.n_conf) ** 0.5, 0.0625 * (2 / (3 * B * a.n_conf)) ** 0.5)
        L_(f"  sd_init (per-UNIT pairwise fresh differences, {len(d_unit)} values) = {sd_init:.4f} | per-goal sd "
           f"(diagnostic, {len(d_goal)} values) = {sd_goal:.4f}")
        L_(f"  b* median fresh per unit: tier2 {med_b[0]} tier3 {med_b[1]} tier4 {med_b[2]} -> B_max(test) {B_test}, N {N}")
        L_(f"  se_res {se_res:.4f} | se_proj(B_test {B_test}, N {N}) {se_proj:.4f} | diagnostic se_proj(B {B}, N {a.n_conf}) {se_proj_diag:.4f}")
        L_(f"  FRESH-LEARNS = {fresh_learns}  (rule {a.rule}: " +
           ("median b*(tier2) <= 2 and median b*(tier3) <= 3 on both units" if a.rule == "v61" else
            f"all 3 arms b*(tier2) <= 2 and >= 2 of 3 arms b*(tier3) <= {B} on both units") +
           "; tier 4 is the reach position, printed, not required)")
        if a.k0:
            H_u = [A[w]["L"] - st.mean(A[w][arm] for arm in arms) for w in units]
            H_t = [st.mean(Ag[w]["L"][p] - st.mean(Ag[w][arm][p] for arm in arms) for w in units) for p in range(3)]
            Hp = [A[w]["L"] - A[w]["G"] for w in units]
            H = st.mean(H_u); room = H >= 4 * se_proj
            L_(f"  H = mean_u[A_L - mean fresh] = {H:.4f} (per unit {[round(x,3) for x in H_u]}; per tier "
               f"{[round(x,3) for x in H_t]}) | H' = A_L - A_G' = {st.mean(Hp):.4f} | 4*se_proj = {4*se_proj:.4f}")
            L_(f"  ROOM = {room}")
            proceed = bool(reachable and fresh_learns and room)
            L_(f"\n  GATE -> {'PROCEED' if proceed else 'STOP'}" +
               ("" if proceed else f"  (REACHABLE {reachable}, FRESH-LEARNS {fresh_learns}, ROOM {room}; see notch list)"))
            res["summary"] = dict(reachable=reachable, fresh_learns=fresh_learns, room=room, proceed=proceed,
                                  H=H, H_unit=H_u, H_tier=H_t, H_prime=st.mean(Hp), sd_init=sd_init, sd_goal=sd_goal,
                                  se_res=se_res, se_proj=se_proj, B_test=B_test, N=N, b_star_median=med_b, A=A,
                                  G_first_median=[res["units"][w]["k0"]["G_first"] for w in units])
        json.dump(res, open(os.path.join(a.out_dir, f"v61_gate{sfx}.json"), "w"), indent=1)
    L_(f"  total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
