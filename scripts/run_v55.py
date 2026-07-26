"""v55 driver — HIDDEN-RECIPE persistent world. Prereg FROZEN (preregistration.md) + AMENDMENT 1.

Arms, all sharing ONE frozen childhood nav skill per seed:
  A  accumulating  — one composer+buffer across every goal in the world (the treatment)
  B  amnesic       — fresh composer+buffer per goal, IDENTICAL hidden observation (paired same-goal
                     control, so goal difficulty cancels by construction)
  D  strong        — the UNMODIFIED v54 mechanism WITH the affordance oracle restored, fresh per goal.
                     STRICTLY better informed than A: quantifies A's memory in units of "the granted DAG"
  C  foreign       — warm-started on an equal-VOLUME buffer collected in a DIFFERENT world
  E  compute-match — fresh composer given A's ENTIRE cumulative spend on the hardest GOAL-NECESSARY goal

Usage:
  python -m scripts.run_v55 --gate            # KILL-0 feasibility gate (one shot, cheap)
  python -m scripts.run_v55 --seed 0 [--resume]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.childhood_v50 import train_childhood
from scripts.hidden_recipe_v55 import (permute_spec, admitted_goals, nav_gate, run_goal,
                                       goal_swap_separation, Composer, Buffer, collect_episode,
                                       relabel, HiddenEnv)

WORLDS = {0: 3002, 1: 3003, 2: 3016}          # AMENDMENT 1 (frozen before any run)
FOREIGN = {0: 3019, 1: 3021, 2: 3024}
FALLBACK = [3041, 3048, 3050, 3059]


def build_cfg(a):
    cfg = dict(num_envs=a.num_envs, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
               nav_max_steps=40, skill_iters=a.skill_iters, option_timeout=16, macro_budget=26,
               episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
               epsilon=0.05, temp=1.0, thresh=0.6, r_max=a.r_max, skill_stochastic=True,
               mgr_entropy=0.03, router_iters=0)
    return cfg


def world_for(seed, cfg, skill_specs, a):
    """Pick the seed's world, applying the pre-registered nav gate (the only permitted pre-run action)."""
    order = [WORLDS[seed]] + [w for w in FALLBACK if w not in WORLDS.values()]
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--skill-iters", type=int, default=400)
    p.add_argument("--r-max", type=int, default=10)
    p.add_argument("--gate", action="store_true", help="KILL-0 feasibility gate only")
    p.add_argument("--nav-min", type=float, default=0.85,
                   help="FROZEN at 0.85 for the experiment; lower ONLY to smoke-test code paths")
    p.add_argument("--arms", default="ABDCE")
    p.add_argument("--max-goals", type=int, default=0, help="smoke only: cap the goal stream")
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = build_cfg(a)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()

    ni = 14
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    skill, c_skill = train_childhood(skill_specs, cfg, a.seed)
    print(f"[v55] device={DEVICE} | childhood skill ready ({c_skill/1e6:.2f}M) | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    # ---- world + pre-registered nav gate (only permitted pre-run action) --------------------------
    chosen, nav = None, None
    for w in world_for(a.seed, cfg, skill_specs, a):
        spec = permute_spec(gen_tree(w, n_items=ni), w)
        nav = nav_gate(skill, spec, cfg, a.seed)
        lo = min(nav.values())
        print(f"  nav gate world {w}: per-cell-type {nav} | min {lo:.3f} "
              f"{'PASS' if lo >= a.nav_min else 'FAIL -> next world'}", flush=True)
        if lo >= a.nav_min:
            chosen = w
            break
    if chosen is None:
        print("  NAV GATE FAILED ON ALL WORLDS — abort (do not tune; report)", flush=True)
        return
    spec = permute_spec(gen_tree(chosen, n_items=ni), chosen)
    adm = admitted_goals(spec)
    nec = [g for g, pc, b in adm if b > cfg["macro_budget"]]
    goals = [g for g, _, _ in adm]
    if a.max_goals:                                   # smoke only
        goals = goals[:a.max_goals]; nec = [g for g in nec if g in goals]
    print(f"  world {chosen} | {len(adm)} admitted goals | GOAL-NECESSARY {len(nec)} = {nec}\n"
          f"    pc    {[pc for _, pc, _ in adm]}\n"
          f"    blind {[round(b,1) for _, _, b in adm]}", flush=True)

    res = dict(seed=a.seed, world=chosen, foreign=FOREIGN[a.seed], nav=nav, c_skill=c_skill,
               admitted=[(g, pc, round(b, 2)) for g, pc, b in adm], necessary=nec, arms={})
    jp = os.path.join(a.out_dir, f"v55_s{a.seed}.json")
    ck = os.path.join(a.out_dir, f"v55_ckpt_s{a.seed}.pt")

    # ---- KILL-0 feasibility gate ------------------------------------------------------------------
    if a.gate:
        g0 = goals[0]
        r = run_goal(spec, skill, Composer("memory"), Buffer(cap=400_000), cfg, a.seed, g0, hidden=True)
        ok = r["master"] >= cfg["thresh"]
        print(f"\n  KILL-0 GATE world {chosen} easiest goal {g0} (pc {adm[0][1]}): master {r['master']} "
              f"in {r['rounds']} rounds -> {'PROCEED' if ok else 'KILL-1 FIRES'}", flush=True)
        res["gate"] = dict(goal=g0, **r, proceed=ok)
        json.dump(res, open(os.path.join(a.out_dir, f"v55_gate_s{a.seed}.json"), "w"), indent=2)
        return

    if a.resume and os.path.exists(jp) and os.path.exists(ck):
        res = json.load(open(jp))
        print(f"  RESUME: arms done {list(res['arms'].keys())}", flush=True)

    def save(cA=None, bA=None):
        json.dump(res, open(jp, "w"), indent=2)
        if cA is not None:
            torch.save(dict(net=cA.net.state_dict(), opt=cA.opt.state_dict(), buf=bA.state_dict()), ck)

    def arm(tag, hidden, gamma, net, accumulate, goal_list, r_max=None, warm=None):
        comp, buf = warm if warm is not None else (Composer(net), Buffer())
        rows, tot = [], 0
        for k, g in enumerate(goal_list):
            if not accumulate and warm is None:
                comp, buf = Composer(net), Buffer(cap=400_000)
            r = run_goal(spec, skill, comp, buf, cfg, a.seed + 11 * k + 1, g,
                         hidden=hidden, gamma=gamma, r_max=r_max)
            r["necessary"] = g in nec
            rows.append(r); tot += r["prim"]
            print(f"    [{tag}] goal {g:>2} (pc {dict((x,y) for x,y,_ in adm)[g]:>2}"
                  f"{', NEC' if r['necessary'] else '     '}): zs {r['zero_shot']:.2f} -> "
                  f"master {r['master']:.2f} in {r['rounds']:>2} rounds "
                  f"({'M' if r['mastered'] else 'x'}) | buf {buf.n} | {time.perf_counter()-t0:.0f}s",
                  flush=True)
            res["arms"][tag] = dict(rows=rows, total_prim=tot)
            save(comp if tag == "A" else None, buf if tag == "A" else None)
        return comp, buf, rows, tot

    # ---- A: accumulating (the treatment) -----------------------------------------------------------
    if "A" in a.arms and "A" not in res["arms"]:
        print("\n  === ARM A (accumulating: one composer+buffer across all goals) ===", flush=True)
        cA, bA, rowsA, totA = arm("A", True, 0.7, "memory", True, goals)
    elif os.path.exists(ck):
        st = torch.load(ck, map_location=DEVICE)
        cA, bA = Composer("memory"), Buffer()
        cA.net.load_state_dict(st["net"]); cA.opt.load_state_dict(st["opt"]); bA.load_state_dict(st["buf"])
        rowsA, totA = res["arms"]["A"]["rows"], res["arms"]["A"]["total_prim"]
    else:
        cA = bA = None; rowsA, totA = [], 0

    # ---- B: amnesic control on EVERY goal ----------------------------------------------------------
    if "B" in a.arms and "B" not in res["arms"]:
        print("\n  === ARM B (amnesic: fresh composer+buffer per goal, same hidden observation) ===", flush=True)
        arm("B", True, 0.7, "memory", False, goals)

    # ---- D: strong baseline (v54 mechanism WITH the oracle) ----------------------------------------
    if "D" in a.arms and "D" not in res["arms"]:
        print("\n  === ARM D (STRONG baseline: v54 mechanism, affordance oracle RESTORED) ===", flush=True)
        arm("D", False, 1.0, "router", False, goals)

    # ---- C: foreign memory (equal volume, different world) -----------------------------------------
    if "C" in a.arms and "C" not in res["arms"] and cA is not None:
        print(f"\n  === ARM C (foreign memory from world {FOREIGN[a.seed]}, equal volume) ===", flush=True)
        fspec = permute_spec(gen_tree(FOREIGN[a.seed], n_items=ni), FOREIGN[a.seed])
        fgoals = [g for g, _, _ in admitted_goals(fspec)]
        cC, bC = Composer("memory"), Buffer()
        fenv = HiddenEnv(cfg["num_envs"], fspec, skill, cfg, seed=a.seed + 5000,
                         goal=fgoals[0], hidden=True)
        k = 0
        while bC.n < bA.n and k < 400:
            g = fgoals[k % len(fgoals)]
            s, act, us = collect_episode(fenv, cC, cfg["epsilon"], cfg["temp"], g)
            ss, aa = relabel(s, act, us, cfg["max_samples_per_ep"], gamma=0.7)
            if ss is not None:
                bC.add(ss, aa)
            if k % cfg["episodes_per_round"] == cfg["episodes_per_round"] - 1:
                cC.train_steps(bC, cfg["train_steps_per_round"])
            k += 1
        print(f"    foreign buffer {bC.n} vs A {bA.n} ({k} episodes) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        res["arms"]["C_foreign_buffer"] = dict(n=bC.n, target=bA.n, episodes=k)
        arm("C", True, 0.7, "memory", True, nec, warm=(cC, bC))

    # ---- E: compute-matched from scratch on the hardest GOAL-NECESSARY goal -------------------------
    if "E" in a.arms and "E" not in res["arms"] and nec:
        hardest = max(nec, key=lambda g: dict((x, y) for x, y, _ in adm)[g])
        per_round = cfg["episodes_per_round"] * cfg["num_envs"] * cfg["macro_budget"] * cfg["option_timeout"]
        rE = max(1, int(totA // per_round))
        print(f"\n  === ARM E (compute-matched: fresh composer, A's total {totA/1e6:.1f}M "
              f"= {rE} rounds, on hardest GOAL-NECESSARY goal {hardest}) ===", flush=True)
        r = run_goal(spec, skill, Composer("memory"), Buffer(), cfg, a.seed + 99, hardest,
                     hidden=True, gamma=0.7, r_max=rE)
        print(f"    [E] goal {hardest}: master {r['master']:.2f} in {r['rounds']} rounds "
              f"({'M' if r['mastered'] else 'x'})", flush=True)
        res["arms"]["E"] = dict(r, r_max=rE)          # r already carries `goal`
        save()

    # ---- P2: goal-swap separation on A (hidden) and D (oracle) -------------------------------------
    if "A" in res["arms"] and cA is not None and "swap" not in res:
        probe = nec if len(nec) >= 2 else goals[:4]
        sA, ownA = goal_swap_separation(spec, skill, cA, cfg, a.seed, probe, hidden=True)
        res["swap"] = dict(A=round(sA, 4), A_own={int(k): v for k, v in ownA.items()}, goals=probe)
        print(f"\n  P2 goal-swap separation | A = {sA:+.3f} "
              f"(>=0.30 => the goal became load-bearing; v54's analogue was ~0)", flush=True)
        save()

    # ---- summary ------------------------------------------------------------------------------------
    def mrate(tag):
        rows = res["arms"].get(tag, {}).get("rows", [])
        rn = [r for r in rows if r.get("necessary")]
        return (sum(r["mastered"] for r in rn) / len(rn) if rn else float("nan")), len(rn)

    for tag in ("A", "B", "D", "C"):
        m, n = mrate(tag)
        if n:
            print(f"  {tag}: GOAL-NECESSARY master rate {m:.2f} ({n} goals)", flush=True)
    res["elapsed_s"] = round(time.perf_counter() - t0)
    save()
    print(f"\n  done in {res['elapsed_s']}s -> {jp}", flush=True)


if __name__ == "__main__":
    main()
