"""v55 driver — HIDDEN-RECIPE persistent world. Prereg FROZEN (preregistration.md) + AMENDMENTS 1-2.

Arms, all sharing ONE frozen childhood nav skill per seed (saved/reloaded so a resume cannot swap it):
  A  accumulating  — one composer+buffer across every goal in the world (the treatment)
  B  amnesic       — fresh composer+buffer per goal, IDENTICAL hidden observation (paired same-goal
                     control: same world, same goal, same grids, differing ONLY in memory)
  D  strong        — the UNMODIFIED v54 mechanism WITH the affordance oracle restored, fresh per goal.
                     STRICTLY better informed than A: quantifies A's memory in units of "the granted DAG"
  C  foreign       — warm-started on an equal-VOLUME buffer collected in a DIFFERENT world, then run on
                     the SAME full goal stream as A (so A-vs-C isolates buffer PROVENANCE, not stream length)
  E  compute-match — fresh composer given A's entire cumulative spend on the hardest GOAL-NECESSARY goal

Usage:
  python -m scripts.run_v55 --gate            # KILL-0 feasibility gate (one shot, world 3002)
  python -m scripts.run_v55 --seed 0 [--resume]
"""

import argparse
import copy
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import train_childhood, nav_env, NAV_ACTIONS
from scripts.hidden_recipe_v55 import (permute_spec, admitted_goals, nav_gate, run_goal, eval_goal,
                                       Composer, Buffer, collect_episode, relabel, HiddenEnv)

WORLDS = {0: 3002, 1: 3003, 2: 3016}          # AMENDMENT 1 (frozen before any run)
FOREIGN = {0: 3019, 1: 3021, 2: 3024}
FALLBACK = [3041, 3048, 3050, 3059]


def get_skill(cfg, seed, specs, out_dir):
    """Train the childhood skill ONCE per seed and persist it. A resume must reuse the SAME skill:
    it is a shared factor of every arm, and CUDA conv backward is not bit-reproducible."""
    p = os.path.join(out_dir, f"v55_skill_s{seed}.pt")
    if os.path.exists(p):
        obs_dim = nav_env(specs[0], cfg, seed, 2).obs_dim
        net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
        ppo = DiscretePPO(obs_dim, NAV_ACTIONS, net=net, entropy=cfg["entropy"], gamma=0.99, lam=0.95)
        ppo.net.load_state_dict(torch.load(p, map_location=DEVICE))
        return ppo, 0, True
    skill, c = train_childhood(specs, cfg, seed)
    torch.save(skill.net.state_dict(), p)
    return skill, c, False


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
    p.add_argument("--max-hours", type=float, default=10.0, help="per-seed slice of the 32 GPU-hour cap")
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = dict(num_envs=a.num_envs, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
               nav_max_steps=40, skill_iters=a.skill_iters, option_timeout=16, macro_budget=26,
               episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
               epsilon=0.05, temp=1.0, thresh=0.6, r_max=a.r_max, skill_stochastic=True,
               mgr_entropy=0.03, router_iters=0)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()

    ni = 14
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    skill, c_skill, reused = get_skill(cfg, a.seed, skill_specs, a.out_dir)
    print(f"[v55] device={DEVICE} | childhood skill {'RELOADED' if reused else f'trained ({c_skill/1e6:.2f}M)'}"
          f" | {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- world + pre-registered nav gate (the only permitted pre-run action) ----------------------
    chosen, nav = None, None
    for w in [WORLDS[a.seed]] + [x for x in FALLBACK if x not in WORLDS.values()]:
        spec = permute_spec(gen_tree(w, n_items=ni), w)
        nav = nav_gate(skill, spec, cfg, a.seed)
        lo = min(nav.values())
        print(f"  nav gate world {w}: {nav} | min {lo:.3f} "
              f"{'PASS' if lo >= a.nav_min else 'FAIL -> next world'}", flush=True)
        if lo >= a.nav_min:
            chosen = w
            break
    if chosen is None:
        print("  NAV GATE FAILED ON ALL WORLDS — abort (do not tune; report)", flush=True)
        return
    spec = permute_spec(gen_tree(chosen, n_items=ni), chosen)
    adm = admitted_goals(spec)
    pc_of = {g: pc for g, pc, _ in adm}
    nec = [g for g, _, b in adm if b > cfg["macro_budget"]]
    goals = [g for g, _, _ in adm]
    if a.max_goals:                                   # smoke only
        goals = goals[:a.max_goals]; nec = [g for g in nec if g in goals]
    probe = nec if len(nec) >= 2 else goals[:4]       # P2 probe set, fixed BEFORE any arm runs
    print(f"  world {chosen} | {len(adm)} admitted | GOAL-NECESSARY {len(nec)} = {nec} | probe {probe}\n"
          f"    pc    {[pc for _, pc, _ in adm]}\n"
          f"    blind {[round(b,1) for _, _, b in adm]}", flush=True)

    # env seed is a function of GOAL IDENTITY, never list position, so every arm sees the SAME grids
    # for the same goal (arm C runs a shorter list; without this its evals would differ from A's)
    gseed = lambda g: a.seed + 11 * goals.index(g) + 1                                   # noqa: E731

    # ---- KILL-0 feasibility gate ------------------------------------------------------------------
    if a.gate:
        assert chosen == 3002, f"KILL-0 is pre-registered on world 3002, got {chosen}"
        g0 = goals[0]
        r = run_goal(spec, skill, Composer("memory"), Buffer(cap=400_000), cfg, gseed(g0), g0, hidden=True)
        ok = r["master"] >= cfg["thresh"]
        print(f"\n  KILL-0 GATE world {chosen} easiest goal {g0} (pc {pc_of[g0]}): master {r['master']} "
              f"in {r['rounds']} rounds -> {'PROCEED' if ok else 'KILL-1 FIRES'}", flush=True)
        json.dump(dict(world=chosen, nav=nav, gate=dict(r, proceed=ok)),
                  open(os.path.join(a.out_dir, f"v55_gate_s{a.seed}.json"), "w"), indent=2)
        return

    res = dict(seed=a.seed, world=chosen, foreign=FOREIGN[a.seed], nav=nav, c_skill=c_skill,
               admitted=[(g, pc, round(b, 2)) for g, pc, b in adm], necessary=nec, probe=probe, arms={})
    jp = os.path.join(a.out_dir, f"v55_s{a.seed}.json")
    ck = os.path.join(a.out_dir, f"v55_ckpt_s{a.seed}.pt")
    if a.resume and os.path.exists(jp):
        res = json.load(open(jp)); res["resumed"] = True
        print(f"  RESUME: complete arms "
              f"{[t for t in ('A','B','D','C') if res['arms'].get(t, {}).get('complete')]}", flush=True)

    done = lambda t: bool(res["arms"].get(t, {}).get("complete"))                          # noqa: E731
    d_composers = {}

    def save(cA=None, bA=None):
        json.dump(res, open(jp, "w"), indent=2)
        if cA is not None:
            torch.save(dict(net=cA.net.state_dict(), opt=cA.opt.state_dict(), buf=bA.state_dict(),
                            rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                       ck)

    def arm(tag, hidden, gamma, net, accumulate, goal_list, r_max=None, warm=None):
        comp, buf = warm if warm is not None else (Composer(net), Buffer())
        rows, tot = [], 0
        for k, g in enumerate(goal_list):
            if not accumulate:
                comp, buf = Composer(net), Buffer(cap=400_000)     # fresh per goal, small buffer
            r = run_goal(spec, skill, comp, buf, cfg, gseed(g), g, hidden=hidden, gamma=gamma, r_max=r_max)
            r["necessary"] = g in nec
            rows.append(r); tot += r["prim"]
            if tag == "D" and g in probe:
                d_composers[g] = copy.deepcopy(comp)               # P2 needs D's own composer per goal
            print(f"    [{tag}] goal {g:>2} (pc {pc_of[g]:>2}{', NEC' if r['necessary'] else '    '}): "
                  f"zs {r['zero_shot']:.2f} -> master {r['master']:.2f} in {r['rounds']:>2} rounds "
                  f"({'M' if r['mastered'] else 'x'}) | demos r1 {r['demos_per_round'][0] if r['demos_per_round'] else 0}"
                  f" | buf {buf.n} | {time.perf_counter()-t0:.0f}s", flush=True)
            res["arms"][tag] = dict(rows=rows, total_prim=tot, complete=(k == len(goal_list) - 1),
                                    collect_prim=sum(x["collect_prim"] for x in rows),
                                    eval_prim=sum(x["eval_prim"] for x in rows),
                                    buf_capped=bool(buf.n >= buf.cap))
            save(comp if tag == "A" else None, buf if tag == "A" else None)
            if time.perf_counter() - t0 > a.max_hours * 3600:
                print(f"    !! {a.max_hours}h cap reached — aborting cleanly with state saved "
                      f"(prereg: do not extend mid-run)", flush=True)
                raise SystemExit(3)
        return comp, buf, rows, tot

    # ---- A: accumulating (the treatment) -----------------------------------------------------------
    cA = bA = None
    if "A" in a.arms and not done("A"):
        print("\n  === ARM A (accumulating: one composer+buffer across all goals) ===", flush=True)
        cA, bA, _, totA = arm("A", True, 0.7, "memory", True, goals)
    elif done("A") and os.path.exists(ck):
        st = torch.load(ck, map_location=DEVICE)
        cA, bA = Composer("memory"), Buffer()
        cA.net.load_state_dict(st["net"]); cA.opt.load_state_dict(st["opt"]); bA.load_state_dict(st["buf"])
        totA = res["arms"]["A"]["total_prim"]
        print(f"  arm A reloaded from checkpoint (buffer {bA.n}, spend {totA/1e6:.1f}M)", flush=True)
    else:
        totA = 0

    # ---- B: amnesic control on EVERY goal ----------------------------------------------------------
    if "B" in a.arms and not done("B"):
        print("\n  === ARM B (amnesic: fresh composer+buffer per goal, same hidden observation) ===", flush=True)
        arm("B", True, 0.7, "memory", False, goals)

    # ---- D: strong baseline (v54 mechanism WITH the oracle) ----------------------------------------
    if "D" in a.arms and not done("D"):
        print("\n  === ARM D (STRONG baseline: v54 mechanism, affordance oracle RESTORED) ===", flush=True)
        arm("D", False, 1.0, "router", False, goals)

    # ---- P2: goal-swap separation, on A (hidden) and on D (oracle) ---------------------------------
    def swap(comp_of, hidden):
        own = {Y: eval_goal(spec, skill, comp_of(Y), cfg, gseed(Y), Y, hidden) for Y in probe}
        d = [own[Y] - eval_goal(spec, skill, comp_of(Y), cfg, gseed(Y), Y, hidden, command=X)
             for Y in probe for X in probe if X != Y]
        return (sum(d) / len(d) if d else 0.0), own

    if "swap" not in res and cA is not None and d_composers:
        sA, ownA = swap(lambda Y: cA, True)
        sD, ownD = swap(lambda Y: d_composers[Y], False)
        res["swap"] = dict(A=round(sA, 4), D=round(sD, 4), goals=probe,
                           A_own={int(k): v for k, v in ownA.items()},
                           D_own={int(k): v for k, v in ownD.items()})
        print(f"\n  P2 goal-swap | S_A {sA:+.3f} (need >=0.30) | S_D {sD:+.3f} (need <0.10) | "
              f"mean own A {sum(ownA.values())/len(ownA):.2f} D {sum(ownD.values())/len(ownD):.2f}", flush=True)
        save()

    # ---- C: foreign memory (equal volume, different world), SAME full goal stream as A -------------
    if "C" in a.arms and not done("C") and bA is not None:
        print(f"\n  === ARM C (foreign memory from world {FOREIGN[a.seed]}, equal volume) ===", flush=True)
        fspec = permute_spec(gen_tree(FOREIGN[a.seed], n_items=ni), FOREIGN[a.seed])
        fgoals = [g for g, _, _ in admitted_goals(fspec)]
        cC, bC = Composer("memory"), Buffer()
        fenv = HiddenEnv(cfg["num_envs"], fspec, skill, cfg, seed=a.seed + 5000, goal=fgoals[0], hidden=True)
        k = 0
        while bC.n < bA.n and k < 1200:
            g = fgoals[k % len(fgoals)]
            s, act, us = collect_episode(fenv, cC, cfg["epsilon"], cfg["temp"], g)
            ss, aa = relabel(s, act, us, cfg["max_samples_per_ep"], gamma=0.7)
            if ss is not None:
                bC.add(ss, aa)
            if k % cfg["episodes_per_round"] == cfg["episodes_per_round"] - 1:
                cC.train_steps(bC, cfg["train_steps_per_round"])
            k += 1
        print(f"    foreign buffer {bC.n} vs A {bA.n} (ratio {bC.n/max(1,bA.n):.2f}, {k} episodes) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        res["arms"]["C_foreign_buffer"] = dict(n=bC.n, target=bA.n, episodes=k,
                                               ratio=round(bC.n / max(1, bA.n), 3))
        arm("C", True, 0.7, "memory", True, goals, warm=(cC, bC))

    # ---- E: compute-matched from scratch on the hardest GOAL-NECESSARY goal -------------------------
    if "E" in a.arms and "E" not in res["arms"] and nec and totA:
        hardest = max(nec, key=lambda g: pc_of[g])
        per_round = cfg["episodes_per_round"] * cfg["num_envs"] * cfg["macro_budget"] * cfg["option_timeout"]
        rE = max(1, int(totA // per_round))     # totA mixes collection+eval => E is OVER-granted (conservative for P6)
        print(f"\n  === ARM E (compute-matched: A's total {totA/1e6:.1f}M = {rE} rounds on hardest "
              f"GOAL-NECESSARY goal {hardest}) ===", flush=True)
        r = run_goal(spec, skill, Composer("memory"), Buffer(), cfg, gseed(hardest), hardest,
                     hidden=True, gamma=0.7, r_max=rE)
        print(f"    [E] goal {hardest}: master {r['master']:.2f} in {r['rounds']} rounds "
              f"({'M' if r['mastered'] else 'x'})", flush=True)
        res["arms"]["E"] = dict(r, r_max=rE)
        save()

    res["elapsed_s"] = round(time.perf_counter() - t0)
    save()
    print(f"\n  done in {res['elapsed_s']}s -> {jp}  (score with: python -m scripts.score_v55)", flush=True)


if __name__ == "__main__":
    main()
