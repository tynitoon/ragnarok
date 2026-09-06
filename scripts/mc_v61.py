"""ARC 3 — CPU Monte Carlo of the law-world generator (ARC3_PLAN.md 2.2, 3.2, 4.7). No GPU, no learning.

Prints the MEASURED tables the gate is checked against: roster statistics, world admission over the
named seed windows, the chain-blind stream per admitted world, lineage coverage of the bond table, and
the symbolic cost of an ideal law-free sweep. Frozen numbers land in preregistration.md (v61).

Usage: python -m scripts.mc_v61 [--n 200] [--json craft_v6_out/v61_mc.json]
"""

import argparse
import json
import statistics as st

import numpy as np

from ragnarok.environments.law_world import (sample_laws, make_world, goal_stream, T_MAX, INERT, BOTCH)

LAWS_SEED = 31337
WINDOWS = dict(gate=list(range(8100, 8110)), probe=[8150],
               pre0=list(range(8200, 8210)), pre1=list(range(8210, 8220)), pre2=list(range(8220, 8230)),
               test0=list(range(8300, 8310)), test1=list(range(8310, 8320)), test2=list(range(8320, 8330)))


def exposed_cells(spec):
    """Bond-table cells (unordered element pairs) this world can EXERCISE: both items real, same tier < T_MAX.
    A lineage's coverage is the union over its worlds, divided by the number of true bond cells."""
    n, el, tier, decoy = spec["n_items"], spec["el"], spec["tier"], spec["decoy"]
    cells = set()
    for i in range(n):
        for j in range(i + 1, n):
            if decoy[i] or decoy[j] or tier[i] != tier[j] or tier[i] >= T_MAX:
                continue
            cells.add((min(el[i], el[j]), max(el[i], el[j])))
    return cells


def true_cells(laws):
    K = laws["K"]
    return {(a, b) for a in range(K) for b in range(a + 1, K) if laws["Bond"][a, b]}


def sweep_cost(spec, goal, rng, n_mc=50):
    """Symbolic IDEAL law-free sweep from an EMPTY store (ARC3_PLAN 2.7 G'): perfect nav, quota, three
    outcomes; the sweeper knows the tier law and the gate law, not Bond/Out. It tries untried equal-tier
    pairs among what it can build, cheapest-to-rebuild first, random ties; remembers products; retries
    a gated raw only while holding a tier >= 2 item. Returns macro-attempts until the goal is first
    obtained (cap 576 = 3 rounds), median over n_mc runs."""
    n, tier, gate, decoy, po, q = (spec["n_items"], spec["tier"], spec["gate"], spec["decoy"],
                                   spec["pair_out"], spec["quota"])
    raws = [i for i in range(n) if tier[i] == 1]
    costs = []
    for _ in range(n_mc):
        known = {}                     # (i,j) -> product slot (or INERT)
        dead = set()                   # botched pairs
        steps, got = 0, False
        while steps < 576 and not got:
            inv = {i: 0 for i in range(n)}
            quota = {r: q for r in raws}
            ep = 0
            while ep < 48 and steps < 576 and not got:
                # 1. can we build toward the goal with known recipes? (needed closure)
                need = {goal}
                frontier = [goal]
                while frontier:
                    k = frontier.pop()
                    if inv[k] > 0:
                        continue
                    for (i, j), p in known.items():
                        if p == k:
                            for s in (i, j):
                                if s not in need:
                                    need.add(s); frontier.append(s)
                act = None
                for (i, j), p in known.items():
                    if p in need and p >= 0 and inv[p] == 0 and inv[i] > 0 and inv[j] > 0:
                        act = ("combine", i, j); break
                if act is None:
                    for s in need:
                        if tier[s] == 1 and inv[s] == 0 and quota[s] > 0 and (gate[s] == 0 or any(inv[k] > 0 and tier[k] >= 2 for k in range(n))):
                            act = ("collect", s); break
                # 2. otherwise experiment: an untried equal-tier pair among held items, else collect
                if act is None:
                    held = [i for i in range(n) if inv[i] > 0]
                    cands = [(i, j) for a, i in enumerate(held) for j in held[a + 1:]
                             if tier[i] == tier[j] < T_MAX and (min(i, j), max(i, j)) not in known
                             and (min(i, j), max(i, j)) not in dead]
                    if cands:
                        i, j = cands[int(rng.integers(len(cands)))]
                        act = ("combine", i, j)
                if act is None:
                    # collect a raw that enables a new pair: any raw with quota (gated needs a tool)
                    has_tool = any(inv[k] > 0 and tier[k] >= 2 for k in range(n))
                    opts = [r for r in raws if quota[r] > 0 and (gate[r] == 0 or has_tool)]
                    if not opts:
                        break                                    # episode exhausted
                    act = ("collect", opts[int(rng.integers(len(opts)))])
                steps += 1; ep += 1
                if act[0] == "collect":
                    s = act[1]; inv[s] += 1; quota[s] -= 1
                else:
                    i, j = min(act[1], act[2]), max(act[1], act[2])
                    p = po[i, j]
                    if p == BOTCH:
                        inv[i] -= 1; inv[j] -= 1; dead.add((i, j))
                    elif p == INERT:
                        known[(i, j)] = INERT
                    else:
                        inv[i] -= 1; inv[j] -= 1; inv[p] += 1; known[(i, j)] = int(p)
                        if p == goal:
                            got = True
        costs.append(steps if got else 576)
    return int(st.median(costs)), sum(c >= 576 for c in costs) / n_mc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json", default="craft_v6_out/v61_mc.json")
    ap.add_argument("--sweep", action="store_true", help="also run the symbolic sweep cost (slow)")
    a = ap.parse_args()
    laws = sample_laws(LAWS_SEED)
    tc = true_cells(laws)
    print(f"LAWS {LAWS_SEED}: K={laws['K']} true bond cells {laws['n_true']}/91 | partners per element "
          f"{sorted(int(x) for x in laws['Bond'].sum(1))} | sampled in {laws['attempts']} attempts")

    # ---- roster statistics over random seeds ----------------------------------------------------
    per_tier = {t: [] for t in range(1, T_MAX + 1)}
    drops, ok, adm = [], 0, {t: [] for t in range(2, T_MAX + 1)}
    exposed, steps4, rawuse4 = [], [], []
    for s in range(20_000, 20_000 + a.n):
        sp = make_world(s, laws)
        for t in per_tier:
            per_tier[t].append(sum(1 for i in range(sp["n_items"]) if sp["tier"][i] == t and not sp["decoy"][i]))
        drops.append(len(sp["dropped"]))
        ok += sp["world_ok"]
        for t in adm:
            adm[t].append(len(sp["admitted"][t]))
        exposed.append(len(exposed_cells(sp) & tc) / len(tc))
        if sp["admitted"][T_MAX]:
            g = sp["admitted"][T_MAX][0]
            p = sp["plans"][g]
            steps4.append(p["steps"]); rawuse4.append(max(p["units_by_type"].values()))
    print(f"\nROSTER over {a.n} seeds: real items per tier " +
          " / ".join(f"{st.mean(per_tier[t]):.2f}" for t in per_tier) +
          f" | products dropped {st.mean(drops):.2f}/world | P(world admitted) {ok / a.n:.2f}")
    print("  admitted goals per tier (mean, P>=1): " +
          " | ".join(f"t{t}: {st.mean(adm[t]):.2f}, {sum(x >= 1 for x in adm[t]) / a.n:.2f}" for t in adm))
    print(f"  bond cells EXPOSED per world: {st.mean(exposed):.3f} (sd {st.pstdev(exposed):.3f}) of {len(tc)}")
    print(f"  tier-4 goal (lowest index): planner macro-steps median {st.median(steps4)} "
          f"(min {min(steps4)}, max {max(steps4)}) | max raw units of one type median {st.median(rawuse4)}")

    # ---- coverage of random 4-world lineages -----------------------------------------------------
    rng = np.random.default_rng(0)
    cov = {k: [] for k in range(1, 5)}
    for _ in range(100):
        seeds = rng.integers(30_000, 60_000, size=12)
        worlds = [make_world(int(s), laws) for s in seeds]
        worlds = [w for w in worlds if w["world_ok"]][:4]
        if len(worlds) < 4:
            continue
        u = set()
        for k, w in enumerate(worlds, 1):
            u |= exposed_cells(w) & tc
            cov[k].append(len(u) / len(tc))
    print("  cumulative coverage of random admitted 4-world lineages: " +
          " / ".join(f"{st.mean(cov[k]):.2f}" for k in cov) + f"  (sd {st.pstdev(cov[4]):.2f} at 4)")

    # ---- the named windows -----------------------------------------------------------------------
    print("\nADMISSION over the named windows (mechanical; [n admitted goals t2/t3/t4], stream, planner steps):")
    res = dict(laws_seed=LAWS_SEED, n_true=laws["n_true"], windows={})
    for name, seeds in WINDOWS.items():
        rows = []
        for s in seeds:
            sp = make_world(s, laws)
            row = dict(seed=s, ok=bool(sp["world_ok"]),
                       n_adm=[len(sp["admitted"][t]) for t in range(2, T_MAX + 1)],
                       exposed=round(len(exposed_cells(sp) & tc) / len(tc), 3))
            if sp["world_ok"]:
                g = goal_stream(sp)
                row["stream"] = g
                row["steps"] = [sp["plans"][x]["steps"] for x in g]
                row["raw_use"] = [max(sp["plans"][x]["units_by_type"].values()) for x in g]
                row["tool_needed"] = [sp["plans"][x]["tool"] is not None for x in g]
                if a.sweep:
                    rr = np.random.default_rng(s)
                    row["sweep"] = [sweep_cost(sp, x, rr) for x in g]
            rows.append(row)
        res["windows"][name] = rows
        line = " ".join(f"{r['seed']}{'' if r['ok'] else 'x'}[{'/'.join(map(str, r['n_adm']))}]" for r in rows)
        print(f"  {name:6s} {line}")
        for r in rows:
            if r["ok"]:
                extra = f" sweep {r['sweep']}" if a.sweep else ""
                print(f"         {r['seed']}: stream {r['stream']} steps {r['steps']} raw_use {r['raw_use']} "
                      f"tool {r['tool_needed']} exposed {r['exposed']}{extra}")
    # lineage coverage for the named pretrain windows
    for name in ("pre0", "pre1", "pre2"):
        seeds = [r["seed"] for r in res["windows"][name] if r["ok"]][:4]
        u, cum = set(), []
        for s in seeds:
            u |= exposed_cells(make_world(s, laws)) & tc
            cum.append(round(len(u) / len(tc), 2))
        res["windows"][name + "_lineage"] = dict(seeds=seeds, coverage=cum)
        print(f"  lineage {name}: worlds {seeds} cumulative coverage {cum}")
    json.dump(res, open(a.json, "w"), indent=1, default=int)
    print(f"\nwritten {a.json}")


if __name__ == "__main__":
    main()
