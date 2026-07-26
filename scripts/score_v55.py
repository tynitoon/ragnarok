"""v55 scorer — the FROZEN analysis spec, committed BEFORE the confirmatory arms ran.

No choices remain at analysis time: this file mechanically evaluates the validity preconditions, P1-P6,
the KILL criteria and the frozen verdict rule, and prints every mandatory report field whether the news
is good or bad. v53 and v54 both computed their criteria in code — that is what keeps a team that has
retracted 6+ overclaims honest.

Usage: python -m scripts.score_v55 [--out-dir craft_v6_out]
"""

import argparse
import json
import os


def spearman(xs, ys):
    """Spearman rho with average ranks for ties (no scipy dependency)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load(out_dir):
    seeds = {}
    for s in (0, 1, 2):
        p = os.path.join(out_dir, f"v55_s{s}.json")
        if os.path.exists(p):
            seeds[s] = json.load(open(p))
    return seeds


def validity(d):
    """V1-V5. Returns (ok, [failures])."""
    f = []
    goals = [g for g, _, _ in d.get("admitted", [])]
    for t in ("A", "B", "D", "C"):
        arm = d.get("arms", {}).get(t)
        if not arm or not arm.get("complete"):
            f.append(f"V1 arm {t} incomplete/missing")
        elif [r["goal"] for r in arm["rows"]] != goals:
            f.append(f"V1 arm {t} goal list != admitted stream")
    nec = d.get("necessary", [])
    if nec != [g for g, _, b in d.get("admitted", []) if b > 26]:
        f.append("V2 stratum does not match blind > 26")
    if len(nec) < 4:
        f.append(f"V2 only {len(nec)} GOAL-NECESSARY goals (need >=4)")
    if not d.get("nav") or min(d["nav"].values()) < 0.85:
        f.append("V3 nav gate below 0.85")
    sw = d.get("swap", {})
    if not all(k in sw for k in ("A", "A_own", "D", "D_own")):
        f.append("V4 swap missing A/A_own/D/D_own")
    if "E" not in d.get("arms", {}) or "goal" not in d["arms"].get("E", {}):
        f.append("V5 arm E missing")
    return (not f), f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="craft_v6_out")
    a = ap.parse_args()
    seeds = load(a.out_dir)
    if not seeds:
        print("no v55_s*.json found"); return

    gp = os.path.join(a.out_dir, "v55_gate_s0.json")
    print("=" * 100)
    print("v55 — HIDDEN-RECIPE PERSISTENT WORLD | frozen analysis spec (score_v55.py, committed pre-run)")
    print("=" * 100)
    if os.path.exists(gp):
        g = json.load(open(gp))["gate"]
        print(f"KILL-0 gate: world 3002 goal {g['goal']} master {g['master']} in {g['rounds']} rounds "
              f"-> {'PROCEED' if g['proceed'] else 'KILL-1 FIRES'}")
    else:
        print("KILL-0 gate: world 3002, easiest goal 6 (pc 3), master 1.0 in 2 rounds -> PROCEED "
              "(recovered verbatim from craft_v6_out/v55_gate.log after the writer crashed)")

    valid, P = {}, {}
    for s, d in sorted(seeds.items()):
        ok, fails = validity(d)
        valid[s] = ok
        pc = {g: p for g, p, _ in d["admitted"]}
        bl = {g: b for g, _, b in d["admitted"]}
        NEC = d["necessary"]; OPT = [g for g, _, _ in d["admitted"] if g not in NEC]
        rows = lambda t: d["arms"].get(t, {}).get("rows", [])                              # noqa: E731
        mrate = lambda t, S: (sum(r["mastered"] for r in rows(t) if r["goal"] in S)         # noqa: E731
                              / max(1, len([r for r in rows(t) if r["goal"] in S])))
        mcnt = lambda t, S: sum(r["mastered"] for r in rows(t) if r["goal"] in S)           # noqa: E731

        print(f"\n{'-'*100}\nSEED {s} | world {d['world']} | {'VALID' if ok else 'INVALID: ' + '; '.join(fails)}"
              f"{' | RESUMED — reproducibility and P5 downgraded' if d.get('resumed') else ''}")
        print(f"  nav gate (40-step): {d['nav']}")
        print(f"  goals (pc/blind, * = GOAL-NECESSARY): " +
              ", ".join(f"{g}{'*' if g in NEC else ''}({pc[g]}/{bl[g]:.0f})" for g, _, _ in d["admitted"]))
        for t in ("A", "B", "D", "C"):
            if rows(t):
                print(f"   {t}: " + " ".join(
                    f"{r['goal']}{'*' if r['necessary'] else ''}:{'M' if r['mastered'] else 'x'}"
                    f"{r['rounds']}r" for r in rows(t)) +
                    f"  | NEC master {mrate(t,NEC):.2f} ({mcnt(t,NEC)}/{len(NEC)})"
                    f" | OPT {mrate(t,OPT):.2f}")
        E = d["arms"].get("E", {})
        sw = d.get("swap", {})
        hardest = E.get("goal")
        aH = next((r for r in rows("A") if r["goal"] == hardest), None)
        p = dict(
            P1=(mrate("A", NEC) >= 0.70 and mrate("B", NEC) <= 0.30),
            P2a=(sw.get("A", -9) >= 0.30),
            P2=(sw.get("A", -9) >= 0.30 and sw.get("D", 9) < 0.10),
            P3=(abs(mrate("A", OPT) - mrate("D", OPT)) < 0.15),
            P4=(mcnt("A", NEC) - mcnt("C", NEC) >= 2),
            C3=(mcnt("A", NEC) >= mcnt("D", NEC)),
            P6=(bool(aH and aH["mastered"]) and not E.get("mastered", True)),
        )
        P[s] = p
        print(f"  E: goal {hardest} r_max {E.get('r_max')} -> master {E.get('master')} "
              f"({'M' if E.get('mastered') else 'x'})")
        print(f"  swap: S_A {sw.get('A')} (need>=0.30) | S_D {sw.get('D')} (need<0.10) | "
              f"mean own A {sum(sw.get('A_own',{0:0}).values())/max(1,len(sw.get('A_own',{1:1}))):.2f} "
              f"D {sum(sw.get('D_own',{0:0}).values())/max(1,len(sw.get('D_own',{1:1}))):.2f}")
        print(f"  predictions: " + "  ".join(f"{k}={'T' if v else 'F'}" for k, v in p.items()))

        # R7 pre-committed deflating explanation ---------------------------------------------------
        for t in ("A", "B"):
            rn = [r for r in rows(t) if r["necessary"]]
            att = sum(r.get("att", 0) for r in rn) or 1
            print(f"  R7 {t} on NEC: fail {sum(r.get('fail',0) for r in rn)/att:.2f} | "
                  f"repeat-already-succeeded {sum(r.get('repeat_prev_succ',0) for r in rn)/att:.2f} | "
                  f"first-try-correct {sum(r.get('first_try_ok',0) for r in rn)/att:.2f}")
        # R8 no-signal diagnosis --------------------------------------------------------------------
        for g in NEC:
            ra = next((r for r in rows("A") if r["goal"] == g), {})
            rb = next((r for r in rows("B") if r["goal"] == g), {})
            da = (ra.get("demos_per_round") or [0])[0]; db = (rb.get("demos_per_round") or [0])[0]
            tag = ("NO-SIGNAL (null here is a MEASUREMENT limit, not a memory result)"
                   if da + db == 0 else "")
            print(f"  R8 goal {g}: round-1 demos A {da} B {db} {tag}")
        cf = d["arms"].get("C_foreign_buffer", {})
        if cf and cf.get("ratio", 1) < 0.90:
            print(f"  R4 !! arm C under-provisioned (ratio {cf['ratio']}) — a P4 pass is confounded "
                  f"with data volume")
        if d["arms"].get("A", {}).get("buf_capped"):
            print("  R5 !! A's 2M buffer hit its cap — earliest goals were evicted; 'lifelong' means "
                  "a sliding window")
        print(f"  R9 collect vs eval primitives: " + ", ".join(
            f"{t} {d['arms'][t].get('collect_prim',0)/1e6:.0f}M/{d['arms'][t].get('eval_prim',0)/1e6:.0f}M"
            for t in ("A", "B", "D", "C") if t in d["arms"]))

    # ---- aggregation ------------------------------------------------------------------------------
    V = [s for s in seeds if valid[s]]
    print(f"\n{'='*100}\nVALID SEEDS: {V}")
    if len(V) < 3:
        print("VERDICT: NULL-UNDECIDABLE — fewer than 3 valid seeds; no 3/3 prediction may be declared.")
        return
    agg = {k: all(P[s][k] for s in V) for k in P[V[0]]}
    n_ok = {k: sum(P[s][k] for s in V) for k in P[V[0]]}
    P6ok = n_ok["P6"] >= 2

    xs, ys = [], []
    for s in V:
        for k, r in enumerate(seeds[s]["arms"]["A"]["rows"]):
            xs.append(k); ys.append(r["rounds"])
    rho = spearman(xs, ys)
    print(f"P5 (pooled Spearman rho of arm-A rounds vs presentation index): {rho:+.3f} (n={len(xs)}) "
          f"-> {'TRUE' if rho < -0.40 else 'FALSE'}")
    print("  NOTE: the stream is presented in ASCENDING difficulty, so a negative P5 is conservative; "
          "a non-negative P5 is NOT evidence against compounding.")
    for k in ("P1", "P2a", "P2", "P3", "P4", "C3"):
        print(f"{k}: {n_ok[k]}/{len(V)} seeds -> {'TRUE' if agg[k] else 'FALSE'}")
    print(f"P6: {n_ok['P6']}/{len(V)} seeds (needs >=2) -> {'TRUE' if P6ok else 'FALSE'}")

    kill1 = all(seeds[s].get("swap", {}).get("A", 9) < 0.10 for s in V)
    kill2 = sum(1 for s in V
                if sum(r["mastered"] for r in seeds[s]["arms"]["A"]["rows"] if r["necessary"])
                <= sum(r["mastered"] for r in seeds[s]["arms"]["B"]["rows"] if r["necessary"])) >= 2
    if kill1:
        print("\n!! KILL-1 FIRES: S_A < 0.10 on 3/3 — goal-directedness unreachable on this substrate.")
    if kill2:
        print("\n!! KILL-2 FIRES: A's GOAL-NECESSARY master count <= B's on >=2/3 — memory buys nothing "
              "on the right side of our own boundary.")

    if agg["P1"] and agg["P2a"] and agg["C3"] and agg["P4"] and P6ok:
        v = "POSITIVE"
    elif agg["P2"]:
        v = "PARTIAL-MECHANISM"
    elif agg["P1"] and not agg["P2a"]:
        v = "PARTIAL-ENABLEMENT"
    else:
        v = "NULL"
    print(f"\n{'='*100}\nFROZEN VERDICT: {v}\n{'='*100}")

    ceiling = {
        "POSITIVE": ('In one persistent world with HIDDEN recipes, a composer that accumulates its own '
                     'experience masters GOAL-NECESSARY goals that an amnesic composer with identical '
                     'observations cannot, and its behaviour is goal-dependent (S_A >= 0.30). '
                     'NOT transfer, NOT cross-world, NOT developmental compounding.'),
        "PARTIAL-MECHANISM": ('The composer became genuinely goal-directed (a real new finding that flips '
                              'the v54 goal-agnostic result) WITHOUT the enablement result.'),
        "PARTIAL-ENABLEMENT": ('Discovered-and-remembered recipes enable otherwise-unreachable goals via a '
                               'STILL-GOAL-BLIND policy. Nothing stronger may be claimed.'),
        "NULL": 'No claim. Report the numbers and the kill criteria.'}[v]
    print(f"CLAIM CEILING: {ceiling}")
    if v == "POSITIVE" and all(
            sum(r["mastered"] for r in seeds[s]["arms"]["A"]["rows"] if r["necessary"])
            == sum(r["mastered"] for r in seeds[s]["arms"]["D"]["rows"] if r["necessary"]) for s in V):
        print("  A merely TIES the oracle arm D: maximum claim is 'discovered-and-remembered knowledge "
              "recovers exactly the value of the granted DAG'.")
    for s in V:
        own = seeds[s].get("swap", {}).get("A_own", {})
        if own and sum(own.values()) / len(own) < 0.30:
            print(f"  R3 seed {s}: S_A was COMPETENCE-bounded (S <= mean own[Y] = "
                  f"{sum(own.values())/len(own):.2f}); a low S_A here cannot be read as "
                  f"'the composer is a goal-blind reflex'.")
    print("\nR10 FROZEN CAVEATS (binding regardless of outcome): within-world only (per-item embeddings "
          "are world-specific; cross-world transfer OUT OF SCOPE); n=3 worlds, so seed variance mixes "
          "init and world variance and is not a CI over worlds; the shared frozen nav skill is a common "
          "failure axis capping every arm; the GOAL-NECESSARY boundary is CALIBRATED via a blind "
          "reference policy, not proven, and does not bind a goal-CONDITIONED learner; resources remain "
          "infinite and the DAG acyclic, so this is about remembering RULES, never search or "
          "irreversibility; right-censoring at R_max=10 makes cost differences lower bounds; arm D has "
          "strictly MORE information than A; M6's gamma=0.7 discards long-lag credit, so any real effect "
          "is expected on the SHALLOW GOAL-NECESSARY goals; v49_depth_scaling.json is NOT cited (its own "
          "verdict string contradicts its single row).")


if __name__ == "__main__":
    main()
