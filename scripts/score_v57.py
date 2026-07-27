"""v57 scorer — the FROZEN decision rule, committed BEFORE any confirmatory arm ran.

Mechanically evaluates P1-P5, the kill criteria and the verdict on craft_v6_out/v57_s{0,1,2}.json.
No choices remain at analysis time. Prints every mandatory field whether the news is good or bad, and
refuses to emit a verdict on incomplete data.

Usage: python -m scripts.score_v57
"""

import argparse
import json
import os

DEEP_PC = 10          # frozen stratum boundary
STARVED_DEMOS = 50    # frozen: a deep cell where K's TOTAL demos over its budget is <= this
THRESH = 0.60


def load(out_dir):
    d = {}
    for s in (0, 1, 2):
        p = os.path.join(out_dir, f"v57_s{s}.json")
        if os.path.exists(p):
            d[s] = json.load(open(p))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="craft_v6_out")
    a = ap.parse_args()
    seeds = load(a.out_dir)
    print("=" * 104)
    print("v57 — HONEST CREDIT (train only on what the agent was asked to do)")
    print("frozen analysis spec, committed before any confirmatory arm | scripts/score_v57.py")
    print("=" * 104)
    if not seeds:
        print("no v57_s*.json found"); return

    P, valid = {}, {}
    for s, d in sorted(seeds.items()):
        rowsA = {r["goal"]: r for r in d.get("arms", {}).get("A", {}).get("rows", [])}
        rowsK = {r["goal"]: r for r in d.get("arms", {}).get("K", {}).get("rows", [])}
        U = d.get("arms", {}).get("U", [])
        pc = {g: c for g, c, _ in d["admitted"]}
        goals = [g for g, _, _ in d["admitted"]]
        DEEP = [g for g in goals if pc[g] >= DEEP_PC]
        SHAL = [g for g in goals if pc[g] < DEEP_PC]
        ok = (d.get("arms", {}).get("A", {}).get("complete") and
              d.get("arms", {}).get("K", {}).get("complete") and
              set(rowsA) == set(goals) and set(rowsK) == set(goals))
        valid[s] = bool(ok)

        print(f"\n{'-'*104}\nWORLD {d['world']} (seed {s}) | {'COMPLETE' if ok else 'INCOMPLETE'}"
              f" | {len(SHAL)} shallow, {len(DEEP)} deep (pc>={DEEP_PC})")
        print(f"  {'goal':>5} {'pc':>3} | {'A master':>9} {'A rnds':>7} {'A demos':>9} | "
              f"{'K master':>9} {'K rnds':>7} {'K demos':>9} | starved")
        for g in goals:
            ra, rk = rowsA.get(g, {}), rowsK.get(g, {})
            da = sum(ra.get("demos_per_round", []) or [0])
            dk = sum(rk.get("demos_per_round", []) or [0])
            st = "STARVED" if (g in DEEP and dk <= STARVED_DEMOS) else ""
            print(f"  {g:>5} {pc[g]:>3} | {ra.get('master',0):>9.2f} {ra.get('rounds',0):>7} {da:>9} | "
                  f"{rk.get('master',0):>9.2f} {rk.get('rounds',0):>7} {dk:>9} | {st}")

        mA_sh = sum(rowsA.get(g, {}).get("mastered", False) for g in SHAL) / max(1, len(SHAL))
        starved = [g for g in DEEP if sum(rowsK.get(g, {}).get("demos_per_round", []) or [0])
                   <= STARVED_DEMOS]
        p2 = len(starved) >= (2 / 3) * len(DEEP) if DEEP else False
        mA_st = (sum(rowsA.get(g, {}).get("mastered", False) for g in starved) / len(starved)
                 if starved else 0.0)
        p3 = mA_st >= 0.50
        mA_deep = sum(rowsA.get(g, {}).get("mastered", False) for g in DEEP) / max(1, len(DEEP))
        P[s] = dict(P1=mA_sh >= 0.80, P2=p2, P3=p3, mA_sh=mA_sh, mA_st=mA_st, mA_deep=mA_deep,
                    n_deep=len(DEEP), n_starved=len(starved), U=U)
        print(f"  P1 shallow mastery A {mA_sh:.2f} (need >=0.80) -> {P[s]['P1']}")
        print(f"  P2 K starved on {len(starved)}/{len(DEEP)} deep cells (need >=2/3) -> {p2}")
        print(f"  P3 A masters {mA_st:.2f} of the starved cells (need >=0.50) -> {p3}")
        for u in U:
            print(f"  U  goal {u['goal']} at r_max {u.get('r_max')}: master {u.get('master',0):.2f} "
                  f"({'M' if u.get('mastered') else 'x'}) | total demos "
                  f"{sum(u.get('demos_per_round', []) or [0])}")
        # P5 pre-committed deflation check
        for tag, rows in (("A", rowsA), ("K", rowsK)):
            rn = [rows[g] for g in DEEP if g in rows]
            att = sum(r.get("att", 0) for r in rn) or 1
            print(f"  P5 {tag} on DEEP: first-try-correct "
                  f"{sum(r.get('first_try_ok',0) for r in rn)/att:.3f} | repeats "
                  f"{sum(r.get('repeat_prev_succ',0) for r in rn)/att:.3f} | fail "
                  f"{sum(r.get('fail',0) for r in rn)/att:.3f}")

    V = [s for s in seeds if valid[s]]
    print(f"\n{'='*104}\nCOMPLETE WORLDS: {V}")
    if len(V) < 3:
        print("VERDICT: NULL-UNDECIDABLE — fewer than 3 complete worlds; no 3/3 prediction may be declared.")
        return

    p1 = all(P[s]["P1"] for s in V)
    p2 = all(P[s]["P2"] for s in V)
    p3 = all(P[s]["P3"] for s in V)
    allU = [u for s in V for u in P[s]["U"]]
    nU = len(allU) or 1
    p4 = (sum(u.get("mastered", False) for u in allU) / nU) <= (1 / 3)
    print(f"P1 replication gate : {sum(P[s]['P1'] for s in V)}/3 -> {p1}")
    print(f"P2 starvation       : {sum(P[s]['P2'] for s in V)}/3 -> {p2}")
    print(f"P3 contrast         : {sum(P[s]['P3'] for s in V)}/3 -> {p3}")
    print(f"P4 not-just-budget  : arm U mastered {sum(u.get('mastered',False) for u in allU)}/{nU} "
          f"(need <=1/3) -> {p4}")

    if not p1:
        print("\n!! KILL-1 FIRES: the credit fix breaks the AGENT, not the control. VOID. Line ends.")
        verdict = "VOID"
    else:
        if sum(1 for s in V if P[s]["mA_deep"] < 1 / 3) >= 2:
            print("\n!! KILL-2 FIRES: under honest credit arm A masters <1/3 of deep goals on >=2/3 "
                  "worlds — nobody learns at depth here. Line ends.")
        if p1 and p2 and p3 and p4:
            verdict = "POSITIVE"
        elif p1 and p2 and p3:
            verdict = "PARTIAL-AMORTISATION"
            print("\n!! KILL-3 FIRES: arm U closes the gap — memory buys a finite factor, not "
                  "possibility. Line ends; publish next to v55's NULL.")
        else:
            verdict = "NULL"

    print(f"\n{'='*104}\nFROZEN VERDICT: {verdict}\n{'='*104}")
    ceiling = {
        "POSITIVE": ("In a persistent world whose recipes must be discovered by failing, an agent that "
                     "accumulated its own discovered knowledge masters deep goals for which a genuinely "
                     "knowledge-free agent — same observations, same credit rule — receives essentially "
                     "NO learning signal at all, and 6x the budget does not close the gap. NOT "
                     "'impossible' (absence of signal AT OUR BUDGET), NOT cross-world, NOT a causal claim "
                     "about chain length."),
        "PARTIAL-AMORTISATION": ("Memory buys a large but FINITE speed-up; the from-scratch path exists "
                                 "at >=6x cost. Nothing stronger."),
        "NULL": "No claim. Report the numbers and the kill criteria.",
        "VOID": "No claim in either direction. The instrument, not the hypothesis, failed."}[verdict]
    print(f"CLAIM CEILING: {ceiling}")
    print("\nBINDING CAVEATS: n=3 worlds (seed variance mixes init and world variance, not a CI over "
          "worlds); the pc=10 stratum boundary is calibrated, not proven; pc is collinear with max item "
          "multiplicity (r=0.979) so depth and multiplicity are NOT separable and no causal claim about "
          "chain length is licensed; the shared frozen nav skill caps every arm; the composer sees "
          "(inv>0) not counts and mastery is known to collapse at multiplicity>=4; right-censoring at "
          "R_max makes any cost difference a lower bound (mastery and demo count are primary); resources "
          "are infinite and the DAG acyclic, so literal NECESSITY remains false by construction on a "
          "world that resets every episode; and the credit fix costs every arm — in the gate, arm A lost "
          "a deep goal it had mastered under the leaky rule.")
    print("\nWhatever this verdict, v57 is the LAST experiment in this arc. No bigger world, no longer "
          "chain, no new substrate. (Pre-committed in the v57 prereg.)")


if __name__ == "__main__":
    main()
