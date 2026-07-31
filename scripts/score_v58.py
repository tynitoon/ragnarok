"""ARC 2 — the FROZEN scorer. Committed BEFORE any confirmatory arm runs.

Mechanically evaluates P1-P6, the kill criteria and the verdict on craft_v6_out/v58_test_*.json.
No choices remain at analysis time. It refuses to emit a verdict on incomplete data.

STATISTICS ARE THE REPAIRED ONES (ARC2_PLAN sections 8-9). The v1 design was VOIDED by the verification
audit for a fatal censoring asymmetry and a threshold fitted on the wrong unit, so:
  - cost is the SCORED cost (run_goal_v58 `cost`): 0 if the goal was mastered on arrival, rounds*192
    if mastered, censor_cap*192 if censored — identical rule for every arm. Raw `attempts` is still
    recorded for the spend ledger but is NOT the outcome variable.
  - P2 is a CONTINUOUS paired statistic: the mean over paired goals of (M.discovery.frac -
    F.discovery.frac), the fraction of envs reaching the goal at first exposure. Win rates are dead
    (26% of pairs tied at the ceiling in calibration v1) and `min_step` is banned (saturates low).
  - thresholds are fitted on the POOLED 3-world unit the scorer actually uses, never a single world.

THRESHOLDS are read off calibration v2's measured null (scripts/calibrate2_v58.py ->
craft_v6_out/v58_calibration2.json). Two independent arm-F runs differ only by initialisation and
sampling noise, so the spread of their pooled pairwise ratios IS what "no transfer whatsoever" looks
like; the thresholds sit beyond its tail. This is the repair for the defect that decided v54, v55 and
v57 — all three lost to thresholds chosen from a pencil rather than from data.

Usage: python -m scripts.score_v58
"""

import argparse
import glob
import json
import os

# ---------------------------------------------------------------- FITTED AT CALIBRATION v2
# Filled verbatim from craft_v6_out/v58_calibration2.json -> "fitted", in the SAME commit as the
# amended prereg, before any confirmatory arm. Never edited afterwards.
P1_MAX_RATIO = None        # pooled M/F SCORED-cost ratio must be <= this
REFUTE_RATIO = None        # at or above this, M is indistinguishable from a second fresh run
P2_MIN_DFRAC = None        # mean paired (M - F) discovery.frac must be >= this
CALIBRATION_STAMP = None

# ---------------------------------------------------------------- STRUCTURAL (design, not data)
P3_FACTOR = 2.0            # saving over F must be >= this x the saving a degenerate pretrain buys
P4_MIN_Z_RATIO = 1.5       # zeroing the store must cost the treatment at least this factor
P6_MAX_LOSS_FRAC = 0.25    # M may be worse than F on at most this fraction of paired goals

DEFAULT_WORLDS = [6000, 6001, 6002]
SHIFTED_WORLDS = [7000, 7001]
NEED_ARMS = ("M", "F", "Mdeg", "Z")


def _cost(rows):
    """SCORED cost — see the module docstring. Falls back loudly if a run predates the repair."""
    if any("cost" not in r for r in rows):
        raise SystemExit("REFUSING TO SCORE: a result row has no 'cost' field — it was produced by the "
                         "VOIDED pre-repair pipeline (ARC2_PLAN section 8). Re-run it.")
    return sum(r["cost"] for r in rows)


def _dfrac(rows_a, rows_b):
    """P2: mean paired difference of first-exposure discovery fraction (a - b), and the pair count.
    Goals where either side lacks a discovery record are skipped."""
    ds = []
    for ra, rb in zip(rows_a, rows_b):
        fa = (ra.get("discovery") or {}).get("frac")
        fb = (rb.get("discovery") or {}).get("frac")
        if fa is not None and fb is not None:
            ds.append(fa - fb)
    return (sum(ds) / len(ds) if ds else 0.0), len(ds)


def _worse(rows_a, rows_b):
    """P6: goals where a costs strictly more than b, and the pair count."""
    n = sum(1 for _ in zip(rows_a, rows_b))
    return sum(1 for ra, rb in zip(rows_a, rows_b) if ra["cost"] > rb["cost"]), n


def load(out_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "v58_test_*.json"))):
        d = json.load(open(p))
        out[d["world"]] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="craft_v6_out")
    a = ap.parse_args()

    print("=" * 104)
    print("ARC 2 — EvidenceNet: change world, keep skills | frozen scorer, committed before any arm ran")
    print("=" * 104)
    if P1_MAX_RATIO is None or REFUTE_RATIO is None or P2_MIN_DFRAC is None:
        print("REFUSING TO SCORE: thresholds are not fitted. Run scripts/calibrate2_v58.py, copy its")
        print("fitted values here, and commit them with the amended prereg BEFORE any confirmatory arm.")
        print("(The v1 values 0.712/0.750 were VOIDED — wrong unit and a fatal censoring asymmetry;")
        print(" see ARC2_PLAN sections 8-9.)")
        return
    print(f"thresholds (fitted on the POOLED unit, calibration {CALIBRATION_STAMP}):")
    print(f"  P1_MAX_RATIO {P1_MAX_RATIO} | REFUTE_RATIO {REFUTE_RATIO} | P2_MIN_DFRAC {P2_MIN_DFRAC}")

    worlds = load(a.out_dir)
    if not worlds:
        print("\nno craft_v6_out/v58_test_*.json found"); return

    valid = {}
    for w, d in sorted(worlds.items()):
        miss = [t for t in NEED_ARMS if not d.get("arms", {}).get(t, {}).get("complete")]
        nav_ok = bool(d.get("nav")) and min(d["nav"].values()) >= 0.85
        valid[w] = (not miss) and nav_ok
        print(f"\n{'-'*104}\nWORLD {w} ({'default' if w in DEFAULT_WORLDS else 'param-shifted'}) | "
              f"{'VALID' if valid[w] else 'INVALID'}"
              f"{' | incomplete: ' + ','.join(miss) if miss else ''}"
              f"{'' if nav_ok else ' | nav gate below 0.85'}")
        if not valid[w]:
            continue
        rows = {t: d["arms"][t]["rows"] for t in NEED_ARMS}
        cM, cF, cMd, cZ = (_cost(rows["M"]), _cost(rows["F"]),
                           _cost(rows["Mdeg"]), _cost(rows["Z"]))
        df, npair = _dfrac(rows["M"], rows["F"])
        wo, nw = _worse(rows["M"], rows["F"])
        arr = {t: sum(r.get("mastered_on_arrival", False) for r in rows[t]) for t in ("M", "F")}
        cen = {t: sum(1 for r in rows[t] if not r["mastered"]) for t in ("M", "F")}
        print(f"  scored cost: M {cM} | F {cF} | Mdeg {cMd} | Z {cZ}   -> M/F {cM/max(1,cF):.3f}, "
              f"Z/M {cZ/max(1,cM):.3f}")
        print(f"  mastered on arrival: M {arr['M']}/{len(rows['M'])}, F {arr['F']}/{len(rows['F'])} | "
              f"censored: M {cen['M']}, F {cen['F']}")
        print(f"  discovery.frac paired (M-F): {df:+.4f} over {npair} goals | "
              f"M costlier on {wo}/{nw}")
        for t in ("G", "F6"):
            if d.get("arms", {}).get(t):
                print(f"  {t}: {json.dumps(d['arms'][t])[:150]}")

    V = [w for w in worlds if valid[w]]
    Vd = [w for w in V if w in DEFAULT_WORLDS]
    Vs = [w for w in V if w in SHIFTED_WORLDS]
    print(f"\n{'='*104}\nVALID WORLDS: default {Vd} | param-shifted {Vs}")
    if len(Vd) < 3:
        print("VERDICT: NULL-UNDECIDABLE — fewer than 3 valid default held-out worlds.")
        return

    pool = lambda t: sum(_cost(worlds[w]["arms"][t]["rows"]) for w in Vd)      # noqa: E731
    cM, cF, cMd, cZ = pool("M"), pool("F"), pool("Mdeg"), pool("Z")
    ratio = cM / max(1, cF)
    dsum = dn = wo = nw = 0
    for w in Vd:
        d_, n_ = _dfrac(worlds[w]["arms"]["M"]["rows"], worlds[w]["arms"]["F"]["rows"])
        dsum += d_ * n_; dn += n_
        a_, b_ = _worse(worlds[w]["arms"]["M"]["rows"], worlds[w]["arms"]["F"]["rows"])
        wo += a_; nw += b_
    dfrac = dsum / max(1, dn)

    p1 = ratio <= P1_MAX_RATIO
    p2 = dfrac >= P2_MIN_DFRAC
    p3 = (cF - cM) >= P3_FACTOR * max(0, cF - cMd)
    p4 = (cZ / max(1, cM)) >= P4_MIN_Z_RATIO
    p6 = (wo / max(1, nw)) <= P6_MAX_LOSS_FRAC
    p5 = None
    if Vs:
        sM = sum(_cost(worlds[w]["arms"]["M"]["rows"]) for w in Vs)
        sF = sum(_cost(worlds[w]["arms"]["F"]["rows"]) for w in Vs)
        p5_ratio = sM / max(1, sF)
        p5_thr = (P1_MAX_RATIO + 1.0) / 2.0
        p5 = p5_ratio <= p5_thr
        print(f"P5 param-shifted M/F   {p5_ratio:.3f} (need <= {p5_thr:.3f}) -> {p5}")

    print(f"P1 pooled M/F          {ratio:.3f} (need <= {P1_MAX_RATIO}) -> {p1}")
    print(f"P2 paired dFRAC (M-F)  {dfrac:+.4f} over {dn} goals (need >= {P2_MIN_DFRAC}) -> {p2}")
    print(f"P3 saving vs degenerate (F-M) {cF-cM} vs {P3_FACTOR}x(F-Mdeg) "
          f"{P3_FACTOR*max(0,cF-cMd):.0f} -> {p3}")
    print(f"P4 store-read Z/M      {cZ/max(1,cM):.3f} (need >= {P4_MIN_Z_RATIO}) -> {p4}")
    print(f"P6 M costlier on       {wo}/{nw} goals (allowed <= {P6_MAX_LOSS_FRAC:.0%}) -> {p6}")

    if p1 and p2 and p3 and p4:
        verdict = "SUPPORTED"
    elif ratio >= REFUTE_RATIO:
        verdict = "REFUTED"
    else:
        verdict = "NULL"

    print(f"\n{'='*104}\nFROZEN VERDICT: {verdict}\n{'='*104}")
    if verdict == "SUPPORTED":
        print("CLAIM, verbatim and nothing beyond it: 'Meta-trained across same-family hidden-recipe")
        print("worlds, a FROZEN-weight agent that writes its own per-world evidence store discovers and")
        print(f"masters a NEW world of that family {100*(1-ratio):.0f}% cheaper (scored cost, discovery")
        print("included) than an identical fresh agent — same architecture, same store, same credit")
        print("rule, equal in-world budget.'")
        print("NEVER: enablement, cross-family or cross-domain transfer, symbolic DAG induction,")
        print("zero-shot mastery. Disclosed grants: frozen nav skill, item->cell / item->craft-action /")
        print("is_resource / is_valid bindings, the one-of-each-input family invariant, and the")
        print("hand-designed store write schema (given identically to every arm).")
    elif verdict == "REFUTED":
        print("The treatment is indistinguishable from another fresh run. Published as a frozen NULL,")
        print("next to ARC 1's three. Arc closed.")
    else:
        print("Neither supported nor refuted at the frozen thresholds. No claim.")

    if not p3:
        print("\nDOWNGRADE: 'grammar/cheap transfer only — a meta-RL-101 re-demonstration, not")
        print("portability'. The degenerate-family pretrain bought most of the same saving.")
    if p5 is False:
        print("\nDOWNGRADE: 'generator-constants prior, family-memorization not procedure'. The saving")
        print("did not survive shifted generator constants.")
    if not p4:
        print("\nKILL K4 — 'store-ignoring collapse: the portability claim dies'. Reported even if P1")
        print("passed: if zeroing the store costs the treatment nothing, no world knowledge is being")
        print("carried and the weights are not portable skill.")
    if not p6:
        print(f"\nINTERFERENCE: M was costlier than F on {wo}/{nw} goals — above the "
              f"{P6_MAX_LOSS_FRAC:.0%} bound. Report the losses symmetrically; ARC 1 measured that")
        print("accumulation can hurt.")
    for w in Vd + Vs:
        f6 = worlds[w].get("arms", {}).get("F6")
        if f6:
            got = [c for c in (f6 if isinstance(f6, list) else [f6]) if c.get("mastered")]
            if got:
                print(f"\nENABLEMENT GUARD (world {w}): a fresh agent at 6x budget mastered "
                      f"{len(got)} cell(s) F could not. Nothing here may be described as enablement —")
                print("the gap is amortisation, not possibility.")

    print("\nBINDING CAVEATS: within-family only (one gen_tree generator); n=3 default held-out worlds,")
    print("so seed variance mixes init with world variance and is not a CI over worlds; the frozen nav")
    print("skill is a common failure axis capping every arm and is a DISCLOSED grant, so the claim is")
    print("composer-level only; the evidence store's write schema is hand-designed and given IDENTICALLY")
    print("to every arm, so the claim is 'a learned policy over self-gathered evidence', never 'the")
    print("agent invented the representation'; censored goals are capped in the SCORED cost, so a cost")
    print("difference understates a treatment that fails outright — censoring counts are reported")
    print("per world above and must be read alongside the ratio.")


if __name__ == "__main__":
    main()
