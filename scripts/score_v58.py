"""ARC 2 — the FROZEN scorer. Committed BEFORE any confirmatory arm runs.

Mechanically evaluates P1-P6, the kill criteria and the verdict on craft_v6_out/v58_test_*.json.
No choices remain at analysis time. It refuses to emit a verdict on incomplete data.

THRESHOLDS. P1_MAX_RATIO, P2_MIN_WINRATE and REFUTE_RATIO are not opinions — they are read off the
calibration's measured null distribution (scripts/calibrate_v58.py, craft_v6_out/v58_calibration.json).
Two independent arm-F runs on the same world differ only by initialisation and sampling noise, so the
spread of their pairwise ratios IS what "no transfer whatsoever" looks like on this statistic; the
thresholds sit beyond its tail. This is the repair for the defect that decided v54, v55 and v57 — all
three were lost to thresholds and strata chosen from a pencil rather than from data.

Usage: python -m scripts.score_v58
"""

import argparse
import glob
import json
import os

# ---------------------------------------------------------------- FITTED AT CALIBRATION (Step 4)
# Filled in from craft_v6_out/v58_calibration.json -> "fitted". Never edited afterwards.
P1_MAX_RATIO = 0.712       # FITTED: 0.95 x the lowest ratio the null ever produced (0.750)
REFUTE_RATIO = 0.750       # FITTED: at or above the null's minimum, M is indistinguishable from a
                           #         second fresh run (null median 1.002, max 1.333)
CALIBRATION_STAMP = "v58_calibration.json / 2 worlds x 3 inits / K2 1.14x / null min 0.750"

# P2 IS BLOCKED — NOT fitted, deliberately. Calibration proved the specified statistic
# (attempts-to-first-demo) has ZERO resolution: all 54 measured values were exactly 48, one episode,
# because with 256 parallel envs some env always reaches the goal in the first episode. Freezing it
# would put a permanently-false conjunct into the SUPPORTED rule and make success unreachable by
# construction. Redefining it AFTER seeing calibration data is the exact failure mode that cost v54,
# v55 and v57 their verdicts, so the replacement is an audit decision, not an implementer's. Evidence
# for that decision is in the handoff (ARC2_PLAN.md section 7). While this is None the scorer reports
# P1/P3/P4/P5/P6 but MAY NOT declare SUPPORTED.
P2_MIN_WINRATE = None

# ---------------------------------------------------------------- STRUCTURAL (design, not data)
P3_FACTOR = 2.0            # saving over F must be >= this x the saving a degenerate pretrain buys
P4_MIN_Z_RATIO = 1.5       # zeroing the store must cost the treatment at least this factor
P6_MAX_LOSS_FRAC = 0.25    # M may be worse than F on at most this fraction of paired goals
P2_MIN_WIN_LOSS = 2.0      # and must win at least this many times per loss
REFUTE_MIN_WIN_LOSS = 1.5  # below this win/loss ratio the result is refuted outright

DEFAULT_WORLDS = [6000, 6001, 6002]
SHIFTED_WORLDS = [7000, 7001]


def _cost(rows):
    return sum(r["attempts"] for r in rows)


def _paired(rows_a, rows_b):
    """Paired attempts-to-first-demo. Ties (neither side ever reached the goal) are excluded."""
    wins = losses = 0
    for ra, rb in zip(rows_a, rows_b):
        fa, fb = ra.get("first_demo_attempt"), rb.get("first_demo_attempt")
        if fa is None and fb is None:
            continue
        if fb is None:
            wins += 1
        elif fa is None:
            losses += 1
        elif fa < fb:
            wins += 1
        elif fb < fa:
            losses += 1
    return wins, losses


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
    if P1_MAX_RATIO is None or REFUTE_RATIO is None:
        print("REFUSING TO SCORE: thresholds were never fitted. Run scripts/calibrate_v58.py, copy its")
        print("fitted values into this file, and commit BOTH before any confirmatory arm runs.")
        return
    if P2_MIN_WINRATE is None:
        print("\n!! P2 IS BLOCKED (see the constant's comment): its specified statistic had zero")
        print("   resolution at calibration. P1/P3/P4/P5/P6 are reported below, but SUPPORTED cannot")
        print("   be declared until the audit resolves P2. A confirmatory run may still be scored for")
        print("   REFUTED, which needs only P1's fitted REFUTE_RATIO.")
    print(f"thresholds (fitted from the measured null, calibration {CALIBRATION_STAMP}):")
    print(f"  P1_MAX_RATIO {P1_MAX_RATIO} | P2_MIN_WINRATE {P2_MIN_WINRATE} | "
          f"REFUTE_RATIO {REFUTE_RATIO}")

    worlds = load(a.out_dir)
    if not worlds:
        print("\nno craft_v6_out/v58_test_*.json found"); return

    # ---- validity ---------------------------------------------------------------------------------
    need_arms = ("M", "F", "Mdeg", "Z")
    valid = {}
    for w, d in sorted(worlds.items()):
        miss = [t for t in need_arms if not d.get("arms", {}).get(t, {}).get("complete")]
        nav_ok = d.get("nav") and min(d["nav"].values()) >= 0.85
        valid[w] = (not miss) and nav_ok
        print(f"\n{'-'*104}\nWORLD {w} ({'default' if w in DEFAULT_WORLDS else 'param-shifted'}) | "
              f"{'VALID' if valid[w] else 'INVALID'}"
              f"{' | missing/incomplete: ' + ','.join(miss) if miss else ''}"
              f"{'' if nav_ok else ' | nav gate below 0.85'}")
        if not valid[w]:
            continue
        rows = {t: d["arms"][t]["rows"] for t in need_arms}
        cM, cF, cMd, cZ = (_cost(rows["M"]), _cost(rows["F"]),
                           _cost(rows["Mdeg"]), _cost(rows["Z"]))
        wins, losses = _paired(rows["M"], rows["F"])
        worse = sum(1 for rm, rf in zip(rows["M"], rows["F"]) if rm["attempts"] > rf["attempts"])
        print(f"  stream cost: M {cM} | F {cF} | Mdeg {cMd} | Z {cZ}   -> M/F {cM/max(1,cF):.3f}, "
              f"Z/M {cZ/max(1,cM):.3f}")
        print(f"  first-demo paired: M wins {wins}, loses {losses} "
              f"({wins/max(1,wins+losses):.2f} win rate) | M worse on {worse}/{len(rows['M'])} goals")
        for t in ("G", "F6"):
            if d.get("arms", {}).get(t):
                print(f"  {t}: {json.dumps(d['arms'][t])[:150]}")

    V = [w for w in worlds if valid[w]]
    Vd = [w for w in V if w in DEFAULT_WORLDS]
    print(f"\n{'='*104}\nVALID WORLDS: default {Vd} | param-shifted {[w for w in V if w in SHIFTED_WORLDS]}")
    if len(Vd) < 3:
        print("VERDICT: NULL-UNDECIDABLE — fewer than 3 valid default held-out worlds.")
        return

    # ---- pooled primaries over the DEFAULT held-out worlds ----------------------------------------
    pool = lambda t: sum(_cost(worlds[w]["arms"][t]["rows"]) for w in Vd)      # noqa: E731
    cM, cF, cMd, cZ = pool("M"), pool("F"), pool("Mdeg"), pool("Z")
    ratio = cM / max(1, cF)
    wins = losses = 0
    worse = total = 0
    for w in Vd:
        a_, b_ = _paired(worlds[w]["arms"]["M"]["rows"], worlds[w]["arms"]["F"]["rows"])
        wins += a_; losses += b_
        for rm, rf in zip(worlds[w]["arms"]["M"]["rows"], worlds[w]["arms"]["F"]["rows"]):
            total += 1
            worse += rm["attempts"] > rf["attempts"]
    wr = wins / max(1, wins + losses)
    wl = wins / max(1, losses)

    p1 = ratio <= P1_MAX_RATIO
    p2 = None if P2_MIN_WINRATE is None else ((wr >= P2_MIN_WINRATE) and (wl >= P2_MIN_WIN_LOSS))
    p3 = (cF - cM) >= P3_FACTOR * max(0, cF - cMd)
    p4 = (cZ / max(1, cM)) >= P4_MIN_Z_RATIO
    p6 = (worse / max(1, total)) <= P6_MAX_LOSS_FRAC
    Vs = [w for w in V if w in SHIFTED_WORLDS]
    p5 = None
    if Vs:
        sM = sum(_cost(worlds[w]["arms"]["M"]["rows"]) for w in Vs)
        sF = sum(_cost(worlds[w]["arms"]["F"]["rows"]) for w in Vs)
        p5_ratio = sM / max(1, sF)
        p5 = p5_ratio <= (P1_MAX_RATIO + 1.0) / 2.0
        print(f"P5 param-shifted M/F {p5_ratio:.3f} (need <= {(P1_MAX_RATIO+1.0)/2.0:.3f}) -> {p5}")

    print(f"P1 pooled M/F           {ratio:.3f} (need <= {P1_MAX_RATIO}) -> {p1}")
    print(f"P2 first-demo win rate  {wr:.3f} ({wins}W/{losses}L, ratio {wl:.2f}) -> "
          f"{'BLOCKED (statistic had zero resolution at calibration)' if p2 is None else p2}")
    print(f"P3 saving vs degenerate (F-M) {cF-cM} vs {P3_FACTOR}x(F-Mdeg) "
          f"{P3_FACTOR*max(0,cF-cMd):.0f} -> {p3}")
    print(f"P4 store-read Z/M       {cZ/max(1,cM):.3f} (need >= {P4_MIN_Z_RATIO}) -> {p4}")
    print(f"P6 M worse on           {worse}/{total} goals (allowed <= {P6_MAX_LOSS_FRAC:.0%}) -> {p6}")

    refuted = ratio >= REFUTE_RATIO
    if p2 is None:
        verdict = "REFUTED" if refuted else "P2-BLOCKED — no SUPPORTED verdict is available"
    elif p1 and p2 and p3 and p4:
        verdict = "SUPPORTED"
    elif refuted or (wl < REFUTE_MIN_WIN_LOSS):
        verdict = "REFUTED"
    else:
        verdict = "NULL"

    print(f"\n{'='*104}\nFROZEN VERDICT: {verdict}\n{'='*104}")
    if verdict == "SUPPORTED":
        print("CLAIM, verbatim and nothing beyond it: 'Meta-trained across same-family hidden-recipe")
        print("worlds, a FROZEN-weight agent that writes its own per-world evidence store discovers and")
        print(f"masters a NEW world of that family {100*(1-ratio):.0f}% cheaper (attempts-to-master,")
        print("discovery included) than an identical fresh agent — same architecture, same store, same")
        print("credit rule, equal in-world budget.'")
        print("NEVER: enablement, cross-family or cross-domain transfer, symbolic DAG induction,")
        print("zero-shot mastery. Disclosed grants: frozen nav skill, item->cell / item->craft-action /")
        print("is_resource / is_valid bindings, the one-of-each-input family invariant.")
    elif verdict == "REFUTED":
        print("The treatment is indistinguishable from another fresh run. Published as a frozen NULL,")
        print("next to ARC 1's three. Arc closed.")
    else:
        print("Neither supported nor refuted at the frozen thresholds. No claim.")

    # ---- pre-specified downgrade labels, used verbatim --------------------------------------------
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
        print(f"\nINTERFERENCE: M was worse than F on {worse}/{total} goals — above the {P6_MAX_LOSS_FRAC:.0%}")
        print("bound. Report the losses symmetrically; ARC 1 measured that accumulation can hurt.")
    for w in Vd + Vs:
        f6 = worlds[w].get("arms", {}).get("F6")
        if f6:
            got = [c for c in (f6 if isinstance(f6, list) else [f6]) if c.get("mastered")]
            if got:
                print(f"\nENABLEMENT GUARD (world {w}): a fresh agent at 6x budget mastered "
                      f"{len(got)} cell(s) F could not. Nothing in this report may be described as")
                print("enablement — the gap is amortisation, not possibility.")

    print("\nBINDING CAVEATS: within-family only (same gen_tree generator); n=3 default held-out worlds,")
    print("so seed variance mixes init with world variance and is not a CI over worlds; the frozen nav")
    print("skill is a common failure axis capping every arm and is a DISCLOSED grant, so the claim is")
    print("composer-level only; the evidence store's write schema is hand-designed and given IDENTICALLY")
    print("to every arm, so the claim is 'a learned policy over self-gathered evidence', never 'the agent")
    print("invented the representation'; attempts-to-master is right-censored at the per-goal budget, so")
    print("cost differences are lower bounds.")


if __name__ == "__main__":
    main()
