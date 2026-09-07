"""ARC 3 — the FROZEN scorer (ARC3_PLAN.md 4.4). Written and frozen BEFORE pretraining; reads only JSON.

Input: craft_v6_out/v61_confirm.json  (files with a _smoke suffix are never read)
    {"b_max": B, "units": [{"id", "lineage", "world", "goals", "tiers",
                            "arms": {"M4": rows, "Fa": rows, "Fb": rows, "L": rows, "G": rows|null}}]}
    rows = [{"goal", "tier", "curve": [m_0 .. m_B]}, ...] in stream order (tier 2, 3, 4)
Optional: craft_v6_out/v61_probe.json for the mechanism caption.

PRIMARY (monotone: an arm whose curves dominate pointwise can never score lower — self-test below):
    A_a(u)      = mean over the unit's goals of mean over b = 1..B of m_a(g, b)
    Delta_learn = mean_u [A_M4(u) - 0.5 (A_Fa(u) + A_Fb(u))]
    se_null     = sd_u [A_Fa(u) - A_Fb(u)] / sqrt(2N);  se_res = 0.0625 sqrt(2 / (3 B N));  se = max
    DEMONSTRATED iff Delta_learn >= 2 se AND Delta_s > 0 for every lineage; NOT iff Delta_learn <= 0;
    else INCONCLUSIVE. Instrument line: |t_init| >= 2 -> INCONCLUSIVE.
Reported, never in the verdict: Delta per tier with its own se, Delta_0 per tier, endpoint gaps,
Delta_s with se_s, the robustness null, the full 0..B area, the wording rule, the mechanism caption.

Usage: python -m scripts.score_v61 [--json craft_v6_out/v61_confirm.json] [--selftest]
"""

import argparse
import json
import os
import statistics as st


def _sd(xs):
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def area(curve, b_from, b_to):
    m = curve[b_from:b_to + 1]
    return sum(m) / len(m)


def unit_area(rows, b_from, b_to):
    return st.mean(area(r["curve"], b_from, b_to) for r in rows)


def score(data):
    B, units = data["b_max"], data["units"]
    N = len(units)
    if N < 2:
        raise ValueError("need >= 2 units")
    A = {a: [unit_area(u["arms"][a], 1, B) for u in units] for a in ("M4", "Fa", "Fb")}
    A_full = {a: [unit_area(u["arms"][a], 0, B) for u in units] for a in ("M4", "Fa", "Fb")}
    dM = [A["M4"][i] - 0.5 * (A["Fa"][i] + A["Fb"][i]) for i in range(N)]
    dN = [A["Fa"][i] - A["Fb"][i] for i in range(N)]
    delta = st.mean(dM)
    se_null = _sd(dN) / (2 * N) ** 0.5
    se_res = 0.0625 * (2 / (3 * B * N)) ** 0.5
    se = max(se_null, se_res)
    t_init = st.mean(dN) / (_sd(dN) / N ** 0.5) if _sd(dN) > 0 else 0.0
    robust = _sd(dM) / N ** 0.5
    # per lineage
    lin = sorted({u["lineage"] for u in units})
    per_lin = {}
    for s in lin:
        idx = [i for i, u in enumerate(units) if u["lineage"] == s]
        ds = [dM[i] for i in idx]
        per_lin[s] = dict(n=len(idx), delta=st.mean(ds), se=(_sd(ds) / len(ds) ** 0.5) if len(ds) > 1 else float("nan"))
    # per tier position
    per_tier = {}
    n_pos = len(units[0]["arms"]["M4"])
    for p in range(n_pos):
        tier = units[0]["arms"]["M4"][p]["tier"]
        m = [area(u["arms"]["M4"][p]["curve"], 1, B) for u in units]
        fa = [area(u["arms"]["Fa"][p]["curve"], 1, B) for u in units]
        fb = [area(u["arms"]["Fb"][p]["curve"], 1, B) for u in units]
        d = [m[i] - 0.5 * (fa[i] + fb[i]) for i in range(N)]
        dn = [fa[i] - fb[i] for i in range(N)]
        se_t = max(_sd(dn) / (2 * N) ** 0.5, 0.0625 * (2 / (B * N)) ** 0.5)
        d0 = [u["arms"]["M4"][p]["curve"][0] - 0.5 * (u["arms"]["Fa"][p]["curve"][0] + u["arms"]["Fb"][p]["curve"][0])
              for u in units]
        end = [u["arms"]["M4"][p]["curve"][B] - 0.5 * (u["arms"]["Fa"][p]["curve"][B] + u["arms"]["Fb"][p]["curve"][B])
               for u in units]
        per_tier[tier] = dict(delta=st.mean(d), se=se_t, delta_0=st.mean(d0), endpoint_gap=st.mean(end),
                              M4=st.mean(m), fresh=st.mean(0.5 * (fa[i] + fb[i]) for i in range(N)),
                              L=st.mean(area(u["arms"]["L"][p]["curve"], 1, B) for u in units if u["arms"].get("L")),
                              G=(st.mean(area(u["arms"]["G"][p]["curve"], 1, B) for u in units if u["arms"].get("G"))
                                 if any(u["arms"].get("G") for u in units) else None))
    # verdict
    if abs(t_init) >= 2:
        verdict = "INCONCLUSIVE"; reason = f"instrument line: |t_init| = {abs(t_init):.2f} >= 2 (fresh arms not exchangeable)"
    elif delta <= 0:
        verdict = "NOT-DEMONSTRATED"; reason = "Delta_learn <= 0"
    elif delta >= 2 * se and all(v["delta"] > 0 for v in per_lin.values()):
        verdict = "DEMONSTRATED"; reason = f"Delta_learn {delta:.4f} >= 2 se {2*se:.4f} and Delta_s > 0 in every lineage"
    else:
        verdict = "INCONCLUSIVE"; reason = f"0 < Delta_learn {delta:.4f} < 2 se {2*se:.4f} or a lineage <= 0"
    t3 = per_tier.get(3)
    learns_faster_wording = bool(t3 and t3["delta"] >= 2 * t3["se"])
    if verdict == "DEMONSTRATED":
        sentence = ("learns faster thanks to prior experience" if learns_faster_wording else
                    "solves with prior knowledge what a fresh agent does not learn in the budget")
    elif verdict == "NOT-DEMONSTRATED":
        sentence = "prior experience does not make learning faster here"
    else:
        sentence = "not distinguishable from zero at this N"
    return dict(N=N, B=B, delta_learn=delta, se_null=se_null, se_res=se_res, se=se, t_init=t_init,
                robust_se=robust, per_lineage=per_lin, per_tier=per_tier, verdict=verdict, reason=reason,
                sentence=sentence, learns_faster_wording=learns_faster_wording,
                delta_full_area=st.mean(A_full["M4"][i] - 0.5 * (A_full["Fa"][i] + A_full["Fb"][i]) for i in range(N)),
                A=A, per_unit_delta=dM, per_unit_null=dN)


def selftest(n_trials=20_000, seed=0):
    """Monotonicity: if M4's curves dominate the fresh curves pointwise on every unit, Delta_learn >= 0 and
    the verdict is never NOT-DEMONSTRATED; and a unit-wise pointwise improvement of M4 never lowers Delta."""
    import random
    rng = random.Random(seed)
    B = 3
    for _ in range(n_trials):
        units = []
        for u in range(4):
            arms = {}
            for a in ("Fa", "Fb"):
                arms[a] = [dict(goal=p, tier=p + 2, curve=[rng.random() for _ in range(B + 1)]) for p in range(3)]
            arms["M4"] = [dict(goal=p, tier=p + 2, curve=[min(1.0, max(arms["Fa"][p]["curve"][b], arms["Fb"][p]["curve"][b]) + rng.random() * 0.1)
                                                        for b in range(B + 1)]) for p in range(3)]
            arms["L"] = [dict(goal=p, tier=p + 2, curve=[1.0] * (B + 1)) for p in range(3)]
            units.append(dict(id=str(u), lineage=u % 2, world=u, arms=arms))
        r = score(dict(b_max=B, units=units))
        assert r["delta_learn"] >= 0 and r["verdict"] != "NOT-DEMONSTRATED", "dominance scored as a loss"
        # improve M4 pointwise on one unit -> Delta cannot fall
        d0 = r["delta_learn"]
        for row in units[0]["arms"]["M4"]:
            row["curve"] = [min(1.0, m + 0.05) for m in row["curve"]]
        assert score(dict(b_max=B, units=units))["delta_learn"] >= d0 - 1e-12, "pointwise improvement lowered Delta"
    print(f"selftest ok: {n_trials} random dominance trials, monotone")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="craft_v6_out/v61_confirm.json")
    ap.add_argument("--probe", default="craft_v6_out/v61_probe.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    assert "_smoke" not in a.json, "smoke files are never scored"
    data = json.load(open(a.json))
    lineages = {u["lineage"] for u in data["units"]}
    N, B = len(data["units"]), data["b_max"]
    # design guard: v61 (N 12 at B 3 / N 9 at B 4) or v62 (N 9 at B 5 or 6). Widened for v62 at its
    # STAGE-0 freeze, before any v62 data existed (audit finding); the verdict rule below is untouched.
    ok = len(lineages) == 3 and ((N, B) in ((12, 3), (9, 4)) or (N == 9 and B in (5, 6)))
    if not ok:
        print(f"PARTIAL — no verdict: {N} units, lineages {sorted(lineages)}, B_max {B} "
              f"(the frozen designs: v61 N 12 @ B 3 or N 9 @ B 4; v62 N 9 @ B 5 or 6)")
        return
    r = score(data)
    print("=" * 96)
    print(f"ARC 3 CONFIRMATORY — N {r['N']} units, B_max {r['B']}")
    print(f"  Delta_learn = {r['delta_learn']:+.4f} | se_null {r['se_null']:.4f} | se_res {r['se_res']:.4f} | "
          f"se {r['se']:.4f} | robustness se {r['robust_se']:.4f} | t_init {r['t_init']:+.2f}")
    for s, v in r["per_lineage"].items():
        print(f"  lineage {s}: Delta_s {v['delta']:+.4f} (se_s {v['se']:.4f}, n {v['n']})")
    for t, v in sorted(r["per_tier"].items()):
        print(f"  tier {t}: Delta {v['delta']:+.4f} (se {v['se']:.4f}) | Delta_0 {v['delta_0']:+.4f} | endpoint gap "
              f"{v['endpoint_gap']:+.4f} | M4 {v['M4']:.3f} fresh {v['fresh']:.3f} L {v['L']:.3f}"
              + (f" G' {v['G']:.3f}" if v["G"] is not None else ""))
    print(f"  full 0..B area Delta (secondary) = {r['delta_full_area']:+.4f}")
    print(f"\n  VERDICT: {r['verdict']}  ({r['reason']})")
    print(f"  wording: \"{r['sentence']}\"")
    if os.path.exists(a.probe):
        p = json.load(open(a.probe))
        print(f"  mechanism (probe): {p.get('summary', p)}")
    print("=" * 96)
    json.dump(r, open(a.json.replace(".json", "_score.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
