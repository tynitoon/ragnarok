"""G0 stages shared by the driver (real store) and the planted control T8 (docs/ARC4_V84_PROTOCOL_EN.md
A.9-A.13): tuning curves, kappa selection, unit eligibility on F, arm curves with integrity checks, and
the aggregation. The same code runs on real and planted labels."""
import math

import numpy as np

from ragnarok.evaluation import g0_v84 as G
from ragnarok.evaluation.g0_pipeline_v84 import arm_sigmas, curve_for, direction_data, k_star

TUNED = ("F", "X", "P", "U", "K*A", "X-mean", "P_shift", "P_rot")
DECISION = ("F", "X", "P", "U", "K*A")


def directions(g, head, streams, seeds, geo, gi_of):
    return [direction_data(g, head, streams, seeds, geo, gi_of, a, b) for a, b in ((0, 1), (1, 0))]


def tune_rows(g, head_list, streams, seeds, geo, sources, src, gi_of, kappas):
    """A.9 tuning curves of one tuning game: mean AUC over checkpoints per (head, dir, arm, kappa)."""
    res = []
    for head in head_list:
        arms = arm_sigmas(g, head, geo, sources, src, gi_of)
        if arms is None:
            continue
        for di, dd in enumerate(directions(g, head, streams, seeds, geo, gi_of)):
            if min(dd["counts"]) < 30:
                continue
            ks = k_star(dd["ZA"], dd["yA"], dd["blockA"], geo["d"])
            sig = {a: arms[a] for a in TUNED if a in arms and a != "K*A"}
            if ks is not None:
                sig["K*A"] = (ks, None)
            for arm, s in sig.items():
                for kap in kappas:
                    a = curve_for(arm, s, dd, kap)
                    res.append(dict(game=g, head=head, dir=di, arm=arm, kappa=kap, mean_auc=float(np.mean(a))))
    return res


def select_kappa(rows, kappas, arms=TUNED):
    """Highest mean over tuning curves of the mean AUC over checkpoints; exact ties to the smaller kappa."""
    out = {}
    for arm in arms:
        best, bk = -1.0, None
        for k in sorted(kappas):
            v = [r["mean_auc"] for r in rows if r["arm"] == arm and r["kappa"] == k]
            if v and float(np.mean(v)) > best + 1e-15:
                best, bk = float(np.mean(v)), k
        out[arm] = bk
    return out


def edge_extension(sel, kappas):
    """Edge rule: if a decision arm sits at a grid end, extend once by a factor 4 on that side."""
    lo, hi = min(kappas), max(kappas)
    extra = []
    if any(sel.get(a) == lo for a in DECISION):
        extra.append(lo / 4)
    if any(sel.get(a) == hi for a in DECISION):
        extra.append(hi * 4)
    return extra


def unit_rows(g, head_list, streams, seeds, geo, gi_of, kF, outside):
    """A.10 eligibility on F alone (both directions)."""
    res = []
    for head in head_list:
        dirs = []
        for di, dd in enumerate(directions(g, head, streams, seeds, geo, gi_of)):
            if min(dd["counts"]) < 30:
                dirs.append(dict(dir=di, counts=dd["counts"], ok=False))
                continue
            a = curve_for("F", (np.eye(geo["d"]), None), dd, kF)
            end = float(np.mean(a[-3:]))
            thr = 0.60 if head == "good" else 0.65
            ok = end >= thr and end - dd["slot_auc"] >= 0.03
            dirs.append(dict(dir=di, counts=dd["counts"], auc_F=a, auc_F_end=end, auc_slot=dd["slot_auc"],
                             clocks=dd["clocks"], c_end=dd["c_end"], removed=dd["removed"], ok=bool(ok),
                             prefix_ok=_prefix_ok({"F": a}, dd)))
        res.append(dict(game=g, head=head, eligible=len(dirs) == 2 and all(x["ok"] for x in dirs), dirs=dirs,
                        d=geo["d"], outside=outside))
    return res


def _prefix_ok(cur, dd):
    """Prefix certificate on the real stream: every zero-mean arm has AUC exactly 0.5 at every checkpoint
    before the learning rows contain both classes."""
    y = dd["yA"]
    first, seen0, seen1 = None, False, False
    for i in np.nonzero(y >= 0)[0]:
        seen0 |= bool(y[i] == 0)
        seen1 |= bool(y[i] == 1)
        if seen0 and seen1:
            first = int(i) + 1
            break
    for arm, a in cur.items():
        if a is None or arm == "X-mean":
            continue
        for c in (a if arm == "X_8" else [a]):
            for n, v in zip(dd["cks"], c):
                if (first is None or n < first) and v != 0.5:
                    return False
    return True


def arm_curve_rows(g, head_list, streams, seeds, geo, sources, src, gi_of, kap):
    """All non-F arms (A.8), with the integrity checks of (vi)."""
    res = []
    for head in head_list:
        arms = arm_sigmas(g, head, geo, sources, src, gi_of)
        for di, dd in enumerate(directions(g, head, streams, seeds, geo, gi_of)):
            cur = {}
            for arm in ("X", "P", "U", "X-mean", "P_shift", "P_rot"):
                if arm in arms:
                    cur[arm] = curve_for(arm, arms[arm], dd, kap[arm])
            cur["X_8"] = [curve_for("X_8", (S, None), dd, kap["X_8"]) for S in arms["X_8"]]
            ks = k_star(dd["ZA"], dd["yA"], dd["blockA"], geo["d"])
            cur["K*A"] = curve_for("K*A", (ks, None), dd, kap["K*A"]) if ks is not None else None
            sB = streams[1 - di]
            t_idx, _ = G.test_set(streams[di], sB, head, gi_of[g], seeds[1 - di])
            ev = np.nonzero((sB["block"] % 2 == 0) & ~np.isin(sB["hash"], sB["hash"][t_idx]))[0]
            kb = k_star(G.project(sB["X"][ev], geo), sB["y"][head][ev], sB["block"][ev], geo["d"],
                        fold=(sB["block"][ev] // 2) % 2) if len(ev) else None
            cur["K*Beven"] = curve_for("K*Beven", (kb, None), dd, kap["K*Beven"]) if kb is not None else None
            lam_X = arms["_lam_X"]
            integ = dict(trace_ok=all(abs(np.trace(arms[a][0]) - geo["d"]) < 1e-6 * geo["d"] for a in ("X", "P", "U", "P_rot")),
                         spectrum_ok=all(float(np.max(np.abs(G.eig_desc(arms[a][0])[0] - lam_X) / np.maximum(np.abs(lam_X), 1e-300))) <= 1e-9
                                         for a in ("P", "U", "P_rot")),
                         prefix_ok=_prefix_ok(cur, dd))
            res.append(dict(game=g, head=head, dir=di, curves=cur, integrity=integ, n_sources=arms["_n_sources"]))
    return res


def summarize(units, curve_rows):
    """A.11-A.12 aggregation from unit rows (holding the F curves) and the other arms' curve rows."""
    elig = [u for u in units if u["eligible"]]
    games = sorted({u["game"] for u in elig})
    unit_game = {(u["game"], u["head"]): u["game"] for u in elig}
    base = {(u["game"], u["head"], d["dir"]): d for u in elig for d in u["dirs"]}
    lam, late, ends, same = {}, {}, {}, []
    lvl = {q: {} for q in G.QS}
    n_kstar_none = 0
    for r in curve_rows:
        key = (r["game"], r["head"], r["dir"])
        if key not in base:
            continue
        b = base[key]
        cl, c_end, fend = b["clocks"], b["c_end"], b["auc_F_end"]
        arms = dict(r["curves"], F=b["auc_F"])
        n_kstar_none += arms.get("K*A") is None
        N25 = {}
        for arm, a in arms.items():
            if a is None:
                continue
            if arm == "X_8":
                if a:
                    ls = [G.lambdas(x, cl, fend, True, c_end) for x in a]
                    lam[(arm,) + key] = float(np.mean([x[0] for x in ls]))
                    late[(arm,) + key] = float(np.mean([x[1] for x in ls]))
                    for q in G.QS:
                        lvl[q][(arm,) + key] = float(np.mean([math.log(x[2][q]) for x in ls]))
                continue
            l, ll, N, _ = G.lambdas(a, cl, fend, arm in G.CARRIED, c_end)
            lam[(arm,) + key], late[(arm,) + key] = l, ll
            for q in G.QS:
                lvl[q][(arm,) + key] = math.log(N[q])
            ends[(arm,) + key] = float(np.mean(a[-3:]))
            if arm in DECISION:
                N25[arm] = N[0.25]
        same.append(len({round(v, 6) for v in N25.values()}) == 1)

    def lg(arm_x, comps, table):
        lu = {}
        for u in elig:
            k = (u["game"], u["head"])

            def m(a):
                v = [table.get((a,) + k + (d,)) for d in (0, 1)]
                return None if any(x is None for x in v) else float(np.mean(v))
            lx, lc = m(arm_x), [m(c) for c in comps]
            if lx is None or any(x is None for x in lc):
                continue
            lu[k] = min(lc) - lx
        return G.per_game(lu, unit_game)

    def summ(d):
        if not d:
            return None
        lo, hi = G.boot(d)
        return dict(R=G.ratio(d), lo=lo, hi=hi, p=G.signflip_p(d), n_games=len(d), logo_min=G.logo_min(d),
                    HL=G.hodges_lehmann(d), per_game={k: math.exp(v) for k, v in d.items()})

    res = dict(n_units=len(elig), n_games=len(games), games=games)
    res["X:C*"] = summ(lg("X", ("F", "P", "U"), lam))
    res["X:C*_late"] = summ(lg("X", ("F", "P", "U"), late))
    for a in ("K*A", "X", "P", "U", "P_rot", "X-mean", "X_8", "P_shift", "K*Beven"):
        res[f"{a}:F"] = summ(lg(a, ("F",), lam))
    res["X:F_late"] = summ(lg("X", ("F",), late))
    guard = [ends[("X",) + k] - base[k]["auc_F_end"] for k in base if ("X",) + k in ends]
    res["ceiling_guard"] = float(np.mean(guard)) if guard else None
    res["share_same_N025"] = float(np.mean(same)) if same else None
    res["n_dirs_kstar_none"] = n_kstar_none
    res["profile_X:C*"] = {str(q): summ(lg("X", ("F", "P", "U"), lvl[q])) for q in G.QS}   # C* per level
    res["profile_X:F"] = {str(q): summ(lg("X", ("F",), lvl[q])) for q in G.QS}
    return res
