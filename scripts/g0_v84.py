"""G0 driver (docs/ARC4_V84_PROTOCOL_EN.md A.16): sums, sources, tune, units, arms, verdict.

  python scripts/g0_v84.py sums
  python scripts/g0_v84.py sources [workers]
  python scripts/g0_v84.py tune [workers]          -> g0_kappa.json (commit before any development curve)
  python scripts/g0_v84.py units [workers]         -> g0_units.json (commit before any non-F development curve)
  python scripts/g0_v84.py arms [workers]          -> curves.json
  python scripts/g0_v84.py verdict                 -> g0_verdict.json
"""
import json
import math
import os
import pathlib
import pickle
import sys
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import g0_v84 as G                                       # noqa: E402
from ragnarok.evaluation.g0_pipeline_v84 import arm_sigmas, curve_for, direction_data, fit_sources, k_star  # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV                         # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp                    # noqa: E402

OUT = ROOT / "craft_v6_out" / "v84" / "g0"
STORE = OUT / "store"
GAMES = DEV + CALIBRATION
GI = {g: i for i, g in enumerate(GAMES)}
SEEDS = (9131, 9132)
TUNED = ("F", "X", "P", "U", "K*A", "X-mean", "P_shift", "P_rot")
DECISION = ("F", "X", "P", "U", "K*A")
CLAIM = ("Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 "
         "(score rises, lives count). It is not live play without savestates.")
SENSES = "pixels, own buttons, max(0, r_step) per step, ale.lives(); game over only as an episode boundary"


def load_game(g):
    out = []
    for s in SEEDS:
        z = np.load(STORE / f"{g}_{s}.npz")
        out.append(dict(X=z["X"], y=dict(good=z["y_good"], bad=z["y_bad"]), clock=z["clock"], block=z["block"],
                        it=z["it"], hash=z["hash"], slot=z["slot"]))
    return out


def load_sums():
    z = np.load(OUT / "sums.npz")
    return {g: dict(n=int(z[f"{g}_n"]), s=z[f"{g}_s"], ss=z[f"{g}_ss"]) for g in GAMES}


def load_sources():
    with open(OUT / "sources.pkl", "rb") as f:
        return pickle.load(f)


def save_json(name, obj):
    p = OUT / name
    with open(str(p) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(str(p) + ".tmp", p)


# ---------------------------------------------------------------- sums, sources
def cmd_sums():
    out = {}
    for g in GAMES:
        q = G.game_sums(load_game(g))
        out[f"{g}_n"], out[f"{g}_s"], out[f"{g}_ss"] = q["n"], q["s"], q["ss"]
    np.savez(OUT / "sums.npz", **out)
    print("sums ok")


def _sources_one(g):
    return g, fit_sources({g: load_game(g)}, [g], GI, SEEDS)


def cmd_sources(workers):
    prov = stamp()
    res = {}
    with Pool(workers) as pool:
        for g, r in pool.imap_unordered(_sources_one, GAMES):
            res.update(r)
            print("sources", g, {k[1] + "/" + k[2]: (None if v is None else v["rho"]) for k, v in r.items()}, flush=True)
    with open(OUT / "sources.pkl", "wb") as f:
        pickle.dump(res, f)
    save_json("sources.json", dict(provenance=prov, included={f"{k[0]}|{k[1]}|{k[2]}": v is not None for k, v in res.items()},
                                   rho={f"{k[0]}|{k[1]}|{k[2]}": (v["rho"] if v else None) for k, v in res.items()},
                                   sources_sha256=full_hash(OUT / "sources.pkl")))


# ---------------------------------------------------------------- per held-out game work
def held_out_setup(g, sums, sources):
    src = [h for h in GAMES if h != g]
    geo = G.geometry(sums, src)
    streams = load_game(g)
    return src, geo, streams


def _directions(g, head, streams, geo):
    out = []
    for a, b in ((0, 1), (1, 0)):
        dd = direction_data(g, head, streams, SEEDS, geo, GI, a, b)
        out.append(dd)
    return out


def _tune_one(args):
    g, kappas = args
    sums, sources = load_sums(), load_sources()
    src, geo, streams = held_out_setup(g, sums, sources)
    res = []
    for head in G.HEADS:
        arms = arm_sigmas(g, head, geo, sources, src, GI)
        if arms is None:
            continue
        for di, dd in enumerate(_directions(g, head, streams, geo)):
            pl, nl, pt, nt = dd["counts"]
            if min(pl, nl, pt, nt) < 30:
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


def select_kappa(rows, kappas):
    out = {}
    for arm in TUNED:
        best, bk = -1.0, None
        for k in sorted(kappas):
            v = [r["mean_auc"] for r in rows if r["arm"] == arm and r["kappa"] == k]
            if not v:
                continue
            m = float(np.mean(v))
            if m > best + 1e-15:
                best, bk = m, k
        out[arm] = bk
    return out


def cmd_tune(workers):
    prov = stamp()
    kappas = list(G.KAPPA_GRID)
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_tune_one, [(c, kappas) for c in CALIBRATION]) for r in rs]
    sel = select_kappa(rows, kappas)
    extended = None
    lo, hi = min(kappas), max(kappas)
    edge_lo = any(sel.get(a) == lo for a in DECISION)
    edge_hi = any(sel.get(a) == hi for a in DECISION)
    if edge_lo or edge_hi:                                        # edge rule: extend once, re-select
        extra = ([lo / 4] if edge_lo else []) + ([hi * 4] if edge_hi else [])
        with Pool(workers) as pool:
            rows += [r for rs in pool.map(_tune_one, [(c, extra) for c in CALIBRATION]) for r in rs]
        kappas = sorted(kappas + extra)
        sel = select_kappa(rows, kappas)
        extended = extra
    sel["X_8"], sel["K*Beven"] = sel["X"], sel["K*A"]
    save_json("g0_kappa.json", dict(provenance=prov, kappa=sel, grid=kappas, extended=extended, rows=rows,
                                    claim_header=CLAIM, senses=SENSES))
    print(json.dumps(sel))


def _units_one(args):
    g, kF = args
    sums, sources = load_sums(), load_sources()
    src, geo, streams = held_out_setup(g, sums, sources)
    res = []
    for head in G.HEADS:
        dirs = []
        for di, dd in enumerate(_directions(g, head, streams, geo)):
            pl, nl, pt, nt = dd["counts"]
            if min(pl, nl, pt, nt) < 30:
                dirs.append(dict(dir=di, counts=dd["counts"], ok=False))
                continue
            a = curve_for("F", (np.eye(geo["d"]), None), dd, kF)
            end = float(np.mean(a[-3:]))
            thr = 0.60 if head == "good" else 0.65
            ok = end >= thr and end - dd["slot_auc"] >= 0.03
            dirs.append(dict(dir=di, counts=dd["counts"], auc_F=a, auc_F_end=end, auc_slot=dd["slot_auc"],
                             clocks=dd["clocks"], c_end=dd["c_end"], removed=dd["removed"], ok=bool(ok)))
        res.append(dict(game=g, head=head, eligible=all(x["ok"] for x in dirs) and len(dirs) == 2, dirs=dirs,
                        d=geo["d"], outside=G.outside_share(streams[0]["X"], geo)))
    return res


def cmd_units(workers):
    prov = stamp()
    kap = json.load(open(OUT / "g0_kappa.json", encoding="utf-8"))
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_units_one, [(g, kap["kappa"]["F"]) for g in DEV]) for r in rs]
    save_json("g0_units.json", dict(provenance=prov, kappa_sha256=full_hash(OUT / "g0_kappa.json"), units=rows,
                                    n_eligible=sum(r["eligible"] for r in rows),
                                    games_eligible=sorted({r["game"] for r in rows if r["eligible"]})))
    print("eligible", sum(r["eligible"] for r in rows), "games", len({r["game"] for r in rows if r["eligible"]}))


def _arms_one(args):
    g, heads, kap = args
    sums, sources = load_sums(), load_sources()
    src, geo, streams = held_out_setup(g, sums, sources)
    res = []
    for head in heads:
        arms = arm_sigmas(g, head, geo, sources, src, GI)
        for di, dd in enumerate(_directions(g, head, streams, geo)):
            cur = {}
            for arm in ("X", "P", "U", "X-mean", "P_shift", "P_rot"):
                if arm in arms:
                    cur[arm] = curve_for(arm, arms[arm], dd, kap[arm])
            cur["X_8"] = [curve_for("X_8", (S, None), dd, kap["X_8"]) for S in arms["X_8"]]
            ks = k_star(dd["ZA"], dd["yA"], dd["blockA"], geo["d"])
            cur["K*A"] = curve_for("K*A", (ks, None), dd, kap["K*A"]) if ks is not None else None
            # K*Beven (report only): B's even blocks, deduplicated against the test set
            sB = streams[1 - di]
            t_idx, _ = G.test_set(streams[di], sB, head, GI[g], SEEDS[1 - di])
            ev = np.nonzero((sB["block"] % 2 == 0) & ~np.isin(sB["hash"], sB["hash"][t_idx]))[0]
            ZB = G.project(sB["X"][ev], geo)
            kb = k_star(ZB, sB["y"][head][ev], sB["block"][ev], geo["d"])
            cur["K*Beven"] = curve_for("K*Beven", (kb, None), dd, kap["K*Beven"]) if kb is not None else None
            lam_X = arms["_lam_X"]
            integ = dict(trace_ok=all(abs(np.trace(arms[a][0]) - geo["d"]) < 1e-6 * geo["d"] for a in ("X", "P", "U", "P_rot")),
                         spectrum_ok=all(np.max(np.abs(G.eig_desc(arms[a][0])[0] - lam_X) / np.maximum(np.abs(lam_X), 1e-300)) <= 1e-9
                                         for a in ("P", "U", "P_rot")))
            res.append(dict(game=g, head=head, dir=di, curves=cur, integrity=integ, n_sources=arms["_n_sources"]))
    return res


def cmd_arms(workers):
    prov = stamp()
    kap = json.load(open(OUT / "g0_kappa.json", encoding="utf-8"))["kappa"]
    units = json.load(open(OUT / "g0_units.json", encoding="utf-8"))
    todo = {}
    for u in units["units"]:
        if u["eligible"]:
            todo.setdefault(u["game"], []).append(u["head"])
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_arms_one, [(g, hs, kap) for g, hs in todo.items()]) for r in rs]
    save_json("curves.json", dict(provenance=prov, units_sha256=full_hash(OUT / "g0_units.json"), rows=rows))
    print("curves", len(rows))


# ---------------------------------------------------------------- verdict
def cmd_verdict():
    prov = stamp()
    units = json.load(open(OUT / "g0_units.json", encoding="utf-8"))
    cur = json.load(open(OUT / "curves.json", encoding="utf-8"))["rows"]
    elig = [u for u in units["units"] if u["eligible"]]
    games = sorted({u["game"] for u in elig})
    unit_game = {(u["game"], u["head"]): u["game"] for u in elig}
    base = {(u["game"], u["head"], d["dir"]): d for u in elig for d in u["dirs"]}
    lam, late, ends, sameN = {}, {}, {}, []
    for r in cur:
        key = (r["game"], r["head"], r["dir"])
        b = base[key]
        cl, c_end, fend = b["clocks"], b["c_end"], b["auc_F_end"]
        arms = dict(r["curves"], F=b["auc_F"])
        N25 = {}
        for arm, a in arms.items():
            if a is None:
                continue
            if arm == "X_8":
                ls = [G.lambdas(x, cl, fend, True, c_end) for x in a]
                lam[(arm,) + key] = float(np.mean([x[0] for x in ls]))
                late[(arm,) + key] = float(np.mean([x[1] for x in ls]))
                continue
            l, ll, N, ok = G.lambdas(a, cl, fend, arm in G.CARRIED, c_end)
            lam[(arm,) + key], late[(arm,) + key] = l, ll
            ends[(arm,) + key] = float(np.mean(a[-3:]))
            if arm in ("F", "X", "P", "U", "K*A"):
                N25[arm] = N[0.25]
        sameN.append(len(set(round(v, 6) for v in N25.values())) == 1)

    def unit_l(arm_x, comps, table):
        lu = {}
        for u in elig:
            k = (u["game"], u["head"])
            def m(a):
                v = [table.get((a,) + k + (d,)) for d in (0, 1)]
                return None if any(x is None for x in v) else float(np.mean(v))
            lx = m(arm_x)
            lc = [m(c) for c in comps]
            if lx is None or any(x is None for x in lc):
                continue
            lu[k] = min(lc) - lx
        return G.per_game(lu, unit_game)

    def summ(lg):
        if not lg:
            return None
        lo, hi = G.boot(lg)
        return dict(R=G.ratio(lg), lo=lo, hi=hi, p=G.signflip_p(lg), n_games=len(lg), logo_min=G.logo_min(lg),
                    HL=G.hodges_lehmann(lg))

    res = dict(n_units=len(elig), n_games=len(games))
    res["X:C*"] = summ(unit_l("X", ("F", "P", "U"), lam))
    res["X:C*_late"] = summ(unit_l("X", ("F", "P", "U"), late))
    res["K*A:F"] = summ(unit_l("K*A", ("F",), lam))
    for a in ("X", "P", "U", "P_rot", "X-mean", "X_8", "P_shift", "K*Beven"):
        res[f"{a}:F"] = summ(unit_l(a, ("F",), lam))
    res["X:F_late"] = summ(unit_l("X", ("F",), late))
    guard = [ends[("X",) + (u["game"], u["head"], d)] - base[(u["game"], u["head"], d)]["auc_F_end"]
             for u in elig for d in (0, 1) if ("X",) + (u["game"], u["head"], d) in ends]
    res["ceiling_guard"] = float(np.mean(guard)) if guard else None
    res["share_same_N025"] = float(np.mean(sameN)) if sameN else None
    integ = all(r["integrity"]["trace_ok"] and r["integrity"]["spectrum_ok"] for r in cur)
    tests_ok = (OUT / "tests_passed.json").exists()
    i_ok = len(elig) >= 12 and len(games) >= 8
    ii_ok = res["K*A:F"] is not None and res["K*A:F"]["R"] >= 2.0 and res["K*A:F"]["lo"] >= 1.25
    iii = res["X:C*"] is not None and res["X:C*"]["R"] >= 1.5 and res["X:C*"]["lo"] > 1.1 and res["X:C*"]["p"] <= 0.05
    iv = res["X:C*_late"] is not None and res["X:C*_late"]["lo"] > 1 and res["X:C*_late"]["p"] <= 0.05
    v_ok = res["ceiling_guard"] is not None and res["ceiling_guard"] >= -0.01
    if not (integ and tests_ok):
        label = "Interruption"
    elif not i_ok:
        label = "KC-knowledge-room"
    elif not ii_ok:
        label = "KC-knowledge-worth"
    elif iii and iv and v_ok:
        label = "PASS"
    else:
        label = "KC-knowledge-speed"
    caps = []
    if label == "KC-knowledge-speed":
        xf, pf, uf = res.get("X:F"), res.get("P:F"), res.get("U:F")
        if xf and xf["lo"] > 1 and ((pf and pf["lo"] > 1) or (uf and uf["lo"] > 1)):
            caps.append("GENERIC-SHRINKAGE")
        if iii and not iv:
            caps.append("FIRST-RULES-ONLY")
        if res["X:C*"] and res["X:C*"]["lo"] > 1 and not iii:
            caps.append("SUB-THRESHOLD")
        if res.get("P:F") and res["P:F"]["hi"] < 1:
            caps.append("WEAK-PLACEBO")
    res.update(conditions=dict(i=i_ok, ii=ii_ok, iii=iii, iv=iv, v=v_ok, vi=bool(integ and tests_ok)),
               label=label, captions=caps, claim_header=CLAIM, senses=SENSES, games=games,
               inputs=dict(units_sha256=full_hash(OUT / "g0_units.json"), curves_sha256=full_hash(OUT / "curves.json"),
                           kappa_sha256=full_hash(OUT / "g0_kappa.json")), provenance=prov)
    save_json("g0_verdict.json", res)
    print(json.dumps({k: res[k] for k in ("label", "captions", "n_units", "n_games", "conditions")}))
    print(json.dumps({k: (None if res[k] is None else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in res[k].items()})
                      for k in ("X:C*", "X:C*_late", "K*A:F", "X:F", "P:F", "U:F")}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    OUT.mkdir(parents=True, exist_ok=True)
    dict(sums=lambda: cmd_sums(), sources=lambda: cmd_sources(w), tune=lambda: cmd_tune(w), units=lambda: cmd_units(w),
         arms=lambda: cmd_arms(w), verdict=lambda: cmd_verdict())[cmd]()
