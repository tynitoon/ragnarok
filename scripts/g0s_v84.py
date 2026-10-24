"""G0-S, the search-sense knowledge clock (docs/ARC4_V84_PROTOCOL_EN.md §6.2), offline, zero frames.

The Part A machinery on the A1' (G1b, V2, seeds 9712-9713) EXPANSION logs: one head (ADV of an
expansion, from its 44 search-sense features); learning stream = one seed's expansions in order; test =
the other seed's odd 50-expansion blocks; K*_A = a head on the whole learning stream; comparator C* =
per-unit best of {F, P (ADV permuted within each run), U, T_clock (Sigma from source heads fitted on the
clock and run-context features only)}; units = development games with AUC_F,end >= 0.60 and >= 0.03 over
the per-process base rate; condition (i) >= 12 games; every other threshold as in A.13. Planted control
(T8-S, build note 13) before any real-label computation.

  python scripts/g0s_v84.py store          (expansion streams from the committed G1b outputs)
  python scripts/g0s_v84.py planted        (T8-S)
  python scripts/g0s_v84.py run            (sources, kappa, units, arms, verdict; refuses if T8-S failed)
"""
import hashlib
import json
import os
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import g0_stages_v84 as ST                                  # noqa: E402
from ragnarok.evaluation import g0_v84 as G                                         # noqa: E402
from ragnarok.evaluation.g0_pipeline_v84 import fit_sources                          # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV                           # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp                      # noqa: E402

G1B = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84\g1b")
OUT = ROOT / "craft_v6_out" / "v84" / "g0s"
GAMES = DEV + CALIBRATION
GI = {g: i for i, g in enumerate(GAMES)}
SEEDS = (9712, 9713)
CLOCK_COLS = tuple(range(19, 26))       # budget share .. new-cell rate (explorer2._features order)
HEAD = ("good",)                        # the single ADV head, carried in the "good" slot of the G0 code
PATHS = ("ragnarok", "scripts", "tests")
CLAIM = ("Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 "
         "(score rises, lives count). It is not live play without savestates.")


def prov(*inputs):
    p = stamp(inputs, cwd=str(ROOT), paths=PATHS, full=True)
    if not p["code_sha"] or p["dirty"]:
        sys.exit("Interruption: code tree not clean")
    return p


def save(name, obj):
    obj = dict(obj, provenance_end=stamp((), cwd=str(ROOT), paths=PATHS))
    p = OUT / name
    with open(str(p) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(str(p) + ".tmp", p)


def cmd_store():
    p = prov()
    (OUT / "store").mkdir(parents=True, exist_ok=True)
    info = []
    for g in GAMES:
        for s in SEEDS:
            js = json.load(open(G1B / f"{g}_{s}.json", encoding="utf-8"))
            z = np.load(G1B / f"{g}_{s}.npz")
            meta = z["exp_meta"]                                   # exp_n, cell, q, adv, billed, prop, losses, len
            X = z["exp_feat"].astype(np.float32)
            entry = int(js["entry"]["entry_frames"])
            hs = np.array([np.frombuffer(hashlib.blake2b(x.tobytes(), digest_size=8).digest(), np.uint64)[0] for x in X],
                          np.uint64)
            np.savez(OUT / "store" / f"{g}_{s}.npz", X=X, y=meta[:, 3].astype(np.int8), clock=(meta[:, 4] - entry).astype(np.int64),
                     block=(meta[:, 0] // 50).astype(np.int32), it=meta[:, 0].astype(np.int32), slot=meta[:, 2].astype(np.int16),
                     hash=hs)
            info.append(dict(game=g, seed=s, n=len(X), adv_rate=float(meta[:, 3].mean()) if len(meta) else None,
                             source_npz_sha256=full_hash(G1B / f"{g}_{s}.npz"), file_sha256=full_hash(OUT / "store" / f"{g}_{s}.npz")))
    save("store.json", dict(provenance=p, streams=info, claim_header=CLAIM))
    print("expansion streams", len(info), "rows", sum(x["n"] for x in info))


def load_game(g):
    out = []
    for s in SEEDS:
        z = np.load(OUT / "store" / f"{g}_{s}.npz")
        y = z["y"]
        out.append(dict(X=z["X"], y=dict(good=y, bad=np.full(len(y), -1, np.int8)), clock=z["clock"], block=z["block"],
                        it=z["it"], hash=z["hash"], slot=z["slot"]))
    return out


def run_pipeline(data, pool, tune_games, unit_games, kinds=("real", "perm", "shift", "clock")):
    sums = {g: G.game_sums(data[g]) for g in pool}
    sources = fit_sources(data, pool, GI, SEEDS, heads=HEAD, kinds=kinds, clock_cols=CLOCK_COLS)

    def geo_of(g):
        src = [h for h in pool if h != g]
        return src, G.geometry(sums, src)
    kappas = list(G.KAPPA_GRID)
    rows = []
    for g in tune_games:
        src, geo = geo_of(g)
        rows += ST.tune_rows(g, HEAD, data[g], SEEDS, geo, sources, src, GI, kappas)
    sel = ST.select_kappa(rows, kappas)
    lo, hi = min(kappas), max(kappas)
    dec = ("F", "X", "P", "U", "K*A", "T_clock")
    extra = ([lo / 4] if any(sel.get(a) == lo for a in dec) else []) + ([hi * 4] if any(sel.get(a) == hi for a in dec) else [])
    if extra:
        for g in tune_games:
            src, geo = geo_of(g)
            rows += ST.tune_rows(g, HEAD, data[g], SEEDS, geo, sources, src, GI, extra)
        kappas = sorted(kappas + extra)
        sel = ST.select_kappa(rows, kappas)
    sel["X_8"], sel["K*Beven"] = sel["X"], sel["K*A"]
    units, curves = [], []
    for g in unit_games:
        src, geo = geo_of(g)
        u = ST.unit_rows(g, HEAD, data[g], SEEDS, geo, GI, sel["F"], None)
        units += u
        if u[0]["eligible"]:
            curves += ST.arm_curve_rows(g, HEAD, data[g], SEEDS, geo, sources, src, GI, sel)
    res = ST.summarize(units, curves, comps=("F", "P", "U", "T_clock"))
    return sel, units, curves, res, rows


def cmd_planted():
    """T8-S: planted controls on the real expansion features (same generator as T8: 5-dim shared subspace,
    logit SD 1.5, base rate 0.15; null: independent directions). 12 games (seed 9708), kappa on 4, metric
    on 8. Pass: positive R(K*A:F) >= 2.0 and R(X:C*) >= 2.0; null R(X:F) <= 1.25 and R(X:C*) <= 1.1."""
    from scipy.optimize import brentq
    p = prov()
    res_all = []
    for scenario in ("positive", "null"):
        rng = np.random.default_rng(9708)
        pool = [GAMES[i] for i in sorted(rng.choice(len(GAMES), 12, replace=False))]
        tune = [pool[i] for i in sorted(rng.choice(12, 4, replace=False))]
        data = {g: load_game(g) for g in pool}
        geo_all = G.geometry({g: G.game_sums(data[g]) for g in pool}, pool)
        d = geo_all["d"]
        Vb = np.linalg.qr(rng.standard_normal((d, 5)))[0]
        for g in pool:
            w = Vb @ rng.standard_normal(5) if scenario == "positive" else rng.standard_normal(d)
            Z = [G.project(st["X"], geo_all) for st in data[g]]
            w = w * (1.5 / float(np.std(np.concatenate([z @ w for z in Z]))))
            lo_all = np.concatenate([z @ w for z in Z])
            b = brentq(lambda bb: float(np.mean(1 / (1 + np.exp(-(bb + lo_all))))) - 0.15, -30, 30)
            for s, st, z in zip(SEEDS, data[g], Z):
                pr = 1 / (1 + np.exp(-(b + z @ w)))
                st["y"]["good"] = (np.random.default_rng([9708, GI[g], s]).random(len(pr)) < pr).astype(np.int8)
        sel, units, curves, res, _ = run_pipeline(data, pool, tune, [g for g in pool if g not in tune], kinds=("real", "perm", "clock"))
        R = lambda k: None if res.get(k) is None else res[k]["R"]
        if scenario == "positive":
            ok = bool(R("K*A:F") and R("K*A:F") >= 2.0 and R("X:C*") and R("X:C*") >= 2.0)
        else:
            ok = bool(R("X:F") is not None and R("X:F") <= 1.25 and R("X:C*") is not None and R("X:C*") <= 1.1)
        res_all.append(dict(scenario=scenario, passed=ok, pool=pool, tune=tune, kappa=sel, n_units=res["n_units"],
                            R={k: R(k) for k in ("K*A:F", "X:C*", "X:F", "P:F", "U:F", "T_clock:F")}))
    save("planted_s.json", dict(provenance=p, results=res_all, passed=all(r["passed"] for r in res_all), claim_header=CLAIM))
    print(json.dumps([{k: r[k] for k in ("scenario", "passed", "n_units", "R")} for r in res_all], default=float))


def cmd_run():
    p = prov(OUT / "store.json", OUT / "planted_s.json")
    pl = json.load(open(OUT / "planted_s.json", encoding="utf-8"))
    if not pl["passed"]:
        save("g0s_verdict.json", dict(provenance=p, label="RECIPE-INSENSITIVE",
                                      sentence="the recipe cannot detect planted shared structure on the expansion features",
                                      consequence="KC-search-sense path: route S closed; v84 ends at feasibility",
                                      planted_sha256=full_hash(OUT / "planted_s.json"), claim_header=CLAIM))
        print("RECIPE-INSENSITIVE (T8-S failed)")
        return
    data = {g: load_game(g) for g in GAMES}
    sel, units, curves, res, rows = run_pipeline(data, GAMES, CALIBRATION, DEV)
    integ = all(all(r["integrity"].values()) for r in curves)
    n_games = res["n_games"]
    i_ok = n_games >= 12
    ii_ok = res["K*A:F"] is not None and res["K*A:F"]["R"] >= 2.0 and res["K*A:F"]["lo"] >= 1.25
    iii = res["X:C*"] is not None and res["X:C*"]["R"] >= 1.5 and res["X:C*"]["lo"] > 1.1 and res["X:C*"]["p"] <= 0.05
    iv = res["X:C*_late"] is not None and res["X:C*_late"]["lo"] > 1 and res["X:C*_late"]["p"] <= 0.05
    v_ok = res["ceiling_guard"] is not None and res["ceiling_guard"] >= -0.01
    if not integ:
        label = "Interruption"
    elif not i_ok:
        label = "KC-knowledge-room"
    elif not ii_ok:
        label = "KC-knowledge-worth"
    elif iii and iv and v_ok:
        label = "PASS"
    else:
        label = "KC-search-sense"
    with open(OUT / "curves_s.pkl", "wb") as f:
        pickle.dump(dict(units=units, curves=curves, tune=rows), f)
    res.update(label=label, conditions=dict(i=i_ok, ii=ii_ok, iii=iii, iv=iv, v=v_ok, vi=integ), kappa=sel,
               curves_sha256=full_hash(OUT / "curves_s.pkl"), claim_header=CLAIM, provenance=p)
    save("g0s_verdict.json", res)
    print(json.dumps({k: res[k] for k in ("label", "n_units", "n_games", "conditions")}))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dict(store=cmd_store, planted=cmd_planted, run=cmd_run)[sys.argv[1]]()
