"""G0 test T8: planted controls on the real feature geometry (docs/ARC4_V84_PROTOCOL_EN.md A.15 T8).

12 games drawn from DEV + CALIBRATION with default_rng(9708); their real kept rows (streams capped at
6,000 by the A.3 rule, drawn from the stored rows); synthetic labels, one planted head per game,
y ~ Bernoulli(sigmoid(b + x_hat^T w)) with the logit SD over that game's rows set to 1.5 and b giving a
base rate of 0.15 (labels default_rng([9708, game_index, seed])), the GOOD keep mask (t + 24 <= 100 and
avail >= 24, label-free). The planting basis x_hat is the A.5 geometry of all 12 games. Positive: w_g =
V a_g, V a Haar-random 5-dim orthonormal basis, a_g ~ N(0, I5). Null: each w_g Haar-random. Pipeline:
the G0 code (sources, A.9 kappa tuning on 4 games drawn with 9708, A.10 units, A.11-A.12 on the other 8).
Pass: positive R(K*A:F) >= 2.0 and R(X:C*) >= 2.0; null R(X:F) <= 1.25 and R(X:C*) <= 1.1.

  python scripts/g0_planted_v84.py
"""
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import g0_v84 as G                                 # noqa: E402
from ragnarok.evaluation import g0_stages_v84 as ST                         # noqa: E402
from ragnarok.evaluation.g0_pipeline_v84 import fit_sources                  # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                         # noqa: E402
from scripts.g0_v84 import GAMES, GI, OUT, SEEDS, load_game, save_json       # noqa: E402

CAP = 6000


def planted_dataset(scenario):
    rng = np.random.default_rng(9708)
    pool = [GAMES[i] for i in sorted(rng.choice(len(GAMES), 12, replace=False))]
    tune = [pool[i] for i in sorted(rng.choice(12, 4, replace=False))]
    data = {}
    for g in pool:
        streams = []
        for s, st in zip(SEEDS, load_game(g)):
            n = len(st["clock"])
            keep = np.arange(n) if n <= CAP else np.sort(np.random.default_rng([9701, GI[g], s]).choice(n, CAP, replace=False))
            streams.append({k: (v[keep] if k != "y" else None) for k, v in st.items()})
        data[g] = streams
    sums = {g: G.game_sums(data[g]) for g in pool}
    geo_all = G.geometry(sums, pool)
    d = geo_all["d"]
    Vb = np.linalg.qr(rng.standard_normal((d, 5)))[0]
    for g in pool:
        w = Vb @ rng.standard_normal(5) if scenario == "positive" else rng.standard_normal(d)
        Z_all = [G.project(st["X"], geo_all) for st in data[g]]
        sdl = float(np.std(np.concatenate([Z @ w for Z in Z_all])))
        w = w * (1.5 / sdl)
        lo_all = np.concatenate([Z @ w for Z in Z_all])
        b = brentq(lambda bb: float(np.mean(1 / (1 + np.exp(-(bb + lo_all))))) - 0.15, -30, 30)
        for s, st, Z in zip(SEEDS, data[g], Z_all):
            p = 1 / (1 + np.exp(-(b + Z @ w)))
            y = (np.random.default_rng([9708, GI[g], s]).random(len(p)) < p).astype(np.int8)
            mask = (st["t"].astype(int) + 24 <= 100) & (st["avail"].astype(int) >= 24)
            y[~mask] = -1
            st["y"] = dict(good=y, bad=np.full(len(y), -1, np.int8))
    return pool, tune, data


def run(scenario):
    pool, tune, data = planted_dataset(scenario)
    units_g = [g for g in pool if g not in tune]
    sums = {g: G.game_sums(data[g]) for g in pool}
    sources = fit_sources(data, pool, GI, SEEDS, heads=("good",))
    kappas = list(G.KAPPA_GRID)

    def geo_of(g):
        src = [h for h in pool if h != g]
        return src, G.geometry(sums, src)
    rows = []
    for g in tune:
        src, geo = geo_of(g)
        rows += ST.tune_rows(g, ("good",), data[g], SEEDS, geo, sources, src, GI, kappas)
    sel = ST.select_kappa(rows, kappas)
    extra = ST.edge_extension(sel, kappas)
    if extra:
        for g in tune:
            src, geo = geo_of(g)
            rows += ST.tune_rows(g, ("good",), data[g], SEEDS, geo, sources, src, GI, extra)
        kappas = sorted(kappas + extra)
        sel = ST.select_kappa(rows, kappas)
    sel["X_8"], sel["K*Beven"] = sel["X"], sel["K*A"]
    units, curves = [], []
    for g in units_g:
        src, geo = geo_of(g)
        u = ST.unit_rows(g, ("good",), data[g], SEEDS, geo, GI, sel["F"], None)
        units += u
        if u[0]["eligible"]:
            curves += ST.arm_curve_rows(g, ("good",), data[g], SEEDS, geo, sources, src, GI, sel)
    res = ST.summarize(units, curves)
    R = lambda k: None if res.get(k) is None else res[k]["R"]
    if scenario == "positive":
        ok = R("K*A:F") is not None and R("K*A:F") >= 2.0 and R("X:C*") is not None and R("X:C*") >= 2.0
    else:
        ok = R("X:F") is not None and R("X:F") <= 1.25 and R("X:C*") is not None and R("X:C*") <= 1.1
    return dict(scenario=scenario, pool=pool, tune=tune, kappa=sel, n_units=res["n_units"], passed=bool(ok),
                R={k: R(k) for k in ("K*A:F", "X:C*", "X:F", "P:F", "U:F", "X:C*_late")},
                lo={k: (res[k]["lo"] if res.get(k) else None) for k in ("K*A:F", "X:C*", "X:F")},
                units=[dict(game=u["game"], eligible=u["eligible"], ends=[d.get("auc_F_end") for d in u["dirs"]]) for u in units])


if __name__ == "__main__":
    prov = stamp()
    with Pool(2) as p:
        out = p.map(run, ["positive", "null"])
    passed = all(o["passed"] for o in out)
    save_json("planted.json", dict(provenance=prov, results=out, passed=passed))
    print(json.dumps([{k: o[k] for k in ("scenario", "passed", "n_units", "R")} for o in out]))
