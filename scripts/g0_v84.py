"""G0 driver (docs/ARC4_V84_PROTOCOL_EN.md A.16): sums, sources, tune, units, arms, verdict.

  python scripts/g0_v84.py sums
  python scripts/g0_v84.py sources [workers]
  python scripts/g0_v84.py tune [workers]          -> g0_kappa.json (commit before any development curve)
  python scripts/g0_v84.py units [workers]         -> g0_units.json (commit before any non-F development curve)
  python scripts/g0_v84.py arms [workers]          -> curves.json
  python scripts/g0_v84.py verdict                 -> g0_verdict.json
"""
import json
import os
import pathlib
import pickle
import sys
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation import g0_v84 as G                                       # noqa: E402
from ragnarok.evaluation import g0_stages_v84 as ST                               # noqa: E402
from ragnarok.evaluation.g0_pipeline_v84 import fit_sources                        # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV                         # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp                    # noqa: E402

OUT = ROOT / "craft_v6_out" / "v84" / "g0"
STORE = OUT / "store"
GAMES = DEV + CALIBRATION
GI = {g: i for i, g in enumerate(GAMES)}
SEEDS = (9131, 9132)
CLAIM = ("Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 "
         "(score rises, lives count). It is not live play without savestates.")
SENSES = "pixels, own buttons, max(0, r_step) per step, ale.lives(); game over only as an episode boundary"


def load_game(g):
    out = []
    for s in SEEDS:
        z = np.load(STORE / f"{g}_{s}.npz")
        out.append(dict(X=z["X"], y=dict(good=z["y_good"], bad=z["y_bad"]), clock=z["clock"], block=z["block"],
                        it=z["it"], hash=z["hash"], slot=z["slot"], t=z["t"], avail=z["avail"]))
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


def require_tests():
    if not (OUT / "tests_passed.json").exists():
        sys.exit("T1-T9 have not passed: no real-label computation (A.15)")


def setup(g):
    sums = load_sums()
    src = [h for h in GAMES if h != g]
    return src, G.geometry(sums, src), load_game(g)


# ---------------------------------------------------------------- stages
def cmd_sums():
    out = {}
    for g in GAMES:
        q = G.game_sums(load_game(g))
        out[f"{g}_n"], out[f"{g}_s"], out[f"{g}_ss"] = q["n"], q["s"], q["ss"]
    np.savez(OUT / "sums.npz", **out)
    print("sums ok")


def _sources_one(g):
    return fit_sources({g: load_game(g)}, [g], GI, SEEDS)


def cmd_sources(workers):
    require_tests()
    prov = stamp()
    res = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_sources_one, GAMES):
            res.update(r)
            print("sources", sorted({k[0] for k in r}), flush=True)
    with open(OUT / "sources.pkl", "wb") as f:
        pickle.dump(res, f)
    save_json("sources.json", dict(provenance=prov,
                                   included={f"{k[0]}|{k[1]}|{k[2]}": v is not None for k, v in res.items()},
                                   rho={f"{k[0]}|{k[1]}|{k[2]}": (v["rho"] if v else None) for k, v in res.items()},
                                   sources_sha256=full_hash(OUT / "sources.pkl")))


def _tune_one(args):
    g, kappas = args
    src, geo, streams = setup(g)
    return ST.tune_rows(g, G.HEADS, streams, SEEDS, geo, load_sources(), src, GI, kappas)


def cmd_tune(workers):
    require_tests()
    prov = stamp()
    kappas = list(G.KAPPA_GRID)
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_tune_one, [(c, kappas) for c in CALIBRATION]) for r in rs]
    sel = ST.select_kappa(rows, kappas)
    extra = ST.edge_extension(sel, kappas)
    if extra:
        with Pool(workers) as pool:
            rows += [r for rs in pool.map(_tune_one, [(c, extra) for c in CALIBRATION]) for r in rs]
        kappas = sorted(kappas + extra)
        sel = ST.select_kappa(rows, kappas)
    sel["X_8"], sel["K*Beven"] = sel["X"], sel["K*A"]
    save_json("g0_kappa.json", dict(provenance=prov, kappa=sel, grid=kappas, extended=extra or None, rows=rows,
                                    sources_sha256=full_hash(OUT / "sources.pkl"), claim_header=CLAIM, senses=SENSES))
    print(json.dumps(sel))


def _units_one(args):
    g, kF = args
    src, geo, streams = setup(g)
    return ST.unit_rows(g, G.HEADS, streams, SEEDS, geo, GI, kF, G.outside_share(np.vstack([s["X"] for s in streams]), geo))


def cmd_units(workers):
    require_tests()
    prov = stamp()
    kap = json.load(open(OUT / "g0_kappa.json", encoding="utf-8"))
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_units_one, [(g, kap["kappa"]["F"]) for g in DEV]) for r in rs]
    save_json("g0_units.json", dict(provenance=prov, kappa_sha256=full_hash(OUT / "g0_kappa.json"), units=rows,
                                    n_eligible=sum(r["eligible"] for r in rows),
                                    games_eligible=sorted({r["game"] for r in rows if r["eligible"]}),
                                    claim_header=CLAIM, senses=SENSES))
    print("eligible", sum(r["eligible"] for r in rows), "games", len({r["game"] for r in rows if r["eligible"]}))


def _arms_one(args):
    g, heads, kap = args
    src, geo, streams = setup(g)
    return ST.arm_curve_rows(g, heads, streams, SEEDS, geo, load_sources(), src, GI, kap)


def cmd_arms(workers):
    require_tests()
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


def cmd_verdict():
    prov = stamp()
    units = json.load(open(OUT / "g0_units.json", encoding="utf-8"))
    cur = json.load(open(OUT / "curves.json", encoding="utf-8"))["rows"]
    res = ST.summarize(units["units"], cur)
    integ = all(all(r["integrity"].values()) for r in cur)
    tests_ok = (OUT / "tests_passed.json").exists()
    c = res
    i_ok = c["n_units"] >= 12 and c["n_games"] >= 8
    ii_ok = c["K*A:F"] is not None and c["K*A:F"]["R"] >= 2.0 and c["K*A:F"]["lo"] >= 1.25
    iii = c["X:C*"] is not None and c["X:C*"]["R"] >= 1.5 and c["X:C*"]["lo"] > 1.1 and c["X:C*"]["p"] <= 0.05
    iv = c["X:C*_late"] is not None and c["X:C*_late"]["lo"] > 1 and c["X:C*_late"]["p"] <= 0.05
    v_ok = c["ceiling_guard"] is not None and c["ceiling_guard"] >= -0.01
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
        xf, pf, uf = c.get("X:F"), c.get("P:F"), c.get("U:F")
        if xf and xf["lo"] > 1 and ((pf and pf["lo"] > 1) or (uf and uf["lo"] > 1)):
            caps.append("GENERIC-SHRINKAGE")
        if iii and not iv:
            caps.append("FIRST-RULES-ONLY")
        if c["X:C*"] and c["X:C*"]["lo"] > 1 and not iii:
            caps.append("SUB-THRESHOLD")
        if pf and pf["hi"] < 1:
            caps.append("WEAK-PLACEBO")
    res.update(conditions=dict(i=i_ok, ii=ii_ok, iii=iii, iv=iv, v=v_ok, vi=bool(integ and tests_ok)),
               label=label, captions=caps, claim_header=CLAIM, senses=SENSES,
               inputs=dict(units_sha256=full_hash(OUT / "g0_units.json"), curves_sha256=full_hash(OUT / "curves.json"),
                           kappa_sha256=full_hash(OUT / "g0_kappa.json"), sources_sha256=full_hash(OUT / "sources.pkl")),
               provenance=prov)
    save_json("g0_verdict.json", res)
    print(json.dumps({k: res[k] for k in ("label", "captions", "n_units", "n_games", "conditions")}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    OUT.mkdir(parents=True, exist_ok=True)
    dict(sums=cmd_sums, sources=lambda: cmd_sources(w), tune=lambda: cmd_tune(w), units=lambda: cmd_units(w),
         arms=lambda: cmd_arms(w), verdict=cmd_verdict)[cmd]()
