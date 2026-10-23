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
PATHS = ("ragnarok", "scripts", "tests")                 # code trees of the clean-tree check (A.13 vi)


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
    if isinstance(obj, dict) and "provenance" in obj:            # git state when the output is written
        obj = dict(obj, provenance_end=stamp((), cwd=str(ROOT), paths=PATHS))
    p = OUT / name
    with open(str(p) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(str(p) + ".tmp", p)


def prov_v84(*inputs, strict=True):
    """A.2 provenance: git state of this worktree over ragnarok/, scripts/, tests/; full SHA-256 of every
    input. A stage refuses to start on an unclean or unknown tree."""
    p = stamp(inputs, cwd=str(ROOT), paths=PATHS, full=True)
    if strict and (not p["code_sha"] or p["dirty"]):
        sys.exit(f"Interruption: code tree not clean or unknown (HEAD {p['code_sha'] or '?'}, dirty {p['dirty']})")
    return p


def _load(name):
    p = OUT / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _same_tree(out, prov):
    ps = [out.get("provenance"), out.get("provenance_end")] if out else [None]
    return all(q is not None and q.get("dirty") is False and q.get("code_tree") == prov["code_tree"] for q in ps)


def tests_status(prov):
    """(vi) tests T1-T9 passed (T8 included): tests_record.json written by `tests` on this clean tree (pytest
    exit 0, nothing skipped, a T9 review reference); planted.json from the same tree, passed, unchanged."""
    t, p = _load("tests_record.json"), _load("planted.json")
    if t is None or p is None:
        return False, ["tests_record.json or planted.json missing"]
    why = []
    if t.get("pytest_rc") != 0 or t.get("pytest_skipped") != 0 or not t.get("pytest_passed"):
        why.append(f"T1-T7: {t.get('pytest_summary')}")
    if not t.get("t9_review"):
        why.append("T9: no review reference")
    if p.get("passed") is not True:
        why.append("T8 not passed")
    if t.get("planted_sha256") != full_hash(OUT / "planted.json"):
        why.append("planted.json changed after the tests record")
    for name, q in (("tests_record.json", t), ("planted.json", p)):
        if not _same_tree(q, prov):
            why.append(f"{name} not from this clean code tree")
    return not why, why


def chain_status(prov):
    """(vi) hashes match and clean tree: every stage output on this clean tree; every recorded input hash
    equals the file now; store files equal store.json."""
    why = [] if (prov["code_sha"] and not prov["dirty"]) else ["verdict run on an unclean tree"]
    for name in ("sums.json", "sources.json", "g0_kappa.json", "g0_units.json", "curves.json"):
        q = _load(name)
        if q is None:
            why.append(f"{name} missing")
            continue
        if not _same_tree(q, prov):
            why.append(f"{name} not from this clean code tree")
        for path, h in q["provenance"].get("inputs", {}).items():
            if not os.path.exists(path) or full_hash(path) != h:
                why.append(f"{name}: input changed since the stage ran: {path}")
    st = json.load(open(STORE / "store.json", encoding="utf-8"))
    if st["provenance"].get("dirty") is not False:
        why.append("store built from an unclean tree")
    for s in st["streams"]:
        if full_hash(STORE / f"{s['game']}_{s['seed']}.npz") != s["file_sha256"]:
            why.append(f"store file differs from store.json: {s['game']}_{s['seed']}")
    return not why, why


def require_tests(prov):
    ok, why = tests_status(prov)
    if not ok:
        sys.exit("T1-T9 not passed on this code tree, no real-label computation (A.15): " + "; ".join(why))


def cmd_tests(review_ref):
    """tests_record.json from machine results (A.15): pytest tests/test_g0_v84.py (nothing skipped), the hash
    of planted.json (T8), the T9 review reference."""
    import re
    import subprocess
    prov = prov_v84(OUT / "planted.json")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests/test_g0_v84.py"],
                       cwd=str(ROOT), capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    tail = lines[-1] if lines else ""

    def count(word):
        m = re.search(rf"(\d+) {word}", tail)
        return int(m.group(1)) if m else 0
    save_json("tests_record.json", dict(provenance=prov, pytest_rc=r.returncode, pytest_passed=count("passed"),
                                        pytest_skipped=count("skipped"), pytest_summary=tail,
                                        planted_sha256=full_hash(OUT / "planted.json"), t9_review=review_ref,
                                        claim_header=CLAIM, senses=SENSES))
    print(r.returncode, tail)


def setup(g):
    sums = load_sums()
    src = [h for h in GAMES if h != g]
    return src, G.geometry(sums, src), load_game(g)


# ---------------------------------------------------------------- stages
def cmd_sums():
    prov = prov_v84(STORE / "store.json")
    out = {}
    for g in GAMES:
        q = G.game_sums(load_game(g))
        out[f"{g}_n"], out[f"{g}_s"], out[f"{g}_ss"] = q["n"], q["s"], q["ss"]
    np.savez(OUT / "sums.npz", **out)
    save_json("sums.json", dict(provenance=prov, sums_sha256=full_hash(OUT / "sums.npz"), claim_header=CLAIM, senses=SENSES))
    print("sums ok")


def _sources_one(g):
    return fit_sources({g: load_game(g)}, [g], GI, SEEDS)


def cmd_sources(workers):
    prov = prov_v84(STORE / "store.json")
    require_tests(prov)
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
                                   sources_sha256=full_hash(OUT / "sources.pkl"), claim_header=CLAIM, senses=SENSES))


def _tune_one(args):
    g, kappas = args
    src, geo, streams = setup(g)
    return ST.tune_rows(g, G.HEADS, streams, SEEDS, geo, load_sources(), src, GI, kappas)


def cmd_tune(workers):
    prov = prov_v84(STORE / "store.json", OUT / "sums.npz", OUT / "sources.pkl")
    require_tests(prov)
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
    prov = prov_v84(STORE / "store.json", OUT / "sums.npz", OUT / "g0_kappa.json")
    require_tests(prov)
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
    prov = prov_v84(STORE / "store.json", OUT / "sums.npz", OUT / "sources.pkl", OUT / "g0_kappa.json", OUT / "g0_units.json")
    require_tests(prov)
    kap = json.load(open(OUT / "g0_kappa.json", encoding="utf-8"))["kappa"]
    units = json.load(open(OUT / "g0_units.json", encoding="utf-8"))
    todo = {}
    for u in units["units"]:
        if u["eligible"]:
            todo.setdefault(u["game"], []).append(u["head"])
    with Pool(workers) as pool:
        rows = [r for rs in pool.map(_arms_one, [(g, hs, kap) for g, hs in todo.items()]) for r in rs]
    save_json("curves.json", dict(provenance=prov, units_sha256=full_hash(OUT / "g0_units.json"), rows=rows,
                                  claim_header=CLAIM, senses=SENSES))
    print("curves", len(rows))


def cmd_verdict():
    prov = prov_v84(STORE / "store.json", *(OUT / n for n in ("sums.npz", "sources.pkl", "g0_kappa.json", "g0_units.json",
                                                              "curves.json", "planted.json", "tests_record.json")), strict=False)
    units = json.load(open(OUT / "g0_units.json", encoding="utf-8"))
    cur = json.load(open(OUT / "curves.json", encoding="utf-8"))["rows"]
    res = ST.summarize(units["units"], cur)
    f_prefix = all(d.get("prefix_ok", False) for u in units["units"] if u["eligible"] for d in u["dirs"])
    integ = all(all(r["integrity"].values()) for r in cur) and f_prefix
    t_ok, t_why = tests_status(prov)
    h_ok, h_why = chain_status(prov)
    tests_ok = t_ok and h_ok
    res["vi_failures"] = t_why + h_why + ([] if integ else ["trace, spectrum or prefix certificate failed"])
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
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "tests":                       # python scripts/g0_v84.py tests <T9 review reference>
        cmd_tests(sys.argv[2])
        sys.exit(0)
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    dict(sums=cmd_sums, sources=lambda: cmd_sources(w), tune=lambda: cmd_tune(w), units=lambda: cmd_units(w),
         arms=lambda: cmd_arms(w), verdict=cmd_verdict)[cmd]()
