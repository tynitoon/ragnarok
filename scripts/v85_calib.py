"""v85 calibration (docs/ARC4_V85_PROTOCOL_EN.md A.10.5-A.10.6, A.10.1(c), A.12), evaluator side.

On the I-stage V2s runs at the F1 rungs K_{g,s}, per simulation b (generator default_rng([9802, b])):
- comparator: for each (g, s), 3 run vectors drawn with replacement from that cell's V2s runs, each
  mean + centred residual * sqrt(3/2) in log T over the counted rungs;
- three steered rows: n/2 base vectors per (g, s), resampled the same way;
- planting: T_k <- T_k / D_eff, D_eff = D * exp(tau * z), z ~ N(0, 1) per (unit, row); D_late divides the gaps
  after the run's own M_{k_first} by D_eff with D = 2; D_jump sets T(M_{k_first}) to 5000, later gaps unchanged;
- extra run noise: every T of a planted run times exp(eps), eps ~ N(0, 0.5 * sigma2_{g,s}) per run;
- censoring: "stay" keeps a censored base T at the cap (a miss); "divide" plants it from the cap value
  (cap / D_eff for the D scenarios); a planted T at or above the cap is a miss;
- labels by the frozen verdict functions (scripts/v85_verdict.py: ssc_core, a11_clauses, a11), 1000 resamples.
Statistics are stored once per simulation and evaluated on the grid n in {4, 6}, tau_pt in {1.6, 1.7, 1.8, 2.0},
tau_lb in {1.15, 1.25, 1.35}; tau = 0.2 is primary (0 and 0.4 reported). Selection: the smallest n, then the
largest tau_pt, then the largest tau_lb meeting (i)-(viii); otherwise TOP-UP once, then UNDERPOWERED.

Choices where the text leaves room (each listed in the build result):
- the resampling is in log T; the residual inflation is sqrt(m / (m - 1)) for a cell with m runs, which is
  the frozen sqrt(3/2) at the I-stage's m = 3; a cell with fewer than 2 runs is refused; no monotone repair.
- a base element is censored when its resampled T is at or above the cap (an element from a run that missed
  that rung always is); sigma2_{g,s} = sample variance (ddof 1) of the cell's V2s lam.
- one simulation draws the comparator, 3 base runs per (row, cell), z, eps and the bootstrap draws once; they
  are shared by every n (the first n/2 base runs are used), tau and scenario (common random numbers).
- per-row probabilities average the 3 rows (exchangeable); (iv) is any of the 3 rows at D = 1.
- treatments: "stay" for P(ROOM | D2), P(NO-ROOM | D2), mean R(D2), mean R_late(D_late); "divide" for the
  false-ROOM rates (ii)-(v); (vii) P(NO-ROOM | D1), listed under neither, must hold under both (minimum).
  Every operating characteristic is reported under both treatments.
- D_jump plants no tau heterogeneity (the scenario fixes T(M_{k_first}) = 5000 and the later gaps).
- A.12 rerun: run noise variance max(0, v - 1) * sigma2 and tau = tau_K at the frozen (n, tau_pt, tau_lb).
- U for the S0 natural spike-ins: the caller passes the per-(g, s) vectors of natural G1b runs timed from their
  own M1 (cells need >= 2 runs, e.g. both reference runs of a unit in one cell).

  python scripts/v85_calib.py run [--sims 1000] [--topped-up]   # writes craft_v6_out/v85/calib.json
  python scripts/v85_calib.py f1 [--budget P] [--hashes P] [--closures P] [--seed-grep P]
  python scripts/v85_calib.py time [--sims 20]                  # runtime of a reduced run, extrapolated
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.v85_verdict import (CAP, CAP8, CLAIM, FLOOR, LOG15, OUT, SENSES, STARTS, a11, a11_clauses,  # noqa: E402
                                 boot_draws, canon, group, istage_kgs, kgs_from_json, kgs_to_json, lam_arrays,
                                 load_runs, run_times, ssc_core, stamp_out, _write)

SEED_SIM, SIMS, B_CAL = 9802, 1000, 1000
N_GRID, TAU_PT, TAU_LB = (4, 6), (1.6, 1.7, 1.8, 2.0), (1.15, 1.25, 1.35)
TAUS = (0.2, 0.0, 0.4)
SCEN = {"D2": 2.0, "D1.5": 1.5, "D1.25": 1.25, "D1": 1.0, "Dlate": 2.0, "Djump": None}
COMBOS = tuple((s, t) for s in SCEN for t in ("stay", "divide"))
N_COMP, N_ROWS = 3, 3
STAT = ("R", "LB90", "UB95", "UB95_8M", "R_late", "LB90_late", "breadth", "cens_a", "cens_c")
RUNTIME_LIMIT_S = 1800


def cells_from_runs(runs, K_gs, arm="v2s"):
    """{(g, s): array (m, |K_gs|) of T(M_k), NaN = miss} from the runs of one arm at the given rungs."""
    out = {}
    for (g, s), ks in sorted(K_gs.items()):
        rs = runs.get((g, s, canon(arm)))
        if rs and len(ks) >= 2:
            out[(g, s)] = np.array([[np.nan if t is None else t for t in run_times(r, ks)] for r in rs], float)
    return out


def prepare(cells, units, cap=CAP):
    """Pack the base cells (first units in the given order) for simulate()."""
    keys = [(g, s) for g in units for s in STARTS if (g, s) in cells]
    units = [g for g in units if any(k[0] == g for k in keys)]
    C = len(keys)
    K = max(np.asarray(cells[k]).shape[1] for k in keys)
    M = max(np.asarray(cells[k]).shape[0] for k in keys)
    X, miss = np.full((C, M, K), math.log(cap)), np.zeros((C, M, K), bool)
    kmask, m = np.zeros((C, K), bool), np.zeros(C, np.int64)
    for c, key in enumerate(keys):
        a = np.asarray(cells[key], float)
        if a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError(f"cell {key} needs >= 2 runs and >= 2 counted rungs")
        mm = np.isnan(a) | (a >= cap)
        X[c, :a.shape[0], :a.shape[1]] = np.log(np.where(mm, cap, a))
        miss[c, :a.shape[0], :a.shape[1]] = mm
        kmask[c, :a.shape[1]], m[c] = True, a.shape[0]
    rv = np.arange(M)[None, :] < m[:, None]                             # (C, M) real runs
    mean = np.where(rv[..., None], X, 0.0).sum(1) / m[:, None]
    lam = lam_arrays(np.exp(X), miss, kmask[:, None, :])[0]
    lm = np.where(rv, lam, 0.0).sum(1) / m
    sig2 = np.where(rv, (lam - lm[:, None]) ** 2, 0.0).sum(1) / (m - 1)
    top = np.zeros((C, K), bool)
    top[np.arange(C), kmask.sum(1) - 1] = True
    return dict(keys=keys, units=units, U=len(units), C=C, K=K, X=X, mean=mean, resid=X - mean[:, None, :],
                infl=np.sqrt(m / (m - 1.0)), m=m, kmask=kmask, top=top, sig2=sig2, cap=float(cap),
                u=np.array([units.index(g) for g, _ in keys]), s=np.array([STARTS.index(s) for _, s in keys]))


def resample(base, j):
    """Resampled vectors for source-run indices j (..., C, R) -> T, miss of shape (..., C, R, K)."""
    C = base["C"]
    r = base["resid"][np.arange(C)[:, None], j]                          # (..., C, R, K) via broadcasting
    x = base["mean"][:, None, :] + base["infl"][:, None, None] * r
    lc = math.log(base["cap"])
    miss = (x >= lc - 1e-12) & base["kmask"][:, None, :]
    return np.exp(np.minimum(x, lc)), miss


def plant(T, miss, scen, treat, deff, eps, cap=CAP):
    """A.10.5 planting. T, miss (..., C, R, K); deff (..., C) per (unit, row); eps (..., C, R) run noise."""
    D = deff[..., None, None]
    first = T[..., :1]
    if scen == "Djump":
        Tp = FLOOR + (T - first)
    elif scen == "Dlate":
        Tp = first + (T - first) / D
    else:
        Tp = T / D
    Tp = Tp * np.exp(eps[..., None])
    keep = miss if treat == "stay" else np.zeros_like(miss)
    Tp = np.where(keep, cap, Tp)
    return np.minimum(Tp, cap), keep | (Tp >= cap)


def _to_units(vals, base, N):
    """(M, C, >=N) cell values -> (M, U, 2, N)."""
    out = np.zeros((vals.shape[0], base["U"], 2, N))
    out[:, base["u"], base["s"], :] = vals[..., :N]
    return out


def simulate(base, sims=SIMS, n_list=N_GRID, taus=TAUS, combos=COMBOS, B=B_CAL, v=1.5, seed=SEED_SIM, b0=0):
    """Statistics (len(n_list), len(taus), sims, 3 rows, len(combos), len(STAT)) per A.10.5."""
    C, U, cap = base["C"], base["U"], base["cap"]
    km, top, m = base["kmask"][:, None, :], base["top"][:, None, :], base["m"]
    taus = np.asarray(taus, float)
    nt, ncb = len(taus), len(combos)
    out = np.zeros((len(n_list), nt, sims, N_ROWS, ncb, len(STAT)))
    noise = np.sqrt(max(v - 1.0, 0.0) * base["sig2"])                   # (C,)
    pairs = [(2 * i + d, c) for i in range(nt * ncb) for d, c in ((0, 0), (1, 1), (0, 2))]
    for b in range(sims):
        gen = np.random.default_rng([seed, b0 + b])
        jc = (gen.random((C, N_COMP)) * m[:, None]).astype(np.int64)
        jr = (gen.random((N_ROWS, C, 3)) * m[None, :, None]).astype(np.int64)
        z = gen.standard_normal((N_ROWS, U))
        e = gen.standard_normal((N_ROWS, C, 3))
        draws = [boot_draws(gen, B, U, 2, 3, N_COMP) for _ in range(N_ROWS)]
        Tc, mc = resample(base, jc)
        lam_c, late_c = lam_arrays(Tc, mc, km)
        lam8 = lam_arrays(Tc, mc, km, CAP8)[0]
        cens_c = float((mc & top).any(-1).mean())
        lc = _to_units(np.stack([lam_c, late_c, lam8]), base, N_COMP)
        Tr, mr = resample(base, jr)                                       # (3, C, 3, K)
        hz = np.exp(taus[:, None, None] * z[None])[..., base["u"]]       # (nt, 3, C)
        eps = e * noise[None, :, None]
        L = np.zeros((nt, ncb, 2, N_ROWS, C, 3))
        cens = np.zeros((nt, ncb, N_ROWS, C, 3), bool)
        for ci, (sc, tr) in enumerate(combos):
            deff = (SCEN[sc] or 1.0) * (hz if SCEN[sc] else np.ones_like(hz))
            Tp, mp = plant(Tr, mr, sc, tr, deff, eps, cap)
            L[:, ci, 0], L[:, ci, 1] = lam_arrays(Tp, mp, km)
            cens[:, ci] = (mp & top).any(-1)
        for ni, n in enumerate(n_list):
            h = n // 2
            na = np.zeros((U, 2), np.int64)
            na[base["u"], base["s"]] = h
            nc = np.where(na > 0, N_COMP, 0)
            for r in range(N_ROWS):
                la = _to_units(L[:, :, :, r].reshape(nt * ncb * 2, C, 3), base, h)
                _, lr, lR, bR = ssc_core(la, lc, na, nc, na > 0, draws[r], pairs)
                q = np.percentile(np.exp(bR), (10.0, 95.0), axis=-1).reshape(2, nt, ncb, 3)
                lR, lr = lR.reshape(nt, ncb, 3), lr.reshape(nt, ncb, 3, U)
                o = out[ni, :, b, r]
                o[..., 0], o[..., 1], o[..., 2], o[..., 3] = np.exp(lR[..., 0]), q[0, ..., 0], q[1, ..., 0], q[1, ..., 2]
                o[..., 4], o[..., 5] = np.exp(lR[..., 1]), q[0, ..., 1]
                o[..., 6] = (lr[..., 0, :] >= LOG15 - 1e-12).mean(-1)
                o[..., 7], o[..., 8] = cens[:, :, r, :, :h].mean((-1, -2)), cens_c
    return out


def operating(st, combos, tau_pt, tau_lb):
    """Operating characteristics at (tau_pt, tau_lb) from stored statistics st (sims, 3, combos, STAT)."""
    g = {k: st[..., i] for i, k in enumerate(STAT)}
    cl = a11_clauses(g["R"], g["LB90"], g["LB90_late"], g["breadth"], g["cens_a"], g["cens_c"], g["UB95"],
                     g["UB95_8M"], tau_pt, tau_lb)
    room, noroom, _ = a11(cl)
    name = [f"{s}/{t}" for s, t in combos]
    return dict(P_room=dict(zip(name, room.mean((0, 1)).tolist())), P_noroom=dict(zip(name, noroom.mean((0, 1)).tolist())),
                P_any_room=dict(zip(name, room.any(1).mean(0).tolist())),
                mean_R=dict(zip(name, g["R"].mean((0, 1)).tolist())), mean_R_late=dict(zip(name, g["R_late"].mean((0, 1)).tolist())))


def criteria(oc):
    """A.10.5 (i)-(viii) from the operating characteristics (treatments per the module docstring)."""
    P, N, A, R, RL = oc["P_room"], oc["P_noroom"], oc["P_any_room"], oc["mean_R"], oc["mean_R_late"]
    return dict(i=P["D2/stay"] >= 0.80, ii=P["D1.5/divide"] <= 0.25, iii=P["D1.25/divide"] <= 0.10,
                iv=A["D1/divide"] <= 0.05, v=P["Djump/divide"] <= 0.05, vi=N["D2/stay"] <= 0.05,
                vii=min(N["D1/stay"], N["D1/divide"]) >= 0.80,
                viii=bool(1.8 <= R["D2/stay"] <= 2.2 and 1.8 <= RL["Dlate/stay"] <= 2.2))


def select(table):
    """The smallest n, then the largest tau_pt, then the largest tau_lb meeting every criterion; None if none."""
    ok = [t for t in table if all(t["criteria"].values())]
    return min(ok, key=lambda t: (t["n"], -t["tau_pt"], -t["tau_lb"])) if ok else None


def calibrate(base, sims=SIMS, B=B_CAL, topped_up=False):
    t0 = time.time()
    st = simulate(base, sims, N_GRID, TAUS, COMBOS, B)
    table = []
    for ni, n in enumerate(N_GRID):
        for ti, tau in enumerate(TAUS):
            for tp in TAU_PT:
                for tl in TAU_LB:
                    oc = operating(st[ni, ti], COMBOS, tp, tl)
                    cr = criteria(oc)
                    table.append(dict(n=n, tau=float(tau), tau_pt=tp, tau_lb=tl, oc=oc,
                                      criteria={k: bool(v) for k, v in cr.items()}))
    sel = select([t for t in table if t["tau"] == 0.2])
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.5", units=base["units"], U=base["U"],
                cells=[f"{g}/{s}" for g, s in base["keys"]], sims=sims, B=B, seeds=f"[{SEED_SIM}, b], b < {sims}",
                taus=list(TAUS), tau_primary=0.2, grid=dict(n=list(N_GRID), tau_pt=list(TAU_PT), tau_lb=list(TAU_LB)),
                table=table, selection=sel, topped_up=topped_up,
                outcome="SELECTED" if sel else ("UNDERPOWERED" if topped_up else "TOP-UP"),
                runtime_s=time.time() - t0)


def rerun_a12(cells, units, thr, v, tau, sims=SIMS, B=B_CAL):
    """A.12: the frozen calibration at (v_K, tau_K) for one row, at the frozen (n, tau_pt, tau_lb)."""
    base = prepare(cells, units)
    st = simulate(base, sims, (thr["n"],), (tau,), COMBOS, B, v=v)
    oc = operating(st[0, 0], COMBOS, thr["tau_pt"], thr["tau_lb"])
    cr = criteria(oc)
    return dict(v=v, tau=tau, oc=oc, passed={k: bool(cr[k]) for k in ("ii", "iii", "iv", "v", "vi")})


def p_room_x2(cells, units, n=4, tau=0.2, tau_pt=1.75, tau_lb=1.25, sims=SIMS, B=B_CAL):
    """A.10.1(c): P(ROOM | x2) at n, tau and thresholds tau_pt / tau_lb, for the given units (first U in order)."""
    base = prepare(cells, units)
    cb = (("D2", "stay"),)
    st = simulate(base, sims, (n,), (tau,), cb, B)
    return operating(st[0, 0], cb, tau_pt, tau_lb)["P_room"]["D2/stay"]


def assemble_f1(calib, K_gs, budget=None, hashes=None, closures=None, seed_grep=None, provenance=None):
    """A.10.6 f1_thresholds.json; budget, hashes, closures and seed grep come from the stage scripts.
    §13.1 O_4 rule: if budget["projection_frames"] > 0.6G, O_4 runs at n = 4 whatever n is chosen."""
    sel = calib.get("selection") or {}
    n = sel.get("n")
    proj = (budget or {}).get("projection_frames")
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.6 (F1)", outcome=calib["outcome"], U=calib["U"],
                units=calib["units"], K_gs=kgs_to_json(K_gs), n=n, tau_pt=sel.get("tau_pt"), tau_lb=sel.get("tau_lb"),
                operating_characteristics=sel.get("oc"), criteria=sel.get("criteria"), calibration=calib,
                budget=budget, n_O4=(4 if proj is not None and proj > 0.6e9 else n) if n else None,
                hashes=hashes, closure_hashes=closures, seed_grep=seed_grep, claim_header=CLAIM, senses=SENSES,
                provenance=provenance)


def synthetic_cells(U=16, K=5, m=3, seed=9800):
    """Synthetic V2s-like cells for the runtime check (seed 9800: tests and smoke only)."""
    rng = np.random.default_rng([seed, 85])
    cells = {}
    for u in range(U):
        for s in STARTS:
            gaps = np.exp(rng.normal(11.5, 0.6, size=(m, K)))
            cells[(f"u{u}", s)] = np.cumsum(gaps, 1)
    return cells, [f"u{u}" for u in range(U)]


def time_run(sims=20, U=16, K=5):
    """Runtime of `sims` simulations over the full grid (both n, three tau, every scenario), extrapolated."""
    cells, units = synthetic_cells(U, K)
    base = prepare(cells, units)
    t0 = time.time()
    simulate(base, sims)
    dt = time.time() - t0
    return dict(sims=sims, seconds=dt, extrapolated_s=dt * SIMS / sims, limit_s=RUNTIME_LIMIT_S)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "f1", "time"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--topped-up", action="store_true")
    for k in ("budget", "hashes", "closures", "seed-grep"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if a.cmd == "time":
        print(json.dumps(time_run(a.sims or 20)))
        return
    ist, paths = load_runs(out / "istage")
    runs = group(ist)
    iv = out / "istage" / "istage_verdict.json"
    K_gs = kgs_from_json(json.load(open(iv, encoding="utf-8"))["K_gs"]) if iv.exists() else istage_kgs(runs)[0]
    if a.cmd == "run":
        from ragnarok.evaluation.start_v85 import unit_order
        units = unit_order({g for g, s in K_gs})
        cal = calibrate(prepare(cells_from_runs(runs, K_gs), units), a.sims or SIMS, B_CAL, a.topped_up)
        cal.update(claim_header=CLAIM, senses=SENSES, provenance=stamp_out(paths + ([iv] if iv.exists() else [])))
        _write(out / "calib.json", cal)
        print(json.dumps(dict(outcome=cal["outcome"], selection=cal["selection"] and {
            k: cal["selection"][k] for k in ("n", "tau_pt", "tau_lb")}, runtime_s=cal["runtime_s"])))
        return
    cal = json.load(open(out / "calib.json", encoding="utf-8"))
    if cal["outcome"] != "SELECTED":
        sys.exit(f"calibration outcome {cal['outcome']}: no thresholds, G1 does not run")
    ld = {k: (json.load(open(getattr(a, k.replace('-', '_')), encoding="utf-8")) if getattr(a, k.replace('-', '_')) else None)
          for k in ("budget", "hashes", "closures", "seed-grep")}
    f1 = assemble_f1(cal, K_gs, ld["budget"], ld["hashes"], ld["closures"], ld["seed-grep"],
                     stamp_out([out / "calib.json"] + [p for p in (a.budget, a.hashes, a.closures, a.seed_grep) if p]))
    _write(out / "f1_thresholds.json", f1)
    print(json.dumps({k: f1[k] for k in ("U", "n", "n_O4", "tau_pt", "tau_lb")}))


if __name__ == "__main__":
    main()
