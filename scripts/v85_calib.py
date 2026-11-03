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
- `run` needs a complete I-stage verdict (istage/istage_verdict.json with K_gs; A.10.4 is never recomputed
  here) and calibrates on cells of exactly 3 V2s runs (A.10.5 "3 run vectors"), units in the A.2 order of
  s0/starts.json (calib.json "units" feeds f1 "units", K~mac's derangement).
- §13.1 budget projection (f1): the cost of the G1 core rows, in emulated frames, projected from the I-stage
  mean V2s billed per run (a run that was not evaluator-stopped counts at the 4M cap): mean billed x
  sum over kept (g, s) of n/2 x (5 for O_4, whose 4 look-aheads emulate 4x its billed frames, + 1 for K*map
  + 1 for K*mac where the other start is valid). Above 0.6G, O_4 runs at n = 4 (n_O4).
- closures (A.10.6, §13.4): per arm type (every module of arms_v85: the G1 arms and, once written, the family
  arms of Part B item 12), `scripts/v85_job.py UNIT A ARM SEED --closure-only` in its own light process (one at
  a time), on the first kept unit with both starts valid: the closure hash of the job built for that arm (the
  same import graph as a run). Written by `closures` (before the I-stage, as craft_v6_out/v85/closures.json,
  which the I-stage verdict checks V2s and D against; `--out <sealed root>` at F2) and recomputed by `f1`
  unless --closures gives a file written after the last code commit.
- f1 refuses unless the I-stage stage is PASS, calib.json is SELECTED, and closures (every gated arm type),
  hashes (--hashes: "maps" and "libraries" from `inputs`, "family_code", "development_objects", "kappa" from
  scripts/v85_objects.py) and the seed grep (--seed-grep, v85_stage0 grep) are present. The s1 and entry-cache
  SHA-256 of the F1 cells come from s0/starts.json.

  python scripts/v85_calib.py run [--sims 1000] [--topped-up]   # writes craft_v6_out/v85/calib.json
  python scripts/v85_calib.py closures                          # writes craft_v6_out/v85/closures.json
  python scripts/v85_calib.py inputs                            # writes craft_v6_out/v85/f1_inputs.json
  python scripts/v85_calib.py f1 --hashes P [P ..] --seed-grep P [--closures P]
  python scripts/v85_calib.py time [--sims 20]                  # runtime of a reduced run, extrapolated
"""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")       # before numpy: BLAS reads it when it loads (§13.2)
import numpy as np                                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.v85_verdict import (CAP, CAP8, CLAIM, FLOOR, LOG15, OUT, SENSES, STARTS, a11, a11_clauses,  # noqa: E402
                                 a2_order, boot_draws, canon, group, kgs_from_json, kgs_to_json, lam_arrays,
                                 load_runs, run_times, ssc_core, stamp_out, _write)

GATE_ARMS = ("v2s", "d", "o4", "kmap", "kmac", "o4_dice", "kmac_exact", "gf4", "gf", "kmac_tilde")
ARM_SEED = {"v2s": 9806, "o4": 9812, "kmap": 9812, "kmac": 9812, "o4_dice": 9818, "kmac_exact": 9818, "gf4": 9812,
            "gf": 9812, "kmac_tilde": 9812}
HASH_KEYS = ("maps", "libraries", "family_code", "development_objects", "kappa")
O_FACTOR, BUDGET_G = 5.0, 0.6e9

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


def prepare(cells, units, cap=CAP, require_m=None):
    """Pack the base cells (first units in the given order) for simulate(). require_m: every cell must have
    exactly that many runs (3 for the I-stage V2s cells, A.10.5)."""
    keys = [(g, s) for g in units for s in STARTS if (g, s) in cells]
    if require_m is not None:
        bad = [f"{g}/{s}: {np.asarray(cells[(g, s)]).shape[0]}" for g, s in keys if np.asarray(cells[(g, s)]).shape[0] != require_m]
        if bad:
            raise ValueError(f"cells need exactly {require_m} V2s runs (A.10.5): {bad}")
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


def rerun_a12(cells, units, thr, v, tau, sims=SIMS, B=B_CAL, require_m=3):
    """A.12: the frozen calibration at (v_K, tau_K) for one row, at the frozen (n, tau_pt, tau_lb) (O_4: n_O4)."""
    base = prepare(cells, units, require_m=require_m)
    st = simulate(base, sims, (thr["n"],), (tau,), COMBOS, B, v=v)
    oc = operating(st[0, 0], COMBOS, thr["tau_pt"], thr["tau_lb"])
    cr = criteria(oc)
    return dict(v=v, tau=tau, oc=oc, passed={k: bool(cr[k]) for k in ("ii", "iii", "iv", "v", "vi")})


def p_room_x2(cells, units, n=4, tau=0.2, tau_pt=1.75, tau_lb=1.25, sims=SIMS, B=B_CAL, seed=SEED_SIM):
    """A.10.1(c): P(ROOM | x2) at n, tau and thresholds tau_pt / tau_lb, for the given units (first U in order);
    simulations on [seed, b] (9802, §12; 9800 in tests)."""
    base = prepare(cells, units)
    cb = (("D2", "stay"),)
    st = simulate(base, sims, (n,), (tau,), cb, B, seed=seed)
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


def budget_projection(istage_runs, K_gs, starts, n, cap=CAP):
    """§13.1 at F1 (module docstring): G1 core cost in emulated frames from the I-stage mean V2s billed per run
    (a run not stopped by the evaluator counts at the cap); O_4 at n = 4 if the projection exceeds 0.6G."""
    v = [r for r in istage_runs if canon(r["arm"]) == "v2s"]
    if not v or not n:
        return None
    billed = [float(cap) if (not r.get("stopped") or r.get("billed") is None) else min(float(cap), float(r["billed"]))
              for r in v]
    mb = float(np.mean(billed))
    cells = [(g, s) for (g, s), ks in sorted(K_gs.items()) if len(ks) >= 2]
    kmac = [(g, s) for g, s in cells if ((starts["units"][g].get("B" if s == "A" else "A")) or {}).get("valid")]

    def proj(n_o4):
        return mb * ((n_o4 // 2) * O_FACTOR * len(cells) + (n // 2) * (len(cells) + len(kmac)))
    p = proj(n)
    n_o4 = 4 if p > BUDGET_G else n
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §13.1 budget rule (G1)", mean_v2s_billed=mb, n_v2s_runs=len(v),
                censored_at_cap=int(sum(1 for r in v if not r.get("stopped"))), cells=len(cells), kmac_cells=len(kmac),
                n=int(n), o_factor=O_FACTOR, projection_frames=p, threshold=BUDGET_G, n_O4=int(n_o4),
                projection_at_n_O4=proj(n_o4))


def closure_unit(starts):
    """The first kept unit in A.2 order with both starts valid (K*mac and K~mac build there)."""
    kept = a2_order(starts, starts["order"])
    ok = [g for g in kept if all((starts["units"][g].get(s) or {}).get("valid") for s in STARTS)]
    if not ok:
        raise SystemExit("no kept unit with both starts valid: closures cannot be built")
    return ok[0]


def _job_closure(v85, unit, arm, seed):
    """One light process: scripts/v85_job.py --closure-only (the run's own entry point and imports)."""
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "v85_job.py"), unit, "A", arm, str(seed), "--v85",
                        str(v85), "--closure-only"], cwd=str(ROOT), env=env, capture_output=True, text=True,
                       timeout=1800, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(f"closure of {arm}: exit {r.returncode}: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def arm_modules():
    """Every arm module of ragnarok/evaluation/arms_v85 (the G1 arms, and the family arms of Part B item 12)."""
    return tuple(sorted(p.stem for p in (ROOT / "ragnarok" / "evaluation" / "arms_v85").glob("*.py")
                        if p.stem != "__init__"))


def arm_closures(v85=OUT, arms=None, runner=_job_closure):
    """{arm: closure hash} per arm type (A.10.6, §13.4), one process at a time; arms default: every arm module."""
    v85 = pathlib.Path(v85)
    arms = arm_modules() if arms is None else arms
    starts = json.load(open(v85 / "s0" / "starts.json", encoding="utf-8"))
    unit = closure_unit(starts)
    E = starts["units"][unit]["A"]["E"]
    out, files = {}, {}
    for arm in arms:
        if not (ROOT / "ragnarok" / "evaluation" / "arms_v85" / f"{arm}.py").exists():
            continue
        c = runner(v85, unit, arm, E if arm == "d" else ARM_SEED.get(arm, 9826))
        if c.get("closure_foreign"):
            raise RuntimeError(f"closure of {arm}: foreign modules {c['closure_foreign']}")
        out[arm], files[arm] = c["closure_hash"], c.get("closure_files")
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.6, §13.4", unit=unit, closures=out, closure_files=files)


def start_hashes(starts, K_gs):
    """SHA-256 of the s1 blobs and entry caches of the F1 cells (from s0/starts.json, hash-checked at load)."""
    out = {}
    for (g, s), ks in sorted(K_gs.items()):
        st = starts["units"][g][s]
        out[f"{g}/{s}"] = dict(s1=st["s1_sha256"], entry=st["entry"]["sha256"])
    return out


def input_hashes(v85, K_gs):
    """A.10.6 'maps and libraries' of the F1 cells: the K*map map (oracle_v85.lineage_map on SenseEnv85(g, E_s))
    and the K*mac library (the other start's lineage through this start's adapter), hashed as the arms do."""
    from ragnarok.evaluation.arms_v85.kmap import map_sha256
    from ragnarok.evaluation.oracle_v85 import kmac_library, lineage_map
    from scripts.v85_job import Starts
    S = Starts(pathlib.Path(v85) / "s0")
    maps, libs = {}, {}
    for (g, s), ks in sorted(K_gs.items()):
        if len(ks) < 2:
            continue
        s1, st = S.load_s1(g, s), S.start(g, s)
        maps[f"{g}/{s}"] = map_sha256(lineage_map(g, int(st["E"]), s1))
        other = S.load_s1(g, "B" if s == "A" else "A")
        libs[f"{g}/{s}"] = kmac_library(other, S.load_body(g, s).adapter)[1]["sha256"] if other is not None else None
    return dict(maps=maps, libraries=libs)


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


def _json(p):
    return json.load(open(p, encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "f1", "time", "closures", "inputs"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--topped-up", action="store_true")
    ap.add_argument("--hashes", nargs="+", default=None, help="f1: JSON files merged into F1 'hashes' (HASH_KEYS)")
    ap.add_argument("--closures", default=None, help="f1: closures.json of `closures` (else recomputed now)")
    ap.add_argument("--seed-grep", default=None, help="f1: seed_grep.json of v85_stage0 grep")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if a.cmd == "time":
        print(json.dumps(time_run(a.sims or 20)))
        return
    if a.cmd == "closures":
        c = arm_closures(out)
        c.update(provenance=stamp_out([out / "s0" / "starts.json"]))
        _write(out / "closures.json", c)
        print(json.dumps(c["closures"]))
        return
    sp = out / "s0" / "starts.json"
    if not sp.exists():
        raise SystemExit("no s0/starts.json (A.2 order, validity)")
    starts = _json(sp)
    ist, paths = load_runs(out / "istage")
    runs = group(ist)
    iv = out / "istage" / "istage_verdict.json"
    ivd = _json(iv) if iv.exists() else None
    if ivd is None or ivd.get("K_gs") is None:
        raise SystemExit(f"I-stage verdict {'missing' if ivd is None else ivd.get('stage')}: no K_gs to calibrate on "
                         "(run v85_verdict.py istage on the complete I-stage)")
    K_gs = kgs_from_json(ivd["K_gs"])
    if a.cmd == "inputs":
        h = input_hashes(out, K_gs)
        h.update(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.6 (maps and libraries)", provenance=stamp_out([sp, iv]))
        _write(out / "f1_inputs.json", h)
        print(json.dumps(dict(maps=len(h["maps"]), libraries=len(h["libraries"]))))
        return
    if a.cmd == "run":
        units = a2_order(starts, {g for g, s in K_gs})
        cal = calibrate(prepare(cells_from_runs(runs, K_gs), units, require_m=3), a.sims or SIMS, B_CAL, a.topped_up)
        cal.update(claim_header=CLAIM, senses=SENSES, provenance=stamp_out(paths + [iv, sp]))
        _write(out / "calib.json", cal)
        print(json.dumps(dict(outcome=cal["outcome"], selection=cal["selection"] and {
            k: cal["selection"][k] for k in ("n", "tau_pt", "tau_lb")}, runtime_s=cal["runtime_s"])))
        return
    if ivd.get("stage") != "PASS":
        sys.exit(f"I-stage stage {ivd.get('stage')}: G1 does not run (§10, Part C)")
    cal = _json(out / "calib.json")
    if cal["outcome"] != "SELECTED":
        sys.exit(f"calibration outcome {cal['outcome']}: no thresholds, G1 does not run")
    if not a.hashes or not a.seed_grep:
        sys.exit("f1 needs --hashes (maps, libraries, family code, development objects, kappa) and --seed-grep (A.10.6)")
    hashes = {}
    for p in a.hashes:
        hashes.update({k: v for k, v in _json(p).items() if k not in ("provenance", "rule")})
    lack = [k for k in HASH_KEYS if hashes.get(k) is None]
    if lack:
        sys.exit(f"F1 hashes lack {lack} (A.10.6)")
    hashes["starts"] = start_hashes(starts, K_gs)
    clo = _json(a.closures) if a.closures else arm_closures(out)
    need = [x for x in GATE_ARMS if x != "d" and (ROOT / "ragnarok" / "evaluation" / "arms_v85" / f"{x}.py").exists()]
    have = {canon(k) for k in (clo.get("closures") or {})}
    if [x for x in need if canon(x) not in have]:
        sys.exit(f"closures lack {[x for x in need if canon(x) not in have]} (A.10.6: per arm type)")
    budget = budget_projection(ist, K_gs, starts, (cal.get("selection") or {}).get("n"))
    if budget is None:
        sys.exit("no I-stage V2s runs: no §13.1 budget projection")
    grep = _json(a.seed_grep)
    f1 = assemble_f1(cal, K_gs, budget, hashes, clo["closures"], grep,
                     stamp_out([out / "calib.json", iv, sp] + [pathlib.Path(p) for p in a.hashes]
                               + ([pathlib.Path(a.closures)] if a.closures else []) + [pathlib.Path(a.seed_grep)]))
    f1["istage_stage"] = ivd["stage"]
    f1["closures_unit"] = clo.get("unit")
    _write(out / "f1_thresholds.json", f1)
    print(json.dumps({k: f1[k] for k in ("U", "n", "n_O4", "tau_pt", "tau_lb")}))


if __name__ == "__main__":
    main()
