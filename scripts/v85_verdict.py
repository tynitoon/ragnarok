"""v85 verdict (docs/ARC4_V85_PROTOCOL_EN.md §5, §7, A.8-A.13, A.10.2-A.10.4, Part C), evaluator side.

Pure functions (also used, frozen, by scripts/v85_calib.py) and a CLI:
  python scripts/v85_verdict.py istage   # cross-fitted K_{g,s}, D control, instrument line, regime ratio
                                         # (K_g and start validity from s0/starts.json when present)
  python scripts/v85_verdict.py g1       # rows O_4, K*map, K*mac against V2s: A.11 labels, A.12, A.13
  python scripts/v85_verdict.py g1p      # G1-P placebos (§7.2) and routing (§7.3)
Run JSONs follow the shared v85 schema (scripts/v85_job.py) and are grouped by (game, start, arm).

Yardstick (A.8). T(M_k) = billed frames since the clock start at the first record with cum >= M_k (from the
run's records and ladder; the "T" field when a run has no records); a miss is censored at the cap (4M).
lam = mean over K_{g,s} of log max(5000, T(M_k)); lam_late = mean over K_{g,s} minus k_first of
log max(5000, T(M_k) - T(M_{k_first})), each term log(4M) when M_{k_first} is missed.
Statistics (A.9). l_g(a:c) = mean over valid starts of [mean lam(c) - mean lam(a)]; r_g = exp(l_g) clipped
to [1/8, 8]; R = exp(mean_g log r_g). Two-level bootstrap: units with replacement, then, for every drawn
copy of a unit, runs with replacement inside each of its (start, arm) cells; LB90/UB90/UB95 = 10th/90th/95th
percentiles of the resampled R (linear interpolation). Every contrast is computed and labelled on its own
(intersection-union): a label needs every clause of every named contrast.

Choices where the text leaves room (each listed in the build result):
- bootstrap stream of a contrast: default_rng([9805, crc32("<gate>:<arm>:<comparator>")]); lam, lam_late and
  the 8M sensitivity of one contrast share their draws.
- "V2s misses counted at 8M" (A.11 NO-ROOM b): the comparator's missed T are 8M in lam; generally a missed
  M_{k_first} gives log(miss value) terms in lam_late (log 4M in the primary analysis, as written).
- total emulated frames (instrument report only): T + the record's oracle frames; a miss is cap + the run's
  oracle frames.
- a unit enters a contrast when at least one (g, s) has |K_{g,s}| >= 2 and runs of both arms; the D control
  and the instrument line use the cross-fitted K_{g,s} (A.8 defines lam on K_{g,s}); in the D control a
  unit without a D run, or a D run without an exact confirmation, counts as a failure.
- cross-fit with fewer than 3 V2s runs at the other start: all present runs must reach k (flagged).
- K*map coverage pooled as sum(found) / sum(coins) over the row's runs; unknown coverage counts as MAP-WEAK
  (it cannot establish "not MAP-WEAK").
- a run without a dirty flag or without a closure hash is an Interruption, as is a resumed run whose segments
  disagree on the closure (provenance.closure_consistent false: a mixed closure inside one run).
- an instrument line that cannot be computed counts as INSTRUMENT-FAIL.
- A.12: v_K = df-pooled within-(g, s) variance of lam(K) / that of V2s; tau_K = DerSimonian-Laird
  method-of-moments SD of l_g, within-unit variances from the pooled variances; the rerun uses run noise
  variance max(0, v_K - 1) * sigma2_{g,s} and tau = tau_K at the frozen (n, tau_pt, tau_lb).
- arm names match case-insensitively with '*', '_', '-' and spaces ignored ("~" reads "tilde").
- caption rows (O_4-dice, K*mac-exact) are reported against V2s without labels and kept out of the
  Interruption check (they never delay or change the verdict).
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "craft_v6_out" / "v85"
V84 = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84")
CAP, CAP8, FLOOR = 4_000_000, 8_000_000, 5_000
LOG8, LOG15 = math.log(8.0), math.log(1.5)
B_VERDICT, SEED_BOOT = 2000, 9805
QS = (0.5, 10.0, 90.0, 95.0, 99.5)
STARTS = ("A", "B")
REF = {"A": 9712, "B": 9713}
ROWS = ("o4", "kmap", "kmac")
CAPTION_ROWS = ("o4dice", "kmacexact")
LINE = ((9806, 9809), (9807, 9810))
NAMES = {"v2s": "V2s", "d": "D", "o4": "O_4", "kmap": "K*map", "kmac": "K*mac", "gf4": "GF4", "gf": "GF",
         "kmactilde": "K~mac", "natural": "natural V2", "o4dice": "O_4-dice", "kmacexact": "K*mac-exact"}
KIND = {"o4": "which proposal to bill", "kmap": "where to restart", "kmac": "how to move"}
PLACEBO = {"o4": "gf4", "kmap": "gf", "kmac": "kmactilde"}
CLAIM = ("Every v85 claim concerns a SEARCH WITH BILLED REWINDS (savestates; 60 frames per restore; every emulated "
         "frame of the agent billed). It starts from a savestate shared by all arms: a state, chosen by the evaluator, "
         "at the game's first matched rung M1 on an earlier V2 run's lineage that reached the top rung. The root's "
         "sensed rise is set to that lineage's own sensed rise and is identical for all arms. It uses the senses "
         "granted on 22 October 2026: pixels, its own buttons, max(0, r_step) per step, lives. Evaluator instruments "
         "(D, O, K*) additionally use unbilled look-ahead or future-path data, reported as oracle frames; they are not "
         "agents. It is not live play without savestates.")
SENSES = ("pixels, own buttons, max(0, r_step) per step, lives (granted 22 October 2026); root S = S(s1), handed over "
          "by the evaluator; s1 chosen by the evaluator with hindsight on a lineage that reached the top rung; "
          "instruments only: unbilled look-ahead and future-path data (oracle frames)")
NO_ROOM_TEXT = (
    "On these {U} dense-reward development units, from a savestate shared by all arms at M1, three evaluator "
    "instruments did not make this billed-rewind climber 2x faster at the next counted rungs (95% upper bounds below "
    "2, also with V2s misses counted at 2x cap; geometric mean over units). The instruments: best-of-4 of its own "
    "proposals by one unbilled look-ahead each, executed on a fresh stream; restarting half the time from the cell "
    "matching the furthest screen of a future successful path; and motor patterns of another successful path of the "
    "same game, held 4 steps. Effects smaller than 2x, up to the reported upper bounds, are not excluded.")


def canon(arm):
    return "".join(ch for ch in str(arm).lower().replace("~", "tilde") if ch not in "*_- ")


# ---------------------------------------------------------------- yardstick (A.5, A.8)

def run_times(run, ks, total=False):
    """[T(M_k) for k in ks], None = miss (A.5); total=True adds the record's oracle frames (A.8)."""
    cap = float(run.get("cap", CAP))
    recs, lad = run.get("records"), run.get("ladder")
    if recs is not None and lad is not None:
        out = []
        for k in ks:
            r = next((r for r in recs if r[1] >= lad[k]), None)
            out.append(None if r is None or r[0] > cap else float(r[0] + (r[4] if total and len(r) > 4 else 0)))
        return out
    d = (run.get("T_total") if total and run.get("T_total") is not None else run.get("T")) or {}   # no T_total: 0 oracle
    return [None if d.get(str(k)) is None else float(d[str(k)]) for k in ks]


def lam_arrays(T, miss, kmask, miss_value=CAP):
    """A.8 on arrays. T, miss, kmask: (..., K) over the counted rungs in ascending order (k_first at index 0;
    kmask False on padding). Returns (lam, lam_late), each of shape (...)."""
    kmask = np.asarray(kmask, bool)
    miss = np.asarray(miss, bool) & kmask
    Tm = np.where(miss | ~kmask, miss_value, T)
    lam = np.where(kmask, np.log(np.maximum(FLOOR, Tm)), 0.0).sum(-1) / np.maximum(kmask.sum(-1), 1)
    late = kmask.copy()
    late[..., 0] = False
    lg = np.where(miss[..., :1], math.log(miss_value), np.log(np.maximum(FLOOR, Tm - Tm[..., :1])))
    lam_late = np.where(late, lg, 0.0).sum(-1) / np.maximum(late.sum(-1), 1)
    return lam, lam_late


def run_measures(run, ks, miss_value=None, total=False):
    """lam, lam_late and 'misses M_{k_top}' of one run over the counted rungs ks."""
    cap = float(run.get("cap", CAP))
    T = run_times(run, ks, total)
    mv = (cap if miss_value is None else float(miss_value)) + (float(run.get("oracle_frames", 0)) if total else 0.0)
    miss = np.array([t is None for t in T])
    lam, late = lam_arrays(np.array([mv if t is None else t for t in T], float), miss, np.ones(len(ks), bool), mv)
    return dict(lam=float(lam), lam_late=float(late), top_missed=bool(miss[-1]), T=T)


# ---------------------------------------------------------------- statistics (A.9)

def boot_draws(rng, B, U, S=2, Na=1, Nc=1):
    """Two-level bootstrap draws: unit positions (B, U) and run uniforms per (B, U, S, N) for each arm."""
    return rng.integers(0, U, size=(B, U)), rng.random((B, U, S, Na)), rng.random((B, U, S, Nc))


def _cmean(x, n):
    m = np.arange(x.shape[-1]) < n[..., None]
    return np.where(m, x, 0.0).sum(-1) / np.maximum(n, 1)


def _bmean(x, n, I, F, wt):
    """Resampled unit means (M, B, U): runs drawn with replacement inside each cell of each drawn unit copy,
    cell means averaged over the unit's valid starts (wt (B, U, S) = valid / number of valid starts)."""
    M, U, S, N = x.shape
    cnt = n[I]                                                           # (B, U, S)
    j = np.minimum((F[..., :N] * cnt[..., None]).astype(np.int64), N - 1)
    flat = (I[:, :, None, None] * S + np.arange(S)[None, None, :, None]) * N + j
    g = x.reshape(M, -1)[:, flat]                                        # (M, B, U, S, N)
    if not np.all((n == N) | (n == 0)):
        g = np.where(np.arange(N) < cnt[..., None], g, 0.0)
    return np.einsum("mbusn,bus->mbu", g, wt / np.maximum(cnt, 1))


def ssc_core(la, lc, na, nc, valid, draws, pairs, clip=True):
    """A.9 point estimate and bootstrap on packed arrays (shared by the verdict and the calibration).
    la (Ma, U, S, Na), lc (Mc, U, S, Nc): per-run measures (padding ignored); na, nc (U, S) run counts;
    valid (U, S) cells of the contrast (every unit has one); draws = boot_draws(...) or None;
    pairs [(ia, ic)]: arm measure ia against comparator measure ic.
    Returns l (P, U) raw l_g, lr (P, U) clipped log r_g, lR (P,) log R, bR (P, B) resampled log R or None."""
    ia, ic = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    lim = LOG8 if clip else np.inf
    va = np.asarray(valid, float)
    d = _cmean(lc, nc)[ic] - _cmean(la, na)[ia]                          # (P, U, S)
    l = (d * va).sum(-1) / va.sum(-1)
    lr = np.clip(l, -lim, lim)
    if draws is None:
        return l, lr, lr.mean(-1), None
    I, Fa, Fc = draws
    vI = va[I]                                                           # (B, U, S)
    wt = vI / vI.sum(-1, keepdims=True)
    bR = np.clip(_bmean(lc, nc, I, Fc, wt)[ic] - _bmean(la, na, I, Fa, wt)[ia], -lim, lim).mean(-1)
    return l, lr, lr.mean(-1), bR


def _cells(runs, K_gs, a, c):
    return {(g, s): list(ks) for (g, s), ks in sorted(K_gs.items())
            if ks and len(ks) >= 2 and runs.get((g, s, a)) and runs.get((g, s, c))}


def _pack(runs, cells, units, arm, miss8=False, total=False):
    N = max(len(runs[(g, s, arm)]) for g, s in cells)
    x, n, top = np.zeros((3 if miss8 else 2, len(units), 2, N)), np.zeros((len(units), 2), np.int64), []
    for (g, s), ks in cells.items():
        u, si = units.index(g), STARTS.index(s)
        for i, r in enumerate(runs[(g, s, arm)]):
            m = run_measures(r, ks, total=total)
            x[0, u, si, i], x[1, u, si, i] = m["lam"], m["lam_late"]
            if miss8:
                x[2, u, si, i] = run_measures(r, ks, CAP8, total)["lam"]
            top.append(m["top_missed"])
        n[u, si] = len(runs[(g, s, arm)])
    return x, n, float(np.mean(top))


def contrast(runs, K_gs, arm, comp, gate="G1", B=B_VERDICT, clip=True, total=False, rng=None):
    """R(arm:comp) per A.9 over the (g, s) cells of K_gs where both arms have runs (None if there are none).
    runs: {(game, start, canon arm): [run dicts]}; K_gs: {(game, start): [k, ...]}."""
    a, c = canon(arm), canon(comp)
    cells = _cells(runs, K_gs, a, c)
    if not cells:
        return None
    units = sorted({g for g, _ in cells})
    la, na, cens_a = _pack(runs, cells, units, a, False, total)
    lc, nc, cens_c = _pack(runs, cells, units, c, True, total)
    draws = None
    if B:
        if rng is None:
            rng = np.random.default_rng([SEED_BOOT, zlib.crc32(f"{gate}:{a}:{c}{':total' if total else ''}".encode())])
        draws = boot_draws(rng, B, len(units), 2, la.shape[-1], lc.shape[-1])
    l, lr, lR, bR = ssc_core(la, lc, na, nc, (na > 0) & (nc > 0), draws, [(0, 0), (1, 1), (0, 2)], clip)
    st = dict(gate=gate, arm=NAMES.get(a, arm), comp=NAMES.get(c, comp), total_frames=total, U=len(units), units=units,
              cells=[f"{g}/{s}" for g, s in cells], n_arm=int(na.sum()), n_comp=int(nc.sum()),
              R=float(np.exp(lR[0])), R_late=float(np.exp(lR[1])), R_8M=float(np.exp(lR[2])),
              r_g={g: float(np.exp(lr[0, i])) for i, g in enumerate(units)},
              l_g={g: float(l[0, i]) for i, g in enumerate(units)},
              breadth=float(np.mean(lr[0] >= LOG15 - 1e-12)), cens_arm=cens_a, cens_comp=cens_c, B=int(B))
    if B:
        q = np.percentile(np.exp(bR), QS, axis=-1)                       # (5, 3)
        st.update(LB90=float(q[1, 0]), UB90=float(q[2, 0]), UB95=float(q[3, 0]), CI99=[float(q[0, 0]), float(q[4, 0])],
                  LB90_late=float(q[1, 1]), UB95_late=float(q[3, 1]), UB95_8M=float(q[3, 2]))
    return st


# ---------------------------------------------------------------- labels (A.11, §7.2, §8, A.12, A.13)

def a11_clauses(R, LB90, LB90_late, breadth, cens_a, cens_c, UB95, UB95_8M, tau_pt, tau_lb, map_weak=False):
    """A.11 clauses on scalars or broadcastable arrays (the calibration evaluates them on its grid)."""
    R, LB90, LB90_late, breadth, cens_a, cens_c, UB95, UB95_8M = map(
        np.asarray, (R, LB90, LB90_late, breadth, cens_a, cens_c, UB95, UB95_8M))
    return dict(room_a=R >= tau_pt, room_b=LB90 >= tau_lb, room_c=LB90_late > 1, room_d=breadth >= 0.4,
                room_e=(cens_a < 0.4) & (cens_c < 0.4),
                noroom_a=UB95 < 2, noroom_b=UB95_8M < 2, noroom_c=(cens_a <= 0.25) & (cens_c <= 0.25),
                noroom_d=np.logical_not(np.asarray(map_weak, bool)))


def a11(cl):
    """-> (room, noroom, below2x): NO-ROOM-K only where not ROOM-K (ROOM-K takes precedence)."""
    room = cl["room_a"] & cl["room_b"] & cl["room_c"] & cl["room_d"] & cl["room_e"]
    noroom = np.logical_not(room) & cl["noroom_a"] & cl["noroom_b"] & cl["noroom_c"] & cl["noroom_d"]
    return room, noroom, room & cl["noroom_a"]


def row_label(st, tau_pt, tau_lb, map_weak=False):
    cl = a11_clauses(st["R"], st["LB90"], st["LB90_late"], st["breadth"], st["cens_arm"], st["cens_comp"],
                     st["UB95"], st["UB95_8M"], tau_pt, tau_lb, map_weak)
    room, noroom, below = a11(cl)
    label = "ROOM-K" if room else "NO-ROOM-K" if noroom else "INCONCLUSIVE-K"
    return dict(label=label, clauses={k: bool(v) for k, v in cl.items()},
                captions=(["BELOW-2X"] if below else []) + (["MAP-WEAK"] if map_weak else []))


def g1_label(row_labels, reasons=(), o_dropped=False):
    """A.13: Interruption first; ROOM if any ROOM-K; NO-ROOM if every tested row is NO-ROOM-K."""
    if reasons:
        return "Interruption"
    tested = [r for r in ROWS if not (o_dropped and r == "o4")]
    if any(row_labels.get(r) == "ROOM-K" for r in tested):
        return "ROOM"
    if all(row_labels.get(r) == "NO-ROOM-K" for r in tested):
        return "NO-ROOM"
    return "INCONCLUSIVE"


def g1p_label(st):
    """§7.2 on R(K:placebo)."""
    if st["UB90"] < 1.25:
        return "GENERIC-K"
    return "SEPARATED-K" if st["LB90"] > 1 else "NOT-SEPARATED-K"


def routing(g1_rows, g1p_rows):
    """§7.3. g1_rows: {row: A.11 label}; g1p_rows: {row: G1-P label}."""
    if not any(v == "ROOM-K" for v in g1_rows.values()):
        return dict(result="v85 ends (§11)", families=[])
    live = [r for r in ROWS if g1_rows.get(r) == "ROOM-K" and g1p_rows.get(r) != "GENERIC-K"]
    fam = [f for r, f in (("o4", "W"), ("kmac", "M")) if r in live]
    if not fam:
        ch = ", ".join(NAMES[r] for r in ROWS if g1_rows.get(r) == "ROOM-K")
        return dict(result="ROOM-UNCARRIED", families=[],
                    sentence=f"room exists in channel(s) {ch}; no v85 family holds that information.")
    return dict(result="G2a for " + " and ".join(fam), families=fam)


def g2a_label(st_KF, st_KV):
    """§8.1 OBJECT-ROOM and the ceiling-to-label rule."""
    ok = st_KF["R"] >= 1.4 and st_KF["LB90"] > 1 and st_KV["LB90"] > 1
    top = "FASTER-2X" if (st_KF["R"] >= 2.0 and st_KF["LB90"] >= 1.5) else "FASTER-PARTIAL"
    return dict(label="OBJECT-ROOM" if ok else "KC-object", sealed_top_label=top)


def g2b_label(st_XF, others):
    """§8.2 GO; others: {name: stats of R(X:c)} for c = V2s, each tuned default and each placebo."""
    cl = dict(XF_point=st_XF["R"] >= 1.4, XF_LB90=st_XF["LB90"] > 1, XF_late=st_XF["LB90_late"] > 1,
              **{f"vs_{k}": v["LB90"] > 1 for k, v in others.items()})
    return dict(label="GO" if all(cl.values()) else "KC-dev", clauses=cl)


def vocab_caption(st_on_frozen):
    return "HEAD-START-VOCABULARY" if st_on_frozen["LB90"] <= 1 else "ADAPTS"


def a12_params(runs, K_gs, arm, comp="v2s"):
    """A.12: v_K (pooled within-(g, s) variance ratio of lam) and tau_K (method-of-moments SD of l_g)."""
    a, c = canon(arm), canon(comp)
    cells = _cells(runs, K_gs, a, c)
    lam = {(g, s, x): np.array([run_measures(r, ks)["lam"] for r in runs[(g, s, x)]])
           for (g, s), ks in cells.items() for x in (a, c)}

    def pooled(x):
        ss = sum(float(((v - v.mean()) ** 2).sum()) for (g, s, y), v in lam.items() if y == x)
        df = sum(len(v) - 1 for (g, s, y), v in lam.items() if y == x)
        return ss / df if df > 0 else None
    va, vc = pooled(a), pooled(c)
    if va is None or vc is None or vc <= 0:
        return dict(v_K=None, tau_K=None)
    units = sorted({g for g, _ in cells})
    l, se2 = [], []
    for g in units:
        ss = [s for s in STARTS if (g, s) in cells]
        l.append(np.mean([lam[(g, s, c)].mean() - lam[(g, s, a)].mean() for s in ss]))
        se2.append(sum(vc / len(lam[(g, s, c)]) + va / len(lam[(g, s, a)]) for s in ss) / len(ss) ** 2)
    l, w = np.array(l), 1.0 / np.array(se2)
    lbar = (w * l).sum() / w.sum()
    Q = float((w * (l - lbar) ** 2).sum())
    den = w.sum() - (w ** 2).sum() / w.sum()
    t2 = max(0.0, (Q - (len(units) - 1)) / den) if den > 0 else 0.0
    return dict(v_K=va / vc, tau_K=math.sqrt(t2), var_arm=va, var_comp=vc, Q=Q)


def a12_triggered(p):
    return p.get("v_K") is not None and (p["v_K"] > 1.5 or p["tau_K"] > 0.4)


def a12_downgrade(label, oc):
    """oc: {criterion: passed} of the rerun at the frozen (n, tau_pt, tau_lb); only downgrades."""
    if label == "ROOM-K" and not all(oc[k] for k in ("ii", "iii", "iv", "v")):
        return "INCONCLUSIVE-K"
    if label == "NO-ROOM-K" and not oc["vi"]:
        return "INCONCLUSIVE-K"
    return label


def cross_fit(K_g, k_ref, other_runs, other_valid=True):
    """A.10.4 for one (g, s): k_top = the highest k in K_g reached by all V2s runs at the other start s'
    (k_ref if s' is invalid); K_{g,s} = {k in K_g : k <= k_top}; [] (dropped) if fewer than 2."""
    if not other_valid:
        kt, n = k_ref, 0
    else:
        n = len(other_runs)
        tops = []
        for r in other_runs:
            hit = [k for k, t in zip(K_g, run_times(r, K_g)) if t is not None]
            tops.append(max(hit) if hit else None)
        kt = None if (not tops or None in tops) else min(tops)
    K = [k for k in K_g if kt is not None and k <= kt]
    return dict(k_top=kt, K=K if len(K) >= 2 else [], n_other=n, complete=(not other_valid) or n == 3)


def _exp_meta(r):
    h = (r.get("exp") or {}).get("meta_sha256") or r.get("exp_meta_sha256") or (r.get("summary") or {}).get("exp_meta_sha256")
    if h is None and r.get("exp_meta") is not None:
        h = hashlib.sha256(json.dumps(r["exp_meta"], sort_keys=True).encode()).hexdigest()
    return h


def certificate(originals, reruns, first_unit):
    """§13.4 reuse certificate: V2s seeds 9806 (A) and 9809 (B) on the first unit in draw order, re-executed under
    the new closure, must equal the originals' records and exp_meta byte for byte."""
    checks = []
    for s, seed in (("A", 9806), ("B", 9809)):
        pick = lambda rs: [r for r in rs if r["game"] == first_unit and r["start"] == s and int(r["seed"]) == seed
                           and canon(r["arm"]) == "v2s"]
        o, n = pick(originals), pick(reruns)
        if not o:
            continue                                     # invalid start: no such I-stage run
        ok = bool(n) and json.dumps(o[0]["records"]) == json.dumps(n[0]["records"]) \
            and _exp_meta(o[0]) is not None and _exp_meta(o[0]) == _exp_meta(n[0])
        checks.append(dict(start=s, seed=seed, rerun=bool(n), ok=bool(ok)))
    return dict(ok=bool(checks) and all(c["ok"] for c in checks), checks=checks)


def interruptions(run_list, committed=None, cert=None, reused=("v2s",)):
    """A.13 / §13.4: replay mismatch, dirty tree, missing or mixed closure hashes per arm type, a closure differing
    from the gate's committed one, and (for reused arms) a missing or failed comparator certificate."""
    why, by_arm = [], {}
    for r in run_list:
        tag = f"{r.get('game')}/{r.get('start')}/{r.get('arm')}/{r.get('seed')}"
        if r.get("interruption") or any(not c.get("ok") for c in r.get("replay") or []):
            why.append(f"replay mismatch: {tag}")
        p = r.get("provenance") or {}
        if p.get("dirty") is not False:
            why.append(f"dirty tree (or no dirty flag): {tag}")
        if p.get("closure_consistent") is False:
            why.append(f"mixed closure hashes within a resumed run: {tag}")
        by_arm.setdefault(canon(r.get("arm")), set()).add(p.get("closure_hash"))
    com = {canon(k): v for k, v in (committed or {}).items()}
    for a, hs in sorted(by_arm.items()):
        if None in hs:
            why.append(f"missing closure hash: {NAMES.get(a, a)}")
        elif len(hs) > 1:
            why.append(f"mixed closure hashes: {NAMES.get(a, a)}")
        elif a in com and hs != {com[a]}:
            if a not in reused:
                why.append(f"changed closure hash: {NAMES.get(a, a)}")
            elif cert is None:
                why.append(f"missing comparator certificate: {NAMES.get(a, a)} closure changed")
            elif not cert.get("ok"):
                why.append(f"comparator certificate failed: all {NAMES.get(a, a)} runs must be redone")
    return why


def room_sentence(row, st, tau_pt, tau_lb, R_total=None):
    o = (f" with 4 unbilled look-ahead rollouts per expansion (on total emulated frames R = {R_total:.3g})"
         if row == "o4" and R_total is not None else "")
    return (f"From a savestate shared by all arms at M1 on {st['U']} development games, same-game information of kind "
            f"{NAMES[row]} ({KIND[row]}), supplied by an evaluator instrument{o}, made this billed-rewind climber reach "
            f"the next counted rungs faster: point R = {st['R']:.3g} (>= {tau_pt}), 90% lower bound {st['LB90']:.3g} "
            f"(>= {tau_lb}) (calibrated: P(ROOM | 1.5x) <= 0.25, P(ROOM | 2x) >= 0.8). This is room: not transfer, "
            f"not a carried object, not faster learning.")


def no_room_sentence(U, ub95, o_dropped=False):
    s = NO_ROOM_TEXT.format(U=U) + " UB95: " + ", ".join(f"{NAMES[r]} {v:.3g}" for r, v in ub95.items()) + "."
    return s + (" Selection channel not tested." if o_dropped else "")


# ---------------------------------------------------------------- reports

def profiles(runs, K_gs, arms):
    """Per (g, s), per counted rung: mean log T (misses at the cap) and misses, for each arm."""
    out = {}
    for (g, s), ks in sorted(K_gs.items()):
        for a in map(canon, arms):
            rs = runs.get((g, s, a))
            if not ks or not rs:
                continue
            T = np.array([[CAP if t is None else t for t in run_times(r, ks)] for r in rs], float)
            miss = np.array([[t is None for t in run_times(r, ks)] for r in rs])
            out.setdefault(f"{g}/{s}", {})[NAMES.get(a, a)] = {
                str(k): dict(mean_logT=float(np.log(T[:, i]).mean()), misses=int(miss[:, i].sum())) for i, k in enumerate(ks)}
    return out


def coverage(rs):
    """Pooled K*map coverage over runs (found / coins), else the mean of per-run coverages; None if unknown."""
    ins = [r.get("instrument") or {} for r in rs]
    if ins and all("coins" in i and "found" in i for i in ins):
        coins = sum(i["coins"] for i in ins)
        return sum(i["found"] for i in ins) / coins if coins else None
    cov = [i.get("coverage") for i in ins if i.get("coverage") is not None]
    return float(np.mean(cov)) if cov else None


def instrument_summary(rs):
    """Mean over runs of every numeric field of the runs' "instrument" captions (fidelity, library size, ...)."""
    keys = sorted({k for r in rs for k, v in (r.get("instrument") or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return {k: float(np.mean([r["instrument"][k] for r in rs if isinstance((r.get("instrument") or {}).get(k), (int, float))]))
            for k in keys}


def o_ties(rs):
    """Pooled O-TIES share: per-run o_ties weighted by expansions_scored, else their mean."""
    ins = [r.get("instrument") or {} for r in rs]
    if ins and all(i.get("o_ties") is not None and "expansions_scored" in i for i in ins):
        n = sum(i["expansions_scored"] for i in ins)
        return sum(i["o_ties"] * i["expansions_scored"] for i in ins) / n if n else None
    v = [i.get("o_ties") for i in ins if i.get("o_ties") is not None]
    return float(np.mean(v)) if v else None


def group(run_list):
    runs = {}
    for r in run_list:
        runs.setdefault((r["game"], r["start"], canon(r["arm"])), []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["seed"]))
    return runs


def load_runs(*dirs):
    out, paths = [], []
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("*.json")):
            j = json.load(open(p, encoding="utf-8"))
            if isinstance(j, dict) and all(k in j for k in ("game", "start", "arm", "seed")):
                out.append(j)
                paths.append(p)
    return out, paths


def kgs_to_json(K_gs):
    d = {}
    for (g, s), ks in sorted(K_gs.items()):
        d.setdefault(g, {})[s] = [int(k) for k in ks]
    return d


def kgs_from_json(d):
    return {(g, s): list(ks) for g, v in d.items() for s, ks in v.items()}


def unit_rungs(runs, rungs=None):
    """K_g and k_ref per unit: from S0 data when given (s0/starts.json {"units": {game: {"K_g": [..]}}}, or
    {game: [k..]} / {game: {"K_g": [..]}}), else from the I-stage V2s runs' "rungs" and "k_ref" fields (which
    must agree within a unit). k_ref = max K_g (A.4)."""
    out = {}
    rungs = (rungs or {}).get("units", rungs) if isinstance(rungs, dict) else rungs
    for (g, s, a), rs in runs.items():
        if a != "v2s":
            continue
        for r in rs:
            kg = list(r["rungs"])
            kr = int(r.get("k_ref", max(kg)))
            if g in out and out[g] != (kg, kr):
                raise ValueError(f"I-stage V2s runs of {g} disagree on K_g / k_ref")
            out[g] = (kg, kr)
    for g, v in (rungs or {}).items():
        kg = list(v if isinstance(v, list) else v.get("K_g") or v.get("K"))
        out[g] = (kg, int(v.get("k_ref", max(kg))) if isinstance(v, dict) else max(kg))
    return out


def start_validity(starts):
    """{(game, start): valid} from S0's starts.json ({"units": {game: {"A": {"valid": ..} or null, ...}}})."""
    return {(g, s): bool(u.get(s) and u[s].get("valid", True)) for g, u in (starts or {}).get("units", {}).items()
            for s in STARTS}


def istage_kgs(runs, rungs=None, valid=None):
    """A.10.4 for every (g, s) of the I-stage. valid {(g, s): bool} from S0 (start_validity); without it a start
    is valid iff it has V2s runs. A valid other start without V2s runs gives no k_top (the cell is dropped)."""
    ur = unit_rungs(runs, rungs)
    K_gs, detail = {}, {}
    for g, (kg, kr) in sorted(ur.items()):
        ok = {s: (valid[(g, s)] if valid is not None and (g, s) in valid else bool(runs.get((g, s, "v2s"))))
              for s in STARTS}
        for s in STARTS:
            if not ok[s]:
                continue
            o = "B" if s == "A" else "A"
            cf = cross_fit(kg, kr, runs.get((g, o, "v2s"), []), ok[o])
            detail[f"{g}/{s}"] = cf
            if cf["K"]:
                K_gs[(g, s)] = cf["K"]
    return K_gs, detail


def _confirmed(r):
    rep = r.get("replay") or []
    return bool(rep) and not r.get("interruption") and all(c.get("ok") for c in rep)


def d_control(runs, K_gs, units):
    """A.10.2: r_g(D:V2s) >= 4 on >= 80% of units, every D confirmation exact."""
    st = contrast(runs, K_gs, "d", "v2s", gate="I", B=0)
    r = {g: (st["r_g"].get(g) if st else None) for g in units}
    share = float(np.mean([v is not None and v >= 4 for v in r.values()])) if units else 0.0
    exact = all(_confirmed(x) for (g, s, a), rs in runs.items() if a == "d" for x in rs)
    return dict(r_g=r, share=share, exact=exact, passed=bool(share >= 0.8 and exact))


def instrument_line(runs, K_gs, B=B_VERDICT):
    """A.10.2: V2s {9806, 9809} against {9807, 9810}; INSTRUMENT-FAIL if the 99% interval excludes 1."""
    sub = {}
    for (g, s, a), rs in runs.items():
        if a == "v2s":
            for tag, seeds in zip(("linea", "linec"), LINE):
                sub.setdefault((g, s, tag), []).extend(r for r in rs if int(r["seed"]) in seeds)
    st = contrast(sub, K_gs, "linea", "linec", gate="I", B=B)
    if st is None:
        return dict(R=None, CI99=None, instrument_fail=None)
    lo, hi = st["CI99"]
    return dict(R=st["R"], CI99=[lo, hi], U=st["U"], instrument_fail=bool(lo > 1 or hi < 1))


def natural_runs(units, ladders, v84=V84):
    """Natural V2 (G1b 9712 -> start A, 9713 -> B) timed from its own M1, as pseudo-runs with a "T" field."""
    out = {}
    for g in units:
        lad = ladders[g]
        for s, seed in REF.items():
            recs = json.load(open(v84 / "g1b" / f"{g}_{seed}.json", encoding="utf-8"))["records"]
            first = lambda m: next((r[0] for r in recs if r[1] >= m), None)
            t1 = first(lad[1])
            T = {str(k): (None if t1 is None or first(lad[k]) is None else first(lad[k]) - t1) for k in range(len(lad))}
            out[(g, s, "natural")] = [dict(game=g, start=s, arm="natural", seed=seed, T=T, cap=CAP)]
    return out


def istage_report(runs, units, rungs=None, natural=None, B=B_VERDICT, valid=None):
    K_gs, detail = istage_kgs(runs, rungs, valid)
    out = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.2-A.10.4", units=units, K_gs=kgs_to_json(K_gs), cross_fit=detail,
               d_control=d_control(runs, K_gs, units), instrument_line=instrument_line(runs, K_gs, B))
    if natural:
        st = contrast({**runs, **natural}, K_gs, "natural", "v2s", gate="I", B=B)
        out["regime_ratio"] = dict(name="R(natural V2 from its own M1 : V2s from s1), on the counted rungs",
                                   R=st["R"], LB90=st["LB90"], UB90=st["UB90"], r_g=st["r_g"]) if st else None
    D = out["d_control"]
    # D failing: first time one dated fix and D reruns; second time Interruption (D) (A.10.2)
    out["stage"] = ("D-CONTROL-FAIL" if not D["passed"] else
                    "INSTRUMENT-FAIL" if out["instrument_line"]["instrument_fail"] is not False else "PASS")
    return out


def g1_report(runs, K_gs, thr, istage_runs=None, committed=None, cert=None, o_dropped=False, B=B_VERDICT,
              a12_rerun=None):
    """A.11-A.13. runs: G1 rows plus the reused V2s; thr: {"n", "tau_pt", "tau_lb"}; a12_rerun(row, v_K, tau_K)
    -> {criterion: passed} (default: scripts/v85_calib.rerun_a12 on the I-stage V2s)."""
    tp, tl = thr["tau_pt"], thr["tau_lb"]
    rows, labels, ub = {}, {}, {}
    rd = contrast(istage_runs or runs, K_gs, "d", "v2s", gate="I", B=0, clip=False)
    for row in ROWS:
        if o_dropped and row == "o4":
            continue
        st = contrast(runs, K_gs, row, "v2s", gate="G1", B=B)
        if st is None:
            rows[row] = dict(label="INCONCLUSIVE-K", note="no runs")
            labels[row] = "INCONCLUSIVE-K"
            continue
        rs = [r for (g, s, a), v in runs.items() if a == row for r in v]
        mw = False
        if row == "kmap":
            cov = coverage(rs)
            st["coverage"] = cov
            mw = cov is None or cov < 0.30
        lab = row_label(st, tp, tl, mw)
        tot = contrast(runs, K_gs, row, "v2s", gate="G1", B=B, total=True)
        st.update(R_total_frames=tot["R"], LB90_total_frames=tot["LB90"], instrument=instrument_summary(rs),
                  oracle_frames=float(sum(r.get("oracle_frames", 0) for r in rs)))
        if row == "o4":
            st["o_ties"] = o_ties(rs)
            if st["o_ties"] is not None and st["o_ties"] > 0.70:
                lab["captions"].append("O-TIES")
        p = a12_params(runs, K_gs, row)
        lab["a12"] = dict(p, triggered=a12_triggered(p))
        if a12_triggered(p) and lab["label"] != "INCONCLUSIVE-K":
            if a12_rerun is None:
                from scripts.v85_calib import cells_from_runs, rerun_a12
                cells = cells_from_runs(istage_runs or runs, K_gs)
                a12_rerun = lambda r_, v_, t_: rerun_a12(cells, sorted({g for g, _ in cells}), thr, v_, t_)["passed"]  # noqa: E731
            oc = a12_rerun(row, p["v_K"], p["tau_K"])
            lab["a12"]["oc"] = oc
            lab["label"] = a12_downgrade(lab["label"], oc)
            if lab["label"] != "ROOM-K" and "BELOW-2X" in lab["captions"]:
                lab["captions"].remove("BELOW-2X")
        if rd and rd["R"] != 1 and st["R"] > 0:
            st["logR_over_logR_D"] = math.log(st["R"]) / math.log(rd["R"])
        rows[row] = dict(stats=st, **lab)
        labels[row], ub[row] = lab["label"], st["UB95"]
    caps = {NAMES.get(c, c): contrast(runs, K_gs, c, "v2s", gate="G1C", B=B) for c in CAPTION_ROWS
            if any(a == c for (g, s, a) in runs)}                    # caption rows: reported, never labelled
    gated = [r for (g, s, a), v in runs.items() if a in ROWS + ("v2s",) for r in v]
    reasons = interruptions(gated, committed, cert)
    verdict = g1_label(labels, reasons, o_dropped)
    U = max((x["stats"]["U"] for x in rows.values() if "stats" in x), default=0)
    if verdict == "ROOM":
        sentences = [room_sentence(r, rows[r]["stats"], tp, tl, rows[r]["stats"].get("R_total_frames"))
                     + (" [BELOW-2X]" if "BELOW-2X" in rows[r]["captions"] else "")
                     for r in ROWS if labels.get(r) == "ROOM-K"]
    elif verdict == "NO-ROOM":
        sentences = [no_room_sentence(U, ub, o_dropped)]
    elif verdict == "INCONCLUSIVE":
        sentences = ["Room for 2x was neither shown nor excluded in any tested channel."]
    else:
        sentences = ["Interruption: no verdict. " + "; ".join(reasons)]
    return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.11-A.13", G1=verdict, rows=rows, interruptions=reasons,
                thresholds=thr, sentences=sentences, R_D_unclipped=rd["R"] if rd else None, caption_rows=caps,
                note_logR_ratio="log R_K / log R_D (unclipped): a ratio, not a decomposition",
                profiles=profiles(runs, K_gs, ("v2s",) + ROWS), claim_header=CLAIM, senses=SENSES)


def closure_hash():
    """SHA-256 over sorted (path, file SHA-256) of the project modules loaded now, plus this script (§13.4)."""
    from ragnarok.evaluation.provenance_v81 import full_hash
    files = {pathlib.Path(__file__).resolve()}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f:
            p = pathlib.Path(f).resolve()
            if ROOT in p.parents and ".venv" not in p.parts:
                files.add(p)
    items = sorted((p.relative_to(ROOT).as_posix(), full_hash(p)) for p in files)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def stamp_out(paths):
    from ragnarok.evaluation.provenance_v81 import stamp
    p = stamp([str(x) for x in paths], cwd=str(ROOT), paths=("ragnarok", "scripts", "tests"), full=True)
    p["closure_hash"] = closure_hash()
    return p


def _write(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("istage", "g1", "g1p"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rungs", default=None, help="S0 rungs JSON (default: the I-stage runs' rungs fields)")
    ap.add_argument("--o-dropped", action="store_true", help="O-confined KC-build: selection row not tested")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    sp = out / "s0" / "starts.json"
    starts = json.load(open(sp, encoding="utf-8")) if sp.exists() else None
    rungs = json.load(open(a.rungs, encoding="utf-8")) if a.rungs else starts
    ist, ipaths = load_runs(out / "istage")
    iruns = group(ist)
    from ragnarok.evaluation.start_v85 import ladder, unit_order
    if a.cmd == "istage":
        units = unit_order({g for g, s, x in iruns})
        have = [g for g in units if (V84 / "bars" / f"{g}_9706.json").exists()
                and all((V84 / "g1b" / f"{g}_{x}.json").exists() for x in REF.values())]
        nat = natural_runs(have, {g: ladder(g) for g in have}) if have else None
        rep = istage_report(iruns, units, rungs, nat, valid=start_validity(starts) if starts else None)
        rep.update(claim_header=CLAIM, senses=SENSES, provenance=stamp_out(ipaths + ([sp] if starts else [])))
        _write(out / "istage" / "istage_verdict.json", rep)
        print(json.dumps({k: rep[k] for k in ("stage", "d_control", "instrument_line")}, default=str)[:2000])
        return
    f1 = json.load(open(out / "f1_thresholds.json", encoding="utf-8"))
    K_gs = kgs_from_json(f1["K_gs"])
    thr = {k: f1[k] for k in ("n", "tau_pt", "tau_lb")}
    g1, gpaths = load_runs(out / "g1")
    if a.cmd == "g1":
        runs = group(g1 + [r for r in ist if canon(r["arm"]) == "v2s"])
        cert_runs, cpaths = load_runs(out / "g1" / "certificate")
        units = unit_order({g for g, s in K_gs})
        cert = certificate(ist, cert_runs, units[0]) if cert_runs else None
        from scripts.v85_calib import cells_from_runs, rerun_a12
        cells = cells_from_runs(iruns, K_gs)

        def rerun(row, v, tau):
            return rerun_a12(cells, units, thr, v, tau)["passed"]
        rep = g1_report(runs, K_gs, thr, iruns, f1.get("closure_hashes"), cert, a.o_dropped, a12_rerun=rerun)
        rep["certificate"] = cert
        rep["provenance"] = stamp_out(ipaths + gpaths + cpaths + [out / "f1_thresholds.json"])
        _write(out / "g1" / "g1_verdict.json", rep)
        print(json.dumps(dict(G1=rep["G1"], rows={r: v["label"] for r, v in rep["rows"].items()},
                              interruptions=rep["interruptions"])))
        return
    v = json.load(open(out / "g1" / "g1_verdict.json", encoding="utf-8"))
    g1p, ppaths = load_runs(out / "g1p")
    runs = group(g1 + g1p)
    res = {}
    for row, x in v["rows"].items():
        if x["label"] != "ROOM-K":
            continue
        st = contrast(runs, K_gs, row, PLACEBO[row], gate="G1P", B=B_VERDICT)
        res[row] = dict(stats=st, label=g1p_label(st) if st else None)
    reasons = interruptions([r for rs in runs.values() for r in rs], f1.get("closure_hashes"), None, reused=())
    rep = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §7.2-7.3", rows=res, interruptions=reasons,
               routing=routing({r: x["label"] for r, x in v["rows"].items()}, {r: x["label"] for r, x in res.items()}),
               claim_header=CLAIM, senses=SENSES, provenance=stamp_out(gpaths + ppaths))
    _write(out / "g1p" / "g1p_verdict.json", rep)
    print(json.dumps(dict(rows={r: x["label"] for r, x in res.items()}, routing=rep["routing"])))


if __name__ == "__main__":
    main()
