"""v85 verdict (docs/ARC4_V85_PROTOCOL_EN.md §5, §7, A.8-A.13, A.10.2-A.10.4, Part C), evaluator side.

Pure functions (also used, frozen, by scripts/v85_calib.py) and a CLI:
  python scripts/v85_verdict.py istage [--d-fix DATE]   # cross-fitted K_{g,s}, D control, instrument line,
                                         # regime ratio (K_g, validity and A.2 order from s0/starts.json)
  python scripts/v85_verdict.py g1       # rows O_4, K*map, K*mac against V2s: A.11 labels, A.12, A.13
  python scripts/v85_verdict.py g1p      # G1-P placebos (§7.2) and routing (§7.3)
  python scripts/v85_verdict.py g2a      # §8.1 for each family licensed by G1-P (runs in g2a/)
  python scripts/v85_verdict.py g2b      # §8.2 for the G2b family (runs in g2b/, F and K reused from g2a/)
  python scripts/v85_verdict.py b1 --family M|W [--sealed DIR]   # §9 B1 label ladder (Part C), sealed runs
Run JSONs follow the shared v85 schema (scripts/v85_job.py) and are grouped by (game, start, arm).
s0/starts.json is required by every command (A.2 draw order, start validity, K_g; kept units only).

Completeness (§7.1, A.10.4-A.10.5): a command labels nothing while an expected run is missing. It writes
status/stage "INCOMPLETE" with the missing runs: I-stage = 3 V2s (9806-9808 / 9809-9811) and 1 D (seed E_s)
per valid start of every I-stage unit (the first U kept units in A.2 order, U from s0/natural_spikeins.json,
extended in order to every unit with runs: the top-up); G1 = n/2 runs per kept (g, s) per core row (n_O4 for
O_4; K*mac only where the other start is valid) on 9812.. / 9815..; G1-P = the same (g, s, seed) for each ROOM
row's placebo; G2 and B1 = the §12 seeds of each arm. Two runs of one (game, start, arm, seed) are refused.
I-stage stages, first match: INCOMPLETE; Interruption (§13.4 on the V2s and D runs: dirty, closure per arm
type, closure inside a run; a V2s replay mismatch; D's own confirmations belong to the D control); D failing:
"D-CONTROL-FAIL" before the one dated D fix, "Interruption (D)" after it (--d-fix DATE records the fix; the
first-pass D outputs must be moved out of istage/, e.g. to istage/d_first_pass/); INSTRUMENT-FAIL; PASS.
The I-stage checks V2s and D against craft_v6_out/v85/closures.json when present (v85_calib closures, written
after the last code commit before the launch); a V2s closure changed by the D fix needs the §13.4 certificate
(reruns of 9806 / 9809 on the first unit in istage/certificate/ or g1/certificate/, made under the committed
closure).

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
  and the instrument line use the cross-fitted K_{g,s} (A.8 defines lam on K_{g,s}). The D control precedes
  the cross-fit (A.10.2 before A.10.4): an I-stage unit with no cross-fitted cell (dropped by A.10.4, a V2s
  censoring event, not a D failure) is scored on its uncross-fitted K_g at its valid starts and listed in
  "uncross_fitted_units". A unit without a D run, or a D run without an exact confirmation, counts as a
  failure.
- cross-fit: k_top needs exactly the 3 V2s runs of the other valid start (A.10.4 "all 3"); otherwise the cell
  has no K_{g,s} (incomplete). Only kept units (s0 "kept", |K_g| >= 2) that have I-stage V2s runs get cells.
- the unit order of every report, of calib.json / f1 "units" (K~mac's derangement) and the certificate's
  "first unit in draw order" is the A.2 order of s0/starts.json ("order", drops removed), never a
  re-permutation of a subset.
- §13.1 O_4 budget rule: O_4's A.12 rerun runs at f1 "n_O4"; when n_O4 != n, O_4's operating characteristics
  at n = 4 (tau = 0.2, the frozen tau_pt / tau_lb) are reported from calib.json's stored grid.
- G1 and G1-P check the closures against F1's closure_hashes: a gated arm type without a committed hash is an
  Interruption. G1-P checks only the ROOM rows' K runs and their placebos (caption rows never interrupt).
- O_4-dice (caption): runs aborted by DiceMismatch are counted and reported, and left out of its R, which is
  reported beside S0's G_4 (s0/dice_g4.json, mean of the per-unit G_4 over the contrast's (g, s)).
- instrument captions: only measurements are averaged (fidelity, priv/NOOP shares, q5 expansions, oracle
  frames, rank correlation, coverage); constants and library sizes are listed per (g, s).
- G2 / B1 CENSORING-SENSITIVE (§8.2, §9, Part C): every claim label is recomputed with all misses (both arms
  of every named contrast, the B1 p-value and leave-one-out included) at 2x cap = 8M; if it changes, the
  primary label drops one step (OBJECT-ROOM -> KC-object; GO -> KC-dev; FASTER-2X -> FASTER-PARTIAL ->
  NOT-SIGNIFICANT); other labels are not claims and never move.
- G2b interaction I = mean over units of [lam(X_frozen) - lam(X_on)] - [lam(V2s) - lam(F_M)] (start means,
  unclipped), two-level bootstrap (units, then runs per cell), stream [9805, crc32("G2b:interaction")];
  LB90 / UB90 = 10th / 90th percentiles; phi = log R(X:F) / log R(K:F) with R(K:F) from G2a.
- B1: p = one-sided exact sign-flip permutation test of mean log r_g(X:F) over units (random 2^16 flips on
  [9859, 2] above 16 units); leave-one-game-out minimum of the point R(X:F); the bootstrap on [9859, .]; the
  sealed instrument line is V2s {9853, 9856} against {9854, 9857}; ACCUMULATES uses Page's L over doses
  0 (F) / 4 / 12 / 41 (X) on unit means of lam, p by 20,000 within-unit permutations on [9859, 3].
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
import os
import pathlib
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "2")       # before numpy: BLAS reads it when it loads (§13.2)
import numpy as np                                     # noqa: E402

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
LINE_SEALED = ((9853, 9856), (9854, 9857))
SEED_SEALED = 9859
V2S_SEEDS = {"A": (9806, 9807, 9808), "B": (9809, 9810, 9811)}
G1_SEED0 = {"A": 9812, "B": 9815}
G2_SEEDS = {6: {"A": (9826, 9827, 9828), "B": (9829, 9830, 9831)}, 4: {"A": (9826, 9827), "B": (9829, 9830)}}
B1_SEEDS = {8: {"A": (9853, 9854, 9855, 9873, 9874), "B": (9856, 9857, 9858, 9875, 9876)},
            6: {"A": (9853, 9854, 9855), "B": (9856, 9857, 9858)}, 4: {"A": (9853, 9854), "B": (9856, 9857)},
            2: {"A": (9853,), "B": (9856,)}}
# Family arms (§8, §9). The names are the arm modules of Part B item 12 (matched by canon()).
FAM = {"W": dict(K="K_S", F="F_S", X="X_S", frozen=None, tuned=("T", "S", "GF"), placebos=("P_S", "U_S"),
                 X4="X4_S", X12="X12_S", g2b_runs={"X_S": 6, "T": 4, "S": 4, "GF": 4, "P_S": 4, "U_S": 4, "X4_S": 4},
                 b1_runs={"X_S": 6, "F_S": 6, "V2s": 6, "T": 4, "S": 4, "GF": 4, "P_S": 4, "U_S": 4, "X4_S": 2,
                          "X12_S": 2}),
       "M": dict(K="K_mine", F="F_M", X="X_on", frozen="X_frozen", tuned=("DM_on", "S"), placebos=("P_on",),
                 X4="X4_on", X12="X12_on", g2b_runs={"X_on": 6, "X_frozen": 4, "DM_on": 4, "S": 4, "P_on": 4,
                                                     "X4_on": 4},
                 b1_runs={"X_on": 6, "F_M": 6, "V2s": 6, "X_frozen": 4, "DM_on": 4, "S": 4, "P_on": 4, "X4_on": 2,
                          "X12_on": 2})}
CLAIM_STEPS = {"OBJECT-ROOM": "KC-object", "GO": "KC-dev", "FASTER-2X": "FASTER-PARTIAL",
               "FASTER-PARTIAL": "NOT-SIGNIFICANT"}
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


def _pack(runs, cells, units, arm, miss8=False, total=False, miss_value=None):
    N = max(len(runs[(g, s, arm)]) for g, s in cells)
    x, n, top = np.zeros((3 if miss8 else 2, len(units), 2, N)), np.zeros((len(units), 2), np.int64), []
    for (g, s), ks in cells.items():
        u, si = units.index(g), STARTS.index(s)
        for i, r in enumerate(runs[(g, s, arm)]):
            m = run_measures(r, ks, miss_value, total=total)
            x[0, u, si, i], x[1, u, si, i] = m["lam"], m["lam_late"]
            if miss8:
                x[2, u, si, i] = run_measures(r, ks, CAP8, total)["lam"]
            top.append(m["top_missed"])
        n[u, si] = len(runs[(g, s, arm)])
    return x, n, float(np.mean(top))


def contrast(runs, K_gs, arm, comp, gate="G1", B=B_VERDICT, clip=True, total=False, rng=None, miss_value=None,
             seed=SEED_BOOT):
    """R(arm:comp) per A.9 over the (g, s) cells of K_gs where both arms have runs (None if there are none).
    runs: {(game, start, canon arm): [run dicts]}; K_gs: {(game, start): [k, ...]}. miss_value: the T of a
    miss for both arms (default the cap; 8M for the G2 / B1 censoring sensitivity, same bootstrap draws);
    seed: the bootstrap seed family ([9805, .] development, [9859, .] sealed)."""
    a, c = canon(arm), canon(comp)
    cells = _cells(runs, K_gs, a, c)
    if not cells:
        return None
    units = sorted({g for g, _ in cells})
    la, na, cens_a = _pack(runs, cells, units, a, False, total, miss_value)
    lc, nc, cens_c = _pack(runs, cells, units, c, True, total, miss_value)
    draws = None
    if B:
        if rng is None:
            rng = np.random.default_rng([seed, zlib.crc32(f"{gate}:{a}:{c}{':total' if total else ''}".encode())])
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
    """§7.3. g1_rows: {row: A.11 label}; g1p_rows: {row: G1-P label}. A ROOM-K row that could license a family
    (O_4, K*mac) must carry its G1-P label: without it there is no routing (ValueError), never a licence."""
    if not any(v == "ROOM-K" for v in g1_rows.values()):
        return dict(result="v85 ends (§11)", families=[])
    lack = [r for r in ("o4", "kmac") if g1_rows.get(r) == "ROOM-K"
            and g1p_rows.get(r) not in ("GENERIC-K", "SEPARATED-K", "NOT-SEPARATED-K")]
    if lack:
        raise ValueError(f"no G1-P label for ROOM-K row(s) {lack}: G1-P incomplete, no routing (§7.2-7.3)")
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
    """A.10.4 for one (g, s): k_top = the highest k in K_g reached by all 3 V2s runs at the other start s'
    (k_ref if s' is invalid); K_{g,s} = {k in K_g : k <= k_top}; [] (dropped) if fewer than 2. With a valid s'
    and other than exactly 3 V2s runs there, no k_top and no K (complete False: incomplete, not a drop)."""
    if not other_valid:
        kt, n = k_ref, 0
    else:
        n = len(other_runs)
        tops = []
        for r in other_runs:
            hit = [k for k, t in zip(K_g, run_times(r, K_g)) if t is not None]
            tops.append(max(hit) if hit else None)
        kt = None if (n != 3 or None in tops) else min(tops)
    K = [k for k in K_g if kt is not None and k <= kt]
    return dict(k_top=kt, K=K if len(K) >= 2 else [], n_other=n, complete=(not other_valid) or n == 3)


def _exp_meta(r):
    h = (r.get("exp") or {}).get("meta_sha256") or r.get("exp_meta_sha256") or (r.get("summary") or {}).get("exp_meta_sha256")
    if h is None and r.get("exp_meta") is not None:
        h = hashlib.sha256(json.dumps(r["exp_meta"], sort_keys=True).encode()).hexdigest()
    return h


def certificate(originals, reruns, first_unit, closure=None):
    """§13.4 reuse certificate: V2s seeds 9806 (A) and 9809 (B) on the first unit in draw order, re-executed under
    the new closure, must equal the originals' records and exp_meta byte for byte. closure: the gate's committed
    V2s closure; only reruns made under it count."""
    checks = []
    for s, seed in (("A", 9806), ("B", 9809)):
        pick = lambda rs: [r for r in rs if r["game"] == first_unit and r["start"] == s and int(r["seed"]) == seed
                           and canon(r["arm"]) == "v2s"]
        o, n = pick(originals), pick(reruns)
        if closure is not None:
            n = [r for r in n if (r.get("provenance") or {}).get("closure_hash") == closure]
        if not o:
            continue                                     # invalid start: no such I-stage run
        ok = bool(n) and json.dumps(o[0]["records"]) == json.dumps(n[0]["records"]) \
            and _exp_meta(o[0]) is not None and _exp_meta(o[0]) == _exp_meta(n[0])
        checks.append(dict(start=s, seed=seed, rerun=bool(n), ok=bool(ok)))
    return dict(ok=bool(checks) and all(c["ok"] for c in checks), unit=first_unit, checks=checks)


def certificate_unit(starts, istage_runs):
    """§13.4: the first unit in A.2 draw order that has I-stage V2s runs."""
    have = {r["game"] for r in istage_runs if canon(r["arm"]) == "v2s"}
    return next((g for g in a2_order(starts, have)), None)


def interruptions(run_list, committed=None, cert=None, reused=("v2s",), require_committed=False, replay_exempt=()):
    """A.13 / §13.4: replay mismatch, dirty tree, missing or mixed closure hashes per arm type, a closure differing
    from the gate's committed one, and (for reused arms) a missing or failed comparator certificate.
    require_committed: the gate has committed closures (G1, G1-P): none, or none for an arm type, is an
    Interruption. replay_exempt: arm types whose confirmations are judged elsewhere (D, A.10.2)."""
    why, by_arm = [], {}
    if require_committed and not committed:
        why.append("no committed closure hashes for this gate (F1 closure_hashes missing)")
    for r in run_list:
        tag = f"{r.get('game')}/{r.get('start')}/{r.get('arm')}/{r.get('seed')}"
        if canon(r.get("arm")) not in replay_exempt and (
                r.get("interruption") or any(not c.get("ok") for c in r.get("replay") or [])):
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
        elif require_committed and committed and a not in com:
            why.append(f"no committed closure hash: {NAMES.get(a, a)}")
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


MEASURES = ("fidelity", "fidelity_held", "fidelity_slot", "share_priv_or_noop", "share_priv", "share_unmapped",
            "q5_expansions", "oracle_frames", "rank_corr", "coverage", "greedy_share", "share_not_first",
            "share_one_cell")
PER_START = ("size", "hold", "bandit_k", "map_size", "windows", "rises", "m")


def instrument_summary(rs):
    """Run means of the measurement fields of the runs' "instrument" captions (A.7, A.13: fidelity, priv / NOOP
    shares, Q5 use, oracle frames, ...); constants and library facts (size, hold, bandit k, m, map size) are
    listed per (g, s), never averaged as measurements."""
    out = {}
    for k in MEASURES:
        v = [r["instrument"][k] for r in rs if isinstance((r.get("instrument") or {}).get(k), (int, float))
             and not isinstance(r["instrument"][k], bool)]
        if v:
            out[k] = float(np.mean(v))
    per = {}
    for r in rs:
        i = r.get("instrument") or {}
        d = {k: i[k] for k in PER_START if k in i}
        if d:
            per.setdefault(f"{r['game']}/{r['start']}", d)
    if per:
        out["per_start"] = per
    return out


def o_ties(rs):
    """Pooled O-TIES share (A.7: share of expansions where all scores tie): per-run o_ties weighted by the
    run's scored expansions (OChooser.caption "expansions"), else their mean."""
    ins = [r.get("instrument") or {} for r in rs]
    if ins and all(i.get("o_ties") is not None and isinstance(i.get("expansions"), int) for i in ins):
        n = sum(i["expansions"] for i in ins)
        return sum(i["o_ties"] * i["expansions"] for i in ins) / n if n else None
    v = [i.get("o_ties") for i in ins if i.get("o_ties") is not None]
    return float(np.mean(v)) if v else None


def group(run_list):
    """{(game, start, canon arm): runs sorted by seed}; two runs of one (game, start, arm, seed) are refused
    (e.g. first-pass D outputs left beside the reruns, §10 allowed change 5)."""
    runs, seen = {}, set()
    for r in run_list:
        key = (r["game"], r["start"], canon(r["arm"]), int(r["seed"]))
        if key in seen:
            raise ValueError(f"two runs of {key}: move first-pass outputs to a subdirectory")
        seen.add(key)
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
        if isinstance(v, dict) and v.get("kept") is False:
            out.pop(g, None)                             # dropped by S0 (A.3.5, A.4): no cell, never replaced
            continue
        kg = list(v if isinstance(v, list) else (v.get("K_g") if v.get("K_g") is not None else v.get("K")) or [])
        if len(kg) < 2:
            out.pop(g, None)
            continue
        kr = v.get("k_ref") if isinstance(v, dict) else None
        out[g] = (kg, int(kr) if kr is not None else max(kg))
    return out


def a2_order(starts, games):
    """The A.2 draw order (s0/starts.json "order", drops removed) restricted to `games`."""
    if not starts or "order" not in starts:
        raise SystemExit("s0/starts.json with the A.2 'order' is required (A.2, §13.4)")
    games = set(games)
    stray = games - set(starts["order"])
    if stray:
        raise SystemExit(f"units outside s0/starts.json 'order': {sorted(stray)}")
    kept = set(starts["kept"]) if "kept" in starts else {g for g in starts["order"]
                                                          if starts["units"][g].get("kept", True)}
    return [g for g in starts["order"] if g in games and g in kept]


def kept_order(starts):
    """Kept units in A.2 order (s0 'kept', else 'order' filtered by each unit's kept flag)."""
    return a2_order(starts, starts["order"])


def istage_units(starts, istage_runs, U=None):
    """The I-stage units: the first U kept units in A.2 order, extended in order to every kept unit with I-stage
    runs (the one top-up takes the next units in order). Returns (units, stray units with runs not kept)."""
    kept = kept_order(starts)
    have = {r["game"] for r in istage_runs}
    last = max((kept.index(g) for g in have if g in kept), default=-1)
    return kept[:max(int(U or 0), last + 1)], sorted(have - set(kept))


def _seeds_present(runs, g, s, arm, seeds):
    have = {int(r["seed"]) for r in runs.get((g, s, canon(arm)), [])}
    return [f"{g}/{s}/{NAMES.get(canon(arm), arm)}/{sd}" for sd in seeds if int(sd) not in have]


def missing_istage(runs, starts, units):
    """A.10.2: 3 V2s (9806-9808 / 9809-9811) and 1 D (seed E_s) per valid start of each I-stage unit."""
    miss = []
    for g in units:
        for s in STARTS:
            st = starts["units"][g].get(s)
            if st and st.get("valid"):
                miss += _seeds_present(runs, g, s, "v2s", V2S_SEEDS[s]) + _seeds_present(runs, g, s, "d", [st["E"]])
    return miss


def stale_d(runs, d_fix):
    """After the one dated D fix (A.10.2): D runs launched before it are first-pass outputs left in istage/."""
    if not d_fix:
        return []
    return [f"first-pass D output launched {(r.get('provenance') or {}).get('launched')} before the D fix {d_fix}: "
            f"{r['game']}/{r['start']}/{r['seed']} (move it to istage/d_first_pass/)"
            for (g, s, a), rs in sorted(runs.items()) if a == "d" for r in rs
            if str((r.get("provenance") or {}).get("launched") or "") < str(d_fix)]


def g1_expected(K_gs, starts, n, n_O4=None, o_dropped=False):
    """§7.1: {(g, s, row): seeds} for the core rows: n/2 per kept (g, s) (n_O4/2 for O_4, §13.1), K*mac only
    where the other start is valid (A.7)."""
    out = {}
    for (g, s), ks in sorted(K_gs.items()):
        if len(ks) < 2:
            continue
        o = "B" if s == "A" else "A"
        other_ok = bool((starts["units"][g].get(o) or {}).get("valid"))
        for row in ROWS:
            if (row == "o4" and o_dropped) or (row == "kmac" and not other_ok):
                continue
            m = int(n_O4 or n) if row == "o4" else int(n)
            out[(g, s, row)] = [G1_SEED0[s] + i for i in range(m // 2)]
    return out


def missing_g1(runs, K_gs, starts, n, n_O4=None, o_dropped=False):
    miss = []
    for (g, s, row), seeds in g1_expected(K_gs, starts, n, n_O4, o_dropped).items():
        miss += _seeds_present(runs, g, s, row, seeds)
    for (g, s), ks in sorted(K_gs.items()):
        if len(ks) >= 2:
            miss += _seeds_present(runs, g, s, "v2s", V2S_SEEDS[s])
    return miss


def start_validity(starts):
    """{(game, start): valid} from S0's starts.json ({"units": {game: {"A": {"valid": ..} or null, ...}}})."""
    return {(g, s): bool(u.get(s) and u[s].get("valid", True)) for g, u in (starts or {}).get("units", {}).items()
            for s in STARTS}


def istage_kgs(runs, rungs=None, valid=None):
    """A.10.4 for every (g, s) of the I-stage. valid {(g, s): bool} from S0 (start_validity); without it a start
    is valid iff it has V2s runs. A valid other start without its 3 V2s runs gives no k_top (no cell). Only
    kept units with I-stage V2s runs get cells (units outside the I-stage never do)."""
    ur = unit_rungs(runs, rungs)
    K_gs, detail = {}, {}
    have = {g for (g, s, a) in runs if a == "v2s"}
    for g, (kg, kr) in sorted(ur.items()):
        if g not in have:
            continue
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


def d_control(runs, K_gs, units, K_g=None):
    """A.10.2: r_g(D:V2s) >= 4 on >= 80% of units, every D confirmation exact. K_g {unit: [k..]}: a unit with
    no cross-fitted cell is scored on its uncross-fitted K_g (the D control precedes the cross-fit)."""
    cells, fb = dict(K_gs), []
    for g in units:
        if not any(x == g and len(ks) >= 2 for (x, _), ks in K_gs.items()) and K_g and len(K_g.get(g) or []) >= 2:
            for s in STARTS:
                if runs.get((g, s, "d")) and runs.get((g, s, "v2s")):
                    cells[(g, s)] = list(K_g[g])
            fb.append(g)
    st = contrast(runs, cells, "d", "v2s", gate="I", B=0)
    r = {g: (st["r_g"].get(g) if st else None) for g in units}
    share = float(np.mean([v is not None and v >= 4 for v in r.values()])) if units else 0.0
    exact = all(_confirmed(x) for (g, s, a), rs in runs.items() if a == "d" for x in rs)
    return dict(r_g=r, share=share, exact=exact, passed=bool(share >= 0.8 and exact), uncross_fitted_units=fb)


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


def istage_stage(missing, reasons, D, line, d_fix=None):
    """First match: INCOMPLETE; Interruption; D failing (D-CONTROL-FAIL before the one dated fix,
    Interruption (D) after it, A.10.2 / Part C); INSTRUMENT-FAIL; PASS."""
    if missing:
        return "INCOMPLETE"
    if reasons:
        return "Interruption"
    if not D["passed"]:
        return "Interruption (D)" if d_fix else "D-CONTROL-FAIL"
    if line["instrument_fail"] is not False:
        return "INSTRUMENT-FAIL"
    return "PASS"


def istage_report(runs, units, rungs=None, natural=None, B=B_VERDICT, valid=None, missing=(), committed=None,
                  d_fix=None, cert=None):
    """A.10.2-A.10.4. missing: expected runs not present (missing_istage); committed: the closures committed
    before the I-stage, if any; d_fix: the date of the one dated D fix (A.10.2) when D has been rerun."""
    K_gs, detail = istage_kgs(runs, rungs, valid)
    ur = unit_rungs(runs, rungs)
    out = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.10.2-A.10.4", units=units, K_gs=kgs_to_json(K_gs), cross_fit=detail,
               d_control=d_control(runs, K_gs, units, {g: kg for g, (kg, kr) in ur.items()}),
               instrument_line=instrument_line(runs, K_gs, B), missing=list(missing), d_fix=d_fix)
    if natural:
        st = contrast({**runs, **natural}, K_gs, "natural", "v2s", gate="I", B=B)
        out["regime_ratio"] = dict(name="R(natural V2 from its own M1 : V2s from s1), on the counted rungs",
                                   R=st["R"], LB90=st["LB90"], UB90=st["UB90"], r_g=st["r_g"]) if st else None
    gated = [r for (g, s, a), v in runs.items() if a in ("v2s", "d") for r in v]
    out["interruptions"] = interruptions(gated, committed, cert, replay_exempt=("d",))   # V2s: certificate rule
    out["certificate"] = cert
    out["stage"] = istage_stage(missing, out["interruptions"], out["d_control"], out["instrument_line"], d_fix)
    if out["stage"] in ("INCOMPLETE", "Interruption"):
        out["K_gs"] = None                               # nothing may be calibrated or frozen from it
    return out


def dice_caption(runs, K_gs, B=B_VERDICT, g4=None):
    """A.7 O_4-dice caption: runs aborted by DiceMismatch counted and left out of R; R beside S0's G_4."""
    dice = {k: v for k, v in runs.items() if k[2] == "o4dice"}
    ab = [f"{r['game']}/{r['start']}/{r['seed']}: {r['aborted']}" for v in dice.values() for r in v if r.get("aborted")]
    clean = dict(runs)
    for k, v in dice.items():
        clean[k] = [r for r in v if not r.get("aborted")]
    st = contrast(clean, K_gs, "o4dice", "v2s", gate="G1C", B=B) or {}
    per = (g4 or {}).get("per_unit") or {}
    cells = [c.split("/") for c in st.get("cells", [])]
    pred = [per[g][s] for g, s in cells if (per.get(g) or {}).get(s) is not None]
    return dict(st, runs=sum(len(v) for v in dice.values()), aborted=len(ab), aborted_runs=ab,
                G4_pred_cells=float(np.mean(pred)) if pred else None,
                G4_S0=(g4 or {}).get("G4"), G4_S0_dice_starts=(g4 or {}).get("G4_dice_starts"))


def g1_report(runs, K_gs, thr, istage_runs=None, committed=None, cert=None, o_dropped=False, B=B_VERDICT,
              a12_rerun=None, missing=(), require_committed=False, g4=None):
    """A.11-A.13. runs: G1 rows plus the reused V2s; thr: {"n", "tau_pt", "tau_lb"} (+ "n_O4", §13.1);
    a12_rerun(row, v_K, tau_K) -> {criterion: passed} (default: scripts/v85_calib.rerun_a12 on the I-stage V2s,
    at n_O4 for O_4). missing: expected runs not present: nothing is labelled (G1 = "INCOMPLETE")."""
    if missing:
        return dict(rule="docs/ARC4_V85_PROTOCOL_EN.md A.11-A.13, §7.1", G1="INCOMPLETE", rows={}, missing=list(missing),
                    interruptions=[], thresholds=thr, sentences=["Incomplete: no label until every expected run exists."],
                    caption_rows={}, claim_header=CLAIM, senses=SENSES)
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
                a12_rerun = lambda r_, v_, t_: rerun_a12(cells, sorted({g for g, _ in cells}), row_thr(thr, r_), v_,  # noqa: E731
                                                         t_)["passed"]
            oc = a12_rerun(row, p["v_K"], p["tau_K"])
            lab["a12"]["oc"] = oc
            lab["label"] = a12_downgrade(lab["label"], oc)
            if lab["label"] != "ROOM-K" and "BELOW-2X" in lab["captions"]:
                lab["captions"].remove("BELOW-2X")
        if rd and rd["R"] != 1 and st["R"] > 0:
            st["logR_over_logR_D"] = math.log(st["R"]) / math.log(rd["R"])
        rows[row] = dict(stats=st, **lab)
        labels[row], ub[row] = lab["label"], st["UB95"]
    caps = {}                                                        # caption rows: reported, never labelled
    if any(a == "o4dice" for (g, s, a) in runs):
        caps["O_4-dice"] = dice_caption(runs, K_gs, B, g4)
    if any(a == "kmacexact" for (g, s, a) in runs):
        caps["K*mac-exact"] = contrast(runs, K_gs, "kmacexact", "v2s", gate="G1C", B=B)
    gated = [r for (g, s, a), v in runs.items() if a in ROWS + ("v2s",) and not (o_dropped and a == "o4") for r in v]
    reasons = interruptions(gated, committed, cert, require_committed=require_committed)
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


def row_thr(thr, row):
    """The frozen (n, tau_pt, tau_lb) of a row: O_4 at n_O4 (§13.1 budget rule)."""
    return dict(thr, n=int(thr.get("n_O4") or thr["n"])) if canon(row) == "o4" else dict(thr)


def g1p_expected(g1_exp, live):
    """§7.2: the placebo of each ROOM row at the row's own (g, s, seeds)."""
    return {(g, s, canon(PLACEBO[row])): seeds for (g, s, row), seeds in g1_exp.items() if row in live}


def g1p_report(v, runs, K_gs, committed=None, missing=(), B=B_VERDICT, require_committed=False):
    """§7.2-7.3 after a ROOM verdict. v: g1_verdict.json. Routing only when every ROOM row has its complete
    placebo contrast and no Interruption; otherwise status INCOMPLETE / Interruption / NOT-RUN and no routing."""
    base = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §7.2-7.3", claim_header=CLAIM, senses=SENSES)
    if v.get("G1") != "ROOM":
        return dict(base, status="NOT-RUN", rows={}, interruptions=[], routing=None,
                    reason=f"G1 is {v.get('G1')!r}: G1-P runs only after a ROOM verdict (§7.2)")
    live = [r for r, x in v["rows"].items() if isinstance(x, dict) and x.get("label") == "ROOM-K"]
    res = {}
    for row in live:
        st = contrast(runs, K_gs, row, PLACEBO[row], gate="G1P", B=B)
        res[row] = dict(stats=st, label=g1p_label(st) if st else None)
    arms = set(live) | {canon(PLACEBO[r]) for r in live}
    reasons = interruptions([r for (g, s, a), rs in runs.items() if a in arms for r in rs], committed, None,
                            reused=(), require_committed=require_committed)
    out = dict(base, rows=res, interruptions=reasons, missing=list(missing), routing=None)
    if missing or any(x["label"] is None for x in res.values()):
        return dict(out, status="INCOMPLETE")
    if reasons:
        return dict(out, status="Interruption")
    return dict(out, status="COMPLETE", routing=routing({r: x["label"] for r, x in v["rows"].items()},
                                                        {r: x["label"] for r, x in res.items()}))


# ---------------------------------------------------------------- G2a, G2b, B1 (§8, §9, Part C)

def drop_step(label):
    return CLAIM_STEPS.get(label, label)


def sensitive(label, label8):
    """CENSORING-SENSITIVE (Part C): a claim label that changes with misses at 2x cap drops one step."""
    s = label in CLAIM_STEPS and label8 != label
    return dict(label=drop_step(label) if s else label, primary=label, label_8M=label8,
                captions=["CENSORING-SENSITIVE"] if s else [])


def expected_runs(K_gs, plan, seeds_by_n, which=None):
    """{(g, s, arm): seeds} for plan {arm: runs per unit} (half per start, §12 seed sets)."""
    out = {}
    for (g, s), ks in sorted(K_gs.items()):
        if len(ks) < 2:
            continue
        for arm, n in plan.items():
            if which is None or arm in which:
                out[(g, s, canon(arm))] = list(seeds_by_n[n][s])
    return out


def missing_runs(runs, exp):
    miss = []
    for (g, s, a), seeds in exp.items():
        miss += _seeds_present(runs, g, s, a, seeds)
    return miss


def g2a_report(runs, K_gs, fam, B=B_VERDICT, missing=()):
    """§8.1 for one family: OBJECT-ROOM / KC-object, the ceiling-to-label rule, CENSORING-SENSITIVE."""
    A = FAM[fam]
    if missing:
        return dict(family=fam, status="INCOMPLETE", missing=list(missing), label=None)
    kf = contrast(runs, K_gs, A["K"], A["F"], gate="G2a", B=B)
    kv = contrast(runs, K_gs, A["K"], "v2s", gate="G2a", B=B)
    if kf is None or kv is None:
        return dict(family=fam, status="INCOMPLETE", missing=["no runs"], label=None)
    lab = g2a_label(kf, kv)
    kf8 = contrast(runs, K_gs, A["K"], A["F"], gate="G2a", B=B, miss_value=CAP8)
    kv8 = contrast(runs, K_gs, A["K"], "v2s", gate="G2a", B=B, miss_value=CAP8)
    sens = sensitive(lab["label"], g2a_label(kf8, kv8)["label"])
    return dict(family=fam, status="COMPLETE", **sens, sealed_top_label=lab["sealed_top_label"],
                stats={"K:F": kf, "K:V2s": kv}, stats_8M={"K:F": kf8, "K:V2s": kv8})


def g2b_family(reports):
    """§7.3: G2b for the family that passes G2a with the larger point log R(K:F); a tie goes to M."""
    ok = {f: r["stats"]["K:F"]["R"] for f, r in reports.items() if r.get("label") == "OBJECT-ROOM"}
    if not ok:
        return None
    best = max(ok.values())
    return "M" if ok.get("M") == best else next(f for f, R in ok.items() if R == best)


def interaction(runs, K_gs, B=B_VERDICT, seed=SEED_BOOT, miss_value=None):
    """§8.2 family M: I = mean_g of [lam(X_frozen) - lam(X_on)] - [lam(V2s) - lam(F_M)], with a two-level
    bootstrap interval (units, then runs within each (unit, start, arm))."""
    arms = [canon(x) for x in ("X_on", "X_frozen", "F_M", "v2s")]
    cells = {(g, s): list(ks) for (g, s), ks in sorted(K_gs.items())
             if len(ks) >= 2 and all(runs.get((g, s, a)) for a in arms)}
    if not cells:
        return None
    units = sorted({g for g, _ in cells})
    P = [_pack(runs, cells, units, a, False, False, miss_value)[:2] for a in arms]
    valid = np.asarray(P[0][1] > 0, float)
    m = [_cmean(x[0], n) for x, n in P]                                 # (U, S)
    d = (m[1] - m[0]) - (m[3] - m[2])
    I = float(((d * valid).sum(-1) / valid.sum(-1)).mean())
    out = dict(I=I, U=len(units), units=units)
    if B:
        rng = np.random.default_rng([seed, zlib.crc32(b"G2b:interaction")])
        Iu = rng.integers(0, len(units), size=(B, len(units)))
        vI = valid[Iu]
        wt = vI / vI.sum(-1, keepdims=True)
        bm = [_bmean(x[:1], n, Iu, rng.random((B, len(units), 2, x.shape[-1])), wt)[0] for x, n in P]
        bI = ((bm[1] - bm[0]) - (bm[3] - bm[2])).mean(-1)
        q = np.percentile(bI, (5.0, 10.0, 90.0, 95.0))
        out.update(LB90=float(q[1]), UB90=float(q[2]), CI90=[float(q[0]), float(q[3])], B=int(B))
    return out


def g2b_report(runs, K_gs, fam, B=B_VERDICT, R_KF=None, missing=()):
    """§8.2: GO / KC-dev, CENSORING-SENSITIVE; reported phi, the X4 dose, and for family M R(X_on:X_frozen),
    HEAD-START-VOCABULARY / ADAPTS and the interaction I."""
    A = FAM[fam]
    if missing:
        return dict(family=fam, status="INCOMPLETE", missing=list(missing), label=None)
    comps = ("v2s",) + A["tuned"] + A["placebos"]

    def stats(mv):
        xf = contrast(runs, K_gs, A["X"], A["F"], gate="G2b", B=B, miss_value=mv)
        return xf, {NAMES.get(canon(c), c): contrast(runs, K_gs, A["X"], c, gate="G2b", B=B, miss_value=mv) for c in comps}
    xf, others = stats(None)
    if xf is None or any(v is None for v in others.values()):
        return dict(family=fam, status="INCOMPLETE", missing=["no runs for a named contrast"], label=None)
    lab = g2b_label(xf, others)
    xf8, others8 = stats(CAP8)
    sens = sensitive(lab["label"], g2b_label(xf8, others8)["label"])
    out = dict(family=fam, status="COMPLETE", **sens, clauses=lab["clauses"], stats=dict({"X:F": xf}, **others),
               phi=(math.log(xf["R"]) / math.log(R_KF)) if R_KF and R_KF != 1 and xf["R"] > 0 else None,
               dose={"X4:F": contrast(runs, K_gs, A["X4"], A["F"], gate="G2b", B=B),
                     "X:X4": contrast(runs, K_gs, A["X"], A["X4"], gate="G2b", B=B)})
    if A["frozen"]:
        of = contrast(runs, K_gs, A["X"], A["frozen"], gate="G2b", B=B)
        I = interaction(runs, K_gs, B)
        out.update(on_frozen=of, interaction=I,
                   vocab=vocab_caption(of) if of else None,
                   accelerates_in_run_learning=bool(of and of["LB90"] > 1 and I and I.get("LB90", -1) > 0))
    return out


def sign_flip_p(lr, seed=SEED_SEALED, draws=65536):
    """One-sided p of mean(lr) > 0 under unit-level sign flips: exact up to 16 units, else random on [seed, 2]."""
    lr = np.asarray(lr, float)
    U = len(lr)
    if U <= 16:
        signs = 1 - 2 * ((np.arange(2 ** U)[:, None] >> np.arange(U)[None, :]) & 1)
    else:
        signs = np.where(np.random.default_rng([seed, 2]).random((draws, U)) < 0.5, -1, 1)
    t = (signs * lr).mean(-1)
    return float(np.mean(t >= lr.mean() - 1e-12))


def page_L(means, seed=SEED_SEALED, draws=20000):
    """Page's L for the increasing order of speed over the dose columns of means (U, k) of lam (lower lam =
    faster): ranks within unit by speed (1 = slowest), L = sum_j j * R_j; p by within-unit permutations."""
    means = np.asarray(means, float)
    U, k = means.shape
    ranks = np.argsort(np.argsort(-means, axis=1, kind="stable"), axis=1) + 1.0   # largest lam -> rank 1
    w = np.arange(1, k + 1)
    L = float((ranks.sum(0) * w).sum())
    rng = np.random.default_rng([seed, 3])
    perm = np.argsort(rng.random((draws, U, k)), axis=-1) + 1.0
    Ls = (perm.sum(1) * w).sum(-1)
    return dict(L=L, p=float(np.mean(Ls >= L - 1e-12)), U=U)


def b1_label(xf, p, logo_min, allow_2x, vs, tuned_v2s):
    """Part C B1 ladder, first match. xf: R(X:F) stats; vs: {"V2s"|tuned|placebo name: R(X:c) stats} with
    keys grouped by tuned_v2s (tuned names -> R(tuned:V2s) stats); placebos are the other keys."""
    tuned = list(tuned_v2s)
    plac = [c for c in vs if c != "V2s" and c not in tuned]
    late = xf["LB90_late"] > 1
    partial_l = (xf["R"] >= 1.25 and xf["LB90"] > 1 and p <= 0.025
                 and all(vs[c]["LB90"] > 1 for c in ["V2s"] + tuned + plac))
    if xf["UB95"] < 1:
        return "NEGATIVE-TRANSFER"
    if (allow_2x and xf["R"] >= 2.0 and xf["LB90"] >= 1.3 and p <= 0.025
            and all(vs[c]["R"] >= 1.5 and vs[c]["LB90"] > 1 for c in ["V2s"] + tuned)
            and all(vs[c]["LB90"] > 1 for c in plac) and late and logo_min is not None and logo_min > 1):
        return "FASTER-2X"
    if partial_l and late:
        return "FASTER-PARTIAL"
    if partial_l:
        return "EARLY-RUNGS-ONLY"
    if any(tuned_v2s[c]["LB90"] > 1 and vs[c]["LB90"] <= 1 for c in tuned):
        return "TUNED-DEFAULTS"
    if xf["LB90"] > 1 and all(vs[c]["LB90"] > 1 for c in ["V2s"] + tuned) and any(vs[c]["LB90"] <= 1 for c in plac):
        return "GENERIC"
    if xf["R"] >= 1.25:
        return "NOT-SIGNIFICANT"
    return "NULL"


def b1_report(runs, K_gs, fam, allow_2x, B=B_VERDICT, missing=(), line=None):
    """§9 B1 for one family on sealed runs (bootstrap on [9859, .]); the label ladder of Part C with
    CENSORING-SENSITIVE, ACCUMULATES, the vocabulary caption (M); INSTRUMENT-FAIL makes it INCONCLUSIVE."""
    A = FAM[fam]
    if missing:
        return dict(family=fam, status="INCOMPLETE", missing=list(missing), label=None)

    def one(mv):
        xf = contrast(runs, K_gs, A["X"], A["F"], gate="B1", B=B, miss_value=mv, seed=SEED_SEALED)
        if xf is None:
            return None
        vs = {NAMES.get(canon(c), c): contrast(runs, K_gs, A["X"], c, gate="B1", B=B, miss_value=mv, seed=SEED_SEALED)
              for c in ("v2s",) + A["tuned"] + A["placebos"]}
        tv = {NAMES.get(canon(c), c): contrast(runs, K_gs, c, "v2s", gate="B1", B=B, miss_value=mv, seed=SEED_SEALED)
              for c in A["tuned"]}
        if any(v is None for v in list(vs.values()) + list(tv.values())):
            return None
        lr = np.log(np.clip([xf["r_g"][g] for g in xf["units"]], 1 / 8, 8))
        p = sign_flip_p(lr)
        logo = min(float(np.exp(np.delete(lr, i).mean())) for i in range(len(lr))) if len(lr) > 1 else None
        return dict(xf=xf, vs=vs, tv=tv, p=p, logo_min=logo,
                    label=b1_label(xf, p, logo, allow_2x, vs, {k: tv[k] for k in tv}))
    s, s8 = one(None), one(CAP8)
    if s is None or s8 is None:
        return dict(family=fam, status="INCOMPLETE", missing=["no runs for a named contrast"], label=None)
    out = dict(family=fam, status="COMPLETE", **sensitive(s["label"], s8["label"]), p=s["p"], logo_min=s["logo_min"],
               allow_2x=bool(allow_2x), stats=dict({"X:F": s["xf"]}, **s["vs"]), tuned_vs_V2s=s["tv"],
               header="SEARCH WITH BILLED REWINDS FROM A SHARED SAVESTATE", senses=SENSES)
    if line is not None and line.get("instrument_fail") is not False:
        out.update(label="INCONCLUSIVE", instrument_line=line, captions=out["captions"] + ["INSTRUMENT-FAIL"])
    elif line is not None:
        out["instrument_line"] = line
    doses = [A["F"], A["X4"], A["X12"], A["X"]]
    cells = {k: v for k, v in K_gs.items() if len(v) >= 2 and all(runs.get((k[0], k[1], canon(a))) for a in doses)}
    units = sorted({g for g, _ in cells})
    if units:
        mean = np.zeros((len(units), 4))
        for j, a in enumerate(doses):
            x, n, _ = _pack(runs, cells, units, canon(a))
            v = (n > 0).astype(float)
            mean[:, j] = (_cmean(x[0], n) * v).sum(-1) / v.sum(-1)
        pl = page_L(mean)
        x41 = contrast(runs, K_gs, A["X"], A["X4"], gate="B1", B=B, seed=SEED_SEALED)
        out["accumulates"] = dict(page=pl, X41_X4=x41, ACCUMULATES=bool(pl["p"] <= 0.05 and x41 and x41["LB90"] > 1))
    if A["frozen"]:
        of = contrast(runs, K_gs, A["X"], A["frozen"], gate="B1", B=B, seed=SEED_SEALED)
        out.update(on_frozen=of, vocab=vocab_caption(of) if of else None,
                   interaction=interaction(runs, K_gs, B, SEED_SEALED))
    return out


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


def _load(p):
    p = pathlib.Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _cert(out, starts, ist, committed):
    """§13.4 certificate from the reruns in istage/certificate/ and g1/certificate/ made under the committed V2s
    closure, on the first unit in A.2 order with I-stage V2s runs (None without reruns)."""
    reruns = load_runs(out / "istage" / "certificate", out / "g1" / "certificate")[0]
    if not reruns:
        return None
    clo = {canon(k): v for k, v in (committed or {}).items()}.get("v2s")
    return certificate(ist, reruns, certificate_unit(starts, ist), clo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("istage", "g1", "g1p", "g2a", "g2b", "b1"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rungs", default=None, help="S0 rungs JSON (default: s0/starts.json)")
    ap.add_argument("--o-dropped", action="store_true", help="O-confined KC-build: selection row not tested")
    ap.add_argument("--d-fix", default=None, help="istage: date of the one dated D fix (A.10.2), D rerun since")
    ap.add_argument("--U", type=int, default=None, help="istage: U when s0/natural_spikeins.json has none")
    ap.add_argument("--family", choices=("M", "W"), help="b1: the family carried to the sealed test")
    ap.add_argument("--sealed", default=None, help="b1: sealed root with s0/starts.json and b1/ (default OUT/sealed)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    sp = out / "s0" / "starts.json"
    starts = _load(sp)
    if starts is None and a.cmd != "b1":
        raise SystemExit("no s0/starts.json: every verdict needs S0's A.2 order, validity and K_g")
    rungs = _load(a.rungs) if a.rungs else starts
    ist, ipaths = load_runs(out / "istage")
    iruns = group(ist)
    if a.cmd == "istage":
        U = a.U or (_load(out / "s0" / "natural_spikeins.json") or {}).get("U")
        units, stray = istage_units(starts, ist, U)
        missing = (missing_istage(iruns, starts, units) + [f"run of a unit that is not kept: {g}" for g in stray]
                   + stale_d(iruns, a.d_fix))
        from ragnarok.evaluation.start_v85 import ladder
        have = [g for g in units if (V84 / "bars" / f"{g}_9706.json").exists()
                and all((V84 / "g1b" / f"{g}_{x}.json").exists() for x in REF.values())]
        nat = natural_runs(have, {g: ladder(g) for g in have}) if have else None
        com = (_load(out / "closures.json") or {}).get("closures")      # committed before the I-stage (Part B)
        cert = _cert(out, starts, ist, com)                             # after a D fix that changed V2s (A.10.2)
        rep = istage_report(iruns, units, rungs, nat, valid=start_validity(starts), missing=missing,
                            committed=com, d_fix=a.d_fix, cert=cert)
        rep.update(U=U, claim_header=CLAIM, senses=SENSES,
                   provenance=stamp_out(ipaths + [sp] + ([out / "closures.json"] if com else [])))
        _write(out / "istage" / "istage_verdict.json", rep)
        print(json.dumps({k: rep[k] for k in ("stage", "d_control", "instrument_line", "interruptions")},
                         default=str)[:2000] + (f" missing={len(missing)}" if missing else ""))
        return
    if a.cmd == "b1":
        return main_b1(a, out)
    f1 = _load(out / "f1_thresholds.json")
    if f1 is None:
        raise SystemExit("no f1_thresholds.json")
    K_gs = kgs_from_json(f1["K_gs"])
    thr = {k: f1[k] for k in ("n", "tau_pt", "tau_lb")}
    thr["n_O4"] = int(f1.get("n_O4") or f1["n"])
    com = f1.get("closure_hashes")
    g1, gpaths = load_runs(out / "g1")
    g1_exp = g1_expected(K_gs, starts, thr["n"], thr["n_O4"], a.o_dropped)
    if a.cmd == "g1":
        runs = group(g1 + [r for r in ist if canon(r["arm"]) == "v2s"])
        missing = missing_g1(runs, K_gs, starts, thr["n"], thr["n_O4"], a.o_dropped)
        cpaths = load_runs(out / "istage" / "certificate", out / "g1" / "certificate")[1]
        units = a2_order(starts, {g for g, s in K_gs})
        cu = certificate_unit(starts, ist)
        cert = _cert(out, starts, ist, com)
        from scripts.v85_calib import cells_from_runs, rerun_a12
        cells = cells_from_runs(iruns, K_gs)

        def rerun(row, v, tau):
            return rerun_a12(cells, units, row_thr(thr, row), v, tau)["passed"]
        g4 = _load(out / "s0" / "dice_g4.json")
        rep = g1_report(runs, K_gs, thr, iruns, com, cert, a.o_dropped, a12_rerun=rerun, missing=missing,
                        require_committed=True, g4=g4)
        if thr["n_O4"] != thr["n"] and rep["rows"].get("o4"):
            cal = _load(out / "calib.json") or {}
            rep["rows"]["o4"]["power_at_n4"] = next(
                (dict(oc=t["oc"], criteria=t["criteria"]) for t in cal.get("table", [])
                 if (t["n"], t["tau"], t["tau_pt"], t["tau_lb"]) == (4, 0.2, thr["tau_pt"], thr["tau_lb"])), None)
        rep.update(units=units, certificate=cert, certificate_unit=cu,
                   provenance=stamp_out(ipaths + gpaths + cpaths + [out / "f1_thresholds.json", sp]))
        _write(out / "g1" / "g1_verdict.json", rep)
        print(json.dumps(dict(G1=rep["G1"], rows={r: v["label"] for r, v in rep["rows"].items()},
                              interruptions=rep["interruptions"], missing=len(rep.get("missing", [])))))
        return
    v = _load(out / "g1" / "g1_verdict.json")
    if v is None:
        raise SystemExit("no g1/g1_verdict.json")
    if a.cmd == "g1p":
        g1p, ppaths = load_runs(out / "g1p")
        runs = group(g1 + g1p)
        live = [r for r, x in v.get("rows", {}).items() if isinstance(x, dict) and x.get("label") == "ROOM-K"]
        missing = missing_runs(runs, g1p_expected(g1_exp, live)) + missing_runs(
            runs, {k: s for k, s in g1_exp.items() if k[2] in live})
        rep = g1p_report(v, runs, K_gs, com, missing, require_committed=True)
        rep["provenance"] = stamp_out(gpaths + ppaths + [out / "g1" / "g1_verdict.json"])
        _write(out / "g1p" / "g1p_verdict.json", rep)
        print(json.dumps(dict(status=rep["status"], rows={r: x["label"] for r, x in rep["rows"].items()},
                              routing=rep["routing"], missing=len(rep.get("missing", [])))))
        return
    v2s = [r for r in ist if canon(r["arm"]) == "v2s"]
    cert = _cert(out, starts, ist, com)                                # V2s reused under the certificate rule
    gp = _load(out / "g1p" / "g1p_verdict.json") or {}
    fams = ((gp.get("routing") or {}).get("families") or []) if gp.get("status") == "COMPLETE" else []
    g2a, apaths = load_runs(out / "g2a")
    if a.cmd == "g2a":
        if not fams:
            raise SystemExit("G1-P licensed no family (or is incomplete): G2a does not run (§7.3)")
        runs = group(g2a + v2s)
        reps = {}
        for f in fams:
            A = FAM[f]
            exp = expected_runs(K_gs, {A["K"]: 4, A["F"]: 6}, G2_SEEDS)
            reps[f] = g2a_report(runs, K_gs, f, missing=missing_runs(runs, exp))
            why = interruptions([r for (g, s, x), rs in runs.items() if x in {k[2] for k in exp} | {"v2s"}
                                 for r in rs], com, cert, require_committed=True)
            if why:
                reps[f].update(status="Interruption", label=None, interruptions=why)
        rep = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §8.1", families=reps, g2b_family=g2b_family(reps),
                   claim_header=CLAIM, senses=SENSES, provenance=stamp_out(apaths + [out / "g1p" / "g1p_verdict.json"]))
        _write(out / "g2a" / "g2a_verdict.json", rep)
        print(json.dumps(dict(labels={f: r["label"] for f, r in reps.items()}, g2b_family=rep["g2b_family"])))
        return
    va = _load(out / "g2a" / "g2a_verdict.json") or {}
    f = va.get("g2b_family")
    if not f:
        raise SystemExit("no G2a family passed OBJECT-ROOM: G2b does not run (§8.1)")
    g2b, bpaths = load_runs(out / "g2b")
    runs = group(g2a + g2b + v2s)
    A = FAM[f]
    exp = expected_runs(K_gs, A["g2b_runs"], G2_SEEDS)
    R_KF = ((va["families"][f].get("stats") or {}).get("K:F") or {}).get("R")
    why = interruptions([r for (g, s, x), rs in runs.items() if x in {canon(k) for k in A["g2b_runs"]}
                         | {canon(A["F"]), canon(A["K"]), "v2s"} for r in rs], com, cert, require_committed=True)
    g2b = g2b_report(runs, K_gs, f, R_KF=R_KF, missing=missing_runs(runs, exp))
    if why:
        g2b.update(status="Interruption", label=None)
    rep = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §8.2", **g2b, interruptions=why,
               claim_header=CLAIM, senses=SENSES,
               provenance=stamp_out(apaths + bpaths + [out / "g2a" / "g2a_verdict.json"]))
    _write(out / "g2b" / "g2b_verdict.json", rep)
    print(json.dumps(dict(family=f, label=rep["label"], status=rep["status"])))


def main_b1(a, out):
    """§9 B1: sealed starts and runs under --sealed (s0/starts.json, b1/*.json, closures.json of the F2 freeze);
    K_{g,s} cross-fitted from the sealed V2s runs alone (A.10.4 rule) before any other arm is read; the ceiling
    from g2a_verdict.json; G2 and B1 check closures like G1 (committed, per arm type; V2s reuse by certificate)."""
    if not a.family:
        raise SystemExit("b1 needs --family")
    root = pathlib.Path(a.sealed) if a.sealed else out / "sealed"
    st = _load(root / "s0" / "starts.json")
    if st is None:
        raise SystemExit(f"no {root / 's0' / 'starts.json'}")
    allr, paths = load_runs(root / "b1")
    v2s = group([r for r in allr if canon(r["arm"]) == "v2s"])
    K_gs, detail = istage_kgs(v2s, st, start_validity(st))
    runs = group(allr)
    A = FAM[a.family]
    exp = expected_runs(K_gs, dict(A["b1_runs"]), B1_SEEDS)
    if any(int(r["seed"]) in (9873, 9874, 9875, 9876) for r in allr):   # the 8-seed rule (§9 F2): X and F at 8
        exp.update(expected_runs(K_gs, {A["X"]: 8, A["F"]: 8}, B1_SEEDS))
    va = _load(out / "g2a" / "g2a_verdict.json") or {}
    allow = ((va.get("families") or {}).get(a.family) or {}).get("sealed_top_label") == "FASTER-2X"
    sub = {}
    for (g, s, x), rs in runs.items():
        if x == "v2s":
            for tag, seeds in zip(("linea", "linec"), LINE_SEALED):
                sub.setdefault((g, s, tag), []).extend(r for r in rs if int(r["seed"]) in seeds)
    lst = contrast(sub, K_gs, "linea", "linec", gate="B1", seed=SEED_SEALED)
    line = dict(R=lst["R"], CI99=lst["CI99"], instrument_fail=bool(lst["CI99"][0] > 1 or lst["CI99"][1] < 1)) \
        if lst else dict(R=None, CI99=None, instrument_fail=None)
    com = (_load(root / "closures.json") or {}).get("closures")      # the F2 sealed freeze's closures
    why = interruptions(allr, com, None, reused=(), require_committed=True)
    b1 = b1_report(runs, K_gs, a.family, allow, missing=missing_runs(runs, exp), line=line)
    if why:
        b1.update(status="Interruption", label=None)
    rep = dict(rule="docs/ARC4_V85_PROTOCOL_EN.md §9, Part C (B1)", K_gs=kgs_to_json(K_gs), cross_fit=detail,
               **b1, interruptions=why,
               claim_header=CLAIM, provenance=stamp_out(paths + [root / "s0" / "starts.json"]))
    _write(root / "b1" / "b1_verdict.json", rep)
    print(json.dumps(dict(family=a.family, label=rep["label"], status=rep["status"])))


if __name__ == "__main__":
    main()
