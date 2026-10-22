"""v83 climb statistics (docs/ARC4_V83_PROTOCOL.md §3; synthesis §2.4, §3 B1), evaluator side.

Per run: C = climb(records, ladder, k_star, cap, carried) from milestones_v83 (censoring always against
the carried arm, floor 5,000). Per game: l_g = mean log C(comparator) - mean log C(carried);
r_g = exp(l_g) clipped to [1/8, 8]; pooled R = exp(mean_g l_g). 95% interval: two-level bootstrap
(games, then seeds within each arm), 2000 resamples, percentiles. 'The better of A and B' as the
comparator (A3 WORTH) is taken per game: the comparator arm with the smaller mean log C on that game
(and on each resample) -- declared in docs/ARC4_V83_BUILD_NOTES.md before A3.
"""
import itertools

import numpy as np

from ragnarok.evaluation.milestones_v83 import climb, first_T

N_BOOT = 2000


def run_climbs(runs, lad, k_star, cap, carried, k_from=1):
    """runs: list of record lists ([(billed, true), ...]) -> list of C."""
    return [climb(r, lad, k_star, cap, carried, k_from) for r in runs]


def _lg(Cc, Cx):
    return float(np.mean(np.log(Cc)) - np.mean(np.log(Cx)))


def per_game(Cx, comps):
    """Cx: carried arm's C list; comps: list of comparator C lists (the better is used)."""
    best = min(comps, key=lambda c: np.mean(np.log(c)))
    return _lg(best, Cx)


def pooled(units):
    """units: {game: (Cx, [Ccomp...])} -> R, r_g dict, l_g dict."""
    l = {g: per_game(cx, cc) for g, (cx, cc) in units.items()}
    r = {g: float(np.clip(np.exp(v), 1 / 8, 8)) for g, v in l.items()}
    return float(np.exp(np.mean(list(l.values())))), r, l


def bootstrap(units, seed=9149, n=N_BOOT):
    """Two-level percentile interval of R."""
    rng = np.random.default_rng(seed)
    games = list(units)
    out = []
    for _ in range(n):
        ls = []
        for g in rng.choice(games, size=len(games), replace=True):
            cx, cc = units[g]
            cxb = [cx[i] for i in rng.integers(len(cx), size=len(cx))]
            ccb = [[c[i] for i in rng.integers(len(c), size=len(c))] for c in cc]
            ls.append(per_game(cxb, ccb))
        out.append(np.exp(np.mean(ls)))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summary(units):
    R, r, l = pooled(units)
    lo, hi = bootstrap(units)
    return dict(R=R, lo=lo, hi=hi, r_g=r, l_g=l, n_units=len(units))


def head_start(T1x, T1f):
    """Geometric mean over runs of T_F(M1) / T_X(M1) (pairs where both reach M1)."""
    v = [np.log(f / x) for x, f in zip(T1x, T1f) if x and f]
    return float(np.exp(np.mean(v))) if v else None


def permutation_p(units, n_mc=10000, seed=9149):
    """One-sided p for R(X:F) >= observed: X/F labels swapped within each game (single comparator)."""
    obs = np.mean([_lg(cc[0], cx) for cx, cc in units.values()])
    per = []
    for cx, cc in units.values():
        pool = list(cx) + list(cc[0])
        k = len(cx)
        splits = [(list(s), [i for i in range(len(pool)) if i not in s])
                  for s in itertools.combinations(range(len(pool)), k)]
        per.append([(_lg([pool[i] for i in b], [pool[i] for i in a])) for a, b in splits])
    total = int(np.prod([len(p) for p in per]))
    if total <= n_mc:
        vals = [np.mean(c) for c in itertools.product(*per)]
    else:
        rng = np.random.default_rng(seed)
        vals = [np.mean([p[rng.integers(len(p))] for p in per]) for _ in range(n_mc)]
    return float((np.sum(np.array(vals) >= obs - 1e-12)) / len(vals))


def logo_min(units):
    if len(units) < 2:
        return None
    return float(min(pooled({h: u for h, u in units.items() if h != g})[0] for g in units))


def T1s(runs, lad):
    return [first_T(r, lad[1]) for r in runs]
