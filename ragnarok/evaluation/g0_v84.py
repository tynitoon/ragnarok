"""G0, the knowledge clock (docs/ARC4_V84_PROTOCOL_EN.md, Part A), evaluator side.

Data model: a `stream` is a dict with X (n x 427 raw features, float16 or float64), y = {"good", "bad"}
labels in {-1, 0, 1}, clock (billed frames since entry), block (row_it // 50), hash (dedup), slot. A
dataset maps game -> [stream_A(9131), stream_B(9132)]. The same functions serve the real store, the
planted controls (T8) and the Gaussian unit test (T7).
"""
import math

import numpy as np
from scipy.optimize import minimize

from ragnarok.agent_v84.consequence import auc, checkpoints, run_curve

HEADS = ("good", "bad")
HEAD_INDEX = {"good": 0, "bad": 1}
SD_FLOOR = 0.05
EIG_REL = 1e-5
QS = (0.25, 0.375, 0.5, 0.625, 0.75)
LATE_QS = (0.625, 0.75)
FLOOR = 5000
KAPPA_GRID = tuple(4.0 ** k for k in range(-5, 1))
RHO_GRID = (1e-3, 1e-2, 1e-1, 1.0)
CARRIED = ("X", "K*A", "K*Beven", "X-mean", "X_8")
CONTROLS = ("F", "P", "U", "P_shift", "P_rot")


# ---------------------------------------------------------------- geometry (A.5)
def stream_sums(X):
    X = np.asarray(X, np.float64)
    return dict(n=len(X), s=X.sum(0), ss=X.T @ X)


def game_sums(streams):
    out = dict(n=0, s=0.0, ss=0.0)
    for st in streams:
        q = stream_sums(st["X"])
        out["n"] += q["n"]
        out["s"] = out["s"] + q["s"]
        out["ss"] = out["ss"] + q["ss"]
    return out


def geometry(sums, sources):
    """S_-g (pooled mean, floored SD over all source rows), C_-g (equal-game-weighted covariance of
    x_tilde) and Q_-g (eigenvectors with eigenvalue > 1e-5 lambda_max), eigenvalues descending."""
    n = sum(sums[h]["n"] for h in sources)
    s = sum(sums[h]["s"] for h in sources)
    ss = sum(sums[h]["ss"] for h in sources)
    m = s / n
    sd_raw = np.sqrt(np.maximum(np.diag(ss) / n - m ** 2, 0.0))
    sd = np.maximum(sd_raw, SD_FLOOR)
    mus = [sums[h]["s"] / sums[h]["n"] for h in sources]
    Ms = [sums[h]["ss"] / sums[h]["n"] for h in sources]
    a = np.mean(mus, axis=0)
    C_raw = np.mean(Ms, axis=0) - np.outer(a, a)
    C = C_raw / np.outer(sd, sd)
    lam, U = np.linalg.eigh((C + C.T) / 2)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    keep = lam > EIG_REL * lam[0]
    return dict(m=m, sd=sd, sd_raw=sd_raw, Q=U[:, keep], lam=lam[keep], d=int(keep.sum()))


def project(X, geo):
    return ((np.asarray(X, np.float64) - geo["m"]) / geo["sd"]) @ geo["Q"]


def outside_share(X, geo):
    """Share of x_tilde variance (of these rows) outside span(Q)."""
    Xt = (np.asarray(X, np.float64) - geo["m"]) / geo["sd"]
    Xt = Xt - Xt.mean(0)
    tot = float((Xt ** 2).sum())
    inside = float(((Xt @ geo["Q"]) ** 2).sum())
    return 1.0 - inside / tot if tot > 0 else 0.0


# ---------------------------------------------------------------- source heads (A.7)
def _fit_multi(Z, y, grp, n_grp, rho, w0=None, b0=None, maxiter=300, fix_w=None):
    """Logistic, one intercept per group, ridge (rho/2)||w||^2 on the MEAN log-loss. fix_w: fit only the
    intercepts with w fixed."""
    n, d = Z.shape
    yf = y.astype(np.float64)
    if fix_w is not None:
        zw = Z @ fix_w

        def f(b):
            z = zw + b[grp]
            g = (1.0 / (1.0 + np.exp(-z)) - yf) / n
            return float(np.mean(np.logaddexp(0.0, z) - yf * z)), np.bincount(grp, weights=g, minlength=n_grp)
        r = minimize(f, np.zeros(n_grp) if b0 is None else b0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        return fix_w, r.x

    def f(th):
        w, b = th[:d], th[d:]
        z = Z @ w + b[grp]
        g = (1.0 / (1.0 + np.exp(-z)) - yf) / n
        loss = float(np.mean(np.logaddexp(0.0, z) - yf * z)) + 0.5 * rho * float(w @ w)
        return loss, np.concatenate([Z.T @ g + rho * w, np.bincount(grp, weights=g, minlength=n_grp)])
    th0 = np.concatenate([np.zeros(d) if w0 is None else w0, np.zeros(n_grp) if b0 is None else b0])
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return r.x[:d], r.x[d:]


def _logloss(Z, y, w, b):
    z = Z @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_two_fold(Z1, y1, Z2, y2, maxiter=300):
    """rho by 2-fold (fit on one fold, refit the other fold's intercept only, log-loss there; mean of both
    directions; ties to the larger rho; rho = 0.1 if a fold has < 10 rows of a class). Final fit on both
    folds with one intercept each. Returns (w, intercepts, rho, cv losses)."""
    small = any(min(int((y == 1).sum()), int((y == 0).sum())) < 10 for y in (y1, y2))
    cv = {}
    if small:
        rho = 0.1
    else:
        for rho_c in RHO_GRID:
            ls = []
            for (Za, ya), (Zb, yb) in (((Z1, y1), (Z2, y2)), ((Z2, y2), (Z1, y1))):
                w, _ = _fit_multi(Za, ya, np.zeros(len(ya), np.int64), 1, rho_c, maxiter=maxiter)
                _, b = _fit_multi(Zb, yb, np.zeros(len(yb), np.int64), 1, rho_c, fix_w=w, maxiter=maxiter)
                ls.append(_logloss(Zb, yb, w, b[0]))
            cv[rho_c] = float(np.mean(ls))
        best = min(cv.values())
        rho = max(r for r, v in cv.items() if v <= best + 1e-12)
    Z = np.vstack([Z1, Z2])
    y = np.concatenate([y1, y2])
    grp = np.concatenate([np.zeros(len(y1), np.int64), np.ones(len(y2), np.int64)])
    w, b = _fit_multi(Z, y, grp, 2, rho, maxiter=maxiter)
    return w, b, rho, cv


def source_head(streams, head, labels_override=None):
    """Head of one source game in its own geometry (A.7). Returns dict(w_raw coefficients on x_h, m_h,
    sd_h, mask, rho, n_pos, n_neg) or None if fewer than 30 positives or negatives over both streams."""
    ys = [labels_override[i] if labels_override is not None else st["y"][head] for i, st in enumerate(streams)]
    oks = [y >= 0 for y in ys]
    pos = sum(int((y[o] == 1).sum()) for y, o in zip(ys, oks))
    neg = sum(int((y[o] == 0).sum()) for y, o in zip(ys, oks))
    if pos < 30 or neg < 30:
        return None
    Xall = np.vstack([np.asarray(st["X"], np.float64) for st in streams])
    m_h = Xall.mean(0)
    sd_h = Xall.std(0)
    mask = sd_h >= SD_FLOOR
    Zs = [((np.asarray(st["X"], np.float64)[o] - m_h) / np.where(mask, sd_h, 1.0))[:, mask] for st, o in zip(streams, oks)]
    yv = [y[o] for y, o in zip(ys, oks)]
    w, b, rho, cv = fit_two_fold(Zs[0], yv[0], Zs[1], yv[1])
    wf = np.zeros(len(mask))
    wf[mask] = w
    return dict(w=wf, m=m_h, sd=sd_h, mask=mask, rho=rho, cv=cv, n_pos=pos, n_neg=neg)


def map_head(hd, geo):
    """w' = w * (sd_-g / sd_h) on the raw-feature function, then v = Q^T w'."""
    wp = np.where(hd["mask"], hd["w"] * geo["sd"] / np.where(hd["mask"], hd["sd"], 1.0), 0.0)
    return geo["Q"].T @ wp


def permuted_labels(y, rng):
    y = y.copy()
    ok = np.nonzero(y >= 0)[0]
    y[ok] = y[ok][rng.permutation(len(ok))]
    return y


def shifted_labels(y, it, rng):
    """Circular shift of the uncensored labels by >= 10 rollouts of rows (v83 placebo, per stream)."""
    y = y.copy()
    ok = np.nonzero(y >= 0)[0]
    if len(ok) == 0:
        return None
    per = len(ok) / max(1, len(np.unique(it[ok])))
    lo = int(math.ceil(10 * per))
    if len(ok) <= 2 * lo:
        return None
    s = int(rng.integers(lo, len(ok) - lo + 1))
    y[ok] = y[ok][(np.arange(len(ok)) + s) % len(ok)]
    return y


# ---------------------------------------------------------------- arms (A.8)
def norm_trace(M, d):
    return M * (d / np.trace(M))


def sigma_from_dirs(V, d):
    """norm[(1/G) sum u u^T + tau^2 I] from unit vectors (rows of V), tau^2 = 0.1 / d."""
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    M = (U.T @ U) / len(U) + (0.1 / d) * np.eye(d)
    return norm_trace(M, d)


def eig_desc(S):
    lam, U = np.linalg.eigh((S + S.T) / 2)
    o = np.argsort(lam)[::-1]
    return lam[o], U[:, o]


def spectrum_matched(S_shape, lam_target):
    """Eigenvectors of S_shape (by its own eigenvalue order) with lam_target (descending)."""
    _, U = eig_desc(S_shape)
    return (U * lam_target) @ U.T


def sigma_U(geo, lam_target):
    """Q axes ordered by C eigenvalue (geo['lam'] is descending and Q columns follow it): diagonal."""
    return np.diag(lam_target)


def haar(d, rng):
    A = rng.standard_normal((d, d))
    Qm, R = np.linalg.qr(A)
    return Qm * np.sign(np.diag(R))


def stream_head_dir(Z, y, block):
    """K*: a head fitted in x_hat on one stream, rho by 2-fold over its even and odd blocks."""
    ok = y >= 0
    Z, y, block = Z[ok], y[ok], block[ok]
    ev = block % 2 == 0
    if min(int((y == 1).sum()), int((y == 0).sum())) < 2 or ev.all() or (~ev).all():
        return None
    w, _, rho, _ = fit_two_fold(Z[ev], y[ev], Z[~ev], y[~ev])
    return w


# ---------------------------------------------------------------- splits (A.4)
def test_set(stream_A, stream_B, head, gi, seed_B):
    yB = stream_B["y"][head]
    odd = (stream_B["block"] % 2 == 1) & (yB >= 0)
    inA = np.isin(stream_B["hash"], stream_A["hash"])
    cand = np.nonzero(odd)[0]
    keep = cand[~inA[cand]]
    removed = 1.0 - len(keep) / max(1, len(cand))
    if len(keep) > 5000:
        keep = np.sort(np.random.default_rng([9702, gi, seed_B, HEAD_INDEX[head]]).choice(keep, 5000, replace=False))
    return keep, removed


def slot_auc(stream_A, stream_B, head, test_idx):
    yA = stream_A["y"][head]
    ok = yA >= 0
    rate_all = float(yA[ok].mean()) if ok.any() else 0.0
    rates = {}
    for s in np.unique(stream_A["slot"][ok]):
        sel = ok & (stream_A["slot"] == s)
        rates[int(s)] = float(yA[sel].mean())
    sc = np.array([rates.get(int(s), rate_all) for s in stream_B["slot"][test_idx]])
    return auc(sc, stream_B["y"][head][test_idx])


# ---------------------------------------------------------------- metric (A.11)
def crossing(aucs, clocks, L, carried, c_end):
    """N(L): first k with AUC(k) >= L and AUC(k+1) >= L; log-clock interpolation from k-1; floor 5000."""
    K = len(aucs)
    for k in range(K - 1):
        if aucs[k] >= L and aucs[k + 1] >= L:
            if k == 0:
                N = clocks[0]
            else:
                a0, a1 = aucs[k - 1], aucs[k]
                c0, c1 = math.log(max(clocks[k - 1], 1)), math.log(max(clocks[k], 1))
                f = 0.0 if a1 == a0 else (L - a0) / (a1 - a0)
                N = math.exp(c0 + min(max(f, 0.0), 1.0) * (c1 - c0))
            return max(FLOOR, N), True
    return (2.0 * c_end if carried else float(c_end)), False


def lambdas(aucs, clocks, auc_f_end, carried, c_end):
    """(lambda over QS, lambda_late over LATE_QS, N per q, crossed per q)."""
    N, ok = {}, {}
    for q in QS:
        N[q], ok[q] = crossing(aucs, clocks, 0.5 + q * (auc_f_end - 0.5), carried, c_end)
    lam = float(np.mean([math.log(N[q]) for q in QS]))
    D = []
    for q in LATE_QS:
        if carried:
            n_q = N[q] if ok[q] else 2.0 * c_end
            D.append(2.0 * c_end if not ok[0.5] else n_q - N[0.5])
        else:
            if not ok[0.5]:
                D.append(FLOOR)
            elif not ok[q]:
                D.append(c_end - N[0.5])
            else:
                D.append(N[q] - N[0.5])
    lam_late = float(np.mean([math.log(max(FLOOR, x)) for x in D]))
    return lam, lam_late, N, ok


# ---------------------------------------------------------------- statistics (A.11-A.12)
def ratio(lg):
    """R = exp(mean over games of l_g)."""
    return float(math.exp(np.mean(list(lg.values()))))


def boot(lg, n=2000, seed=9705):
    v = np.array(list(lg.values()))
    rng = np.random.default_rng(seed)
    bs = [math.exp(v[rng.integers(len(v), size=len(v))].mean()) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def signflip_p(lg, seed=9705, n_mc=100000):
    v = np.array(list(lg.values()))
    obs = v.mean()
    G = len(v)
    if G <= 20:
        import itertools
        vals = [np.mean(v * np.array(s)) for s in itertools.product((1, -1), repeat=G)]
        return float(np.mean(np.array(vals) >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    sg = rng.choice((1, -1), size=(n_mc, G))
    return float(np.mean((sg * v).mean(1) >= obs - 1e-12))


def hodges_lehmann(lg):
    v = np.array(list(lg.values()))
    w = [(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i, len(v))]
    return float(math.exp(np.median(w)))


def per_game(lu, units):
    """lu: {unit: l}; units: {unit: game} -> {game: mean l}."""
    g = {}
    for u, l in lu.items():
        g.setdefault(units[u], []).append(l)
    return {k: float(np.mean(v)) for k, v in g.items()}


def logo_min(lg):
    if len(lg) < 2:
        return None
    return float(min(ratio({k: v for k, v in lg.items() if k != g}) for g in lg))
