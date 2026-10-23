"""Consequence learner with a Gaussian prior (docs/ARC4_V84_PROTOCOL_EN.md, A.6).

MAP logistic regression on reduced features x_hat (d dims): minimise
    sum_i [softplus(z_i) - y_i z_i] + (1/(2 kappa)) (v - v0)^T Sigma^-1 (v - v0),   z_i = b + x_hat_i^T v,
with a free intercept. The loss is a SUM, so the prior's weight fades as 1/n. Implemented in the whitened
coordinates x' = Sigma^(1/2) x_hat, v = Sigma^(1/2) v' + v0, with an isotropic penalty on v'.

`curve` fits at the checkpoints n_k = ceil(128 * 2^(k/4)) (while n_k < n) and n, warm-starting each fit
from the previous one, and scores the test AUC at every checkpoint. Both-classes rule: while the
uncensored rows seen so far lack either class, the head is intercept-only (test AUC exactly 0.5),
except for a prior with a non-zero mean (X-mean, report only).
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


def checkpoints(n):
    ks, k = [], 0
    while True:
        c = int(math.ceil(128 * 2 ** (k / 4)))
        if c >= n:
            break
        if not ks or c > ks[-1]:
            ks.append(c)
        k += 1
    ks.append(int(n))
    return ks


def auc(score, y):
    """Mann-Whitney AUC, ties counted half."""
    y = np.asarray(y)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def sqrt_psd(S):
    lam, U = np.linalg.eigh((S + S.T) / 2)
    lam = np.clip(lam, 0.0, None)
    return (U * np.sqrt(lam)) @ U.T


def fit(Xw, y, kappa, theta0=None, offset=None, maxiter=200):
    """MAP fit in whitened coordinates. Xw: (n, d) = x_hat Sigma^(1/2); offset: x_hat v0 (prior mean).
    Returns theta = (v', b)."""
    n, d = Xw.shape
    y = y.astype(np.float64)
    off = np.zeros(n) if offset is None else offset
    inv2k = 1.0 / (2.0 * kappa)

    def f(th):
        v, b = th[:d], th[d]
        z = Xw @ v + b + off
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z)) + inv2k * float(v @ v)
        g = 1.0 / (1.0 + np.exp(-z)) - y
        return loss, np.concatenate([Xw.T @ g + 2.0 * inv2k * v, [g.sum()]])

    th0 = np.zeros(d + 1) if theta0 is None else theta0
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return r.x


def run_curve(X_learn, y_learn, X_test, y_test, Sigma, kappa, v0=None, both_classes=True, maxiter=200):
    """X_learn: (nA, d) every kept row of the learning stream in order; y_learn: labels in {-1, 0, 1}
    (-1 = censored, ignored); X_test, y_test: uncensored test rows. Returns (checkpoints, AUC list).
    v0: prior mean (X-mean arm) or None (zero mean)."""
    nA = len(X_learn)
    cks = checkpoints(nA)
    S12 = sqrt_psd(Sigma)
    unc = np.nonzero(y_learn >= 0)[0]
    Xw_all = X_learn[unc] @ S12
    yu = y_learn[unc].astype(np.int64)
    Xw_test = X_test @ S12
    off_all = None if v0 is None else X_learn[unc] @ v0
    off_test = None if v0 is None else X_test @ v0
    theta, out = None, []
    for n in cks:
        m = int(np.searchsorted(unc, n))            # uncensored rows among the first n kept rows
        ys = yu[:m]
        two = m > 0 and 0 < ys.sum() < m
        if both_classes and not two:
            out.append(0.5)                             # intercept-only: constant test scores
            continue
        if m == 0:                                      # X-mean before any data: scores from the prior mean
            out.append(auc(off_test, y_test))
            continue
        theta = fit(Xw_all[:m], ys, kappa, theta, None if off_all is None else off_all[:m], maxiter)
        z = Xw_test @ theta[:-1] + theta[-1] + (0.0 if off_test is None else off_test)
        out.append(auc(z, y_test))
    return cks, out
