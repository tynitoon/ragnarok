"""ARC 3 — test (14): the ROSTER CEILING (ARC3_PLAN.md 2.4, 4.3). CPU only, numpy only.

The roster (which (element, tier) items exist in a world) is a consequence of the law and is disclosed to
every arm. How much of the Bond table can be read off the roster ALONE, without ever proposing a pair?
A logistic classifier is trained across 2,000 random law tables (never 31337) to predict Bond[a, b] from
roster features of (a, b), and evaluated on held-out tables — at coverage c it also sees a random
fraction c of the table as known cells (partner counts) and is scored on the UNEXERCISED cells only.
That held-out AUC is the number printed beside M's outcome-head AUC in the probe and the caption.

Usage: python -m scripts.roster_v61 [--n 2000]
"""

import argparse
import json

import numpy as np

from ragnarok.environments.law_world import sample_laws, gen_law_world, T_MAX

COVERAGES = (0.0, 0.38, 0.62, 0.76, 0.85)


def roster_features(spec, laws, rng, coverage):
    """Rows for every unordered element pair (a, b): symmetric roster features (+ known-partner counts
    at `coverage`), label Bond[a,b], and a mask of UNEXERCISED cells."""
    K = laws["K"]
    n, el, tier = spec["n_items"], spec["el"], spec["tier"]
    present = np.zeros((K, T_MAX + 1))
    for i in range(n):
        present[el[i], tier[i]] = 1.0                       # real AND decoy: what the agent sees
    cells = [(a, b) for a in range(K) for b in range(a + 1, K)]
    known = rng.random(len(cells)) < coverage
    bond = np.array([laws["Bond"][a, b] for (a, b) in cells], dtype=float)
    kb = np.zeros(K); kn = np.zeros(K)                       # known bond / non-bond partner counts
    for (a, b), k, y in zip(cells, known, bond):
        if k:
            if y:
                kb[a] += 1; kb[b] += 1
            else:
                kn[a] += 1; kn[b] += 1
    X = []
    for (a, b) in cells:
        pa, pb = present[a, 1:], present[b, 1:]              # tiers 1..4
        f = np.concatenate([pa + pb, pa * pb, [pa.sum() + pb.sum(), pa.sum() * pb.sum()],
                            [kb[a] + kb[b], kb[a] * kb[b], kn[a] + kn[b], kn[a] * kn[b], coverage]])
        X.append(f)
    return np.array(X), bond, ~known


def fit_logreg(X, y, iters=400, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xn @ w))
        g = Xn.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return lambda Z: 1 / (1 + np.exp(-(np.c_[(Z - mu) / sd, np.ones(len(Z))] @ w)))


def auc(p, y):
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--json", default="craft_v6_out/v61_roster.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for c in COVERAGES:
        Xtr, ytr, Xte, yte, mte = [], [], [], [], []
        for s in range(a.n):
            laws = sample_laws(50_000 + s)
            spec = gen_law_world(90_000 + s, laws)
            X, y, unex = roster_features(spec, laws, rng, c)
            if s < int(0.8 * a.n):
                Xtr.append(X[unex]); ytr.append(y[unex])
            else:
                Xte.append(X); yte.append(y); mte.append(unex)
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        f = fit_logreg(Xtr, ytr)
        Xte, yte, mte = np.concatenate(Xte), np.concatenate(yte), np.concatenate(mte)
        p = f(Xte)
        out[str(c)] = dict(auc_unexercised=round(auc(p[mte], yte[mte]), 4), n_test_cells=int(mte.sum()),
                           base_rate=round(float(yte[mte].mean()), 3))
        print(f"coverage {c:.2f}: roster-only AUC on unexercised cells = {out[str(c)]['auc_unexercised']:.3f} "
              f"({int(mte.sum())} held-out cells, base rate {yte[mte].mean():.2f})", flush=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"written {a.json}")


if __name__ == "__main__":
    main()
