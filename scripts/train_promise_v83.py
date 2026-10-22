"""v83 carried objects from the A1 logs (docs/ARC4_V83_PROTOCOL.md §1 'Objet transporté'; synthesis
§2.2), offline, no frames.

  features   python scripts/train_promise_v83.py features        (rows -> memory-mapped feature store)
  fit        python scripts/train_promise_v83.py fit [GAME ...]    (W_dev, W_-g, W_perm, W_perm_-g, W_same)

Declared before any A1 result (build notes §6): rows where I am lost are dropped; at most 15,000 rows
per run (30,000 per game) drawn uniformly with seed 9149; features float16 on disk; standardisation
(mean, scale) pooled over the training games' rows; each training game weighs equally (row weights
1/n_g), intercepts free per game, ridge 1e-4 on the weights; L-BFGS, 300 iterations at most. The
placebo shifts each game's label rows circularly by s rows, s uniform between the rows of 10 rollouts
and n minus that (seed 9149), then each head is rescaled to the norm of the matching real head.
"""
import json
import os
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.logs import labels                         # noqa: E402
from ragnarok.agent_v83.promise import HEADS, HORIZON, feats       # noqa: E402
from ragnarok.agent_v83.views import N_FEAT                        # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEEDS  # noqa: E402

OUT = "craft_v6_out/v83"
STORE = f"{OUT}/promise_store"
PER_RUN = 15000
RIDGE = 1e-4
GAMES = DEV + CALIBRATION
SHAPE = (210, 160)


def build_store():
    os.makedirs(STORE, exist_ok=True)
    rng = np.random.default_rng(SEEDS["placebo_shift"])
    meta = dict(game=[], seed=[], it=[], rise_lag=[], drop_lag=[], avail=[], slot=[])
    chunks = []
    for gi, g in enumerate(GAMES):
        for s in SEEDS["room"]:
            z = np.load(f"{OUT}/room/room_{g}_{s}.npz")
            keep = np.nonzero(~z["row_lost"])[0]
            if len(keep) > PER_RUN:
                keep = np.sort(rng.choice(keep, PER_RUN, replace=False))
            X = np.zeros((len(keep), N_FEAT), np.float16)
            for j, i in enumerate(keep):
                X[j] = feats(z["row_view"][i], z["row_screen"][i], False, (float(z["row_mx"][i]), float(z["row_my"][i])),
                             int(z["row_lives"][i]), int(z["row_slot"][i]), SHAPE)
            chunks.append(X)
            for k, src in (("rise_lag", "row_rise_lag"), ("drop_lag", "row_drop_lag"), ("avail", "row_avail"),
                           ("it", "row_it"), ("slot", "row_slot")):
                meta[k].append(z[src][keep])
            meta["game"].append(np.full(len(keep), gi, np.int16))
            meta["seed"].append(np.full(len(keep), s, np.int32))
            print(g, s, len(keep), flush=True)
    n = sum(len(c) for c in chunks)
    mm = np.lib.format.open_memmap(f"{STORE}/X.npy", mode="w+", dtype=np.float16, shape=(n, N_FEAT))
    o = 0
    for c in chunks:
        mm[o:o + len(c)] = c
        o += len(c)
    mm.flush()
    np.savez(f"{STORE}/meta.npz", games=np.array(GAMES), **{k: np.concatenate(v) for k, v in meta.items()})


def load_store():
    X = np.load(f"{STORE}/X.npy", mmap_mode="r")
    m = dict(np.load(f"{STORE}/meta.npz"))
    return X, m


def head_labels(m, h, shift=None):
    lag = m["rise_lag"] if h == "good" else m["drop_lag"]
    y = labels(lag, m["avail"], HORIZON[h])
    if shift is not None:
        y = y[shift]
    return y


def placebo_index(m):
    """Row index permutation: within each game, labels circularly shifted by >= 10 rollouts of rows."""
    rng = np.random.default_rng(SEEDS["placebo_shift"])
    idx = np.arange(len(m["game"]))
    for gi in np.unique(m["game"]):
        rows = np.nonzero(m["game"] == gi)[0]
        roll = m["seed"][rows].astype(np.int64) * 10**7 + m["it"][rows]
        per = len(rows) / max(1, len(np.unique(roll)))
        lo = int(np.ceil(10 * per))
        if len(rows) <= 2 * lo:
            raise ValueError("game too small for a 10-rollout placebo shift")
        s = int(rng.integers(lo, len(rows) - lo + 1))
        idx[rows] = rows[(np.arange(len(rows)) + s) % len(rows)]
    return idx


def fit(X, m, games, h, shift=None, w0=None, maxiter=300, chunk=200000):
    """Pooled logistic head over `games` (indices): equal weight per game, intercept per game."""
    y_all = head_labels(m, h, shift)
    rows = np.nonzero(np.isin(m["game"], games) & (y_all >= 0))[0]
    gid = m["game"][rows]
    ug = np.unique(gid)
    gpos = np.searchsorted(ug, gid)
    cnt = np.bincount(gpos, minlength=len(ug)).astype(np.float64)
    sw = 1.0 / (len(ug) * cnt[gpos])
    y = y_all[rows].astype(np.float64)
    mean = np.zeros(N_FEAT)
    sq = np.zeros(N_FEAT)
    for a in range(0, len(rows), chunk):
        c = X[rows[a:a + chunk]].astype(np.float64)
        mean += c.sum(0)
        sq += (c ** 2).sum(0)
    mean /= len(rows)
    sd = np.sqrt(np.maximum(sq / len(rows) - mean ** 2, 0))
    scale = np.where(sd > 1e-6, sd, 1.0)
    d, G = N_FEAT, len(ug)

    def f(theta):
        w, b = theta[:d], theta[d:]
        loss, gw, gb = RIDGE * float(w @ w), 2 * RIDGE * w, np.zeros(G)
        for a in range(0, len(rows), chunk):
            Z = (X[rows[a:a + chunk]].astype(np.float64) - mean) / scale
            z = Z @ w + b[gpos[a:a + chunk]]
            yy, ss = y[a:a + chunk], sw[a:a + chunk]
            loss += float(ss @ (np.logaddexp(0.0, z) - yy * z))
            g = ss * (1 / (1 + np.exp(-z)) - yy)
            gw += Z.T @ g
            gb += np.bincount(gpos[a:a + chunk], weights=g, minlength=G)
        return loss, np.concatenate([gw, gb])

    th0 = np.concatenate([np.zeros(d) if w0 is None else w0, np.zeros(G)])
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return dict(w=r.x[:d], mean=mean, scale=scale, n=len(rows), games=ug.tolist(), converged=bool(r.success))


def fit_all(only=None):
    X, m = load_store()
    names = list(m["games"])
    allg = np.arange(len(names))
    shift = placebo_index(m)
    res = {}
    for h in HEADS:
        dev = fit(X, m, allg, h)
        perm = fit(X, m, allg, h, shift=shift)
        perm["w"] = perm["w"] * np.linalg.norm(dev["w"]) / max(1e-12, np.linalg.norm(perm["w"]))
        res[("dev", h)], res[("perm", h)] = dev, perm
        for g in (only or names):
            gi = names.index(g)
            rest = allg[allg != gi]
            mg = fit(X, m, rest, h, w0=dev["w"])
            pg = fit(X, m, rest, h, shift=shift, w0=perm["w"])
            pg["w"] = pg["w"] * np.linalg.norm(mg["w"]) / max(1e-12, np.linalg.norm(pg["w"]))
            res[("minus_" + g, h)], res[("perm_minus_" + g, h)] = mg, pg
            res[("same_" + g, h)] = fit(X, m, np.array([gi]), h)
            print(h, g, mg["converged"], flush=True)
    out = {}
    for (k, h), v in res.items():
        out[f"{k}|{h}|w"], out[f"{k}|{h}|mean"], out[f"{k}|{h}|scale"] = v["w"], v["mean"], v["scale"]
    np.savez(f"{OUT}/promise_W.npz", **out)
    with open(f"{OUT}/promise_W.json", "w", encoding="utf-8") as fh:
        json.dump({f"{k}|{h}": dict(n=v["n"], converged=v["converged"], norm=float(np.linalg.norm(v["w"])))
                   for (k, h), v in res.items()}, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "features":
        build_store()
    elif sys.argv[1] == "fit":
        fit_all(sys.argv[2:] or None)
