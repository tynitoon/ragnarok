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
from ragnarok.agent_v83.promise import HEADS, HORIZON, SD_FLOOR, feats   # noqa: E402
from ragnarok.agent_v83.views import N_FEAT                        # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEEDS  # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp    # noqa: E402

OUT = str(ROOT / "craft_v6_out" / "v83")
STORE = f"{OUT}/promise_store"
PER_RUN = 15000
RIDGE = 1e-4
GAMES = DEV + CALIBRATION
SHAPE = (210, 160)


def build_store():
    os.makedirs(STORE, exist_ok=True)
    rng = np.random.default_rng(SEEDS["placebo_shift"])
    meta = dict(game=[], seed=[], it=[], rise_lag=[], drop_lag=[], avail=[], slot=[], t=[])
    chunks, paths = [], []
    for gi, g in enumerate(GAMES):
        for s in SEEDS["room"]:
            paths.append(f"{OUT}/room/room_{g}_{s}.npz")
            z = np.load(paths[-1])
            V, S, MX, MY, LV, SL = (z[k] for k in ("row_view", "row_screen", "row_mx", "row_my", "row_lives", "row_slot"))
            keep = np.nonzero(~z["row_lost"])[0]                 # rows where I am lost are dropped
            if len(keep) > PER_RUN:
                keep = np.sort(rng.choice(keep, PER_RUN, replace=False))
            X = np.zeros((len(keep), N_FEAT), np.float16)
            for j, i in enumerate(keep):                         # each array inflated once per run, not per row
                X[j] = feats(V[i], S[i], False, (float(MX[i]), float(MY[i])), int(LV[i]), int(SL[i]), SHAPE)
            chunks.append(X)
            for k, src in (("rise_lag", "row_rise_lag"), ("drop_lag", "row_drop_lag"), ("avail", "row_avail"),
                           ("it", "row_it"), ("slot", "row_slot"), ("t", "row_t")):
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
    with open(f"{STORE}/provenance.json", "w", encoding="utf-8") as fh:
        json.dump(dict(stamp(), per_run=PER_RUN, seed=SEEDS["placebo_shift"],
                       inputs_sha256={p: full_hash(p) for p in paths}), fh, indent=1)


def load_store():
    X = np.load(f"{STORE}/X.npy", mmap_mode="r")
    m = dict(np.load(f"{STORE}/meta.npz"))
    return X, m


def head_labels(m, h):
    lag = m["rise_lag"] if h == "good" else m["drop_lag"]
    return labels(lag, m["avail"], m["t"], HORIZON[h], stops_on_event=(h == "bad"))


def placebo_labels(m):
    """Placebo labels per head (declared before any A1 result). Within each game, the head's uncensored
    labels are shifted circularly by s uncensored rows, s uniform between the rows of 10 rollouts and n
    minus that (seed 9149, drawn game by game, GOOD then BAD); censored rows stay censored, so W_perm
    trains on exactly W's rows, same base rate, same clustering of the kept labels. A game whose kept
    rows for a head come from <= 20 rollouts has no shift: it is left out of that head's pooled fits
    (real and placebo) and listed. Returns ({head: labels}, {head: [game index]})."""
    rng = np.random.default_rng(SEEDS["placebo_shift"])
    y = {h: head_labels(m, h) for h in HEADS}
    yp = {h: np.full_like(y[h], -1) for h in HEADS}
    out = {h: [] for h in HEADS}
    for gi in np.unique(m["game"]):
        for h in HEADS:
            kept = np.nonzero((m["game"] == gi) & (y[h] >= 0))[0]
            roll = m["seed"][kept].astype(np.int64) * 10**7 + m["it"][kept]
            lo = int(np.ceil(10 * len(kept) / max(1, len(np.unique(roll)))))
            if len(kept) <= 2 * lo:
                out[h].append(int(gi))
                continue
            s = int(rng.integers(lo, len(kept) - lo + 1))
            yp[h][kept] = y[h][kept][(np.arange(len(kept)) + s) % len(kept)]
    return yp, out


def standardise(X, m, games, chunk=200000):
    """Mean and scale over every (non-lost) row of the training games, shared by both heads."""
    rows = np.nonzero(np.isin(m["game"], games))[0]
    mean, sq = np.zeros(N_FEAT), np.zeros(N_FEAT)
    for a in range(0, len(rows), chunk):
        c = X[rows[a:a + chunk]].astype(np.float64)
        mean += c.sum(0)
        sq += (c ** 2).sum(0)
    if len(rows) == 0:                                   # a game without non-lost rows: neutral standardisation
        return mean, np.ones(N_FEAT)
    mean /= len(rows)
    sd = np.sqrt(np.maximum(sq / len(rows) - mean ** 2, 0))
    return mean, np.maximum(sd, SD_FLOOR)


def fit(X, m, games, h, y=None, w0=None, maxiter=300, chunk=200000, stand=None):
    """Pooled logistic head over `games` (indices): equal weight per game, intercept per game; `y`
    replaces the head's labels (placebo)."""
    y_all = head_labels(m, h) if y is None else y
    rows = np.nonzero(np.isin(m["game"], games) & (y_all >= 0))[0]
    gid = m["game"][rows]
    ug = np.unique(gid)
    gpos = np.searchsorted(ug, gid)
    cnt = np.bincount(gpos, minlength=len(ug)).astype(np.float64)
    sw = 1.0 / (len(ug) * cnt[gpos])
    y = y_all[rows].astype(np.float64)
    mean, scale = stand if stand is not None else standardise(X, m, games, chunk)
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
    yp, left_out = placebo_labels(m)
    res = {}
    st_all = standardise(X, m, allg)
    for h in HEADS:
        use = allg[~np.isin(allg, left_out[h])]           # standardisation stays common to both heads (note 18)
        dev = fit(X, m, use, h, stand=st_all)
        perm = fit(X, m, use, h, y=yp[h], stand=st_all)
        perm["w"] = perm["w"] * np.linalg.norm(dev["w"]) / max(1e-12, np.linalg.norm(perm["w"]))
        res[("dev", h)], res[("perm", h)] = dev, perm
        for g in (only or names):
            gi = names.index(g)
            rest = allg[allg != gi]
            st_rest, st_g = standardise(X, m, rest), standardise(X, m, np.array([gi]))
            mg = fit(X, m, use[use != gi], h, stand=st_rest)          # from zero: no warm start that saw g
            pg = fit(X, m, use[use != gi], h, y=yp[h], stand=st_rest)
            pg["w"] = pg["w"] * np.linalg.norm(mg["w"]) / max(1e-12, np.linalg.norm(pg["w"]))
            res[("minus_" + g, h)], res[("perm_minus_" + g, h)] = mg, pg
            res[("same_" + g, h)] = fit(X, m, np.array([gi]), h, stand=st_g)
            print(h, g, mg["converged"], flush=True)
    out = {}
    for (k, h), v in res.items():
        out[f"{k}|{h}|w"], out[f"{k}|{h}|mean"], out[f"{k}|{h}|scale"] = v["w"], v["mean"], v["scale"]
    np.savez(f"{OUT}/promise_W.npz", **out)
    info = {f"{k}|{h}": dict(n=v["n"], converged=v["converged"], norm=float(np.linalg.norm(v["w"])))
            for (k, h), v in res.items()}
    info["_left_out"] = {h: [names[i] for i in left_out[h]] for h in HEADS}
    info["_provenance"] = dict(stamp(), store=json.load(open(f"{STORE}/provenance.json", encoding="utf-8")),
                               store_sha256={p: full_hash(p) for p in (f"{STORE}/X.npy", f"{STORE}/meta.npz")},
                               W_sha256=full_hash(f"{OUT}/promise_W.npz"))
    with open(f"{OUT}/promise_W.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "features":
        build_store()
    elif sys.argv[1] == "fit":
        fit_all(sys.argv[2:] or None)
