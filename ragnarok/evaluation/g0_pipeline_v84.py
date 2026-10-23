"""G0 pipeline (docs/ARC4_V84_PROTOCOL_EN.md A.4-A.11): sources, arms, curves for one held-out game.
Works on any dataset {game: [stream_A, stream_B]} (real store, planted controls, Gaussian test)."""
import numpy as np

from ragnarok.agent_v84.consequence import checkpoints, run_curve
from ragnarok.evaluation.g0_v84 import (HEAD_INDEX, HEADS, eig_desc, geometry, haar, map_head, norm_trace,
                                        permuted_labels, project, shifted_labels, sigma_from_dirs, slot_auc,
                                        source_head, spectrum_matched, stream_head_dir, test_set)

DECISION = ("F", "X", "P", "U", "K*A")
REPORT = ("X-mean", "X_8", "P_shift", "P_rot", "K*Beven")


def fit_sources(dataset, games, gi_of, seeds, heads=HEADS, kinds=("real", "perm", "shift")):
    """Source heads once per (game, head, kind) in the game's own geometry (A.7)."""
    out = {}
    for h in games:
        streams = dataset[h]
        for head in heads:
            for kind in kinds:
                if kind == "real":
                    lab = None
                elif kind == "perm":
                    lab = [permuted_labels(st["y"][head], np.random.default_rng([9703, gi_of[h], s, HEAD_INDEX[head]]))
                           for st, s in zip(streams, seeds)]
                else:
                    lab = [shifted_labels(st["y"][head], st["it"], np.random.default_rng([9707, gi_of[h], s, HEAD_INDEX[head]]))
                           for st, s in zip(streams, seeds)]
                    if any(x is None for x in lab):
                        out[(h, head, kind)] = None
                        continue
                out[(h, head, kind)] = source_head(streams, head, lab)
    return out


def arm_sigmas(g, head, geo, sources, source_games, gi_of):
    """Decision and report Sigmas that do not depend on the direction (X, P, U, X-mean, X_8, P_shift, P_rot)."""
    d = geo["d"]
    real = [(h, sources[(h, head, "real")]) for h in source_games if sources.get((h, head, "real")) is not None]
    perm = [sources[(h, head, "perm")] for h in source_games if sources.get((h, head, "perm")) is not None]
    shif = [sources[(h, head, "shift")] for h in source_games if sources.get((h, head, "shift")) is not None]
    if len(real) < 2 or len(perm) < 2:
        return None
    V = np.array([map_head(hd, geo) for _, hd in real])
    S_X = sigma_from_dirs(V, d)
    lam_X, _ = eig_desc(S_X)
    S_P = spectrum_matched(sigma_from_dirs(np.array([map_head(hd, geo) for hd in perm]), d), lam_X)
    S_U = np.diag(lam_X)                                  # Q axes are ordered by C eigenvalue (descending)
    out = dict(F=(np.eye(d), None), X=(S_X, None), P=(S_P, None), U=(S_U, None),
               **{"X-mean": (np.eye(d), V.mean(0))})
    if len(shif) >= 2:
        out["P_shift"] = (spectrum_matched(sigma_from_dirs(np.array([map_head(hd, geo) for hd in shif]), d), lam_X), None)
    R = haar(d, np.random.default_rng([9709, gi_of[g], HEAD_INDEX[head]]))
    out["P_rot"] = (R @ S_X @ R.T, None)
    rng8 = np.random.default_rng([9704, gi_of[g]])
    draws = []
    for _ in range(3):
        if len(real) >= 8:
            pick = rng8.choice(len(real), 8, replace=False)
            draws.append(sigma_from_dirs(V[pick], d))
    out["X_8"] = draws                                       # list of Sigmas, lambda averaged over draws
    out["_n_sources"] = len(real)
    out["_lam_X"] = lam_X
    return out


def k_star(Z_learn, y_learn, block, d):
    """K*: norm[u u^T + tau^2 I] from a head on one stream (or its even blocks)."""
    w = stream_head_dir(Z_learn, y_learn, block)
    if w is None or not np.any(w):
        return None
    u = w / np.linalg.norm(w)
    return norm_trace(np.outer(u, u) + (0.1 / d) * np.eye(d), d)


def direction_data(g, head, streams, seeds, geo, gi_of, a, b):
    """Learning stream a, test rows of stream b (A.4). Returns dict or None if too few rows."""
    sA, sB = streams[a], streams[b]
    yA = sA["y"][head]
    t_idx, removed = test_set(sA, sB, head, gi_of[g], seeds[b])
    yT = sB["y"][head][t_idx]
    ZA = project(sA["X"], geo)
    ZT = project(sB["X"][t_idx], geo)
    n_pos_l, n_neg_l = int((yA == 1).sum()), int((yA == 0).sum())
    n_pos_t, n_neg_t = int((yT == 1).sum()), int((yT == 0).sum())
    cks = checkpoints(len(yA))
    clocks = [float(sA["clock"][n - 1]) for n in cks]
    return dict(ZA=ZA, yA=yA, ZT=ZT, yT=yT, removed=removed, counts=(n_pos_l, n_neg_l, n_pos_t, n_neg_t),
                cks=cks, clocks=clocks, c_end=float(sA["clock"][-1]), slot_auc=slot_auc(sA, sB, head, t_idx),
                blockA=sA["block"], idxB_even=None)


def curve_for(arm, sig, dd, kappa, maxiter=200):
    S, v0 = sig
    _, a = run_curve(dd["ZA"], dd["yA"], dd["ZT"], dd["yT"], S, kappa, v0=v0, both_classes=(v0 is None), maxiter=maxiter)
    return a
