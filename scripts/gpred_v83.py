"""v83 A2, G-pred (docs/ARC4_V83_PROTOCOL.md §4 A2; synthesis §2.6), offline, no frames.

On each room game g, W_-g is scored on g's A1 rows (feature store of train_promise_v83). A head passes
on g if its AUC >= 0.60 (GOOD) or >= 0.65 (BAD) and it exceeds the better of {per-slot base rate fitted
in-sample on g, placebo W_perm,-g} by >= 0.03 with a cluster-bootstrap lower bound > 0 (clusters:
blocks of 50 iterations per run; 2000 resamples, percentile 2.5%). A head passes overall on >= 50% of
room games and on at least 6. G-pred passes if at least one head passes; otherwise KC-pred.

  python scripts/gpred_v83.py
"""
import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.promise import HEADS                     # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp             # noqa: E402
from scripts.train_promise_v83 import OUT, head_labels, load_store, placebo_index   # noqa: E402

AUC_MIN = dict(good=0.60, bad=0.65)
MARGIN = 0.03
B = 2000


def auc_parts(score, y):
    """Mann-Whitney AUC with ties counted half."""
    ranks = rankdata(score)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def score_rows(X, rows, W, key, h, chunk=200000):
    w, mean, scale = W[f"{key}|{h}|w"], W[f"{key}|{h}|mean"], W[f"{key}|{h}|scale"]
    out = np.zeros(len(rows))
    for a in range(0, len(rows), chunk):
        Z = (X[rows[a:a + chunk]].astype(np.float64) - mean) / scale
        out[a:a + chunk] = Z @ w
    return out


def slot_rate(slots, y):
    r = {s: y[slots == s].mean() for s in np.unique(slots)}
    return np.array([r[s] for s in slots])


def game_test(X, m, W, g, h):
    names = list(m["games"])
    gi = names.index(g)
    y_all = head_labels(m, h)
    rows = np.nonzero((m["game"] == gi) & (y_all >= 0))[0]
    y = y_all[rows].astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return dict(game=g, head=h, n=len(y), passed=False, why="one class")
    s_x = score_rows(X, rows, W, "minus_" + g, h)
    s_p = score_rows(X, rows, W, "perm_minus_" + g, h)
    s_b = slot_rate(m["slot"][rows], y)
    clus = m["seed"][rows].astype(np.int64) * 10**7 + m["it"][rows] // 50
    uc, cidx = np.unique(clus, return_inverse=True)
    groups = [np.nonzero(cidx == c)[0] for c in range(len(uc))]
    a_x, a_p, a_b = auc_parts(s_x, y), auc_parts(s_p, y), auc_parts(s_b, y)
    rng = np.random.default_rng(9149)
    diffs = []
    for _ in range(B):
        pick = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        yy = y[pick]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(auc_parts(s_x[pick], yy) - max(auc_parts(s_p[pick], yy), auc_parts(s_b[pick], yy)))
    lo = float(np.percentile(diffs, 2.5)) if diffs else np.nan
    best_base = max(a_p, a_b)
    passed = bool(a_x >= AUC_MIN[h] and a_x - best_base >= MARGIN and lo > 0)
    return dict(game=g, head=h, n=len(y), pos_rate=float(y.mean()), auc=a_x, auc_placebo=a_p, auc_slot=a_b,
                diff=a_x - best_base, diff_lo=lo, clusters=len(groups), passed=passed)


def main():
    room = json.load(open(f"{OUT}/room_verdict.json", encoding="utf-8"))
    if room["G_room"] != "PASS" or room["interruptions"]:
        sys.exit(f"G-room is {room['G_room']}: A2 does not run")
    games = room["room_dev"] + room["room_cal"]
    X, m = load_store()
    W = np.load(f"{OUT}/promise_W.npz")
    res = {h: [game_test(X, m, W, g, h) for g in games] for h in HEADS}
    heads = {}
    for h in HEADS:
        k = sum(r["passed"] for r in res[h])
        heads[h] = dict(passed_games=k, of=len(games), passed=bool(len(games) and k >= 0.5 * len(games) and k >= 6))
    out = dict(rule="docs/ARC4_V83_PROTOCOL.md §4 A2", games=games, heads=heads,
               G_pred="PASS" if any(v["passed"] for v in heads.values()) else "KC-pred",
               carried_heads=[h for h in HEADS if heads[h]["passed"]], per_game=res, provenance=stamp())
    with open(f"{OUT}/gpred_verdict.json.tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{OUT}/gpred_verdict.json.tmp", f"{OUT}/gpred_verdict.json")
    print(json.dumps({k: out[k] for k in ("G_pred", "heads", "carried_heads")}))


if __name__ == "__main__":
    main()
