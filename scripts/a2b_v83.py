"""v83 A2b (report only; synthesis §2.6): held-out log-loss of the carried prior plus in-game refit
(L2-SP, n0 = 50,000, kappa0 = 1, w0 = W_-g) against in-game only (w0 = 0), at 2k, 10k and 50k in-game
rows. Written prediction: the prior's advantage falls below 25% of its value at 2k rows by 50k rows.

Declared before any A1 result: in-game training rows = the first n non-lost rows (in time order) of the
room seed 9131 run; test rows = every labelled non-lost row of the 9132 run; the in-game fit uses the
prior's standardisation for both arms; intercept free.

  python scripts/a2b_v83.py
"""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.logs import labels                         # noqa: E402
from ragnarok.agent_v83.promise import HEADS, HORIZON, feats, fit_head   # noqa: E402
from ragnarok.evaluation.pools_v83 import SEEDS                    # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp               # noqa: E402

OUT = ROOT / "craft_v6_out" / "v83"
SIZES = (2000, 10000, 50000)


def rows_of(g, s):
    z = np.load(OUT / "room" / f"room_{g}_{s}.npz")
    idx = np.nonzero(~z["row_lost"])[0]                  # every non-lost row, in time order
    V, S, MX, MY, LV, SL = (z[k] for k in ("row_view", "row_screen", "row_mx", "row_my", "row_lives", "row_slot"))
    X = np.stack([feats(V[i], S[i], False, (float(MX[i]), float(MY[i])), int(LV[i]), int(SL[i]), (210, 160))
                  for i in idx]) if len(idx) else None
    lab = {h: labels(z["row_rise_lag" if h == "good" else "row_drop_lag"][idx], z["row_avail"][idx], z["row_t"][idx],
                     HORIZON[h], stops_on_event=(h == "bad")) for h in HEADS}
    return X, lab


def logloss(Z, y, w, b):
    z = Z @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def main():
    gp = json.load(open(OUT / "gpred_verdict.json", encoding="utf-8"))
    W = np.load(OUT / "promise_W.npz")
    res = {}
    for g in gp["games"]:
        a, s = SEEDS["room"]
        Xa, la = rows_of(g, a)
        Xb, lb = rows_of(g, s)
        if Xa is None or Xb is None:
            continue
        for h in HEADS:
            key = f"minus_{g}|{h}"
            mean, scale, w0 = W[key + "|mean"], W[key + "|scale"], W[key + "|w"]
            Za, Zb = (Xa - mean) / scale, (Xb - mean) / scale
            okb = lb[h] >= 0
            out = []
            for n in SIZES:
                if len(Xa) < n:                          # the run has fewer than n non-lost rows
                    out.append(dict(n=n, available=int(len(Xa))))
                    continue
                oka = np.nonzero(la[h][:n] >= 0)[0]      # labelled rows among the first n non-lost rows
                ya = la[h][oka]
                if len(oka) == 0 or ya.min() == ya.max():
                    out.append(dict(n=n, labelled=int(len(oka)), one_class=True))
                    continue
                wp, bp = fit_head(Za[oka], ya, w0)
                wf, bf = fit_head(Za[oka], ya, np.zeros_like(w0))
                lp, lf = logloss(Zb[okb], lb[h][okb], wp, bp), logloss(Zb[okb], lb[h][okb], wf, bf)
                out.append(dict(n=n, labelled=int(len(oka)), prior=lp, fresh=lf, advantage=lf - lp))
            res[f"{g}|{h}"] = out
    pred = {}
    for k, v in res.items():
        adv = [x.get("advantage") for x in v]
        if adv[0] is not None and adv[-1] is not None and adv[0] > 0:
            pred[k] = bool(adv[-1] < 0.25 * adv[0])
    out = dict(rule="synthesis §2.6 A2b (report only)", per_game=res, prediction_holds=pred,
               share_holds=float(np.mean(list(pred.values()))) if pred else None, provenance=stamp())
    with open(OUT / "a2b_report.json.tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT / "a2b_report.json.tmp", OUT / "a2b_report.json")
    print(json.dumps(dict(share_holds=out["share_holds"], n=len(pred))))


if __name__ == "__main__":
    main()
