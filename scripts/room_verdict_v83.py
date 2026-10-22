"""v83 A1 verdict, G-room (docs/ARC4_V83_PROTOCOL.md §4 A1; synthesis §2.5). Frozen rules:

A game is a room game if (a) each run has >= 10 sensed rises in its first 300k billed frames, (b) M*_g,
the highest rung reached by both seeds within 2M, is >= M3, (c) median T(M1) <= 1M, (d) median
C = T(M*) - T(M1) >= 300k. Pooled spread s2 = mean over room games of (log C1 - log C2)^2 / 2.
G-room passes if >= 13 of the 35 development games are room games and
se_proj = sqrt((s2/2 + 0.09)/G_proj) <= 0.23 with G_proj = 17 n_room / 35. Otherwise KC-room.
T(M) of the records used (M1, M2, M*) is confirmed by replay; a mismatch is an Interruption.

  python scripts/room_verdict_v83.py [--no-replay]
"""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.milestones_v83 import confirm, first_index, first_T, ladder, top_rung   # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEEDS                                # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                                             # noqa: E402

OUT = "craft_v6_out/v83"
CAP = 2_000_000


def game_row(g, replay=True):
    bp = f"{OUT}/bars/{g}_{SEEDS['dev_bars']}.json"
    runs = [f"{OUT}/room/room_{g}_{s}" for s in SEEDS["room"]]
    if not os.path.exists(bp) or not all(os.path.exists(r + ".json") for r in runs):
        return dict(game=g, complete=False)
    bar = json.load(open(bp, encoding="utf-8"))["bars"][g]
    R = [json.load(open(r + ".json", encoding="utf-8")) for r in runs]
    B, u = bar["B"], bar["u"]
    u_src = "bars"
    if u is None:                       # no increment in the reference episodes: the runs' sensed rises
        meds = [r["median_rise"] for r in R if r.get("median_rise") is not None]
        u, u_src = (max(0.5 * abs(B), float(np.median(meds))), "runs") if meds else (None, "none")
    row = dict(game=g, complete=True, B=B, u=u, u_src=u_src, rises_300k=[r["rises_300k"] for r in R],
               best=[r["records"][-1][1] if r["records"] else None for r in R],
               entry_frames=[r["entry"]["entry_frames"] for r in R], lost_frac=[r["entry"]["lost_frac"] for r in R],
               probe_share=[r["summary"]["probe_frames"] / max(1, r["summary"]["billed"]) for r in R],
               cells=[r["summary"]["cells"] for r in R], iters=[r["summary"]["iters"] for r in R])
    if u is None or u <= 0:
        row.update(room=False, why="no step")
        return row
    lad = ladder(B, u)
    recs = [r["records"] for r in R]
    k_star = min(top_rung(x, lad) for x in recs)
    row["k_star"] = k_star
    T1 = [first_T(x, lad[1]) for x in recs]
    row["T1"] = T1
    a = all(n >= 10 for n in row["rises_300k"])
    b = k_star >= 3
    c = all(t is not None for t in T1) and float(np.median(T1)) <= 1_000_000
    C = None
    if b:
        C = [first_T(x, lad[k_star]) - first_T(x, lad[1]) for x in recs]
        row["C"] = C
        row["C_late"] = [first_T(x, lad[k_star]) - first_T(x, lad[2]) for x in recs]
    d = C is not None and float(np.median(C)) >= 300_000
    row.update(a=a, b=b, c=c, d=d, room=bool(a and b and c and d))
    if replay and k_star >= 1:
        checks = []
        for s, x, stem in zip(SEEDS["room"], recs, runs):
            npz = np.load(stem + ".npz")
            for k in sorted({1, 2, k_star} & set(range(k_star + 1))):
                i = first_index(x, lad[k])
                got, want, ok = confirm(g, s, npz, x, i)
                checks.append(dict(seed=s, k=k, replayed=got, recorded=want, ok=ok))
        row["replay"] = checks
        row["interruption"] = not all(c["ok"] for c in checks)
    return row


def verdict(replay=True):
    rows = [game_row(g, replay) for g in DEV + CALIBRATION]
    dev = [r for r in rows if r["game"] in DEV]
    complete = all(r["complete"] for r in dev)
    room = [r for r in dev if r.get("room")]
    n_room = len(room)
    s2 = float(np.mean([(np.log(r["C"][0]) - np.log(r["C"][1])) ** 2 / 2 for r in room])) if room else None
    g_proj = 17 * n_room / 35
    se = float(np.sqrt((s2 / 2 + 0.09) / g_proj)) if room else None
    passed = complete and n_room >= 13 and se is not None and se <= 0.23
    interrupted = [r["game"] for r in rows if r.get("interruption")]
    return dict(rule="docs/ARC4_V83_PROTOCOL.md §4 A1", complete=complete, n_room_dev=n_room,
                room_dev=[r["game"] for r in room], room_cal=[r["game"] for r in rows if r["game"] in CALIBRATION and r.get("room")],
                sigma2=s2, sigma=None if s2 is None else float(np.sqrt(s2)), G_proj=g_proj, se_proj=se,
                G_room="PASS" if passed else ("KC-room" if complete else "incomplete"),
                interruptions=interrupted, senses="score rises and lives given as senses (owner decision 2026-10-22)",
                rows=rows, provenance=stamp())


if __name__ == "__main__":
    v = verdict(replay="--no-replay" not in sys.argv)
    path = f"{OUT}/room_verdict.json"
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: v[k] for k in ("complete", "n_room_dev", "room_dev", "sigma", "se_proj", "G_room", "interruptions")}))
