"""v84 G1a report and G1b room verdict (docs/ARC4_V84_PROTOCOL_EN.md §7.1-7.3, §8), evaluator side.

Ladder: B' and u' from bars_v84 (seed 9706), M_k = B' + 2^k u'. T(M) = billed frames at the first record
(new best of the cumulative SENSED rise) >= M, replay-confirmed for M1, M2 and M_top (one NOOP before
restoring the base; a mismatch is an Interruption).

G1b, per game on both seeds 9712-9713 (cap 4M): (a) >= 10 sensed rises in the first 300k billed frames;
(b) H = min over seeds of the best cumulative rise >= M3; (c) median T(M1) <= 1.5M; (d) median
T(M_k*) - T(M1) >= 300k, k* = highest rung reached by both seeds. PROCEED if >= 13 of the 35 development
games are room games AND se_proj = sqrt((s2/3 + 0.09)/(17 n/35)) <= 0.23, s2 = pooled (lambda1 -
lambda2)^2 / 2 over room development games, winsorised at the 90th percentile, lambda = mean over
k = 2..k* of log max(5000, T(M_k) - T(M1)). Caption "climbs on most dev games" if >= 18/35 with M3
reached at lineage depth <= 7,000 steps on both seeds. Otherwise KC-room-84.

  python scripts/room_verdict_v84.py g1a
  python scripts/room_verdict_v84.py g1b
"""
import json
import math
import os
import pathlib
import pickle
import sys
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v71.env import AtariEnv                                     # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV                       # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp                  # noqa: E402
from scripts.queue_v84 import G1A                                                # noqa: E402

OUT = ROOT / "craft_v6_out" / "v84"
CAP = 4_000_000
PATHS = ("ragnarok", "scripts", "tests")
A1_ROOM = pathlib.Path(r"F:\dev\ragnarok_v83\craft_v6_out\v83\room")
CLAIM = ("Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 "
         "(score rises, lives count). It is not live play without savestates.")


def ladder(g):
    b = json.load(open(OUT / "bars" / f"{g}_9706.json", encoding="utf-8"))
    return b["ladder"], b


def first(records, m):
    """records [[billed, cum, true, depth], ...] -> (billed, depth) of the first record with cum >= m."""
    for r in records:
        if r[1] >= m:
            return r[0], r[3]
    return None, None


def top_rung(records, lad):
    best = max((r[1] for r in records), default=-1.0)
    ks = [k for k, m in enumerate(lad) if best >= m]
    return max(ks) if ks else -1


def replay_cum(game, seed, npz, idx):
    off = npz["rec_offsets"]
    acts = npz["rec_actions"][off[idx]:off[idx + 1]]
    env = AtariEnv(game, seed=seed)
    env.step(0)
    env.ale.restoreState(pickle.loads(zlib.decompress(bytes(npz["base_state"]))))
    env._true_score, cum = 0.0, 0.0
    for a in acts:
        before = env._true_score
        env.step(int(a))
        cum += max(0.0, env._true_score - before)
    return cum


def lam(records, lad, ks, k_from=1):
    t1 = first(records, lad[k_from])[0]
    v = [math.log(max(5000, first(records, lad[k])[0] - t1)) for k in range(k_from + 1, ks + 1)]
    return float(np.mean(v)) if v else None


def game_row(g, stems, seeds, replay=True):
    lad, bar = ladder(g)
    R = [json.load(open(str(s) + ".json", encoding="utf-8")) for s in stems]
    recs = [r["records"] for r in R]
    ks = min(top_rung(x, lad) for x in recs)
    row = dict(game=g, B_prime=bar["B_prime"], u_prime=bar["u_prime"], k_star=ks,
               best=[x[-1][1] if x else None for x in recs], best_true=[x[-1][2] if x else None for x in recs],
               rises_300k=[r["rises_300k"] for r in R], cells=[r["summary"]["cells"] for r in R],
               expansions=[r["summary"]["expansions"] for r in R], billed=[r["summary"]["billed"] for r in R],
               code=sorted({(r["provenance"]["code_tree"], r["provenance"]["dirty"]) for r in R}),
               me_lost=[r["entry"]["lost_frac"] for r in R])
    T1 = [first(x, lad[1])[0] for x in recs]
    row["T1"] = T1
    row["reached_M1"] = [t is not None for t in T1]
    row["reached_M3"] = [first(x, lad[3])[0] is not None for x in recs]
    row["depth_M3"] = [first(x, lad[3])[1] for x in recs]
    a = all(n >= 10 for n in row["rises_300k"])
    b = ks >= 3
    c = all(t is not None for t in T1) and float(np.median(T1)) <= 1_500_000
    d = False
    if b:
        C = [first(x, lad[ks])[0] - first(x, lad[1])[0] for x in recs]
        row["C"] = C
        row["lambda"] = [lam(x, lad, ks) for x in recs]
        d = float(np.median(C)) >= 300_000
    row.update(a=a, b=b, c=c, d=d, room=bool(a and b and c and d))
    if replay and ks >= 1:
        checks = []
        for s, x, stem in zip(seeds, recs, stems):
            npz = np.load(str(stem) + ".npz")
            for k in sorted({1, 2, ks} & set(range(ks + 1))):
                i = next(j for j, r in enumerate(x) if r[1] >= lad[k])
                got = replay_cum(g, s, npz, i)
                checks.append(dict(seed=s, k=k, replayed=got, recorded=x[i][1], ok=bool(got == x[i][1])))
        row["replay"] = checks
        row["interruption"] = not all(c["ok"] for c in checks)
    return row


def a1_T_M3(g, lad):
    """V's A1 median T(M3) re-scored on the v84 ladder (A1 records are true-score bests; the controls,
    ms_pacman and amidar, have no negative rewards, so true score = sensed rises)."""
    ts = []
    for s in (9131, 9132):
        r = json.load(open(A1_ROOM / f"room_{g}_{s}.json", encoding="utf-8"))["records"]
        t = next((b for b, v in r if v >= lad[3]), None)
        ts.append(t)
    return ts


def cmd_g1a():
    prov = stamp((), cwd=str(ROOT), paths=PATHS)
    rows = []
    for g in G1A:
        stem = OUT / "g1a" / f"{g}_9710"
        lad, _ = ladder(g)
        r = json.load(open(str(stem) + ".json", encoding="utf-8"))
        best = r["records"][-1][1] if r["records"] else 0.0
        row = dict(game=g, best=best, M1=lad[1], M3=lad[3], reached_M1=best >= lad[1], reached_M3=best >= lad[3],
                   T_M3=first(r["records"], lad[3])[0], cells=r["summary"]["cells"], wall_s=r["wall_s"],
                   peak_commit_mb=r["peak_commit_mb"], billed=r["summary"]["billed"])
        if g in ("ms_pacman", "amidar"):
            row["A1_T_M3"] = a1_T_M3(g, lad)
        rows.append(row)
    panel = [x for x in rows if x["game"] not in ("ms_pacman", "amidar")]
    out = dict(rule="G1a (exploratory, no kill)", panel_M1=sum(x["reached_M1"] for x in panel),
               panel_M3=sum(x["reached_M3"] for x in panel), n_panel=len(panel), rows=rows,
               caption=f"V2 at its frozen constants reaches M3 on {sum(x['reached_M3'] for x in panel)}/{len(panel)} games of the A1-failure panel",
               claim_header=CLAIM, provenance=prov)
    with open(OUT / "g1a_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("panel_M1", "panel_M3", "n_panel", "caption")}))


def cmd_g1b():
    prov = stamp((), cwd=str(ROOT), paths=PATHS)
    rows = [game_row(g, [OUT / "g1b" / f"{g}_{s}" for s in (9712, 9713)], (9712, 9713)) for g in DEV + CALIBRATION]
    dev = [r for r in rows if r["game"] in DEV]
    room = [r for r in dev if r["room"]]
    n = len(room)
    d2 = sorted((r["lambda"][0] - r["lambda"][1]) ** 2 / 2 for r in room if None not in r["lambda"])
    if d2:
        cap90 = float(np.percentile(d2, 90))
        s2 = float(np.mean(np.minimum(d2, cap90)))
    else:
        s2 = None
    se = float(math.sqrt((s2 / 3 + 0.09) / (17 * n / 35))) if (s2 is not None and n) else None
    codes = sorted({tuple(c) for r in rows for c in r["code"]})
    interrupted = [r["game"] for r in rows if r.get("interruption")]
    climbs = sum(1 for r in dev if r["room"] and all(x is not None and x <= 7000 for x in r["depth_M3"]))
    proceed = n >= 13 and se is not None and se <= 0.23
    if interrupted or len(codes) > 1 or any(c[1] for c in codes):
        label = "Interruption"
    else:
        label = "PROCEED" if proceed else "KC-room-84"
    out = dict(rule="docs/ARC4_V84_PROTOCOL_EN.md §8 G1b", n_room_dev=n, room_dev=[r["game"] for r in room],
               room_cal=[r["game"] for r in rows if r["game"] in CALIBRATION and r["room"]], sigma2_winsor=s2,
               se_proj=se, G1b=label, climbs_most=climbs >= 18, n_climbs=climbs, interruptions=interrupted, codes=codes,
               rows=rows, claim_header=CLAIM, provenance=prov)
    with open(OUT / "room_verdict_v84.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("G1b", "n_room_dev", "room_dev", "se_proj", "n_climbs", "interruptions")}))


if __name__ == "__main__":
    dict(g1a=cmd_g1a, g1b=cmd_g1b)[sys.argv[1]]()
