"""One V2 run (docs/ARC4_V84_PROTOCOL_EN.md §5, §8 G1a/G1b), evaluator side: SenseEnv84 with a billed
cap and Trace84, v83 entry (Body.enter), V2 explorer; writes OUT_STEM.json (summary, entry, records of the
cumulative sensed rise with true score and depth), OUT_STEM.npz (expansion rows, record actions, base
state) and OUT_STEM_rows/ (promise row chunks).

  python scripts/v2_job_v84.py GAME SEED CAP OUT_STEM
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.entry import Body                                   # noqa: E402
from ragnarok.agent_v84.explorer2 import Explorer2                          # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp             # noqa: E402
from ragnarok.evaluation.senses_v84 import SenseEnv84, Spent, Trace84       # noqa: E402
from scripts.room_job_v83 import peak_rss_mb                                # noqa: E402

PATHS = ("ragnarok", "scripts", "tests")
CLAIM = ("Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 "
         "(score rises, lives count). It is not live play without savestates.")


def run(game, seed, cap, stem, stop_at=None, steer=None, extra=None):
    prov = stamp((), cwd=str(ROOT), paths=PATHS)
    t0 = time.time()
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tr = Trace84()
    env = SenseEnv84(game, seed=seed, cap=cap, trace=tr)
    rng = np.random.default_rng(seed)
    rng_prop = np.random.default_rng([seed, 1])
    body = Body()
    entry = body.enter(env, rng, seed)
    tr.stop_at = stop_at
    ex = Explorer2(body, str(stem) + "_rows", budget=cap, steer=steer)
    summ = ex.run(env, rng, rng_prop, spent=Spent)
    recs = tr.records
    acts, offs, base = [], [0], None
    for r in recs:
        b, a = tr.actions(r[2])
        base = b if base is None else base
        assert b == base
        acts.append(np.frombuffer(a, np.uint8))
        offs.append(offs[-1] + len(a))
    np.savez_compressed(str(stem) + ".npz", exp_feat=np.array(ex.exp_rows, np.float32).reshape(-1, 44),
                        exp_meta=np.array(ex.exp_meta, np.float64).reshape(-1, 8),
                        rec_actions=np.concatenate(acts) if acts else np.zeros(0, np.uint8),
                        rec_offsets=np.array(offs, np.int64),
                        base_state=np.frombuffer(tr.bases[base], np.uint8) if base is not None else np.zeros(0, np.uint8))
    ws, commit = peak_rss_mb()
    out = dict(game=game, seed=seed, cap=cap, stop_at=stop_at, steer=None if steer is None else type(steer).__name__,
               entry=entry, summary=summ, records=[[b, c, t, d] for b, c, _, t, d in recs],
               n_rises=tr.n_rises, rises_300k=sum(b <= 300000 for b in tr.rise_billed),
               median_rise=float(np.median(tr.rise_sizes)) if tr.rise_sizes else None,
               peak_ws_mb=ws, peak_commit_mb=commit, wall_s=time.time() - t0, claim_header=CLAIM,
               provenance=prov, provenance_end=stamp((), cwd=str(ROOT), paths=PATHS), npz_sha256=full_hash(str(stem) + ".npz"),
               **(extra or {}))
    with open(str(stem) + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(str(stem) + ".json.tmp", str(stem) + ".json")
    return out


if __name__ == "__main__":
    g, s, cap, stem = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    o = run(g, s, cap, stem)
    print(json.dumps(dict(summary=o["summary"], best=o["records"][-1] if o["records"] else None,
                          peak_ws_mb=round(o["peak_ws_mb"]), peak_commit_mb=round(o["peak_commit_mb"]), wall_s=round(o["wall_s"]))))
