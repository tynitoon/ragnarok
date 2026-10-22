"""One v83 explorer run (evaluator side): senses with a billed cap, entry, explorer V (or a promise arm),
then the trace of the explorer epoch (records + replayable action strings), rows and iteration logs.

  python scripts/room_job_v83.py GAME SEED CAP OUT_STEM [--stop-at M]

Writes OUT_STEM.json (summary, entry report, records) and OUT_STEM.npz (rows, iterations, record
actions, base state). The agent sees only the senses; this script holds the game name.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragnarok.agent_v83.entry import Body                      # noqa: E402
from ragnarok.agent_v83.explorer import Explorer                # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp            # noqa: E402
from ragnarok.evaluation.senses_v83 import SenseEnv, Spent, Trace   # noqa: E402


def peak_rss_mb():
    """Peak working set and peak commit of this process (Windows), MB."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20, c.PeakPagefileUsage / 2**20


def run(game, seed, cap, stem, stop_at=None, max_iters=None, steer=None):
    prov = stamp()                                 # the code at launch, not HEAD at the end
    t0 = time.time()
    tr = Trace()
    env = SenseEnv(game, seed=seed, cap=cap, trace=tr)
    rng = np.random.default_rng(seed)
    body = Body()
    entry = body.enter(env, rng, seed)
    tr.stop_at = stop_at
    ex = Explorer(body, steer=steer)
    summ = ex.run(env, rng, spent=Spent, max_iters=max_iters)
    recs = tr.records
    acts, offs = [], [0]
    base = None
    for _, _, h in recs:
        b, a = tr.actions(h)
        base = b if base is None else base
        assert b == base
        acts.append(np.frombuffer(a, np.uint8))
        offs.append(offs[-1] + len(a))
    rows, iters = ex.rows.arrays(), ex.iters.arrays()
    peak_ws, peak_commit = peak_rss_mb()
    out = dict(game=game, seed=seed, cap=cap, stop_at=stop_at, max_iters=max_iters,
               steer=None if steer is None else type(steer).__name__, entry=entry, summary=summ,
               records=[[b, t] for b, t, _ in recs], n_rises=tr.n_rises,
               rises_300k=sum(b <= 300000 for b in tr.rise_billed),
               median_rise=float(np.median(tr.rise_sizes)) if tr.rise_sizes else None, n_rows=int(len(rows["it"])), peak_ws_mb=peak_ws, peak_commit_mb=peak_commit,
               wall_s=time.time() - t0, provenance=prov)
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(stem) + ".npz", **{f"row_{k}": v for k, v in rows.items()},
                        **{f"iter_{k}": v for k, v in iters.items()},
                        rec_actions=np.concatenate(acts) if acts else np.zeros(0, np.uint8),
                        rec_offsets=np.array(offs, np.int64),
                        base_state=np.frombuffer(tr.bases[base], np.uint8) if base is not None else np.zeros(0, np.uint8))
    tmp = str(stem) + ".json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, str(stem) + ".json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("seed", type=int)
    ap.add_argument("cap", type=int)
    ap.add_argument("stem")
    ap.add_argument("--stop-at", type=float, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    a = ap.parse_args()
    o = run(a.game, a.seed, a.cap, a.stem, a.stop_at, a.max_iters)
    print(json.dumps(dict(summary=o["summary"], entry={k: o["entry"][k] for k in ("entry_frames", "lost_frac", "n_resp")},
                          n_rows=o["n_rows"], best=o["records"][-1] if o["records"] else None,
                          peak_ws_mb=round(o["peak_ws_mb"]), peak_commit_mb=round(o["peak_commit_mb"]), rises_300k=o["rises_300k"], wall_s=round(o["wall_s"]))))
