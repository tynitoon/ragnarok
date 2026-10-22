"""v83 A1 queue (docs/ARC4_V83_PROTOCOL.md §4 A1): bars on the dev bar seed, then arm V on every
development and calibration game with the room seeds, cap 2M billed frames. Resumable (a job whose
output exists is skipped), one subprocess per job, waits for memory (gate copied from slice_v74:
physical RAM and commit) rather than skipping, provenance stamped by each job. Shard k/n takes every
n-th job so two workers share the queue without overlap.

  python scripts/room_v83.py run k/n
"""
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV, SEEDS   # noqa: E402

OUT = "craft_v6_out/v83"
CAP = 2_000_000
GAMES = DEV + CALIBRATION


def free_gb():
    import ctypes
    import ctypes.wintypes as wt

    class M(ctypes.Structure):
        _fields_ = [("l", wt.DWORD), ("m", wt.DWORD), ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                    ("x", ctypes.c_ulonglong * 5)]
    m = M(); m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.a, m.x[1]) / 2 ** 30      # physical RAM and commit (x[1] = available page file)


def gate(min_gb=14.0, wait_s=6 * 3600):
    """Do not start while less than `min_gb` is free (Tavern keeps 12 GB). Waits rather than skipping."""
    t0 = time.time()
    while free_gb() < min_gb and time.time() - t0 < wait_s:
        time.sleep(60)
    if free_gb() < min_gb:
        sys.exit(f"RAM gate: only {free_gb():.1f} GB free after {wait_s}s, not starting")


def queue():
    jobs = [(f"{OUT}/bars/{g}_{SEEDS['dev_bars']}.json",
             ["scripts/bars_v83.py", str(SEEDS["dev_bars"]), f"{OUT}/bars/{g}_{SEEDS['dev_bars']}.json", g])
            for g in GAMES]
    for s in SEEDS["room"]:
        for g in GAMES:
            stem = f"{OUT}/room/room_{g}_{s}"
            jobs.append((stem + ".json", ["scripts/room_job_v83.py", g, str(s), str(CAP), stem]))
    return jobs


def run_all(shard, passes=3):
    k, n = (int(x) for x in shard.split("/"))
    os.makedirs(f"{OUT}/bars", exist_ok=True)
    os.makedirs(f"{OUT}/room", exist_ok=True)
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    for _ in range(passes):
        todo = [(o, a) for i, (o, a) in enumerate(queue()) if i % n == k and not os.path.exists(o)]
        if not todo:
            break
        for out, args in todo:
            if os.path.exists(out):
                continue
            gate()
            name = pathlib.Path(out).stem
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            with open(f"{OUT}/logs/{name}.log", "w") as f:
                r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_all(sys.argv[2])
    elif sys.argv[1] == "status":
        q = queue()
        done = sum(os.path.exists(o) for o, _ in q)
        print(f"{done}/{len(q)} jobs done")
