"""v84 queue (docs/ARC4_V84_PROTOCOL_EN.md §8): resumable, one subprocess per job, waits for memory
(13 GB free physical RAM and commit) rather than skipping, k/n sharding.

  python scripts/queue_v84.py bars k/n      bars B' for the 41 dev + calibration games (seed 9706)
  python scripts/queue_v84.py smoke7 k/n    G-build test 7: one 4M V2 run (seed 9700, amidar)
  python scripts/queue_v84.py g1a k/n       G1a: 10 panel games + 2 controls, seed 9710, cap 4M
  python scripts/queue_v84.py g1b k/n       G1b: 41 games x seeds 9712-9713, cap 4M
"""
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV          # noqa: E402
from scripts.room_v83 import free_gb                                # noqa: E402

OUT = ROOT / "craft_v6_out" / "v84"
CAP = 4_000_000
GAMES = DEV + CALIBRATION
G1A = ("atlantis", "krull", "centipede", "video_pinball", "gopher", "name_this_game", "zaxxon", "bowling",
       "seaquest", "phoenix", "ms_pacman", "amidar")


def gate(min_gb=13.0, wait_s=6 * 3600):
    t0 = time.time()
    while free_gb() < min_gb and time.time() - t0 < wait_s:
        time.sleep(60)
    if free_gb() < min_gb:
        sys.exit(f"RAM gate: only {free_gb():.1f} GB free after {wait_s}s, not starting")


def jobs(kind):
    if kind == "bars":
        return [(OUT / "bars" / f"{g}_9706.json", ["scripts/bars_v84.py", "9706", str(OUT / "bars"), g]) for g in GAMES]
    if kind == "smoke7":
        stem = OUT / "smoke" / "t7_amidar_9700"
        return [(pathlib.Path(str(stem) + ".json"), ["scripts/v2_job_v84.py", "amidar", "9700", str(CAP), str(stem)])]
    if kind == "g1a":
        return [(OUT / "g1a" / f"{g}_9710.json", ["scripts/v2_job_v84.py", g, "9710", str(CAP), str(OUT / "g1a" / f"{g}_9710")])
                for g in G1A]
    if kind == "g1b":
        return [(OUT / "g1b" / f"{g}_{s}.json", ["scripts/v2_job_v84.py", g, str(s), str(CAP), str(OUT / "g1b" / f"{g}_{s}")])
                for s in (9712, 9713) for g in GAMES]
    raise SystemExit(kind)


def run_all(kind, shard, passes=3):
    k, n = (int(x) for x in shard.split("/"))
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    for _ in range(passes):
        todo = [(o, a) for i, (o, a) in enumerate(jobs(kind)) if i % n == k and not o.exists()]
        if not todo:
            break
        for out, args in todo:
            if out.exists():
                continue
            gate()
            name = f"{kind}_{out.stem}"
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            with open(OUT / "logs" / f"{name}.log", "w") as f:
                r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    run_all(sys.argv[1], sys.argv[2])
