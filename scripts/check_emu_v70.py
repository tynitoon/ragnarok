"""Bounded CPU development check; no RAM, scorer, training, gate or held-out game."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig, EmulatorPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("craft_v6_out/v70_cpu_instrument.json"))
    args = parser.parse_args()
    if not 160 <= args.steps <= 2000:
        parser.error("--steps must be in 160..2000 for this bounded development check")
    if args.out.exists():
        parser.error("output already exists; keep the old record and choose a new path")
    configs = [EmulatorConfig(seed=17001, max_charged_frames=20_000)] * args.workers
    started = time.perf_counter()
    with EmulatorPool(configs) as pool:
        startup_seconds = time.perf_counter() - started
        first = [pool.accounting(i) for i in range(args.workers)]
        tic = time.perf_counter()
        for step in range(args.steps):
            pool.step([step % 18] * args.workers)
        step_seconds = time.perf_counter() - tic
        before = [pool.accounting(i) for i in range(args.workers)]
        raw, pooled = pool.observe(raw=True), pool.observe()
        intervention, control = pool.branch_pair(3, control=3)
        replay_equal = bool(np.array_equal(intervention, control))
        restored_equal = bool(np.array_equal(raw, pool.observe(raw=True)) and np.array_equal(pooled, pool.observe()))
        end = [pool.accounting(i) for i in range(args.workers)]
        final_step = pool.step([4] * args.workers)
        unchanged = all(np.array_equal(final_step[0], image) for image in final_step[1:])
    root = Path(__file__).resolve().parents[1]
    paths = ["ragnarok/environments/emu_pool.py", "ragnarok/environments/frame_budget_v70.py",
             "scripts/check_emu_v70.py"]
    result = dict(kind="development_cpu_instrument_check", game="pong", mode=0,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  audited_rom_md5="60e0ea3cbe0913d39803477945e9e5ec",
                  python=sys.executable, workers=args.workers, steps_per_worker=args.steps,
                  versions={name: importlib.metadata.version(name) for name in ("ale-py", "numpy")},
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
                  startup_seconds=startup_seconds, ordinary_step_seconds=step_seconds,
                  agent_steps_per_second=args.steps * args.workers / step_seconds,
                  ordinary_emulated_frames_per_second=sum(b["emulated_frames"] - a["emulated_frames"]
                                                         for a, b in zip(first, before)) / step_seconds,
                  replay_equal=replay_equal, restored_pixels_equal=restored_equal,
                  ordinary_trajectory_unchanged=(unchanged if args.workers > 1 else None),
                  before_pair=before, after_pair=end, gate_result=None,
                  limitations=["Only Pong mode 0 bootstrap is audited; other ROMs disabled.",
                               "Sticky actions use worker RNG, not ALE's native sticky RNG stream.",
                               "CPU pool throughput excludes parser, scorer, labels and training.",
                               "No learned perception, competence or transfer claim."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if not replay_equal or not restored_equal or not unchanged:
        raise SystemExit("Instrument check failed; no gate may use these results")


if __name__ == "__main__":
    main()
