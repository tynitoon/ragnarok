"""Bounded Pong CPU chain benchmark: emulation, IPC, reference, RAM scorer and C pairs.

It scores the last raw frame at repeat4. Max-pooled temporal calibration and the
rest of gate0-A remain separate work; no gate verdict is emitted here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.evaluation.reference_pool_v70 import ReferencePool


def benchmark(workers, steps):
    configs = [EmulatorConfig(seed=18000+i, frame_skip=4, max_charged_frames=100_000) for i in range(workers)]
    started = time.perf_counter()
    with ReferencePool(configs) as pool:
        startup_seconds = time.perf_counter() - started
        tic = time.perf_counter()
        for step in range(steps):
            # A real request/response and full image payload at each ordinary decision.
            outputs = pool.step([(step // 4 + i) % 18 for i in range(workers)])
            if any(rgb.shape != (210,160,3) or boxes.ndim != 2 or boxes.shape[1] != 4 for rgb, boxes in outputs):
                raise RuntimeError("Malformed pixel-derived observation")
        work_seconds = time.perf_counter() - tic
        reports = [pool.report(i) for i in range(workers)]
    root = Path(__file__).resolve().parents[1]
    sources = ["scripts/bench_reference_pool_v70.py", "ragnarok/evaluation/reference_pool_v70.py",
               "scripts/ref_extract_v70.py", "ragnarok/evaluation/perception_v70.py",
               "ragnarok/evaluation/pong_v70.py", "ragnarok/environments/emu_pool.py",
               "ragnarok/environments/frame_budget_v70.py"]
    return dict(kind="pong_raw_repeat4_cpu_chain_benchmark", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                workers=workers, ordinary_decisions_per_worker=steps, action_repeat=4,
                branch_every_decisions=128, startup_seconds=startup_seconds, work_seconds=work_seconds,
                ordinary_agent_steps_per_second=workers*steps/work_seconds,
                including_experiment_agent_steps_per_second=sum(r["accounting"]["ordinary_agent_steps"] +
                    r["accounting"]["experiment_agent_steps"] for r in reports)/work_seconds,
                emulated_frames_per_second=sum(r["accounting"]["emulated_frames"]-72 for r in reports)/work_seconds,
                logical_processors=os.cpu_count(), worker_reports=reports,
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy","scipy")},
                gate_result=None, gpu_used=False, reference_uses_oracle_reset=False,
                ordinary_and_branch_images="last_raw_frame",
                limitations=["Pong development only; action-repeat4 but last raw frame, not max-pooled truth.",
                             "Includes per-step full RGB IPC, causal reference, every-real-frame RAM decoding, "
                             "end-of-repeat scoring, charged branch pairs and pixel C targets.",
                             "Startup separately reported; ARC3 GPU pretraining is concurrently active.",
                             "No learned parser, other games, reader/drives or complete gate0-A verdict."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,choices=(1,2,4,8,16),default=2)
    parser.add_argument("--steps",type=int,default=512)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if not 256 <= args.steps <= 2048:
        parser.error("Keep this development check in256..2048 steps per worker")
    if args.out.exists():
        parser.error("Keep old measurements; choose a new output")
    result = benchmark(args.workers,args.steps)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("kind","workers","startup_seconds","work_seconds","ordinary_agent_steps_per_second")},indent=2))
