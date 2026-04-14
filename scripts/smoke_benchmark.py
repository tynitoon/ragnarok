"""Compute-budget smoke benchmark (preregistration §12.5).

Phase 5 (H1 primary + secondary + ablations) is gated on a measured
compute table for the target hardware. This script runs a small
multi-seed job per env and writes `compute_budget.json` with wall-clock
per env-step — callers can then extrapolate to the full 20-seed primary
+ 10-seed secondary + 5-seed ablations budget before committing.

Usage (full smoke, ~60-90 min total on a modern GPU):
    python -m scripts.smoke_benchmark \
        --envs cartpole mountaincar acrobot pendulum mountaincar-continuous \
        --seeds 3 --steps 50000 --output compute_budget.json

Fast sanity check (~1 min):
    python -m scripts.smoke_benchmark --envs cartpole --seeds 1 --steps 1000

Output schema (compute_budget.json):
    {
      "hardware": {"device": "cuda|cpu", "gpu_name": "...", "python": "3.14.3"},
      "runs": [
        {"env": "cartpole", "seed": 42, "steps": 50000,
         "wall_clock_sec": 123.4, "steps_per_sec": 405.2,
         "episodes_completed": 312}
      ],
      "summary": {
        "<env>": {"mean_steps_per_sec": ..., "std_steps_per_sec": ...}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.registry import get_env_spec, make_env
from ragnarok.core.agent import RagnarokAgent


@dataclass
class SmokeRun:
    env: str
    seed: int
    steps: int
    wall_clock_sec: float
    steps_per_sec: float
    episodes_completed: int


def _run_one(env_name: str, seed: int, target_steps: int) -> SmokeRun:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim
    # Smoke runs must reflect the default (benchmark-clean) code path —
    # no env_overrides, no reward shaping (preregistration §6.1 fix #3).
    config.reward_shaping.enabled = False
    config.env_overrides.enabled = False

    env = make_env(env_name, seed=seed)
    agent = RagnarokAgent(config, env)

    start = time.time()
    while agent.total_steps < target_steps:
        agent.collect_episode()
        # Train on a small schedule matching real training loops
        if agent.total_episodes % 10 == 0 and agent.replay_buffer.num_episodes >= 5:
            agent.train_world_model(steps=5)
        if agent.replay_buffer.num_episodes >= 5:
            agent.train_policy_dream(steps=2)
    elapsed = time.time() - start

    steps = agent.total_steps
    env.close()

    return SmokeRun(
        env=env_name,
        seed=seed,
        steps=steps,
        wall_clock_sec=elapsed,
        steps_per_sec=steps / max(elapsed, 1e-6),
        episodes_completed=agent.total_episodes,
    )


def _summarize(runs: list[SmokeRun]) -> dict[str, dict[str, float]]:
    by_env: dict[str, list[float]] = {}
    for r in runs:
        by_env.setdefault(r.env, []).append(r.steps_per_sec)
    summary = {}
    for env, sps_list in by_env.items():
        arr = np.array(sps_list)
        summary[env] = {
            "mean_steps_per_sec": float(arr.mean()),
            "std_steps_per_sec": float(arr.std(ddof=0)),
            "n_seeds": len(arr),
        }
    return summary


def _hardware_info() -> dict[str, str]:
    info = {
        "device": str(DEVICE),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    if DEVICE.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = str(torch.cuda.get_device_capability(0))
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=["cartpole"],
                        help="Env names to benchmark")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds per env")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Target env-steps per run")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="First seed (seeds run from base_seed..base_seed+seeds-1)")
    parser.add_argument("--output", type=Path, default=Path("compute_budget.json"),
                        help="Output JSON path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output instead of overwriting")
    args = parser.parse_args(argv)

    runs: list[SmokeRun] = []
    if args.append and args.output.exists():
        existing = json.loads(args.output.read_text())
        runs = [SmokeRun(**r) for r in existing.get("runs", [])]

    print(f"Hardware: {_hardware_info()}", flush=True)
    print(f"Target: {len(args.envs)} env(s) × {args.seeds} seed(s) "
          f"× {args.steps:,} steps", flush=True)

    for env_name in args.envs:
        for s in range(args.seeds):
            seed = args.base_seed + s
            print(f"[{env_name} seed={seed}] running {args.steps:,} steps...",
                  end=" ", flush=True)
            r = _run_one(env_name, seed, args.steps)
            runs.append(r)
            print(f"{r.wall_clock_sec:.1f}s "
                  f"({r.steps_per_sec:.0f} steps/s, "
                  f"{r.episodes_completed} eps)", flush=True)

            # Flush after every run so partial results survive interruption
            payload = {
                "hardware": _hardware_info(),
                "runs": [asdict(r) for r in runs],
                "summary": _summarize(runs),
            }
            args.output.write_text(json.dumps(payload, indent=2))

    print(f"\nWrote {args.output}", flush=True)
    print("Summary:", flush=True)
    for env, s in _summarize(runs).items():
        print(f"  {env:>30s}: {s['mean_steps_per_sec']:7.1f} ± "
              f"{s['std_steps_per_sec']:5.1f} steps/s  (n={s['n_seeds']})",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
