"""Spike benchmark, Phase 2: device-resident PPO loop throughput.

Measures env-steps/sec for the full accelerator-resident PPO loop —
collect_rollout -> train_on_rollout -> normalizer.update — across
parallel-env counts N.

scripts/bench_device_env.py measured COLLECTION alone. This adds the PPO
training step (the Stage 2 deliverable). Two questions:
  - does the device train loop keep the accelerator fed, or stall it?
  - where is the large-N crossover vs a CUDA GPU? A TPU is throughput-
    optimised — it wins at large N (big matmuls); a GPU wins at small N
    (low dispatch latency). bench_device_env.py showed this for
    collection; this checks it for collect+train.

Per N: a collect-only timed loop and a collect+train timed loop — the
gap between them is the PPO-update cost.

Usage:  python -m scripts.bench_device_ppo
"""

import time
import torch

from ragnarok.infrastructure.device import DEVICE, IS_XLA
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceRunningNormalizer)
from ragnarok.learning.rollout import collect_rollout
from ragnarok.learning.real_experience import RealExperienceTrainer

N_VALUES = [256, 1024, 4096, 16384]
HORIZON = 128
WARMUP = 3
TIMED = 12


def _sync():
    """Block until all device work completes — for honest wall-clock timing."""
    if IS_XLA:
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench(n: int) -> dict:
    trainer = RealExperienceTrainer(obs_dim=4, action_dim=2, discrete=True)
    env = DeviceVecCartPole(n)
    normalizer = DeviceRunningNormalizer(obs_dim=4)

    def collect():
        return collect_rollout(env, trainer.device_policy_fn, HORIZON,
                               normalizer=normalizer)

    def collect_train():
        batch = collect()
        trainer.train_on_rollout(batch)
        normalizer.update(batch.raw_obs.reshape(-1, 4))

    for _ in range(WARMUP):       # pay XLA compile for both graphs
        collect_train()
    _sync()

    t0 = time.perf_counter()
    for _ in range(TIMED):
        collect()
    _sync()
    collect_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(TIMED):
        collect_train()
    _sync()
    full_wall = time.perf_counter() - t0

    steps = n * HORIZON * TIMED
    return {
        "n": n,
        "collect_sps": steps / collect_wall,
        "full_sps": steps / full_wall,
    }


def main():
    print(f"[bench-device-ppo] device={DEVICE}")
    print(f"  horizon={HORIZON}  warmup={WARMUP}  timed={TIMED}\n")
    print(f"{'N':>7} | {'transitions':>12} | {'collect/s':>14} | "
          f"{'collect+train/s':>16}")
    print("-" * 58)
    for n in N_VALUES:
        r = bench(n)
        print(f"{n:>7} | {n * HORIZON:>12,} | {r['collect_sps']:>14,.0f} | "
              f"{r['full_sps']:>16,.0f}")
    print("\n[bench-device-ppo] done")


if __name__ == "__main__":
    main()
