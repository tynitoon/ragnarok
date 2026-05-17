"""Spike benchmark, Phase 1: device-resident collection throughput.

Measures env-steps/sec for a fully on-device collection loop using the
batched DeviceVecCartPole — no CPU env-loop, no per-step host sync.

Comparison point: scripts/bench_vec_collection.py measured the gym-based
vectorized path at ~10k env-steps/s (TPU) / ~18k (GPU) at N=256,
bottlenecked ~96% by the serial env-loop. This benchmark removes that
loop entirely — the question is how high throughput goes once the whole
collection is accelerator-resident.

`sync/K` = env-steps unrolled into one XLA graph per mark_step. K>1
amortises dispatch overhead — the canonical way to keep a TPU fed.

Usage:  python -m scripts.bench_device_env
"""

import time
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.environments.device_env import DeviceVecCartPole
from ragnarok.core.agent import RagnarokAgent
from ragnarok.infrastructure.device import DEVICE, mark_step

N_VALUES = [256, 1024, 4096]
SYNC_EVERY = [1, 20]   # env-steps unrolled per mark_step
WARMUP = 4
TIMED = 40             # timed sync-points (each = K env-steps)


def build_policy():
    """The real_trainer's discrete CartPole policy (DirectPolicyNet)."""
    spec = get_env_spec("cartpole")
    config = RagnarokConfig()
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim
    env0 = RagnarokEnv(spec.gym_name, seed=0)
    agent = RagnarokAgent(config, env0)
    return agent.real_trainer.policy


def bench(policy, n: int, k: int) -> float:
    env = DeviceVecCartPole(n)
    obs = env.reset()

    def block():
        nonlocal obs
        for _ in range(k):
            logits, _ = policy(obs)
            action = torch.distributions.Categorical(logits=logits).sample()
            obs, _, _, _, _ = env.step(action)
        mark_step()

    with torch.no_grad():
        for _ in range(WARMUP):
            block()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            block()
        wall = time.perf_counter() - t0
    return n * k * TIMED / wall


def main():
    print(f"[bench-device] device={DEVICE}")
    policy = build_policy()
    print(f"\n{'N':>6} | {'sync/K':>7} | {'env-steps/sec':>15}")
    print("-" * 38)
    for n in N_VALUES:
        for k in SYNC_EVERY:
            eps = bench(policy, n, k)
            print(f"{n:>6} | {k:>7} | {eps:>15,.0f}")
    print("\n[bench-device] done")


if __name__ == "__main__":
    main()
