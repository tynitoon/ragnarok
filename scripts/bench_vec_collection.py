"""Spike benchmark: batched vectorized-collection throughput.

Measures env-steps/sec for vectorized RL collection at increasing parallel
env counts N, splitting the per-iteration cost into:
  - inference: the batched policy forward on (N, obs_dim) + sample,
  - env-loop:  VecRagnarokEnv.step (a synchronous loop of N gym steps).

This is the spike that gates the "vectorize for the TPU" decision. The
question it answers: does batched collection scale on the TPU (inference
amortizes across N), or does the serial env-loop become the bottleneck?

Run on the TPU and on the CUDA box; compare env-steps/sec.

Usage:  python -m scripts.bench_vec_collection
"""

import time
import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.vec_wrapper import VecRagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent
from ragnarok.infrastructure.device import DEVICE, mark_step

N_VALUES = [1, 16, 64, 256]
WARMUP_ITERS = 8       # first iters pay XLA compile + caches; excluded
TIMED_ITERS = 60       # collection iterations actually timed


def build_policy():
    """Build a discrete CartPole policy (the real_trainer's DirectPolicyNet)."""
    spec = get_env_spec("cartpole")
    config = RagnarokConfig()
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim
    env0 = RagnarokEnv(spec.gym_name, seed=0)
    agent = RagnarokAgent(config, env0)
    return agent.real_trainer.policy, spec


def bench_one(policy, spec, n: int) -> dict:
    """Time WARMUP+TIMED collection iters for n parallel envs."""
    vec = VecRagnarokEnv(spec.gym_name, num_envs=n, seed=100)
    obs = vec.reset()
    a_dim = spec.action_dim

    def collect_iter(obs):
        t0 = time.perf_counter()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            idx = torch.distributions.Categorical(logits=logits).sample()
            idx = idx.cpu().numpy()
        mark_step()
        t1 = time.perf_counter()
        acts = np.zeros((n, a_dim), dtype=np.float32)
        acts[np.arange(n), idx] = 1.0
        obs, _, _, _, _ = vec.step(acts)
        t2 = time.perf_counter()
        return obs, (t1 - t0), (t2 - t1)

    for _ in range(WARMUP_ITERS):
        obs, _, _ = collect_iter(obs)

    t_inf = t_env = 0.0
    wall0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        obs, di, de = collect_iter(obs)
        t_inf += di
        t_env += de
    wall = time.perf_counter() - wall0
    vec.close()

    steps = n * TIMED_ITERS
    return {
        "n": n,
        "env_steps_per_sec": steps / wall,
        "wall": wall,
        "inference_frac": t_inf / wall,
        "env_loop_frac": t_env / wall,
    }


def main():
    print(f"[bench] device={DEVICE}")
    policy, spec = build_policy()
    print(f"[bench] CartPole obs_dim={spec.obs_dim} action_dim={spec.action_dim}")
    print(f"[bench] warmup={WARMUP_ITERS} timed={TIMED_ITERS} iters/N\n")
    print(f"{'N':>5} | {'env-steps/s':>12} | {'wall(s)':>8} | "
          f"{'infer%':>7} | {'envloop%':>9}")
    print("-" * 56)
    base = None
    for n in N_VALUES:
        r = bench_one(policy, spec, n)
        if base is None:
            base = r["env_steps_per_sec"]
        speedup = r["env_steps_per_sec"] / base
        print(f"{r['n']:>5} | {r['env_steps_per_sec']:>12.0f} | "
              f"{r['wall']:>8.2f} | {r['inference_frac']*100:>6.1f}% | "
              f"{r['env_loop_frac']*100:>8.1f}%   (x{speedup:.1f} vs N=1)")
    print("\n[bench] done")


if __name__ == "__main__":
    main()
