"""Phase 2 Stage 5 validation: a full DeviceAgent trains on the device path.

validate_device_wm checks the world model in isolation, on random-policy
rollouts (a stationary data distribution). This checks the *full*
DeviceAgent: the real policy (PPO for the discrete CartPole, SAC for the
continuous MountainCar), the RSSM world model and the latent policy all
training together on the same rollout — and crucially the world model
now trains on NON-stationary data, since the real policy that collects
the rollouts keeps improving.

  --env cartpole : PPO + world model + latent policy.
  --env mcc      : SAC + world model + latent policy + curiosity (the
                   continuous path; MountainCar's reward is too sparse
                   for a short run to master, so the bar is "runs clean,
                   world model stable").

Pass criteria:
  - the world model stays stable — KL never runs away (the lr 3e-5 fix
    holds under non-stationary data, not just the stationary random-policy
    data validate_device_wm tested);
  - the agent is healthy — CartPole greedy eval climbs well above the
    ~20-40 a random policy scores; MCC just has to run NaN-free.

Usage:  python -m scripts.validate_device_agent [--env cartpole|mcc] [--iters N]
"""

import argparse
import math
import time

import torch

from ragnarok.infrastructure.device import DEVICE, IS_XLA
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)

ENVS = {"cartpole": DeviceVecCartPole, "mcc": DeviceVecMountainCarContinuous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["cartpole", "mcc"], default="cartpole")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--sac-updates", type=int, default=256,
                        help="SAC updates per iteration (continuous env only).")
    parser.add_argument("--xla-precision", choices=["default", "high", "highest"],
                        default="default",
                        help="TPU MXU fp32-matmul precision (XLA only).")
    args = parser.parse_args()

    if IS_XLA:
        import torch_xla.backends
        torch_xla.backends.set_mat_mul_precision(args.xla_precision)

    env_cls = ENVS[args.env]
    num_envs = 128 if args.env == "cartpole" else 256
    print(f"[validate-device-agent] device={DEVICE}  env={args.env}  "
          f"({env_cls.__name__})")
    torch.manual_seed(0)

    agent = DeviceAgent(env_cls, num_envs=num_envs, horizon=128,
                        sac_updates=args.sac_updates)

    print(f"\n{'iter':>5} | {'wm_kl':>8} | {'wm_total':>9} | "
          f"{'wm_grad':>9} | {'eval':>9}")
    print("-" * 54)

    t0 = time.perf_counter()
    first_eval = last_eval = None
    kl_max = last_kl = 0.0
    for i in range(1, args.iters + 1):
        m = agent.train_iteration()
        last_kl = float(m.get("wm/kl_loss", 0.0))
        kl_max = max(kl_max, last_kl)
        if i == 1 or i % 5 == 0:
            ev = agent.evaluate(steps=500)
            first_eval = ev if first_eval is None else first_eval
            last_eval = ev
            print(f"{i:>5} | {m.get('wm/kl_loss', 0.0):>8.3f} | "
                  f"{m.get('wm/total_loss', 0.0):>9.3f} | "
                  f"{m.get('wm/grad_norm', 0.0):>9.3f} | {ev:>9.2f}")
    wall = time.perf_counter() - t0

    print(f"\n  {args.iters} iters in {wall:.1f}s  |  "
          f"eval {first_eval:.2f} -> {last_eval:.2f}")
    print(f"  world-model KL: final {last_kl:.3f}, max {kl_max:.3f}")
    # WM stable: KL did not run away. The TPU divergence is a monotonic
    # runaway (KL -> 10-70), so the *final* KL is the tell — a transient
    # spike that recovers (common early in curiosity-driven MCC
    # exploration, when the agent first reaches novel states) is not it.
    wm_stable = last_kl < 5.0
    if args.env == "cartpole":
        agent_ok = last_eval > 200.0          # random policy ~20-40
        crit = "eval > 200"
    else:
        agent_ok = math.isfinite(last_eval)   # sparse reward: just NaN-free
        crit = "eval finite (NaN-free)"
    verdict = ("PASS — full device agent trains on the device path"
               if (wm_stable and agent_ok) else
               "FAIL — see wm_stable / agent_ok below")
    print(f"  wm_stable (final KL<5): {wm_stable}   agent_ok ({crit}): {agent_ok}")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
