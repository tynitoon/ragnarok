"""Phase 2 Stage 5 validation: a full DeviceAgent trains on the device path.

validate_device_wm checks the world model in isolation, on random-policy
rollouts (a stationary data distribution). This checks the *full*
DeviceAgent: the real policy (PPO), the RSSM world model and the latent
policy all training together on the same rollout — and crucially the
world model now trains on NON-stationary data, since the PPO policy that
collects the rollouts keeps improving.

Pass criteria:
  - the world model stays stable — KL never runs away (the lr 3e-5 fix
    holds under non-stationary data, not just the stationary random-policy
    data validate_device_wm tested);
  - the agent learns — greedy CartPole eval return climbs well above the
    ~20-40 a random policy scores.

Usage:  python -m scripts.validate_device_agent [--iters N]
"""

import argparse
import time

import torch

from ragnarok.infrastructure.device import DEVICE, IS_XLA
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import DeviceVecCartPole


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--xla-precision", choices=["default", "high", "highest"],
                        default="default",
                        help="TPU MXU fp32-matmul precision (XLA only).")
    args = parser.parse_args()

    if IS_XLA:
        import torch_xla.backends
        torch_xla.backends.set_mat_mul_precision(args.xla_precision)

    print(f"[validate-device-agent] device={DEVICE}  CartPole DeviceAgent")
    torch.manual_seed(0)

    agent = DeviceAgent(DeviceVecCartPole, num_envs=128, horizon=128)

    print(f"\n{'iter':>5} | {'wm_kl':>8} | {'wm_total':>9} | "
          f"{'wm_grad':>9} | {'eval':>7}")
    print("-" * 52)

    t0 = time.perf_counter()
    first_eval = last_eval = None
    kl_max = 0.0
    for i in range(1, args.iters + 1):
        m = agent.train_iteration()
        kl_max = max(kl_max, float(m.get("wm/kl_loss", 0.0)))
        if i == 1 or i % 5 == 0:
            ev = agent.evaluate(steps=500)
            first_eval = ev if first_eval is None else first_eval
            last_eval = ev
            print(f"{i:>5} | {m.get('wm/kl_loss', 0.0):>8.3f} | "
                  f"{m.get('wm/total_loss', 0.0):>9.3f} | "
                  f"{m.get('wm/grad_norm', 0.0):>9.3f} | {ev:>7.1f}")
    wall = time.perf_counter() - t0

    print(f"\n  {args.iters} iters in {wall:.1f}s  |  "
          f"eval {first_eval:.1f} -> {last_eval:.1f}")
    print(f"  world-model KL max over training: {kl_max:.3f}")
    # WM stable: KL never ran away (the divergence drove it to 10-70).
    # Learned: CartPole eval climbed well above the random-policy ~20-40.
    wm_stable = kl_max < 5.0
    learned = last_eval > 200.0
    verdict = ("PASS — full device agent trains on the device path"
               if (wm_stable and learned) else
               "FAIL — see wm_stable / learned below")
    print(f"  wm_stable (KL<5): {wm_stable}   learned (eval>200): {learned}")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
