"""Phase 2 Stage 3 validation: the RSSM world model learns on the device path.

End-to-end check of accelerator-resident world-model training:

    DeviceVecCartPole  ->  collect_rollout  ->
                           WorldModelTrainer.train_world_model_on_rollout

No gym envs, no host ReplayBuffer, no host->device transfer of training
batches — collection and the RSSM update both run batched on the device.
If reconstruction and reward losses fall and plateau low, the device
world-model path is correct and Stage 3's core is validated.

Usage:  python -m scripts.validate_device_wm
"""

import argparse
import time
import torch

from ragnarok.infrastructure.device import DEVICE, IS_XLA
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceRunningNormalizer)
from ragnarok.learning.rollout import collect_rollout
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer

N_ENVS = 256
HORIZON = 128
ROLLOUTS = 30
REPORT_EVERY = 5


def random_policy_fn(obs):
    """Uniform-random discrete policy — wide state coverage for the WM."""
    n = obs.shape[0]
    dist = torch.distributions.Categorical(
        logits=torch.zeros(n, 2, device=obs.device))
    action = dist.sample()
    return action, dist.log_prob(action), torch.zeros(n, device=obs.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xla-precision", choices=["default", "high", "highest"],
        default="highest",
        help="TPU MXU fp32-matmul precision (XLA only; no-op on CUDA/CPU). "
             "'default' is a single bf16 pass — the TPU default; its "
             "rounding error compounds across the RSSM's 128-step GRU "
             "unroll and the world model diverges. 'highest' is a six-pass "
             "~fp32-faithful matmul that matches the CUDA path the agent "
             "was calibrated on. Pass 'default' to reproduce the historical "
             "TPU divergence.")
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="rollout length T — also the RSSM world-model "
                             "training unroll depth. The device path unrolls "
                             "the full 128-step row; the calibrated gym WM "
                             "trained on 50-step subsequences. Shorten this "
                             "to test whether unroll depth triggers the TPU "
                             "divergence.")
    args = parser.parse_args()

    if IS_XLA:
        import torch_xla.backends
        torch_xla.backends.set_mat_mul_precision(args.xla_precision)

    print(f"[validate-device-wm] device={DEVICE}  "
          f"xla_matmul_precision={args.xla_precision if IS_XLA else 'n/a'}")
    torch.manual_seed(0)

    rssm = RSSM(obs_dim=4, action_dim=2).to(DEVICE)
    wm = WorldModelTrainer(rssm, ReplayBuffer())
    env = DeviceVecCartPole(N_ENVS)
    normalizer = DeviceRunningNormalizer(obs_dim=4)

    print(f"  N={N_ENVS}  horizon={args.horizon}  "
          f"({N_ENVS * args.horizon:,} transitions/rollout)\n")
    print(f"{'rollout':>8} | {'recon':>9} | {'reward':>9} | "
          f"{'continue':>9} | {'kl':>9} | {'total':>9}")
    print("-" * 64)

    first = last = None
    t0 = time.perf_counter()
    for it in range(1, args.rollouts + 1):
        batch = collect_rollout(env, random_policy_fn, args.horizon,
                                normalizer=normalizer)
        m = wm.train_world_model_on_rollout(batch)
        normalizer.update(batch.raw_obs.reshape(-1, 4))
        first = first or m
        last = m
        if it == 1 or it % REPORT_EVERY == 0:
            print(f"{it:>8} | {m['wm/recon_loss']:>9.4f} | "
                  f"{m['wm/reward_loss']:>9.4f} | {m['wm/continue_loss']:>9.4f} | "
                  f"{m['wm/kl_loss']:>9.4f} | {m['wm/total_loss']:>9.4f}")
    wall = time.perf_counter() - t0

    print(f"\n  recon  {first['wm/recon_loss']:.4f} -> {last['wm/recon_loss']:.4f}")
    print(f"  reward {first['wm/reward_loss']:.4f} -> {last['wm/reward_loss']:.4f}")
    print(f"  total  {first['wm/total_loss']:.4f} -> {last['wm/total_loss']:.4f}")
    print(f"  {args.rollouts} rollouts in {wall:.1f}s")
    # CartPole pays a constant +1 reward — the reward head should nail it;
    # recon must fall well below its starting value.
    learned = (last["wm/reward_loss"] < 0.1
               and last["wm/recon_loss"] < first["wm/recon_loss"]
               and last["wm/total_loss"] < first["wm/total_loss"])
    verdict = ("PASS — device world model learned"
               if learned else "FAIL — losses did not converge")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
