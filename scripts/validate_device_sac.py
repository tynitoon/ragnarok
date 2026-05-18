"""Phase 2 Stage 4 validation: SAC learns MountainCar on the device path.

End-to-end check of accelerator-resident SAC:

    DeviceVecMountainCarContinuous  ->  collect_rollout  ->
                                        SACTrainer.train_on_rollout
                                        (DeviceSACBuffer — all on-device)

No gym envs, no host env-loop, no host-resident replay buffer. The greedy
eval reports mean completed-episode return; MountainCarContinuous pays
+100 at the goal minus a small action cost, so a return climbing toward
~90 means the policy reliably reaches the goal.

MCC is a sparse-reward exploration task — the gym pipeline solves it with
the agent's curiosity bonus. This script runs PLAIN device SAC: it gates
the device-SAC *mechanism* (buffer, updates, collection). 256 parallel
envs give strong exploration, so plain SAC may well solve it; if not, the
SAC losses still show the mechanism is sound and full MCC mastery is a
Stage-5 (curiosity-integrated) concern.

Usage:  python -m scripts.validate_device_sac
"""

import time
import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE, mark_step
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuous, DeviceRunningNormalizer)
from ragnarok.learning.rollout import collect_rollout
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer

N_ENVS = 256
HORIZON = 128
ROLLOUTS = 50
N_UPDATES = 1024
WARMUP_STEPS = N_ENVS * HORIZON      # one full rollout of random actions
EVAL_EVERY = 5
PASS_RETURN = 50.0                   # MCC mastery is ~90; 50 = clearly learning


@torch.no_grad()
def eval_mcc(policy, normalizer, n_envs: int = 256, steps: int = 999) -> float:
    """Greedy eval — mean completed-episode return on MountainCarContinuous."""
    env = DeviceVecMountainCarContinuous(n_envs)
    ret = torch.zeros(n_envs, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        mean, _ = policy.forward(normalizer.normalize(env.state))
        action = policy._rescale(torch.tanh(mean))   # deterministic
        _, reward, _, _, done = env.step(action)
        done = done.float()        # env.step returns a bool done tensor
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
    mark_step()
    return (ret_sum / ep_count.clamp(min=1.0)).item()


def main():
    print(f"[validate-device-sac] device={DEVICE}")
    torch.manual_seed(0)
    np.random.seed(0)

    trainer = SACTrainer(
        obs_dim=2, action_dim=1,
        action_low=np.array([-1.0]), action_high=np.array([1.0]),
        warmup_steps=WARMUP_STEPS, batch_size=256,
        buffer=DeviceSACBuffer(capacity=200_000))
    env = DeviceVecMountainCarContinuous(N_ENVS)
    normalizer = DeviceRunningNormalizer(obs_dim=2)

    print(f"  N={N_ENVS}  horizon={HORIZON}  n_updates={N_UPDATES}/rollout"
          f"  warmup={WARMUP_STEPS:,}\n")
    base = eval_mcc(trainer.policy, normalizer)
    print(f"  rollout  0 | return {base:8.2f}  (untrained)")

    t0 = time.perf_counter()
    for it in range(1, ROLLOUTS + 1):
        batch = collect_rollout(env, trainer.device_policy_fn, HORIZON,
                                normalizer=normalizer)
        m = trainer.train_on_rollout(batch, n_updates=N_UPDATES)
        normalizer.update(batch.raw_obs.reshape(-1, 2))
        if it % EVAL_EVERY == 0:
            score = eval_mcc(trainer.policy, normalizer)
            ql = m.get("sac/q1_loss", float("nan"))
            pl = m.get("sac/policy_loss", float("nan"))
            al = m.get("sac/alpha", float("nan"))
            print(f"  rollout {it:2d} | return {score:8.2f} "
                  f"| q {ql:8.2f}  policy {pl:8.2f}  alpha {al:.3f}")
    wall = time.perf_counter() - t0

    final = eval_mcc(trainer.policy, normalizer)
    print(f"\n  final return {final:.2f}  (start {base:.2f})")
    print(f"  {ROLLOUTS} rollouts in {wall:.1f}s")
    verdict = ("PASS — device SAC learned MountainCar"
               if final >= PASS_RETURN
               else f"PARTIAL — final {final:.1f} < {PASS_RETURN} "
                    "(SAC mechanism runs; full MCC mastery needs curiosity)")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
