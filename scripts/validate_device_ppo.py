"""Phase 2 Stage 2 validation: PPO learns CartPole on the device path.

End-to-end check of the accelerator-resident training path:

    DeviceVecCartPole  ->  collect_rollout  ->  compute_gae_batched
                       ->  RealExperienceTrainer.train_on_rollout

No gym envs, no host env-loop, no per-step host sync — collection and the
PPO update both run batched on the device. If CartPole's mean episode
length climbs to ~500 (the +1/step reward makes length == score), the
device PPO path is correct and Stage 2's core is validated.

Usage:  python -m scripts.validate_device_ppo
"""

import time
import torch

from ragnarok.infrastructure.device import DEVICE, mark_step
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceRunningNormalizer)
from ragnarok.learning.rollout import collect_rollout
from ragnarok.learning.real_experience import RealExperienceTrainer

N_ENVS = 128
HORIZON = 128
ROLLOUTS = 60
EVAL_EVERY = 5
PASS_THRESHOLD = 450.0   # mean episode length; CartPole-v1 caps at 500


@torch.no_grad()
def eval_mean_ep_len(policy, normalizer, n_envs: int = 256,
                     steps: int = 500) -> float:
    """Greedy eval: mean episode length for CartPole (== mean return).

    Runs `steps` greedy steps across `n_envs` auto-resetting envs and
    counts episode boundaries. mean_ep_len = total transitions / #episodes
    — ~500 when solved, ~20 for a random policy.
    """
    env = DeviceVecCartPole(n_envs)
    n_done = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        logits, _ = policy(normalizer.normalize(env.state))
        action = logits.argmax(dim=-1)
        _, _, _, _, done = env.step(action)
        n_done = n_done + done.float().sum()
    mark_step()
    return n_envs * steps / max(n_done.item(), 1.0)


def main():
    print(f"[validate-device-ppo] device={DEVICE}")
    torch.manual_seed(0)

    trainer = RealExperienceTrainer(obs_dim=4, action_dim=2, discrete=True)
    env = DeviceVecCartPole(N_ENVS)
    normalizer = DeviceRunningNormalizer(obs_dim=4)

    print(f"  N={N_ENVS}  horizon={HORIZON}  "
          f"({N_ENVS * HORIZON:,} transitions/rollout)\n")
    base = eval_mean_ep_len(trainer.policy, normalizer)
    print(f"  rollout  0 | mean_ep_len {base:6.1f}  (untrained)")

    t0 = time.perf_counter()
    for it in range(1, ROLLOUTS + 1):
        batch = collect_rollout(env, trainer.device_policy_fn, HORIZON,
                                normalizer=normalizer)
        trainer.train_on_rollout(batch)
        normalizer.update(batch.raw_obs.reshape(-1, 4))
        if it % EVAL_EVERY == 0:
            score = eval_mean_ep_len(trainer.policy, normalizer)
            print(f"  rollout {it:2d} | mean_ep_len {score:6.1f} "
                  f"| {it * N_ENVS * HORIZON:,} env-steps")
    wall = time.perf_counter() - t0

    final = eval_mean_ep_len(trainer.policy, normalizer)
    total = ROLLOUTS * N_ENVS * HORIZON
    print(f"\n  final mean_ep_len {final:.1f}  (start {base:.1f})")
    print(f"  {total:,} env-steps in {wall:.1f}s "
          f"| {total / wall:,.0f} steps/s (collect+train)")
    verdict = ("PASS — device PPO learned CartPole" if final >= PASS_THRESHOLD
               else f"FAIL — final {final:.1f} < {PASS_THRESHOLD}")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
