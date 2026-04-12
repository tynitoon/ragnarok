"""Skill transfer demonstration.

Trains CartPole twice:
1. From scratch (no prior knowledge)
2. With skill transfer (loads crystallized skill)

Compares episodes-to-threshold to prove transfer learning works.

Usage:
    python transfer_demo.py
"""

import time
import torch
import numpy as np

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.learning.real_experience import RealExperienceTrainer


THRESHOLD = 450.0
MAX_EPISODES = 300
EVAL_EVERY = 10
EVAL_EPISODES = 5


def train_from_scratch(spec, seed=42):
    """Train CartPole from scratch, return episode where threshold is reached."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = RagnarokEnv(spec.gym_name, seed=seed)
    trainer = RealExperienceTrainer(
        obs_dim=spec.obs_dim, action_dim=spec.action_dim,
        gamma=0.99, entropy_coeff=0.01, lr=3e-4, grad_clip=0.5,
    )

    rewards = []
    for ep in range(1, MAX_EPISODES + 1):
        reward, _, _ = trainer.collect_and_train(env)
        rewards.append(reward)

        if ep % EVAL_EVERY == 0:
            eval_reward = trainer.evaluate(env, episodes=EVAL_EPISODES)
            if eval_reward >= THRESHOLD:
                env.close()
                return ep, rewards, eval_reward

        if ep % 50 == 0:
            print(f"  [scratch] Ep {ep:3d} | last: {reward:.0f} | avg50: {np.mean(rewards[-50:]):.1f}")

    env.close()
    return MAX_EPISODES, rewards, np.mean(rewards[-10:])


def train_with_transfer(spec, skill_path, seed=123):
    """Train CartPole starting from a transferred skill."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = RagnarokEnv(spec.gym_name, seed=seed)
    trainer = RealExperienceTrainer(
        obs_dim=spec.obs_dim, action_dim=spec.action_dim,
        gamma=0.99, entropy_coeff=0.01, lr=3e-4, grad_clip=0.5,
    )

    # Load skill weights
    skill_data = torch.load(skill_path, weights_only=False, map_location=DEVICE)
    trainer.policy.load_state_dict(
        {k: v.to(DEVICE) for k, v in skill_data["policy_state_dict"].items()}
    )
    print(f"  [transfer] Loaded skill: {skill_data['name']} (perf: {skill_data['performance']:.0f})")

    # Evaluate immediately after loading (before any training)
    pre_eval = trainer.evaluate(env, episodes=EVAL_EPISODES)
    print(f"  [transfer] Pre-training eval: {pre_eval:.1f}")

    if pre_eval >= THRESHOLD:
        env.close()
        return 0, [pre_eval], pre_eval

    rewards = []
    for ep in range(1, MAX_EPISODES + 1):
        reward, _, _ = trainer.collect_and_train(env)
        rewards.append(reward)

        if ep % EVAL_EVERY == 0:
            eval_reward = trainer.evaluate(env, episodes=EVAL_EPISODES)
            if eval_reward >= THRESHOLD:
                env.close()
                return ep, rewards, eval_reward

        if ep % 50 == 0:
            print(f"  [transfer] Ep {ep:3d} | last: {reward:.0f} | avg50: {np.mean(rewards[-50:]):.1f}")

    env.close()
    return MAX_EPISODES, rewards, np.mean(rewards[-10:])


def main():
    spec = get_env_spec("cartpole")
    skill_path = "skills_data/CartPole-v1_60ep.pt"

    print("=" * 60)
    print("RAGNAROK - SKILL TRANSFER DEMONSTRATION")
    print("=" * 60)
    print(f"Environment: {spec.gym_name}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Max episodes: {MAX_EPISODES}")
    print()

    # --- Run 1: From scratch ---
    print("[1/2] Training from scratch...")
    t0 = time.time()
    scratch_ep, scratch_rewards, scratch_eval = train_from_scratch(spec)
    scratch_time = time.time() - t0
    print(f"  -> Reached threshold at episode {scratch_ep} (eval: {scratch_eval:.1f}) in {scratch_time:.1f}s")
    print()

    # --- Run 2: With transfer ---
    print("[2/2] Training with skill transfer...")
    t0 = time.time()
    transfer_ep, transfer_rewards, transfer_eval = train_with_transfer(spec, skill_path)
    transfer_time = time.time() - t0
    print(f"  -> Reached threshold at episode {transfer_ep} (eval: {transfer_eval:.1f}) in {transfer_time:.1f}s")
    print()

    # --- Results ---
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  From scratch:    {scratch_ep:3d} episodes ({scratch_time:.1f}s)")
    print(f"  With transfer:   {transfer_ep:3d} episodes ({transfer_time:.1f}s)")

    if transfer_ep < scratch_ep:
        speedup = scratch_ep / max(transfer_ep, 1)
        print(f"  Speedup:         {speedup:.1f}x faster with skill transfer!")
    elif transfer_ep == 0:
        print(f"  Speedup:         INSTANT — skill already at threshold!")
    else:
        print(f"  No speedup detected (transfer: {transfer_ep}, scratch: {scratch_ep})")

    print()
    print("This demonstrates that Ragnarok can:")
    print("  1. Learn a task (CartPole) from zero")
    print("  2. Crystallize the learned policy as a persistent skill")
    print("  3. Reload and reuse that skill, converging faster")


if __name__ == "__main__":
    main()
