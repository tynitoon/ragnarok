"""Ragnarok training entry point.

Hybrid training approach:
1. Direct A2C on raw observations (provides reliable policy learning)
2. World model (RSSM) training on collected experience
3. Dream training to augment learning once world model is mature
4. Skill crystallization when proficiency is reached

Usage:
    python train.py --env cartpole --episodes 500
    python train.py --env mountaincar --episodes 1000
"""

import argparse
import time
import torch
import numpy as np

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.logger import Logger
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent


def train(env_name: str, max_episodes: int = 500, seed: int = 42,
          transfer: bool = True, log_dir: str = "logs",
          checkpoint_dir: str = "checkpoints",
          wm_train_every: int = 10,
          wm_train_steps: int = 50,
          dream_train_every: int = 20,
          dream_train_steps: int = 20):
    """Main training loop."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed, log_dir=log_dir, checkpoint_dir=checkpoint_dir)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)

    total_params = (sum(p.numel() for p in agent.rssm.parameters()) +
                    sum(p.numel() for p in agent.real_trainer.policy.parameters()))
    print(f"[Ragnarok] Environment: {spec.gym_name}")
    print(f"[Ragnarok] Device: {DEVICE}")
    print(f"[Ragnarok] RSSM: {sum(p.numel() for p in agent.rssm.parameters()):,} params")
    print(f"[Ragnarok] Policy: {sum(p.numel() for p in agent.real_trainer.policy.parameters()):,} params")
    print(f"[Ragnarok] Total: {total_params:,} params")

    # Try skill transfer
    if transfer:
        transferred = agent.try_transfer()
        if transferred:
            print(f"[Ragnarok] Transferred skill: {transferred.name} (perf: {transferred.performance:.1f})")
        else:
            print("[Ragnarok] No matching skill found, starting from scratch")

    run_name = f"{env_name}_{int(time.time())}"
    logger = Logger(log_dir, run_name)

    best_reward = -float("inf")
    start_time = time.time()

    try:
        for episode in range(1, max_episodes + 1):
            # === 1. Collect + train from real experience (A2C) ===
            ep_reward, real_metrics = agent.train_policy_real()

            metrics = {"episode_reward": ep_reward, "total_steps": agent.total_steps}
            metrics.update(real_metrics)

            # === 2. Train world model periodically ===
            if episode % wm_train_every == 0 and agent.replay_buffer.num_episodes >= 10:
                wm_metrics = agent.train_world_model(steps=wm_train_steps)
                for k, v in wm_metrics.items():
                    metrics[f"wm/{k}"] = v

            # === 3. Dream augmentation (train direct policy on imagined data) ===
            if (episode % dream_train_every == 0 and
                    episode >= 50 and
                    agent.replay_buffer.num_episodes >= 20):
                dream_metrics = agent.train_policy_dream(steps=dream_train_steps)
                for k, v in dream_metrics.items():
                    metrics[k] = v

            logger.log(episode, metrics)

            # === 4. Check skill crystallization ===
            skill = agent.check_crystallization()
            if skill:
                print(f"\n[Ragnarok] SKILL CRYSTALLIZED: {skill.name} (reward: {skill.performance:.1f})")

            if ep_reward > best_reward:
                best_reward = ep_reward

            # === 5. Progress report ===
            if episode % 50 == 0:
                eval_mean = agent.real_trainer.evaluate(env, episodes=5)
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                print(f"[Ep {episode:4d}] reward: {ep_reward:7.1f} | "
                      f"eval: {eval_mean:7.1f} | best: {best_reward:7.1f} | "
                      f"steps: {agent.total_steps} | "
                      f"skills: {agent.skill_library.num_skills} | "
                      f"replay: {agent.replay_buffer.num_episodes} ep | "
                      f"{eps_per_sec:.1f} ep/s")

            # Save checkpoint periodically
            if episode % 200 == 0:
                ckpt_path = f"{checkpoint_dir}/{run_name}_ep{episode}.pt"
                agent.save(ckpt_path)

    except KeyboardInterrupt:
        print("\n[Ragnarok] Training interrupted by user")
    finally:
        agent.save(f"{checkpoint_dir}/{run_name}_final.pt")
        logger.close()
        env.close()

    print(f"\n[Ragnarok] Training complete. {agent.total_episodes} episodes, {agent.total_steps} steps")
    print(f"[Ragnarok] Best reward: {best_reward:.1f}")
    print(f"[Ragnarok] Skills: {agent.skill_library.list_skills()}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train Ragnarok agent")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--episodes", type=int, default=500, help="Max episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-transfer", action="store_true", help="Disable skill transfer")
    args = parser.parse_args()

    train(
        env_name=args.env,
        max_episodes=args.episodes,
        seed=args.seed,
        transfer=not args.no_transfer,
    )


if __name__ == "__main__":
    main()
