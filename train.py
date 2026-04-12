"""Ragnarok training entry point.

Usage:
    python train.py --env cartpole --episodes 500
    python train.py --env mountaincar --episodes 1000
    python train.py --env cartpole --episodes 500 --transfer
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
          warmup_episodes: int = 5,
          train_every_episodes: int = 1,
          explore_start: float = 1.0,
          explore_end: float = 0.05,
          explore_decay_episodes: int = 200):
    """Main training loop."""

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed, log_dir=log_dir, checkpoint_dir=checkpoint_dir)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)

    print(f"[Ragnarok] Environment: {spec.gym_name}")
    print(f"[Ragnarok] Obs dim: {spec.obs_dim}, Action dim: {spec.action_dim}, Discrete: {spec.is_discrete}")
    print(f"[Ragnarok] Device: {DEVICE}")
    print(f"[Ragnarok] RSSM params: {sum(p.numel() for p in agent.rssm.parameters()):,}")
    print(f"[Ragnarok] Policy params: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")
    print(f"[Ragnarok] Total params: {sum(p.numel() for p in agent.rssm.parameters()) + sum(p.numel() for p in agent.actor_critic.parameters()):,}")

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
            # Exploration schedule (linear decay)
            explore_ratio = max(
                explore_end,
                explore_start - (explore_start - explore_end) * episode / explore_decay_episodes
            )

            # Collect episode
            ep_reward = agent.collect_episode(explore_ratio=explore_ratio)

            # Log
            metrics = {
                "episode_reward": ep_reward,
                "explore_ratio": explore_ratio,
                "total_steps": agent.total_steps,
                "replay_size": len(agent.replay_buffer),
                "episodic_memory_size": agent.episodic_memory.size,
            }

            # Train after warmup
            if episode > warmup_episodes and episode % train_every_episodes == 0:
                # Train world model
                wm_metrics = agent.train_world_model()
                for k, v in wm_metrics.items():
                    metrics[f"wm/{k}"] = v

                # Train policy via dreaming
                dream_metrics = agent.train_policy()
                for k, v in dream_metrics.items():
                    metrics[f"dream/{k}"] = v

            logger.log(episode, metrics)

            # Check crystallization
            skill = agent.check_crystallization()
            if skill:
                print(f"\n[Ragnarok] SKILL CRYSTALLIZED: {skill.name} (reward: {skill.performance:.1f})")

            # Progress
            if ep_reward > best_reward:
                best_reward = ep_reward

            if episode % 10 == 0:
                mean_recent = np.mean(list(agent.episode_rewards)[-50:]) if agent.episode_rewards else 0
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                print(f"[Ep {episode:4d}] reward: {ep_reward:7.1f} | "
                      f"mean50: {mean_recent:7.1f} | best: {best_reward:7.1f} | "
                      f"explore: {explore_ratio:.2f} | "
                      f"steps: {agent.total_steps} | "
                      f"skills: {agent.skill_library.num_skills} | "
                      f"{eps_per_sec:.1f} ep/s")

            # Save checkpoint periodically
            if episode % 100 == 0:
                ckpt_path = f"{checkpoint_dir}/{run_name}_ep{episode}.pt"
                agent.save(ckpt_path)

    except KeyboardInterrupt:
        print("\n[Ragnarok] Training interrupted by user")
    finally:
        # Final save
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
    parser.add_argument("--warmup", type=int, default=5, help="Warmup episodes before training")
    parser.add_argument("--explore-start", type=float, default=1.0, help="Initial exploration ratio")
    parser.add_argument("--explore-end", type=float, default=0.05, help="Final exploration ratio")
    parser.add_argument("--explore-decay", type=int, default=200, help="Episodes to decay exploration")
    args = parser.parse_args()

    train(
        env_name=args.env,
        max_episodes=args.episodes,
        seed=args.seed,
        transfer=not args.no_transfer,
        warmup_episodes=args.warmup,
        explore_start=args.explore_start,
        explore_end=args.explore_end,
        explore_decay_episodes=args.explore_decay,
    )


if __name__ == "__main__":
    main()
