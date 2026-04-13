"""Ragnarok Phase 4: Autonomous multi-task sequential training.

Trains the agent across multiple environments in sequence, measuring
how skill transfer accelerates learning on each new task.

Curriculum: CartPole → MountainCar → Acrobot → Pendulum → MountainCarContinuous

Usage:
    python multi_train.py
    python multi_train.py --no-transfer   # baseline without skill transfer
    python multi_train.py --seed 123
"""

import argparse
import time
import shutil
from pathlib import Path

import torch
import numpy as np

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent


# Curriculum order: easy discrete → hard discrete → continuous
CURRICULUM = [
    ("cartpole", 500),
    ("mountaincar", 1000),
    ("acrobot", 500),
    ("pendulum", 500),
    ("mountaincar-continuous", 500),
]


def train_single_env(env_name: str, max_episodes: int, seed: int,
                     transfer: bool, skills_dir: str) -> dict:
    """Train one environment, return results."""
    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed, checkpoint_dir="checkpoints")
    config.skill.skills_dir = skills_dir
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed, pixel_obs=spec.pixel_obs)
    agent = RagnarokAgent(config, env)

    # Determine algorithm
    if agent.pixel_ppo is not None:
        algo = "PPO"
    elif agent.sac_trainer:
        algo = "SAC"
    else:
        algo = "A2C"

    # Try skill transfer
    transferred = None
    if transfer:
        transferred = agent.try_transfer()

    transfer_str = f"← {transferred.name}" if transferred else "from scratch"
    print(f"\n{'='*60}")
    print(f"  {spec.gym_name} ({algo}) | {transfer_str}")
    print(f"{'='*60}")

    # Training
    start_time = time.time()
    crystallized_at = None
    best_eval = -float("inf")

    is_pixel = spec.pixel_obs
    report_interval = 20 if is_pixel else 50

    for iteration in range(1, max_episodes + 1):
        ep_reward, metrics = agent.train_policy_real()

        # Check crystallization
        skill = agent.check_crystallization()
        if skill and crystallized_at is None:
            crystallized_at = agent.total_episodes
            print(f"  ✓ CRYSTALLIZED at ep {crystallized_at} "
                  f"(reward: {skill.performance:.1f})")

        # Progress report
        if iteration % report_interval == 0:
            if is_pixel:
                eval_mean = agent._evaluate_pixel(episodes=5)
            elif agent.sac_trainer:
                eval_mean = agent.sac_trainer.evaluate(env, episodes=5)
            else:
                eval_mean = agent.real_trainer.evaluate(env, episodes=5)
            best_eval = max(best_eval, eval_mean)

            elapsed = time.time() - start_time
            eps = agent.total_episodes / elapsed if elapsed > 0 else 0
            label = f"Iter {iteration:4d}" if is_pixel else f"Ep {agent.total_episodes:4d}"
            print(f"  [{label}] eval: {eval_mean:7.1f} | "
                  f"best: {best_eval:7.1f} | "
                  f"steps: {agent.total_steps:6d} | {eps:.1f} ep/s")

        # Early stop if crystallized and well past threshold
        if crystallized_at and iteration > crystallized_at + 50:
            break

    elapsed = time.time() - start_time
    env.close()

    # Final eval
    env2 = RagnarokEnv(spec.gym_name, seed=seed + 1000, pixel_obs=spec.pixel_obs)
    if is_pixel:
        final_eval = agent.pixel_ppo.evaluate(env2, episodes=10)
    elif agent.sac_trainer:
        final_eval = agent.sac_trainer.evaluate(env2, episodes=10)
    else:
        final_eval = agent.real_trainer.evaluate(env2, episodes=10)
    env2.close()

    result = {
        "env": spec.gym_name,
        "algo": algo,
        "transfer_from": transferred.name if transferred else None,
        "crystallized_at": crystallized_at,
        "total_episodes": agent.total_episodes,
        "total_steps": agent.total_steps,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "elapsed_sec": elapsed,
        "num_skills": agent.skill_library.num_skills,
    }

    status = f"crystallized ep {crystallized_at}" if crystallized_at else "not crystallized"
    print(f"  → {status} | final eval: {final_eval:.1f} | "
          f"{elapsed:.0f}s | skills: {agent.skill_library.num_skills}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-task sequential training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-transfer", action="store_true",
                        help="Disable skill transfer (baseline)")
    parser.add_argument("--clean", action="store_true",
                        help="Clear skill library before starting")
    args = parser.parse_args()

    transfer = not args.no_transfer
    skills_dir = "skills_data" if transfer else "skills_data_baseline"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Clean start
    if args.clean:
        p = Path(skills_dir)
        if p.exists():
            shutil.rmtree(p)
            print(f"[Ragnarok] Cleared {skills_dir}/")
    Path(skills_dir).mkdir(exist_ok=True)

    mode = "WITH transfer" if transfer else "WITHOUT transfer (baseline)"
    print(f"[Ragnarok] Multi-task training — {mode}")
    print(f"[Ragnarok] Device: {DEVICE}")
    print(f"[Ragnarok] Curriculum: {' → '.join(name for name, _ in CURRICULUM)}")
    print(f"[Ragnarok] Skills dir: {skills_dir}")

    results = []
    total_start = time.time()

    for env_name, max_eps in CURRICULUM:
        result = train_single_env(
            env_name, max_eps, args.seed,
            transfer=transfer, skills_dir=skills_dir,
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS — {mode}")
    print(f"{'='*60}")
    print(f"{'Env':<28} {'Algo':<5} {'Crystal.':<10} {'Eval':>7} {'Transfer'}")
    print(f"{'-'*60}")
    for r in results:
        crystal = f"ep {r['crystallized_at']}" if r['crystallized_at'] else "—"
        xfer = r['transfer_from'] or "—"
        print(f"{r['env']:<28} {r['algo']:<5} {crystal:<10} "
              f"{r['final_eval']:>7.1f} {xfer}")
    print(f"{'-'*60}")
    print(f"Total time: {total_elapsed:.0f}s | "
          f"Skills: {results[-1]['num_skills']}")


if __name__ == "__main__":
    main()
