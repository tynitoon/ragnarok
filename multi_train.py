"""Ragnarok Phase 4: Autonomous multi-task sequential training.

Two-phase approach:
  Phase A: Train all environments from scratch, building the skill library.
  Phase B: Re-train all environments WITH transfer from saved skills.
           Compare episodes-to-threshold to prove transfer acceleration.

Curriculum: CartPole -> MountainCar -> Acrobot -> Pendulum -> MountainCarContinuous

Usage:
    python multi_train.py                # full two-phase comparison
    python multi_train.py --phase A      # only build skill library
    python multi_train.py --phase B      # only transfer test (requires existing skills)
    python multi_train.py --seed 123
"""

import argparse
import sys
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


# Curriculum order: easy discrete -> hard discrete -> continuous
CURRICULUM = [
    ("cartpole", 500),
    ("mountaincar", 800),
    ("acrobot", 500),
    ("pendulum", 500),
    ("mountaincar-continuous", 500),
]


_builtin_print = print


def fprint(*args, **kwargs):
    """Print with immediate flush for background process visibility."""
    kwargs.setdefault('flush', True)
    _builtin_print(*args, **kwargs)


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

    transfer_str = f"<- {transferred.name}" if transferred else "from scratch"
    fprint(f"\n{'='*60}")
    fprint(f"  {spec.gym_name} ({algo}) | {transfer_str}")
    fprint(f"{'='*60}")

    # Training
    start_time = time.time()
    crystallized_at = None
    threshold_at = None  # First episode where eval >= threshold
    best_eval = -float("inf")
    threshold = spec.reward_threshold

    is_pixel = spec.pixel_obs
    report_interval = 20 if is_pixel else 50

    for iteration in range(1, max_episodes + 1):
        ep_reward, metrics = agent.train_policy_real()

        # Train world model periodically (vector mode only, lighter than train.py)
        if (not is_pixel and
                iteration % 25 == 0 and
                agent.replay_buffer.num_episodes >= 10):
            agent.train_world_model(steps=30)

        # Dream augmentation (vector mode only, skip SAC)
        if (not is_pixel and
                agent.sac_trainer is None and
                iteration % 25 == 0 and
                iteration >= 100 and
                agent.replay_buffer.num_episodes >= 20):
            agent.train_policy_dream(steps=15)

        # Check crystallization periodically (runs eval internally, so not every ep)
        skill = None
        if iteration % 10 == 0:
            skill = agent.check_crystallization()
        if skill and crystallized_at is None:
            crystallized_at = agent.total_episodes
            fprint(f"  * CRYSTALLIZED at ep {crystallized_at} "
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

            # Track threshold reaching (independently of crystallization)
            if threshold_at is None and eval_mean >= threshold:
                threshold_at = agent.total_episodes
                fprint(f"  * THRESHOLD REACHED at ep {threshold_at} "
                      f"(eval: {eval_mean:.1f} >= {threshold:.1f})")

            elapsed = time.time() - start_time
            eps = agent.total_episodes / elapsed if elapsed > 0 else 0
            label = f"Iter {iteration:4d}" if is_pixel else f"Ep {agent.total_episodes:4d}"
            fprint(f"  [{label}] eval: {eval_mean:7.1f} | "
                  f"best: {best_eval:7.1f} | "
                  f"steps: {agent.total_steps:6d} | {eps:.1f} ep/s")

        # Early stop only on crystallization (robust signal, requires sustained performance)
        if crystallized_at and iteration > (crystallized_at + 50):
            break

    elapsed = time.time() - start_time

    # Final eval (share normalizer so obs distribution matches training)
    env2 = RagnarokEnv(spec.gym_name, seed=seed + 1000, pixel_obs=spec.pixel_obs,
                       normalizer=env.normalizer, normalize=env.normalize)
    if is_pixel:
        final_eval = agent.pixel_ppo.evaluate(env2, episodes=10)
    elif agent.sac_trainer:
        final_eval = agent.sac_trainer.evaluate(env2, episodes=10)
    else:
        final_eval = agent.real_trainer.evaluate(env2, episodes=10)
    env2.close()
    env.close()

    result = {
        "env": spec.gym_name,
        "algo": algo,
        "transfer_from": transferred.name if transferred else None,
        "crystallized_at": crystallized_at,
        "threshold_at": threshold_at,
        "total_episodes": agent.total_episodes,
        "total_steps": agent.total_steps,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "elapsed_sec": elapsed,
        "num_skills": agent.skill_library.num_skills,
    }

    status = f"threshold ep {threshold_at}" if threshold_at else "not reached"
    fprint(f"  -> {status} | final eval: {final_eval:.1f} | "
          f"{elapsed:.0f}s | skills: {agent.skill_library.num_skills}")

    return result


def print_results(results: list[dict], label: str):
    """Print results table."""
    fprint(f"\n{'='*70}")
    fprint(f"  {label}")
    fprint(f"{'='*70}")
    fprint(f"{'Env':<25} {'Algo':<5} {'Threshold':>10} {'Eval':>8} {'Transfer'}")
    fprint(f"{'-'*70}")
    for r in results:
        th = f"ep {r['threshold_at']}" if r['threshold_at'] else "-"
        xfer = r['transfer_from'] or "-"
        fprint(f"{r['env']:<25} {r['algo']:<5} {th:>10} "
              f"{r['final_eval']:>8.1f} {xfer}")
    fprint(f"{'-'*70}")


def main():
    parser = argparse.ArgumentParser(description="Multi-task sequential training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=str, default="AB", choices=["A", "B", "AB"],
                        help="A=build skills, B=test transfer, AB=both")
    parser.add_argument("--clean", action="store_true",
                        help="Clear skill library before starting")
    args = parser.parse_args()

    skills_dir = "skills_data"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Clean start
    if args.clean:
        p = Path(skills_dir)
        if p.exists():
            shutil.rmtree(p)
            fprint(f"[Ragnarok] Cleared {skills_dir}/")
    Path(skills_dir).mkdir(exist_ok=True)

    fprint(f"[Ragnarok] Multi-task training")
    fprint(f"[Ragnarok] Device: {DEVICE}")
    fprint(f"[Ragnarok] Curriculum: {' -> '.join(name for name, _ in CURRICULUM)}")
    fprint(f"[Ragnarok] Skills dir: {skills_dir}")

    scratch_results = []
    transfer_results = []
    total_start = time.time()

    # === PHASE A: Train from scratch, build skill library ===
    if "A" in args.phase:
        fprint(f"\n{'#'*60}")
        fprint(f"  PHASE A: Training from scratch (building skill library)")
        fprint(f"{'#'*60}")

        for env_name, max_eps in CURRICULUM:
            result = train_single_env(
                env_name, max_eps, args.seed,
                transfer=False, skills_dir=skills_dir,
            )
            scratch_results.append(result)

        print_results(scratch_results, "PHASE A RESULTS (from scratch)")

    # === PHASE B: Re-train with transfer from saved skills ===
    if "B" in args.phase:
        fprint(f"\n{'#'*60}")
        fprint(f"  PHASE B: Re-training WITH skill transfer")
        fprint(f"{'#'*60}")

        for env_name, max_eps in CURRICULUM:
            result = train_single_env(
                env_name, max_eps, args.seed,
                transfer=True, skills_dir=skills_dir,
            )
            transfer_results.append(result)

        print_results(transfer_results, "PHASE B RESULTS (with transfer)")

    total_elapsed = time.time() - total_start

    # === COMPARISON ===
    if scratch_results and transfer_results:
        fprint(f"\n{'='*70}")
        fprint(f"  TRANSFER ACCELERATION COMPARISON")
        fprint(f"{'='*70}")
        fprint(f"{'Env':<25} {'Scratch':>10} {'Transfer':>10} {'Speedup':>10}")
        fprint(f"{'-'*70}")
        for s, t in zip(scratch_results, transfer_results):
            s_ep = s['threshold_at'] or s['total_episodes']
            t_ep = t['threshold_at'] or t['total_episodes']
            if s_ep > 0 and t_ep > 0 and t_ep < s_ep:
                speedup = f"{s_ep / t_ep:.1f}x"
            else:
                speedup = "-"
            fprint(f"{s['env']:<25} {s_ep:>8} ep {t_ep:>8} ep {speedup:>10}")
        fprint(f"{'-'*70}")
        fprint(f"Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
