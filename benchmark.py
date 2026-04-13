"""Ragnarok Rigorous Benchmark Suite.

Multi-seed benchmark with statistical analysis for measuring
transfer learning acceleration and comparing against SB3 baselines.

Usage:
    python benchmark.py                          # Full benchmark (10 seeds, all envs)
    python benchmark.py --envs cartpole acrobot  # Specific envs
    python benchmark.py --seeds 5                # Fewer seeds (faster)
    python benchmark.py --export-csv results.csv # Export results
    python benchmark.py --quick                  # Quick mode (3 seeds, 200 eps)
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec, REGISTRY
from ragnarok.core.agent import RagnarokAgent


# ── Config ──────────────────────────────────────────────────────────

BENCHMARK_ENVS = [
    ("cartpole", 500),
    ("mountaincar", 400),
    ("acrobot", 400),
    ("pendulum", 300),
]


@dataclass
class BenchmarkResult:
    env_name: str
    gym_name: str
    seed: int
    condition: str  # "scratch", "transfer", "sb3_ppo", "sb3_sac"
    threshold_ep: int | None
    final_eval: float
    total_episodes: int
    elapsed_sec: float
    eval_curve: list = field(default_factory=list)  # [(ep, reward), ...]


# ── Training runners ────────────────────────────────────────────────

def train_ragnarok(env_name: str, max_episodes: int, seed: int,
                   transfer: bool, skills_dir: str = "skills_data",
                   num_envs: int = 1) -> BenchmarkResult:
    """Train Ragnarok agent, return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed)
    config.skill.skills_dir = skills_dir
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)

    # Vectorized env for parallel collection (A2C discrete only)
    vec_env = None
    use_vec = (num_envs > 1 and not spec.pixel_obs
               and agent.sac_trainer is None)
    if use_vec:
        from ragnarok.environments.vec_wrapper import VecRagnarokEnv
        vec_env = VecRagnarokEnv(
            spec.gym_name, num_envs=num_envs, seed=seed,
            normalizer=env.normalizer, normalize=env.normalize,
        )

    if transfer:
        agent.try_transfer()
        if vec_env is not None:
            vec_env.normalizer = env.normalizer
            for v in vec_env.envs:
                v.normalizer = env.normalizer

    threshold = spec.reward_threshold
    eval_curve = []
    threshold_ep = None
    best_eval = -float("inf")
    start = time.time()

    eval_interval = 25
    eps_per_iter = num_envs if use_vec else 1
    max_iters = max(1, max_episodes // eps_per_iter)

    for it in range(1, max_iters + 1):
        if use_vec:
            results = agent.train_policy_real_vec(vec_env)
        else:
            agent.train_policy_real()

        ep = agent.total_episodes
        if ep % eval_interval < eps_per_iter or it == max_iters:
            if agent.sac_trainer:
                eval_r = agent.sac_trainer.evaluate(env, episodes=5)
            else:
                eval_r = agent.real_trainer.evaluate(env, episodes=5)
            eval_curve.append((ep, eval_r))
            best_eval = max(best_eval, eval_r)

            if threshold_ep is None and eval_r >= threshold:
                threshold_ep = ep

            if threshold_ep and ep > threshold_ep + 75:
                break

    elapsed = time.time() - start
    if vec_env is not None:
        vec_env.close()
    env.close()

    condition = "transfer" if transfer else "scratch"
    return BenchmarkResult(
        env_name=env_name,
        gym_name=spec.gym_name,
        seed=seed,
        condition=condition,
        threshold_ep=threshold_ep,
        final_eval=eval_curve[-1][1] if eval_curve else -float("inf"),
        total_episodes=agent.total_episodes,
        elapsed_sec=elapsed,
        eval_curve=eval_curve,
    )


def train_sb3(env_name: str, max_episodes: int, seed: int) -> BenchmarkResult:
    """Train SB3 baseline (PPO for discrete, SAC for continuous)."""
    try:
        from stable_baselines3 import PPO, SAC
        import gymnasium as gym
    except ImportError:
        print("  [SB3 not available, skipping]", flush=True)
        spec = get_env_spec(env_name)
        return BenchmarkResult(
            env_name=env_name, gym_name=spec.gym_name, seed=seed,
            condition="sb3", threshold_ep=None, final_eval=float("nan"),
            total_episodes=0, elapsed_sec=0,
        )

    spec = get_env_spec(env_name)
    env = gym.make(spec.gym_name)

    # Choose algo based on action space
    if spec.is_discrete:
        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        algo_name = "sb3_ppo"
    else:
        model = SAC("MlpPolicy", env, seed=seed, verbose=0)
        algo_name = "sb3_sac"

    # Estimate total timesteps from episodes * avg steps
    avg_steps = {"cartpole": 200, "mountaincar": 200, "acrobot": 200, "pendulum": 200}
    total_timesteps = max_episodes * avg_steps.get(env_name, 200)

    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start

    # Evaluate
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset(seed=seed + 1000)
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward
        eval_rewards.append(total_r)

    final_eval = float(np.mean(eval_rewards))
    threshold_ep = None  # SB3 doesn't track per-episode threshold
    if final_eval >= spec.reward_threshold:
        threshold_ep = max_episodes  # Reached but we don't know when exactly

    env.close()

    return BenchmarkResult(
        env_name=env_name, gym_name=spec.gym_name, seed=seed,
        condition=algo_name, threshold_ep=threshold_ep,
        final_eval=final_eval, total_episodes=max_episodes,
        elapsed_sec=elapsed,
    )


# ── Statistics ──────────────────────────────────────────────────────

def bootstrap_ci(data: list[float], n_resamples: int = 10000,
                 confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(data) < 2:
        return (data[0] if data else float("nan"), data[0] if data else float("nan"))
    data = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    return (float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)))


def wilcoxon_test(scratch: list, transfer: list) -> float:
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    from scipy.stats import wilcoxon
    # Use threshold_ep; if None, use max_episodes as penalty
    if len(scratch) < 3:
        return float("nan")
    try:
        _, p = wilcoxon(scratch, transfer, alternative="greater")
        return float(p)
    except ValueError:
        return float("nan")


# ── Main benchmark ──────────────────────────────────────────────────

def run_benchmark(envs: list[tuple[str, int]], n_seeds: int = 10,
                  skills_dir: str = "skills_data",
                  run_sb3: bool = True,
                  num_envs: int = 1) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    all_results = []

    for env_name, max_eps in envs:
        spec = get_env_spec(env_name)
        print(f"\n{'='*60}", flush=True)
        print(f"  BENCHMARK: {spec.gym_name} ({max_eps} eps, {n_seeds} seeds)", flush=True)
        print(f"{'='*60}", flush=True)

        scratch_results = []
        transfer_results = []

        for seed in range(n_seeds):
            actual_seed = 42 + seed

            # Scratch
            print(f"  [seed {actual_seed}] scratch...", end=" ", flush=True)
            r = train_ragnarok(env_name, max_eps, actual_seed,
                               transfer=False, skills_dir=skills_dir,
                               num_envs=num_envs)
            scratch_results.append(r)
            all_results.append(r)
            th = f"ep {r.threshold_ep}" if r.threshold_ep else "---"
            print(f"eval={r.final_eval:7.1f} threshold={th} ({r.elapsed_sec:.0f}s)", flush=True)

            # Transfer
            print(f"  [seed {actual_seed}] transfer...", end=" ", flush=True)
            r = train_ragnarok(env_name, max_eps, actual_seed,
                               transfer=True, skills_dir=skills_dir,
                               num_envs=num_envs)
            transfer_results.append(r)
            all_results.append(r)
            th = f"ep {r.threshold_ep}" if r.threshold_ep else "---"
            print(f"eval={r.final_eval:7.1f} threshold={th} ({r.elapsed_sec:.0f}s)", flush=True)

        # SB3 baseline (single run, just for final eval comparison)
        if run_sb3:
            print(f"  [SB3 baseline]...", end=" ", flush=True)
            sb3_r = train_sb3(env_name, max_eps, seed=42)
            all_results.append(sb3_r)
            print(f"eval={sb3_r.final_eval:7.1f} ({sb3_r.elapsed_sec:.0f}s)", flush=True)

        # ── Stats ──
        s_thresh = [r.threshold_ep or max_eps for r in scratch_results]
        t_thresh = [r.threshold_ep or max_eps for r in transfer_results]
        s_evals = [r.final_eval for r in scratch_results]
        t_evals = [r.final_eval for r in transfer_results]

        s_mean, s_std = np.mean(s_thresh), np.std(s_thresh)
        t_mean, t_std = np.mean(t_thresh), np.std(t_thresh)
        s_ci = bootstrap_ci(s_thresh)
        t_ci = bootstrap_ci(t_thresh)
        p_value = wilcoxon_test(s_thresh, t_thresh)

        speedups = [s / t if t > 0 else 1.0 for s, t in zip(s_thresh, t_thresh)]
        sp_mean, sp_std = np.mean(speedups), np.std(speedups)
        sp_ci = bootstrap_ci(speedups)

        print(f"\n  --- {spec.gym_name} Results ({n_seeds} seeds) ---", flush=True)
        print(f"  Scratch  threshold: {s_mean:.0f} +/- {s_std:.0f} ep  "
              f"CI95=[{s_ci[0]:.0f}, {s_ci[1]:.0f}]", flush=True)
        print(f"  Transfer threshold: {t_mean:.0f} +/- {t_std:.0f} ep  "
              f"CI95=[{t_ci[0]:.0f}, {t_ci[1]:.0f}]", flush=True)
        print(f"  Speedup: {sp_mean:.2f}x +/- {sp_std:.2f}  "
              f"CI95=[{sp_ci[0]:.2f}, {sp_ci[1]:.2f}]", flush=True)
        print(f"  Wilcoxon p-value: {p_value:.4f} "
              f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})", flush=True)
        print(f"  Scratch  eval: {np.mean(s_evals):.1f} +/- {np.std(s_evals):.1f}", flush=True)
        print(f"  Transfer eval: {np.mean(t_evals):.1f} +/- {np.std(t_evals):.1f}", flush=True)
        if run_sb3:
            print(f"  SB3 baseline eval: {sb3_r.final_eval:.1f}", flush=True)

    return all_results


def export_csv(results: list[BenchmarkResult], path: str):
    """Export results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_name", "gym_name", "seed", "condition",
                     "threshold_ep", "final_eval", "total_episodes", "elapsed_sec"])
        for r in results:
            w.writerow([r.env_name, r.gym_name, r.seed, r.condition,
                        r.threshold_ep or "", f"{r.final_eval:.2f}",
                        r.total_episodes, f"{r.elapsed_sec:.1f}"])
    print(f"\nExported {len(results)} results to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ragnarok Benchmark Suite")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (default: 10)")
    parser.add_argument("--envs", type=str, nargs="+",
                        help="Specific environments (default: all)")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="Export results to CSV file")
    parser.add_argument("--no-sb3", action="store_true",
                        help="Skip SB3 baseline comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, 200 eps")
    parser.add_argument("--skills-dir", type=str, default="skills_data",
                        help="Skills directory for transfer")
    parser.add_argument("--vec", type=int, default=1,
                        help="Number of parallel envs for vectorized collection")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3

    # Filter envs
    if args.envs:
        envs = [(name, eps) for name, eps in BENCHMARK_ENVS if name in args.envs]
    else:
        envs = BENCHMARK_ENVS

    if args.quick:
        envs = [(name, min(eps, 200)) for name, eps in envs]

    print(f"[Benchmark] Seeds: {args.seeds}", flush=True)
    print(f"[Benchmark] Envs: {', '.join(name for name, _ in envs)}", flush=True)
    print(f"[Benchmark] Device: {DEVICE}", flush=True)
    print(f"[Benchmark] Skills: {args.skills_dir}", flush=True)

    results = run_benchmark(
        envs=envs,
        n_seeds=args.seeds,
        skills_dir=args.skills_dir,
        run_sb3=not args.no_sb3,
        num_envs=args.vec,
    )

    if args.export_csv:
        export_csv(results, args.export_csv)

    print(f"\n[Benchmark] Total: {len(results)} runs completed", flush=True)


if __name__ == "__main__":
    main()
