"""Transfer Learning Benchmark.

Measures how much faster the agent learns when starting from a
transferred skill vs from scratch.

Usage:
    python benchmark_transfer.py --env cartpole --episodes 200
    python benchmark_transfer.py --env mountaincar --episodes 500
"""

import argparse
import time
import torch
import numpy as np

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec
from ragnarok.core.agent import RagnarokAgent


def run_training(env_name: str, max_episodes: int, seed: int,
                 transfer: bool) -> dict:
    """Run training and return episode-by-episode rewards."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)

    transferred_skill = None
    if transfer:
        transferred_skill = agent.try_transfer()
        # If RSSM-based transfer didn't find a match (fresh RSSM),
        # try direct env-name matching from the skill library
        if transferred_skill is None:
            from ragnarok.skills.library import SkillLibrary
            library = SkillLibrary()
            for name in library.list_skills():
                skill = library.load_skill(name)
                if skill and skill.env_name == spec.gym_name:
                    try:
                        agent._active_policy.load_state_dict(
                            {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
                        )
                        transferred_skill = skill
                    except RuntimeError:
                        pass
                    break

    rewards = []
    eval_rewards = []
    start = time.time()

    for episode in range(1, max_episodes + 1):
        ep_reward, _ = agent.train_policy_real()
        rewards.append(ep_reward)

        # Eval every 20 episodes
        if episode % 20 == 0:
            if agent.sac_trainer:
                eval_r = agent.sac_trainer.evaluate(env, episodes=5)
            else:
                eval_r = agent.real_trainer.evaluate(env, episodes=5)
            eval_rewards.append((episode, eval_r))

    elapsed = time.time() - start
    env.close()

    return {
        "rewards": rewards,
        "eval_rewards": eval_rewards,
        "transferred": transferred_skill.name if transferred_skill else None,
        "elapsed": elapsed,
        "final_eval": eval_rewards[-1][1] if eval_rewards else None,
    }


def benchmark(env_name: str, max_episodes: int = 200, seed: int = 42):
    """Compare training with and without transfer."""
    spec = get_env_spec(env_name)
    threshold = {
        "CartPole-v1": 450.0,
        "MountainCar-v0": -120.0,
        "Acrobot-v1": -100.0,
        "Pendulum-v1": -200.0,
        "MountainCarContinuous-v0": 90.0,
    }.get(spec.gym_name, float("inf"))

    print("=" * 60)
    print(f"TRANSFER LEARNING BENCHMARK: {spec.gym_name}")
    print(f"Threshold: {threshold}, Episodes: {max_episodes}")
    print("=" * 60)

    # === Run WITHOUT transfer ===
    print("\n--- Training FROM SCRATCH ---")
    scratch = run_training(env_name, max_episodes, seed, transfer=False)
    print(f"  Time: {scratch['elapsed']:.1f}s")
    print(f"  Final eval: {scratch['final_eval']:.1f}")

    # Find episodes to threshold
    scratch_threshold_ep = None
    for ep, eval_r in scratch["eval_rewards"]:
        if eval_r >= threshold:
            scratch_threshold_ep = ep
            break
    if scratch_threshold_ep:
        print(f"  Reached threshold at episode {scratch_threshold_ep}")
    else:
        print(f"  Did NOT reach threshold in {max_episodes} episodes")

    # === Run WITH transfer ===
    print("\n--- Training WITH TRANSFER ---")
    transfer = run_training(env_name, max_episodes, seed, transfer=True)
    print(f"  Transferred skill: {transfer['transferred']}")
    print(f"  Time: {transfer['elapsed']:.1f}s")
    print(f"  Final eval: {transfer['final_eval']:.1f}")

    transfer_threshold_ep = None
    for ep, eval_r in transfer["eval_rewards"]:
        if eval_r >= threshold:
            transfer_threshold_ep = ep
            break
    if transfer_threshold_ep:
        print(f"  Reached threshold at episode {transfer_threshold_ep}")
    else:
        print(f"  Did NOT reach threshold in {max_episodes} episodes")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if scratch_threshold_ep and transfer_threshold_ep:
        speedup = scratch_threshold_ep / transfer_threshold_ep
        saved = scratch_threshold_ep - transfer_threshold_ep
        print(f"  Scratch:  {scratch_threshold_ep} episodes to threshold")
        print(f"  Transfer: {transfer_threshold_ep} episodes to threshold")
        print(f"  Speedup:  {speedup:.1f}x ({saved} episodes saved)")
    else:
        # Compare mean reward over last 50 episodes
        scratch_last50 = np.mean(scratch["rewards"][-50:])
        transfer_last50 = np.mean(transfer["rewards"][-50:])
        print(f"  Scratch  mean (last 50 ep): {scratch_last50:.1f}")
        print(f"  Transfer mean (last 50 ep): {transfer_last50:.1f}")
        improvement = transfer_last50 - scratch_last50
        print(f"  Improvement: {improvement:+.1f}")

    # Detailed eval trajectory
    print(f"\n{'Episode':>8} {'Scratch':>10} {'Transfer':>10} {'Delta':>10}")
    print("-" * 42)
    s_dict = dict(scratch["eval_rewards"])
    t_dict = dict(transfer["eval_rewards"])
    for ep in sorted(set(list(s_dict.keys()) + list(t_dict.keys()))):
        s = s_dict.get(ep, float("nan"))
        t = t_dict.get(ep, float("nan"))
        delta = t - s if not (np.isnan(s) or np.isnan(t)) else float("nan")
        print(f"{ep:>8} {s:>10.1f} {t:>10.1f} {delta:>+10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Transfer learning benchmark")
    parser.add_argument("--env", type=str, default="cartpole",
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Max episodes per run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark(args.env, args.episodes, args.seed)


if __name__ == "__main__":
    main()
