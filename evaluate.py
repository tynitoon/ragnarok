"""Evaluate Ragnarok skills and agents.

Usage:
    # Evaluate a specific skill
    python evaluate.py --skill CartPole-v1_60ep --episodes 10

    # Evaluate from a checkpoint
    python evaluate.py --checkpoint checkpoints/cartpole_xxx_final.pt --env cartpole --episodes 10

    # List all available skills
    python evaluate.py --list-skills

    # Evaluate all skills
    python evaluate.py --all-skills --episodes 5
"""

import argparse
import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import get_env_spec, REGISTRY
from ragnarok.skills.library import SkillLibrary
from ragnarok.learning.real_experience import RealExperienceTrainer, DirectPolicyNet


def evaluate_skill(skill_name: str, episodes: int = 10, seed: int = 42):
    """Evaluate a crystallized skill."""
    library = SkillLibrary()
    skill = library.load_skill(skill_name)
    if skill is None:
        print(f"Skill '{skill_name}' not found. Available: {library.list_skills()}")
        return

    # Find env spec by matching gym_name
    env_spec = None
    for name, spec in REGISTRY.items():
        if spec.gym_name == skill.env_name:
            env_spec = spec
            break

    if env_spec is None:
        print(f"Environment '{skill.env_name}' not in registry")
        return

    # Load normalizer state from skill (policy was trained with this normalizer)
    from ragnarok.core.normalizer import RunningNormalizer
    normalizer = None
    if skill.normalizer_state:
        try:
            normalizer = RunningNormalizer.from_state_dict(skill.normalizer_state)
        except Exception:
            pass

    # SAC (continuous) skills use fixed normalization (from obs space bounds)
    # instead of running normalizer to avoid replay buffer distribution shift.
    normalize = env_spec.is_discrete
    env = RagnarokEnv(env_spec.gym_name, seed=seed, normalizer=normalizer,
                      normalize=normalize)
    if not normalize:
        # Freeze the loaded normalizer so stats don't update during eval
        env.normalizer.freeze()
    env.normalizer.freeze()

    # Load appropriate policy type (auto-detect from saved weights)
    if env_spec.is_discrete:
        policy = DirectPolicyNet(env_spec.obs_dim, env_spec.action_dim).to(DEVICE)
    else:
        # Detect SAC vs ContinuousPolicyNet by checking for critic_head key
        has_critic = any("critic_head" in k for k in skill.policy_state_dict)
        if has_critic:
            from ragnarok.learning.real_experience import ContinuousPolicyNet
            policy = ContinuousPolicyNet(
                env_spec.obs_dim, env_spec.action_dim,
                action_low=env.action_low, action_high=env.action_high,
            ).to(DEVICE)
        else:
            from ragnarok.learning.sac import SACPolicy
            policy = SACPolicy(
                env_spec.obs_dim, env_spec.action_dim,
                action_low=env.action_low, action_high=env.action_high,
            ).to(DEVICE)
    policy.load_state_dict({k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()})
    policy.eval()

    print(f"Skill: {skill.name}")
    print(f"Environment: {skill.env_name}")
    print(f"Training performance: {skill.performance:.1f}")
    print(f"Episodes trained: {skill.episodes_trained}")
    print()

    rewards = []
    lengths = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if env_spec.is_discrete:
                    action_idx = policy.act(obs_t, deterministic=True)
                    action = env.action_to_onehot(action_idx)
                else:
                    action = policy.act(obs_t, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep:2d}: reward = {total_reward:7.1f}, steps = {steps}")

    env.close()

    print(f"\nResults over {episodes} episodes:")
    print(f"  Mean reward:  {np.mean(rewards):7.1f} +/- {np.std(rewards):.1f}")
    print(f"  Min/Max:      {np.min(rewards):7.1f} / {np.max(rewards):.1f}")
    print(f"  Mean length:  {np.mean(lengths):7.1f}")
    return np.mean(rewards)


def evaluate_checkpoint(checkpoint_path: str, env_name: str,
                        episodes: int = 10, seed: int = 42):
    """Evaluate an agent from checkpoint using the direct policy."""
    from ragnarok.core.agent import RagnarokAgent

    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed)
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim

    env = RagnarokEnv(spec.gym_name, seed=seed)
    agent = RagnarokAgent(config, env)
    agent.load(checkpoint_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {spec.gym_name}")
    print(f"Training episodes: {agent.total_episodes}")
    print()

    eval_reward = agent.real_trainer.evaluate(env, episodes=episodes)
    print(f"\nMean reward over {episodes} episodes: {eval_reward:.1f}")

    env.close()
    return eval_reward


def list_skills():
    """List all available skills."""
    library = SkillLibrary()
    skills = library.list_skills()
    if not skills:
        print("No skills found in skill library.")
        return

    print(f"{'Name':<30} {'Environment':<20} {'Performance':>12} {'Episodes':>10}")
    print("-" * 75)
    for name in sorted(skills):
        skill = library.load_skill(name)
        if skill:
            print(f"{skill.name:<30} {skill.env_name:<20} {skill.performance:>12.1f} {skill.episodes_trained:>10}")


def evaluate_all_skills(episodes: int = 5, seed: int = 42):
    """Evaluate all skills in the library."""
    library = SkillLibrary()
    skills = library.list_skills()
    if not skills:
        print("No skills found.")
        return

    print("=" * 60)
    print("RAGNAROK - SKILL EVALUATION REPORT")
    print("=" * 60)
    print()

    results = {}
    for name in sorted(skills):
        print(f"--- {name} ---")
        mean_reward = evaluate_skill(name, episodes=episodes, seed=seed)
        results[name] = mean_reward
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, reward in results.items():
        print(f"  {name:<30} -> {reward:7.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ragnarok agent/skills")
    parser.add_argument("--skill", type=str, help="Skill name to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--env", type=str, default="cartpole", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list-skills", action="store_true", help="List available skills")
    parser.add_argument("--all-skills", action="store_true", help="Evaluate all skills")
    args = parser.parse_args()

    if args.list_skills:
        list_skills()
    elif args.all_skills:
        evaluate_all_skills(episodes=args.episodes, seed=args.seed)
    elif args.skill:
        evaluate_skill(args.skill, episodes=args.episodes, seed=args.seed)
    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args.env, episodes=args.episodes, seed=args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
