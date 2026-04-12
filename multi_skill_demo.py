"""Demo: Multi-skill agent switching between environments.

Shows the agent loading all crystallized skills and executing them
across different environments — no retraining needed.

Usage:
    python multi_skill_demo.py
    python multi_skill_demo.py --episodes 10
"""

import argparse
from ragnarok.skills.multi_agent import MultiSkillAgent


def main():
    parser = argparse.ArgumentParser(description="Multi-skill agent demo")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per environment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("RAGNAROK - MULTI-SKILL AGENT DEMO")
    print("=" * 60)

    agent = MultiSkillAgent()
    agent.load_all_skills()

    skills = list(agent.loaded_skills.keys())
    if not skills:
        print("No skills found. Train some environments first!")
        return

    print(f"\nLoaded {len(skills)} skills:")
    for name, ls in agent.loaded_skills.items():
        print(f"  {name}: {ls.skill.env_name} "
              f"(perf: {ls.skill.performance:.1f}, "
              f"{'discrete' if ls.is_discrete else 'continuous'})")

    # Evaluate all skills in their environments
    print(f"\n--- Evaluating all skills ({args.episodes} episodes each) ---")
    results = agent.evaluate_all(
        episodes_per_skill=args.episodes, seed=args.seed
    )

    # Multi-task run: cycle through all environments
    env_names = [ls.skill.env_name for ls in agent.loaded_skills.values()]
    unique_envs = list(dict.fromkeys(env_names))  # Deduplicate, preserve order

    print(f"\n--- Multi-task run: {' -> '.join(unique_envs)} ---")
    multi_results = agent.run_multi_task(
        unique_envs, episodes_per_env=args.episodes, seed=args.seed
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for env_name, rewards in multi_results.items():
        import numpy as np
        print(f"  {env_name}: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")


if __name__ == "__main__":
    main()
