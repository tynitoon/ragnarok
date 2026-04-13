"""Learned Router Demo.

Trains a small MLP to map observations -> best skill,
then tests it on multi-environment switching.

Usage:
    python demo_router.py
"""

import numpy as np
import torch

from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.router import CentroidRouter, LearnedRouter
from ragnarok.skills.multi_agent import MultiSkillAgent, LoadedSkill
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import REGISTRY
from ragnarok.infrastructure.device import DEVICE


def collect_router_training_data(agent: MultiSkillAgent,
                                 episodes_per_skill: int = 20,
                                 seed: int = 42):
    """Collect (obs, best_skill_idx) pairs by running each skill in its env.

    For each skill, collect observations and label them with the skill index.
    """
    skill_names = list(agent.loaded_skills.keys())
    obs_data = []
    label_data = []

    for idx, name in enumerate(skill_names):
        ls = agent.loaded_skills[name]
        env = RagnarokEnv(ls.skill.env_name, seed=seed,
                          normalize=ls.is_discrete)

        for ep in range(episodes_per_skill):
            obs = env.reset()
            done = False
            while not done:
                obs_data.append(obs.copy())
                label_data.append(idx)
                # Use the skill's policy
                action = agent.act(env.last_raw_obs, name, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        env.close()
        print(f"  Collected {len([l for l in label_data if l == idx])} "
              f"samples for {name} (idx={idx})")

    return (
        np.array(obs_data, dtype=np.float32),
        np.array(label_data, dtype=np.int64),
        skill_names,
    )


def train_router(obs: np.ndarray, labels: np.ndarray,
                 num_skills: int, epochs: int = 50,
                 batch_size: int = 128) -> LearnedRouter:
    """Train a LearnedRouter on collected data."""
    obs_dim = obs.shape[1]
    router = LearnedRouter(obs_dim, num_skills).to(DEVICE)

    n = len(obs)
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            obs_batch = torch.tensor(obs[batch_idx], device=DEVICE)
            label_batch = torch.tensor(labels[batch_idx], device=DEVICE)
            loss = router.train_step(obs_batch, label_batch)
            total_loss += loss
            batches += 1

        if epoch % 10 == 0:
            # Compute accuracy
            with torch.no_grad():
                all_obs = torch.tensor(obs, device=DEVICE)
                pred = router(all_obs).argmax(dim=-1).cpu().numpy()
                acc = (pred == labels).mean() * 100
            print(f"  Epoch {epoch:3d}: loss={total_loss/batches:.3f}, "
                  f"accuracy={acc:.1f}%")

    return router


def test_router(router: LearnedRouter, skill_names: list[str],
                agent: MultiSkillAgent, episodes: int = 5, seed: int = 42):
    """Test the learned router by selecting skills dynamically."""
    # Test on each environment
    env_names = list({ls.skill.env_name for ls in agent.loaded_skills.values()})

    print(f"\n{'Environment':<25} {'Selected Skill':<30} {'Correct':>8} {'Reward':>10}")
    print("-" * 78)

    for env_name in sorted(env_names):
        # Find the actual best skill for this env
        best_skill = None
        for name, ls in agent.loaded_skills.items():
            if ls.skill.env_name == env_name:
                best_skill = name
                break

        # Get discrete flag
        is_discrete = any(
            ls.is_discrete for ls in agent.loaded_skills.values()
            if ls.skill.env_name == env_name
        )

        env = RagnarokEnv(env_name, seed=seed, normalize=is_discrete)

        obs = env.reset()
        # Router selects based on first observation (zero-padded to match training)
        padded = np.zeros(router.net[0].in_features, dtype=np.float32)
        padded[:len(obs)] = obs
        obs_t = torch.tensor(padded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        selected_idx = router.select(obs_t)
        selected_name = skill_names[selected_idx]
        correct = "YES" if selected_name == best_skill else "NO"

        # Run episode with selected skill
        rewards = []
        for _ in range(episodes):
            try:
                r = agent.run_episode(env, skill_name=selected_name)
                rewards.append(r)
            except (ValueError, KeyError):
                rewards.append(float("nan"))

        mean_r = np.nanmean(rewards) if rewards else float("nan")
        print(f"{env_name:<25} {selected_name:<30} {correct:>8} {mean_r:>10.1f}")

        env.close()


def main():
    print("=" * 60)
    print("LEARNED ROUTER DEMO")
    print("=" * 60)

    # Load all skills
    agent = MultiSkillAgent()
    agent.load_all_skills()
    print(f"\nLoaded {len(agent.loaded_skills)} skills:")
    for name, ls in agent.loaded_skills.items():
        print(f"  {name}: {ls.skill.env_name} "
              f"(perf={ls.skill.performance:.1f}, "
              f"{'discrete' if ls.is_discrete else 'continuous'})")

    # Only use discrete skills for router demo (shared obs space needed)
    discrete_skills = {
        name: ls for name, ls in agent.loaded_skills.items()
        if ls.is_discrete
    }
    # For learned router, we need a common observation space
    # Group by obs_dim
    obs_dims = {name: ls.skill.policy_state_dict[
        list(ls.skill.policy_state_dict.keys())[0]
    ].shape[-1] for name, ls in discrete_skills.items()}
    print(f"\nDiscrete skills obs dims: {obs_dims}")

    # === 1. Centroid Router ===
    print("\n--- Centroid Router ---")
    centroids = {
        name: ls.skill.latent_centroid
        for name, ls in agent.loaded_skills.items()
    }
    centroid_router = CentroidRouter(centroids)

    # Test with random latent vectors
    for name, ls in list(agent.loaded_skills.items())[:3]:
        probs = centroid_router.select_soft(ls.skill.latent_centroid)
        top = max(probs, key=probs.get)
        print(f"  Centroid of '{name}' -> routes to '{top}' "
              f"(p={probs[top]:.2f})")

    # === 2. Collect training data ===
    print("\n--- Collecting Router Training Data ---")
    # Use the largest obs_dim for zero-padded unified input
    max_obs_dim = max(
        RagnarokEnv(ls.skill.env_name).obs_dim
        for ls in agent.loaded_skills.values()
    )

    all_obs = []
    all_labels = []
    skill_names = list(agent.loaded_skills.keys())

    for idx, name in enumerate(skill_names):
        ls = agent.loaded_skills[name]
        is_disc = ls.is_discrete
        env = RagnarokEnv(ls.skill.env_name, normalize=is_disc)
        for ep in range(10):
            obs = env.reset()
            done = False
            while not done:
                # Zero-pad to max_obs_dim
                padded = np.zeros(max_obs_dim, dtype=np.float32)
                padded[:len(obs)] = obs
                all_obs.append(padded)
                all_labels.append(idx)
                action = agent.act(env.last_raw_obs, name, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()
        count = sum(1 for l in all_labels if l == idx)
        print(f"  {name}: {count} samples")

    obs_arr = np.array(all_obs, dtype=np.float32)
    label_arr = np.array(all_labels, dtype=np.int64)
    print(f"  Total: {len(obs_arr)} samples, {len(skill_names)} classes")

    # === 3. Train Learned Router ===
    print("\n--- Training Learned Router ---")
    router = train_router(obs_arr, label_arr, len(skill_names), epochs=50)

    # === 4. Test Router ===
    print("\n--- Testing Learned Router ---")
    test_router(router, skill_names, agent)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
