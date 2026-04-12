"""Multi-skill agent: orchestrates multiple learned skills.

Loads skills from the library, routes observations to the most
appropriate skill, and executes the selected skill's policy.
Can also learn new skills while leveraging existing ones.
"""

import numpy as np
import torch

from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.skill import Skill
from ragnarok.skills.router import CentroidRouter, LearnedRouter
from ragnarok.learning.real_experience import DirectPolicyNet, ContinuousPolicyNet
from ragnarok.core.normalizer import RunningNormalizer
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.environments.registry import REGISTRY
from ragnarok.infrastructure.device import DEVICE


class LoadedSkill:
    """A skill loaded into memory with its policy ready to execute."""

    def __init__(self, skill: Skill, policy, normalizer: RunningNormalizer,
                 is_discrete: bool):
        self.skill = skill
        self.policy = policy
        self.normalizer = normalizer
        self.is_discrete = is_discrete

    @staticmethod
    def from_skill(skill: Skill) -> "LoadedSkill":
        """Load a skill from its stored data."""
        # Find env spec to determine obs/action dims and type
        env_spec = None
        for spec in REGISTRY.values():
            if spec.gym_name == skill.env_name:
                env_spec = spec
                break
        if env_spec is None:
            raise ValueError(f"No registry entry for {skill.env_name}")

        # Build policy
        if env_spec.is_discrete:
            policy = DirectPolicyNet(env_spec.obs_dim, env_spec.action_dim).to(DEVICE)
        else:
            policy = ContinuousPolicyNet(env_spec.obs_dim, env_spec.action_dim).to(DEVICE)

        policy.load_state_dict(
            {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
        )
        policy.eval()

        normalizer = RunningNormalizer.from_state_dict(skill.normalizer_state)

        return LoadedSkill(skill, policy, normalizer, env_spec.is_discrete)


class MultiSkillAgent:
    """Agent that dynamically switches between multiple learned skills.

    Usage:
        agent = MultiSkillAgent()
        agent.load_all_skills()
        # or agent.load_skills(["CartPole-v1_60ep", "Acrobot-v1_200ep"])

        # Run an episode with automatic skill switching
        reward = agent.run_episode(env, routing="centroid")
    """

    def __init__(self, library: SkillLibrary | None = None):
        self.library = library or SkillLibrary()
        self.loaded_skills: dict[str, LoadedSkill] = {}
        self.centroid_router: CentroidRouter | None = None
        self.learned_router: LearnedRouter | None = None

    def load_skills(self, skill_names: list[str]):
        """Load specific skills into memory."""
        for name in skill_names:
            skill = self.library.load_skill(name)
            if skill is None:
                print(f"Warning: skill '{name}' not found, skipping")
                continue
            self.loaded_skills[name] = LoadedSkill.from_skill(skill)

        self._build_centroid_router()

    def load_all_skills(self):
        """Load all available skills."""
        self.load_skills(self.library.list_skills())

    def _build_centroid_router(self):
        """Build centroid-based router from loaded skills."""
        if not self.loaded_skills:
            return
        centroids = {
            name: ls.skill.latent_centroid
            for name, ls in self.loaded_skills.items()
        }
        self.centroid_router = CentroidRouter(centroids)

    def select_skill_for_env(self, env_name: str) -> LoadedSkill | None:
        """Select the best skill for a given environment name."""
        for ls in self.loaded_skills.values():
            if ls.skill.env_name == env_name:
                return ls
        return None

    def act(self, obs: np.ndarray, skill_name: str,
            deterministic: bool = True) -> np.ndarray:
        """Execute an action using a specific skill's policy.

        Args:
            obs: raw observation (will be normalized with skill's normalizer)
            skill_name: which skill to use
            deterministic: whether to act greedily
        Returns:
            action (one-hot for discrete, raw for continuous)
        """
        ls = self.loaded_skills[skill_name]
        # Normalize with the skill's own normalizer
        norm_obs = ls.normalizer.normalize(obs)
        obs_t = torch.tensor(norm_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            if ls.is_discrete:
                action_idx = ls.policy.act(obs_t, deterministic)
                action = np.zeros(ls.skill.policy_state_dict[
                    "actor_head.weight"
                ].shape[0], dtype=np.float32)
                action[action_idx] = 1.0
            else:
                action = ls.policy.act(obs_t, deterministic)

        return action

    def run_episode(self, env: RagnarokEnv, skill_name: str | None = None,
                    deterministic: bool = True) -> float:
        """Run a full episode using a specific skill (or auto-select for env).

        Args:
            env: environment to run in
            skill_name: skill to use (auto-selects by env_name if None)
            deterministic: greedy actions
        Returns:
            total episode reward
        """
        if skill_name is None:
            ls = self.select_skill_for_env(env.env_name)
            if ls is None:
                raise ValueError(f"No skill found for {env.env_name}")
            skill_name = ls.skill.name

        obs = env.reset()
        raw_obs = env.last_raw_obs
        total_reward = 0.0
        done = False

        while not done:
            action = self.act(raw_obs, skill_name, deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            raw_obs = env.last_raw_obs
            done = terminated or truncated
            total_reward += reward

        return total_reward

    def evaluate_all(self, episodes_per_skill: int = 5,
                     seed: int = 42) -> dict[str, float]:
        """Evaluate all loaded skills in their respective environments.

        Returns:
            {skill_name: mean_reward}
        """
        results = {}
        for name, ls in self.loaded_skills.items():
            env = RagnarokEnv(ls.skill.env_name, seed=seed)
            rewards = []
            for _ in range(episodes_per_skill):
                r = self.run_episode(env, skill_name=name)
                rewards.append(r)
            env.close()
            results[name] = float(np.mean(rewards))
            print(f"  {name}: {results[name]:.1f} "
                  f"(trained: {ls.skill.performance:.1f})")
        return results

    def run_multi_task(self, env_sequence: list[str],
                       episodes_per_env: int = 1,
                       seed: int = 42) -> dict[str, list[float]]:
        """Run the agent across a sequence of environments,
        automatically switching skills.

        Args:
            env_sequence: list of env names (e.g., ["CartPole-v1", "Acrobot-v1"])
            episodes_per_env: episodes per environment
            seed: random seed
        Returns:
            {env_name: [rewards]}
        """
        results: dict[str, list[float]] = {}

        for env_name in env_sequence:
            env = RagnarokEnv(env_name, seed=seed)
            ls = self.select_skill_for_env(env_name)

            if ls is None:
                print(f"  {env_name}: no skill available, skipping")
                env.close()
                continue

            rewards = []
            for ep in range(episodes_per_env):
                # Use the skill's normalizer for this env
                env.normalizer = RunningNormalizer.from_state_dict(
                    ls.skill.normalizer_state
                )
                r = self.run_episode(env, skill_name=ls.skill.name)
                rewards.append(r)

            results[env_name] = rewards
            mean_r = np.mean(rewards)
            print(f"  {env_name} ({ls.skill.name}): "
                  f"{mean_r:.1f} over {episodes_per_env} episodes")
            env.close()

        return results
