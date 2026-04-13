"""Automatic curriculum selection.

The agent chooses its next training environment based on:
1. Transfer utility: how close is the nearest existing skill?
2. Novelty: has this environment been mastered yet?
3. Cold start: when no skills exist, pick the simplest environment.

This replaces the hardcoded CURRICULUM list with intelligent ordering.
"""

import numpy as np

from ragnarok.skills.library import SkillLibrary
from ragnarok.environments.registry import REGISTRY, EnvSpec


class CurriculumSelector:
    """Selects the next environment to train based on skill library state."""

    def __init__(self, skill_library: SkillLibrary,
                 available_envs: list[str] | None = None):
        self.library = skill_library
        # Default: all non-pixel environments from registry
        if available_envs is None:
            self.available = [
                name for name, spec in REGISTRY.items()
                if not spec.pixel_obs
            ]
        else:
            self.available = list(available_envs)

    def _is_mastered(self, env_name: str) -> bool:
        """Check if we have a crystallized skill for this environment."""
        spec = REGISTRY[env_name]
        for skill in self.library._cache.values():
            if skill.env_name == spec.gym_name:
                return True
        return False

    def _simplicity_score(self, env_name: str) -> float:
        """Heuristic: simpler envs have lower obs_dim. Used for cold start."""
        spec = REGISTRY[env_name]
        return -spec.obs_dim  # Lower obs_dim = higher score

    def _transfer_score(self, env_name: str) -> float:
        """Score based on proximity to existing skills.

        High score = close to an existing skill = easy to transfer to.
        """
        if self.library.num_skills == 0:
            return 0.0

        spec = REGISTRY[env_name]
        # Use a dummy centroid query (zero vector) — we don't have the
        # actual latent state for an unvisited env, so we score based on
        # obs_dim similarity as a proxy for latent similarity.
        best_distance = float("inf")
        for skill in self.library._cache.values():
            # Obs-dim distance as proxy (same dim = more likely compatible)
            skill_spec = None
            for name, s in REGISTRY.items():
                if s.gym_name == skill.env_name:
                    skill_spec = s
                    break
            if skill_spec is None:
                continue

            # Feature similarity: same action type, similar obs dim
            dim_dist = abs(spec.obs_dim - skill_spec.obs_dim)
            type_penalty = 0 if spec.is_discrete == skill_spec.is_discrete else 10
            distance = dim_dist + type_penalty

            if distance < best_distance:
                best_distance = distance

        # Convert distance to score: close = high score
        return 1.0 / (1.0 + best_distance)

    def select_next(self) -> str | None:
        """Select the best next environment to train.

        Returns environment name, or None if all are mastered.
        """
        candidates = [env for env in self.available if not self._is_mastered(env)]
        if not candidates:
            return None

        # Cold start: no skills at all -> pick simplest environment
        if self.library.num_skills == 0:
            return max(candidates, key=self._simplicity_score)

        # Score each candidate
        scores = {}
        for env in candidates:
            transfer = self._transfer_score(env)
            novelty = 1.0  # Not mastered -> full novelty bonus
            scores[env] = transfer + novelty

        return max(scores, key=scores.get)

    def get_ordered_curriculum(self, max_episodes_per_env: int = 500
                               ) -> list[tuple[str, int]]:
        """Generate full curriculum ordering.

        Returns list of (env_name, max_episodes) in recommended order.
        """
        curriculum = []
        # Simulate selection to get full ordering
        selected = set()
        for _ in range(len(self.available)):
            candidates = [e for e in self.available
                          if e not in selected and not self._is_mastered(e)]
            if not candidates:
                break

            if not selected and self.library.num_skills == 0:
                # Cold start
                best = max(candidates, key=self._simplicity_score)
            else:
                scores = {e: self._transfer_score(e) + 1.0 for e in candidates}
                best = max(scores, key=scores.get)

            curriculum.append((best, max_episodes_per_env))
            selected.add(best)

        return curriculum
