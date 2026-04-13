"""Automatic curriculum selection.

The agent chooses its next training environment based on:
1. Transfer utility: latent centroid distance to nearest skill (when RSSM available)
2. Novelty: has this environment been mastered yet?
3. Cold start: when no skills exist, pick the simplest environment.

When an RSSM is available, the selector probes unvisited environments
with random rollouts and encodes them into latent space for actual
distance-based scoring. Falls back to action-type + dimension heuristics
when no RSSM is available.
"""

import numpy as np
import torch

from ragnarok.skills.library import SkillLibrary
from ragnarok.environments.registry import REGISTRY, EnvSpec


class CurriculumSelector:
    """Selects the next environment to train based on skill library state."""

    def __init__(self, skill_library: SkillLibrary,
                 available_envs: list[str] | None = None,
                 rssm=None):
        """
        Args:
            skill_library: Persistent skill storage.
            available_envs: Env names to consider (default: all non-pixel).
            rssm: Optional trained RSSM for latent-space probing.
        """
        self.library = skill_library
        self.rssm = rssm
        self._probe_cache: dict[str, np.ndarray] = {}  # env_name -> probe centroid

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
        return -spec.obs_dim

    def _probe_centroid(self, env_name: str) -> np.ndarray | None:
        """Run random rollouts and encode through RSSM to get probe centroid.

        Returns mean hidden state (hidden_dim,), or None if RSSM unavailable.
        """
        if self.rssm is None:
            return None

        if env_name in self._probe_cache:
            return self._probe_cache[env_name]

        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.infrastructure.device import DEVICE

        spec = REGISTRY[env_name]
        # RSSM dimensions must match — skip if obs/action dims differ
        if (spec.obs_dim != self.rssm.obs_dim or
                spec.action_dim != self.rssm.action_dim):
            return None

        env = RagnarokEnv(spec.gym_name, seed=0, pixel_obs=False)
        hidden_states = []

        try:
            for _ in range(5):  # 5 random episodes
                obs = env.reset()
                h, z = self.rssm.initial_state(1, DEVICE)
                done = False
                steps = 0
                while not done and steps < 200:
                    action = env.sample_random_action()
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=DEVICE).unsqueeze(0)
                    act_t = torch.tensor(action, dtype=torch.float32,
                                         device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        h, z = self.rssm.encode_observation(obs_t, h, z, act_t)
                    hidden_states.append(h.squeeze(0).cpu().numpy())

                    obs, _, term, trunc, _ = env.step(action)
                    done = term or trunc
                    steps += 1
        finally:
            env.close()

        if not hidden_states:
            return None

        centroid = np.mean(hidden_states, axis=0)
        self._probe_cache[env_name] = centroid
        return centroid

    def _transfer_score(self, env_name: str) -> float:
        """Score based on proximity to existing skills.

        Uses latent centroid distance when RSSM is available (probe the
        target env with random rollouts, then find nearest skill).
        Falls back to action-type + dimension heuristics otherwise.
        """
        if self.library.num_skills == 0:
            return 0.0

        spec = REGISTRY[env_name]

        # Try latent-space scoring with RSSM probe
        probe = self._probe_centroid(env_name)
        if probe is not None:
            nearest, distance = self.library.find_nearest(probe)
            if nearest is not None:
                return 1.0 / (1.0 + distance)

        # Fallback: heuristic scoring based on env properties
        best_score = 0.0
        for skill in self.library._cache.values():
            skill_spec = None
            for name, s in REGISTRY.items():
                if s.gym_name == skill.env_name:
                    skill_spec = s
                    break
            if skill_spec is None:
                continue

            # Hard filter: action type must match for policy transfer
            if spec.is_discrete != skill_spec.is_discrete:
                continue

            # Architecture compatibility: same dims = direct weight transfer
            exact_match = (spec.obs_dim == skill_spec.obs_dim and
                           spec.action_dim == skill_spec.action_dim)

            # Dimension similarity (soft)
            obs_sim = 1.0 / (1.0 + abs(spec.obs_dim - skill_spec.obs_dim))
            act_sim = 1.0 / (1.0 + abs(spec.action_dim - skill_spec.action_dim))

            score = (2.0 if exact_match else 0.0) + obs_sim + act_sim

            if score > best_score:
                best_score = score

        return best_score

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
        selected = set()
        for _ in range(len(self.available)):
            candidates = [e for e in self.available
                          if e not in selected and not self._is_mastered(e)]
            if not candidates:
                break

            if not selected and self.library.num_skills == 0:
                best = max(candidates, key=self._simplicity_score)
            else:
                scores = {e: self._transfer_score(e) + 1.0 for e in candidates}
                best = max(scores, key=scores.get)

            curriculum.append((best, max_episodes_per_env))
            selected.add(best)

        return curriculum
