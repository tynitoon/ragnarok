"""Skill library: persistent storage and retrieval of learned skills.

Skills are saved as .pt files on disk. The library provides
nearest-neighbor matching to find the most relevant skill for
a new task based on latent space similarity.
"""

from pathlib import Path
import numpy as np
import torch

from ragnarok.skills.skill import Skill


class SkillLibrary:
    """Manages persistent skill storage and retrieval."""

    def __init__(self, skills_dir: str = "skills_data"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Skill] = {}
        self._load_all()

    def _load_all(self):
        """Load all skills from disk into cache."""
        for path in self.skills_dir.glob("*.pt"):
            try:
                data = torch.load(path, weights_only=False)
                skill = Skill(**data)
                self._cache[skill.name] = skill
            except Exception as e:
                print(f"Warning: failed to load skill {path.name}: {e}")

    def save_skill(self, skill: Skill):
        """Save a skill to disk.

        NOTE (Phase 3 pre-launch, smoke #3): the serialized dict must
        include EVERY field the Skill dataclass carries — in
        particular `latent_trunk_state_dict`, which is what enables
        heterogeneous-dim (cross-env) transfer. Without it, the
        check_crystallization → save → load round-trip drops the
        trunk silently, `try_transfer()` hits the
        `if skill.latent_trunk_state_dict:` gate at agent.py:764 and
        returns None, and the §8 mechanism check fails because
        acting_policy_mode never flips to "latent".
        """
        data = {
            "name": skill.name,
            "env_name": skill.env_name,
            "policy_state_dict": skill.policy_state_dict,
            "latent_centroid": skill.latent_centroid,
            "performance": skill.performance,
            "normalizer_state": skill.normalizer_state,
            "created_at": skill.created_at,
            "episodes_trained": skill.episodes_trained,
            "metadata": skill.metadata,
            "latent_trunk_state_dict": skill.latent_trunk_state_dict,
            "rssm_core_state_dict": skill.rssm_core_state_dict,
        }
        path = self.skills_dir / f"{skill.name}.pt"
        torch.save(data, path)
        self._cache[skill.name] = skill

    def load_skill(self, name: str) -> Skill | None:
        """Load a skill by name."""
        if name in self._cache:
            return self._cache[name]
        path = self.skills_dir / f"{name}.pt"
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)
        skill = Skill(**data)
        self._cache[name] = skill
        return skill

    def list_skills(self) -> list[str]:
        """List all available skill names."""
        return list(self._cache.keys())

    def find_nearest(self, latent_state: np.ndarray, exclude_env: str | None = None
                     ) -> tuple[Skill | None, float]:
        """Find the skill with the nearest latent centroid.

        Args:
            latent_state: query state (hidden_dim,)
            exclude_env: optionally exclude skills from this environment

        Returns:
            (nearest_skill, distance) or (None, inf)
        """
        best_skill = None
        best_dist = float("inf")

        for skill in self._cache.values():
            if exclude_env and skill.env_name == exclude_env:
                continue
            dist = np.sqrt(np.sum((skill.latent_centroid - latent_state) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_skill = skill

        return best_skill, best_dist

    @property
    def num_skills(self) -> int:
        return len(self._cache)
