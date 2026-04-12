"""Skill selector: deterministic heuristic for choosing which skill to transfer from.

Runs a few warmup steps in the new environment, encodes the observations
to get the latent state, and finds the nearest skill in the library.
"""

import numpy as np
import torch

from ragnarok.core.rssm import RSSM
from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.skill import Skill
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.infrastructure.device import DEVICE


class SkillSelector:
    """Selects the most relevant existing skill for a new task."""

    def __init__(self, rssm: RSSM, skill_library: SkillLibrary,
                 warmup_steps: int = 10, distance_threshold: float = 50.0):
        self.rssm = rssm
        self.library = skill_library
        self.warmup_steps = warmup_steps
        self.distance_threshold = distance_threshold

    @torch.no_grad()
    def select(self, env: RagnarokEnv) -> Skill | None:
        """Run warmup steps, encode observations, find nearest skill.

        Returns the best matching skill, or None if no skill is close enough.
        """
        if self.library.num_skills == 0:
            return None

        # Collect warmup observations with random actions
        obs = env.reset()
        h, z = self.rssm.initial_state(1, DEVICE)
        h_states = []

        zero_action = torch.zeros(1, env.action_dim, device=DEVICE)

        for step in range(self.warmup_steps):
            obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)
            action = zero_action if step == 0 else torch.tensor(
                env.sample_random_action(), device=DEVICE
            ).unsqueeze(0)

            h, z = self.rssm.encode_observation(obs_t, h, z, action)
            h_states.append(h.squeeze(0).cpu().numpy())

            raw_action = env.sample_random_action()
            obs, _, terminated, truncated, _ = env.step(raw_action)
            if terminated or truncated:
                obs = env.reset()

        # Mean latent state as query
        mean_h = np.mean(h_states, axis=0)
        skill, distance = self.library.find_nearest(mean_h)

        if skill is not None and distance < self.distance_threshold:
            return skill
        return None
