"""Tests for skill system."""

import numpy as np
import torch
import pytest
import tempfile
import os
from ragnarok.skills.skill import Skill
from ragnarok.skills.library import SkillLibrary


class TestSkill:
    def test_create_skill(self):
        skill = Skill(
            name="test_skill",
            env_name="CartPole-v1",
            policy_state_dict={"weight": torch.randn(3, 3)},
            latent_centroid=np.zeros(128),
            performance=450.0,
            normalizer_state={"mean": np.zeros(4)},
        )
        assert skill.name == "test_skill"
        assert skill.performance == 450.0


class TestSkillLibrary:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="test",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0, 2.0])},
                latent_centroid=np.array([1.0, 2.0, 3.0]),
                performance=500.0,
                normalizer_state={},
            )
            lib.save_skill(skill)
            assert "test" in lib.list_skills()

            # Load in new library
            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("test")
            assert loaded is not None
            assert loaded.performance == 500.0

    def test_find_nearest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            for i, name in enumerate(["skill_a", "skill_b", "skill_c"]):
                skill = Skill(
                    name=name,
                    env_name="test",
                    policy_state_dict={},
                    latent_centroid=np.array([float(i * 10)] * 3),
                    performance=100.0,
                    normalizer_state={},
                )
                lib.save_skill(skill)

            # Query close to skill_b (centroid [10, 10, 10])
            nearest, dist = lib.find_nearest(np.array([11.0, 11.0, 11.0]))
            assert nearest.name == "skill_b"
