"""Tests for automatic curriculum selection (Phase 5.5)."""

import numpy as np
import torch
import pytest
import tempfile
from pathlib import Path

from ragnarok.skills.curriculum import CurriculumSelector
from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.skill import Skill


@pytest.fixture
def empty_library(tmp_path):
    """Create an empty skill library in a temp directory."""
    return SkillLibrary(skills_dir=str(tmp_path / "skills"))


@pytest.fixture
def library_with_cartpole(tmp_path):
    """Create a library with a CartPole skill."""
    lib = SkillLibrary(skills_dir=str(tmp_path / "skills"))
    skill = Skill(
        name="CartPole-v1_100ep",
        env_name="CartPole-v1",
        policy_state_dict={},
        latent_centroid=np.zeros(128),
        performance=450.0,
        normalizer_state={},
        episodes_trained=100,
    )
    lib.save_skill(skill)
    return lib


class TestCurriculumSelection:

    def test_cold_start_picks_simplest(self, empty_library):
        """With no skills, should pick simplest env (lowest obs_dim)."""
        selector = CurriculumSelector(
            empty_library,
            available_envs=["cartpole", "acrobot", "mountaincar"]
        )
        # mountaincar has obs_dim=2 (simplest), acrobot has 6, cartpole 4
        choice = selector.select_next()
        assert choice == "mountaincar"

    def test_skips_mastered(self, library_with_cartpole):
        """Should skip environments that already have a crystallized skill."""
        selector = CurriculumSelector(
            library_with_cartpole,
            available_envs=["cartpole", "mountaincar"]
        )
        # CartPole is mastered, should pick mountaincar
        choice = selector.select_next()
        assert choice == "mountaincar"

    def test_all_mastered_returns_none(self, library_with_cartpole):
        """Should return None when all envs are mastered."""
        selector = CurriculumSelector(
            library_with_cartpole,
            available_envs=["cartpole"]
        )
        choice = selector.select_next()
        assert choice is None

    def test_transfer_score_prefers_compatible(self, library_with_cartpole):
        """Should prefer envs with compatible action type to existing skills."""
        selector = CurriculumSelector(
            library_with_cartpole,
            available_envs=["mountaincar", "acrobot", "pendulum"]
        )
        # CartPole is discrete, so discrete envs should score higher
        # than pendulum (continuous)
        choice = selector.select_next()
        assert choice in ["mountaincar", "acrobot"]  # Not pendulum

    def test_ordered_curriculum(self, empty_library):
        """get_ordered_curriculum should return all non-mastered envs."""
        selector = CurriculumSelector(
            empty_library,
            available_envs=["cartpole", "mountaincar", "acrobot"]
        )
        curriculum = selector.get_ordered_curriculum(max_episodes_per_env=300)
        assert len(curriculum) == 3
        # All envs should appear
        env_names = [name for name, _ in curriculum]
        assert set(env_names) == {"cartpole", "mountaincar", "acrobot"}
        # All should have max_episodes
        for _, eps in curriculum:
            assert eps == 300

    def test_cold_start_curriculum_order(self, empty_library):
        """First env in curriculum should be the simplest (cold start)."""
        selector = CurriculumSelector(
            empty_library,
            available_envs=["cartpole", "mountaincar", "acrobot"]
        )
        curriculum = selector.get_ordered_curriculum()
        # First should be mountaincar (obs_dim=2, simplest)
        assert curriculum[0][0] == "mountaincar"

    def test_mastered_env_excluded(self, library_with_cartpole):
        """Mastered envs should not appear in curriculum."""
        selector = CurriculumSelector(
            library_with_cartpole,
            available_envs=["cartpole", "mountaincar", "acrobot"]
        )
        curriculum = selector.get_ordered_curriculum()
        env_names = [name for name, _ in curriculum]
        assert "cartpole" not in env_names
        assert len(curriculum) == 2

    def test_default_available_envs(self, empty_library):
        """Default should include all non-pixel envs from registry."""
        selector = CurriculumSelector(empty_library)
        assert "cartpole" in selector.available
        assert "mountaincar" in selector.available
        assert "cartpole-pixels" not in selector.available
