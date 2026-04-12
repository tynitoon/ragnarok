"""Tests for multi-skill agent and router."""

import numpy as np
import pytest
import torch

from ragnarok.skills.router import CentroidRouter, LearnedRouter
from ragnarok.skills.multi_agent import MultiSkillAgent, LoadedSkill
from ragnarok.skills.library import SkillLibrary


class TestCentroidRouter:
    def test_select_nearest(self):
        centroids = {
            "skill_a": np.array([1.0, 0.0, 0.0]),
            "skill_b": np.array([0.0, 1.0, 0.0]),
            "skill_c": np.array([0.0, 0.0, 1.0]),
        }
        router = CentroidRouter(centroids)

        assert router.select(np.array([0.9, 0.1, 0.0])) == "skill_a"
        assert router.select(np.array([0.0, 0.8, 0.2])) == "skill_b"
        assert router.select(np.array([0.1, 0.1, 0.9])) == "skill_c"

    def test_select_soft_sums_to_one(self):
        centroids = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
        router = CentroidRouter(centroids)
        probs = router.select_soft(np.array([0.5, 0.5]))
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_temperature_affects_distribution(self):
        centroids = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
        # Low temperature -> more decisive
        router_low = CentroidRouter(centroids, temperature=0.1)
        router_high = CentroidRouter(centroids, temperature=10.0)

        query = np.array([0.8, 0.2])
        probs_low = router_low.select_soft(query)
        probs_high = router_high.select_soft(query)

        # Low temp should be more peaked toward "a"
        assert probs_low["a"] > probs_high["a"]


class TestLearnedRouter:
    def test_output_shape(self):
        router = LearnedRouter(obs_dim=4, num_skills=3, hidden=32)
        obs = torch.randn(8, 4)
        logits = router(obs)
        assert logits.shape == (8, 3)

    def test_select_returns_valid_index(self):
        router = LearnedRouter(obs_dim=4, num_skills=3, hidden=32)
        obs = torch.randn(1, 4)
        idx = router.select(obs)
        assert 0 <= idx < 3

    def test_train_step_reduces_loss(self):
        router = LearnedRouter(obs_dim=4, num_skills=2, hidden=32)
        obs = torch.randn(32, 4)
        labels = torch.randint(0, 2, (32,))

        loss1 = router.train_step(obs, labels)
        for _ in range(50):
            router.train_step(obs, labels)
        loss2 = router.train_step(obs, labels)

        assert loss2 < loss1


class TestMultiSkillAgent:
    def test_load_from_library(self):
        agent = MultiSkillAgent()
        skills = agent.library.list_skills()
        if not skills:
            pytest.skip("No skills available for testing")

        agent.load_all_skills()
        assert len(agent.loaded_skills) == len(skills)
        assert agent.centroid_router is not None

    def test_select_skill_for_env(self):
        agent = MultiSkillAgent()
        agent.load_all_skills()
        if not agent.loaded_skills:
            pytest.skip("No skills available")

        # Get the env name of the first loaded skill
        first = next(iter(agent.loaded_skills.values()))
        selected = agent.select_skill_for_env(first.skill.env_name)
        assert selected is not None
        assert selected.skill.env_name == first.skill.env_name

    def test_missing_env_returns_none(self):
        agent = MultiSkillAgent()
        agent.load_all_skills()
        assert agent.select_skill_for_env("NonExistent-v99") is None
