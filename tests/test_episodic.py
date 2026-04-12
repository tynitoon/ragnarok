"""Tests for episodic memory."""

import numpy as np
import pytest
from ragnarok.memory.episodic import EpisodicMemory


class TestEpisodicMemory:
    def test_add_and_query(self):
        mem = EpisodicMemory(state_dim=4, action_dim=2, capacity=100)
        for i in range(10):
            state = np.array([float(i)] * 4, dtype=np.float32)
            action = np.array([1.0, 0.0], dtype=np.float32)
            mem.add(state, action, float(i))

        assert mem.size == 10

        query = np.array([5.0] * 4, dtype=np.float32)
        states, actions, rewards, distances = mem.query(query, k=3)
        assert len(states) == 3
        assert distances[0] == 0.0  # Exact match

    def test_get_context(self):
        mem = EpisodicMemory(state_dim=4, action_dim=2, capacity=100)
        for i in range(10):
            mem.add(np.ones(4) * i, np.zeros(2), 0.0)

        context = mem.get_context(np.ones(4) * 5.0, k=3)
        assert context.shape == (4,)

    def test_empty_query(self):
        mem = EpisodicMemory(state_dim=4, action_dim=2)
        states, actions, rewards, distances = mem.query(np.zeros(4))
        assert len(states) == 0

    def test_serialization(self):
        mem = EpisodicMemory(state_dim=4, action_dim=2, capacity=100)
        for i in range(5):
            mem.add(np.ones(4) * i, np.zeros(2), float(i))

        state = mem.state_dict()
        mem2 = EpisodicMemory.from_state_dict(state)
        assert mem2.size == 5
        np.testing.assert_array_equal(mem.states[:5], mem2.states[:5])
