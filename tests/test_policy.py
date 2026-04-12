"""Tests for the Actor-Critic policy network."""

import torch
import pytest
from ragnarok.core.policy import Actor, Critic, ActorCritic


class TestActor:
    def test_discrete_act_shape(self):
        actor = Actor(state_dim=16, action_dim=3, hidden=32, mid=16, discrete=True)
        state = torch.randn(5, 16)
        action = actor.act(state)
        assert action.shape == (5, 3)
        # One-hot: each row should sum to 1
        assert torch.allclose(action.sum(dim=-1), torch.ones(5))

    def test_discrete_deterministic(self):
        actor = Actor(state_dim=16, action_dim=3, hidden=32, mid=16, discrete=True)
        state = torch.randn(1, 16)
        action = actor.act(state, deterministic=True)
        assert action.shape == (1, 3)
        assert torch.allclose(action.sum(dim=-1), torch.ones(1))

    def test_entropy_shape(self):
        actor = Actor(state_dim=16, action_dim=3, hidden=32, mid=16, discrete=True)
        state = torch.randn(5, 16)
        entropy = actor.entropy(state)
        assert entropy.shape == (5,)
        assert (entropy >= 0).all()

    def test_gumbel_softmax_has_gradients(self):
        actor = Actor(state_dim=16, action_dim=3, hidden=32, mid=16, discrete=True)
        state = torch.randn(5, 16, requires_grad=True)
        action = actor.act(state)
        loss = action.sum()
        loss.backward()
        # Gradients should flow through Gumbel-Softmax
        assert state.grad is not None


class TestCritic:
    def test_output_shape(self):
        critic = Critic(state_dim=16, hidden=32, mid=16)
        state = torch.randn(5, 16)
        value = critic(state)
        assert value.shape == (5,)


class TestActorCritic:
    def test_act_with_memory_context(self):
        ac = ActorCritic(state_dim=32, action_dim=2, hidden=16, mid=8, discrete=True)
        h = torch.randn(3, 16)
        z = torch.randn(3, 16)
        # Without memory context: state_dim should be 32
        action = ac.act(h, z)
        assert action.shape == (3, 2)

    def test_policy_fn(self):
        ac = ActorCritic(state_dim=32, action_dim=2, hidden=16, mid=8, discrete=True)
        h = torch.randn(3, 16)
        z = torch.randn(3, 16)
        action = ac.policy_fn(h, z)
        assert action.shape == (3, 2)
