"""Actor-Critic policy network for the Ragnarok agent.

The policy operates in latent space: it takes (h_t, z_t, memory_context)
and outputs actions. This is the "decision maker" that learns through
dream training (imagined rollouts in the world model).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class Actor(nn.Module):
    """Maps latent state to action distribution.

    Supports both discrete (Categorical) and continuous (tanh-Normal) actions.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128,
                 mid: int = 64, discrete: bool = True):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, mid),
            nn.ELU(),
        )

        if discrete:
            self.head = nn.Linear(mid, action_dim)
        else:
            self.mean_head = nn.Linear(mid, action_dim)
            self.logstd_head = nn.Linear(mid, action_dim)

    def forward(self, state: torch.Tensor) -> Categorical | Normal:
        """Return action distribution given state."""
        features = self.trunk(state)

        if self.discrete:
            logits = self.head(features)
            return Categorical(logits=logits)
        else:
            mean = self.mean_head(features)
            logstd = self.logstd_head(features).clamp(-5.0, 2.0)
            return Normal(mean, logstd.exp())

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action (or take the mode if deterministic).

        For discrete: uses straight-through Gumbel-Softmax during training
        to maintain gradient flow. Returns one-hot vectors.
        For continuous: uses reparameterized sampling.
        """
        features = self.trunk(state)

        if self.discrete:
            logits = self.head(features)
            if deterministic:
                action_idx = logits.argmax(dim=-1)
                return F.one_hot(action_idx, self.action_dim).float()
            else:
                # Straight-through Gumbel-Softmax: differentiable discrete sampling
                # Forward: hard one-hot, Backward: soft probabilities
                soft = F.gumbel_softmax(logits, tau=1.0, hard=True)
                return soft
        else:
            mean = self.mean_head(features)
            logstd = self.logstd_head(features).clamp(-5.0, 2.0)
            if deterministic:
                return torch.tanh(mean)
            else:
                dist = Normal(mean, logstd.exp())
                return torch.tanh(dist.rsample())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of an action given state."""
        dist = self.forward(state)

        if self.discrete:
            # action is one-hot -> get index
            action_idx = action.argmax(dim=-1)
            return dist.log_prob(action_idx)
        else:
            raw_action = torch.atanh(action.clamp(-0.999, 0.999))
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            # tanh correction
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            return log_prob

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the action distribution."""
        features = self.trunk(state)
        if self.discrete:
            logits = self.head(features)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            return -(probs * log_probs).sum(dim=-1)
        else:
            logstd = self.logstd_head(features).clamp(-5.0, 2.0)
            # Entropy of Normal: 0.5 * ln(2*pi*e*sigma^2) per dimension
            return (0.5 + 0.5 * np.log(2 * np.pi) + logstd).sum(dim=-1)


class Critic(nn.Module):
    """Estimates state value V(s) for the actor-critic."""

    def __init__(self, state_dim: int, hidden: int = 128, mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, mid),
            nn.ELU(),
            nn.Linear(mid, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined actor-critic for the Ragnarok agent.

    Input state = concat(h_t, z_t) or concat(h_t, z_t, memory_context).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int = 128, mid: int = 64, discrete: bool = True):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, hidden, mid, discrete)
        self.critic = Critic(state_dim, hidden, mid)
        self.action_dim = action_dim
        self.discrete = discrete

    def act(self, h: torch.Tensor, z: torch.Tensor,
            memory_context: torch.Tensor | None = None,
            deterministic: bool = False) -> torch.Tensor:
        """Produce action from latent state."""
        state = self._build_state(h, z, memory_context)
        return self.actor.act(state, deterministic)

    def evaluate(self, h: torch.Tensor, z: torch.Tensor,
                 memory_context: torch.Tensor | None = None
                 ) -> torch.Tensor:
        """Estimate value of latent state."""
        state = self._build_state(h, z, memory_context)
        return self.critic(state)

    def policy_fn(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Convenience for RSSM.imagine() - returns action from (h, z)."""
        state = torch.cat([h, z], dim=-1)
        return self.actor.act(state)

    def _build_state(self, h: torch.Tensor, z: torch.Tensor,
                     memory_context: torch.Tensor | None) -> torch.Tensor:
        parts = [h, z]
        if memory_context is not None:
            parts.append(memory_context)
        return torch.cat(parts, dim=-1)
