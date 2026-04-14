"""Latent-space policy head operating on RSSM state cat(h, z).

Unlike obs-space policies (DirectPolicyNet, ContinuousPolicyNet), this
policy takes the RSSM hidden state as input. Since (h_dim, z_dim) are
constant across all environments, this enables:
  1. Cross-environment transfer (CartPole -> Acrobot despite different obs/act dims)
  2. Efficient dream training (no lossy decode -> re-encode round-trip)

Architecture:
  cat(h, z) -> shared MLP -> actor head (env-specific) + critic head

The actor head is swapped per-environment (different action dims), but the
shared trunk weights transfer directly across tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE


class LatentPolicyHead(nn.Module):
    """Policy operating on RSSM latent state cat(h, z).

    The shared trunk (latent_dim -> hidden -> hidden) is environment-agnostic
    and transfers across tasks. Only the actor head is env-specific.
    """

    def __init__(self, latent_dim: int, action_dim: int,
                 hidden: int = 128, discrete: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(hidden, 1)

        if discrete:
            self.actor_head = nn.Linear(hidden, action_dim)
        else:
            self.mean_head = nn.Linear(hidden, action_dim)
            self.logstd_head = nn.Linear(hidden, action_dim)
            nn.init.constant_(self.logstd_head.bias, -0.5)

    def forward(self, latent: torch.Tensor):
        """Forward pass on cat(h, z).

        Returns:
            Discrete: (logits, value)
            Continuous: (mean, logstd, value)
        """
        features = self.shared(latent)
        value = self.critic_head(features).squeeze(-1)

        if self.discrete:
            logits = self.actor_head(features)
            return logits, value
        else:
            mean = self.mean_head(features)
            logstd = self.logstd_head(features).clamp(-5.0, 2.0)
            return mean, logstd, value

    @torch.no_grad()
    def act(self, latent: torch.Tensor, deterministic: bool = True):
        """Select an action from cat(h, z) suitable for env.step().

        Discrete: returns int action index.
        Continuous: returns numpy array of shape (action_dim,).

        Mirrors the interface of `real_trainer.policy.act` (discrete: int)
        and `sac_trainer.policy.act` (continuous: ndarray) so the caller in
        `collect_episode` can swap policies without further branching.
        """
        if self.discrete:
            logits, _ = self.forward(latent)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.distributions.Categorical(probs).sample().item())

        mean, logstd, _ = self.forward(latent)
        if deterministic:
            return mean.squeeze(0).cpu().numpy()
        std = logstd.exp()
        sample = torch.distributions.Normal(mean, std).sample()
        return sample.squeeze(0).cpu().numpy()

    def get_trunk_state_dict(self) -> dict:
        """Get only the shared trunk + critic weights (transferable)."""
        trunk_keys = set()
        for name, _ in self.shared.named_parameters():
            trunk_keys.add(f"shared.{name}")
        for name, _ in self.critic_head.named_parameters():
            trunk_keys.add(f"critic_head.{name}")

        return {k: v for k, v in self.state_dict().items()
                if k in trunk_keys}

    def load_trunk_state_dict(self, state_dict: dict):
        """Load only the shared trunk + critic weights (from different env)."""
        current = self.state_dict()
        for k, v in state_dict.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
        self.load_state_dict(current)
