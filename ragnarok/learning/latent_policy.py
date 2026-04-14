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
                 hidden: int = 128, discrete: bool = True,
                 action_low: np.ndarray | None = None,
                 action_high: np.ndarray | None = None):
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
            # Match SACPolicy (-1.0) so latent/obs continuous trainers start
            # from the same exploration scale and H1 comparisons aren't
            # confounded by an inherited -0.5 divergence.
            nn.init.constant_(self.logstd_head.bias, -1.0)
            # Action bounds for tanh-squash + rescale (mirrors SACPolicy).
            # Without this, `act()` emits raw Gaussian samples that violate
            # env.action_space on every continuous target — silent failure.
            if action_low is not None and action_high is not None:
                self.register_buffer(
                    "action_low",
                    torch.as_tensor(action_low, dtype=torch.float32))
                self.register_buffer(
                    "action_high",
                    torch.as_tensor(action_high, dtype=torch.float32))
            else:
                self.register_buffer("action_low", -torch.ones(action_dim))
                self.register_buffer("action_high", torch.ones(action_dim))

    def _rescale(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (tanh_action + 1.0) * 0.5 * (
            self.action_high - self.action_low)

    def _inverse_rescale(self, env_action: torch.Tensor) -> torch.Tensor:
        """Inverse of `_rescale` — env-space action back to tanh range (-1, 1)."""
        return 2.0 * (env_action - self.action_low) / (
            self.action_high - self.action_low) - 1.0

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
            action = self._rescale(torch.tanh(mean))
        else:
            std = logstd.exp()
            raw = torch.distributions.Normal(mean, std).sample()
            action = self._rescale(torch.tanh(raw))
        return action.squeeze(0).cpu().numpy()

    def evaluate_action(self, latent: torch.Tensor, action: torch.Tensor):
        """Compute (log_prob, entropy, value) for an already-taken action.

        For continuous, the stored `action` is env-space (post-tanh+rescale
        from `act()`). Naively evaluating log_prob of the raw Gaussian on
        that action would be a train/inference distribution mismatch — the
        policy would train on the wrong density. This method inverts the
        squash and adds the tanh log-det correction (cf. SACPolicy.sample
        L82-85), so the log-prob reflects the *actual* distribution the
        policy sampled from.

        The affine `_rescale` Jacobian is a constant w.r.t. policy params
        and cancels in the gradient, so we drop it — mirrors SAC.

        Returns:
            (log_prob[B], entropy[scalar], value[B])
        """
        if self.discrete:
            logits, value = self.forward(latent)
            dist = torch.distributions.Categorical(logits=logits)
            idx = action.argmax(dim=-1) if action.dim() == 2 else action.long()
            return dist.log_prob(idx), dist.entropy().mean(), value

        mean, logstd, value = self.forward(latent)
        std = logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        # Invert rescale+tanh to recover the pre-squash sample point.
        tanh_a = self._inverse_rescale(action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(tanh_a)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        # Tanh log-det correction: log|d/dx tanh(x)| = log(1 - tanh(x)^2)
        log_prob = log_prob - torch.log(1.0 - tanh_a.pow(2) + 1e-6).sum(dim=-1)
        # Gaussian entropy as exploration bonus proxy (standard in squashed-
        # Gaussian actors — the exact squashed entropy has no closed form).
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_prob, entropy, value

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
