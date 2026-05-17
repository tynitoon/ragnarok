"""Canonical advantage / lambda-return computations.

Single source of truth for GAE (real-experience, PPO/A2C) and Dreamer-style
lambda-returns (imagination). Before consolidation (preregistration v3 §6.1
fix #2), three near-duplicate implementations lived in `real_experience.py`,
`dream_augmenter.py`, and `dreamer.py`; the two dream-side copies were
byte-identical and the real-side copy used a different algorithm with a
different `continues` convention. Benchmark numbers could not be compared
across methods without confounds from these divergences.

Two distinct functions are exposed here because they ARE different
algorithms, despite superficial similarity:

  * `compute_gae` — numpy, GAE advantage recursion, `continues = 1 - done`,
    used for on-policy real-experience training (PPO, A2C).

  * `compute_lambda_returns` — torch, Dreamer v2 lambda-return recursion,
    `continues ∈ [0, 1]` (typically `sigmoid(continue_logit)`), used for
    imagination rollouts where termination is probabilistic.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman 2016) for real experience.

    Sign convention:
      next_nonterminal = 1.0 - dones[t]
      delta_t = r_t + gamma * V(s_{t+1}) * next_nonterminal - V(s_t)
      A_t     = delta_t + gamma * lam * next_nonterminal * A_{t+1}

    Bootstrapping: at t = n-1, V(s_n) is supplied as `last_value`.

    Args:
      rewards:    shape (T,)
      values:     shape (T,) — V(s_0), ..., V(s_{T-1})
      dones:      shape (T,) — 1.0 at episode boundary, else 0.0
      last_value: scalar — V(s_T), bootstrap target at the rollout tail
      gamma, lam: discount and GAE parameter

    Returns:
      advantages: shape (T,)
      returns:    shape (T,) — advantages + values (the value-head target)

    Does NOT normalize advantages; callers that want mean-0/std-1 advantages
    must do so themselves (PPO typically does; policy-gradient variants may
    not). Keeping normalization out of this function preserves a pure,
    testable core.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values.astype(np.float32)
    return advantages, returns


def compute_gae_batched(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GAE over (N, T) device tensors — the on-device twin of `compute_gae`.

    Identical recursion and sign convention to `compute_gae`, but vectorized
    across N parallel environments and kept on the accelerator (torch, no
    numpy, no host sync). The reverse T-loop unrolls into the XLA graph (T is
    a Python constant); the N dimension stays fully batched.

    This is the advantage primitive for the device-resident rollout path
    (`ragnarok/learning/rollout.py`). Each of the N env rows is an
    independent length-T trajectory, and `dones` resets the recursion at
    every episode boundary within a row — auto-reset envs cross several
    episodes per rollout.

    Sign convention (per row, per step t):
      next_nonterminal = 1.0 - dones[:, t]
      delta_t = r_t + gamma * V(s_{t+1}) * next_nonterminal - V(s_t)
      A_t     = delta_t + gamma * lam * next_nonterminal * A_{t+1}

    Args:
      rewards:    (N, T)
      values:     (N, T) — V(s_0), ..., V(s_{T-1}) per row
      dones:      (N, T) — 1.0 at an episode boundary, else 0.0. With
                  auto-reset envs, termination and truncation are both
                  boundaries: the post-done observation belongs to a fresh
                  episode, so neither bootstraps across the seam.
      last_value: (N,) — V(s_T), the bootstrap target at each row's tail
      gamma, lam: discount and GAE parameter

    Returns:
      advantages: (N, T)
      returns:    (N, T) — advantages + values (the value-head target)

    Does NOT normalize advantages — same contract as `compute_gae`.
    """
    horizon = rewards.shape[1]
    adv_rev: list[torch.Tensor] = []
    gae = torch.zeros_like(last_value)
    for t in reversed(range(horizon)):
        next_val = last_value if t == horizon - 1 else values[:, t + 1]
        next_nonterminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_val * next_nonterminal - values[:, t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv_rev.append(gae)
    advantages = torch.stack(adv_rev[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Dreamer v2 lambda-returns for imagination rollouts (Hafner 2021).

    Recursion:
      V_lambda_t = r_t + gamma * c_t * ((1 - lam) * V(s_{t+1}) + lam * V_lambda_{t+1})

    Sign convention: `continues` is the per-step probability of NOT
    terminating, typically `sigmoid(continue_logit)` from the world model.
    This is the opposite of the `dones` mask used by `compute_gae`.

    Bootstrapping: `values[:, -1]` seeds the recursion at the horizon.

    Gradient flow: rewards and continues pass gradients through; values are
    the caller's responsibility to detach or not. In Dreamer v2, values are
    typically detached so the critic regresses onto the returns and the actor
    sees gradient only through the world model.

    Args:
      rewards:   shape (B, H)
      values:    shape (B, H+1) — V(s_0), ..., V(s_H)
      continues: shape (B, H)
      gamma, lam: discount and lambda parameter

    Returns:
      lambda_returns: shape (B, H)
    """
    horizon = rewards.shape[1]
    last = values[:, -1]

    returns_list = []
    for t in reversed(range(horizon)):
        r = rewards[:, t]
        c = continues[:, t]
        v_next = values[:, t + 1]
        last = r + gamma * c * ((1 - lam) * v_next + lam * last)
        returns_list.append(last)

    returns_list.reverse()
    return torch.stack(returns_list, dim=1)
