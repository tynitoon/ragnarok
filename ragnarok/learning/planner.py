"""Model-predictive planning in a learned RSSM world model (v4.0).

The v4 pivot makes the world model load-bearing for control: instead of
learning a policy from real experience, the agent PLANS in its learned
world model. CEM-MPC (cross-entropy method, receding horizon) samples
action sequences, rolls them through the FROZEN RSSM, scores the
imagined trajectories with a task reward computed on the DECODED
predicted observations, and executes the first action of the best plan.

This is the substrate for "understanding the world makes new goals
cheap": one trained dynamics model solves any goal by planning, with no
new policy/representation learning — the reward function is the only
thing that changes per task, and for goal-reaching it is a known
analytic function of the decoded state (no learning at all).
"""

import torch

from ragnarok.infrastructure.device import DEVICE


@torch.no_grad()
def cem_plan(rssm, h, z, reward_fn, horizon=15, n_cand=256, n_elite=32,
             n_iters=4, action_dim=2, action_low=-1.0, action_high=1.0,
             init_mu=None):
    """Batched CEM-MPC plan over a learned RSSM, for B envs at once.

    Args:
        rssm: a (frozen) RSSM world model.
        h, z: (B, hidden) / (B, stoch) current latent state per env.
        reward_fn: callable(decoded_obs (M, obs_dim)) -> (M,) per-step
            reward. Captures the task/goal; for point-mass goal-reaching
            this is -distance(decoded_xy, goal).
        horizon: planning horizon H (receding).
        n_cand / n_elite / n_iters: CEM population, elite count, refit
            iterations.
        action_dim, action_low, action_high: action space.
        init_mu: optional (B, H, action_dim) warm-start mean (the
            previous step's plan shifted) for MPC continuity.

    Returns:
        (first_action (B, action_dim), plan_mu (B, H, action_dim)) — the
        action to execute now and the full plan (for next-step warm
        start).
    """
    B = h.shape[0]
    mu = (torch.zeros(B, horizon, action_dim, device=DEVICE)
          if init_mu is None else init_mu.clone())
    std = torch.full((B, horizon, action_dim), 0.5, device=DEVICE)

    for _ in range(n_iters):
        eps = torch.randn(B, n_cand, horizon, action_dim, device=DEVICE)
        cand = (mu.unsqueeze(1) + std.unsqueeze(1) * eps).clamp(action_low,
                                                                action_high)
        cf = cand.reshape(B * n_cand, horizon, action_dim)

        hh = h.unsqueeze(1).expand(B, n_cand, -1).reshape(B * n_cand, -1)
        zz = z.unsqueeze(1).expand(B, n_cand, -1).reshape(B * n_cand, -1)
        total = torch.zeros(B * n_cand, device=DEVICE)
        for t in range(horizon):
            a = cf[:, t]
            hh = rssm.core.step(hh, zz, a)
            pm, pl = rssm.core.forward_prior(hh)
            zz = rssm.core.sample(pm, pl)
            obs_pred = rssm.decoder(torch.cat([hh, zz], dim=-1))
            total = total + reward_fn(obs_pred)
        scores = total.reshape(B, n_cand)

        elite_idx = scores.topk(n_elite, dim=1).indices               # (B, E)
        gather_idx = elite_idx.view(B, n_elite, 1, 1).expand(
            B, n_elite, horizon, action_dim)
        elite = torch.gather(cand, 1, gather_idx)                     # (B,E,H,A)
        mu = elite.mean(dim=1)
        std = elite.std(dim=1).clamp(min=1e-3)

    return mu[:, 0], mu
