"""Fixed-shape on-device rollout collection for accelerator-resident envs.

Phase 2 of the TPU re-architecture. ``collect_rollout`` runs a fixed
``horizon``-step loop over a ``DeviceVec*`` env (see
``ragnarok/environments/device_env.py``), producing one ``RolloutBatch``
of device tensors with NO host sync inside the loop. N (num_envs) and
horizon are Python constants, so the whole rollout is a single XLA graph
— compiled once, then reused (and unrolled, which keeps the TPU fed).

This module only *collects*. The ``RolloutBatch`` then fans out to every
training consumer — the PPO/SAC policy trainer, the RSSM world model,
the latent policy, curiosity — which live in their own modules.
"""

from dataclasses import dataclass

import torch

from ragnarok.infrastructure.device import DEVICE, mark_step


@dataclass
class RolloutBatch:
    """One fixed-shape rollout: N parallel envs x T steps, all device tensors.

    ``obs`` is the normalized pre-step observation the policy acted on;
    ``raw_obs`` is the same step's unnormalized env state — feed it to the
    normalizer's ``update`` between rollouts. ``actions`` is the action
    taken; ``rewards``/``dones`` are the step results; ``logp``/``values``
    are the collecting policy's (frozen, no-grad) outputs — used as PPO's
    old log-probs and for GAE. ``last_obs``/``last_value`` carry the
    post-rollout state for value bootstrapping.

    With no normalizer ``obs == raw_obs``. actions shape is env-dependent:
    discrete -> (N, T) int indices; continuous -> (N, T, action_dim) float.
    """

    obs: torch.Tensor         # (N, T, obs_dim) normalized (policy acts/trains on this)
    raw_obs: torch.Tensor     # (N, T, obs_dim) raw env state (for normalizer.update)
    actions: torch.Tensor     # (N, T) or (N, T, action_dim)
    rewards: torch.Tensor     # (N, T)
    dones: torch.Tensor       # (N, T) float 0/1
    logp: torch.Tensor        # (N, T)
    values: torch.Tensor      # (N, T)
    last_obs: torch.Tensor    # (N, obs_dim)
    last_value: torch.Tensor  # (N,)

    @property
    def num_envs(self) -> int:
        return self.obs.shape[0]

    @property
    def horizon(self) -> int:
        return self.obs.shape[1]

    @property
    def total_steps(self) -> int:
        """N x T — the number of real transitions in this rollout."""
        return self.obs.shape[0] * self.obs.shape[1]


@torch.no_grad()
def collect_rollout(device_env, policy_fn, horizon: int, normalizer=None) -> RolloutBatch:
    """Collect a fixed ``horizon``-step rollout from a DeviceVec* env.

    Args:
        device_env: a ``DeviceVecCartPole`` / ``DeviceVecMountainCarContinuous``
            instance — holds an (N, obs_dim) device-resident state, and
            ``.step(action)`` is batched and auto-resets terminated envs.
        policy_fn: ``callable(obs) -> (action, logp, value)``. ``obs`` is the
            (N, obs_dim) device tensor; ``action`` is what ``device_env.step``
            expects (discrete: (N,) int or (N, action_dim) one-hot;
            continuous: (N, action_dim) float); ``logp`` and ``value`` are
            (N,) device tensors. Must run under no-grad / be cheap.
        horizon: number of steps T — a Python int constant, so the rollout
            is a single fixed-shape XLA graph.
        normalizer: optional ``DeviceRunningNormalizer``. If given, the
            policy sees normalized obs (also stored as ``batch.obs``); its
            stats stay READ-ONLY for the whole rollout — one consistent
            scaling — so call ``normalizer.update(batch.raw_obs...)``
            BETWEEN rollouts. If None, ``obs == raw_obs``.

    Returns:
        A ``RolloutBatch`` of device tensors. No ``.cpu()``/``.item()`` is
        called inside the loop, so the collection never leaves the device.
    """
    raw = device_env.state
    raw_l, obs_l, act_l, rew_l, done_l, logp_l, val_l = [], [], [], [], [], [], []

    for _ in range(horizon):
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        action, logp, value = policy_fn(obs)
        next_raw, reward, _terminated, _truncated, done = device_env.step(action)
        raw_l.append(raw)
        obs_l.append(obs)
        act_l.append(action)
        rew_l.append(reward)
        done_l.append(done.float())
        logp_l.append(logp)
        val_l.append(value)
        raw = next_raw

    # Bootstrap value for the post-rollout state (for GAE's final step).
    last_obs = normalizer.normalize(raw) if normalizer is not None else raw
    _, _, last_value = policy_fn(last_obs)
    mark_step()  # XLA: materialize the rollout graph (no-op on CUDA/CPU)

    return RolloutBatch(
        obs=torch.stack(obs_l, dim=1),
        raw_obs=torch.stack(raw_l, dim=1),
        actions=torch.stack(act_l, dim=1),
        rewards=torch.stack(rew_l, dim=1),
        dones=torch.stack(done_l, dim=1),
        logp=torch.stack(logp_l, dim=1),
        values=torch.stack(val_l, dim=1),
        last_obs=last_obs,
        last_value=last_value,
    )


@torch.no_grad()
def device_evaluate(device_env, act_fn, steps: int, normalizer=None) -> float:
    """Mean completed-episode return — greedy eval on a DeviceVec* env.

    The device-path counterpart of the gym ``evaluate()``. ``act_fn(obs) ->
    action`` is a deterministic (greedy) action function: ``obs`` is the
    (N, obs_dim) device tensor, normalized first if a normalizer is given,
    and ``action`` is what ``device_env.step`` expects. Runs ``steps`` steps
    — an env that finishes auto-resets and starts a fresh episode — and
    returns the mean return over every episode that completed in the window
    (each env truncates at least once at its step cap, so the count is
    never zero). No host sync until the final scalar read.
    """
    device_env.reset()
    n = device_env.num_envs
    ret = torch.zeros(n, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        obs = device_env.state
        if normalizer is not None:
            obs = normalizer.normalize(obs)
        _, reward, _, _, done = device_env.step(act_fn(obs))
        done = done.float()
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
    mark_step()
    return (ret_sum / ep_count.clamp(min=1.0)).item()
