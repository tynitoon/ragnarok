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

from ragnarok.infrastructure.device import mark_step


@dataclass
class RolloutBatch:
    """One fixed-shape rollout: N parallel envs x T steps, all device tensors.

    ``obs``/``actions`` are the pre-step observation and the action taken;
    ``rewards``/``dones`` are the step results; ``logp``/``values`` are the
    collecting policy's (frozen, no-grad) outputs — used as PPO's old
    log-probs and for GAE. ``last_obs``/``last_value`` carry the
    post-rollout state for value bootstrapping.

    actions shape is env-dependent: discrete -> (N, T) int indices;
    continuous -> (N, T, action_dim) float.
    """

    obs: torch.Tensor         # (N, T, obs_dim)
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
def collect_rollout(device_env, policy_fn, horizon: int) -> RolloutBatch:
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

    Returns:
        A ``RolloutBatch`` of device tensors. No ``.cpu()``/``.item()`` is
        called inside the loop, so the collection never leaves the device.
    """
    obs = device_env.state
    obs_l, act_l, rew_l, done_l, logp_l, val_l = [], [], [], [], [], []

    for _ in range(horizon):
        action, logp, value = policy_fn(obs)
        next_obs, reward, _terminated, _truncated, done = device_env.step(action)
        obs_l.append(obs)
        act_l.append(action)
        rew_l.append(reward)
        done_l.append(done.float())
        logp_l.append(logp)
        val_l.append(value)
        obs = next_obs

    # Bootstrap value for the post-rollout state (for GAE's final step).
    _, _, last_value = policy_fn(obs)
    mark_step()  # XLA: materialize the rollout graph (no-op on CUDA/CPU)

    return RolloutBatch(
        obs=torch.stack(obs_l, dim=1),
        actions=torch.stack(act_l, dim=1),
        rewards=torch.stack(rew_l, dim=1),
        dones=torch.stack(done_l, dim=1),
        logp=torch.stack(logp_l, dim=1),
        values=torch.stack(val_l, dim=1),
        last_obs=obs,
        last_value=last_value,
    )
