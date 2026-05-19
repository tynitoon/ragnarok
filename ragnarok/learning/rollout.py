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
import torch.nn.functional as F

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
    # v3.10 corrected-transfer mechanism: the augmented observation
    # [obs, h, z] (raw normalized obs ++ RSSM latent state) that SAC
    # acts/trains on. Set only by collect_rollout_augmented; None for the
    # plain/latent collectors, so existing constructor calls stay valid.
    aug_obs: torch.Tensor | None = None    # (N, T, obs_dim + state_dim)
    last_aug: torch.Tensor | None = None   # (N, obs_dim + state_dim)
    # v3.12: the raw (pre-normalization) augmented observation — feed it to
    # the aug-obs normalizer's update() between rollouts. Set by
    # collect_rollout_augmented only when an aug_normalizer is in use.
    raw_aug_obs: torch.Tensor | None = None  # (N, T, obs_dim + state_dim)

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


@torch.no_grad()
def collect_rollout_latent(device_env, rssm, latent_head, horizon: int,
                           normalizer=None) -> RolloutBatch:
    """Collect a rollout where the latent policy acts on RSSM state cat(h, z).

    The post-transfer acting path. Unlike ``collect_rollout`` (a stateless
    ``policy_fn`` on raw obs), the latent policy consumes the RSSM recurrent
    state, so (h, z) and the preceding action are threaded across the horizon
    and zeroed at every episode seam — a rollout row spans several auto-reset
    episodes and the GRU must not bridge them (mirrors ``observe``'s done
    reset).

    Returns the same ``RolloutBatch`` shape as ``collect_rollout``, so the
    world-model / latent-policy / SAC trainers consume it unchanged.
    """
    n = device_env.num_envs
    h, z = rssm.initial_state(n, DEVICE)
    prev_action = torch.zeros(n, rssm.action_dim, device=DEVICE)
    raw = device_env.state
    raw_l, obs_l, act_l, rew_l, done_l, logp_l, val_l = [], [], [], [], [], [], []

    for _ in range(horizon):
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        h, z = rssm.encode_observation(obs, h, z, prev_action)
        action, logp, value = latent_head.device_sample(torch.cat([h, z], dim=-1))
        next_raw, reward, _terminated, _truncated, done = device_env.step(action)
        raw_l.append(raw)
        obs_l.append(obs)
        act_l.append(action)
        rew_l.append(reward)
        done_l.append(done.float())
        logp_l.append(logp)
        val_l.append(value)
        # Thread the RSSM state; zero (h, z) and the preceding action at
        # episode seams — the post-done obs belongs to a fresh episode.
        keep = (1.0 - done.float()).unsqueeze(-1)
        act_oh = (F.one_hot(action.long(), rssm.action_dim).float()
                  if action.dim() == 1 else action)
        h = h * keep
        z = z * keep
        prev_action = act_oh * keep
        raw = next_raw

    last_obs = normalizer.normalize(raw) if normalizer is not None else raw
    h, z = rssm.encode_observation(last_obs, h, z, prev_action)
    _, _, last_value = latent_head.device_sample(torch.cat([h, z], dim=-1))
    mark_step()

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
def device_evaluate_latent(device_env, rssm, latent_head, steps: int,
                           normalizer=None) -> float:
    """Greedy latent-policy eval — mean completed-episode return.

    The latent-acting counterpart of ``device_evaluate``: it threads the RSSM
    recurrent state (h, z) and acts greedily via the latent policy on
    cat(h, z), resetting the state at episode seams.
    """
    device_env.reset()
    n = device_env.num_envs
    h, z = rssm.initial_state(n, DEVICE)
    prev_action = torch.zeros(n, rssm.action_dim, device=DEVICE)
    ret = torch.zeros(n, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        raw = device_env.state
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        h, z = rssm.encode_observation(obs, h, z, prev_action)
        action = latent_head.device_act(torch.cat([h, z], dim=-1))
        _, reward, _t, _tr, done = device_env.step(action)
        done = done.float()
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
        keep = (1.0 - done).unsqueeze(-1)
        act_oh = (F.one_hot(action.long(), rssm.action_dim).float()
                  if action.dim() == 1 else action)
        h = h * keep
        z = z * keep
        prev_action = act_oh * keep
    mark_step()
    return (ret_sum / ep_count.clamp(min=1.0)).item()


@torch.no_grad()
def collect_rollout_augmented(device_env, rssm, sac_trainer, horizon: int,
                              normalizer=None, deterministic: bool = False,
                              aug_normalizer=None) -> RolloutBatch:
    """Collect a rollout where SAC acts on the augmented obs [obs, h, z].

    The v3.10 corrected-transfer collection. The RSSM recurrent state is
    threaded across the horizon (and zeroed at episode seams, like
    ``collect_rollout_latent``); each step the SAC policy acts on
    cat(obs, h, z) — the raw normalized observation concatenated with the
    RSSM latent state. The returned ``RolloutBatch`` carries the raw obs
    in ``obs`` (so ``WorldModelTrainer`` is consumed unchanged — it trains
    the RSSM on raw obs) AND the augmented observation in ``aug_obs`` /
    ``last_aug`` (so SAC trains on the transferred representation).
    ``logp``/``values`` are zeros — SAC recomputes everything from its
    replay buffer.

    NB on representation stationarity: SAC is off-policy, so ``aug_obs``
    is replayed many iterations after collection. If the RSSM is trained
    concurrently the stored (h, z) become stale relative to the current
    RSSM and the off-policy critic is regressed across an inconsistent
    representation — the v3.10 failure mode. v3.11 freezes the RSSM for
    the SAC arm, which makes ``aug_obs`` a stationary function of the
    observation history; pass ``deterministic=True`` so z is the
    posterior mean (no per-step sampling noise) for that frozen use.

    v3.12 ``aug_normalizer``: an optional ``DeviceRunningNormalizer`` over
    the full augmented vector. The raw [h, z] block has an arm-dependent
    scale (a trained vs a random RSSM core emit differently-scaled
    latents); feeding that to SAC's plain MLP un-normalised is a scaling
    confound. When given, SAC acts/trains on ``aug_normalizer.normalize``
    of cat(obs, h, z); the unnormalized vector is returned in
    ``raw_aug_obs`` so the caller can ``aug_normalizer.update`` it BETWEEN
    rollouts (the obs-normalizer contract).
    """
    n = device_env.num_envs
    h, z = rssm.initial_state(n, DEVICE)
    prev_action = torch.zeros(n, rssm.action_dim, device=DEVICE)
    raw = device_env.state
    raw_l, obs_l, aug_l, raw_aug_l, act_l, rew_l, done_l = (
        [], [], [], [], [], [], [])

    for _ in range(horizon):
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        h, z = rssm.encode_observation(obs, h, z, prev_action,
                                       deterministic=deterministic)
        aug_raw = torch.cat([obs, h, z], dim=-1)
        aug = (aug_normalizer.normalize(aug_raw)
               if aug_normalizer is not None else aug_raw)
        action, _, _ = sac_trainer.device_policy_fn(aug)
        next_raw, reward, _terminated, _truncated, done = device_env.step(action)
        raw_l.append(raw)
        obs_l.append(obs)
        aug_l.append(aug)
        raw_aug_l.append(aug_raw)
        act_l.append(action)
        rew_l.append(reward)
        done_l.append(done.float())
        # Thread the RSSM state; zero (h, z) and the preceding action at
        # episode seams — the post-done obs belongs to a fresh episode.
        keep = (1.0 - done.float()).unsqueeze(-1)
        act_oh = (F.one_hot(action.long(), rssm.action_dim).float()
                  if action.dim() == 1 else action)
        h = h * keep
        z = z * keep
        prev_action = act_oh * keep
        raw = next_raw

    last_obs = normalizer.normalize(raw) if normalizer is not None else raw
    h, z = rssm.encode_observation(last_obs, h, z, prev_action,
                                   deterministic=deterministic)
    last_aug_raw = torch.cat([last_obs, h, z], dim=-1)
    last_aug = (aug_normalizer.normalize(last_aug_raw)
                if aug_normalizer is not None else last_aug_raw)
    mark_step()

    zeros = torch.zeros(n, horizon, device=DEVICE)
    return RolloutBatch(
        obs=torch.stack(obs_l, dim=1),
        raw_obs=torch.stack(raw_l, dim=1),
        actions=torch.stack(act_l, dim=1),
        rewards=torch.stack(rew_l, dim=1),
        dones=torch.stack(done_l, dim=1),
        logp=zeros,
        values=zeros,
        last_obs=last_obs,
        last_value=torch.zeros(n, device=DEVICE),
        aug_obs=torch.stack(aug_l, dim=1),
        last_aug=last_aug,
        raw_aug_obs=torch.stack(raw_aug_l, dim=1),
    )


@torch.no_grad()
def evaluate_augmented(device_env, rssm, sac_policy, steps: int,
                       normalizer=None, deterministic: bool = False,
                       aug_normalizer=None) -> float:
    """Greedy SAC-on-[obs,h,z] eval — mean completed-episode return.

    The v3.10 counterpart of ``device_evaluate``: threads the RSSM state
    and acts greedily via the SAC policy on the augmented observation
    cat(obs, h, z). ``aug_normalizer`` (v3.12), when given, normalizes the
    augmented vector exactly as in ``collect_rollout_augmented`` so eval
    and training feed SAC the same scaling.
    """
    device_env.reset()
    n = device_env.num_envs
    h, z = rssm.initial_state(n, DEVICE)
    prev_action = torch.zeros(n, rssm.action_dim, device=DEVICE)
    ret = torch.zeros(n, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        raw = device_env.state
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        h, z = rssm.encode_observation(obs, h, z, prev_action,
                                       deterministic=deterministic)
        aug = torch.cat([obs, h, z], dim=-1)
        if aug_normalizer is not None:
            aug = aug_normalizer.normalize(aug)
        mean, _ = sac_policy.forward(aug)
        action = sac_policy._rescale(torch.tanh(mean))
        _, reward, _t, _tr, done = device_env.step(action)
        done = done.float()
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
        keep = (1.0 - done).unsqueeze(-1)
        act_oh = (F.one_hot(action.long(), rssm.action_dim).float()
                  if action.dim() == 1 else action)
        h = h * keep
        z = z * keep
        prev_action = act_oh * keep
    mark_step()
    return (ret_sum / ep_count.clamp(min=1.0)).item()
