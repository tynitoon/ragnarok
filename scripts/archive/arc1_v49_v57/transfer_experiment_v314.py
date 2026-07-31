"""v3.14 core-only transfer under the v3.13 mechanism: does the
env-agnostic RSSM core ALONE carry v3.13's positive transfer signal?

Preregistered as preregistration.md amendment v3.14 (committed before
this script and any run). v3.13 produced the project's first positive
transfer signal — transfer-minus-permuted AUC +15.67 +/- 15.65 (N=8),
end-of-run mastery 6/8 (transfer) vs 2/8 (permuted) vs 0/8 (scratch) —
under a full-RSSM transferable scope (encoder + core + decoder + heads).
The workshop's heterogeneous-dim claim can transfer only the
env-agnostic core (gru + prior + posterior), because the encoder is
shape-dependent on obs_dim. v3.14 is the necessary same-dim preliminary:
does the env-agnostic core ALONE carry the v3.13 effect, with the
encoder fresh-random in every arm?

Design (identical to v3.13 except the transferable scope):
  - SAC reads ONLY cat([h, z]) — no raw observation (latent load-bearing,
    inherited from v3.13).
  - Transferable subset is the env-agnostic core only (gru + prior +
    posterior). The source is the v3.11 raw-obs-DeviceAgent snapshot
    (already this subset). Encoder, decoder, reward and continue
    predictors are fresh-random in EVERY arm — transfer, permuted, and
    scratch alike.
  - Three arms: transfer (source core loaded), permuted (source core with
    each parameter tensor's elements randomly permuted), scratch (fresh
    random core). All three frozen during SAC; ICM curiosity (fresh
    instance per arm, RSSM-independent); aug-obs running normaliser over
    the 160-d latent; N=8 seeds.
  - Endpoint: per-seed AUC difference transfer-minus-permuted (and
    -minus-scratch), 3-point-smoothed, mean +/- Student-t 95% CI.

Decisive interpretation:
  - Positive => the env-agnostic core alone carries transferable
    structure; heterogeneous-dim transfer (cartpole -> mcc) becomes the
    next natural test.
  - Null => the encoder transfer was essential to v3.13's positive, and
    the heterogeneous-dim mechanism needs a different design.

Usage:  python -m scripts.transfer_experiment_v314 [--seeds N] [--smoke]
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.curiosity import DeviceForwardCuriosity
from ragnarok.learning.rollout import collect_rollout_augmented, evaluate_augmented
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuous, DeviceVecMountainCarContinuousHard,
    DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 2, 1
ARMS = ("scratch", "transfer", "permuted")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


def _permute_state_dict(state, seed):
    """The state dict with each parameter tensor's elements randomly
    permuted — per-tensor weight distribution / norm / rank preserved
    (hence the latent scale), learned structure destroyed."""
    g = torch.Generator().manual_seed(seed + 90_000)
    return {k: v.flatten()[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
            for k, v in state.items()}


def _build_arm(env_cls, cfg, arm, snapshot, seed):
    """Build one frozen-RSSM, latent-only arm. Core-only transferable
    scope: only the env-agnostic core (gru + prior + posterior) is
    loaded; encoder/decoder/heads stay fresh-random in EVERY arm."""
    env = env_cls(cfg["num_envs"])
    rssm = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    state_dim = rssm.state_dim
    sac = SACTrainer(
        obs_dim=state_dim, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    curiosity = DeviceForwardCuriosity(OBS_DIM, ACTION_DIM)
    normalizer = DeviceRunningNormalizer(OBS_DIM)
    aug_normalizer = DeviceRunningNormalizer(state_dim)

    if arm == "transfer":
        core = snapshot
    elif arm == "permuted":
        core = _permute_state_dict(snapshot, seed)
    else:                                        # scratch — fresh core
        core = None
    if core is not None:
        # v3.14: load JUST the env-agnostic core. The encoder/decoder/heads
        # of the target RSSM keep their fresh-random init in every arm.
        rssm.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in core.items()})
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    return env, rssm, sac, curiosity, normalizer, aug_normalizer


def _train_iter(env, rssm, sac, curiosity, normalizer, aug_normalizer,
                horizon, sac_updates):
    """One collect-and-train iteration (frozen RSSM, latent-only)."""
    batch = collect_rollout_augmented(env, rssm, sac, horizon,
                                      normalizer=normalizer, deterministic=True,
                                      aug_normalizer=aug_normalizer,
                                      include_obs=False)
    intrinsic = curiosity.intrinsic_reward(batch)
    sac_batch = dataclasses.replace(batch, rewards=batch.rewards + intrinsic)
    sac.train_on_rollout(sac_batch, n_updates=sac_updates,
                         obs_attr="aug_obs", last_obs_attr="last_aug")
    curiosity.train(batch)
    normalizer.update(batch.raw_obs.reshape(-1, OBS_DIM))
    aug_normalizer.update(
        batch.raw_aug_obs.reshape(-1, batch.raw_aug_obs.shape[-1]))
    return batch.total_steps


def _run_arm(arm, env_cls, cfg, snapshot, seed):
    env, rssm, sac, curiosity, normalizer, aug_normalizer = _build_arm(
        env_cls, cfg, arm, snapshot, seed)
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, rssm, sac, curiosity, normalizer,
                             aug_normalizer, cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_augmented(env_cls(cfg["num_envs"]), rssm,
                                       sac.policy, cfg["eval_steps"],
                                       normalizer=normalizer, deterministic=True,
                                       aug_normalizer=aug_normalizer,
                                       include_obs=False)
            curve.append([total, score])
            print(f"    [{arm:>8}] iter {it:>3}/{cfg['arm_rollouts']} | "
                  f"env_steps {total:>10,} | eval {score:8.2f}", flush=True)
    return curve


def _smooth(ys):
    ys = np.asarray(ys, dtype=np.float64)
    if len(ys) < 3:
        return ys
    out = ys.copy()
    out[1:-1] = (ys[:-2] + ys[1:-1] + ys[2:]) / 3.0
    out[0] = (ys[0] + ys[1]) / 2.0
    out[-1] = (ys[-1] + ys[-2]) / 2.0
    return out


def _auc(curve):
    if len(curve) < 2:
        return 0.0
    xs = np.array([c[0] for c in curve], dtype=np.float64)
    ys = _smooth([c[1] for c in curve])
    return float(_trapz(ys, xs) / (xs[-1] - xs[0]))


def _ci95(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, float("nan")
    sem = float(a.std(ddof=1) / np.sqrt(n))
    return mean, _T95.get(n - 1, 1.96) * sem


def _train_source(cfg, snap_path):
    """Source on standard MCC via the proven raw-obs DeviceAgent path;
    saves the env-agnostic core only (gru + prior + posterior). v3.14
    reuses the v3.11 source snapshot if present at `snap_path`."""
    if os.path.exists(snap_path):
        print(f"[source] cached snapshot found: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source] training on MountainCarContinuous "
          f"({cfg['source_rollouts']} rollouts, raw-obs DeviceAgent)...",
          flush=True)
    torch.manual_seed(1000)
    np.random.seed(1000)
    agent = DeviceAgent(DeviceVecMountainCarContinuous,
                        num_envs=cfg["num_envs"], horizon=cfg["horizon"],
                        sac_updates=cfg["source_sac_updates"],
                        curiosity_warmup=6)
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        agent.train_iteration()
        if it == 1 or it % cfg["eval_every"] == 0:
            score = agent.evaluate(steps=cfg["eval_steps"],
                                   n_envs=cfg["num_envs"])
            print(f"  [source] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | "
                  f"eval {score:8.2f}", flush=True)
    snap = {k: v.detach().cpu()
            for k, v in agent.rssm.transferable_state_dict().items()}
    torch.save(snap, snap_path)
    print(f"[source] done in {time.perf_counter() - t0:.0f}s, "
          f"core-only snapshot ({len(snap)} tensors) -> {snap_path}",
          flush=True)
    return snap


def _summarize(results):
    d_perm = [r["diff_vs_permuted"] for r in results]
    d_scr = [r["diff_vs_scratch"] for r in results]
    mp, cp = _ci95(d_perm)
    ms, cs = _ci95(d_scr)
    print(f"\n{'=' * 64}")
    print(f"  N={len(results)} seeds")
    print(f"  transfer - permuted  AUC diff: mean {mp:+.2f}  95%CI +/-{cp:.2f}"
          f"   ({sum(d > 0 for d in d_perm)}/{len(d_perm)} seeds > 0)")
    print(f"  transfer - scratch   AUC diff: mean {ms:+.2f}  95%CI +/-{cs:.2f}"
          f"   ({sum(d > 0 for d in d_scr)}/{len(d_scr)} seeds > 0)")
    if len(results) >= 2 and mp - cp > 0:
        verdict = ("transfer ACCELERATES via learned structure "
                   "(transfer > permuted, CI excludes 0)")
    elif len(results) >= 2 and mp + cp < 0:
        verdict = "transfer HURTS vs the structure control"
    else:
        verdict = ("no structural transfer effect resolved "
                   "(transfer-permuted CI spans 0)")
    print(f"  -> {verdict}")
    return {"transfer_minus_permuted": {"mean": mp, "ci95": cp},
            "transfer_minus_scratch": {"mean": ms, "ci95": cs},
            "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--source-rollouts", type=int, default=50)
    parser.add_argument("--arm-rollouts", type=int, default=60)
    parser.add_argument("--sac-updates", type=int, default=256)
    parser.add_argument("--source-sac-updates", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=999)
    parser.add_argument("--out-dir", default="transfer_v314_out")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end smoke (1 seed, minimal budget)")
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.source_rollouts, args.arm_rollouts = 1, 4, 5
        args.sac_updates, args.source_sac_updates = 8, 8
        args.num_envs, args.horizon = 64, 32
        args.eval_every, args.eval_steps = 2, 300

    cfg = {"num_envs": args.num_envs, "horizon": args.horizon,
           "sac_updates": args.sac_updates,
           "source_sac_updates": args.source_sac_updates,
           "source_rollouts": args.source_rollouts,
           "arm_rollouts": args.arm_rollouts, "eval_every": args.eval_every,
           "eval_steps": args.eval_steps}

    os.makedirs(args.out_dir, exist_ok=True)
    snap_path = os.path.join(args.out_dir, "source_core.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v314] device={DEVICE}  seeds={args.seeds}  "
          f"source_rollouts={args.source_rollouts}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}",
          flush=True)
    t0 = time.perf_counter()

    snapshot = _train_source(cfg, snap_path)

    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f).get("seeds", [])
        print(f"[resume] {len(results)} seed(s) already done", flush=True)
    done_seeds = {r["seed"] for r in results}

    for seed in range(args.seeds):
        if seed in done_seeds:
            print(f"[seed {seed}] already done — skip", flush=True)
            continue
        print(f"\n[seed {seed}]", flush=True)
        s0 = time.perf_counter()
        curves = {}
        for arm in ARMS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            curves[arm] = _run_arm(arm, DeviceVecMountainCarContinuousHard,
                                   cfg, snapshot, seed)
        auc = {arm: _auc(curves[arm]) for arm in ARMS}
        rec = {"seed": seed,
               "scratch_curve": curves["scratch"],
               "transfer_curve": curves["transfer"],
               "permuted_curve": curves["permuted"],
               "scratch_auc": auc["scratch"],
               "transfer_auc": auc["transfer"],
               "permuted_auc": auc["permuted"],
               "diff_vs_scratch": auc["transfer"] - auc["scratch"],
               "diff_vs_permuted": auc["transfer"] - auc["permuted"]}
        results.append(rec)
        with open(results_path, "w") as f:
            json.dump({"seeds": results}, f, indent=2)
        print(f"  seed {seed} | AUC  scratch {auc['scratch']:.2f}  "
              f"permuted {auc['permuted']:.2f}  transfer {auc['transfer']:.2f} "
              f"| diff vs permuted {rec['diff_vs_permuted']:+.2f} "
              f"| {time.perf_counter() - s0:.0f}s", flush=True)

    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
