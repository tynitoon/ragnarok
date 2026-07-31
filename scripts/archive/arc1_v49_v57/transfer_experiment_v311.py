"""v3.11 frozen-representation transfer experiment: does a transferred
world-model representation accelerate SAC?

Preregistered as preregistration.md amendment v3.11 (committed before this
script and any run). It corrects the v3.10 design, whose source training
failed: SAC reading the augmented observation [obs, h, z] never learned
MCC, because the RSSM trained concurrently and SAC — being off-policy —
replayed a non-stationary representation out of its buffer.

The v3.11 mechanism, implemented and tested here:

  - SOURCE: a full agent trains on standard MountainCarContinuous via the
    proven raw-obs DeviceAgent path (SAC reads the raw obs; the RSSM
    trains concurrently). Its env-agnostic RSSM core is snapshotted.
  - Both target arms (scratch, transfer) train MountainCarContinuous-Hard
    with SAC reading the AUGMENTED observation [obs, h, z] — but the RSSM
    is FROZEN, so [obs, h, z] is a stationary function of the observation
    history and SAC's replay buffer is representation-consistent.
  - The ONLY difference between the arms is the frozen RSSM core: the
    transfer arm loads the source snapshot; the scratch arm stays fresh.
  - Curiosity is a fresh forward-prediction (ICM) module, identical in
    both arms — RSSM-independent, so it is a controlled constant and the
    only inter-arm difference stays the frozen RSSM core.
  - Endpoint: the sample-efficiency AUC of the eval-return-vs-env-steps
    curve, transfer vs scratch.

Both arms for a given seed start from the same RNG, so the transferred
RSSM core is the only thing that differs between them.

Usage:  python -m scripts.transfer_experiment_v311 [--seeds N] [--smoke]
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
_trapz = getattr(np, "trapezoid", np.trapz)   # numpy 2.x renamed trapz


def _build_arm(env_cls, cfg, snapshot):
    """Build one frozen-RSSM arm — env, FROZEN RSSM, augmented-obs SAC,
    ICM curiosity, normalizer."""
    env = env_cls(cfg["num_envs"])
    rssm = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    if snapshot is not None:                          # transfer arm
        rssm.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in snapshot.items()})
    # v3.11: FREEZE the RSSM. A stationary [obs, h, z] is what makes SAC's
    # off-policy replay buffer representation-consistent (the v3.10 fix).
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    sac = SACTrainer(
        obs_dim=OBS_DIM + rssm.state_dim, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    # ICM curiosity — RSSM-independent (see DeviceForwardCuriosity): a
    # controlled constant across arms, so the only inter-arm difference
    # stays the frozen RSSM core.
    curiosity = DeviceForwardCuriosity(OBS_DIM, ACTION_DIM)
    normalizer = DeviceRunningNormalizer(OBS_DIM)
    return env, rssm, sac, curiosity, normalizer


def _train_iter(env, rssm, sac, curiosity, normalizer, horizon, sac_updates):
    """One collect-and-train iteration (frozen RSSM); return env-steps.

    SAC reads the augmented [obs, h, z] (z = posterior mean, deterministic
    — the frozen representation carries no per-step sampling noise). MCC's
    reward is sparse; the ICM intrinsic reward supplies the exploration
    drive and augments only SAC's reward. The RSSM is frozen — not trained.
    """
    batch = collect_rollout_augmented(env, rssm, sac, horizon,
                                      normalizer=normalizer, deterministic=True)
    intrinsic = curiosity.intrinsic_reward(batch)
    sac_batch = dataclasses.replace(batch, rewards=batch.rewards + intrinsic)
    sac.train_on_rollout(sac_batch, n_updates=sac_updates,
                         obs_attr="aug_obs", last_obs_attr="last_aug")
    curiosity.train(batch)
    normalizer.update(batch.raw_obs.reshape(-1, OBS_DIM))
    return batch.total_steps


def _run_arm(arm, env_cls, cfg, snapshot=None):
    """Train one arm (scratch or transfer); return the eval curve
    [[env_steps, eval_return], ...]."""
    env, rssm, sac, curiosity, normalizer = _build_arm(env_cls, cfg, snapshot)
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, rssm, sac, curiosity, normalizer,
                             cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_augmented(env_cls(cfg["num_envs"]), rssm,
                                       sac.policy, cfg["eval_steps"],
                                       normalizer=normalizer, deterministic=True)
            curve.append([total, score])
            print(f"    [{arm}] iter {it:>3}/{cfg['arm_rollouts']} | "
                  f"env_steps {total:>10,} | eval {score:8.2f}", flush=True)
    return curve


def _auc(curve):
    """Normalised sample-efficiency AUC — the mean eval return over the
    env-step budget (area under return-vs-env_steps / step span)."""
    if len(curve) < 2:
        return 0.0
    xs = np.array([c[0] for c in curve], dtype=np.float64)
    ys = np.array([c[1] for c in curve], dtype=np.float64)
    return float(_trapz(ys, xs) / (xs[-1] - xs[0]))


def _train_source(cfg, snap_path):
    """Train the source agent on standard MCC via the proven raw-obs
    DeviceAgent path; return + cache its transferable RSSM core."""
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
    snap = agent.snapshot()["rssm_core"]              # CPU tensors
    torch.save(snap, snap_path)
    print(f"[source] done in {time.perf_counter() - t0:.0f}s, "
          f"snapshot -> {snap_path}", flush=True)
    return snap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--source-rollouts", type=int, default=50)
    parser.add_argument("--arm-rollouts", type=int, default=60)
    parser.add_argument("--sac-updates", type=int, default=256,
                        help="SAC updates per rollout on the target arms")
    parser.add_argument("--source-sac-updates", type=int, default=512,
                        help="SAC updates per rollout for the source agent "
                             "(the proven raw-obs DeviceAgent path)")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=999)
    parser.add_argument("--out-dir", default="transfer_v311_out")
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
    snap_path = os.path.join(args.out_dir, "source_snapshot.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v311] device={DEVICE}  seeds={args.seeds}  "
          f"source_rollouts={args.source_rollouts}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}",
          flush=True)
    t0 = time.perf_counter()

    snapshot = _train_source(cfg, snap_path)

    # Resume: load any already-completed seeds.
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
        # Both arms start from the SAME RNG — the frozen RSSM core is then
        # the only thing that differs between them.
        torch.manual_seed(seed)
        np.random.seed(seed)
        scratch = _run_arm("scratch", DeviceVecMountainCarContinuousHard, cfg)
        torch.manual_seed(seed)
        np.random.seed(seed)
        transfer = _run_arm("transfer", DeviceVecMountainCarContinuousHard,
                            cfg, snapshot=snapshot)
        sa, ta = _auc(scratch), _auc(transfer)
        results.append({
            "seed": seed,
            "scratch_curve": scratch, "transfer_curve": transfer,
            "scratch_auc": sa, "transfer_auc": ta,
            "auc_ratio": (ta / sa) if sa else None,
        })
        with open(results_path, "w") as f:
            json.dump({"seeds": results}, f, indent=2)
        ratio = (ta / sa) if sa else float("nan")
        print(f"  seed {seed} | scratch AUC {sa:.2f} | transfer AUC {ta:.2f} "
              f"| ratio {ratio:.3f} | {time.perf_counter() - s0:.0f}s",
              flush=True)

    # Summary.
    ratios = [r["auc_ratio"] for r in results if r["auc_ratio"]]
    print(f"\n{'=' * 60}")
    if ratios:
        med = float(np.median(ratios))
        verdict = ("transfer ACCELERATES learning" if med > 1.05
                   else "no transfer effect" if med > 0.95
                   else "transfer HURTS / negative")
        print(f"  transfer/scratch AUC ratio: median {med:.3f}  "
              f"(range {min(ratios):.3f}-{max(ratios):.3f}, N={len(ratios)})")
        print(f"  -> {verdict}")
    else:
        print("  no usable seeds (scratch AUC was 0)")
    print(f"  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
