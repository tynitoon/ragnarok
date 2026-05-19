"""v3.10 corrected-transfer experiment: does a transferred world-model
representation accelerate SAC?

Preregistered as preregistration.md amendment v3.10 (SHA 16e4d40). The
mechanism implemented and tested here:

  - Both arms (scratch, transfer) train MountainCarContinuous-Hard with
    the SAME learner: SAC, reading the AUGMENTED observation [obs, h, z]
    (raw obs concatenated with the RSSM latent state).
  - The ONLY difference between the arms is the RSSM env-agnostic core's
    initialisation: the transfer arm warm-starts it from a source skill
    (standard MountainCarContinuous); the scratch arm starts fresh.
  - Source and target share a dynamics family (MCC vs a weaker-engine
    MCC-Hard), so the world-model core genuinely transfers.
  - Endpoint: the sample-efficiency AUC of the eval-return-vs-env-steps
    curve, transfer vs scratch.

Both arms for a given seed start from the same RNG, so the transferred
RSSM core is the only thing that differs between them.

Usage:  python -m scripts.transfer_experiment_v310 [--seeds N] [--smoke]
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
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.curiosity import DeviceLatentCuriosity
from ragnarok.learning.rollout import collect_rollout_augmented, evaluate_augmented
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuous, DeviceVecMountainCarContinuousHard,
    DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 2, 1
_trapz = getattr(np, "trapezoid", np.trapz)   # numpy 2.x renamed trapz


def _build(env_cls, num_envs, horizon):
    """Build one agent's components — env, RSSM, WM trainer, SAC, curiosity,
    normalizer. SAC reads the augmented [obs, h, z] observation."""
    env = env_cls(num_envs)
    rssm = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    # lr 3e-5: the device-WM-stability rate (commit a0bbdb9).
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=3e-5)
    sac = SACTrainer(
        obs_dim=OBS_DIM + rssm.state_dim, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=num_envs * horizon,
        buffer=DeviceSACBuffer(capacity=200_000))
    curiosity = DeviceLatentCuriosity(rssm, warmup=6)
    normalizer = DeviceRunningNormalizer(OBS_DIM)
    return env, rssm, wm, sac, curiosity, normalizer


def _train_iter(env, rssm, wm, sac, curiosity, normalizer, horizon, sac_updates):
    """One collect-and-train iteration; return env-steps collected."""
    batch = collect_rollout_augmented(env, rssm, sac, horizon,
                                      normalizer=normalizer)
    # MCC's reward is sparse — curiosity (RSSM latent KL) supplies the
    # exploration drive. It augments only SAC's reward; the world model
    # trains on the raw env reward.
    intrinsic = curiosity.intrinsic_reward(batch)
    sac_batch = dataclasses.replace(batch, rewards=batch.rewards + intrinsic)
    sac.train_on_rollout(sac_batch, n_updates=sac_updates,
                         obs_attr="aug_obs", last_obs_attr="last_aug")
    wm.train_world_model_on_rollout(batch)
    normalizer.update(batch.raw_obs.reshape(-1, OBS_DIM))
    wm.step_episode()
    return batch.total_steps


def _run_arm(arm, env_cls, cfg, snapshot=None):
    """Train one arm (scratch or transfer); return the eval curve
    [[env_steps, eval_return], ...]."""
    env, rssm, wm, sac, curiosity, normalizer = _build(
        env_cls, cfg["num_envs"], cfg["horizon"])
    if snapshot is not None:
        # Transfer arm: warm-start the env-agnostic RSSM core, and slow its
        # LR for a warmup window so the transferred priors are not wiped
        # before SAC can exploit them.
        rssm.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in snapshot.items()})
        wm.reset_transferable_optimizer_state()
        wm.set_transferable_lr_scale(
            0.1, warmup_episodes=max(1, cfg["arm_rollouts"] // 4))
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, rssm, wm, sac, curiosity, normalizer,
                             cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_augmented(env_cls(cfg["num_envs"]), rssm,
                                       sac.policy, cfg["eval_steps"],
                                       normalizer=normalizer)
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
    """Train the source agent on standard MCC; return + cache its snapshot."""
    if os.path.exists(snap_path):
        print(f"[source] cached snapshot found: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source] training on MountainCarContinuous "
          f"({cfg['source_rollouts']} rollouts)...", flush=True)
    torch.manual_seed(1000)
    np.random.seed(1000)
    env, rssm, wm, sac, curiosity, normalizer = _build(
        DeviceVecMountainCarContinuous, cfg["num_envs"], cfg["horizon"])
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        _train_iter(env, rssm, wm, sac, curiosity, normalizer,
                    cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_augmented(
                DeviceVecMountainCarContinuous(cfg["num_envs"]), rssm,
                sac.policy, cfg["eval_steps"], normalizer=normalizer)
            print(f"  [source] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"eval {score:8.2f}", flush=True)
    snap = {k: v.detach().cpu()
            for k, v in rssm.transferable_state_dict().items()}
    torch.save(snap, snap_path)
    print(f"[source] done in {time.perf_counter() - t0:.0f}s, "
          f"snapshot -> {snap_path}", flush=True)
    return snap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--source-rollouts", type=int, default=60)
    parser.add_argument("--arm-rollouts", type=int, default=50)
    parser.add_argument("--sac-updates", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=999)
    parser.add_argument("--out-dir", default="transfer_v310_out")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end smoke (1 seed, minimal budget)")
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.source_rollouts, args.arm_rollouts = 1, 4, 5
        args.sac_updates, args.num_envs, args.horizon = 8, 64, 32
        args.eval_every, args.eval_steps = 2, 300

    cfg = {"num_envs": args.num_envs, "horizon": args.horizon,
           "sac_updates": args.sac_updates,
           "source_rollouts": args.source_rollouts,
           "arm_rollouts": args.arm_rollouts, "eval_every": args.eval_every,
           "eval_steps": args.eval_steps}

    os.makedirs(args.out_dir, exist_ok=True)
    snap_path = os.path.join(args.out_dir, "source_snapshot.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v310] device={DEVICE}  seeds={args.seeds}  "
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
        # Both arms start from the SAME RNG — the transferred RSSM core is
        # then the only thing that differs between them.
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
