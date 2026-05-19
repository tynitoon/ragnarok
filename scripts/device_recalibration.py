"""Phase 2 re-calibration: the device-path cartpole_mcc transfer experiment.

Confirms the device path reproduces the SCIENCE of the gym pilot. The gym
pilot's cartpole_mcc result was Band C — no transfer effect, scratch/transfer
mastery ratio ~1.0 (GPU median 1.006). This runs the same experiment on the
device path and checks it lands in the same band.

Per seed:
  - source : a CartPole DeviceAgent trains (PPO + world model + latent
    policy) and crystallizes; snapshot() extracts the transferable subset;
  - scratch MCC : a fresh DeviceAgent trains MountainCar, SAC-acting
    ("obs" mode), no transfer;
  - transfer MCC : a fresh DeviceAgent loads the CartPole snapshot
    (load_snapshot -> "latent" mode) and trains MountainCar acting via the
    transferred latent policy — the gym pilot's mode=latent.

Both MCC arms run a fixed iteration budget; mastery = total env-steps at the
first greedy eval >= 90. ratio = scratch_mastery / transfer_mastery (>1 means
transfer reached mastery sooner). Band C = ratio < 1.05.

Usage:  python -m scripts.device_recalibration [--seeds N] [--source-iters N]
        [--mcc-iters N]
"""

import argparse
import json
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)

MASTERY = 90.0          # MountainCarContinuous mastery threshold (gym pilot)
OUT = "device_recalibration.json"


def _run_mcc_arm(agent: DeviceAgent, iters: int):
    """Train an MCC DeviceAgent; return (eval_curve, mastery_env_steps)."""
    curve, mastery = [], None
    for _ in range(iters):
        agent.train_iteration()
        score = agent.evaluate(steps=999)
        curve.append([agent.total_env_steps, score])
        if mastery is None and score >= MASTERY:
            mastery = agent.total_env_steps
    return curve, mastery


def _run_seed(seed: int, source_iters: int, mcc_iters: int,
              curiosity_warmup: int, sac_updates: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Source: CartPole — train to crystallization, snapshot the transferable
    # subset (RSSM core + latent-policy trunk).
    source = DeviceAgent(DeviceVecCartPole, num_envs=128, horizon=128)
    for _ in range(source_iters):
        source.train_iteration()
    src_eval = source.evaluate(steps=500)
    snap = source.snapshot()

    # Scratch MCC — SAC-acting, no transfer.
    scratch = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=256,
                          horizon=128, sac_updates=sac_updates,
                          curiosity_warmup=curiosity_warmup)
    s_curve, s_mastery = _run_mcc_arm(scratch, mcc_iters)

    # Transfer MCC — load the CartPole snapshot; load_snapshot flips the
    # agent to act via the transferred latent policy.
    transfer = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=256,
                           horizon=128, sac_updates=sac_updates,
                           curiosity_warmup=curiosity_warmup)
    transfer.load_snapshot(snap)
    t_curve, t_mastery = _run_mcc_arm(transfer, mcc_iters)

    return {
        "seed": seed,
        "source_eval": src_eval,
        "scratch": {"eval_curve": s_curve, "mastery_env_steps": s_mastery},
        "transfer": {"eval_curve": t_curve, "mastery_env_steps": t_mastery},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--source-iters", type=int, default=25)
    parser.add_argument("--mcc-iters", type=int, default=45)
    parser.add_argument("--curiosity-warmup", type=int, default=6,
                        help="iterations the latent-KL curiosity is gated "
                             "off (RSSM not yet trained)")
    parser.add_argument("--sac-updates", type=int, default=512,
                        help="SAC updates per iteration on the MCC arms "
                             "(lower = faster, lighter on the device path)")
    args = parser.parse_args()

    print(f"[device-recalibration] device={DEVICE}  seeds={args.seeds}  "
          f"source_iters={args.source_iters}  mcc_iters={args.mcc_iters}")
    t0 = time.perf_counter()
    results = []
    for seed in range(args.seeds):
        s0 = time.perf_counter()
        r = _run_seed(seed, args.source_iters, args.mcc_iters,
                      args.curiosity_warmup, args.sac_updates)
        results.append(r)
        s = r["scratch"]["mastery_env_steps"]
        t = r["transfer"]["mastery_env_steps"]
        ratio = (s / t) if (s and t) else None
        print(f"  seed {seed} | src_eval {r['source_eval']:6.1f} | "
              f"scratch mastery {str(s):>9} | transfer mastery {str(t):>9} | "
              f"ratio {('%.3f' % ratio) if ratio else '  n/a':>6} | "
              f"{time.perf_counter() - s0:.0f}s")
        with open(OUT, "w") as f:                  # incremental save
            json.dump({"results": results}, f, indent=2)

    ratios = [r["scratch"]["mastery_env_steps"] / r["transfer"]["mastery_env_steps"]
              for r in results
              if r["scratch"]["mastery_env_steps"]
              and r["transfer"]["mastery_env_steps"]]
    wall = time.perf_counter() - t0
    print(f"\n  {len(ratios)}/{args.seeds} seeds reached mastery in both arms")
    median = float(np.median(ratios)) if ratios else None
    if ratios:
        band = ("C (no transfer effect)" if median < 1.05
                else "B (1.05-1.30)" if median < 1.30 else "A (>=1.30)")
        print(f"  scratch/transfer mastery ratio: median {median:.3f}  "
              f"(range {min(ratios):.3f}-{max(ratios):.3f})")
        print(f"  band: {band}   [GPU pilot: ~1.006, Band C]")
        verdict = ("PASS — device path reproduces the GPU pilot's Band C"
                   if median < 1.05 else
                   "CHECK — device ratio outside Band C")
    else:
        verdict = "CHECK — no seed reached mastery in both arms"
    print(f"  [{verdict}]")
    with open(OUT, "w") as f:
        json.dump({"results": results, "ratios": ratios,
                   "median_ratio": median}, f, indent=2)
    print(f"  wrote {OUT}  |  {wall:.0f}s")


if __name__ == "__main__":
    main()
