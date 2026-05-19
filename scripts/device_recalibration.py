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

Checkpoint/resume (--ckpt): the run pickles the live DeviceAgent and its
progress after every iteration. On a preemptible (spot) TPU, a fresh VM
re-launched with the same --ckpt resumes mid-arm — the whole agent (models,
optimizers, SAC buffer, normalizer, RNG) is in the checkpoint, so the
continuation is exact. Without --ckpt the run is in-memory only (for a
stable machine).

Usage:  python -m scripts.device_recalibration [--seeds N] [--source-iters N]
        [--mcc-iters N] [--sac-updates N] [--ckpt PATH]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)

MASTERY = 90.0          # MountainCarContinuous mastery threshold (gym pilot)


def _save_ckpt(ck: dict, path: str | None) -> None:
    """Atomically write the checkpoint (to .tmp, then rename)."""
    if not path:
        return
    ck["torch_rng"] = torch.get_rng_state()
    ck["np_rng"] = np.random.get_state()
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def _load_or_init(seed: int, path: str | None) -> dict:
    """Resume from the checkpoint at `path`, or start a fresh seed."""
    if path and os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        print(f"  [resume] seed {seed}: phase={ck['phase']} iter={ck['iter']}",
              flush=True)
        return ck
    torch.manual_seed(seed)
    np.random.seed(seed)
    return {"seed": seed, "phase": "source", "iter": 0, "agent": None,
            "snapshot": None, "source_eval": None,
            "scratch": None, "transfer": None}


def _mcc_loop(ck: dict, key: str, mcc_iters: int, path: str | None) -> None:
    """Train ck['agent'] on MCC for mcc_iters, checkpointing every iteration.

    `key` is 'scratch' or 'transfer'. Resumable: picks up at ck['iter'].
    """
    agent, rec = ck["agent"], ck[key]
    while ck["iter"] < mcc_iters:
        agent.train_iteration()
        score = agent.evaluate(steps=999)
        rec["eval_curve"].append([agent.total_env_steps, score])
        if rec["mastery_env_steps"] is None and score >= MASTERY:
            rec["mastery_env_steps"] = agent.total_env_steps
        ck["iter"] += 1
        _save_ckpt(ck, path)
        if ck["iter"] % 5 == 0 or ck["iter"] == mcc_iters:
            print(f"    {key} iter {ck['iter']:>3}/{mcc_iters} | "
                  f"score {score:7.1f} | env_steps {agent.total_env_steps}",
                  flush=True)


def _run_seed(seed: int, source_iters: int, mcc_iters: int,
              curiosity_warmup: int, sac_updates: int,
              ckpt_path: str | None = None) -> dict:
    """Run (or resume) one transfer seed: source -> scratch MCC -> transfer MCC."""
    ck = _load_or_init(seed, ckpt_path)

    # -- source: CartPole -> snapshot the transferable subset --
    if ck["phase"] == "source":
        if ck["agent"] is None:
            ck["agent"] = DeviceAgent(DeviceVecCartPole, num_envs=128,
                                      horizon=128)
        while ck["iter"] < source_iters:
            ck["agent"].train_iteration()
            ck["iter"] += 1
            _save_ckpt(ck, ckpt_path)
        ck["source_eval"] = ck["agent"].evaluate(steps=500)
        ck["snapshot"] = ck["agent"].snapshot()
        print(f"  seed {seed}: source done, eval {ck['source_eval']:.1f}",
              flush=True)
        ck["phase"], ck["iter"], ck["agent"] = "scratch", 0, None
        _save_ckpt(ck, ckpt_path)

    # -- scratch MCC: SAC-acting, no transfer --
    if ck["phase"] == "scratch":
        if ck["agent"] is None:
            ck["agent"] = DeviceAgent(
                DeviceVecMountainCarContinuous, num_envs=256, horizon=128,
                sac_updates=sac_updates, curiosity_warmup=curiosity_warmup)
            ck["scratch"] = {"eval_curve": [], "mastery_env_steps": None}
        _mcc_loop(ck, "scratch", mcc_iters, ckpt_path)
        ck["phase"], ck["iter"], ck["agent"] = "transfer", 0, None
        _save_ckpt(ck, ckpt_path)

    # -- transfer MCC: load the CartPole snapshot, act via the latent policy --
    if ck["phase"] == "transfer":
        if ck["agent"] is None:
            ag = DeviceAgent(
                DeviceVecMountainCarContinuous, num_envs=256, horizon=128,
                sac_updates=sac_updates, curiosity_warmup=curiosity_warmup)
            ag.load_snapshot(ck["snapshot"])
            ck["agent"] = ag
            ck["transfer"] = {"eval_curve": [], "mastery_env_steps": None}
        _mcc_loop(ck, "transfer", mcc_iters, ckpt_path)
        ck["phase"], ck["agent"] = "done", None
        _save_ckpt(ck, ckpt_path)

    return {"seed": seed, "source_eval": ck["source_eval"],
            "scratch": ck["scratch"], "transfer": ck["transfer"]}


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
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint path base. If given, each seed "
                             "checkpoints after every iteration to "
                             "<ckpt>.seed<N>.pt and resumes from it — needed "
                             "to survive spot-TPU preemption. Omit on a "
                             "stable machine.")
    args = parser.parse_args()

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    out = (os.path.join(os.path.dirname(args.ckpt) or ".",
                        "device_recalibration.json")
           if args.ckpt else "device_recalibration.json")

    print(f"[device-recalibration] device={DEVICE}  seeds={args.seeds}  "
          f"source_iters={args.source_iters}  mcc_iters={args.mcc_iters}  "
          f"ckpt={'on' if args.ckpt else 'off'}", flush=True)
    t0 = time.perf_counter()
    results = []
    for seed in range(args.seeds):
        s0 = time.perf_counter()
        ckpt_path = f"{args.ckpt}.seed{seed}.pt" if args.ckpt else None
        r = _run_seed(seed, args.source_iters, args.mcc_iters,
                      args.curiosity_warmup, args.sac_updates, ckpt_path)
        results.append(r)
        s = r["scratch"]["mastery_env_steps"]
        t = r["transfer"]["mastery_env_steps"]
        ratio = (s / t) if (s and t) else None
        print(f"  seed {seed} | src_eval {r['source_eval']:6.1f} | "
              f"scratch mastery {str(s):>9} | transfer mastery {str(t):>9} | "
              f"ratio {('%.3f' % ratio) if ratio else '  n/a':>6} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)
        with open(out, "w") as f:                  # incremental save
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
    with open(out, "w") as f:
        json.dump({"results": results, "ratios": ratios,
                   "median_ratio": median}, f, indent=2)
    print(f"  wrote {out}  |  {wall:.0f}s")


if __name__ == "__main__":
    main()
