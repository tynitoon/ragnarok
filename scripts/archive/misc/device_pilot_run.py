"""Phase 2 Stage 5.5: end-to-end device-path cartpole_mcc pipeline.

Demonstrates the full accelerator-resident agent — every training loop
(PPO/SAC, RSSM world model, latent policy, curiosity) batched on the device,
no gym envs, no host env-loop — and the cross-dim transfer mechanism:

  1. Source: a CartPole DeviceAgent trains PPO + world model + latent policy
     together; its greedy eval reward climbs toward 500.
  2. snapshot(): the env-agnostic transferable subset (RSSM core +
     latent-policy trunk) is extracted.
  3. Target: an MCC DeviceAgent (obs 2 / act 1 — different dimensions) loads
     the CartPole snapshot via load_snapshot() and trains.

The device-path counterpart of scripts/pilot_run.py. It gates the Stage 5
INTEGRATION — the components wired and orchestrated end to end. The
transfer-EFFECT measurement (transfer vs scratch, matching the calibrated
gym pilot) is the separate Phase 2 re-calibration step.

Usage:  python -m scripts.device_pilot_run
"""

import time

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceVecMountainCarContinuous)

SOURCE_ITERS = 30
TARGET_ITERS = 20
EVAL_EVERY = 10


def _train(agent, label: str, iters: int, eval_steps: int) -> float:
    print(f"\n  [{label}]")
    for it in range(1, iters + 1):
        m = agent.train_iteration()
        if it == 1 or it % EVAL_EVERY == 0:
            score = agent.evaluate(steps=eval_steps)
            wm = m.get("wm/total_loss", float("nan"))
            print(f"    iter {it:3d} | eval {score:9.2f} | wm_loss {wm:7.3f}")
    return agent.evaluate(steps=eval_steps)


def main():
    print(f"[device-pilot-run] device={DEVICE}")
    t0 = time.perf_counter()

    # 1. Source: CartPole — the full device agent (PPO + WM + latent policy).
    source = DeviceAgent(DeviceVecCartPole, num_envs=128, horizon=128)
    src_final = _train(source, "source: CartPole", SOURCE_ITERS, eval_steps=500)
    print(f"  source CartPole final eval: {src_final:.1f}")

    # 2. Snapshot the env-agnostic transferable subset.
    snap = source.snapshot()
    print(f"  snapshot: {len(snap['rssm_core'])} RSSM-core tensors, "
          f"{len(snap['latent_trunk'])} latent-trunk tensors")

    # 3. Target: MCC — load the CartPole snapshot (cross-dim, obs 4->2 act 2->1).
    target = DeviceAgent(DeviceVecMountainCarContinuous, num_envs=256,
                         horizon=128, sac_updates=512)
    target.load_snapshot(snap)
    print("  transfer: CartPole snapshot loaded into the MCC agent")
    tgt_final = _train(target, "target: MountainCar (post-transfer)",
                       TARGET_ITERS, eval_steps=999)
    print(f"  target MCC final eval: {tgt_final:.2f}")

    wall = time.perf_counter() - t0
    print(f"\n  done in {wall:.0f}s")
    ok = src_final >= 400.0   # device PPO crystallizes CartPole to ~500
    verdict = ("PASS — full device pipeline ran end to end; "
               "cross-dim transfer loaded and trained" if ok else
               "CHECK — source CartPole did not crystallize")
    print(f"  [{verdict}]")


if __name__ == "__main__":
    main()
