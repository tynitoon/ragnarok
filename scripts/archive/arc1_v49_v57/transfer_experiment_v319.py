"""v3.19 multi-skill composition: do TWO transferred RSSM cores compose
into a skill neither carries alone? (Cart-Pole-on-Hill via weight-averaged
MCC + CartPole cores.)

Preregistered as preregistration.md amendment v3.19 (commit e25af57,
committed before this script and any run). Phase A established that a
single transferred core accelerates SAC iff source and target share
physics. v3.19 tests the project's stated long-term goal — "learn a new
skill faster by reusing OLD skills (plural)" — by combining two source
cores on a composite target.

The composite: DeviceVecCartPoleOnHill — a cart on the MCC hill with a
pole balanced on top. Aggressive driving (needed to climb the hill)
swings the pole; gentle driving preserves the pole but cannot climb.
Neither MCC's energy-pumping skill nor CartPole's balancing skill alone
suffices; the composition is required.

Composition mechanism (preregistered): element-wise weight averaging of
the MCC core and the CartPole core. avg[k] = (mcc[k] + cp[k]) / 2 for
each parameter tensor in `transferable_state_dict`.

Five arms, N=8 (same v3.14 mechanism — latent-only SAC, frozen RSSM,
aug-normalizer, ICM curiosity, 3-point AUC smoothing, Student-t CI):
  - scratch:       fresh random core
  - permuted:      MCC core with each parameter tensor permuted
  - transfer_mcc:  MCC core only (energy-pumping skill alone)
  - transfer_cp:   CartPole core only (balancing skill alone)
  - transfer_avg:  (MCC + CartPole) / 2 — the composition

Decisive comparison: transfer_avg minus max(transfer_mcc, transfer_cp)
per seed, mean +/- Student-t 95% CI. If CI excludes 0 (positive), the
composition adds value over either skill alone — the violin+solfege
claim. If null, averaging is no better than the best single skill
(though either single may still be better than scratch). If
transfer_avg < either single, averaging destroyed structure — v3.20
contingent on concatenation as an alternative mechanism.

Usage: python -m scripts.transfer_experiment_v319 [--seeds N] [--smoke]
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
    DeviceVecMountainCarContinuous, DeviceVecCartPole,
    DeviceVecCartPoleOnHill, DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 4, 1                       # TARGET: cart-pole-on-hill
ARMS = ("scratch", "permuted", "transfer_mcc", "transfer_cp", "transfer_avg")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


def _permute_state_dict(state, seed):
    g = torch.Generator().manual_seed(seed + 90_000)
    return {k: v.flatten()[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
            for k, v in state.items()}


def _average_cores(snap_a, snap_b):
    """Element-wise mean of two transferable cores. Both must share keys
    and per-tensor shapes (they will, because both are
    `transferable_state_dict()` outputs of the same RSSM architecture).
    """
    assert set(snap_a) == set(snap_b), "core keys mismatch"
    return {k: (snap_a[k] + snap_b[k]) / 2.0 for k in snap_a}


def _build_arm(env_cls, cfg, arm, snapshots, seed):
    """Build one frozen-RSSM, latent-only arm. Target RSSM is RSSM(4, 1)
    (cart-pole-on-hill obs_dim=4, action_dim=1). The transferable core
    (gru + prior + posterior) is dim-agnostic so it loads despite the
    obs_dim difference from the sources."""
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

    mcc = snapshots["mcc"]
    cp = snapshots["cp"]
    if arm == "transfer_mcc":
        core = mcc
    elif arm == "transfer_cp":
        core = cp
    elif arm == "transfer_avg":
        core = _average_cores(mcc, cp)
    elif arm == "permuted":
        core = _permute_state_dict(mcc, seed)
    else:                                        # scratch
        core = None
    if core is not None:
        rssm.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in core.items()})
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    return env, rssm, sac, curiosity, normalizer, aug_normalizer


def _train_iter(env, rssm, sac, curiosity, normalizer, aug_normalizer,
                horizon, sac_updates):
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


def _run_arm(arm, env_cls, cfg, snapshots, seed):
    env, rssm, sac, curiosity, normalizer, aug_normalizer = _build_arm(
        env_cls, cfg, arm, snapshots, seed)
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
            print(f"    [{arm:>13}] iter {it:>3}/{cfg['arm_rollouts']} | "
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


def _load_or_train_mcc(cfg, snap_path):
    """Standard MCC source via the proven raw-obs DeviceAgent path; caches
    the env-agnostic core to `snap_path`."""
    if os.path.exists(snap_path):
        print(f"[source/mcc] cached: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source/mcc] training on MCC ({cfg['source_rollouts']} rollouts)...",
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
            score = agent.evaluate(steps=999, n_envs=cfg["num_envs"])
            print(f"  [source/mcc] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | eval {score:8.2f}",
                  flush=True)
    snap = {k: v.detach().cpu()
            for k, v in agent.rssm.transferable_state_dict().items()}
    torch.save(snap, snap_path)
    print(f"[source/mcc] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return snap


def _load_or_train_cp(cfg, snap_path):
    """CartPole source via DeviceAgent (discrete: PPO + WM + latent);
    caches the env-agnostic core."""
    if os.path.exists(snap_path):
        print(f"[source/cp] cached: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source/cp] training on CartPole ({cfg['source_rollouts']} rollouts)...",
          flush=True)
    torch.manual_seed(2000)
    np.random.seed(2000)
    agent = DeviceAgent(DeviceVecCartPole,
                        num_envs=cfg["num_envs"], horizon=cfg["horizon"])
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        agent.train_iteration()
        if it == 1 or it % cfg["eval_every"] == 0:
            score = agent.evaluate(steps=500, n_envs=cfg["num_envs"])
            print(f"  [source/cp] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | eval {score:8.2f}",
                  flush=True)
    snap = {k: v.detach().cpu()
            for k, v in agent.rssm.transferable_state_dict().items()}
    torch.save(snap, snap_path)
    print(f"[source/cp] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return snap


def _summarize(results):
    """Per-arm AUC means, key pairwise differences, and the decisive
    composition comparison (transfer_avg vs max(transfer_mcc,transfer_cp))."""
    # Per-arm AUCs
    arm_mean = {}
    for arm in ARMS:
        vals = [r[f"{arm}_auc"] for r in results]
        arm_mean[arm], arm_ci = _ci95(vals)
        n_solve = sum(1 for v in vals if v > 1.0)  # solving ≈ AUC > 1
        print(f"  {arm:>13}: AUC mean {arm_mean[arm]:+7.2f}  "
              f"95%CI +/-{arm_ci:6.2f}  ({n_solve}/{len(vals)} AUC>1)")

    # Pairwise differences from transfer_avg.
    def _diff_stat(name, vals):
        m, c = _ci95(vals)
        pos = sum(1 for v in vals if v > 0)
        marker = " *" if m - c > 0 else ("  " if m + c >= 0 else " -")
        print(f"  diff {name:>30}: mean {m:+7.2f}  95%CI +/-{c:6.2f}  "
              f"({pos}/{len(vals)} > 0){marker}")
        return {"mean": m, "ci95": c, "n_positive": pos}

    print(f"\n  pairwise diffs:")
    d_vs_scratch = _diff_stat("transfer_avg - scratch",
                              [r["transfer_avg_auc"] - r["scratch_auc"] for r in results])
    d_vs_perm = _diff_stat("transfer_avg - permuted",
                           [r["transfer_avg_auc"] - r["permuted_auc"] for r in results])
    d_vs_mcc = _diff_stat("transfer_avg - transfer_mcc",
                          [r["transfer_avg_auc"] - r["transfer_mcc_auc"] for r in results])
    d_vs_cp = _diff_stat("transfer_avg - transfer_cp",
                         [r["transfer_avg_auc"] - r["transfer_cp_auc"] for r in results])
    # The decisive composition test: transfer_avg > best single.
    d_vs_best_single = _diff_stat(
        "transfer_avg - max(mcc, cp)",
        [r["transfer_avg_auc"] - max(r["transfer_mcc_auc"], r["transfer_cp_auc"])
         for r in results])

    composition_works = (
        len(results) >= 2 and d_vs_best_single["mean"] - d_vs_best_single["ci95"] > 0)
    if composition_works:
        verdict = ("COMPOSITION WORKS: transfer_avg > max(transfer_mcc, transfer_cp), "
                   "CI excludes 0 — the averaged core outperforms either skill alone")
    elif len(results) >= 2 and d_vs_best_single["mean"] + d_vs_best_single["ci95"] < 0:
        verdict = ("COMPOSITION HURTS: averaging actively destroys structure; "
                   "v3.20 contingent on concatenation mechanism")
    else:
        verdict = ("NO COMPOSITION EFFECT RESOLVED: averaging neither helps nor hurts "
                   "vs best single skill (CI spans 0)")
    print(f"\n  -> {verdict}")
    return {"arm_means": arm_mean,
            "diff_avg_vs_scratch": d_vs_scratch,
            "diff_avg_vs_permuted": d_vs_perm,
            "diff_avg_vs_mcc": d_vs_mcc,
            "diff_avg_vs_cp": d_vs_cp,
            "diff_avg_vs_best_single": d_vs_best_single,
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
    parser.add_argument("--out-dir", default="transfer_v319_out")
    parser.add_argument("--smoke", action="store_true")
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
    mcc_snap_path = os.path.join(args.out_dir, "source_mcc_core.pt")
    cp_snap_path = os.path.join(args.out_dir, "source_cp_core.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v319] device={DEVICE}  seeds={args.seeds}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}",
          flush=True)
    t0 = time.perf_counter()

    snapshots = {
        "mcc": _load_or_train_mcc(cfg, mcc_snap_path),
        "cp": _load_or_train_cp(cfg, cp_snap_path),
    }

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
        aucs = {}
        for arm in ARMS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            curves[arm] = _run_arm(arm, DeviceVecCartPoleOnHill, cfg, snapshots, seed)
            aucs[arm] = _auc(curves[arm])
        rec = {"seed": seed}
        for arm in ARMS:
            rec[f"{arm}_curve"] = curves[arm]
            rec[f"{arm}_auc"] = aucs[arm]
        results.append(rec)
        with open(results_path, "w") as f:
            json.dump({"seeds": results}, f, indent=2)
        # Compact per-seed line.
        best_single = max(aucs["transfer_mcc"], aucs["transfer_cp"])
        comp_diff = aucs["transfer_avg"] - best_single
        print(f"  seed {seed} | "
              f"scratch {aucs['scratch']:+6.1f}  permuted {aucs['permuted']:+6.1f}  "
              f"t_mcc {aucs['transfer_mcc']:+6.1f}  t_cp {aucs['transfer_cp']:+6.1f}  "
              f"t_avg {aucs['transfer_avg']:+6.1f}  | "
              f"composition diff {comp_diff:+6.1f} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)

    print(f"\n{'=' * 72}\n  N={len(results)} seeds  |  target=Cart-Pole-on-Hill")
    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"\n  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
