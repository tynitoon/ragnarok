"""v3.21 multi-skill composition via LATENT CONCATENATION (two parallel
frozen RSSMs). The v3.19/v3.20 preregistered fallback after weight
averaging was killed by v3.20 (N=8, 0/8 seeds positive, CI entirely
below 0).

Preregistered as preregistration.md amendment v3.21 (commit 079e7bd,
committed before this script and any run).

Mechanism: two FROZEN RSSMs in parallel, each producing its own [h, z].
SAC reads cat([h_a, z_a, h_b, z_b]) = 2 * state_dim = 320-d. Each core
keeps its learned structure intact (no weight averaging); SAC's
first-layer Linear learns to weight the relevant half of the concat.
The principled composition mechanism — preserves each source's
representation and lets SAC selectively use either or both.

Same target as v3.20: MountainCarContinuous-Hard (where v3.14 and
v3.16 single-skill transfers are solidly positive — isolates the
mechanism question from target-amenability).

Five arms, N=8, two cores per arm (same dual-latent architecture
across arms keeps SAC's input dim equal everywhere):
  - scratch_dual:      2 fresh random cores
  - permuted_dual:     MCC core permuted + Pendulum core permuted
                       (structure control)
  - transfer_mcc_only: trained MCC core + 1 fresh random core
                       (single-skill in the concat architecture)
  - transfer_pen_only: 1 fresh random core + trained Pendulum core
                       (single-skill in the concat architecture)
  - transfer_both:     trained MCC core + trained Pendulum core
                       (THE COMPOSITION TEST)

Decisive comparison: transfer_both AUC minus max(transfer_mcc_only,
transfer_pen_only) AUC, per seed, mean +/- Student-t 95% CI. CI
excluding 0 positive => latent concatenation COMPOSES the two skill
cores into something stronger than either alone — the clean
composition positive the project is after.

Usage: python -m scripts.transfer_experiment_v321 [--seeds N] [--smoke]
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
from ragnarok.learning.rollout import collect_rollout_dual_latent, evaluate_dual_latent
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuous, DeviceVecPendulum,
    DeviceVecMountainCarContinuousHard, DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 2, 1                       # TARGET: MCC-Hard
ARMS = ("scratch_dual", "permuted_dual",
        "transfer_mcc_only", "transfer_pen_only", "transfer_both")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


def _permute_state_dict(state, seed):
    g = torch.Generator().manual_seed(seed + 90_000)
    return {k: v.flatten()[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
            for k, v in state.items()}


def _core_pair(arm, snapshots, seed):
    """Return (core_a, core_b) for the arm; None means "fresh random
    init" (the RSSM constructor's default)."""
    mcc, pen = snapshots["mcc"], snapshots["pen"]
    if arm == "scratch_dual":
        return None, None
    if arm == "permuted_dual":
        return _permute_state_dict(mcc, seed), _permute_state_dict(pen, seed + 1)
    if arm == "transfer_mcc_only":
        return mcc, None
    if arm == "transfer_pen_only":
        return None, pen
    if arm == "transfer_both":
        return mcc, pen
    raise ValueError(f"unknown arm: {arm}")


def _build_arm(env_cls, cfg, arm, snapshots, seed):
    """Two frozen RSSMs; SAC reads cat([h_a, z_a, h_b, z_b]) = 2 * state_dim."""
    env = env_cls(cfg["num_envs"])
    rssm_a = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    rssm_b = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    state_dim = rssm_a.state_dim
    dual_dim = 2 * state_dim
    sac = SACTrainer(
        obs_dim=dual_dim, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    curiosity = DeviceForwardCuriosity(OBS_DIM, ACTION_DIM)
    normalizer = DeviceRunningNormalizer(OBS_DIM)
    aug_normalizer = DeviceRunningNormalizer(dual_dim)

    core_a, core_b = _core_pair(arm, snapshots, seed)
    if core_a is not None:
        rssm_a.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in core_a.items()})
    if core_b is not None:
        rssm_b.load_transferable_state_dict(
            {k: v.to(DEVICE) for k, v in core_b.items()})
    for r in (rssm_a, rssm_b):
        r.eval()
        for p in r.parameters():
            p.requires_grad_(False)
    return env, rssm_a, rssm_b, sac, curiosity, normalizer, aug_normalizer


def _train_iter(env, rssm_a, rssm_b, sac, curiosity, normalizer,
                aug_normalizer, horizon, sac_updates):
    batch = collect_rollout_dual_latent(env, rssm_a, rssm_b, sac, horizon,
                                        normalizer=normalizer,
                                        aug_normalizer=aug_normalizer,
                                        deterministic=True)
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
    env, rssm_a, rssm_b, sac, curiosity, normalizer, aug_normalizer = (
        _build_arm(env_cls, cfg, arm, snapshots, seed))
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, rssm_a, rssm_b, sac, curiosity, normalizer,
                             aug_normalizer, cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_dual_latent(env_cls(cfg["num_envs"]), rssm_a, rssm_b,
                                         sac.policy, cfg["eval_steps"],
                                         normalizer=normalizer,
                                         aug_normalizer=aug_normalizer,
                                         deterministic=True)
            curve.append([total, score])
            print(f"    [{arm:>17}] iter {it:>3}/{cfg['arm_rollouts']} | "
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


def _load_or_train_pen(cfg, snap_path):
    if os.path.exists(snap_path):
        print(f"[source/pen] cached: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source/pen] training on Pendulum ({cfg['source_rollouts']} rollouts)...",
          flush=True)
    torch.manual_seed(3000)
    np.random.seed(3000)
    agent = DeviceAgent(DeviceVecPendulum,
                        num_envs=cfg["num_envs"], horizon=cfg["horizon"],
                        sac_updates=cfg["source_sac_updates"],
                        curiosity_warmup=6)
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        agent.train_iteration()
        if it == 1 or it % cfg["eval_every"] == 0:
            score = agent.evaluate(steps=200, n_envs=cfg["num_envs"])
            print(f"  [source/pen] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | eval {score:8.2f}",
                  flush=True)
    snap = {k: v.detach().cpu()
            for k, v in agent.rssm.transferable_state_dict().items()}
    torch.save(snap, snap_path)
    print(f"[source/pen] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return snap


def _summarize(results):
    arm_mean = {}
    for arm in ARMS:
        vals = [r[f"{arm}_auc"] for r in results]
        arm_mean[arm], arm_ci = _ci95(vals)
        n_pos = sum(1 for v in vals if v > 1.0)
        print(f"  {arm:>17}: AUC mean {arm_mean[arm]:+7.2f}  "
              f"95%CI +/-{arm_ci:6.2f}  ({n_pos}/{len(vals)} AUC>1)")

    def _diff_stat(name, vals):
        m, c = _ci95(vals)
        pos = sum(1 for v in vals if v > 0)
        marker = " *" if m - c > 0 else ("  " if m + c >= 0 else " -")
        print(f"  diff {name:>40}: mean {m:+7.2f}  95%CI +/-{c:6.2f}  "
              f"({pos}/{len(vals)} > 0){marker}")
        return {"mean": m, "ci95": c, "n_positive": pos}

    print(f"\n  pairwise diffs:")
    d_vs_scratch = _diff_stat(
        "transfer_both - scratch_dual",
        [r["transfer_both_auc"] - r["scratch_dual_auc"] for r in results])
    d_vs_perm = _diff_stat(
        "transfer_both - permuted_dual",
        [r["transfer_both_auc"] - r["permuted_dual_auc"] for r in results])
    d_vs_mcc = _diff_stat(
        "transfer_both - transfer_mcc_only",
        [r["transfer_both_auc"] - r["transfer_mcc_only_auc"] for r in results])
    d_vs_pen = _diff_stat(
        "transfer_both - transfer_pen_only",
        [r["transfer_both_auc"] - r["transfer_pen_only_auc"] for r in results])
    d_vs_best = _diff_stat(
        "transfer_both - max(mcc_only, pen_only)",
        [r["transfer_both_auc"]
         - max(r["transfer_mcc_only_auc"], r["transfer_pen_only_auc"])
         for r in results])

    works = (len(results) >= 2 and d_vs_best["mean"] - d_vs_best["ci95"] > 0)
    if works:
        verdict = ("COMPOSITION WORKS via concatenation: transfer_both > "
                   "max(single), CI excludes 0 -- two skill cores compose")
    elif len(results) >= 2 and d_vs_best["mean"] + d_vs_best["ci95"] < 0:
        verdict = ("COMPOSITION HURTS via concatenation; RSSM-substrate "
                   "composition mechanisms exhausted, pivot model class")
    else:
        verdict = ("NO COMPOSITION EFFECT RESOLVED via concatenation "
                   "(CI spans 0)")
    print(f"\n  -> {verdict}")
    return {"arm_means": arm_mean,
            "diff_both_vs_scratch": d_vs_scratch,
            "diff_both_vs_permuted": d_vs_perm,
            "diff_both_vs_mcc_only": d_vs_mcc,
            "diff_both_vs_pen_only": d_vs_pen,
            "diff_both_vs_best_single": d_vs_best,
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
    parser.add_argument("--out-dir", default="transfer_v321_out")
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
    pen_snap_path = os.path.join(args.out_dir, "source_pen_core.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v321] device={DEVICE}  seeds={args.seeds}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}",
          flush=True)
    t0 = time.perf_counter()

    snapshots = {
        "mcc": _load_or_train_mcc(cfg, mcc_snap_path),
        "pen": _load_or_train_pen(cfg, pen_snap_path),
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
            curves[arm] = _run_arm(arm, DeviceVecMountainCarContinuousHard,
                                   cfg, snapshots, seed)
            aucs[arm] = _auc(curves[arm])
        rec = {"seed": seed}
        for arm in ARMS:
            rec[f"{arm}_curve"] = curves[arm]
            rec[f"{arm}_auc"] = aucs[arm]
        results.append(rec)
        with open(results_path, "w") as f:
            json.dump({"seeds": results}, f, indent=2)
        best_single = max(aucs["transfer_mcc_only"], aucs["transfer_pen_only"])
        comp_diff = aucs["transfer_both"] - best_single
        print(f"  seed {seed} | "
              f"scratch {aucs['scratch_dual']:+6.1f}  "
              f"permuted {aucs['permuted_dual']:+6.1f}  "
              f"mcc {aucs['transfer_mcc_only']:+6.1f}  "
              f"pen {aucs['transfer_pen_only']:+6.1f}  "
              f"both {aucs['transfer_both']:+6.1f}  | "
              f"composition diff {comp_diff:+6.1f} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)

    print(f"\n{'=' * 80}\n  N={len(results)} seeds  |  target=MCC-Hard, "
          f"composition mechanism=latent concatenation (320-d)")
    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"\n  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
