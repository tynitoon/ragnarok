"""v3.22 substrate pivot: composition via SAC POLICY weight transfer
(not RSSM-core weight transfer). v3.20 (averaging RSSM cores) was
CLEAN NEGATIVE; v3.21 (concatenating RSSM latents) was NEUTRAL. v3.22
keeps the experimental design parallel to v3.20 (same sources, same
target, same N, same arms structure) but transfers ENV-AGNOSTIC
POLICY WEIGHTS instead of RSSM cores.

Preregistered as preregistration.md amendment v3.22 (commit cf4468e,
committed before this script and any run).

Substrate: SAC's actor (SACPolicy) and critics (two QNetworks). The
env-agnostic subset:
  - actor: shared.2 (hidden->hidden) + mean_head + logstd_head
  - critics: net.2 + net.4 (per Q-network)
The first layer of each network is env-specific (obs_dim varies); the
action heads are shape-compatible because action_dim=1 across all our
envs (MCC, MCC-Hard, Pendulum continuous).

Target = MCC-Hard. NO RSSM at all in the target — SAC reads raw obs.
This is the proven raw-obs DeviceAgent setup (Phase A's source path).

Five arms, N=8:
  - scratch_pol:       fresh random SAC (baseline)
  - permuted_pol:      MCC source policy weights with each tensor permuted
  - transfer_mcc_pol:  env-agnostic subset of MCC source's SAC loaded
  - transfer_pen_pol:  env-agnostic subset of Pendulum source's SAC loaded
  - transfer_avg_pol:  averaged env-agnostic subsets — COMPOSITION TEST

Decisive comparison: transfer_avg_pol - max(transfer_mcc_pol,
transfer_pen_pol), per-seed mean +/- Student-t 95% CI. Single-skill
check: transfer_mcc_pol vs scratch_pol.

Sources must be re-trained (the v3.11 + v3.16 caches saved only the
RSSM core, not the SAC actor + critics). Cached to
transfer_v322_out/source_*_policy.pt.

Usage: python -m scripts.transfer_experiment_v322 [--seeds N] [--smoke]
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.device_agent import DeviceAgent
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.curiosity import DeviceForwardCuriosity
from ragnarok.learning.rollout import collect_rollout
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuous, DeviceVecPendulum,
    DeviceVecMountainCarContinuousHard, DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 2, 1                       # TARGET: MCC-Hard
ARMS = ("scratch_pol", "permuted_pol",
        "transfer_mcc_pol", "transfer_pen_pol", "transfer_avg_pol")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}

# Env-agnostic key sets for the SAC modules. shared.0 / net.0 are
# obs-dim-specific and excluded; action_dim=1 across all our envs so
# the heads are included.
ACTOR_ENV_AGNOSTIC_KEYS = frozenset({
    "shared.2.weight", "shared.2.bias",
    "mean_head.weight", "mean_head.bias",
    "logstd_head.weight", "logstd_head.bias",
})
CRITIC_ENV_AGNOSTIC_KEYS = frozenset({
    "net.2.weight", "net.2.bias",
    "net.4.weight", "net.4.bias",
})


def _extract_env_agnostic(sac):
    """Pull the env-agnostic subset of a SACTrainer's actor + critics
    (CPU tensors, ready to cache to disk)."""
    return {
        "actor": {k: v.detach().cpu()
                  for k, v in sac.policy.state_dict().items()
                  if k in ACTOR_ENV_AGNOSTIC_KEYS},
        "q1": {k: v.detach().cpu()
               for k, v in sac.q1.state_dict().items()
               if k in CRITIC_ENV_AGNOSTIC_KEYS},
        "q2": {k: v.detach().cpu()
               for k, v in sac.q2.state_dict().items()
               if k in CRITIC_ENV_AGNOSTIC_KEYS},
    }


def _load_env_agnostic(sac, snap):
    """Overlay the env-agnostic subset onto a target SACTrainer; also
    syncs the target networks (else they keep their random init while
    the main nets get warm-started, which destroys the boostrap target)."""
    sac.policy.load_state_dict(
        {**sac.policy.state_dict(),
         **{k: v.to(DEVICE) for k, v in snap["actor"].items()}})
    sac.q1.load_state_dict(
        {**sac.q1.state_dict(),
         **{k: v.to(DEVICE) for k, v in snap["q1"].items()}})
    sac.q2.load_state_dict(
        {**sac.q2.state_dict(),
         **{k: v.to(DEVICE) for k, v in snap["q2"].items()}})
    sac.q1_target.load_state_dict(sac.q1.state_dict())
    sac.q2_target.load_state_dict(sac.q2.state_dict())


def _permute_policy(snap, seed):
    """Permute each parameter tensor's elements within itself
    (scale/rank-matched, structure destroyed) — the v3.20-style
    structure control, applied to the policy subset."""
    g = torch.Generator().manual_seed(seed + 90_000)
    return {
        section: {k: v.flatten()[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
                  for k, v in tensors.items()}
        for section, tensors in snap.items()
    }


def _average_policies(snap_a, snap_b):
    """Element-wise mean of two env-agnostic policy subsets."""
    out = {}
    for section in snap_a:
        assert set(snap_a[section]) == set(snap_b[section]), \
            f"key mismatch in section {section}"
        out[section] = {k: (snap_a[section][k] + snap_b[section][k]) / 2.0
                        for k in snap_a[section]}
    return out


def _build_arm(env_cls, cfg, arm, snapshots, seed):
    """Build one raw-obs SAC arm with the appropriate policy weights
    loaded. NO RSSM in the target — SAC reads raw obs directly."""
    env = env_cls(cfg["num_envs"])
    sac = SACTrainer(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    curiosity = DeviceForwardCuriosity(OBS_DIM, ACTION_DIM)
    normalizer = DeviceRunningNormalizer(OBS_DIM)

    mcc = snapshots["mcc"]
    pen = snapshots["pen"]
    if arm == "transfer_mcc_pol":
        snap = mcc
    elif arm == "transfer_pen_pol":
        snap = pen
    elif arm == "transfer_avg_pol":
        snap = _average_policies(mcc, pen)
    elif arm == "permuted_pol":
        snap = _permute_policy(mcc, seed)
    else:                                        # scratch_pol
        snap = None
    if snap is not None:
        _load_env_agnostic(sac, snap)
    return env, sac, curiosity, normalizer


def _train_iter(env, sac, curiosity, normalizer, horizon, sac_updates):
    batch = collect_rollout(env, sac.device_policy_fn, horizon,
                            normalizer=normalizer)
    intrinsic = curiosity.intrinsic_reward(batch)
    sac_batch = dataclasses.replace(batch, rewards=batch.rewards + intrinsic)
    sac.train_on_rollout(sac_batch, n_updates=sac_updates)
    curiosity.train(batch)
    normalizer.update(batch.raw_obs.reshape(-1, OBS_DIM))
    return batch.total_steps


def _greedy_eval(env_cls, sac, normalizer, steps, num_envs):
    """Raw-obs greedy SAC eval — mirrors DeviceAgent.evaluate but here
    we don't have a DeviceAgent wrapper, so do it inline. n*T parallel
    eval episodes; report mean completed-episode return."""
    import torch.nn.functional as F
    env = env_cls(num_envs)
    n = env.num_envs
    ret = torch.zeros(n, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        raw = env.state
        obs = normalizer.normalize(raw) if normalizer is not None else raw
        with torch.no_grad():
            mean, _ = sac.policy.forward(obs)
            action = sac.policy._rescale(torch.tanh(mean))
        _, reward, _t, _tr, done = env.step(action)
        done = done.float()
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
    return (ret_sum / ep_count.clamp(min=1.0)).item()


def _run_arm(arm, env_cls, cfg, snapshots, seed):
    env, sac, curiosity, normalizer = _build_arm(env_cls, cfg, arm, snapshots, seed)
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, sac, curiosity, normalizer,
                             cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = _greedy_eval(env_cls, sac, normalizer,
                                 cfg["eval_steps"], cfg["num_envs"])
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


def _load_or_train_source(env_cls, cfg, snap_path, src_seed, label, eval_steps):
    """Train a DeviceAgent on `env_cls` and cache the env-agnostic
    SAC policy subset."""
    if os.path.exists(snap_path):
        print(f"[source/{label}] cached: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source/{label}] training ({cfg['source_rollouts']} rollouts)...",
          flush=True)
    torch.manual_seed(src_seed)
    np.random.seed(src_seed)
    agent = DeviceAgent(env_cls,
                        num_envs=cfg["num_envs"], horizon=cfg["horizon"],
                        sac_updates=cfg["source_sac_updates"],
                        curiosity_warmup=6)
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        agent.train_iteration()
        if it == 1 or it % cfg["eval_every"] == 0:
            score = agent.evaluate(steps=eval_steps, n_envs=cfg["num_envs"])
            print(f"  [source/{label}] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | eval {score:8.2f}",
                  flush=True)
    # agent.real is the SACTrainer for continuous envs.
    snap = _extract_env_agnostic(agent.real)
    torch.save(snap, snap_path)
    print(f"[source/{label}] done in {time.perf_counter() - t0:.0f}s, "
          f"policy subset ({sum(len(s) for s in snap.values())} tensors) -> {snap_path}",
          flush=True)
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
        print(f"  diff {name:>38}: mean {m:+7.2f}  95%CI +/-{c:6.2f}  "
              f"({pos}/{len(vals)} > 0){marker}")
        return {"mean": m, "ci95": c, "n_positive": pos}

    print(f"\n  pairwise diffs:")
    d_mcc_vs_scratch = _diff_stat(
        "transfer_mcc_pol - scratch_pol",
        [r["transfer_mcc_pol_auc"] - r["scratch_pol_auc"] for r in results])
    d_avg_vs_scratch = _diff_stat(
        "transfer_avg_pol - scratch_pol",
        [r["transfer_avg_pol_auc"] - r["scratch_pol_auc"] for r in results])
    d_avg_vs_perm = _diff_stat(
        "transfer_avg_pol - permuted_pol",
        [r["transfer_avg_pol_auc"] - r["permuted_pol_auc"] for r in results])
    d_avg_vs_best = _diff_stat(
        "transfer_avg_pol - max(mcc, pen)",
        [r["transfer_avg_pol_auc"]
         - max(r["transfer_mcc_pol_auc"], r["transfer_pen_pol_auc"])
         for r in results])

    single_works = (len(results) >= 2 and
                    d_mcc_vs_scratch["mean"] - d_mcc_vs_scratch["ci95"] > 0)
    comp_works = (len(results) >= 2 and
                  d_avg_vs_best["mean"] - d_avg_vs_best["ci95"] > 0)
    if comp_works:
        verdict = ("POLICY-SUBSTRATE COMPOSITION WORKS: transfer_avg_pol > "
                   "max(single), CI excludes 0 -- two SAC policies compose "
                   "where their RSSM cores did not")
    elif single_works:
        verdict = ("Single-skill policy transfer works (transfer_mcc_pol > "
                   "scratch_pol), but averaging composition does not -- "
                   "the substrate matters for single transfer but not "
                   "composition; mechanism is the issue regardless of substrate")
    elif len(results) >= 2 and (d_avg_vs_best["mean"] + d_avg_vs_best["ci95"] < 0):
        verdict = ("POLICY-SUBSTRATE COMPOSITION HURTS; v3.23 contingent")
    else:
        verdict = ("NO POLICY-SUBSTRATE COMPOSITION OR SINGLE TRANSFER "
                   "EFFECT RESOLVED (all CIs span 0)")
    print(f"\n  -> {verdict}")
    return {"arm_means": arm_mean,
            "diff_mcc_vs_scratch": d_mcc_vs_scratch,
            "diff_avg_vs_scratch": d_avg_vs_scratch,
            "diff_avg_vs_permuted": d_avg_vs_perm,
            "diff_avg_vs_best_single": d_avg_vs_best,
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
    parser.add_argument("--out-dir", default="transfer_v322_out")
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
    mcc_snap_path = os.path.join(args.out_dir, "source_mcc_policy.pt")
    pen_snap_path = os.path.join(args.out_dir, "source_pen_policy.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v322] device={DEVICE}  seeds={args.seeds}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}  "
          f"(substrate=POLICY weights)",
          flush=True)
    t0 = time.perf_counter()

    snapshots = {
        "mcc": _load_or_train_source(DeviceVecMountainCarContinuous, cfg,
                                     mcc_snap_path, src_seed=1000, label="mcc",
                                     eval_steps=999),
        "pen": _load_or_train_source(DeviceVecPendulum, cfg,
                                     pen_snap_path, src_seed=3000, label="pen",
                                     eval_steps=200),
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
        best_single = max(aucs["transfer_mcc_pol"], aucs["transfer_pen_pol"])
        comp_diff = aucs["transfer_avg_pol"] - best_single
        print(f"  seed {seed} | "
              f"scratch {aucs['scratch_pol']:+6.1f}  "
              f"permuted {aucs['permuted_pol']:+6.1f}  "
              f"mcc {aucs['transfer_mcc_pol']:+6.1f}  "
              f"pen {aucs['transfer_pen_pol']:+6.1f}  "
              f"avg {aucs['transfer_avg_pol']:+6.1f}  | "
              f"composition diff {comp_diff:+6.1f} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)

    print(f"\n{'=' * 80}\n  N={len(results)} seeds  |  target=MCC-Hard, "
          f"substrate=POLICY weights, composition=averaging")
    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"\n  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
