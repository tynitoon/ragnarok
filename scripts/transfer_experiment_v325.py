"""v3.25 SEQUENTIAL multi-skill composition: a learned MANAGER switches
between two frozen skill policies used in DIFFERENT phases of a target
task.

Preregistered as preregistration.md amendment v3.25 (commit 6a73e1c,
committed before this script and any run).

v3.24 (hierarchical manager on the simultaneous CartPoleOnHill) hit the
"all arms fail" branch: that composite's two skills conflict over one
shared actuator, so a convex action blend cannot resolve them. v3.25
fixes the composite — a SEQUENTIAL task where the skills are used at
disjoint times, the regime where hierarchical / options composition is
sound.

Composite target (DeviceVecNavigateThenBalance, mode='composite'):
  - Phase 0: drive the cart up the MCC hill to the goal (pole rigid).
  - Phase 1 (on goal-reach): cart freezes, pole activates, +1/step
    pole-up, episode ends on pole-fall.
  A good agent navigates FAST then balances LONG — two skills, used
  at disjoint times.

Source single-skill tasks (share the composite's 5-d obs / 1-d action):
  - mode='nav':     phase-0 only, +100 on goal -> the NAVIGATE skill.
  - mode='balance': starts in phase 1, cart frozen -> the BALANCE skill.

Mechanism (unchanged from v3.24): two source SAC policies trained then
FROZEN; a learned SAC MANAGER (1-d action = blend weight w in [0,1])
applies a = w*a_nav + (1-w)*a_balance. Since the skills are used in
disjoint phases the manager's job is a temporal switch (w~1 in phase 0,
w~0 in phase 1; the phase is in the obs).

Four arms, N=8, target = composite:
  - scratch_mgr:           manager + 2 fresh random policies
  - transfer_nav_only:     manager + (nav policy, random)
  - transfer_balance_only: manager + (random, balance policy)
  - transfer_both:         manager + (nav policy, balance policy)

Decisive: transfer_both AUC minus max(transfer_nav_only,
transfer_balance_only) AUC, per seed, mean +/- Student-t 95% CI.

Usage: python -m scripts.transfer_experiment_v325 [--seeds N] [--smoke]
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
from ragnarok.learning.sac import SACTrainer, SACPolicy, DeviceSACBuffer
from ragnarok.learning.curiosity import DeviceForwardCuriosity
from ragnarok.learning.rollout import RolloutBatch
from ragnarok.environments.device_env import (
    DeviceVecNavigateThenBalance, DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 5, 1
ARMS = ("scratch_mgr", "transfer_nav_only",
        "transfer_balance_only", "transfer_both")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


# Thin mode-fixed env subclasses (DeviceAgent / rollouts call env_cls(n)).
class _NavEnv(DeviceVecNavigateThenBalance):
    def __init__(self, num_envs): super().__init__(num_envs, mode="nav")


class _BalanceEnv(DeviceVecNavigateThenBalance):
    def __init__(self, num_envs): super().__init__(num_envs, mode="balance")


class _CompositeEnv(DeviceVecNavigateThenBalance):
    def __init__(self, num_envs): super().__init__(num_envs, mode="composite")


@torch.no_grad()
def _greedy_action(policy: SACPolicy, obs: torch.Tensor) -> torch.Tensor:
    mean, _ = policy.forward(obs)
    return policy._rescale(torch.tanh(mean))


def _make_norm(snap):
    nrm = DeviceRunningNormalizer(OBS_DIM)
    if snap is not None:
        nrm.mean = snap["norm_mean"].to(DEVICE)
        nrm.var = snap["norm_var"].to(DEVICE)
    return nrm


def _load_policy(snap) -> SACPolicy:
    pol = SACPolicy(OBS_DIM, ACTION_DIM,
                    action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
                    action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32)).to(DEVICE)
    if snap is not None:
        pol.load_state_dict({k: v.to(DEVICE) for k, v in snap["policy"].items()})
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


def _train_source(env_cls, cfg, snap_path, name, seed):
    """Train a SAC source policy on a single-skill mode via the proven
    raw-obs DeviceAgent path; cache its policy weights + obs-normalizer."""
    if os.path.exists(snap_path):
        print(f"[source/{name}] cached: {snap_path}", flush=True)
        return torch.load(snap_path, weights_only=False)
    print(f"[source/{name}] training ({cfg['source_rollouts']} rollouts)...",
          flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = DeviceAgent(env_cls, num_envs=cfg["num_envs"], horizon=cfg["horizon"],
                        sac_updates=cfg["source_sac_updates"], curiosity_warmup=6)
    t0 = time.perf_counter()
    for it in range(1, cfg["source_rollouts"] + 1):
        agent.train_iteration()
        if it == 1 or it % cfg["eval_every"] == 0:
            score = agent.evaluate(steps=cfg["eval_steps"], n_envs=cfg["num_envs"])
            print(f"  [source/{name}] iter {it:>3}/{cfg['source_rollouts']} | "
                  f"env_steps {agent.total_env_steps:>10,} | eval {score:8.2f}",
                  flush=True)
    snap = {
        "policy": {k: v.detach().cpu()
                   for k, v in agent.real.policy.state_dict().items()},
        "norm_mean": agent.normalizer.mean.detach().cpu(),
        "norm_var": agent.normalizer.var.detach().cpu(),
    }
    torch.save(snap, snap_path)
    print(f"[source/{name}] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return snap


@torch.no_grad()
def _collect_manager_rollout(env, manager, opt_a, opt_b, norm_a, norm_b,
                             mgr_norm, horizon):
    """One manager rollout; the env steps a = w*a_nav + (1-w)*a_balance."""
    n = env.num_envs
    raw = env.state
    raw_l, obs_l, act_l, rew_l, done_l = [], [], [], [], []
    for _ in range(horizon):
        mgr_obs = mgr_norm.normalize(raw)
        w, _, _ = manager.device_policy_fn(mgr_obs)
        a_a = _greedy_action(opt_a, norm_a.normalize(raw))
        a_b = _greedy_action(opt_b, norm_b.normalize(raw))
        a_final = w * a_a + (1.0 - w) * a_b
        next_raw, reward, _t, _tr, done = env.step(a_final)
        raw_l.append(raw)
        obs_l.append(mgr_obs)
        act_l.append(w)
        rew_l.append(reward)
        done_l.append(done.float())
        raw = next_raw
    last_obs = mgr_norm.normalize(raw)
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
    )


@torch.no_grad()
def _evaluate_manager(env, manager, opt_a, opt_b, norm_a, norm_b,
                      mgr_norm, steps):
    env.reset()
    n = env.num_envs
    ret = torch.zeros(n, device=DEVICE)
    ret_sum = torch.zeros((), device=DEVICE)
    ep_count = torch.zeros((), device=DEVICE)
    for _ in range(steps):
        raw = env.state
        mgr_obs = mgr_norm.normalize(raw)
        mean, _ = manager.policy.forward(mgr_obs)
        w = manager.policy._rescale(torch.tanh(mean))
        a_a = _greedy_action(opt_a, norm_a.normalize(raw))
        a_b = _greedy_action(opt_b, norm_b.normalize(raw))
        a_final = w * a_a + (1.0 - w) * a_b
        _, reward, _t, _tr, done = env.step(a_final)
        done = done.float()
        ret = ret + reward
        ret_sum = ret_sum + (ret * done).sum()
        ep_count = ep_count + done.sum()
        ret = ret * (1.0 - done)
    return (ret_sum / ep_count.clamp(min=1.0)).item()


def _build_arm(cfg, arm, sources, seed):
    manager = SACTrainer(
        obs_dim=OBS_DIM, action_dim=1,
        action_low=np.array([0.0], dtype=np.float32),
        action_high=np.array([1.0], dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    curiosity = DeviceForwardCuriosity(OBS_DIM, 1)
    mgr_norm = DeviceRunningNormalizer(OBS_DIM)
    nav_snap = (sources["nav"]
                if arm in ("transfer_nav_only", "transfer_both") else None)
    bal_snap = (sources["balance"]
                if arm in ("transfer_balance_only", "transfer_both") else None)
    opt_a = _load_policy(nav_snap)
    opt_b = _load_policy(bal_snap)
    norm_a = _make_norm(nav_snap)
    norm_b = _make_norm(bal_snap)
    return manager, opt_a, opt_b, norm_a, norm_b, mgr_norm, curiosity


def _run_arm(arm, cfg, sources, seed):
    manager, opt_a, opt_b, norm_a, norm_b, mgr_norm, curiosity = _build_arm(
        cfg, arm, sources, seed)
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        batch = _collect_manager_rollout(
            _CompositeEnv(cfg["num_envs"]), manager, opt_a, opt_b,
            norm_a, norm_b, mgr_norm, cfg["horizon"])
        intrinsic = curiosity.intrinsic_reward(batch)
        mgr_batch = dataclasses.replace(batch, rewards=batch.rewards + intrinsic)
        manager.train_on_rollout(mgr_batch, n_updates=cfg["sac_updates"],
                                 obs_attr="obs", last_obs_attr="last_obs")
        curiosity.train(batch)
        mgr_norm.update(batch.raw_obs.reshape(-1, OBS_DIM))
        total += batch.total_steps
        if it == 1 or it % cfg["eval_every"] == 0:
            score = _evaluate_manager(
                _CompositeEnv(cfg["num_envs"]), manager, opt_a, opt_b,
                norm_a, norm_b, mgr_norm, cfg["eval_steps"])
            with torch.no_grad():
                peek = _CompositeEnv(min(64, cfg["num_envs"]))
                mean, _ = manager.policy.forward(mgr_norm.normalize(peek.state))
                w_mean = float(manager.policy._rescale(
                    torch.tanh(mean)).mean().item())
            curve.append([total, score])
            print(f"    [{arm:>21}] iter {it:>3}/{cfg['arm_rollouts']} | "
                  f"env_steps {total:>10,} | eval {score:9.2f} | "
                  f"mgr_w {w_mean:.3f}", flush=True)
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


def _summarize(results):
    arm_mean = {}
    for arm in ARMS:
        vals = [r[f"{arm}_auc"] for r in results]
        arm_mean[arm], arm_ci = _ci95(vals)
        print(f"  {arm:>21}: AUC mean {arm_mean[arm]:+9.2f}  95%CI +/-{arm_ci:8.2f}")

    def _diff_stat(name, vals):
        m, c = _ci95(vals)
        pos = sum(1 for v in vals if v > 0)
        marker = " *" if m - c > 0 else ("  " if m + c >= 0 else " -")
        print(f"  diff {name:>44}: mean {m:+9.2f}  95%CI +/-{c:8.2f}  "
              f"({pos}/{len(vals)} > 0){marker}")
        return {"mean": m, "ci95": c, "n_positive": pos}

    print(f"\n  pairwise diffs:")
    d_scr = _diff_stat("transfer_both - scratch_mgr",
                       [r["transfer_both_auc"] - r["scratch_mgr_auc"]
                        for r in results])
    d_nav = _diff_stat("transfer_both - transfer_nav_only",
                       [r["transfer_both_auc"] - r["transfer_nav_only_auc"]
                        for r in results])
    d_bal = _diff_stat("transfer_both - transfer_balance_only",
                       [r["transfer_both_auc"] - r["transfer_balance_only_auc"]
                        for r in results])
    d_best = _diff_stat("transfer_both - max(nav_only, balance_only)",
                        [r["transfer_both_auc"]
                         - max(r["transfer_nav_only_auc"],
                               r["transfer_balance_only_auc"])
                         for r in results])

    works = (len(results) >= 2 and d_best["mean"] - d_best["ci95"] > 0)
    if works:
        verdict = ("SEQUENTIAL COMPOSITION WORKS: a manager orchestrating two "
                   "frozen skills in sequence beats either skill alone "
                   "(CI excludes 0) -- the project's first composition positive")
    elif len(results) >= 2 and d_best["mean"] + d_best["ci95"] < 0:
        verdict = "SEQUENTIAL COMPOSITION HURTS vs the best single skill"
    else:
        verdict = "NO SEQUENTIAL COMPOSITION EFFECT RESOLVED (CI spans 0)"
    print(f"\n  -> {verdict}")
    return {"arm_means": arm_mean,
            "diff_both_vs_scratch": d_scr,
            "diff_both_vs_nav_only": d_nav,
            "diff_both_vs_balance_only": d_bal,
            "diff_both_vs_best_single": d_best,
            "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--source-rollouts", type=int, default=60)
    parser.add_argument("--arm-rollouts", type=int, default=60)
    parser.add_argument("--sac-updates", type=int, default=256)
    parser.add_argument("--source-sac-updates", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=999)
    parser.add_argument("--out-dir", default="transfer_v325_out")
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
    nav_path = os.path.join(args.out_dir, "source_nav.pt")
    balance_path = os.path.join(args.out_dir, "source_balance.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[transfer-v325] device={DEVICE}  seeds={args.seeds}  "
          f"arm_rollouts={args.arm_rollouts}  (sequential hierarchical)",
          flush=True)
    t0 = time.perf_counter()

    sources = {
        "nav": _train_source(_NavEnv, cfg, nav_path, "nav", seed=6000),
        "balance": _train_source(_BalanceEnv, cfg, balance_path, "balance",
                                 seed=7000),
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
        curves, aucs = {}, {}
        for arm in ARMS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            curves[arm] = _run_arm(arm, cfg, sources, seed)
            aucs[arm] = _auc(curves[arm])
        rec = {"seed": seed}
        for arm in ARMS:
            rec[f"{arm}_curve"] = curves[arm]
            rec[f"{arm}_auc"] = aucs[arm]
        results.append(rec)
        with open(results_path, "w") as f:
            json.dump({"seeds": results}, f, indent=2)
        best_single = max(aucs["transfer_nav_only"],
                          aucs["transfer_balance_only"])
        print(f"  seed {seed} | scratch {aucs['scratch_mgr']:+8.1f}  "
              f"nav_only {aucs['transfer_nav_only']:+8.1f}  "
              f"balance_only {aucs['transfer_balance_only']:+8.1f}  "
              f"both {aucs['transfer_both']:+8.1f}  | "
              f"composition diff {aucs['transfer_both'] - best_single:+8.1f} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)

    print(f"\n{'=' * 82}\n  N={len(results)} seeds  |  target=NavigateThenBalance, "
          f"mechanism=SEQUENTIAL hierarchical manager")
    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"\n  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
