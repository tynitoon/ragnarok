"""v4.0 Phase 2 — hierarchical composition: reuse a "reach a point" skill
to learn a sequential (ordered-visit) task fast.

Preregistered as preregistration.md amendment v4.0 Phase 2 (+ the
implementation note: the reusable reach-skill is a fast goal-conditioned
policy, not MPC-in-the-loop). Committed before this run.

Pipeline:
  1. Pre-train the BASIC SKILL pi_lo(obs4, goal) -> action — a
     goal-conditioned SAC on point-mass (reach arbitrary goals). This is
     the "notion already known", learned once, FROZEN.
  2. Solve the COMPLEX task DeviceVecOrderedVisit (visit 3 zones in
     order, sparse reward) three ways:
       - hierarchical_reuse:      high-level SAC proposes sub-goals; the
                                  frozen pi_lo executes each for K steps.
       - hierarchical_untrained:  same, but pi_lo is a FRESH RANDOM policy
                                  (control: the reach-skill is absent).
       - flat_scratch:            flat SAC over primitive actions.
  Endpoint: env-steps to MASTER (completion success >= 0.8). The
  composition thesis: hierarchical_reuse masters in << env-steps than
  flat_scratch (or flat never masters), and >> hierarchical_untrained.

Usage: python -m scripts.hrl_ordered_visit_v4 [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.sac import SACTrainer, SACPolicy, DeviceSACBuffer
from ragnarok.learning.rollout import RolloutBatch, collect_rollout
from ragnarok.environments.device_env import (
    DeviceVecPointMass2D, DeviceVecOrderedVisit)

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306}


# --------------------------------------------------------------------------
# Basic skill: goal-conditioned "reach a point" policy.
# --------------------------------------------------------------------------
def _collect_goal_cond(env, sac, horizon):
    """Rollout on point-mass; obs fed to SAC is [state(4), goal(2)] = 6-d."""
    n = env.num_envs
    obs_l, act_l, rew_l, done_l = [], [], [], []
    for _ in range(horizon):
        lo_obs = torch.cat([env.state, env.goal], dim=-1)
        a, _, _ = sac.device_policy_fn(lo_obs)
        _, r, _t, _tr, done = env.step(a)
        obs_l.append(lo_obs); act_l.append(a); rew_l.append(r)
        done_l.append(done.float())
    last = torch.cat([env.state, env.goal], dim=-1)
    z = torch.zeros(n, horizon, device=DEVICE)
    return RolloutBatch(obs=torch.stack(obs_l, 1), raw_obs=torch.stack(obs_l, 1),
                        actions=torch.stack(act_l, 1), rewards=torch.stack(rew_l, 1),
                        dones=torch.stack(done_l, 1), logp=z, values=z,
                        last_obs=last, last_value=torch.zeros(n, device=DEVICE))


def _pretrain_low_level(cfg, cache):
    """Goal-conditioned SAC on point-mass — the reusable reach-skill."""
    sac = SACTrainer(obs_dim=6, action_dim=2,
                     action_low=np.full(2, -1.0, dtype=np.float32),
                     action_high=np.full(2, 1.0, dtype=np.float32),
                     warmup_steps=cfg["num_envs"] * cfg["horizon"],
                     buffer=DeviceSACBuffer(capacity=200_000))
    if os.path.exists(cache):
        sac.policy.load_state_dict(torch.load(cache, weights_only=False))
        print(f"[lo] loaded cached reach-skill: {cache}", flush=True)
        return sac.policy
    print(f"[lo] pre-training reach-skill ({cfg['lo_rollouts']} rollouts)...",
          flush=True)
    env = DeviceVecPointMass2D(cfg["num_envs"])
    t0 = time.perf_counter()
    for it in range(1, cfg["lo_rollouts"] + 1):
        batch = _collect_goal_cond(env, sac, cfg["horizon"])
        sac.train_on_rollout(batch, n_updates=cfg["lo_updates"])
        if it % 10 == 0:
            succ = _lo_success(sac.policy, cfg)
            print(f"  [lo] iter {it:>3}/{cfg['lo_rollouts']} | reach-success {succ:.2f}",
                  flush=True)
    torch.save({k: v.detach().cpu() for k, v in sac.policy.state_dict().items()},
               cache)
    print(f"[lo] done in {time.perf_counter()-t0:.0f}s", flush=True)
    return sac.policy


@torch.no_grad()
def _lo_action(pi_lo, obs4, g):
    mean, _ = pi_lo.forward(torch.cat([obs4, g], dim=-1))
    return pi_lo._rescale(torch.tanh(mean))


@torch.no_grad()
def _lo_success(pi_lo, cfg, n_trials=128):
    env = DeviceVecPointMass2D(n_trials)
    reached = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["eval_steps"]):
        a = _lo_action(pi_lo, env.state, env.goal)
        _, _, term, _, _ = env.step(a)
        reached = reached | term
    return float(reached.float().mean().item())


def _fresh_low_level():
    pol = SACPolicy(6, 2, action_low=np.full(2, -1.0, dtype=np.float32),
                    action_high=np.full(2, 1.0, dtype=np.float32)).to(DEVICE)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


# --------------------------------------------------------------------------
# Hierarchical high-level over the (frozen) reach-skill.
# --------------------------------------------------------------------------
def _collect_macro(env, hi, pi_lo, cfg):
    n = env.num_envs
    obs = env.state                                       # 5-d
    mo, mg, mr, md = [], [], [], []
    steps = 0
    for _ in range(cfg["n_macro"]):
        g, _, _ = hi.device_policy_fn(obs)                # (n, 2) sub-goal
        obs_start = obs
        r_acc = torch.zeros(n, device=DEVICE)
        d_acc = torch.zeros(n, device=DEVICE)
        for _ in range(cfg["macro_len"]):
            a = _lo_action(pi_lo, obs[:, :4], g)
            obs, r, _t, _tr, done = env.step(a)
            r_acc = r_acc + r
            d_acc = torch.maximum(d_acc, done.float())
            steps += n
        mo.append(obs_start); mg.append(g); mr.append(r_acc); md.append(d_acc)
    z = torch.zeros(n, cfg["n_macro"], device=DEVICE)
    batch = RolloutBatch(obs=torch.stack(mo, 1), raw_obs=torch.stack(mo, 1),
                         actions=torch.stack(mg, 1), rewards=torch.stack(mr, 1),
                         dones=torch.stack(md, 1), logp=z, values=z,
                         last_obs=obs, last_value=torch.zeros(n, device=DEVICE))
    return batch, steps


@torch.no_grad()
def _eval_hier(hi, pi_lo, cfg, n_trials=128):
    env = DeviceVecOrderedVisit(n_trials)
    obs = env.state
    completed = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["n_macro_eval"]):
        mean, _ = hi.policy.forward(obs)
        g = hi.policy._rescale(torch.tanh(mean))
        for _ in range(cfg["macro_len"]):
            a = _lo_action(pi_lo, obs[:, :4], g)
            obs, _, term, _, _ = env.step(a)
            completed = completed | term
    return float(completed.float().mean().item())


def _train_hier(pi_lo, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = DeviceVecOrderedVisit(cfg["num_envs"])
    hi = SACTrainer(obs_dim=5, action_dim=2,
                    action_low=np.full(2, -0.9, dtype=np.float32),
                    action_high=np.full(2, 0.9, dtype=np.float32),
                    warmup_steps=2 * cfg["num_envs"] * cfg["n_macro"],
                    buffer=DeviceSACBuffer(capacity=200_000))
    total, mastered_at = 0, None
    for it in range(1, cfg["hi_rollouts"] + 1):
        batch, steps = _collect_macro(env, hi, pi_lo, cfg)
        hi.train_on_rollout(batch, n_updates=cfg["hi_updates"],
                            obs_attr="obs", last_obs_attr="last_obs")
        total += steps
        if it % cfg["eval_every"] == 0:
            succ = _eval_hier(hi, pi_lo, cfg)
            tag = f"{succ:.2f}"
            if succ >= cfg["mastery"] and mastered_at is None:
                mastered_at = total
            print(f"    hier iter {it:>3}/{cfg['hi_rollouts']} | "
                  f"completion {tag} | env_steps {total:,}", flush=True)
            if mastered_at is not None:
                break
    final = _eval_hier(hi, pi_lo, cfg)
    return (mastered_at if mastered_at is not None else total), final, mastered_at is not None


# --------------------------------------------------------------------------
# Flat from-scratch baseline.
# --------------------------------------------------------------------------
@torch.no_grad()
def _eval_flat(sac, cfg, n_trials=128):
    env = DeviceVecOrderedVisit(n_trials)
    obs = env.state
    completed = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["n_macro_eval"] * cfg["macro_len"]):
        mean, _ = sac.policy.forward(obs)
        a = sac.policy._rescale(torch.tanh(mean))
        obs, _, term, _, _ = env.step(a)
        completed = completed | term
    return float(completed.float().mean().item())


def _train_flat(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = DeviceVecOrderedVisit(cfg["num_envs"])
    sac = SACTrainer(obs_dim=5, action_dim=2,
                     action_low=np.full(2, -1.0, dtype=np.float32),
                     action_high=np.full(2, 1.0, dtype=np.float32),
                     warmup_steps=cfg["num_envs"] * cfg["horizon"],
                     buffer=DeviceSACBuffer(capacity=200_000))
    total, mastered_at = 0, None
    for it in range(1, cfg["flat_rollouts"] + 1):
        batch = collect_rollout(env, sac.device_policy_fn, cfg["horizon"])
        sac.train_on_rollout(batch, n_updates=cfg["flat_updates"])
        total += batch.total_steps
        if it % cfg["eval_every"] == 0:
            succ = _eval_flat(sac, cfg)
            if succ >= cfg["mastery"] and mastered_at is None:
                mastered_at = total
            print(f"    flat iter {it:>3}/{cfg['flat_rollouts']} | "
                  f"completion {succ:.2f} | env_steps {total:,}", flush=True)
            if mastered_at is not None:
                break
    final = _eval_flat(sac, cfg)
    return (mastered_at if mastered_at is not None else total), final, mastered_at is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--lo-rollouts", type=int, default=50)
    parser.add_argument("--lo-updates", type=int, default=128)
    parser.add_argument("--macro-len", type=int, default=15)
    parser.add_argument("--n-macro", type=int, default=10)
    parser.add_argument("--n-macro-eval", type=int, default=12)
    parser.add_argument("--hi-rollouts", type=int, default=80)
    parser.add_argument("--hi-updates", type=int, default=64)
    parser.add_argument("--flat-rollouts", type=int, default=200)
    parser.add_argument("--flat-updates", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--mastery", type=float, default=0.8)
    parser.add_argument("--out-dir", default="hrl_v4_out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.num_envs, args.horizon = 1, 64, 32
        args.lo_rollouts, args.lo_updates = 8, 16
        args.hi_rollouts, args.flat_rollouts = 6, 6
        args.n_macro, args.macro_len, args.n_macro_eval = 6, 10, 8
        args.eval_every = 2

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "horizon", "lo_rollouts", "lo_updates", "macro_len",
            "n_macro", "n_macro_eval", "hi_rollouts", "hi_updates",
            "flat_rollouts", "flat_updates", "eval_every", "eval_steps",
            "mastery")}

    os.makedirs(args.out_dir, exist_ok=True)
    lo_cache = os.path.join(args.out_dir, "reach_skill.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[hrl-v4] device={DEVICE}  seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    pi_lo = _pretrain_low_level(cfg, lo_cache)
    lo_succ = _lo_success(pi_lo, cfg)
    print(f"[lo] reach-skill success on random goals: {lo_succ:.2f}\n", flush=True)
    pi_untrained = _fresh_low_level()

    rows = {"hier_reuse": [], "hier_untrained": [], "flat_scratch": []}
    for seed in range(args.seeds):
        print(f"[seed {seed}]", flush=True)
        c_hr, f_hr, ok_hr = _train_hier(pi_lo, cfg, seed)
        c_hu, f_hu, ok_hu = _train_hier(pi_untrained, cfg, seed)
        c_fs, f_fs, ok_fs = _train_flat(cfg, seed)
        rows["hier_reuse"].append((c_hr, f_hr, ok_hr))
        rows["hier_untrained"].append((c_hu, f_hu, ok_hu))
        rows["flat_scratch"].append((c_fs, f_fs, ok_fs))
        print(f"  seed {seed} | hier_reuse master={ok_hr}@{c_hr:,} (final {f_hr:.2f}) "
              f"| hier_untrained master={ok_hu}@{c_hu:,} (final {f_hu:.2f}) "
              f"| flat master={ok_fs}@{c_fs:,} (final {f_fs:.2f})", flush=True)

    print(f"\n{'=' * 70}\n  N={args.seeds}  |  task=OrderedVisit (sparse, 3 zones)")
    for arm in ("hier_reuse", "hier_untrained", "flat_scratch"):
        costs = [r[0] for r in rows[arm]]
        finals = [r[1] for r in rows[arm]]
        n_master = sum(1 for r in rows[arm] if r[2])
        print(f"  {arm:>16}: mastered {n_master}/{args.seeds} | "
              f"env-steps-to-master mean {int(np.mean(costs)):,} | "
              f"final completion mean {np.mean(finals):.2f}")
    hr_ok = sum(1 for r in rows["hier_reuse"] if r[2])
    fs_ok = sum(1 for r in rows["flat_scratch"] if r[2])
    hu_ok = sum(1 for r in rows["hier_untrained"] if r[2])
    if hr_ok >= max(1, args.seeds - 1) and hr_ok > fs_ok and hr_ok > hu_ok:
        verdict = ("COMPOSITION WORKS — reusing the reach-skill masters the "
                   "ordered visit; flat-scratch and untrained-skill do not "
                   "(or far slower)")
    elif hr_ok > 0 and hr_ok >= fs_ok:
        verdict = "PARTIAL — hierarchical reuse helps but check margins"
    else:
        verdict = "CHECK — hierarchical reuse did not clearly win"
    print(f"  -> {verdict}")
    print(f"  {time.perf_counter()-t0:.0f}s", flush=True)

    with open(results_path, "w") as f:
        json.dump({"reach_skill_success": lo_succ,
                   "rows": {k: [[int(c), float(fin), bool(ok)]
                                for (c, fin, ok) in v] for k, v in rows.items()},
                   "verdict": verdict}, f, indent=2)
    print(f"  results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
