"""v3.23 multi-skill composition via LEARNED SOFT GATING over two frozen
RSSM cores. Three static-composition mechanisms (v3.20 averaging RSSM
cores, v3.21 concatenation, v3.22 averaging policy weights) all failed
to produce additive composition. v3.23 tests whether a LEARNED gate
trained jointly with SAC can extract additive composition from two
skill cores where static fusion rules could not.

Preregistered as preregistration.md amendment v3.23 (commit 348bff7,
committed before this script and any run).

Mechanism. Two frozen RSSM cores (MCC and Pendulum sources, same as
v3.20/v3.21) produce two latents (160-d each). A small Gate MLP
takes the 320-d concat and outputs a scalar w in [0, 1] via sigmoid.
The combined latent is ``mixed = w * latent_a + (1 - w) * latent_b``
(160-d). SAC's actor and critics read `mixed`; the gate is SHARED
between actor and both critics (and their target nets); its
parameters get gradients from the SAC loss via the differentiable
path mixed -> actor/critic -> loss.

Architecture. Standard v3.21 dual-latent collection
(collect_rollout_dual_latent) → buffer stores aug_obs (320-d concat).
SACTrainer is instantiated with obs_dim=160 (the dim AFTER gating).
Its actor and critics are then WRAPPED with the Gate so they accept
320-d inputs internally compressed to 160-d. Gate parameters added to
the policy optimizer.

5 arms, N=8, same target (MCC-Hard):
  - scratch_gated:        2 fresh random cores + gate
  - permuted_gated:       MCC permuted + Pen permuted + gate
  - transfer_mcc_gated:   MCC core + 1 random + gate
  - transfer_pen_gated:   1 random + Pen core + gate
  - transfer_both_gated:  MCC core + Pen core + gate (composition)

Decisive: transfer_both_gated AUC minus max(transfer_mcc_gated,
transfer_pen_gated) AUC, per seed, mean +/- Student-t 95% CI.

Usage: python -m scripts.transfer_experiment_v323 [--seeds N] [--smoke]
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.curiosity import DeviceForwardCuriosity
from ragnarok.learning.rollout import collect_rollout_dual_latent, evaluate_dual_latent
from ragnarok.environments.device_env import (
    DeviceVecMountainCarContinuousHard, DeviceRunningNormalizer)

OBS_DIM, ACTION_DIM = 2, 1                       # TARGET task: MCC-Hard
ARMS = ("scratch_gated", "permuted_gated",
        "transfer_mcc_gated", "transfer_pen_gated", "transfer_both_gated")
_trapz = getattr(np, "trapezoid", np.trapz)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131}


def _permute_state_dict(state, seed):
    g = torch.Generator().manual_seed(seed + 90_000)
    return {k: v.flatten()[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
            for k, v in state.items()}


def _core_pair(arm, snapshots, seed):
    mcc, pen = snapshots["mcc"], snapshots["pen"]
    if arm == "scratch_gated":
        return None, None
    if arm == "permuted_gated":
        return _permute_state_dict(mcc, seed), _permute_state_dict(pen, seed + 1)
    if arm == "transfer_mcc_gated":
        return mcc, None
    if arm == "transfer_pen_gated":
        return None, pen
    if arm == "transfer_both_gated":
        return mcc, pen
    raise ValueError(f"unknown arm: {arm}")


class Gate(nn.Module):
    """Small MLP that maps a 2*state_dim concat to a scalar mix weight
    in [0, 1]. Shared between SAC's actor + critics."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        # Sigmoid -> w in [0, 1]; shape (N, 1).
        return torch.sigmoid(self.net(aug))


class GatedActor(nn.Module):
    """Wraps a SACPolicy that expects 160-d input; takes 320-d aug,
    gates it, forwards. Exposes the same interface as the base
    (forward / sample / _rescale / action_low / action_high)."""

    def __init__(self, base, gate: Gate, half_dim: int):
        super().__init__()
        self.base = base
        self.gate = gate
        self.half_dim = half_dim
        self.action_dim = base.action_dim

    @property
    def action_low(self):
        return self.base.action_low

    @property
    def action_high(self):
        return self.base.action_high

    def _mix(self, aug: torch.Tensor) -> torch.Tensor:
        w = self.gate(aug)
        return w * aug[..., :self.half_dim] + (1.0 - w) * aug[..., self.half_dim:]

    def forward(self, aug: torch.Tensor):
        return self.base(self._mix(aug))

    def sample(self, aug: torch.Tensor):
        return self.base.sample(self._mix(aug))

    def _rescale(self, x):
        return self.base._rescale(x)


class GatedCritic(nn.Module):
    """Wraps a QNetwork. Inputs are 320-d aug + action; gates aug to
    160-d, forwards to base (which expects 160-d obs + action)."""

    def __init__(self, base, gate: Gate, half_dim: int):
        super().__init__()
        self.base = base
        self.gate = gate
        self.half_dim = half_dim

    def _mix(self, aug: torch.Tensor) -> torch.Tensor:
        w = self.gate(aug)
        return w * aug[..., :self.half_dim] + (1.0 - w) * aug[..., self.half_dim:]

    def forward(self, aug: torch.Tensor, action: torch.Tensor):
        return self.base(self._mix(aug), action)


def _build_arm(env_cls, cfg, arm, snapshots, seed):
    """Build a frozen dual-RSSM arm with a learned soft gate over the
    two cores. Same architecture across all arms (5 arms differ only
    by what's loaded into the two RSSMs)."""
    env = env_cls(cfg["num_envs"])
    rssm_a = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    rssm_b = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    state_dim = rssm_a.state_dim
    dual_dim = 2 * state_dim                     # 320

    # SAC sees the GATED latent (160-d).
    sac = SACTrainer(
        obs_dim=state_dim, action_dim=ACTION_DIM,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        warmup_steps=cfg["num_envs"] * cfg["horizon"],
        buffer=DeviceSACBuffer(capacity=200_000))
    # IMPORTANT: the buffer stores dual_dim (320) aug_obs; the wrapped
    # actor/critics internally gate it back to state_dim (160).
    # Override buffer's obs_dim by re-creating it with the right dim.
    sac.replay = DeviceSACBuffer(capacity=200_000)
    # The SACBuffer auto-sizes on first add, so this is fine.

    # Build shared gate + wrap actor/critics.
    gate = Gate(dual_dim, hidden=64).to(DEVICE)
    sac.policy = GatedActor(sac.policy, gate, state_dim).to(DEVICE)
    sac.q1 = GatedCritic(sac.q1, gate, state_dim).to(DEVICE)
    sac.q2 = GatedCritic(sac.q2, gate, state_dim).to(DEVICE)
    sac.q1_target = GatedCritic(sac.q1_target, gate, state_dim).to(DEVICE)
    sac.q2_target = GatedCritic(sac.q2_target, gate, state_dim).to(DEVICE)

    # Add gate parameters to the policy optimizer so they train with the actor.
    sac.policy_optimizer.add_param_group({"params": gate.parameters()})
    # Refresh _q_params / _q_target_params cache (SAC's foreach soft-update
    # iterates over these; we want it to track the BASE critics' params only,
    # not the gate — the gate is updated via the policy optimizer and is
    # shared, so it does NOT participate in the EMA soft update).
    sac._q_params = (list(sac.q1.base.parameters()) +
                     list(sac.q2.base.parameters()))
    sac._q_target_params = (list(sac.q1_target.base.parameters()) +
                            list(sac.q2_target.base.parameters()))

    curiosity = DeviceForwardCuriosity(OBS_DIM, ACTION_DIM)
    normalizer = DeviceRunningNormalizer(OBS_DIM)
    aug_normalizer = DeviceRunningNormalizer(dual_dim)

    # Load cores into the two RSSMs (transfer arms only).
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
    return env, rssm_a, rssm_b, sac, gate, curiosity, normalizer, aug_normalizer


def _train_iter(env, rssm_a, rssm_b, sac, gate, curiosity, normalizer,
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
    env, rssm_a, rssm_b, sac, gate, curiosity, normalizer, aug_normalizer = (
        _build_arm(env_cls, cfg, arm, snapshots, seed))
    curve, total = [], 0
    for it in range(1, cfg["arm_rollouts"] + 1):
        total += _train_iter(env, rssm_a, rssm_b, sac, gate, curiosity,
                             normalizer, aug_normalizer,
                             cfg["horizon"], cfg["sac_updates"])
        if it == 1 or it % cfg["eval_every"] == 0:
            score = evaluate_dual_latent(env_cls(cfg["num_envs"]), rssm_a, rssm_b,
                                         sac.policy, cfg["eval_steps"],
                                         normalizer=normalizer,
                                         aug_normalizer=aug_normalizer,
                                         deterministic=True)
            # Track mean gate weight on a fresh eval batch for diagnostics.
            with torch.no_grad():
                # Use a small batch from the env to peek at the gate.
                env_peek = env_cls(min(64, cfg["num_envs"]))
                obs_peek = env_peek.state
                obs_norm = normalizer.normalize(obs_peek)
                h_a, z_a = rssm_a.initial_state(env_peek.num_envs, DEVICE)
                h_b, z_b = rssm_b.initial_state(env_peek.num_envs, DEVICE)
                prev_a = torch.zeros(env_peek.num_envs, ACTION_DIM, device=DEVICE)
                h_a, z_a = rssm_a.encode_observation(obs_norm, h_a, z_a, prev_a, deterministic=True)
                h_b, z_b = rssm_b.encode_observation(obs_norm, h_b, z_b, prev_a, deterministic=True)
                aug = torch.cat([h_a, z_a, h_b, z_b], dim=-1)
                aug_n = aug_normalizer.normalize(aug)
                w_mean = float(gate(aug_n).mean().item())
            curve.append([total, score])
            print(f"    [{arm:>21}] iter {it:>3}/{cfg['arm_rollouts']} | "
                  f"env_steps {total:>10,} | eval {score:8.2f} | "
                  f"gate_w {w_mean:.3f}", flush=True)
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
        n_pos = sum(1 for v in vals if v > 1.0)
        print(f"  {arm:>21}: AUC mean {arm_mean[arm]:+7.2f}  "
              f"95%CI +/-{arm_ci:6.2f}  ({n_pos}/{len(vals)} AUC>1)")

    def _diff_stat(name, vals):
        m, c = _ci95(vals)
        pos = sum(1 for v in vals if v > 0)
        marker = " *" if m - c > 0 else ("  " if m + c >= 0 else " -")
        print(f"  diff {name:>42}: mean {m:+7.2f}  95%CI +/-{c:6.2f}  "
              f"({pos}/{len(vals)} > 0){marker}")
        return {"mean": m, "ci95": c, "n_positive": pos}

    print(f"\n  pairwise diffs:")
    d_vs_scratch = _diff_stat(
        "transfer_both_gated - scratch_gated",
        [r["transfer_both_gated_auc"] - r["scratch_gated_auc"] for r in results])
    d_vs_perm = _diff_stat(
        "transfer_both_gated - permuted_gated",
        [r["transfer_both_gated_auc"] - r["permuted_gated_auc"] for r in results])
    d_vs_mcc = _diff_stat(
        "transfer_both_gated - transfer_mcc_gated",
        [r["transfer_both_gated_auc"] - r["transfer_mcc_gated_auc"]
         for r in results])
    d_vs_pen = _diff_stat(
        "transfer_both_gated - transfer_pen_gated",
        [r["transfer_both_gated_auc"] - r["transfer_pen_gated_auc"]
         for r in results])
    d_vs_best = _diff_stat(
        "transfer_both_gated - max(mcc, pen)",
        [r["transfer_both_gated_auc"]
         - max(r["transfer_mcc_gated_auc"], r["transfer_pen_gated_auc"])
         for r in results])

    works = (len(results) >= 2 and d_vs_best["mean"] - d_vs_best["ci95"] > 0)
    if works:
        verdict = ("GATED COMPOSITION WORKS: transfer_both_gated > max(single), "
                   "CI excludes 0 -- learned gating composes two skill cores")
    elif len(results) >= 2 and d_vs_best["mean"] + d_vs_best["ci95"] < 0:
        verdict = ("GATED COMPOSITION HURTS: gate adds optimisation difficulty "
                   "without value")
    else:
        verdict = ("NO GATED COMPOSITION EFFECT RESOLVED (CI spans 0)")
    print(f"\n  -> {verdict}")
    return {"arm_means": arm_mean,
            "diff_both_vs_scratch": d_vs_scratch,
            "diff_both_vs_permuted": d_vs_perm,
            "diff_both_vs_mcc": d_vs_mcc,
            "diff_both_vs_pen": d_vs_pen,
            "diff_both_vs_best_single": d_vs_best,
            "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--arm-rollouts", type=int, default=60)
    parser.add_argument("--sac-updates", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=999)
    parser.add_argument("--out-dir", default="transfer_v323_out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.arm_rollouts = 1, 5
        args.sac_updates = 8
        args.num_envs, args.horizon = 64, 32
        args.eval_every, args.eval_steps = 2, 300

    cfg = {"num_envs": args.num_envs, "horizon": args.horizon,
           "sac_updates": args.sac_updates,
           "arm_rollouts": args.arm_rollouts, "eval_every": args.eval_every,
           "eval_steps": args.eval_steps}

    os.makedirs(args.out_dir, exist_ok=True)
    mcc_snap_path = os.path.join(args.out_dir, "source_mcc_core.pt")
    pen_snap_path = os.path.join(args.out_dir, "source_pen_core.pt")
    results_path = os.path.join(args.out_dir, "results.json")

    # Reuse v3.20/v3.21 cached cores.
    if not os.path.exists(mcc_snap_path):
        import shutil
        shutil.copy("transfer_v311_out/source_snapshot.pt", mcc_snap_path)
        print(f"[source/mcc] copied from transfer_v311_out", flush=True)
    if not os.path.exists(pen_snap_path):
        import shutil
        shutil.copy("transfer_v316_out/source_pendulum_core.pt", pen_snap_path)
        print(f"[source/pen] copied from transfer_v316_out", flush=True)
    snapshots = {
        "mcc": torch.load(mcc_snap_path, weights_only=False),
        "pen": torch.load(pen_snap_path, weights_only=False),
    }
    print(f"[transfer-v323] device={DEVICE}  seeds={args.seeds}  "
          f"arm_rollouts={args.arm_rollouts}  sac_updates={args.sac_updates}  "
          f"(composition=LEARNED GATE)", flush=True)
    t0 = time.perf_counter()

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
        best_single = max(aucs["transfer_mcc_gated"], aucs["transfer_pen_gated"])
        comp_diff = aucs["transfer_both_gated"] - best_single
        print(f"  seed {seed} | "
              f"scratch {aucs['scratch_gated']:+6.1f}  "
              f"permuted {aucs['permuted_gated']:+6.1f}  "
              f"mcc {aucs['transfer_mcc_gated']:+6.1f}  "
              f"pen {aucs['transfer_pen_gated']:+6.1f}  "
              f"both {aucs['transfer_both_gated']:+6.1f}  | "
              f"composition diff {comp_diff:+6.1f} | "
              f"{time.perf_counter() - s0:.0f}s", flush=True)

    print(f"\n{'=' * 80}\n  N={len(results)} seeds  |  target=MCC-Hard, "
          f"composition mechanism=LEARNED SOFT GATE")
    summary = _summarize(results) if results else {}
    with open(results_path, "w") as f:
        json.dump({"seeds": results, "summary": summary}, f, indent=2)
    print(f"\n  {time.perf_counter() - t0:.0f}s  |  results -> {results_path}")


if __name__ == "__main__":
    main()
