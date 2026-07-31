"""v4.0 Phase 1 — model-based control: a reused world model solves new
goals by planning (CEM-MPC), demonstrating "understanding the world
makes new goals cheap".

Preregistered as preregistration.md amendment v4.0 (committed before
this script and any run).

Pipeline:
  1. Train an RSSM world model on DeviceVecPointMass2D dynamics,
     GOAL-AGNOSTIC (random-action exploration; goals resampled each
     episode but unused for model training).
  2. FREEZE it.
  3. For each held-out GOAL, solve by CEM-MPC planning in the frozen
     model — reward = analytic -distance(decoded_xy, goal). ZERO policy
     or reward learning on the new goal.
  Arms: mpc_trained (planning in the trained model), mpc_untrained
  (planning in a fresh RANDOM RSSM — control), and (full run) sac_scratch
  (from-scratch SAC per goal — the cost of mastering a goal without
  reuse). Endpoint: per-goal success + the faster-and-faster curve
  (total env-steps to master K goals).

Usage: python -m scripts.mbrl_pointmass_v4 [--validate] [--smoke]
"""

import argparse
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.planner import cem_plan
from ragnarok.learning.rollout import collect_rollout
from ragnarok.environments.device_env import DeviceVecPointMass2D

OBS_DIM, ACTION_DIM = 4, 2


def _random_policy_fn(obs):
    n = obs.shape[0]
    a = (torch.rand(n, ACTION_DIM, device=obs.device) - 0.5) * 2.0
    zeros = torch.zeros(n, device=obs.device)
    return a, zeros, zeros


def _train_world_model(cfg):
    """Train an RSSM on point-mass dynamics, goal-agnostic random data."""
    print(f"[wm] training world model on point-mass dynamics "
          f"({cfg['wm_rollouts']} rollouts, random actions)...", flush=True)
    rssm = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=3e-4)
    env = DeviceVecPointMass2D(cfg["num_envs"])      # goal=None: resampled
    t0 = time.perf_counter()
    for it in range(1, cfg["wm_rollouts"] + 1):
        batch = collect_rollout(env, _random_policy_fn, cfg["horizon"])
        metrics = wm.train_world_model_on_rollout(batch, epochs=cfg["wm_epochs"])
        if it == 1 or it % 10 == 0:
            recon = _recon_error(rssm, env, cfg["horizon"])
            print(f"  [wm] iter {it:>3}/{cfg['wm_rollouts']} | "
                  f"recon_err {recon:.4f} | "
                  f"loss {metrics.get('wm/total_loss', metrics.get('total_loss', float('nan'))):.3f}",
                  flush=True)
    print(f"[wm] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return rssm


@torch.no_grad()
def _recon_error(rssm, env, horizon):
    """One-step decoder reconstruction error on a fresh random rollout —
    a quick proxy for world-model quality."""
    batch = collect_rollout(env, _random_policy_fn, horizon)
    out = rssm.observe(batch.obs, batch.actions)
    recon = rssm.decoder(torch.cat([out["h"], out["z"]], dim=-1))
    return float(((recon - batch.obs) ** 2).mean().item())


@torch.no_grad()
def _mpc_eval(rssm, goal, cfg, n_trials):
    """Run CEM-MPC toward `goal` in `rssm` over n_trials parallel envs
    (random starts). Returns (success_fraction, mean_first_reach_step)."""
    env = DeviceVecPointMass2D(n_trials, goal=goal)
    g = torch.tensor(goal, dtype=torch.float32, device=DEVICE)

    def reward_fn(obs_pred):
        return -torch.norm(obs_pred[:, :2] - g, dim=-1)

    h, z = rssm.initial_state(n_trials, DEVICE)
    prev_a = torch.zeros(n_trials, ACTION_DIM, device=DEVICE)
    reached = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    first = torch.zeros(n_trials, device=DEVICE)
    for t in range(cfg["eval_steps"]):
        obs = env.state
        h, z = rssm.encode_observation(obs, h, z, prev_a, deterministic=True)
        a, _ = cem_plan(rssm, h, z, reward_fn, horizon=cfg["horizon_plan"],
                        n_cand=cfg["n_cand"], n_elite=cfg["n_elite"],
                        n_iters=cfg["n_iters"], action_dim=ACTION_DIM)
        _, _, term, _, _ = env.step(a)
        newly = term & (~reached)
        first = torch.where(newly, torch.full_like(first, float(t + 1)), first)
        reached = reached | term
        prev_a = a
    succ = float(reached.float().mean().item())
    steps = float(first[reached].mean().item()) if bool(reached.any()) else float("nan")
    return succ, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm-rollouts", type=int, default=60)
    parser.add_argument("--wm-epochs", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--horizon-plan", type=int, default=20)
    parser.add_argument("--n-cand", type=int, default=200)
    parser.add_argument("--n-elite", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--n-goals", type=int, default=8)
    parser.add_argument("--validate", action="store_true",
                        help="quick thesis check: train WM + MPC trained vs "
                             "untrained on a few goals")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.wm_rollouts, args.wm_epochs, args.num_envs, args.horizon = 6, 2, 64, 32
        args.n_cand, args.n_iters, args.eval_steps = 64, 2, 40
        args.n_trials, args.n_goals = 16, 2

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "horizon", "horizon_plan", "n_cand", "n_elite",
            "n_iters", "eval_steps", "wm_rollouts", "wm_epochs")}

    print(f"[mbrl-v4] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    rssm = _train_world_model(cfg)
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)

    untrained = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    untrained.eval()

    # Fixed held-out goals (deterministic grid-ish sample).
    rng = np.random.default_rng(0)
    goals = [tuple(rng.uniform(-0.8, 0.8, size=2).round(3)) for _ in range(args.n_goals)]

    print(f"\n[eval] MPC on {len(goals)} held-out goals "
          f"(n_trials={args.n_trials} each)\n", flush=True)
    tr_succ, tr_steps, un_succ = [], [], []
    for gi, g in enumerate(goals):
        s_tr, st_tr = _mpc_eval(rssm, g, cfg, args.n_trials)
        s_un, _ = _mpc_eval(untrained, g, cfg, args.n_trials)
        tr_succ.append(s_tr); un_succ.append(s_un)
        if not np.isnan(st_tr):
            tr_steps.append(st_tr)
        print(f"  goal {gi} {g} | mpc_trained succ {s_tr:.2f} "
              f"(reach@{st_tr:.0f}) | mpc_untrained succ {s_un:.2f}", flush=True)

    print(f"\n{'=' * 64}")
    print(f"  mpc_trained:   mean success {np.mean(tr_succ):.2f}  "
          f"mean steps-to-goal {np.mean(tr_steps) if tr_steps else float('nan'):.1f}")
    print(f"  mpc_untrained: mean success {np.mean(un_succ):.2f}")
    verdict = ("MODEL-BASED FOUNDATION WORKS — a reused world model solves "
               "new goals by planning, with ZERO new learning"
               if np.mean(tr_succ) > 0.6 and np.mean(tr_succ) - np.mean(un_succ) > 0.3
               else "CHECK — trained-model planning not clearly solving goals")
    print(f"  -> {verdict}")
    print(f"  {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
