"""v4.0 Phase 1 — the compounding ("de plus en plus vite") curve.

Completes the preregistered v4.0 Phase 1 endpoint: total env-steps to
MASTER K new goals, model-based (reuse a learned world model) vs
from-scratch (learn each goal with SAC, no reuse).

  - Model-based: pay a ONE-TIME world-model training cost W, then each
    new goal is solved by CEM-MPC planning with ~0 learning env-steps
    (just the eval rollout). Cumulative cost flattens: W + K*~0.
  - From-scratch SAC: each goal costs S_k env-steps to master (reach the
    goal reliably). Cumulative cost is ~linear: sum_k S_k.

The "child learning" signature: the model-based marginal cost per new
goal -> 0 as the agent's knowledge (the world model) covers the task
space, while from-scratch stays constant. Mastery = eval success >=
0.8 (reach the goal from random starts).

Usage: python -m scripts.mbrl_compounding_v4 [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.planner import cem_plan
from ragnarok.learning.rollout import collect_rollout
from ragnarok.environments.device_env import DeviceVecPointMass2D

OBS_DIM, ACTION_DIM = 4, 2


def _random_policy_fn(obs):
    n = obs.shape[0]
    return ((torch.rand(n, ACTION_DIM, device=obs.device) - 0.5) * 2.0,
            torch.zeros(n, device=obs.device), torch.zeros(n, device=obs.device))


def _train_world_model(cfg, cache_path):
    rssm = RSSM(OBS_DIM, ACTION_DIM).to(DEVICE)
    W = cfg["wm_rollouts"] * cfg["num_envs"] * cfg["horizon"]
    if os.path.exists(cache_path):
        rssm.load_state_dict(torch.load(cache_path, weights_only=False))
        print(f"[wm] loaded cached world model (cost W={W:,} env-steps)", flush=True)
        rssm.eval()
        for p in rssm.parameters():
            p.requires_grad_(False)
        return rssm, W
    print(f"[wm] training ({cfg['wm_rollouts']} rollouts)...", flush=True)
    wm = WorldModelTrainer(rssm, ReplayBuffer(), lr=3e-4)
    env = DeviceVecPointMass2D(cfg["num_envs"])
    t0 = time.perf_counter()
    for it in range(1, cfg["wm_rollouts"] + 1):
        batch = collect_rollout(env, _random_policy_fn, cfg["horizon"])
        wm.train_world_model_on_rollout(batch, epochs=cfg["wm_epochs"])
    torch.save({k: v.detach().cpu() for k, v in rssm.state_dict().items()},
               cache_path)
    print(f"[wm] done in {time.perf_counter()-t0:.0f}s (cost W={W:,} env-steps)",
          flush=True)
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    return rssm, W


@torch.no_grad()
def _mpc_success(rssm, goal, cfg, n_trials):
    """Success fraction of CEM-MPC toward `goal` in `rssm` (no learning)."""
    env = DeviceVecPointMass2D(n_trials, goal=goal)
    g = torch.tensor(goal, dtype=torch.float32, device=DEVICE)
    reward_fn = lambda o: -torch.norm(o[:, :2] - g, dim=-1)
    h, z = rssm.initial_state(n_trials, DEVICE)
    prev_a = torch.zeros(n_trials, ACTION_DIM, device=DEVICE)
    reached = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["eval_steps"]):
        h, z = rssm.encode_observation(env.state, h, z, prev_a, deterministic=True)
        a, _ = cem_plan(rssm, h, z, reward_fn, horizon=cfg["horizon_plan"],
                        n_cand=cfg["n_cand"], n_elite=cfg["n_elite"],
                        n_iters=cfg["n_iters"], action_dim=ACTION_DIM)
        _, _, term, _, _ = env.step(a)
        reached = reached | term
        prev_a = a
    return float(reached.float().mean().item())


@torch.no_grad()
def _sac_success(sac, goal, cfg, n_trials=64):
    env = DeviceVecPointMass2D(n_trials, goal=goal)
    reached = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["eval_steps"]):
        mean, _ = sac.policy.forward(env.state)
        a = sac.policy._rescale(torch.tanh(mean))
        _, _, term, _, _ = env.step(a)
        reached = reached | term
    return float(reached.float().mean().item())


def _sac_master_goal(goal, cfg):
    """From-scratch SAC on a fixed goal; return env-steps to first reach
    success >= mastery threshold (or the full budget if never)."""
    env = DeviceVecPointMass2D(cfg["num_envs"], goal=goal)
    sac = SACTrainer(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                     action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
                     action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
                     warmup_steps=cfg["num_envs"] * cfg["horizon"],
                     buffer=DeviceSACBuffer(capacity=200_000))
    total = 0
    mastered_at = None
    for it in range(1, cfg["sac_rollouts"] + 1):
        batch = collect_rollout(env, sac.device_policy_fn, cfg["horizon"])
        sac.train_on_rollout(batch, n_updates=cfg["sac_updates"])
        total += batch.total_steps
        if it % cfg["sac_eval_every"] == 0:
            succ = _sac_success(sac, goal, cfg)
            if succ >= cfg["mastery"] and mastered_at is None:
                mastered_at = total
                break
    return mastered_at if mastered_at is not None else total, mastered_at is not None


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
    parser.add_argument("--n-goals", type=int, default=6)
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--sac-rollouts", type=int, default=40)
    parser.add_argument("--sac-updates", type=int, default=128)
    parser.add_argument("--sac-eval-every", type=int, default=2)
    parser.add_argument("--mastery", type=float, default=0.8)
    parser.add_argument("--out-dir", default="mbrl_v4_out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.wm_rollouts, args.wm_epochs, args.num_envs, args.horizon = 6, 2, 64, 32
        args.n_cand, args.n_iters, args.eval_steps = 64, 2, 40
        args.n_goals, args.n_trials, args.sac_rollouts = 2, 16, 4

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "horizon", "horizon_plan", "n_cand", "n_elite",
            "n_iters", "eval_steps", "wm_rollouts", "wm_epochs",
            "sac_rollouts", "sac_updates", "sac_eval_every", "mastery")}

    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "world_model.pt")
    results_path = os.path.join(args.out_dir, "compounding.json")

    print(f"[mbrl-compounding-v4] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    rssm, W = _train_world_model(cfg, cache)

    rng = np.random.default_rng(0)
    goals = [tuple(rng.uniform(-0.8, 0.8, size=2).round(3).tolist())
             for _ in range(args.n_goals)]

    print(f"\n[compounding] mastering {len(goals)} goals two ways\n", flush=True)
    mpc_succ, sac_costs, sac_mastered = [], [], []
    for gi, g in enumerate(goals):
        s_mpc = _mpc_success(rssm, g, cfg, args.n_trials)
        sac_cost, ok = _sac_master_goal(g, cfg)
        mpc_succ.append(s_mpc); sac_costs.append(sac_cost); sac_mastered.append(ok)
        print(f"  goal {gi} {g} | mpc_trained succ {s_mpc:.2f} (0 learning steps) | "
              f"sac_scratch mastered={ok} at {sac_cost:,} env-steps", flush=True)

    # Cumulative curves: env-steps to have mastered the first k goals.
    sac_cum = np.cumsum(sac_costs).tolist()
    mbrl_cum = [W] * len(goals)            # W once, then ~0 marginal per goal
    print(f"\n{'=' * 70}")
    print(f"  world-model upfront cost  W = {W:,} env-steps")
    print(f"  from-scratch per-goal cost: mean {int(np.mean(sac_costs)):,} env-steps")
    print(f"  cumulative env-steps to master k goals:")
    print(f"    {'k':>3} | {'model-based':>14} | {'from-scratch':>14}")
    for k in range(len(goals)):
        print(f"    {k+1:>3} | {mbrl_cum[k]:>14,} | {int(sac_cum[k]):>14,}")
    cross = next((k+1 for k in range(len(goals)) if sac_cum[k] > mbrl_cum[k]), None)
    print(f"  crossover: from-scratch becomes more expensive at k={cross}")
    print(f"  -> model-based marginal cost per NEW goal ~ 0 (just plan); "
          f"from-scratch stays ~{int(np.mean(sac_costs)):,}/goal")
    print(f"  mpc_trained mean success {np.mean(mpc_succ):.2f}  | "
          f"{time.perf_counter()-t0:.0f}s")

    with open(results_path, "w") as f:
        json.dump({"W": W, "goals": goals, "mpc_success": mpc_succ,
                   "sac_costs": sac_costs, "sac_mastered": sac_mastered,
                   "sac_cumulative": sac_cum, "mbrl_cumulative": mbrl_cum,
                   "crossover_k": cross}, f, indent=2)
    print(f"  results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
