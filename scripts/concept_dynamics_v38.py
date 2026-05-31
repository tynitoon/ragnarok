"""v38 — learn the NOTION of DYNAMICS once, reuse it (by planning) to solve MANY
different tasks with ZERO extra learning. The reliable, large-payoff form of
concept reuse (model-based), in a HARD context (inertia -> exploration is hard for
model-free). This is where reuse genuinely pays (cf. the v36/v37 nulls: easy tasks
don't need stored knowledge; here the dynamics are costly to (re)learn per task).

Setup: a 1-D point with INERTIA. action in {-,0,+} thrust; vel = (vel+force)*drag;
pos += vel; walls stop it. Three DIFFERENT tasks share the SAME dynamics:
  REACH  (be at target),  STOP (be at target AND still),  PARK-CENTER (be at 0.5
  AND still). The 'notion' = a learned forward model M(pos,vel,force)->(pos',vel').

Demonstration:
- Learn M ONCE from random interaction (varied experience), B env-steps total.
- REUSE M by planning (random-shooting MPC) to solve ALL THREE tasks zero-shot —
  no per-task training, just a different planning objective.
- Baseline: model-free PPO must be TRAINED PER TASK from scratch.
Measure success rate per task and the env-steps each path costs. The dynamics
notion is learned once and reused across tasks -> huge, reliable amortisation.

Usage: python -m scripts.concept_dynamics_v38 [--mf-iters 150] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, PPONet

FORCE, DRAG = 0.06, 0.90
TOL, VTOL = 0.06, 0.02


def dyn(pos, vel, action):
    force = (action.float() - 1.0) * FORCE          # action 0,1,2 -> -F,0,+F
    vel = (vel + force) * DRAG
    pos = (pos + vel).clamp(0.0, 1.0)
    vel = torch.where((pos <= 0.0) | (pos >= 1.0), torch.zeros_like(vel), vel)
    return pos, vel


# task targets/costs (all share the same dynamics)
TASKS = {
    "reach": dict(stop=False, center=False),
    "stop": dict(stop=True, center=False),
    "park_center": dict(stop=True, center=True),
}


def success(pos, vel, target, task):
    ok = (pos - target).abs() <= TOL
    if TASKS[task]["stop"]:
        ok = ok & (vel.abs() <= VTOL)
    return ok


class ReachEnv:
    """Model-free substrate for one task. obs = [pos, vel, target]."""
    def __init__(self, n, task, max_steps=60, seed=0):
        self.n, self.task, self.max_steps = n, task, max_steps
        self.action_dim, self.obs_dim = 3, 3
        self._g = torch.Generator(device=DEVICE); self._g.manual_seed(seed)
        self._reset_all()
        self.cum_succ = torch.zeros(n, device=DEVICE)
        self.cum_ep = torch.zeros(n, device=DEVICE)

    def _tgt(self):
        if TASKS[self.task]["center"]:
            return torch.full((self.n,), 0.5, device=DEVICE)
        return torch.rand(self.n, generator=self._g, device=DEVICE) * 0.8 + 0.1

    def _reset_all(self):
        self.pos = torch.rand(self.n, generator=self._g, device=DEVICE)
        self.vel = torch.zeros(self.n, device=DEVICE)
        self.target = self._tgt()
        self.steps = torch.zeros(self.n, dtype=torch.long, device=DEVICE)

    @property
    def state(self):
        return torch.stack([self.pos, self.vel, self.target], -1)

    def step(self, action):
        self.pos, self.vel = dyn(self.pos, self.vel, action)
        self.steps += 1
        succ = success(self.pos, self.vel, self.target, self.task)
        timeout = self.steps >= self.max_steps
        done = succ | timeout
        reward = succ.float() - (self.pos - self.target).abs() * 0.02
        self.cum_succ += succ.float(); self.cum_ep += done.float()
        if bool(done.any()):
            d = done
            self.pos = torch.where(d, torch.rand(self.n, generator=self._g, device=DEVICE), self.pos)
            self.vel = torch.where(d, torch.zeros_like(self.vel), self.vel)
            self.target = torch.where(d, self._tgt(), self.target)
            self.steps = torch.where(d, torch.zeros_like(self.steps), self.steps)
        return self.state, reward, succ, timeout, done

    def succ_rate(self):
        return float(self.cum_succ.sum() / self.cum_ep.sum().clamp(min=1))


def learn_dynamics(steps, n, gen):
    """Learn M(pos,vel,force)->(pos',vel') from RANDOM interaction (varied)."""
    M = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 2)).to(DEVICE)
    opt = torch.optim.Adam(M.parameters(), 1e-3)
    pos = torch.rand(n, generator=gen, device=DEVICE)
    vel = torch.zeros(n, device=DEVICE)
    env_steps, last = 0, 0.0
    for _ in range(steps):
        a = torch.randint(0, 3, (n,), generator=gen, device=DEVICE)
        force = (a.float() - 1.0) * FORCE
        npos, nvel = dyn(pos, vel, a)
        x = torch.stack([pos, vel, force], -1)
        y = torch.stack([npos, nvel], -1)
        loss = F.mse_loss(M(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        env_steps += n
        pos, vel = npos.detach(), nvel.detach()
        # occasionally re-scatter to cover the state space (varied experience)
        resc = torch.rand(n, generator=gen, device=DEVICE) < 0.05
        pos = torch.where(resc, torch.rand(n, generator=gen, device=DEVICE), pos)
        vel = torch.where(resc, torch.zeros_like(vel), vel)
        last = float(loss.detach())
    return M, env_steps, last


@torch.no_grad()
def plan_step(M, pos, vel, target, task, H=16, S=128):
    """Random-shooting MPC in the LEARNED model M: pick the action sequence whose
    rolled-out end best satisfies the task, return its first action."""
    n = pos.shape[0]
    seqs = torch.randint(0, 3, (n, S, H), device=DEVICE)
    p = pos.view(n, 1).expand(n, S).clone()
    v = vel.view(n, 1).expand(n, S).clone()
    for h in range(H):
        a = seqs[:, :, h]
        force = (a.float() - 1.0) * FORCE
        x = torch.stack([p, v, force], -1).reshape(n * S, 3)
        out = M(x).reshape(n, S, 2)
        p, v = out[..., 0].clamp(0, 1), out[..., 1]
    tgt = target.view(n, 1)
    cost = (p - tgt).abs()
    if TASKS[task]["stop"]:
        cost = cost + 2.0 * v.abs()
    best = cost.argmin(1)                                # (n,)
    return seqs[torch.arange(n, device=DEVICE), best, 0]


@torch.no_grad()
def eval_planner(M, task, n=256, steps=300, seed=11):
    env = ReachEnv(n, task, seed=seed)
    obs = env.state
    for _ in range(steps):
        a = plan_step(M, env.pos, env.vel, env.target, task)
        obs, _, _, _, _ = env.step(a)
    return env.succ_rate()


@torch.no_grad()
def eval_policy(ppo, task, n=256, steps=300, seed=11):
    env = ReachEnv(n, task, seed=seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return env.succ_rate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dyn-steps", type=int, default=1500)
    p.add_argument("--mf-iters", type=int, default=150)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.dyn_steps, args.mf_iters, args.num_envs = 300, 30, 64

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    M, dyn_env_steps, mse = learn_dynamics(args.dyn_steps, args.num_envs, gen)
    print(f"[v38] device={DEVICE} | learn DYNAMICS once ({dyn_env_steps:,} env-steps, "
          f"MSE {mse:.2e}) -> REUSE by planning on 3 tasks; vs model-free PER task",
          flush=True)
    t0 = time.perf_counter()

    rows = {}
    for task in TASKS:
        mb = eval_planner(M, task)                       # zero extra training
        # model-free: train PPO from scratch on this task
        env = ReachEnv(args.num_envs, task, seed=args.seed)
        net = PPONet(env.obs_dim, env.action_dim, hidden=128)
        ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
        for _ in range(args.mf_iters):
            ppo.train_iter(env, 32)
        mf = eval_policy(ppo, task)
        mf_steps = ppo.total_steps
        rows[task] = dict(model_based=round(mb, 3), model_free=round(mf, 3),
                          mf_env_steps=mf_steps)
        print(f"  {task:12s}: model-based(plan) {mb:.2f} (0 task-steps) vs "
              f"model-free {mf:.2f} ({mf_steps:,} task env-steps) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    mb_mean = sum(r["model_based"] for r in rows.values()) / len(rows)
    mf_total_steps = sum(r["mf_env_steps"] for r in rows.values())
    mb_str = ", ".join(f"{k}:{v['model_based']}" for k, v in rows.items())
    ok = mb_mean >= 0.7 and all(r["model_based"] >= 0.6 for r in rows.values())
    verdict = (
        f"NOTION LEARNED ONCE, REUSED RELIABLY ACROSS TASKS — one dynamics model "
        f"(learned from {dyn_env_steps:,} random env-steps) solves ALL 3 different "
        f"tasks by planning, zero per-task training, mean success {mb_mean:.0%}; a "
        f"model-free agent must train SEPARATELY per task ({mf_total_steps:,} task "
        f"env-steps total). The dynamics NOTION amortises across every task that uses "
        f"it — reliable reuse in NEW contexts, in the HARD (inertia) regime where it "
        f"actually pays. (Contrast with v36/v37: easy tasks needed no stored notion; "
        f"here the shared dynamics are reused wholesale.)"
        if ok else
        f"PARTIAL — model-based per task [{mb_str}], mean {mb_mean:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v38_concept_dynamics.json"), "w") as f:
        json.dump(dict(dyn_env_steps=dyn_env_steps, dyn_mse=mse, rows=rows,
                       mb_mean=mb_mean, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
