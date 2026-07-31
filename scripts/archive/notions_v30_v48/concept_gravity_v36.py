"""v36 — learn a NOTION (gravity), then show that HAVING it -> solve faster BECAUSE
it USES it. The project's real thesis, cleanly.

Step 1 (learn the notion): a small model M learns gravity — predict where a
projectile will land (analytic landing under gravity + wall bounce), supervised.
Step 2 (a task that uses gravity): DeviceVecProjectileCatch — be at the ball's
LANDING y when it arrives (tracking its current y fails, because it arcs).
Step 3 (the demonstration): train two RL agents at equal conditions —
  WITH-CONCEPT: obs = [catcher_y, M's predicted landing]  (it HAS the notion)
  SCRATCH:      obs = [catcher_y, raw ball state]          (must re-infer gravity)
and measure iterations-to-competence (catch-rate >= 0.7). With the notion it
should solve in FAR fewer iterations.
Step 4 (prove it USES it): feed the with-concept agent a SCRAMBLED landing
prediction; if its catch-rate collapses, it was genuinely using the notion.

Usage: python -m scripts.concept_gravity_v36 [--iters 300] [--smoke]
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
from ragnarok.environments.projectile import DeviceVecProjectileCatch as Env

G, XP = 0.004, 0.97


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def true_landing(ball):
    bx, by, bvx, bvy = ball[:, 0], ball[:, 1], ball[:, 2], ball[:, 3]
    t = ((XP - bx) / bvx.clamp(min=1e-5)).clamp(min=0)
    yraw = by + bvy * t - 0.5 * G * t * t
    m = torch.remainder(yraw, 2.0)
    return torch.where(m <= 1.0, m, 2.0 - m)


def train_concept(steps, batch, g):
    M = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.Adam(M.parameters(), 1e-3)
    last = 0.0
    for _ in range(steps):
        bx = torch.rand(batch, 1, generator=g, device=DEVICE) * 0.9
        by = torch.rand(batch, 1, generator=g, device=DEVICE)
        bvx = torch.rand(batch, 1, generator=g, device=DEVICE) * 0.012 + 0.018
        bvy = torch.rand(batch, 1, generator=g, device=DEVICE) * 0.12 - 0.04
        ball = torch.cat([bx, by, bvx, bvy], 1)
        loss = F.mse_loss(M(ball).squeeze(-1), true_landing(ball))
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    return M, last


def concept_of(M):
    @torch.no_grad()
    def f(ball):
        return M(ball).squeeze(-1)
    return f


def scrambled_concept(M):
    @torch.no_grad()
    def f(ball):                       # wrong landing (shuffled across the batch)
        p = M(ball).squeeze(-1)
        return p[torch.randperm(p.shape[0], device=DEVICE)]
    return f


@torch.no_grad()
def catch_rate(ppo, concept, n=256, steps=420, seed=9):
    env = Env(n, concept=concept, seed=seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return env.catch_rate()


def train_arm(concept, iters, eval_every, num_envs, seed):
    seed_all(seed)
    env = Env(num_envs, concept=concept, seed=seed)
    net = PPONet(env.obs_dim, env.action_dim, hidden=128)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
    curve = [(0, round(catch_rate(ppo, concept), 3))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            curve.append((it, round(catch_rate(ppo, concept), 3)))
    return ppo, curve


def iters_to(curve, thr):
    for it, v in curve:
        if v >= thr:
            return it
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--concept-steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.concept_steps, args.eval_every, args.num_envs = 30, 400, 10, 64

    g = torch.Generator(device=DEVICE); g.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    M, cmse = train_concept(args.concept_steps, 256, g)
    cf = concept_of(M)
    print(f"[v36] device={DEVICE} | LEARN GRAVITY -> SOLVE FASTER | concept (landing) "
          f"MSE {cmse:.4f} | with-concept vs scratch, iters->catch>={args.threshold}",
          flush=True)
    t0 = time.perf_counter()

    warm_ppo, warm = train_arm(cf, args.iters, args.eval_every, args.num_envs, args.seed)
    print(f"  with-concept: {warm[-1][1]:.2f} catch @ {warm[-1][0]} it | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    _, scratch = train_arm(None, args.iters, args.eval_every, args.num_envs, args.seed)
    print(f"  scratch:      {scratch[-1][1]:.2f} catch @ {scratch[-1][0]} it | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    # ablation: same trained with-concept agent, but a SCRAMBLED landing prediction
    abl = round(catch_rate(warm_ppo, scrambled_concept(M)), 3)

    wi, si = iters_to(warm, args.threshold), iters_to(scratch, args.threshold)
    speedup = (si / wi) if (wi and si) else None
    print(f"  iters->{args.threshold}: with-concept {wi}, scratch {si} | "
          f"ablation(scrambled) catch {abl:.2f} (vs with-concept {warm[-1][1]:.2f})",
          flush=True)

    uses_it = abl <= warm[-1][1] - 0.30
    faster = wi is not None and (si is None or wi * 1.5 <= si)
    ok = faster and uses_it
    verdict = (
        f"HAVING THE NOTION -> SOLVES FASTER, AND USES IT — the agent that already "
        f"learned gravity reached competence in {wi} iters vs "
        f"{'scratch ' + str(si) if si else 'scratch NEVER in ' + str(args.iters)} "
        f"({'~' + str(round(speedup, 1)) + 'x fewer' if speedup else 'scratch did not reach it'}). "
        f"Ablation: scrambling the learned landing collapses its catch-rate "
        f"{warm[-1][1]:.2f} -> {abl:.2f}, proving it genuinely USES the concept. This is "
        f"the real thesis: learn a basic notion (gravity) once, and a new task that "
        f"uses it is solved with far fewer trials."
        if ok else
        f"PARTIAL — with-concept iters->{args.threshold} {wi} (final {warm[-1][1]:.2f}), "
        f"scratch {si} (final {scratch[-1][1]:.2f}), ablation {abl:.2f}. faster={faster}, "
        f"uses_it={uses_it}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v36_concept_gravity.json"), "w") as f:
        json.dump(dict(concept_mse=cmse, warm_curve=warm, scratch_curve=scratch,
                       iters_warm=wi, iters_scratch=si, speedup=speedup,
                       ablation_catch=abl, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
