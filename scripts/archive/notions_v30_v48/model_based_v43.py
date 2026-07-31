"""v43 — model-based SAMPLE-EFFICIENCY, FAIR: having a learned dynamics-model means
fewer ENV INTERACTIONS to reach competence than a model-free agent (the owner's
"fewer trials with prior knowledge", done honestly).

Both agents get ONLY the env reward (lines). The ONLY difference is prior knowledge:
- MODEL-BASED: has the dynamics (evaluate_placements = where each placement lands /
  its metrics; v17b showed this is LEARNABLE from the board) and LEARNS a value
  V(metrics)->return from reward via TD; acts greedily. No hand-coded scoring (the
  value is learned) -> addresses the v38 strawman critique.
- MODEL-FREE: PPO from the raw board; must learn dynamics AND value from reward.
Metric: lines cleared vs ENV INTERACTIONS (pieces placed). If model-based reaches
competence in far fewer interactions, prior knowledge (the dynamics) buys
sample-efficiency. >=3 seeds. (Honest: dynamics is analytic here to isolate the
value-learning; the learned-dynamics version is the follow-up.)

Usage: python -m scripts.model_based_v43 [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.tetris import DeviceVecTetris

SCALE = torch.tensor([0.25, 1 / 30.0, 1 / 60.0, 1.0], device=DEVICE)


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class MB:
    """Learns V(placement-metrics) -> return by TD; acts greedily over placements."""
    def __init__(self, action_dim):
        self.V = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEVICE)
        self.opt = torch.optim.Adam(self.V.parameters(), 1e-3)
        self.A = action_dim

    @torch.no_grad()
    def act(self, env, eps):
        s = self.V(env.evaluate_placements() * SCALE).squeeze(-1)      # (N,A)
        g = s.argmax(-1)
        r = torch.randint(0, self.A, g.shape, device=DEVICE)
        return torch.where(torch.rand(g.shape, device=DEVICE) < eps, r, g)


@torch.no_grad()
def eval_lines(act_fn, make_env, n=128, steps=200, seed=9):
    env = make_env(n, seed)
    for _ in range(steps):
        env.step(act_fn(env))
    return float(env.cum_lines.mean())


def train_mb(iters, eval_every, num_envs, seed, gamma=0.9):
    seed_all(seed + 1)
    env = DeviceVecTetris(num_envs, seed=seed)
    mb = MB(env.action_dim)
    n, ar = num_envs, torch.arange(num_envs, device=DEVICE)
    make = lambda k, s: DeviceVecTetris(k, seed=s)
    inter = 0
    curve = [(0, round(eval_lines(lambda e: mb.act(e, 0.0), make), 2))]
    for it in range(1, iters + 1):
        for _ in range(16):
            m = env.evaluate_placements() * SCALE                     # (N,A,4)
            a = mb.act(env, 0.1)
            m_ch = m[ar, a]                                            # (N,4)
            _, rew, term, trunc, done = env.step(a)
            inter += n
            with torch.no_grad():
                m2 = env.evaluate_placements() * SCALE
                tgt = rew + gamma * (~done).float() * mb.V(m2).squeeze(-1).max(-1).values
            loss = F.mse_loss(mb.V(m_ch).squeeze(-1), tgt)
            mb.opt.zero_grad(); loss.backward(); mb.opt.step()
        if it % eval_every == 0:
            curve.append((inter, round(eval_lines(lambda e: mb.act(e, 0.0), make), 2)))
    return curve


def train_mf(iters, eval_every, num_envs, seed):
    seed_all(seed + 2)
    make = lambda k, s: DeviceVecTetris(k, seed=s)
    env = make(num_envs, seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
    curve = [(0, round(eval_lines(lambda e: ppo.act(e.state, deterministic=True), make), 2))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 16)
        if it % eval_every == 0:
            curve.append((ppo.total_steps,
                          round(eval_lines(lambda e: ppo.act(e.state, deterministic=True), make), 2)))
    return curve


def inter_to(curve, thr):
    for x, v in curve:
        if v >= thr:
            return x
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--mb-iters", type=int, default=80)
    p.add_argument("--mf-iters", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--num-envs", type=int, default=160)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.mb_iters, args.mf_iters, args.num_envs = [0], 15, 20, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v43] device={DEVICE} | model-based SAMPLE-EFFICIENCY (Tetris) | both use env "
          f"reward; MB has dynamics+learns value, MF=PPO from board | lines vs ENV "
          f"INTERACTIONS, >={args.threshold}, seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        mb = train_mb(args.mb_iters, args.eval_every, args.num_envs, s)
        mf = train_mf(args.mf_iters, args.eval_every, args.num_envs, s)
        mbi, mfi = inter_to(mb, args.threshold), inter_to(mf, args.threshold)
        rows.append(dict(seed=s, mb_interactions=mbi, mb_final=mb[-1][1],
                         mf_interactions=mfi, mf_final=mf[-1][1]))
        print(f"  seed {s}: MB ->{args.threshold} lines @ {mbi} interactions (final {mb[-1][1]}) "
              f"| MF @ {mfi} (final {mf[-1][1]}) | {time.perf_counter()-t0:.0f}s", flush=True)

    mb_ok = all(r["mb_interactions"] is not None for r in rows)
    fewer = all(r["mb_interactions"] is not None and
                (r["mf_interactions"] is None or r["mb_interactions"] * 2 <= r["mf_interactions"])
                for r in rows)
    ok = mb_ok and fewer
    verdict = (
        f"MODEL-BASED SAMPLE-EFFICIENCY (fair) — having the dynamics, the model-based "
        f"agent reaches the line-threshold in <= half the ENV INTERACTIONS of model-free "
        f"every seed (mb,mf interactions {[(r['mb_interactions'], r['mf_interactions']) for r in rows]}). "
        f"Prior knowledge (the dynamics) -> fewer trials, both on env reward, learned value "
        f"(no hand-coded scoring). REVIEW before reporting; then learned-dynamics follow-up."
        if ok else
        f"PARTIAL/NEG — mb_ok={mb_ok}, fewer={fewer}; rows={rows}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v43_model_based.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, rows=rows, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
