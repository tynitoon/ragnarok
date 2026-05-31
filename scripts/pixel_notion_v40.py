"""v40 — THE DECISIVE PIXEL TEST (frozen design; see preregistration.md).

Does a self-supervised, FACTORED notion learned FROM PIXELS make a NEW pixel
control task reach competence in fewer iterations than from scratch?

- NOTION: a CNN that predicts the ball's LANDING y from a PIXEL frame (ball dot +
  velocity-cue + catcher bar), trained self-supervised (no task reward) on varied
  launches. label = the env's analytic landing.
- WARM arm: PPO on obs=[catcher_y, notion(pixels)-landing] (reuses the notion).
- SCRATCH arm: PPO on the raw pixels (must learn perception+parabola+control).
Metric: iters to catch-rate>=0.7, WARM vs SCRATCH, >=3 seeds.

Usage: python -m scripts.pixel_notion_v40 [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet, PPONet
from ragnarok.environments.projectile import DeviceVecProjectileCatch as Env

IMG = 24


class LandingCNN(nn.Module):
    def __init__(self, img=IMG, hidden=64):
        super().__init__()
        self.img = img
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, 2), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, 2), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img, img)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pix_flat):
        n = pix_flat.shape[0]
        x = pix_flat.view(n, 3, self.img, self.img)
        return self.fc(self.conv(x).reshape(n, -1))


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def collect(n, steps, seed):
    env = Env(n, img=IMG, max_steps=70, seed=seed)
    X, Y = [], []
    for _ in range(steps):
        X.append(env._pixels()); Y.append(env._landing())
        env.step(torch.randint(0, 3, (n,), device=DEVICE))
    return torch.cat(X), torch.cat(Y)


def train_notion(steps, seed):
    seed_all(seed)
    M = LandingCNN().to(DEVICE)
    opt = torch.optim.Adam(M.parameters(), 1e-3)
    Xtr, Ytr = collect(256, steps, seed)
    Xte, Yte = collect(256, 40, seed + 999)
    idx = torch.arange(Xtr.shape[0], device=DEVICE)
    for ep in range(8):
        perm = idx[torch.randperm(idx.shape[0], device=DEVICE)]
        for i in range(0, perm.shape[0], 512):
            j = perm[i:i + 512]
            loss = F.mse_loss(M(Xtr[j]).squeeze(-1), Ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val = float(F.mse_loss(M(Xte).squeeze(-1), Yte))
    return M, val


@torch.no_grad()
def catch_rate(ppo, make_env, n=256, steps=350, seed=7):
    env = make_env(n, seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return float(env.cum_catch.sum() / env.cum_ep.sum().clamp(min=1))


def run_arm(mode, notion, iters, eval_every, num_envs, seed):
    seed_all(seed + (1 if mode == "warm" else 2))
    if mode == "warm":
        make_env = lambda n, s: Env(n, img=IMG, concept=notion, max_steps=70, seed=s)
        net = PPONet(2, 3, hidden=128)
    else:
        make_env = lambda n, s: Env(n, img=IMG, max_steps=70, seed=s)
        net = ConvPPONet(IMG, 3, hidden=128)
    env = make_env(num_envs, seed)
    ppo = DiscretePPO(env.obs_dim, 3, entropy=0.01, net=net)
    curve = [(0, round(catch_rate(ppo, make_env), 3))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            curve.append((it, round(catch_rate(ppo, make_env), 3)))
    return curve


def iters_to(curve, thr):
    for it, v in curve:
        if v >= thr:
            return it
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--notion-steps", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.iters, args.notion_steps, args.num_envs = [0], 30, 30, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v40] device={DEVICE} | PIXEL notion->faster | img={IMG} | WARM(notion) vs "
          f"SCRATCH(pixels), iters->catch>={args.threshold}, seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        M, val = train_notion(args.notion_steps, s)
        warm = run_arm("warm", M, args.iters, args.eval_every, args.num_envs, s)
        scratch = run_arm("scratch", None, args.iters, args.eval_every, args.num_envs, s)
        wi, si = iters_to(warm, args.threshold), iters_to(scratch, args.threshold)
        rows.append(dict(seed=s, notion_val_mse=round(val, 4),
                         warm_iters=wi, warm_final=warm[-1][1],
                         scratch_iters=si, scratch_final=scratch[-1][1]))
        print(f"  seed {s}: notion MSE {val:.3f} | WARM ->0.7 in {wi} (final {warm[-1][1]}) | "
              f"SCRATCH ->0.7 in {si} (final {scratch[-1][1]}) | {time.perf_counter()-t0:.0f}s",
              flush=True)

    warm_ok = all(r["warm_iters"] is not None for r in rows)
    faster = all(r["warm_iters"] is not None and
                 (r["scratch_iters"] is None or r["warm_iters"] * 2 <= r["scratch_iters"])
                 for r in rows)
    ok = warm_ok and faster
    verdict = (
        f"PIXEL NOTION -> FASTER (north-star positive) — across {len(args.seeds)} seeds, the "
        f"agent reusing a self-supervised landing-notion learned FROM PIXELS reaches "
        f"competence every seed, in <= half the iters of (or where) a from-pixels SCRATCH "
        f"agent (per-seed warm vs scratch: "
        f"{[(r['warm_iters'], r['scratch_iters']) for r in rows]}). A factored notion learned "
        f"from pixels makes a NEW pixel task measurably cheaper — reliable reuse in the hard "
        f"regime the v36 principle predicted."
        if ok else
        f"PARTIAL/NEG — warm_ok={warm_ok}, faster={faster}; rows={rows}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v40_pixel_notion.json"), "w") as f:
        json.dump(dict(img=IMG, seeds=args.seeds, rows=rows, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
