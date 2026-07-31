"""v41 — the decisive pixel test, in the regime where a notion CAN win (frozen).

v40 was null: the catch task was learnable-enough from pixels (scratch ~40 iters)
AND the notion was inaccurate. v41 fixes BOTH per the sharpened requirement:
- HARDER task: SPARSE reward (+1 catch / -1 miss, no distance shaping) + TIGHT
  tolerance (0.05) -> scratch must precisely predict the (bouncing) landing from
  pixels, which is slow/hard.
- ACCURATE notion: img=32, a bigger CNN, more self-supervised data/epochs -> low
  landing-prediction error, so WARM gets a clean target.
WARM (obs=[catcher_y, notion(pixels)-landing]) vs SCRATCH (raw pixels). Metric:
iters to catch-rate>=0.5, >=3 seeds. If WARM>>SCRATCH reliably -> the positive.

Usage: python -m scripts.pixel_notion_v41 [--seeds 0 1 2] [--smoke]
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

IMG, TOL = 32, 0.05


class LandingCNN(nn.Module):
    def __init__(self, img=IMG, hidden=128):
        super().__init__()
        self.img = img
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img, img)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pix):
        n = pix.shape[0]
        return self.fc(self.conv(pix.view(n, 3, self.img, self.img)).reshape(n, -1))


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def concept_of(M):
    @torch.no_grad()
    def f(pix):
        return M(pix)
    return f


def make_env(n, seed, concept=None):
    return Env(n, img=IMG, concept=concept, tol=TOL, sparse=True, max_steps=70, seed=seed)


@torch.no_grad()
def collect(n, steps, seed):
    env = make_env(n, seed)
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
    for ep in range(18):
        perm = idx[torch.randperm(idx.shape[0], device=DEVICE)]
        for i in range(0, perm.shape[0], 512):
            j = perm[i:i + 512]
            loss = F.mse_loss(M(Xtr[j]).squeeze(-1), Ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return M, float(F.mse_loss(M(Xte).squeeze(-1), Yte))


@torch.no_grad()
def catch_rate(ppo, concept, n=256, steps=350, seed=7):
    env = make_env(n, seed, concept=concept)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return float(env.cum_catch.sum() / env.cum_ep.sum().clamp(min=1))


def run_arm(mode, notion, iters, eval_every, num_envs, seed):
    seed_all(seed + (1 if mode == "warm" else 2))
    cf = concept_of(notion) if mode == "warm" else None
    net = PPONet(2, 3, hidden=128) if mode == "warm" else ConvPPONet(IMG, 3, hidden=128)
    env = make_env(num_envs, seed, concept=cf)
    ppo = DiscretePPO(env.obs_dim, 3, entropy=0.01, net=net)
    curve = [(0, round(catch_rate(ppo, cf), 3))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            curve.append((it, round(catch_rate(ppo, cf), 3)))
    return curve


def iters_to(curve, thr):
    for it, v in curve:
        if v >= thr:
            return it
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iters", type=int, default=240)
    p.add_argument("--notion-steps", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.iters, args.notion_steps, args.num_envs = [0], 30, 60, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v41] device={DEVICE} | PIXEL notion->faster (HARD: sparse+tol{TOL}, img{IMG}, "
          f"accurate notion) | WARM vs SCRATCH, iters->catch>={args.threshold}, "
          f"seeds {args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        M, val = train_notion(args.notion_steps, s)
        warm = run_arm("warm", M, args.iters, args.eval_every, args.num_envs, s)
        scratch = run_arm("scratch", None, args.iters, args.eval_every, args.num_envs, s)
        wi, si = iters_to(warm, args.threshold), iters_to(scratch, args.threshold)
        rows.append(dict(seed=s, notion_mse=round(val, 4), warm_iters=wi,
                         warm_final=warm[-1][1], scratch_iters=si, scratch_final=scratch[-1][1]))
        print(f"  seed {s}: notion MSE {val:.4f} | WARM ->{args.threshold} in {wi} "
              f"(final {warm[-1][1]}) | SCRATCH in {si} (final {scratch[-1][1]}) | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    warm_ok = all(r["warm_iters"] is not None for r in rows)
    faster = all(r["warm_iters"] is not None and
                 (r["scratch_iters"] is None or r["warm_iters"] * 2 <= r["scratch_iters"])
                 for r in rows)
    ok = warm_ok and faster
    verdict = (
        f"POSITIVE — a factored notion learned FROM PIXELS makes a HARD new pixel task "
        f"reliably cheaper: every seed WARM reaches competence in <= half the iters of "
        f"(or where) SCRATCH fails (warm,scratch per seed: "
        f"{[(r['warm_iters'], r['scratch_iters']) for r in rows]}; notion MSE "
        f"{[r['notion_mse'] for r in rows]}). Reliable reuse in the hard regime — the "
        f"north-star claim, from pixels. REVIEW before reporting."
        if ok else
        f"PARTIAL/NEG — warm_ok={warm_ok}, faster={faster}; rows={rows}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v41_pixel_notion_hard.json"), "w") as f:
        json.dump(dict(img=IMG, tol=TOL, seeds=args.seeds, rows=rows, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
