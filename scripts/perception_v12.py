"""v12 Phase A — PERCEPTION: learn a skill from PIXELS.

The agent sees only a small RGB image (egocentric view, each cell-type a
colour, upscaled to tiles) — it is NOT given cell-type ids. A CNN encoder
learns features; PPO learns the collect_wood skill from the image. Decisive:
it reaches the symbolic-MLP skill's success (~0.96, from v6/M2) -> the agent
learned to SEE.

Usage: python -m scripts.perception_v12 [--iters 300] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.craft_world import DeviceVecCraftWorld, A_WOOD


@torch.no_grad()
def _success(ppo, cfg, n=256):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=A_WOOD,
                              pixel=True, tile=cfg["tile"])
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.eval_every = 8, 64, 4

    cfg = {k: getattr(args, k) for k in
           ("grid", "view", "tile", "max_steps", "rollout", "hidden", "entropy")}
    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecCraftWorld(args.num_envs, grid=args.grid, view=args.view,
                              max_steps=args.max_steps, goal=A_WOOD,
                              pixel=True, tile=args.tile)
    print(f"[v12-A] device={DEVICE} | PIXEL obs {3}x{env.img_hw}x{env.img_hw} "
          f"(dim {env.obs_dim}) | learning collect_wood FROM PIXELS", flush=True)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=args.hidden)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=args.entropy, net=net)

    t0 = time.perf_counter()
    best = 0.0
    curve = []
    for it in range(1, args.iters + 1):
        ppo.train_iter(env, args.rollout)
        if it % args.eval_every == 0:
            s = _success(ppo, cfg)
            best = max(best, s)
            curve.append([it, ppo.total_steps, s])
            print(f"  it {it:>4} | steps {ppo.total_steps:>10,} | "
                  f"collect_wood-from-pixels {s:.2f} | best {best:.2f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    final = _success(ppo, cfg)
    ok = max(best, final) >= 0.8
    verdict = ("PERCEPTION WORKS — the agent learned collect_wood FROM PIXELS "
               "(no cell-types given), matching the symbolic skill. It learned "
               "to SEE." if ok else
               f"CHECK — pixel skill reached {max(best, final):.2f} (<0.8); "
               "needs more iters / encoder tuning.")
    print(f"\n  -> {verdict}\n  final {final:.2f} | best {best:.2f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v12a.json"), "w") as f:
        json.dump(dict(curve=curve, best=best, final=final, verdict=verdict,
                       img=f"3x{env.img_hw}x{env.img_hw}"), f, indent=2)


if __name__ == "__main__":
    main()
