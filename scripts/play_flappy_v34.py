"""v34 — validate the new gravity game (DeviceVecFlappy) and learn it from pixels.

Adds a structurally-different game to the substrate (gravity + timing, vs paddle-
ball / grid). First sanity-check the env: a RANDOM agent should die fast (low
score) and a simple HEURISTIC (flap when below the gap centre) should clear many
pipes (so the game is winnable and fair). Then train a CNN-PPO agent from pixels
and show it learns (cum-score rises from ~random toward/over the heuristic).

Usage: python -m scripts.play_flappy_v34 [--iters 250] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.flappy import DeviceVecFlappy


@torch.no_grad()
def eval_policy(act_fn, n=256, steps=599, seed=0):
    env = DeviceVecFlappy(n, max_steps=600, seed=seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(act_fn(env, obs))
    return float(env.cum_score.mean())


def random_act(env, obs):
    return torch.randint(0, 2, (env.num_envs,), device=DEVICE)


def heuristic_act(env, obs):
    # y: 0=top, 1=bottom; flap (go up) when the bird is BELOW the gap centre
    return (env.by > env.gap_y).long()


def ppo_act(ppo):
    return lambda env, obs: ppo.act(obs, deterministic=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.eval_every = 20, 64, 10

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rnd = eval_policy(random_act, seed=1)
    heur = eval_policy(heuristic_act, seed=1)
    print(f"[v34] device={DEVICE} | NEW GAME DeviceVecFlappy (gravity+timing) | "
          f"random {rnd:.2f} pipes vs heuristic {heur:.2f} pipes", flush=True)
    if heur < 3:
        print("  WARNING: heuristic clears <3 pipes — env may be too hard; check tuning",
              flush=True)
    t0 = time.perf_counter()

    env = DeviceVecFlappy(args.num_envs, img=args.img, max_steps=600, seed=args.seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.02, net=net)
    curve = [(0, round(eval_policy(ppo_act(ppo), seed=1), 2))]
    for it in range(1, args.iters + 1):
        ppo.train_iter(env, 32)
        if it % args.eval_every == 0:
            s = eval_policy(ppo_act(ppo), seed=1)
            curve.append((it, round(s, 2)))
            print(f"  iter {it:>3}: cum-score {s:.2f} pipes | {time.perf_counter()-t0:.0f}s",
                  flush=True)
    final = curve[-1][1]

    ok = final >= max(3.0, rnd + 2.0)
    verdict = (
        f"NEW GAME LEARNED FROM PIXELS — DeviceVecFlappy (gravity+timing) is winnable "
        f"(heuristic {heur:.1f} pipes >> random {rnd:.1f}) and the CNN-PPO agent learned "
        f"it from pixels: cum-score {curve[0][1]:.1f} -> {final:.1f} pipes. The substrate "
        f"now has a 5th, structurally-DIFFERENT game (gravity, not paddle-ball/grid) — a "
        f"genuinely dissimilar target for the cross-game accumulation tests the strategy "
        f"review called for."
        if ok else
        f"PARTIAL — Flappy random {rnd:.1f}, heuristic {heur:.1f}, trained {curve[0][1]:.1f}"
        f"->{final:.1f} pipes. Env built + runs; learning/tuning may need another pass.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v34_flappy.json"), "w") as f:
        json.dump(dict(random=rnd, heuristic=heur, curve=curve, final=final,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
