"""v15 — the SAME agent learns to WIN any registered game, FROM PIXELS.

Generic trainer: pick a game, drop the validated pixel agent (ConvPPONet +
DiscretePPO) on it, and train it from the rendered image to maximize return
(= win / score). Universal mastery metric: mean episode return over a fixed
eval horizon, vs a random baseline (and a per-game summary). This embodies the
general-game-mastery north star: one agent, many games, from pixels.

Games: pong (P0), breakout (P1). More rungs (snake ...) plug in here.

Usage: python -m scripts.play_game_v15 --game breakout [--iters 500] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake

GAMES = {"pong": DeviceVecPong, "breakout": DeviceVecBreakout, "snake": DeviceVecSnake}


@torch.no_grad()
def evaluate(ppo, game_cls, kw, n, H, random_agent=False):
    env = game_cls(n, **kw)
    ret = torch.zeros(n, device=DEVICE)
    obs = env.state
    for _ in range(H):
        a = (torch.randint(0, env.action_dim, (n,), device=DEVICE)
             if random_agent else ppo.act(obs, deterministic=True))
        obs, r, _, _, _ = env.step(a)
        ret += r
    stats = env.stats() if hasattr(env, "stats") else {}
    return float(ret.mean()), stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=list(GAMES), default="breakout")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.eval_every, args.eval_steps = 10, 64, 5, 300

    kw = dict(img=args.img, max_steps=args.max_steps)
    game_cls = GAMES[args.game]
    os.makedirs(args.out_dir, exist_ok=True)
    env = game_cls(args.num_envs, **kw)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=args.hidden)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=args.entropy, net=net)

    rand_ret, rand_stats = evaluate(ppo, game_cls, kw, args.eval_envs,
                                    args.eval_steps, random_agent=True)
    print(f"[v15] device={DEVICE} | GAME={args.game} | same agent, from pixels "
          f"({3}x{env.img_hw}x{env.img_hw}) | random return {rand_ret:+.2f} "
          f"{rand_stats}", flush=True)

    t0 = time.perf_counter()
    best, curve = rand_ret, []
    for it in range(1, args.iters + 1):
        ppo.train_iter(env, args.rollout)
        if it % args.eval_every == 0:
            ret, stats = evaluate(ppo, game_cls, kw, args.eval_envs, args.eval_steps)
            best = max(best, ret)
            curve.append(dict(it=it, steps=ppo.total_steps, ret=ret, stats=stats))
            print(f"  it {it:>4} | steps {ppo.total_steps:>11,} | return {ret:+8.2f} "
                  f"| best {best:+8.2f} | {stats} | {time.perf_counter()-t0:.0f}s",
                  flush=True)

    ret, stats = evaluate(ppo, game_cls, kw, args.eval_envs, args.eval_steps)
    best = max(best, ret)
    ok = ret > 0 and ret > rand_ret + abs(rand_ret) * 0.5 + 1.0
    verdict = (f"MASTERS {args.game.upper()} FROM PIXELS — the same agent reached "
               f"return {ret:+.2f} (best {best:+.2f}) vs random {rand_ret:+.2f}; "
               f"stats {stats}. Generality: one agent, another game, from pixels."
               if ok else
               f"NOT YET — {args.game}: return {ret:+.2f} (best {best:+.2f}) vs "
               f"random {rand_ret:+.2f}; stats {stats}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"v15_{args.game}.json"), "w") as f:
        json.dump(dict(game=args.game, random_return=rand_ret, final_return=ret,
                       best_return=best, final_stats=stats, curve=curve,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
