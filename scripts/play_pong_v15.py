"""v15 M1 — learn to WIN Pong FROM PIXELS.

Drop the validated pixel-RL agent (ConvPPONet + DiscretePPO) onto DeviceVecPong
and train it, from the rendered image alone, to BEAT the scripted opponent.
Decisive: it goes from losing every game (random: win-rate ~0, margin ~ -11)
to winning (win-rate >= 0.8, positive margin) — i.e. it masters the game.

First proof of the general-game-mastery north star: same agent, a game, from
pixels, learns to win.

Usage: python -m scripts.play_pong_v15 [--iters 400] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.pong import DeviceVecPong


@torch.no_grad()
def evaluate(ppo, cfg, n=512, random_agent=False):
    """Play one episode (max_steps) deterministically; return (mean margin,
    win-rate, points scored, points conceded)."""
    env = DeviceVecPong(n, img=cfg["img"], max_steps=cfg["max_steps"],
                        opp_speed=cfg["opp_speed"])
    obs = env.state
    # stop one step BEFORE max_steps so scores aren't reset by truncation
    for _ in range(cfg["max_steps"] - 1):
        if random_agent:
            a = torch.randint(0, env.action_dim, (n,), device=DEVICE)
        else:
            a = ppo.act(obs, deterministic=True)
        obs, _, _, _, _ = env.step(a)
    margin = env.score_a - env.score_o
    return (float(margin.mean()), float((margin > 0).float().mean()),
            float(env.score_a.mean()), float(env.score_o.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--opp-speed", type=float, default=0.020)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.eval_every = 10, 64, 5

    cfg = {k: getattr(args, k) for k in ("img", "max_steps", "opp_speed")}
    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecPong(args.num_envs, img=args.img, max_steps=args.max_steps,
                        opp_speed=args.opp_speed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=args.hidden)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=args.entropy, net=net)

    rand_m, rand_w, _, _ = evaluate(ppo, cfg, random_agent=True)
    print(f"[v15-M1] device={DEVICE} | learning to WIN Pong FROM PIXELS "
          f"({3}x{env.img_hw}x{env.img_hw}) | random baseline: margin "
          f"{rand_m:+.2f}, win-rate {rand_w:.2f}", flush=True)

    t0 = time.perf_counter()
    best_w, curve = 0.0, []
    for it in range(1, args.iters + 1):
        ppo.train_iter(env, args.rollout)
        if it % args.eval_every == 0:
            m, w, sa, so = evaluate(ppo, cfg)
            best_w = max(best_w, w)
            curve.append(dict(it=it, steps=ppo.total_steps, margin=m, winrate=w))
            print(f"  it {it:>4} | steps {ppo.total_steps:>11,} | margin {m:+6.2f} "
                  f"| win-rate {w:.2f} | ({sa:.1f}-{so:.1f}) | best {best_w:.2f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    m, w, sa, so = evaluate(ppo, cfg)
    best_w = max(best_w, w)
    ok = w >= 0.8 and m > 0
    verdict = (f"WINS FROM PIXELS — the agent learned to BEAT the opponent at "
               f"Pong from the raw image (final win-rate {w:.2f}, margin {m:+.2f}; "
               f"random was win-rate {rand_w:.2f}, margin {rand_m:+.2f}). First "
               f"proof of game-mastery-to-win from pixels."
               if ok else
               f"NOT YET — final win-rate {w:.2f}, margin {m:+.2f} (need >=0.80, "
               f">0). best win-rate {best_w:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v15_pong.json"), "w") as f:
        json.dump(dict(random=dict(margin=rand_m, winrate=rand_w),
                       final=dict(margin=m, winrate=w, scored=sa, conceded=so),
                       best_winrate=best_w, curve=curve, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
