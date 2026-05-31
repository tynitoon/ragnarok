"""v27 — SCALE via BROAD VARIETY: does training on many game-variants give a
skill that generalises to UNSEEN variants? (The v19 recipe, in games.)

Single-source cross-game transfer failed (P3/v16) — memorisation. v19 showed the
fix: BROAD VARIETY forces the net to learn the RULE, which then generalises.
Here we apply it in the game domain: Pong is parameterised (ball speed, paddle
size, opponent speed, spin), so we make a FAMILY of variants. Train ONE agent on
MANY TRAIN variants, then test it ZERO-SHOT on HELD-OUT variants. Compare to a
SINGLE-variant agent (trained on one fixed Pong).

Decisive: the variety-trained agent plays unseen variants ~ as well as trained
ones (it learned a general Pong skill), and beats the brittle single-variant
agent on the unseen ones -> broad variety -> a reusable skill that transfers to
new instances for free.

Usage: python -m scripts.variety_efficiency_v27 [--iters 220] [--smoke]
"""

import argparse
import json
import os
import random
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.pong import DeviceVecPong


def gen_variants(n, rng):
    return [dict(ball_speed=rng.uniform(0.022, 0.038),
                 paddle_half=rng.uniform(0.08, 0.16),
                 opp_speed=rng.uniform(0.014, 0.024),
                 spin=rng.uniform(0.3, 0.7)) for _ in range(n)]


@torch.no_grad()
def winrate(ppo, variant, n=256, steps=799, img=48):
    env = DeviceVecPong(n, img=img, max_steps=800, **variant)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return float(((env.score_a - env.score_o) > 0).float().mean())


def new_ppo(img=48):
    env = DeviceVecPong(2, img=img)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    return DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=220)
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.n_train, args.num_envs = 12, 6, 64

    rng = random.Random(args.seed)
    train_v = gen_variants(args.n_train, rng)
    test_v = gen_variants(args.n_test, rng)               # HELD-OUT variants
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v27] device={DEVICE} | BROAD VARIETY in games | train on "
          f"{args.n_train} Pong variants -> ZERO-SHOT on {args.n_test} UNSEEN "
          f"variants (vs a single-variant agent)", flush=True)
    t0 = time.perf_counter()

    # variety agent: one net, a random TRAIN variant each iteration
    var_envs = [DeviceVecPong(args.num_envs, img=args.img, max_steps=800, **v)
                for v in train_v]
    variety = new_ppo(args.img)
    for it in range(args.iters):
        variety.train_iter(var_envs[rng.randrange(args.n_train)], 32)
    # single-variant agent: same budget, one fixed variant
    single_env = DeviceVecPong(args.num_envs, img=args.img, max_steps=800, **train_v[0])
    single = new_ppo(args.img)
    for it in range(args.iters):
        single.train_iter(single_env, 32)
    print(f"  trained variety + single-variant agents | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    var_on_train = sum(winrate(variety, v) for v in train_v) / len(train_v)
    var_on_test = sum(winrate(variety, v) for v in test_v) / len(test_v)
    single_on_test = sum(winrate(single, v) for v in test_v) / len(test_v)
    print(f"  variety: train-variants win {var_on_train:.2f} | UNSEEN {var_on_test:.2f}"
          f" | single-variant on UNSEEN {single_on_test:.2f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    ok = var_on_test >= 0.7 and var_on_test >= single_on_test + 0.1
    verdict = (f"BROAD VARIETY -> GENERAL SKILL — the variety-trained agent wins "
               f"{var_on_test:.0%} on UNSEEN Pong variants (~ {var_on_train:.0%} on "
               f"trained ones), beating the single-variant agent's {single_on_test:.0%}"
               f" on the same unseen variants. Training across variety yields a "
               f"reusable skill that transfers to new instances for free — the "
               f"v19 recipe, confirmed in games (where single-source transfer failed)."
               if ok else
               f"PARTIAL — variety unseen {var_on_test:.2f} vs train {var_on_train:.2f}"
               f" vs single {single_on_test:.2f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v27_variety.json"), "w") as f:
        json.dump(dict(n_train=args.n_train, n_test=args.n_test,
                       variety_on_train=var_on_train, variety_on_unseen=var_on_test,
                       single_on_unseen=single_on_test, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
