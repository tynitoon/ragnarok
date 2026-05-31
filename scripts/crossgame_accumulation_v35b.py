"""v35b — THE NORTH QUESTION: does a library of games make a NEW, dissimilar game
cheaper to learn? (v31 done right: more + more-diverse games, multi-source, seeded.)

Leave-one-out over a diverse 48x48 suite {pong, breakout (paddle-ball), snake
(grid), catcher (falling-object intercept)}. For each HELD-OUT game H: pretrain a
SHARED conv encoder on the OTHER three games (a 3-game 'library'), then LEARN H
two ways at equal budget — WARM (reuse the library encoder + a fresh H head) vs
SCRATCH (fresh encoder + head) — and compare learning curves. early-advantage =
mean(warm - scratch) over the first half of checkpoints (relative to scratch's own
range, so games of different score-scales are comparable). Averaged over seeds.

Decisive: if warm beats scratch on the held-out game (positive early-advantage)
across games/seeds, a multi-game library DOES make a new dissimilar game cheaper —
the north-star claim. If not, cross-game representation transfer does not help even
with a diverse library, and we say so (and the developmental value lies in
recognise-and-reuse of KNOWN games, not blind cross-game transfer).

Usage: python -m scripts.crossgame_accumulation_v35b [--seeds 0 1] [--smoke]
"""

import argparse
import copy
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake
from ragnarok.environments.catcher import DeviceVecCatcher
from scripts.crossgame_probe_v31 import SharedConvEncoder, GameNet, make_ppo

GAMES = {
    "pong": (lambda n, s: DeviceVecPong(n, img=48, seed=s), 3),
    "breakout": (lambda n, s: DeviceVecBreakout(n, img=48, seed=s), 3),
    "snake": (lambda n, s: DeviceVecSnake(n, seed=s), 4),
    "catcher": (lambda n, s: DeviceVecCatcher(n, seed=s), 3),
}


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def mean_return(ppo, env_fn, n, steps=399, seed=7):
    env = env_fn(n, seed)
    obs = env.state
    tot = torch.zeros(n, device=DEVICE)
    for _ in range(steps):
        obs, r, _, _, _ = env.step(ppo.act(obs, deterministic=True))
        tot += r
    return float(tot.mean())


def pretrain_library(train_games, iters, num_envs, img, seed):
    seed_all(seed)
    enc = SharedConvEncoder(img)
    envs = {g: GAMES[g][0](num_envs, seed) for g in train_games}
    ppos = {g: make_ppo(enc, GAMES[g][1], img) for g in train_games}   # share enc
    for i in range(iters):
        g = train_games[i % len(train_games)]
        ppos[g].train_iter(envs[g], 32)
    return enc


def learn_game(encoder, game, iters, eval_every, num_envs, img, seed):
    seed_all(seed + 100)
    env_fn, adim = GAMES[game]
    enc = copy.deepcopy(encoder) if encoder is not None else SharedConvEncoder(img)
    ppo = make_ppo(enc, adim, img)
    env = env_fn(num_envs, seed)
    curve = [round(mean_return(ppo, env_fn, num_envs), 3)]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            curve.append(round(mean_return(ppo, env_fn, num_envs), 3))
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--pre-iters", type=int, default=120)
    p.add_argument("--learn-iters", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=224)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.pre_iters, args.learn_iters = [0], 12, 16
        args.eval_every, args.num_envs = 8, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v35b] device={DEVICE} | CROSS-GAME ACCUMULATION (leave-one-out) | "
          f"library = the other 3 of {list(GAMES)} | warm vs scratch | seeds {args.seeds}",
          flush=True)
    t0 = time.perf_counter()

    per_game = {}
    for held in GAMES:
        train_games = [g for g in GAMES if g != held]
        advs = []
        for s in args.seeds:
            enc = pretrain_library(train_games, args.pre_iters, args.num_envs, args.img, s)
            warm = learn_game(enc, held, args.learn_iters, args.eval_every,
                              args.num_envs, args.img, s)
            scratch = learn_game(None, held, args.learn_iters, args.eval_every,
                                 args.num_envs, args.img, s)
            k = max(1, len(warm) // 2)
            rng = max(1e-6, max(scratch) - min(scratch))      # scratch's own score range
            adv = sum((warm[i] - scratch[i]) for i in range(1, k + 1)) / k / rng
            advs.append(adv)
            print(f"  held={held:9s} seed={s}: warm_final {warm[-1]:.2f} vs scratch_final "
                  f"{scratch[-1]:.2f} | norm early-adv {adv:+.2f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
        per_game[held] = round(sum(advs) / len(advs), 3)

    helped = [g for g, a in per_game.items() if a > 0.10]
    mean_adv = round(sum(per_game.values()) / len(per_game), 3)
    ok = len(helped) >= 3 and mean_adv > 0.10
    verdict = (
        f"LIBRARY HELPS A NEW GAME — a 3-game shared-encoder library gives a positive "
        f"normalised early-learning advantage on the held-out game in {len(helped)}/4 "
        f"cases (per-game {per_game}, mean {mean_adv:+.2f}). Accumulated multi-game "
        f"representation makes a NEW, dissimilar game cheaper to learn — the north-star "
        f"claim, now on a diverse seeded suite."
        if ok else
        f"MIXED/NEGATIVE — library early-advantage per held-out game {per_game} "
        f"(mean {mean_adv:+.2f}); helped in {len(helped)}/4. Cross-game representation "
        f"transfer is {'weak' if mean_adv > 0 else 'absent'} even with a diverse "
        f"library; the developmental value is in recognise-and-reuse of KNOWN games, "
        f"not blind cross-game transfer. Reported honestly.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v35b_crossgame_accum.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, per_game=per_game, mean_advantage=mean_adv,
                       helped=helped, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
