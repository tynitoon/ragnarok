"""v35a — validate DeviceVecCatcher and learn it from pixels.

Dense-reward catching game (no exploration trap). Sanity: random catches few,
a move-toward-fruit heuristic catches many. Then a CNN-PPO agent should learn it
from pixels (cum-catch rises well above random).

Usage: python -m scripts.play_catcher_v35 [--iters 150] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.catcher import DeviceVecCatcher


@torch.no_grad()
def eval_catch(act_fn, n=256, steps=399, seed=1):
    env = DeviceVecCatcher(n, max_steps=400, seed=seed)
    obs = env.state
    for _ in range(steps):
        obs, _, _, _, _ = env.step(act_fn(env, obs))
    return float(env.cum_catch.mean())


def random_act(env, obs):
    return torch.randint(0, 3, (env.num_envs,), device=DEVICE)


def heuristic_act(env, obs):
    # move toward the fruit x: right if paddle left of fruit, else left
    return torch.where(env.px < env.fx - 0.01, torch.full_like(env.px, 2),
                       torch.where(env.px > env.fx + 0.01, torch.ones_like(env.px),
                                   torch.zeros_like(env.px))).long()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--shaping", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.num_envs, args.eval_every = 20, 64, 10

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rnd = eval_catch(random_act)
    heur = eval_catch(heuristic_act)
    print(f"[v35a] device={DEVICE} | NEW GAME DeviceVecCatcher | random {rnd:.1f} "
          f"vs heuristic {heur:.1f} catches", flush=True)
    t0 = time.perf_counter()

    env = DeviceVecCatcher(args.num_envs, max_steps=400, shaping=args.shaping, seed=args.seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
    curve = [(0, round(eval_catch(lambda e, o: ppo.act(o, deterministic=True)), 1))]
    for it in range(1, args.iters + 1):
        ppo.train_iter(env, 32)
        if it % args.eval_every == 0:
            s = eval_catch(lambda e, o: ppo.act(o, deterministic=True))
            curve.append((it, round(s, 1)))
            print(f"  iter {it:>3}: cum-catch {s:.1f} | {time.perf_counter()-t0:.0f}s",
                  flush=True)
    final = curve[-1][1]
    ok = final >= max(8.0, rnd + 5.0)
    verdict = (f"CATCHER LEARNED FROM PIXELS — random {rnd:.1f}, heuristic {heur:.1f}, "
               f"trained {curve[0][1]:.1f} -> {final:.1f} catches. A reliably-learnable, "
               f"structurally-different game is now in the library."
               if ok else
               f"PARTIAL — random {rnd:.1f}, heuristic {heur:.1f}, trained "
               f"{curve[0][1]:.1f}->{final:.1f}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v35a_catcher.json"), "w") as f:
        json.dump(dict(random=rnd, heuristic=heur, curve=curve, final=final,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
