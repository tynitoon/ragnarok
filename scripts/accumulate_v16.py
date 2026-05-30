"""v16 — KNOWLEDGE ACCUMULATION across games: does knowing several games make
a NEW game faster to learn?

The owner's core vision, made measurable: "the more it knows, the more it can
solve." P3 showed NAIVE single-source encoder reuse HURTS (over-specialises).
The principled fix: train a SHARED encoder on SEVERAL games at once (general
game-perception via MultiGameConvNet), then learn a HELD-OUT game two ways —
  ACCUMULATED: reuse the multi-game encoder, fresh head,
  SCRATCH:     fresh net —
and compare speed. If ACCUMULATED is faster, then accumulated game-knowledge
bootstraps a new game (compounding across tasks). Honest: if it does not beat
scratch (and beats single-source P3), encoder-sharing is not enough and the
next step is an explicit skill-library + recognition (the craft-world v4/v14
mechanism).

Usage: python -m scripts.accumulate_v16 [--pretrain pong snake] [--target breakout] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.learning.multigame_net import MultiGameConvNet
from scripts.play_game_v15 import GAMES, evaluate
from scripts.crossgame_v15p3 import train      # target-arm trainer (reuses encoder)


def pretrain_multigame(games, iters, cfg):
    """Train one shared encoder on several games at once (rotating per iter)."""
    kw = dict(img=cfg["img"], max_steps=cfg["max_steps"])
    envs = {g: GAMES[g](cfg["num_envs"], **kw) for g in games}
    adims = {g: envs[g].action_dim for g in games}
    img_hw = next(iter(envs.values())).img_hw
    net = MultiGameConvNet(img_hw, adims, hidden=cfg["hidden"])
    obs_dim = 3 * img_hw * img_hw
    ppo = DiscretePPO(obs_dim, max(adims.values()), entropy=cfg["entropy"], net=net)
    t0 = time.perf_counter()
    for it in range(1, iters + 1):
        for g in games:
            net.set_game(g)
            ppo.train_iter(envs[g], cfg["rollout"])
        if it % 25 == 0:
            rs = {}
            for g in games:
                net.set_game(g)
                rs[g], _ = evaluate(ppo, GAMES[g], kw, cfg["eval_envs"], cfg["eval_steps"])
            print(f"  [pretrain] it {it:>3} | " +
                  " ".join(f"{g} {rs[g]:+.1f}" for g in games) +
                  f" | {time.perf_counter()-t0:.0f}s", flush=True)
    return net.encoder_state()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain", nargs="+", default=["pong", "snake"])
    p.add_argument("--target", default="breakout")
    p.add_argument("--pretrain-iters", type=int, default=200)
    p.add_argument("--target-iters", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--eval-steps", type=int, default=800)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--target-return", type=float, default=0.0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.pretrain_iters, args.target_iters, args.num_envs = 6, 10, 64
        args.eval_every, args.eval_steps = 5, 300

    cfg = {k: getattr(args, k) for k in
           ("img", "max_steps", "num_envs", "rollout", "hidden", "entropy",
            "eval_steps", "eval_envs")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v16] device={DEVICE} | ACCUMULATE {args.pretrain} -> learn "
          f"{args.target} | does knowing more games help?", flush=True)
    t0 = time.perf_counter()

    enc = pretrain_multigame(args.pretrain, args.pretrain_iters, cfg)
    print(f"  pretrained shared encoder on {args.pretrain} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    acc = train(args.target, args.target_iters, cfg, args.eval_every,
                init_state=enc, freeze=False, target_return=args.target_return)
    print(f"  {args.target} ACCUMULATED: auc {acc['auc']:+.2f} final "
          f"{acc['final']:+.2f} steps {acc['steps_to_pos']} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    sc = train(args.target, args.target_iters, cfg, args.eval_every,
               init_state=None, target_return=args.target_return)
    print(f"  {args.target} SCRATCH    : auc {sc['auc']:+.2f} final "
          f"{sc['final']:+.2f} steps {sc['steps_to_pos']} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    auc_gain = acc["auc"] - sc["auc"]
    speedup = (sc["steps_to_pos"] / acc["steps_to_pos"]
               if (acc["steps_to_pos"] and sc["steps_to_pos"]) else None)
    ok = auc_gain > 0 and (speedup is None or speedup > 1.0)
    sp_s = f"{speedup:.2f}x" if speedup else "n/a"
    verdict = (f"ACCUMULATION HELPS — knowing {args.pretrain} makes {args.target} "
               f"faster: AUC gain {auc_gain:+.2f}, speedup {sp_s}. More games "
               f"known -> a new game learned faster (compounding across tasks; "
               f"multi-game pretraining fixes the P3 single-source negative)."
               if ok else
               f"NO ACCUMULATION GAIN — {args.pretrain}->{args.target}: AUC gain "
               f"{auc_gain:+.2f}, speedup {sp_s}. Encoder-sharing across these "
               f"games is not enough; next step = explicit skill-library + "
               f"recognition (the craft-world v4/v14 mechanism).")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v16_accumulate.json"), "w") as f:
        json.dump(dict(pretrain=args.pretrain, target=args.target,
                       accumulated=dict(auc=acc["auc"], final=acc["final"],
                                        steps=acc["steps_to_pos"], curve=acc["curve"]),
                       scratch=dict(auc=sc["auc"], final=sc["final"],
                                    steps=sc["steps_to_pos"], curve=sc["curve"]),
                       auc_gain=auc_gain, speedup=speedup, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
