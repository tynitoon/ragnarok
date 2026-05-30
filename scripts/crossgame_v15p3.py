"""v15 P3 — CROSS-GAME TRANSFER: does playing one game help master another?

The achievable core of "understand games generally / reuse win-lose competence":
train the agent on a SOURCE game, then learn a TARGET game two ways —
  TRANSFER: reuse the source game's CNN encoder (perception), fresh heads,
  SCRATCH:  fresh net —
and compare how fast each masters the target (eval-return curve + steps to a
positive return). If TRANSFER learns faster, then game-playing perception
learned on one game is reused to master another faster (compounding, on games)
— a step toward "drop it on any game and it picks it up quickly".

Honest control: v13 showed NAIVE single-source encoder reuse can HURT (negative
transfer) when features over-specialize; this measures whether cross-GAME reuse
helps or hurts.

Usage: python -m scripts.crossgame_v15p3 --source breakout --target pong [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from scripts.play_game_v15 import GAMES, evaluate


def _copy_encoder(dst, src_state):
    sd = dst.state_dict()
    for k in sd:
        if k.startswith("conv.") or k.startswith("fc."):
            sd[k] = src_state[k].clone()
    dst.load_state_dict(sd)


def train(game, iters, cfg, eval_every, init_state=None, freeze=False,
          target_return=None):
    game_cls = GAMES[game]
    kw = dict(img=cfg["img"], max_steps=cfg["max_steps"])
    env = game_cls(cfg["num_envs"], **kw)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=cfg["hidden"])
    if init_state is not None:
        _copy_encoder(net, init_state)
    if freeze:
        for nm, p in net.named_parameters():
            if nm.startswith("conv.") or nm.startswith("fc."):
                p.requires_grad_(False)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=cfg["entropy"], net=net)
    if freeze:
        ppo.opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                   lr=3e-4, eps=1e-5)
    curve, steps_to_pos = [], None
    for it in range(1, iters + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % eval_every == 0:
            ret, _ = evaluate(ppo, game_cls, kw, cfg["eval_envs"], cfg["eval_steps"])
            curve.append([it, ppo.total_steps, ret])
            if steps_to_pos is None and target_return is not None and ret >= target_return:
                steps_to_pos = ppo.total_steps
    auc = sum(c[2] for c in curve) / max(1, len(curve))
    return dict(curve=curve, auc=auc, steps_to_pos=steps_to_pos,
                final=curve[-1][2] if curve else 0.0, state=net.state_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=list(GAMES), default="breakout")
    p.add_argument("--target", choices=list(GAMES), default="pong")
    p.add_argument("--source-iters", type=int, default=300)
    p.add_argument("--target-iters", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--eval-steps", type=int, default=800)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--target-return", type=float, default=0.0)  # "winning-ish"
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.source_iters, args.target_iters, args.num_envs = 10, 10, 64
        args.eval_every, args.eval_steps = 5, 300

    cfg = {k: getattr(args, k) for k in
           ("img", "max_steps", "num_envs", "rollout", "hidden", "entropy",
            "eval_steps", "eval_envs")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v15-P3] device={DEVICE} | cross-game transfer {args.source} -> "
          f"{args.target}", flush=True)
    t0 = time.perf_counter()

    src = train(args.source, args.source_iters, cfg, args.eval_every)
    print(f"  source {args.source} trained: final return {src['final']:+.2f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    tr = train(args.target, args.target_iters, cfg, args.eval_every,
               init_state=src["state"], freeze=False, target_return=args.target_return)
    print(f"  TARGET {args.target} TRANSFER: auc {tr['auc']:+.2f} final "
          f"{tr['final']:+.2f} steps->{args.target_return} "
          f"{tr['steps_to_pos']} | {time.perf_counter()-t0:.0f}s", flush=True)
    sc = train(args.target, args.target_iters, cfg, args.eval_every,
               init_state=None, target_return=args.target_return)
    print(f"  TARGET {args.target} SCRATCH : auc {sc['auc']:+.2f} final "
          f"{sc['final']:+.2f} steps->{args.target_return} "
          f"{sc['steps_to_pos']} | {time.perf_counter()-t0:.0f}s", flush=True)

    auc_gain = tr["auc"] - sc["auc"]
    speedup = (sc["steps_to_pos"] / tr["steps_to_pos"]
               if (tr["steps_to_pos"] and sc["steps_to_pos"]) else None)
    ok = auc_gain > 0 and (speedup is None or speedup > 1.0)
    sp_s = f"{speedup:.2f}x" if speedup else "n/a"
    verdict = (f"CROSS-GAME REUSE HELPS — playing {args.source} makes {args.target} "
               f"faster to master: AUC gain {auc_gain:+.2f}, steps-to-win speedup "
               f"{sp_s}. Game-playing perception transfers across games."
               if ok else
               f"NO CROSS-GAME GAIN — {args.source}->{args.target}: AUC gain "
               f"{auc_gain:+.2f}, speedup {sp_s}. Reuse did not help here "
               f"(single-source encoder may over-specialize, cf. v13).")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"v15p3_{args.source}_{args.target}.json"), "w") as f:
        json.dump(dict(source=args.source, target=args.target,
                       transfer=dict(auc=tr["auc"], final=tr["final"],
                                     steps=tr["steps_to_pos"], curve=tr["curve"]),
                       scratch=dict(auc=sc["auc"], final=sc["final"],
                                    steps=sc["steps_to_pos"], curve=sc["curve"]),
                       auc_gain=auc_gain, speedup=speedup, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
