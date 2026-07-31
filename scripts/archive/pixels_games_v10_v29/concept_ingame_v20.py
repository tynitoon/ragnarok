"""v20 — concept transfer IN THE GAME, with variety (the v19 recipe applied).

v18 failed (gravity model on 5 tetrominoes -> 7.9 lines zero-shot on 2 unseen).
v19 showed the rule generalises once it can't memorise (enough shapes). Here we
apply that IN THE ACTUAL GAME: generate MANY random 4-cell shapes, train the
shape-conditioned landing model on a TRAIN subset, then PLAY the HELD-OUT shapes
ZERO-SHOT. If it plays unseen shapes well, the gravity concept genuinely
transfers in a real game (turning v18's 7.9 into a real win).

Usage: python -m scripts.concept_ingame_v20 [--n-shapes 120] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tetris import DeviceVecTetris
from scripts.concept_transfer_v18 import (
    ShapeLandingModel, plan_score, landings_from_model)


def random_shapes(n, seed=0):
    """n random 4-cell shapes, each with its 4 rotations -> (n,4,4,2)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        idx = torch.randperm(16, generator=g)[:4]
        cur = [(int(i) // 4, int(i) % 4) for i in idx]
        rots = []
        for _ in range(4):
            mnr = min(r for r, c in cur); mnc = min(c for r, c in cur)
            rots.append([(r - mnr, c - mnc) for r, c in cur])
            mxr = max(r for r, c in cur)
            cur = [(c, mxr - r) for r, c in cur]          # rotate 90 CW
        out.append(rots)
    return torch.tensor(out, dtype=torch.long)            # (n,4,4,2)


def build_geom(cells):                                    # (n,4,4,2)->(n,4,4,4)
    n = cells.shape[0]
    G = torch.zeros(n, 4, 4, 4, device=DEVICE)
    for p in range(n):
        for r in range(4):
            for k in range(4):
                G[p, r, int(cells[p, r, k, 0]), int(cells[p, r, k, 1])] = 1.0
    return G


def make_env(n, shapes, piece_set, cfg):
    return DeviceVecTetris(n, width=cfg["W"], height=cfg["Hb"],
                           max_pieces=cfg["max_pieces"], shapes=shapes,
                           piece_set=piece_set)


@torch.no_grad()
def eval_lines(model, shapes, piece_set, cfg, steps=400, n=128):
    env = make_env(n, shapes, piece_set, cfg)
    for _ in range(steps):
        env.step(plan_score(env.metrics_at(landings_from_model(env, model))).argmax(1))
    return env.stats()["mean_lines"]


@torch.no_grad()
def collect(env, model, steps, explore, use_model):
    boards, pieces, lands = [], [], []
    g0 = float(env.cum_games.sum())
    for _ in range(steps):
        boards.append(env.board.clone()); pieces.append(env.piece.clone())
        lands.append(env.placement_landings().clamp(min=0))
        if use_model:
            a = plan_score(env.metrics_at(landings_from_model(env, model))).argmax(1)
        else:
            a = torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE)
        rnd = torch.rand(env.num_envs, device=DEVICE) < explore
        a = torch.where(rnd, torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE), a)
        env.step(a)
    return torch.cat(boards), torch.cat(pieces), torch.cat(lands), \
        float(env.cum_games.sum()) - g0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-shapes", type=int, default=120)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--rounds", type=int, default=18)
    p.add_argument("--collect-steps", type=int, default=60)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=14)
    p.add_argument("--max-pieces", type=int, default=300)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n_shapes, args.n_test, args.rounds = 30, 6, 3
        args.collect_steps, args.epochs, args.num_envs = 20, 10, 32

    cfg = dict(W=args.width, Hb=args.height, max_pieces=args.max_pieces)
    os.makedirs(args.out_dir, exist_ok=True)
    shapes = random_shapes(args.n_shapes)
    geom = build_geom(shapes)
    train_set = list(range(args.n_shapes - args.n_test))
    test_set = list(range(args.n_shapes - args.n_test, args.n_shapes))   # HELD-OUT
    print(f"[v20] device={DEVICE} | in-game concept transfer | train on "
          f"{len(train_set)} shapes -> ZERO-SHOT play {len(test_set)} UNSEEN "
          f"shapes (vs v18: 5 shapes -> 7.9 zero-shot)", flush=True)

    env = make_env(args.num_envs, shapes, train_set, cfg)
    model = ShapeLandingModel(args.height, args.width, env.action_dim, geom).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, total = time.perf_counter(), 0.0
    for rnd in range(1, args.rounds + 1):
        boards, pieces, lands, games = collect(
            env, model, args.collect_steps, max(0.1, 0.5 - rnd * 0.03), rnd > 1)
        total += games
        tgt = lands.float() / args.height; B = boards.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(B, device=DEVICE)[:4096]
            loss = (model(boards[idx], pieces[idx]) - tgt[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if rnd % 6 == 0 or rnd == args.rounds:
            zs = eval_lines(model, shapes, test_set, cfg)
            print(f"  round {rnd:>2} | games~{total:,.0f} | ZERO-SHOT unseen "
                  f"{zs:.1f} lines | loss {loss.item():.4f} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    on_train = eval_lines(model, shapes, train_set, cfg)
    zeroshot = eval_lines(model, shapes, test_set, cfg)
    ok = zeroshot >= 0.7 * on_train and zeroshot >= 20
    verdict = (f"CONCEPT TRANSFERS IN-GAME — trained on {len(train_set)} shapes, "
               f"the agent plays {zeroshot:.1f} lines ZERO-SHOT on {len(test_set)} "
               f"UNSEEN shapes ({on_train:.1f} on trained). Variety -> the gravity "
               f"concept generalises in the real game (v18's 7.9 was lack of "
               f"variety; v19's rule-learning confirmed in-game)."
               if ok else
               f"PARTIAL — zero-shot {zeroshot:.1f} vs {on_train:.1f} on trained "
               f"shapes. Better than v18's 7.9 if higher, but not full transfer.")
    print(f"\n  -> {verdict}\n  total games ~{total:,.0f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v20_ingame_transfer.json"), "w") as f:
        json.dump(dict(n_shapes=args.n_shapes, n_test=args.n_test,
                       on_train=on_train, zeroshot_unseen=zeroshot,
                       train_games=total, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
