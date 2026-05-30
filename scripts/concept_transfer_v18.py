"""v18 — CONCEPT TRANSFER: does a learned 'gravity/collision' model generalise
to UNSEEN pieces? (The owner's vision: concepts are universal and reusable.)

Pixel-feature transfer failed (P3/v16) because features over-specialise. But a
DYNAMICS concept — "where does a falling shape land on a surface" (gravity +
collision) — is UNIVERSAL: it depends on the surface profile and the shape, not
on which game/piece. Test: train a SHAPE-conditioned landing model (it sees the
piece's GEOMETRY, not an id) on a SUBSET of tetrominoes, then evaluate it
ZERO-SHOT on the UNSEEN pieces. If it plays well on shapes it never trained on,
it learned the general physics — a reusable concept — which transfers where
pixel-features did not.

Reference: perfect/true-landing planner ~105-110 lines; learned (v17b, all
pieces) ~46.

Usage: python -m scripts.concept_transfer_v18 [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tetris import DeviceVecTetris

W_LINES, W_HOLES, W_HEIGHT, W_DEAD = 4.0, 2.0, 0.06, 100.0
TRAIN_PIECES = [0, 1, 2, 3, 4]      # I,O,T,S,Z
TEST_PIECES = [5, 6]                # L,J  (UNSEEN shapes)


def plan_score(m):
    return m[..., 0] * W_LINES - m[..., 1] * W_HOLES - m[..., 2] * W_HEIGHT - m[..., 3] * W_DEAD


def build_geom(cells):                       # cells (7,4,4,2) -> (7,4,4,4) masks
    G = torch.zeros(7, 4, 4, 4, device=DEVICE)
    for p in range(7):
        for r in range(4):
            for k in range(4):
                cr, cc = int(cells[p, r, k, 0]), int(cells[p, r, k, 1])
                G[p, r, cr, cc] = 1.0
    return G


class ShapeLandingModel(nn.Module):
    """board + piece GEOMETRY -> predicted landing of each placement. Conditioning
    on geometry (not a piece-id) is what lets it generalise to unseen shapes."""
    def __init__(self, Hb, W, action_dim, geom, hidden=256):
        super().__init__()
        self.Hb, self.W, self.A = Hb, W, action_dim
        self.register_buffer("geom", geom)               # (7,4,4,4)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.gnet = nn.Sequential(nn.Linear(4 * 4 * 4, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(16 * Hb * W + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, board, piece):
        B = board.shape[0]
        x = self.conv(board.float().view(B, 1, self.Hb, self.W)).reshape(B, -1)
        g = self.gnet(self.geom[piece].reshape(B, -1))
        return self.head(torch.cat([x, g], -1))


def landings_from_model(env, model):
    return (model(env.board, env.piece) * env.Hb).round().long().clamp(0, env.Hb - 1)


@torch.no_grad()
def eval_lines(model, piece_set, cfg, steps=400, n=128):
    env = DeviceVecTetris(n, width=cfg["W"], height=cfg["Hb"],
                          max_pieces=cfg["max_pieces"], piece_set=piece_set)
    for _ in range(steps):
        a = plan_score(env.metrics_at(landings_from_model(env, model))).argmax(1)
        env.step(a)
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


def train_model(piece_set, rounds, cfg, geom):
    env = DeviceVecTetris(cfg["num_envs"], width=cfg["W"], height=cfg["Hb"],
                          max_pieces=cfg["max_pieces"], piece_set=piece_set)
    model = ShapeLandingModel(cfg["Hb"], cfg["W"], env.action_dim, geom).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    total = 0.0
    for rnd in range(1, rounds + 1):
        boards, pieces, lands, games = collect(
            env, model, cfg["collect_steps"], max(0.1, 0.5 - rnd * 0.03), rnd > 1)
        total += games
        tgt = lands.float() / cfg["Hb"]; B = boards.shape[0]
        for _ in range(cfg["epochs"]):
            idx = torch.randperm(B, device=DEVICE)[:4096]
            loss = (model(boards[idx], pieces[idx]) - tgt[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=18)
    p.add_argument("--scratch-rounds", type=int, default=18)
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
        args.rounds, args.scratch_rounds, args.collect_steps = 3, 3, 20
        args.epochs, args.num_envs = 10, 32

    cfg = dict(W=args.width, Hb=args.height, max_pieces=args.max_pieces,
               num_envs=args.num_envs, collect_steps=args.collect_steps,
               epochs=args.epochs)
    os.makedirs(args.out_dir, exist_ok=True)
    geom = build_geom(DeviceVecTetris(2)._cells)
    print(f"[v18] device={DEVICE} | CONCEPT TRANSFER | train pieces "
          f"{TRAIN_PIECES} -> ZERO-SHOT on UNSEEN {TEST_PIECES}", flush=True)
    t0 = time.perf_counter()

    model, train_games = train_model(TRAIN_PIECES, args.rounds, cfg, geom)
    on_train = eval_lines(model, TRAIN_PIECES, cfg)
    zeroshot = eval_lines(model, TEST_PIECES, cfg)            # UNSEEN pieces!
    print(f"  trained on {TRAIN_PIECES} ({train_games:,.0f} games): plays "
          f"{on_train:.1f} on trained pieces | ZERO-SHOT on unseen {TEST_PIECES}: "
          f"{zeroshot:.1f} lines | {time.perf_counter()-t0:.0f}s", flush=True)

    scratch, sc_games = train_model(TEST_PIECES, args.scratch_rounds, cfg, geom)
    on_test_scratch = eval_lines(scratch, TEST_PIECES, cfg)
    print(f"  scratch on {TEST_PIECES} ({sc_games:,.0f} games): {on_test_scratch:.1f} "
          f"lines | {time.perf_counter()-t0:.0f}s", flush=True)

    # decisive: zero-shot on unseen pieces is strong (concept generalised)
    ok = zeroshot >= 0.6 * on_train and zeroshot >= 15
    verdict = (f"CONCEPT TRANSFERS — the gravity/collision model, trained only on "
               f"{TRAIN_PIECES}, plays {zeroshot:.1f} lines ZERO-SHOT on UNSEEN "
               f"pieces {TEST_PIECES} ({on_train:.1f} on trained ones). It learned "
               f"the general physics, not memorised shapes -> a reusable concept "
               f"that transfers (where pixel-features did NOT, cf. P3/v16)."
               if ok else
               f"WEAK TRANSFER — zero-shot {zeroshot:.1f} vs {on_train:.1f} on "
               f"trained pieces (scratch-on-unseen {on_test_scratch:.1f}). The "
               f"learned landing did not fully generalise to unseen shapes.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v18_concept_transfer.json"), "w") as f:
        json.dump(dict(train_pieces=TRAIN_PIECES, test_pieces=TEST_PIECES,
                       on_train=on_train, zeroshot_unseen=zeroshot,
                       scratch_unseen=on_test_scratch, train_games=train_games,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
