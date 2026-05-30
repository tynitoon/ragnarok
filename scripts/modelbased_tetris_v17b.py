"""v17b — model-based Tetris, FACTORIZED: learn ONLY where the piece falls.

v17's model tried to predict all outcomes at once -> too hard, weak planning
(7.8 lines). Fix: learn only the LANDING of each placement (= the pure
gravity+collision concept; a single, always-defined target per placement), then
compute lines/holes/height/death ANALYTICALLY from the predicted landing
(env.metrics_at) and plan. Faithful to "learn the concept" + far more accurate.

Reference points:
  - PERFECT model (env dynamics) + plan: ~110 lines, 0 learning.
  - TRUE-landing planner (env.placement_landings -> metrics_at): sanity, should
    match perfect -> validates the landing->metrics->plan pipeline.
  - model-free PPO: ~63 lines after ~170k games.
Question: how many games does the LEARNED-landing agent need to play well?

Usage: python -m scripts.modelbased_tetris_v17b [--rounds 20] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tetris import DeviceVecTetris

N_PIECE = 7
W_LINES, W_HOLES, W_HEIGHT, W_DEAD = 4.0, 2.0, 0.06, 100.0


def plan_score(m):
    return m[..., 0] * W_LINES - m[..., 1] * W_HOLES - m[..., 2] * W_HEIGHT - m[..., 3] * W_DEAD


class LandingModel(nn.Module):
    """board + piece -> predicted landing (0..1, x height) for each placement."""
    def __init__(self, Hb, W, action_dim, hidden=256):
        super().__init__()
        self.Hb, self.W, self.A = Hb, W, action_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.emb = nn.Embedding(N_PIECE, 16)
        self.head = nn.Sequential(
            nn.Linear(16 * Hb * W + 16, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim))

    def forward(self, board, piece):
        B = board.shape[0]
        x = self.conv(board.float().view(B, 1, self.Hb, self.W)).reshape(B, -1)
        return self.head(torch.cat([x, self.emb(piece)], -1))   # (B, A) in [0,1]-ish


def landings_from_model(env, model):
    pred = model(env.board, env.piece) * env.Hb              # un-normalise
    return pred.round().long().clamp(0, env.Hb - 1)


@torch.no_grad()
def evaluate(env_cls, cfg, mode, model=None, steps=400, n=128):
    env = env_cls(n, width=cfg["W"], height=cfg["Hb"], max_pieces=cfg["max_pieces"])
    for _ in range(steps):
        if mode == "perfect":
            a = plan_score(env.evaluate_placements()).argmax(1)
        elif mode == "true":
            a = plan_score(env.metrics_at(env.placement_landings().clamp(min=0))).argmax(1)
        else:                                                # learned
            a = plan_score(env.metrics_at(landings_from_model(env, model))).argmax(1)
        env.step(a)
    return env.stats()["mean_lines"]


@torch.no_grad()
def collect(env, model, steps, explore, use_model):
    boards, pieces, lands = [], [], []
    g0 = float(env.cum_games.sum())
    for _ in range(steps):
        boards.append(env.board.clone()); pieces.append(env.piece.clone())
        lands.append(env.placement_landings().clamp(min=0))      # TRUE landings (target)
        if use_model:
            a = plan_score(env.metrics_at(landings_from_model(env, model))).argmax(1)
        else:
            a = torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE)
        rnd = torch.rand(env.num_envs, device=DEVICE) < explore
        a = torch.where(rnd, torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE), a)
        env.step(a)
    games = float(env.cum_games.sum()) - g0
    return torch.cat(boards), torch.cat(pieces), torch.cat(lands), games


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--collect-steps", type=int, default=60)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=14)
    p.add_argument("--max-pieces", type=int, default=300)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.rounds, args.collect_steps, args.epochs, args.num_envs = 3, 20, 10, 32

    cfg = dict(W=args.width, Hb=args.height, max_pieces=args.max_pieces)
    os.makedirs(args.out_dir, exist_ok=True)
    env = DeviceVecTetris(args.num_envs, width=args.width, height=args.height,
                          max_pieces=args.max_pieces)
    model = LandingModel(args.height, args.width, env.action_dim, args.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    perfect = evaluate(DeviceVecTetris, cfg, "perfect")
    true_ref = evaluate(DeviceVecTetris, cfg, "true")
    print(f"[v17b] device={DEVICE} | FACTORIZED model-based Tetris | perfect "
          f"{perfect:.1f} lines | true-landing planner {true_ref:.1f} (pipeline "
          f"sanity) | PPO ~63 lines / ~170k games", flush=True)

    t0, total_games, curve = time.perf_counter(), 0.0, []
    for rnd in range(1, args.rounds + 1):
        boards, pieces, lands, games = collect(
            env, model, args.collect_steps, explore=max(0.1, 0.5 - rnd * 0.03),
            use_model=(rnd > 1))
        total_games += games
        B = boards.shape[0]
        tgt = lands.float() / args.height
        for _ in range(args.epochs):
            idx = torch.randperm(B, device=DEVICE)[:4096]
            loss = (model(boards[idx], pieces[idx]) - tgt[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        lines = evaluate(DeviceVecTetris, cfg, "learned", model)
        curve.append(dict(round=rnd, games=total_games, lines=lines))
        print(f"  round {rnd:>2} | games~{total_games:>7,.0f} | learned-landing "
              f"plays {lines:>6.1f} lines | loss {loss.item():.4f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]["lines"]
    reached = next((c for c in curve if c["lines"] >= 50), None)
    ok = final >= 50
    eff = (f"reached 50+ lines after ~{reached['games']:,.0f} games "
           f"(~{170000/max(reached['games'],1):.0f}x fewer than model-free PPO's ~170k)"
           if reached else f"reached {final:.1f} lines (<50)")
    verdict = (f"LEARN THE LANDING (gravity+collision) -> SAMPLE-EFFICIENT TETRIS: "
               f"{eff}. Learning the right concept (where the piece falls) and "
               f"planning = far fewer tries than model-free (owner's hypothesis)."
               if ok else
               f"PARTIAL — learned-landing plays {final:.1f} lines; {eff}. "
               f"(perfect {perfect:.1f}, true-landing {true_ref:.1f}.)")
    print(f"\n  -> {verdict}\n  total games ~{total_games:,.0f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v17b_landing.json"), "w") as f:
        json.dump(dict(perfect=perfect, true_landing=true_ref, curve=curve,
                       final=final, total_games=total_games, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
