"""v17 — MODEL-BASED Tetris: learn the DYNAMICS, then PLAN. Tests the owner's
hypothesis: understanding gravity/collision/rotation -> master Tetris in FAR
fewer games than model-free trial-and-error.

The agent learns a model M(board, piece) -> predicted outcome of each placement
[lines, holes, height, dead] (= "if I place the piece here, what happens" =
gravity + collision + line-completion). It then PLANS: pick the placement the
model says is best (imagine outcomes instead of trial-and-error). We measure
how many GAMES of experience it needs to play well, vs:
  - model-free PPO (v15 P4): ~170k games for ~63 lines/window,
  - a PERFECT model (the env's own dynamics) + planning: ~110 lines, 0 learning.

If the learned model reaches strong play in a few THOUSAND games, then learning
the dynamics (the "concepts") buys a huge sample-efficiency win — the owner's
"more understanding -> fewer tries", measured.

Usage: python -m scripts.modelbased_tetris_v17 [--rounds 20] [--smoke]
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
# planning score over predicted [lines, holes, height, dead]
W_LINES, W_HOLES, W_HEIGHT, W_DEAD = 4.0, 2.0, 0.06, 100.0


def plan_score(m):  # m: (...,4)
    return m[..., 0] * W_LINES - m[..., 1] * W_HOLES - m[..., 2] * W_HEIGHT - m[..., 3] * W_DEAD


class DynModel(nn.Module):
    """Predicts, from (board, piece), the outcome metrics of ALL placements."""
    def __init__(self, Hb, W, action_dim, hidden=256):
        super().__init__()
        self.Hb, self.W, self.A = Hb, W, action_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.emb = nn.Embedding(N_PIECE, 16)
        self.head = nn.Sequential(
            nn.Linear(16 * Hb * W + 16, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim * 4))

    def forward(self, board, piece):
        B = board.shape[0]
        x = self.conv(board.float().view(B, 1, self.Hb, self.W)).reshape(B, -1)
        h = torch.cat([x, self.emb(piece)], -1)
        return self.head(h).view(B, self.A, 4)


@torch.no_grad()
def collect(env, model, steps, explore, use_model):
    """Step the env (planning with the model, or random), recording
    (board, piece) -> true placement metrics. Returns data + games played."""
    boards, pieces, targets = [], [], []
    g0 = float(env.cum_games.sum())
    for _ in range(steps):
        tgt = env.evaluate_placements()                  # (N,A,4) TRUE outcomes
        boards.append(env.board.clone()); pieces.append(env.piece.clone())
        targets.append(tgt)
        if use_model:
            a = plan_score(model(env.board, env.piece)).argmax(1)
        else:
            a = torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE)
        rnd = torch.rand(env.num_envs, device=DEVICE) < explore
        a = torch.where(rnd, torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE), a)
        env.step(a)
    games = float(env.cum_games.sum()) - g0
    return (torch.cat(boards), torch.cat(pieces), torch.cat(targets)), games


@torch.no_grad()
def evaluate(env_cls, model, cfg, perfect=False, steps=400, n=128):
    env = env_cls(n, width=cfg["W"], height=cfg["Hb"], max_pieces=cfg["max_pieces"])
    for _ in range(steps):
        if perfect:
            a = plan_score(env.evaluate_placements()).argmax(1)
        else:
            a = plan_score(model(env.board, env.piece)).argmax(1)
        env.step(a)
    return env.stats()


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
    model = DynModel(args.height, args.width, env.action_dim, args.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    perfect = evaluate(DeviceVecTetris, model, cfg, perfect=True)
    print(f"[v17] device={DEVICE} | MODEL-BASED Tetris | PERFECT-model planner "
          f"(0 learning): {perfect['mean_lines']:.1f} lines | model-free PPO "
          f"(v15): ~63 lines after ~170k games", flush=True)

    t0 = time.perf_counter()
    total_games, curve = 0.0, []
    for rnd in range(1, args.rounds + 1):
        (boards, pieces, targets), games = collect(
            env, model, args.collect_steps, explore=max(0.1, 0.5 - rnd * 0.03),
            use_model=(rnd > 1))
        total_games += games
        B = boards.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(B, device=DEVICE)[:4096]
            pred = model(boards[idx], pieces[idx])
            loss = (pred - targets[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        st = evaluate(DeviceVecTetris, model, cfg)
        curve.append(dict(round=rnd, games=total_games, lines=st["mean_lines"]))
        print(f"  round {rnd:>2} | games~{total_games:>7,.0f} | learned-model "
              f"plays {st['mean_lines']:>6.1f} lines | loss {float(loss):.3f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    final = curve[-1]["lines"]
    # sample-efficiency: games to reach a strong threshold (e.g., 50 lines)
    reached = next((c for c in curve if c["lines"] >= 50), None)
    ok = final >= 50
    eff = (f"reached 50+ lines after ~{reached['games']:,.0f} games "
           f"(~{170000 / max(reached['games'],1):.0f}x fewer than model-free PPO's "
           f"~170k)" if reached else "did not reach 50 lines")
    verdict = (f"UNDERSTANDING THE DYNAMICS -> SAMPLE-EFFICIENT TETRIS — the "
               f"learned model plays {final:.1f} lines; {eff}. Learning the "
               f"concepts (gravity/collision/rotation) buys a large efficiency "
               f"win vs model-free trial-and-error (the owner's hypothesis)."
               if ok else
               f"PARTIAL — learned model reaches {final:.1f} lines; {eff}.")
    print(f"\n  -> {verdict}\n  total games ~{total_games:,.0f} | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v17_modelbased_tetris.json"), "w") as f:
        json.dump(dict(perfect_lines=perfect["mean_lines"], curve=curve,
                       final_lines=final, total_games=total_games,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
