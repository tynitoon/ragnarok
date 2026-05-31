"""v24 — INTEGRATION M4: the loop on the PIXEL GAMES (recognise game -> reuse skill).

Lifts the integrated recognise-and-reuse loop (M1-M3) to the real pixel games:
the agent has a small LIBRARY of game-skills (a trained policy per game); dropped
on a game, it RECOGNISES which one it is from raw pixels (a learned classifier),
then REUSES that game's skill to play. Decisive: recognition ~100% AND the
reused (recognised) skill plays its game well, while a mismatched skill fails.
This is "drop it on a game it knows -> it identifies it -> plays it".

Usage: python -m scripts.integrate_v24 [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout

GAMES = [("pong", DeviceVecPong), ("breakout", DeviceVecBreakout)]


def train_policy(game_cls, iters, n=256, img=48):
    env = game_cls(n, img=img, max_steps=800)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
    for _ in range(iters):
        ppo.train_iter(env, 32)
    return ppo


class Recognizer(nn.Module):
    def __init__(self, img, n_games):
        super().__init__()
        self.img = img
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img, img)).reshape(1, -1).shape[1]
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_games))

    def forward(self, obs):
        B = obs.shape[0]
        x = self.conv(obs.view(B, 3, self.img, self.img)).reshape(B, -1)
        return self.head(x)


@torch.no_grad()
def measure(ppo, game_cls, n=256, H=799, img=48):
    env = game_cls(n, img=img, max_steps=800)
    obs = env.state; ret = torch.zeros(n, device=DEVICE)
    for _ in range(H):
        obs, r, _, _, _ = env.step(ppo.act(obs, deterministic=True)); ret += r
    return float(ret.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pong-iters", type=int, default=80)
    p.add_argument("--breakout-iters", type=int, default=180)
    p.add_argument("--rec-iters", type=int, default=300)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.pong_iters, args.breakout_iters, args.rec_iters = 8, 8, 30

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v24] device={DEVICE} | INTEGRATION M4 | library of game-skills + "
          f"pixel recogniser -> recognise game & reuse skill", flush=True)
    t0 = time.perf_counter()
    # 1. library of skills (one policy per game)
    skills = [train_policy(DeviceVecPong, args.pong_iters, img=args.img),
              train_policy(DeviceVecBreakout, args.breakout_iters, img=args.img)]
    print(f"  trained {len(skills)} game-skills | {time.perf_counter()-t0:.0f}s",
          flush=True)

    # 2. recogniser: classify which game from a single pixel frame
    rec = Recognizer(args.img, len(GAMES)).to(DEVICE)
    ropt = torch.optim.Adam(rec.parameters(), lr=1e-3)
    envs = [gc(256, img=args.img, max_steps=800) for _, gc in GAMES]
    for _ in range(args.rec_iters):
        frames, labels = [], []
        for gi, env in enumerate(envs):
            env.step(torch.randint(0, env.action_dim, (256,), device=DEVICE))
            frames.append(env.state); labels.append(torch.full((256,), gi, device=DEVICE))
        x = torch.cat(frames); y = torch.cat(labels)
        loss = nn.functional.cross_entropy(rec(x), y)
        ropt.zero_grad(); loss.backward(); ropt.step()

    # 3. integrated agent: drop on each game -> recognise -> reuse that skill
    @torch.no_grad()
    def recognise(game_cls):
        env = game_cls(256, img=args.img, max_steps=800)
        for _ in range(5):
            env.step(torch.randint(0, env.action_dim, (256,), device=DEVICE))
        return int(rec(env.state).argmax(1).mode().values)      # majority vote

    results, correct = {}, 0
    for gi, (gname, gc) in enumerate(GAMES):
        rec_id = recognise(gc)
        correct += int(rec_id == gi)
        reused = measure(skills[rec_id], gc)                    # recognised skill
        wrong = measure(skills[1 - gi], gc)                     # mismatched skill
        right = measure(skills[gi], gc)                         # the correct skill (ref)
        results[gname] = dict(recognised=GAMES[rec_id][0], reused_return=reused,
                              mismatched_return=wrong, correct_return=right)
        print(f"  {gname:9s}: recognised as {GAMES[rec_id][0]:9s} | reused-skill "
              f"return {reused:+.2f} | mismatched {wrong:+.2f}", flush=True)

    acc = correct / len(GAMES)
    ok = acc == 1.0 and all(r["reused_return"] > r["mismatched_return"] + 1.0
                            for r in results.values())
    verdict = (f"GAME RECOGNISE-AND-REUSE WORKS — from raw pixels the agent "
               f"recognised every game ({acc:.0%}) and reused the right skill "
               f"(reused >> mismatched return on each). 'Drop it on a known game "
               f"-> it identifies it -> plays it' — the integration on the pixel "
               f"substrate."
               if ok else
               f"PARTIAL — recognition {acc:.0%}; see per-game returns.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v24_integration_m4.json"), "w") as f:
        json.dump(dict(recog_acc=acc, results=results, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
