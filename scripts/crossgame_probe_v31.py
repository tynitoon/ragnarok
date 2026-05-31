"""v31 — is cross-GAME representation transfer GATED BY SIMILARITY? (exploratory)

P3/v16 found naive cross-game transfer fails. v31 probes WHY, cleanly: pretrain a
CNN encoder on Pong, then warm-start it (vs scratch) on a SIMILAR game (Breakout:
also paddle+ball) and a DISSIMILAR one (Snake: grid+food). All three games render
48x48x3; we share the conv ENCODER and give each game a fresh policy head.

Hypothesis: the Pong encoder HELPS Breakout (positive transfer — shared paddle/
ball/motion features) but NOT Snake (no transfer — different visual structure).
If so, cross-game transfer is gated by similarity — which is exactly why a
developmental agent must RECOGNISE which known skill applies (v25) instead of
blindly reusing one. Honest/exploratory: only 3 games, visually clustered (2
paddle-ball + 1 grid), so this maps the transfer-vs-similarity gradient rather
than proving broad cross-genre variety (which needs a larger game suite).

Usage: python -m scripts.crossgame_probe_v31 [--pre-iters 160] [--smoke]
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.environments.breakout import DeviceVecBreakout
from ragnarok.environments.snake import DeviceVecSnake


class SharedConvEncoder(nn.Module):
    """Same conv body as ConvPPONet, exposed as a reusable obs->features module."""
    def __init__(self, img_hw=48, hidden=256):
        super().__init__()
        self.img_hw = img_hw
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, img_hw, img_hw)).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.view(B, 3, self.img_hw, self.img_hw)
        return self.fc(self.conv(x).reshape(B, -1))


class GameNet(nn.Module):
    """Shared encoder + a per-game actor/critic head."""
    def __init__(self, encoder, action_dim, hidden=256):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0); nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def make_ppo(encoder, action_dim, img):
    return DiscretePPO(3 * img * img, action_dim, entropy=0.01,
                       net=GameNet(encoder, action_dim))


GAMES = {
    "pong": (lambda n: DeviceVecPong(n, img=48), 3),
    "breakout": (lambda n: DeviceVecBreakout(n, img=48), 3),
    "snake": (lambda n: DeviceVecSnake(n), 4),
}


@torch.no_grad()
def mean_return(ppo, make_env, steps=400, n=256):
    env = make_env(n)
    obs = env.state
    tot = torch.zeros(n, device=DEVICE)
    for _ in range(steps):
        obs, r, _, _, _ = env.step(ppo.act(obs, deterministic=True))
        tot += r
    return float(tot.mean())


def learn_curve(encoder, game, iters, eval_every, num_envs, img):
    make_env, adim = GAMES[game]
    ppo = make_ppo(encoder, adim, img)
    env = make_env(num_envs)
    curve = [(0, round(mean_return(ppo, make_env), 2))]
    for it in range(1, iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            curve.append((it, round(mean_return(ppo, make_env), 2)))
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pre-iters", type=int, default=160)
    p.add_argument("--transfer-iters", type=int, default=140)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.pre_iters, args.transfer_iters = 10, 20
        args.eval_every, args.num_envs = 10, 64

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v31] device={DEVICE} | cross-game transfer GATED BY SIMILARITY? | "
          f"Pong-encoder -> Breakout (similar) vs Snake (dissimilar), warm vs scratch",
          flush=True)
    t0 = time.perf_counter()

    # pretrain the shared encoder on Pong
    enc = SharedConvEncoder(args.img)
    pong = make_ppo(enc, 3, args.img)
    penv = GAMES["pong"][0](args.num_envs)
    for _ in range(args.pre_iters):
        pong.train_iter(penv, 32)
    print(f"  pretrained Pong encoder | {time.perf_counter()-t0:.0f}s", flush=True)

    out = {}
    for game in ["breakout", "snake"]:
        warm = learn_curve(copy.deepcopy(enc), game, args.transfer_iters,
                           args.eval_every, args.num_envs, args.img)
        scratch = learn_curve(SharedConvEncoder(args.img), game, args.transfer_iters,
                              args.eval_every, args.num_envs, args.img)
        # early-learning advantage: mean(warm - scratch) over the first half of checkpoints
        k = max(1, len(warm) // 2)
        early = sum(warm[i][1] - scratch[i][1] for i in range(1, k + 1)) / k
        out[game] = dict(warm=warm, scratch=scratch,
                         warm_final=warm[-1][1], scratch_final=scratch[-1][1],
                         early_advantage=round(early, 3))
        print(f"  {game:9s}: warm_final {warm[-1][1]:.2f} vs scratch_final "
              f"{scratch[-1][1]:.2f} | early-advantage {early:+.2f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    b, s = out["breakout"]["early_advantage"], out["snake"]["early_advantage"]
    gated = b > 0.05 and b >= s + 0.05
    verdict = (
        f"TRANSFER IS GATED BY SIMILARITY — the Pong encoder gives a positive "
        f"early-learning advantage on the SIMILAR game Breakout ({b:+.2f}) but "
        f"~none on the DISSIMILAR game Snake ({s:+.2f}). Cross-game reuse only "
        f"helps when the games share structure — which is precisely why a "
        f"developmental agent must RECOGNISE which known skill applies (v25) "
        f"before reusing it, rather than transferring blindly. (Exploratory: "
        f"3-game set; maps the gradient, does not prove broad cross-genre variety.)"
        if gated else
        f"INCONCLUSIVE/OTHER — Breakout early-advantage {b:+.2f}, Snake {s:+.2f}. "
        f"(warm vs scratch finals: breakout {out['breakout']['warm_final']:.2f}/"
        f"{out['breakout']['scratch_final']:.2f}, snake {out['snake']['warm_final']:.2f}/"
        f"{out['snake']['scratch_final']:.2f}.) Reported honestly.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v31_crossgame.json"), "w") as f:
        json.dump(dict(pre_iters=args.pre_iters, transfer_iters=args.transfer_iters,
                       results=out, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
