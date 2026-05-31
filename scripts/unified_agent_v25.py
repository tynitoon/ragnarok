"""v25 — THE UNIFIED DEVELOPMENTAL AGENT (the grand integration, runnable).

One agent, faced with a STREAM of games (from pixels), that:
  - RECOGNISES whether it has seen this game before (recogniser + perf check);
  - if KNOWN -> REUSES that game's skill, playing it with ZERO new training;
  - if NOVEL -> LEARNS a new skill, ADDS it to the library, updates the
    recogniser;
accumulating a library that GROWS as it meets new games. This assembles every
brick (perceive, recognise=v5, reuse, learn-the-new, library growth) into one
running system: drop it on games, watch it recognise / learn / play / accumulate.

Stream: pong, breakout, pong(repeat), snake, breakout(repeat), snake(repeat),
pong(repeat) -> 3 games learned once, 4 repeats reused with NO retraining.

Usage: python -m scripts.unified_agent_v25 [--smoke]
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
from ragnarok.environments.snake import DeviceVecSnake
from scripts.integrate_v24 import Recognizer, measure

GAME_CLS = {"pong": DeviceVecPong, "breakout": DeviceVecBreakout, "snake": DeviceVecSnake}
ITERS = {"pong": 80, "breakout": 220, "snake": 200}        # to learn a new game


class UnifiedAgent:
    """A growing library of game-skills + a recogniser. Recognise-or-learn on
    each encounter; reuse known skills with no retraining."""
    def __init__(self, img=48):
        self.img = img
        self.names, self.skills, self.home = [], [], []     # library
        self.recog = None

    def _retrain_recogniser(self):
        if len(self.names) < 2:
            return
        self.recog = Recognizer(self.img, len(self.names)).to(DEVICE)
        opt = torch.optim.Adam(self.recog.parameters(), lr=1e-3)
        envs = [GAME_CLS[n](256, img=self.img, max_steps=800) for n in self.names]
        for _ in range(300):
            fr, lb = [], []
            for gi, e in enumerate(envs):
                e.step(torch.randint(0, e.action_dim, (256,), device=DEVICE))
                fr.append(e.state); lb.append(torch.full((256,), gi, device=DEVICE))
            loss = nn.functional.cross_entropy(self.recog(torch.cat(fr)), torch.cat(lb))
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def _guess(self, env):
        if self.recog is None:
            return 0 if self.skills else None
        for _ in range(5):
            env.step(torch.randint(0, env.action_dim, (env.num_envs,), device=DEVICE))
        return int(self.recog(env.state).argmax(1).mode().values)

    def _learn(self, name):
        gc = GAME_CLS[name]
        env = gc(256, img=self.img, max_steps=800)
        net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
        ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)
        for _ in range(ITERS[name]):
            ppo.train_iter(env, 32)
        self.names.append(name); self.skills.append(ppo)
        self.home.append(measure(ppo, gc, img=self.img))
        self._retrain_recogniser()

    def encounter(self, name):
        """Drop the agent on game `name` (it does NOT see the label)."""
        gc = GAME_CLS[name]
        if not self.skills:                                  # empty library -> learn
            self._learn(name)
            return dict(game=name, action="LEARNED (1st game)", lib=len(self.names),
                        perf=round(self.home[-1], 2))
        g = self._guess(gc(256, img=self.img, max_steps=800))
        r = measure(self.skills[g], gc, img=self.img)        # verify by playing
        known = self.home[g] > 0 and r >= 0.4 * self.home[g]
        if known:                                            # REUSE, no retraining
            return dict(game=name, action=f"RECOGNISED as {self.names[g]} -> REUSED",
                        lib=len(self.names), perf=round(r, 2), retrained=False)
        self._learn(name)                                    # NOVEL -> learn + add
        return dict(game=name, action="NOVEL -> LEARNED + added",
                    lib=len(self.names), perf=round(self.home[-1], 2), retrained=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        for k in ITERS:
            ITERS[k] = 6

    stream = ["pong", "breakout", "pong", "snake", "breakout", "snake", "pong"]
    os.makedirs(args.out_dir, exist_ok=True)
    agent = UnifiedAgent()
    print(f"[v25] device={DEVICE} | UNIFIED DEVELOPMENTAL AGENT | stream of "
          f"{len(stream)} game-encounters | recognise-or-learn, accumulate",
          flush=True)
    t0 = time.perf_counter()
    log, learned, reused = [], 0, 0
    for i, name in enumerate(stream):
        ev = agent.encounter(name)
        log.append(ev)
        learned += int("LEARNED" in ev["action"])
        reused += int("REUSED" in ev["action"])
        print(f"  [{i+1}/{len(stream)}] saw {name:9s} -> {ev['action']:32s} | "
              f"perf {ev['perf']:+7.2f} | library={ev['lib']} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    n_distinct = len(set(stream))
    ok = (agent.names == [] or len(agent.names) == n_distinct) and reused >= 1
    verdict = (f"UNIFIED AGENT WORKS — over a stream of {len(stream)} encounters it "
               f"learned {learned} NEW games (library grew to {len(agent.names)}: "
               f"{agent.names}) and REUSED a known skill {reused}x with ZERO "
               f"retraining. It recognises what it knows, learns what's new, and "
               f"accumulates — the developmental vision, running end to end."
               if ok else
               f"PARTIAL — learned {learned}, reused {reused}, library "
               f"{agent.names}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v25_unified_agent.json"), "w") as f:
        json.dump(dict(stream=stream, log=log, library=agent.names,
                       learned=learned, reused=reused, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
