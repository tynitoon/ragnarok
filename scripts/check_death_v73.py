"""v73 pre-check (evaluator side): can the agent tell, from the screen only, that it lost a life?

Candidate agent-side signals, computed while playing random-with-repeat:
  vanish_k : onset of k consecutive steps where my tracker finds me nowhere (update and reacquire fail)
  hud      : an event on the agent's lives counter, when entry found one
Ground truth (experimenter only, never shown to the agent): env.lives() decreases and game over.
A signal onset is a hit when a true loss happens within [-30, +60] steps of it.

Usage: C:/Python314/python.exe -m scripts.check_death_v73 [games...]
"""
import json
import os
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.learner import Agent
from ragnarok.agent_v71.hud import HudReader

GAMES = ["assault", "freeway", "demon_attack", "name_this_game", "carnival", "breakout"]
KS = (1, 2, 4, 8, 16)
BEFORE, AFTER = 30, 60


def onsets(flags, k):
    out, run = [], 0
    for t, f in enumerate(flags):
        run = run + 1 if f else 0
        if run == k:
            out.append(t - k + 1)
    return out


def score(sig, truth):
    hits = [s for s in sig if any(-BEFORE <= t - s <= AFTER for t in truth)]
    rec = [t for t in truth if any(-BEFORE <= t - s <= AFTER for s in sig)]
    return dict(n=len(sig), precision=round(len(hits) / len(sig), 3) if sig else None,
                recall=round(len(rec) / len(truth), 3) if truth else None)


def check(game, seed=5, steps=8000, repeat=0.95):
    t0 = time.perf_counter()
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    ag = Agent()
    ag.enter(env, rng, seed=seed)
    obs = env.reset()
    tr = ag.perc.tracker
    tr.pos, tr.miss = None, 0
    ag.perc.locate(env, obs)
    hud = HudReader(ag.perc.counters or []); hud.step(obs)
    lives0 = env.lives()
    gone, hud_ev, truth, overs = [], [], [], []
    a = 0
    for t in range(steps):
        me = ag._track(obs)
        gone.append(me is None or tr.miss > 0)
        if rng.random() > repeat:
            a = int(rng.integers(env.n_actions))
        obs, over = env.step(a)
        ev = hud.step(obs)
        if ag.l_idx is not None and ag.l_idx in ev:
            hud_ev.append(t)
        lv = env.lives()                       # experimenter side only
        if lv < lives0:
            truth.append(t)
        lives0 = lv
        if over:
            overs.append(t)
            truth.append(t) if not truth or truth[-1] != t else None
            obs = env.reset()
            lives0 = env.lives()
            tr.pos, tr.miss = None, 0
            ag.perc.locate(env, obs)
            hud = HudReader(ag.perc.counters or []); hud.step(obs)
    res = dict(game=game, seed=seed, steps=steps, true_losses=len(truth), game_overs=len(overs),
               has_lives_counter=ag.l_idx is not None, untracked_frac=round(float(np.mean(gone)), 3),
               hud=score(hud_ev, truth) if ag.l_idx is not None else None,
               vanish={k: score(onsets(gone, k), truth) for k in KS}, secs=round(time.perf_counter() - t0))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    os.makedirs("craft_v6_out/v73", exist_ok=True)
    games = sys.argv[1:] or GAMES
    out = [check(g) for g in games]
    json.dump(out, open("craft_v6_out/v73/death_check.json", "w"), indent=1)
