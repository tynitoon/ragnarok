"""Diagnostic only (evaluator side): offset between the agent's score events and RAM score rises.
For each agent score event, the signed offset (true - agent) to the nearest true rise within 40
steps; and which counter region the agent reads. Usage: python -m scripts.score_lag_v81 GAME"""
import json
import sys

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.hud import HudReader
from ragnarok.agent_v71.learner import Agent

game = sys.argv[1]
SEED = 8101
env = AtariEnv(game, seed=SEED)
rng = np.random.default_rng(SEED)
ag = Agent(vanish_k=2)
ag.enter(env, rng, seed=SEED)
env2 = AtariEnv(game, seed=SEED + 1000)
obs = env2.reset()
hud = HudReader(ag.perc.counters or []); hud.step(obs)
r2 = np.random.default_rng(SEED + 1)
sc_a, sc_t, a, score0 = [], [], 0, env2.true_score()
for t in range(8000):
    if r2.random() > 0.95:
        a = int(r2.integers(env2.n_actions))
    obs, over = env2.step(a)
    ev = hud.step(obs)
    if ag.s_idx is not None and ag.s_idx in ev:
        sc_a.append(t)
    s = env2.true_score()
    if s > score0:
        sc_t.append(t)
    score0 = s
    if over:
        obs = env2.reset(); score0 = env2.true_score(); hud = HudReader(ag.perc.counters or []); hud.step(obs)
off = []
for p in sc_a:
    d = [t - p for t in sc_t if abs(t - p) <= 40]
    off.append(min(d, key=abs) if d else None)
print(json.dumps(dict(game=game, roles={k: list(v) for k, v in ag.roles.items()}, counters=[list(c) for c in (ag.perc.counters or [])],
                      agent=len(sc_a), true=len(sc_t), offsets=off[:60],
                      unmatched=sum(o is None for o in off))))
