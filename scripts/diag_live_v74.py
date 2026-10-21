"""v74 diagnostic: where does a learned policy lose against the constant button over a WHOLE game?

Plays full games in live conditions (sticky 0.25) with (a) the agent's greedy policy and (b) the
agent's constant base slot, both through the agent's own perception and adapter. Per game: true
score and lives (EVALUATOR side, diagnosis only), steps survived, the agent's own score events and
death signals, button mix, and score per 500-step phase (where in the game the gap appears).

Usage: C:/Python314/python.exe -m scripts.diag_live_v74 CACHE.pkl POLICY.pt CONST_SLOT [games] [seed]
"""
import copy
import json
import pickle
import sys

import numpy as np
import torch

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.adapter import SLOTS
from ragnarok.agent_v71.policy import Policy
from ragnarok.agent_v71.hud import HudReader


def play(ag, game, seed, games, mode, slot, max_steps=7000, commit=1):
    env = AtariEnv(game, seed=seed, sticky=0.25)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(games):
        obs = env.reset()
        tr = ag.perc.tracker
        tr.pos, tr.miss, ag.gone = None, 0, 0
        hud = HudReader(ag.perc.counters or []); hud.step(obs)
        prev, ev_n, deaths, lives0, true_losses = None, 0, 0, env.lives(), 0
        phase, counts = [], np.zeros(len(SLOTS), int)
        last_true = 0.0
        for t in range(max_steps):
            me = ag._track(obs)
            if mode == "policy":
                a2, e2, _, _, prev = ag.act(obs, me, prev, rng, temp=0.05, eps=0.02)
                if t % commit == 0:          # re-decide every `commit` steps (the hold the values measured)
                    a, e = a2, e2
            else:
                e = slot
                a = ag.adapter.to_button(slot)
            counts[e] += 1
            obs, over = env.step(a)
            ev = hud.step(obs)
            ev_n += int(ag.s_idx is not None and ag.s_idx in ev)
            if ag._dead(ev):
                deaths += 1
                ag.gone = -10 ** 6 if (ag.vanish_k is not None and ag.gone >= ag.vanish_k) else ag.gone   # count each vanish once
            if ag.gone < 0 and not (me is None or tr.miss > 0):
                ag.gone = 0
            lv = env.lives()
            true_losses += int(lv < lives0); lives0 = lv
            if (t + 1) % 500 == 0:
                phase.append(env.true_score() - last_true); last_true = env.true_score()
            if over:
                break
        out.append(dict(true=env.true_score(), steps=t + 1, events=ev_n, agent_deaths=deaths,
                        true_losses=true_losses, phase500=phase,
                        mix={SLOTS[i]: round(float(c) / max(1, counts.sum()), 3) for i, c in enumerate(counts) if c}))
    return out


def main(cache, pol, slot, games=6, seed=99):
    C = pickle.load(open(cache, "rb"))
    ag = C["agent"]
    if not hasattr(ag, "base_slot"):
        ag.base_slot, ag.live_temp = None, 1.0
    ag.policy = Policy(len(SLOTS))
    ag.policy.net.load_state_dict(torch.load(pol, map_location=ag.policy.device))
    s = SLOTS.index(slot)
    res = {}
    for mode in ("policy", "const"):
        res[mode] = play(ag, C["rec"]["game"], seed, games, mode, s, commit=int(__import__("os").environ.get("COMMIT", "1")))
        r = res[mode]
        print(mode, "true", np.mean([g["true"] for g in r]), "steps", np.mean([g["steps"] for g in r]),
              "events", np.mean([g["events"] for g in r]), "agent_deaths", np.mean([g["agent_deaths"] for g in r]),
              "true_losses", np.mean([g["true_losses"] for g in r]), flush=True)
        print("   phases", [g["phase500"][:8] for g in r][:3], flush=True)
        print("   mix", r[0]["mix"], flush=True)
    json.dump(res, open(pol.replace(".pt", "_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(int(x) for x in a[3:5]))
