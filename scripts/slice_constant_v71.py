"""Best-constant-action baseline (experimenter-side control, exploratory): live score of pressing
one button for the whole game, per action, same conditions as the agent's live eval (seed 7,
sticky 0.25). A learned agent that only matches the best constant action learned one button.
Usage: C:/Python314/python.exe -m scripts.slice_constant_v71
"""
import json
from multiprocessing import Pool

import numpy as np

GAMES = ["freeway", "breakout", "assault", "carnival", "demon_attack", "name_this_game", "seaquest",
         "chopper_command", "amidar", "bank_heist", "crazy_climber", "wizard_of_wor"]


def const(g, games=3, max_steps=7000):
    from ragnarok.agent_v71.env import AtariEnv
    env = AtariEnv(g, seed=7, sticky=0.25)
    out = {}
    for a in range(env.n_actions):
        sc = []
        for _ in range(games):
            env.reset()
            for _ in range(max_steps):
                _, over = env.step(a)
                if over:
                    break
            sc.append(env.true_score())
        out[env.actions[a].name] = float(np.mean(sc))
    return g, out


if __name__ == "__main__":
    with Pool(10) as p:
        res = dict(p.map(const, GAMES))
    json.dump(res, open("craft_v6_out/v71_constant_baseline.json", "w"), indent=1)
    for g, o in res.items():
        best = max(o, key=o.get)
        print(f"{g:16s} best constant {best:10s} {o[best]:8.1f} | {o}")
