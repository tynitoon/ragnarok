"""Development look (agent side + images for me): what is on screen just before my deaths in Assault?

Plays the base slot with 15 % random slots from the seed-98 cache, and at each of the agent's own
death signals saves the 6 frames before it (PNG, x3) and lists the parsed objects near me."""
import pickle
import sys

import numpy as np
from PIL import Image

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.hud import HudReader
from ragnarok.agent_v80.lab import make_parser

OUT = sys.argv[1] if len(sys.argv) > 1 else "craft_v6_out/v80/look"
C = pickle.load(open("craft_v6_out/v80/cache_assault_s98.pkl", "rb"))
ag = C["agent"]
env = AtariEnv("assault", seed=98)
rng = np.random.default_rng(3)
parser = make_parser(ag)
obs = env.reset()
tr = ag.perc.tracker
tr.pos, tr.miss, ag.gone = None, 0, 0
hud = HudReader(ag.perc.counters or []); hud.step(obs)
hist, deaths = [], 0
import os
os.makedirs(OUT, exist_ok=True)
choices = np.nonzero(ag.mask)[0]
for t in range(6000):
    me = ag._track(obs)
    hist.append((obs, me, parser.parse(obs)))
    hist = hist[-30:]
    slot = 1 if rng.random() > 0.15 else int(rng.choice(choices))
    obs, over = env.step(ag.adapter.to_button(slot))
    ev = hud.step(obs)
    if ag._dead(ev):
        deaths += 1
        for i, (o, m, objs) in enumerate(hist[-30:-6:3]):
            Image.fromarray(o).resize((480, 630), Image.NEAREST).save(f"{OUT}/death{deaths}_{i}.png")
        o, m, objs = hist[-14]
        near = [(round(x["cx"]), round(x["cy"]), x["area"], x["x1"] - x["x0"], x["y1"] - x["y0"], hex(x["colour"]))
                for x in objs if m is not None and abs(x["cx"] - m[0]) < 40 and abs(x["cy"] - m[1]) < 40]
        print(f"death {deaths} at t={t}: me={None if m is None else (round(m[0]), round(m[1]))} near objects {near}", flush=True)
        ag.gone = -10 ** 6
        if deaths >= 4:
            break
    if ag.gone < 0 and me is not None and tr.miss == 0:
        ag.gone = 0
    if over:
        obs = env.reset(); tr.pos, tr.miss = None, 0
