"""v80 gate G0 (docs/ARC4_V80_DESIGN.md §4), EVALUATOR side: does the agent see the objects?

Freeway: the agent's visible tracks against the RAM car positions (ragnarok/evaluation/
freeway_v70.decode_freeway_ram, calibrated judge). A RAM car counts as seen if a visible track's
centre lies within 4 px of the centre of its on-screen box. Pass: >= 90 % of cars seen.
Assault: my own track. A step counts as bad if the agent's tracked 'me' has no object within 6 px
or the object there is a merge (area > 2 x my learned size). Pass: <= 30 % bad steps.
Play: random actions with repeat 0.95, development seed 98 only, 2000 steps. RAM is read by this
script for scoring only; nothing from it reaches the agent's objects.
Usage: C:/Python314/python.exe -m scripts.check_objects_v80 freeway|assault [steps]
"""
import json
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.learner import Agent
from ragnarok.agent_v80.objects import Parser, Tracker, mark_me

SEED = 98


def run(game, steps=2000):
    t0 = time.perf_counter()
    env = AtariEnv(game, seed=SEED)
    rng = np.random.default_rng(SEED)
    ag = Agent(vanish_k=2)
    ag.enter(env, rng, seed=SEED)
    parser = Parser(ag.perc.bg, boxes=ag.perc.counters or [], me_colours=ag.perc.tracker.colours)
    tracker = Tracker()
    obs = env.reset()
    tr = ag.perc.tracker
    tr.pos, tr.miss, ag.gone = None, 0, 0
    a, seen, total, bad, mesteps, nobj, areas = 0, 0, 0, 0, 0, [], []
    for t in range(steps):
        me = ag._track(obs)
        objs = parser.parse(obs)
        tracker.step(objs)
        vis = tracker.visible()
        nobj.append(len(vis))
        if game == "freeway":
            from ragnarok.evaluation.freeway_v70 import decode_freeway_ram
            for o in decode_freeway_ram(env.ram()):             # evaluator only
                if not o.kind.startswith("car_"):
                    continue
                x, y, w, h = o.box
                cx, cy = x + w / 2, y + h / 2
                total += 1
                seen += int(any(abs(v["cx"] - cx) <= 4 and abs(v["cy"] - cy) <= 4 for v in vis))
        if game == "assault":                    # every step counts; a lost me is a bad step
            mesteps += 1
            k = None if (me is None or tr.miss > 0) else mark_me(vis, me)
            if k is not None and len(areas) < 200:
                areas.append(tracker.tracks[k]["area"])
            ref = float(np.median(areas)) if len(areas) >= 20 else None   # my size as the parser sees it
            if k is None or (ref is not None and tracker.tracks[k]["area"] > 2 * ref):
                bad += 1
        if rng.random() > 0.95:
            a = int(rng.integers(env.n_actions))
        obs, over = env.step(a)
        if over:
            obs = env.reset(); tracker.reset()
            tr.pos, tr.miss = None, 0
    res = dict(game=game, seed=SEED, steps=steps, objects_mean=round(float(np.mean(nobj)), 1),
               objects_max=int(np.max(nobj)), secs=round(time.perf_counter() - t0))
    if game == "freeway":
        res.update(cars=total, seen=seen, frac=round(seen / max(1, total), 3), passed=seen / max(1, total) >= 0.9,
                   note="single-frame RAM decoder vs 2-frame max-pooled screens; matches may be many-to-one")
    if game == "assault":
        res.update(me_steps=mesteps, bad=bad, bad_frac=round(bad / max(1, mesteps), 3),
                   passed=bad / max(1, mesteps) <= 0.3,
                   judge="self-referential (the v71 tracker's me); there is no Assault RAM judge")
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"craft_v6_out/v80/g0_{game}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    run(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
