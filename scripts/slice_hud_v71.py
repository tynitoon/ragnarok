"""Milestone A, step 2 (exploratory): does the agent find its score and lives counters on screen?

Fit counter regions on random play, assign roles (score / lives) from their own statistics,
then score the reader's events experimenter-side: score events vs steps where the true
score increased; lives events vs ALE life losses (each within +-3 agent steps).

Usage: .venv-arc4/Scripts/python.exe -m scripts.slice_hud_v71
"""
import argparse
import json
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.hud import CounterFinder, HudReader


def match(pred, true, tol=3):
    pred, true = sorted(pred), sorted(true)
    hit_t = sum(any(abs(p - t) <= tol for p in pred) for t in true)
    hit_p = sum(any(abs(p - t) <= tol for t in true) for p in pred)
    rec = hit_t / len(true) if true else None
    prec = hit_p / len(pred) if pred else None
    return prec, rec


def run(game, seed, fit_steps=20000, test_steps=20000, repeat=0.0):
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    act = [0]

    def pick():
        if rng.random() >= repeat:
            act[0] = int(rng.integers(env.n_actions))
        return act[0]
    t0 = time.perf_counter()
    obs = env.reset()
    cf = CounterFinder(obs.shape[:2])
    for _ in range(fit_steps):
        cf.update(obs)
        obs, over = env.step(pick())
        if over:
            obs = env.reset()
            cf.ring = []
    regions = cf.fit()
    hud = HudReader(regions)
    for _ in range(fit_steps // 2):                # role statistics on more ordinary play
        hud.step(obs)
        obs, over = env.step(pick())
        if over:
            obs = env.reset(); hud.reset_tracking()
    roles = hud.assign_roles()
    ev = {"score": [], "lives": []}
    true_score, true_death = [], []
    last_s, last_l = env.true_score(), env.lives()
    for t in range(test_steps):
        for i in hud.step(obs):
            if i in roles:
                ev[roles[i]].append(t)
        obs, over = env.step(pick())
        s, l = env.true_score(), env.lives()
        if s > last_s:
            true_score.append(t + 1)
        if l < last_l:
            true_death.append(t + 1)
        last_s, last_l = s, l
        if over:
            obs = env.reset(); hud.reset_tracking()
            last_s, last_l = env.true_score(), env.lives()
    ps, rs = match(ev["score"], true_score)
    pl, rl = match(ev["lives"], true_death)
    return dict(game=game, regions=regions, roles={str(k): v for k, v in roles.items()},
                states=[len(s) for s in hud.states], events=hud.events,
                score_events=len(ev["score"]), true_score_events=len(true_score), score_prec=ps, score_rec=rs,
                lives_events=len(ev["lives"]), true_deaths=len(true_death), lives_prec=pl, lives_rec=rl,
                secs=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["breakout", "space_invaders", "freeway", "pong"])
    ap.add_argument("--repeat", type=float, default=0.0)
    ap.add_argument("--out", default="craft_v6_out/v71_slice_hud.json")
    a = ap.parse_args()
    res = []
    for g in a.games:
        r = run(g, 0, repeat=a.repeat)
        res.append(r)
        print(f"\n{g}: {len(r['regions'])} regions {r['regions']} states {r['states']} roles {r['roles']}")
        print(f"   score: {r['score_events']} events vs {r['true_score_events']} true | precision {r['score_prec']} recall {r['score_rec']}")
        print(f"   lives: {r['lives_events']} events vs {r['true_deaths']} true | precision {r['lives_prec']} recall {r['lives_rec']} | {r['secs']}s")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
