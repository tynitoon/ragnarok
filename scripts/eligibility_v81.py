"""v81 gate E, EVALUATOR side (docs/ARC4_V81_ATLAS.md §4), train seed 8101.

For one candidate game:
  1. the agent's entry (v73/v74 perception: me, buttons, counters, death signal) on its own emulator;
  2. its labels against RAM over 8000 steps of random-with-repeat play: score events (a true-score rise
     within +-4 steps) and deaths (a lives() drop or game over within [-30, +60] steps);
  3. references, practice conditions (sticky 0, 10 episodes, 0-30 random NOOPs, 7000 steps max): every
     constant button, uniform random, random with repeat 0.95;
  4. the existing v71 explorer with 2 M billed frames (entry included): best practice score = true score
     of its best trajectory, replayed from its root.
Eligible if score precision and recall >= 0.9, death precision >= 0.9, and best practice score >=
2 x max(best constant, random-repeat). RAM and the true score are read here only, never by the agent.
Usage: C:/Python314/python.exe -m scripts.eligibility_v81 GAME
"""
import json
import os
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.learner import Agent
from ragnarok.evaluation.provenance_v81 import stamp

SEED = 8101
OUT = "craft_v6_out/v81"
BUDGET = 2_000_000


def pr(pred, truth, before, after):
    """precision of predicted steps (a truth within [-before, +after]) and recall of truth steps."""
    hit = [p for p in pred if any(-before <= t - p <= after for t in truth)]
    rec = [t for t in truth if any(-before <= t - p <= after for p in pred)]
    return (len(hit) / len(pred) if pred else None), (len(rec) / len(truth) if truth else None)


def labels(ag, env, rng, steps=8000):
    obs = env.reset()
    tr = ag.perc.tracker
    tr.pos, tr.miss, ag.gone = None, 0, 0
    from ragnarok.agent_v71.hud import HudReader
    hud = HudReader(ag.perc.counters or []); hud.step(obs)
    sc_a, de_a, sc_t, de_t = [], [], [], []
    lives0, score0, a, dead_latch = env.lives(), env.true_score(), 0, False
    for t in range(steps):
        me = ag._track(obs)
        if rng.random() > 0.95:
            a = int(rng.integers(env.n_actions))
        obs, over = env.step(a)
        ev = hud.step(obs)
        if ag.s_idx is not None and ag.s_idx in ev:
            sc_a.append(t)
        d = ag._dead(ev)
        if d and not dead_latch:
            de_a.append(t)
        dead_latch = d or (dead_latch and (me is None or tr.miss > 0))
        lv, s = env.lives(), env.true_score()
        if s > score0:
            sc_t.append(t)
        if lv < lives0 or over:
            de_t.append(t)
        lives0, score0 = lv, s
        if over:
            obs = env.reset(); lives0, score0 = env.lives(), env.true_score()
            tr.pos, tr.miss, ag.gone = None, 0, 0
            hud = HudReader(ag.perc.counters or []); hud.step(obs)
    sp, sr = pr(sc_a, sc_t, 4, 4)
    dp, dr = pr(de_a, de_t, 30, 60)
    return dict(score_prec=sp, score_rec=sr, score_events=len(sc_a), true_score_events=len(sc_t),
                death_prec=dp, death_rec=dr, death_labels=len(de_a), true_deaths=len(de_t))


def episode(game, policy, rng, max_steps=7000):
    env = AtariEnv(game, seed=SEED)
    env.reset()
    for _ in range(int(rng.integers(0, 31))):
        env.step(0)
    st = {"a": 0}
    for _ in range(max_steps):
        _, over = env.step(policy(rng, st, env))
        if over:
            break
    return env.true_score()


def references(game, rng, n=10):
    env = AtariEnv(game, seed=SEED)
    names = [x.name for x in env.actions]
    out = {"constant": {}}
    for b in range(env.n_actions):
        out["constant"][names[b]] = float(np.mean([episode(game, lambda r, s, e, b=b: b, rng) for _ in range(n)]))
    out["uniform"] = float(np.mean([episode(game, lambda r, s, e: int(r.integers(e.n_actions)), rng) for _ in range(n)]))

    def rep(r, s, e):
        if r.random() > 0.95:
            s["a"] = int(r.integers(e.n_actions))
        return s["a"]
    out["random_repeat"] = float(np.mean([episode(game, rep, rng) for _ in range(n)]))
    best = max(out["constant"], key=out["constant"].get)
    out["best_constant"] = dict(button=best, mean=out["constant"][best])
    out["bar"] = max(out["best_constant"]["mean"], out["random_repeat"])
    return out


FRAMES_PER_ITER = 560     # v72-v74 preps: 1500 explorer iterations cost ~0.84 M frames


def practice(ag, env, rng):
    """ONE run of the v71 explorer sized to the remaining budget (its archive must not restart);
    best trajectory's true score, replayed from its root on the evaluator side."""
    used = env.frames + ag.frames_adapter
    iters = max(100, int((BUDGET - used) / FRAMES_PER_ITER))
    demo, ev, root = ag.explore(env, rng, iters)
    billed = int(env.frames + ag.frames_adapter)
    env.restore(root)
    for a in demo:
        _, over = env.step(a)
        if over:
            break
    return dict(best_practice_score=float(env.true_score()), best_len=len(demo), agent_events=int(ev),
                iterations=iters, billed_frames=billed)


def main(game):
    os.makedirs(OUT, exist_ok=True)
    prov = stamp()
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    env = AtariEnv(game, seed=SEED)
    ag = Agent(vanish_k=2)
    ag.enter(env, rng, seed=SEED)
    res = dict(game=game, seed=SEED, provenance=prov,
               entry=dict(lives_counter=ag.l_idx is not None, vanish_k=ag.vanish_k, lost_frac=round(ag.lost_frac, 3),
                          slots=ag.adapter.mapping()))
    res["labels"] = labels(ag, AtariEnv(game, seed=SEED + 1000), np.random.default_rng(SEED + 1))
    res["references"] = references(game, np.random.default_rng(SEED + 2))
    res["practice"] = practice(ag, env, rng)
    L, R, P = res["labels"], res["references"], res["practice"]
    res["E"] = dict(score_ok=bool((L["score_prec"] or 0) >= 0.9 and (L["score_rec"] or 0) >= 0.9),
                    death_ok=bool((L["death_prec"] or 0) >= 0.9),
                    room_ok=bool(P["best_practice_score"] >= 2 * R["bar"]),
                    ratio=round(P["best_practice_score"] / max(1e-9, R["bar"]), 2))
    res["E"]["eligible"] = bool(res["E"]["score_ok"] and res["E"]["death_ok"] and res["E"]["room_ok"])
    res["secs"] = round(time.perf_counter() - t0)
    json.dump(res, open(f"{OUT}/eligibility_{game}.json", "w"), indent=1)
    print(json.dumps(dict(game=game, E=res["E"], labels=L, bar=R["bar"], best=R["best_constant"],
                          practice=P["best_practice_score"], secs=res["secs"])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
