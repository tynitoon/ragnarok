"""Which Atari games does the v71 perception chain support? (exploratory survey)

Per game, with random play with action repeats: (1) me found by probes (fraction of probes) and
the effect table; (2) counter reader: score events vs true score increments (ALE reward,
experimenter-side), lives events vs ALE lives. Reserved held-out games are never opened.

Usage: C:/Python314/python.exe -m scripts.slice_survey_v71 [--workers 10]
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

GAMES = ["alien", "amidar", "assault", "asterix", "atlantis", "bank_heist", "beam_rider", "berzerk",
         "bowling", "breakout", "carnival", "centipede", "chopper_command", "crazy_climber", "demon_attack",
         "fishing_derby", "freeway", "gopher", "hero", "ice_hockey", "jamesbond", "krull", "kung_fu_master",
         "ms_pacman", "name_this_game", "phoenix", "pong", "pooyan", "qbert", "road_runner", "seaquest",
         "space_invaders", "star_gunner", "tennis", "time_pilot", "up_n_down", "video_pinball",
         "wizard_of_wor", "yars_revenge", "zaxxon"]


def match(pred, true, tol=3):
    hit_t = sum(any(abs(p - t) <= tol for p in pred) for t in true)
    hit_p = sum(any(abs(p - t) <= tol for t in true) for p in pred)
    return (hit_p / len(pred) if pred else None), (hit_t / len(true) if true else None)


def survey(game, seed=0, steps=20000, repeat=0.95):
    try:
        return _survey(game, seed, steps, repeat)
    except Exception as e:  # noqa: BLE001
        return dict(game=game, error=f"{type(e).__name__}: {e}"[:200])


def _survey(game, seed, steps, repeat):
    from ragnarok.agent_v71.env import AtariEnv
    from ragnarok.agent_v71.perceiver import Perceiver
    from ragnarok.agent_v71.me_buttons import effect_table
    from ragnarok.agent_v71.hud import HudReader
    t0 = time.perf_counter()
    try:
        env = AtariEnv(game, seed=seed)
    except Exception as e:  # noqa: BLE001
        return dict(game=game, error=str(e)[:200])
    rng = np.random.default_rng(seed)
    obs = env.reset()
    perc = Perceiver(obs.shape[:2])
    perc.warm(env, rng)
    roles = perc.fit_counters(env, rng, steps=steps, repeat=repeat)
    # me probes along ordinary play
    obs = env.reset()
    found, a = 0, 0
    for i in range(60):
        for _ in range(8):
            if rng.random() > repeat:
                a = int(rng.integers(env.n_actions))
            obs, over = env.step(a)
            if over:
                obs = env.reset()
        perc.tracker.pos = None
        if perc.locate(env, obs) is not None:
            found += 1
    table = effect_table(perc.probes) if perc.probes else []
    # reader accuracy on fresh play
    hud = HudReader(perc.counters or [])
    s_i = perc.counters.index(roles["score"]) if "score" in roles else None
    l_i = perc.counters.index(roles["lives"]) if "lives" in roles else None
    ev_s, ev_l, tr_s, tr_l = [], [], [], []
    obs = env.reset()
    ls, ll = env.true_score(), env.lives()
    for t in range(steps):
        for i in hud.step(obs):
            if i == s_i:
                ev_s.append(t)
            if i == l_i:
                ev_l.append(t)
        if rng.random() > repeat:
            a = int(rng.integers(env.n_actions))
        obs, over = env.step(a)
        if env.true_score() > ls:
            tr_s.append(t + 1)
        if env.lives() < ll:
            tr_l.append(t + 1)
        ls, ll = env.true_score(), env.lives()
        if over:
            obs = env.reset(); hud.reset_tracking(); ls, ll = env.true_score(), env.lives()
    ps, rs = match(ev_s, tr_s)
    pl, rl = match(ev_l, tr_l)
    return dict(game=game, actions=len(env.actions), me_found=found / 60,
                effects=[e["effect"] for e in table], score_region=roles.get("score"), lives_region=roles.get("lives"),
                score_events=len(ev_s), true_score_events=len(tr_s), score_prec=ps, score_rec=rs,
                lives_events=len(ev_l), true_deaths=len(tr_l), lives_prec=pl, lives_rec=rl,
                secs=round(time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--games", nargs="*", default=GAMES)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        res = pool.map(survey, a.games)
    json.dump(res, open("craft_v6_out/v71_survey.json", "w"), indent=1, default=str)
    for r in res:
        if "error" in r:
            print(f"{r['game']:16s} ERROR {r['error']}")
            continue
        f = lambda x: "  -  " if x is None else f"{x:.2f}"   # noqa: E731
        print(f"{r['game']:16s} me {r['me_found']:.2f} | score P/R {f(r['score_prec'])}/{f(r['score_rec'])} "
              f"({r['score_events']}/{r['true_score_events']}) | lives P/R {f(r['lives_prec'])}/{f(r['lives_rec'])} "
              f"({r['lives_events']}/{r['true_deaths']}) | {r['secs']}s")


if __name__ == "__main__":
    main()
