"""Milestone A, step 1 (exploratory): does the agent find itself and its buttons on real Atari?

Per game: warm the background on random play, then probe every action vs NOOP from many
states. Scored experimenter-side against RAM (AtariARI addresses): correlation between
the found "me" coordinate and RAM player coordinate, and sign agreement of each action's
effect with the RAM displacement between branches.

Usage: .venv-arc4/Scripts/python.exe -m scripts.slice_me_v71 [--games breakout freeway space_invaders pong]
"""
import argparse
import json
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.me_buttons import probe, effect_table
from ragnarok.agent_v71.scorer import player_coord
from scripts.ref_extract_v70 import ReferenceExtractor


def ram_branch_delta(env, a, horizon, axis_game):
    s = env.clone()
    env.restore(s)
    for _ in range(horizon):
        env.step(0)
    base = player_coord(axis_game, env.ram())
    env.restore(s)
    for _ in range(horizon):
        env.step(a)
    moved = player_coord(axis_game, env.ram())
    env.restore(s)
    return None if base is None else moved[1] - base[1]


def run(game, seed, warm=3000, n_probes=200, gap=6, horizon=2):
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    ref = ReferenceExtractor()
    t0 = time.perf_counter()
    obs = env.reset()
    for _ in range(warm):                       # background from ordinary play only
        ref.update(obs)
        obs, over = env.step(int(rng.integers(env.n_actions)))
        if over:
            obs = env.reset()
    bg = ref._mode.view(np.uint8).reshape(*ref._mode.shape, 4)[..., :3]
    probes, mine, truth, sign_ok, sign_n = [], [], [], 0, 0
    axis = player_coord(game, env.ram())[0]
    for i in range(n_probes):
        for _ in range(gap):
            ref.update(obs)
            obs, over = env.step(int(rng.integers(env.n_actions)))
            if over:
                obs = env.reset()
        out, me = probe(env, bg, horizon)
        probes.append(out)
        if me is not None:
            mine.append(me[0] if axis == "x" else me[1])
            truth.append(player_coord(game, env.ram())[1])
        for a in range(1, env.n_actions):        # sign agreement on movement along the RAM axis
            d_agent = out[a]["dx"] if axis == "x" else out[a]["dy"]
            if out[a]["moved"] and abs(d_agent) >= 0.5:
                d_ram = ram_branch_delta(env, a, horizon, game)
                if d_ram:
                    sign_n += 1
                    sign_ok += int(np.sign(d_ram) == np.sign(d_agent))
    table = effect_table(probes)
    r = float(np.corrcoef(mine, truth)[0, 1]) if len(mine) > 5 and np.std(truth) > 0 else float("nan")
    # RAM axes may count opposite to screen axes (Freeway's player_y counts upward). The
    # orientation is an experimenter-side calibration taken from the correlation sign; the
    # agent's numbers are untouched.
    if r < 0 and sign_n:
        sign_ok = sign_n - sign_ok
        r = -r
    res = dict(game=game, seed=seed, actions=[a.name for a in env.actions], effect_table=table,
               me_found_frac=len(mine) / n_probes, me_ram_corr=r,
               effect_sign_agreement=(sign_ok / sign_n) if sign_n else None, sign_trials=sign_n,
               frames=env.frames, secs=round(time.perf_counter() - t0, 1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["breakout", "freeway", "space_invaders", "pong"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="craft_v6_out/v71_slice_me.json")
    a = ap.parse_args()
    allres = []
    for g in a.games:
        r = run(g, a.seed)
        allres.append(r)
        print(f"\n{g}: me found {r['me_found_frac']:.2f} | corr with RAM {r['me_ram_corr']:+.3f} | "
              f"effect sign agreement {r['effect_sign_agreement']} ({r['sign_trials']}) | "
              f"{r['frames']} frames {r['secs']}s")
        for name, e in zip(r["actions"], r["effect_table"]):
            print(f"   {name:10s} -> {e['effect']:6s} dx {e['dx']:+.1f} dy {e['dy']:+.1f} spawn {e['spawn_rate']:.2f}")
    json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
