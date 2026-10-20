"""Milestone A, step 3 (exploratory): does the novelty drive alone reach the score?

Equal emulated-frame budgets: novelty explorer (savestate returns charged) vs random play.
Experimenter-side measure: the best true score reached in any trajectory (from the archive's
savestates) and the number of distinct cells. No reward is given to the explorer.

Usage: .venv-arc4/Scripts/python.exe -m scripts.slice_explore_v71
"""
import argparse
import json
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.explore import explore, explore_me, cell_of
from ragnarok.agent_v71.perceiver import Perceiver


def best_score(env, arch):
    best = 0.0
    for c in arch.cells.values():
        best = max(best, c["state"][3])         # experimenter-side true score stored in the savestate
    return best


def random_play(game, seed, frame_budget):
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    best, cells = 0.0, set()
    a = 0
    while env.frames < frame_budget:
        if rng.random() > 0.95:
            a = int(rng.integers(env.n_actions))
        obs, over = env.step(a)
        cells.add(cell_of(obs))
        best = max(best, env.true_score())
        if over:
            env.reset()
    return best, len(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["freeway", "breakout", "space_invaders"])
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = []
    for g in a.games:
        env = AtariEnv(g, seed=a.seed)
        rng = np.random.default_rng(a.seed)
        t0 = time.perf_counter()
        obs = env.reset()
        perc = Perceiver(obs.shape[:2])
        perc.warm(env, rng)
        roles = perc.fit_counters(env, rng)
        print(f"   counters: {roles}", flush=True)
        arch = explore_me(env, perc, a.iters, rng, refit_every=0,
                          log=lambda it, n, r: print(f"   it {it}: {n} cells, {r} counter regions, best {best_score(env, None) if False else ''}", flush=True))
        bs = best_score(env, arch)
        top = max(arch.cells.values(), key=lambda c: (c["events"], -c["depth"]))
        json.dump(dict(game=g, seed=a.seed, events=top["events"], depth=top["depth"], true_score=top["state"][3],
                       traj=list(top.get("traj", b"")), counters=perc.roles),
                  open(f"craft_v6_out/v71_demo_{g}.json", "w"))
        frames = env.frames
        rb, rc = random_play(g, a.seed, frames)
        r = dict(game=g, frames=frames, explorer_best=bs, explorer_cells=len(arch.cells),
                 random_best=rb, random_cells=rc, secs=round(time.perf_counter() - t0, 1))
        res.append(r)
        print(f"{g}: {frames} frames | explorer best score {bs} cells {len(arch.cells)} | "
              f"random best {rb} cells {rc} | {r['secs']}s", flush=True)
    json.dump(res, open("craft_v6_out/v71_slice_explore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
