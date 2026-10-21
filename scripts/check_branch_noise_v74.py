"""v74 diagnostic: is the gain measured by single deterministic branches real, or selection noise?

From roots visited by the agent's own best-constant policy, every slot is held `hold` steps then
the base continues `horizon` steps. Value = the agent's own HUD score events - 2 x its own death
signal (screen only). Two evaluations of the same (root, slot):
  det   : one branch in the deterministic lab emulator (what v72-v74 rules rank on);
  stoch : K branches with sticky actions 0.25 (as in live play; ALE's RNG is not part of the
          savestate, so each branch draws its own stickiness).
Reported: the deterministic 'gain' of the argmax over the base slot, the stochastic gain of that
SAME chosen slot (an unbiased estimate of what the choice is worth), and the stochastic argmax.

Usage: C:/Python314/python.exe -m scripts.check_branch_noise_v74 CACHE.pkl [roots] [K]
"""
import json
import pickle
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.adapter import SLOTS


def main(path, n_roots=30, K=8, hold=8, horizon=96, dp=2.0):
    t0 = time.perf_counter()
    C = pickle.load(open(path, "rb"))
    ag, demo, root = C["agent"], C["demo"], C["root"]
    game, seed = C["rec"]["game"], C["rec"]["seed"]
    det = AtariEnv(game, seed=seed)
    sto = AtariEnv(game, seed=seed + 500, sticky=0.25)
    rng = np.random.default_rng(7)
    D, states = ag.replay_demo(det, demo, root)
    choices = [i for i in range(len(SLOTS)) if ag.mask[i]]
    # the agent's own best constant slot (deterministic, as in _learn_rpi)
    sel = [states[int(i)] for i in rng.choice(len(states), size=min(16, len(states)), replace=False)]
    cv = {}
    for c in choices:
        ag.base_slot = c
        cv[c] = float(np.mean([ag._q(det, r, c, hold, horizon, rng, dp)[0] for r in sel]))
    base = max(choices, key=lambda c: (cv[c], -c))
    ag.base_slot = base
    roots = ag._base_roots(det, rng, states, n_roll=6)
    roots = [roots[int(i)] for i in rng.choice(len(roots), size=min(n_roots, len(roots)), replace=False)]
    rows = []
    for r in roots:
        qd = {c: ag._q(det, r, c, hold, horizon, rng, dp)[0] for c in choices}
        qs = {c: [ag._q(sto, r, c, hold, horizon, rng, dp)[0] for _ in range(K)] for c in choices}
        ms = {c: float(np.mean(v)) for c, v in qs.items()}
        top = max(qd.values())
        cd = base if qd[base] == top else min(c for c in choices if qd[c] == top)
        cs = max(choices, key=lambda c: ms[c])
        rows.append(dict(det_gain=top - qd[base], det_choice_true_gain=ms[cd] - ms[base],
                         sto_gain=ms[cs] - ms[base], det_moved=cd != base, agree=cd == cs,
                         sd_within=float(np.mean([np.std(v, ddof=1) for v in qs.values()]))))
    f = lambda k: round(float(np.mean([x[k] for x in rows])), 3)
    moved = [x for x in rows if x["det_moved"]]
    res = dict(game=game, seed=seed, base=SLOTS[base], const_values={SLOTS[c]: round(v, 2) for c, v in cv.items()},
               roots=len(rows), K=K, det_gain=f("det_gain"), det_choice_true_gain=f("det_choice_true_gain"),
               det_choice_true_gain_when_moved=round(float(np.mean([x["det_choice_true_gain"] for x in moved])), 3) if moved else None,
               det_moved_frac=round(len(moved) / len(rows), 3), sto_gain_K=f("sto_gain"), agree=f("agree"),
               sd_within=f("sd_within"), secs=round(time.perf_counter() - t0))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:3]))
