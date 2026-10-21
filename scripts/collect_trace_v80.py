"""v80 lab traces (docs/ARC4_V80_DESIGN.md §3-4).

train   : roots on the agent's own base trajectories (15 % random slots), every slot held 8 steps
          then the base 56 steps, recorded object by object (agent side only). Also chooses and
          stores the agent's own best constant slot (the base), valued without the `over` penalty.
heldout : EVALUATOR-owned test roots from fresh starts (rng stream 50), screen twins of the
          training roots removed, same recording; plus, in a SEPARATE file *_truth.pkl, the
          evaluator's truth per branch step (RAM lives decrements, true-score increments), used only
          to check the agent's own labels. The agent-side loader refuses *_truth files.
Usage: C:/Python314/python.exe -m scripts.collect_trace_v80 train CACHE.pkl OUT.pkl [n_roots]
       C:/Python314/python.exe -m scripts.collect_trace_v80 heldout CACHE.pkl OUT.pkl TRAIN.pkl [n_roots]
"""
import hashlib
import os
import pickle
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v80 import lab

HOLD, HORIZON = 8, 56
EXTRA = 60          # evaluator truth replays this far past a branch's end: RAM may lag the screen
DEV_SEEDS = (98, 99)


def screen_id(obs):
    """Stable across processes (Python's hash() of bytes is salted per process)."""
    return hashlib.md5(obs.tobytes()).hexdigest()


def load_agent_file(p):
    if "_truth" in p:
        raise ValueError("agent-side code never loads evaluator truth files")
    return pickle.load(open(p, "rb"))


def _setup(cache):
    C = pickle.load(open(cache, "rb"))
    ag, demo, root = C["agent"], C["demo"], C["root"]
    for k, v in dict(base_slot=None, live_temp=1.0, vanish_k=2, gone=0, commit=1, _held=None).items():
        if not hasattr(ag, k):
            setattr(ag, k, v)
    game, seed = C["rec"]["game"], C["rec"]["seed"]
    if seed not in DEV_SEEDS and os.environ.get("V80_PROTOCOL") != "1":
        raise SystemExit(f"seed {seed}: development uses seeds 98/99 only (design §4)")
    env = AtariEnv(game, seed=seed)
    _, states = ag.replay_demo(env, demo, root)
    return ag, env, states, game, seed, root


def collect(ag, env, roots, base, parser, t0, cap):
    sibsets, R = [], []
    for r in roots[:cap]:
        sibsets.append(lab.siblings(ag, env, r, base, parser, HOLD, HORIZON))
        R.append(dict(obs=r[3], pme=r[1], prev_obs=r[2], rid=r[7]))   # pme: my position one step before
        if len(sibsets) % 25 == 0:
            print(len(sibsets), f"{time.perf_counter() - t0:.0f}s", flush=True)
    return sibsets, R


def truth_for(ag, env, roots, sibsets, base):
    """EVALUATOR only: replay each branch from its savestate (deterministic lab, sticky 0), then
    EXTRA base steps, reading RAM lives, the true score and game over at every step."""
    out = []
    for r, sib in zip(roots, sibsets):
        tr_t = {}
        for s, tr in sib.items():
            env.restore(r[0])
            lives0, score0, dl, ds, go = env.lives(), env.true_score(), [], [], []
            for a in list(tr["slot"]) + [base] * EXTRA:
                _, over = env.step(ag.adapter.to_button(int(a)))
                lv, sc = env.lives(), env.true_score()
                dl.append(lv < lives0); ds.append(sc - score0); go.append(over)
                lives0, score0 = lv, sc
                if over:
                    break
            tr_t[s] = dict(lives_lost=np.array(dl, bool), score_gain=np.array(ds, np.float32), game_over=np.array(go, bool))
        out.append(tr_t)
    return out


def train(cache, out, n_roots=400):
    t0 = time.perf_counter()
    ag, env, states, game, seed, root = _setup(cache)
    rng = np.random.default_rng(seed)
    parser = lab.make_parser(ag)
    f0 = env.frames
    base, vals = lab.best_constant(ag, env, states, rng, parser, HOLD, HORIZON)
    print(game, seed, "base", base, {int(k): round(v, 2) for k, v in vals.items()}, flush=True)
    roots, seen = [], set()
    while len(roots) < n_roots:
        roots += lab.roots_from(ag, env, states, base, rng, parser, n_roll=4, seen=seen, rid0=len(roots) * 10 + 1000)
        print("roots", len(roots), flush=True)
    rng.shuffle(roots)
    sibsets, R = collect(ag, env, roots, base, parser, t0, n_roots)
    # evaluator truth for a fixed sample of TRAIN branches too (the labels the theory learns from)
    sample = sorted(np.random.default_rng(7).choice(len(sibsets), min(60, len(sibsets)), replace=False).tolist())
    pickle.dump(dict(game=game, idx=sample, truth=truth_for(ag, env, [roots[i] for i in sample], [sibsets[i] for i in sample], base)),
                open(out.replace(".pkl", "_truth.pkl"), "wb"))
    pickle.dump(dict(game=game, seed=seed, base=int(base), base_vals=vals, mask=ag.mask, bg=ag.perc.bg, sibsets=sibsets, roots=R,
                     screens=[screen_id(r["obs"]) for r in R], frames=env.frames - f0,
                     hold=HOLD, horizon=HORIZON), open(out, "wb"))
    print("saved", out, len(sibsets), "roots", env.frames - f0, "frames", f"{time.perf_counter() - t0:.0f}s", flush=True)


def heldout(cache, out, train_file, n_roots=300):
    t0 = time.perf_counter()
    ag, env, _, game, seed, root = _setup(cache)
    T = load_agent_file(train_file)
    base = T["base"]
    banned = set(T["screens"])
    rng = np.random.default_rng(50)
    parser = lab.make_parser(ag)
    env.restore(root)                                # fresh starts: the reset state only
    obs = env.obs()
    ag.perc.tracker.pos, ag.perc.tracker.miss = None, 0
    ag._track(obs)
    from ragnarok.agent_v71.hud import HudReader
    hud = HudReader(ag.perc.counters or []); hud.step(obs)
    start = [(env.clone(), ag.perc.tracker.pos, None, obs, hud, 0)]
    roots, seen = [], set()
    while len(roots) < n_roots:
        new = lab.roots_from(ag, env, start, base, rng, parser, n_roll=4, length=600, seen=seen, rid0=len(roots) * 10 + 5000)
        roots += [r for r in new if screen_id(r[3]) not in banned]
        print("roots", len(roots), flush=True)
    rng.shuffle(roots)
    sibsets, R = collect(ag, env, roots, base, parser, t0, n_roots)
    truth = truth_for(ag, env, roots[:n_roots], sibsets, base)
    pickle.dump(dict(game=game, seed=seed, base=int(base), mask=ag.mask, bg=ag.perc.bg, sibsets=sibsets, roots=R, hold=HOLD,
                     horizon=HORIZON), open(out, "wb"))
    pickle.dump(dict(game=game, idx=list(range(len(sibsets))), truth=truth), open(out.replace(".pkl", "_truth.pkl"), "wb"))
    print("saved", out, len(sibsets), f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "train":
        train(a[1], a[2], *(int(x) for x in a[3:4]))
    else:
        heldout(a[1], a[2], a[3], *(int(x) for x in a[4:5]))
