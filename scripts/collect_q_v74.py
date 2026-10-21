"""v74 offline dataset: roots with the measured value of EVERY slot (agent-side values only).

Roots are states visited by the agent's own best-constant policy, plus states visited by a noisy
version of it (each step, with prob. `mix`, a random slot) so the data cover deviations. Value of
(root, slot) = hold the slot `hold` steps, then the constant base `horizon` steps; the agent's HUD
score events minus 2 x its own death signal. Saved: the root screen, the previous screen, my tracked
position, the value vector, the slot mask, the base slot, the agent's background and counter regions.
Used offline to compare how well different learners turn these values into a policy (regret).

Usage: C:/Python314/python.exe -m scripts.collect_q_v74 CACHE.pkl OUT.npz [n_roots] [rng_seed]
"""
import pickle
import sys
import time

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v71.adapter import SLOTS


def main(cache, out, n_roots=1200, rs=0, hold=8, horizon=96, dp=2.0, mix=0.15, every=8, length=400):
    t0 = time.perf_counter()
    C = pickle.load(open(cache, "rb"))
    ag, demo, root = C["agent"], C["demo"], C["root"]
    for k, v in dict(base_slot=None, live_temp=1.0, vanish_k=2, gone=0).items():
        if not hasattr(ag, k):                     # agents cached by v72 predate these attributes
            setattr(ag, k, v)
    game, seed = C["rec"]["game"], C["rec"]["seed"]
    env = AtariEnv(game, seed=seed)
    rng = np.random.default_rng(rs)
    _, states = ag.replay_demo(env, demo, root)
    choices = [i for i in range(len(SLOTS)) if ag.mask[i]]
    sel = [states[int(i)] for i in rng.choice(len(states), size=min(16, len(states)), replace=False)]
    cv = {}
    for c in choices:
        ag.base_slot = c
        cv[c] = float(np.mean([ag._q(env, r, c, hold, horizon, rng, dp)[0] for r in sel]))
    base = max(choices, key=lambda c: (cv[c], -c))
    ag.base_slot = base
    print(game, seed, "base", SLOTS[base], {SLOTS[c]: round(v, 2) for c, v in cv.items()}, flush=True)
    import copy
    from ragnarok.agent_v71.hud import HudReader  # noqa: F401  (reader objects travel inside roots)
    O, P, ME, PME, Q, T, RID, SID = [], [], [], [], [], [], [], []
    rid = -1
    perc = ag.perc
    while len(Q) < n_roots:
        rid += 1
        si = int(rng.integers(len(states)))
        st, mpos, prev, obs, hud0, gone0 = states[si]
        env.restore(st)
        perc.tracker.pos, perc.tracker.miss = mpos, 0
        hud, ag.gone = copy.deepcopy(hud0), gone0
        prev_obs = obs
        for k in range(length):
            if k % every == every - 1:
                r = (env.clone(), perc.tracker.pos, prev, obs, copy.deepcopy(hud), ag.gone)
                q = np.full(len(SLOTS), np.nan, np.float32)
                for c in choices:
                    q[c] = ag._q(env, r, c, hold, horizon, rng, dp)[0]
                env.restore(r[0])
                perc.tracker.pos, perc.tracker.miss = r[1], 0
                hud, ag.gone = copy.deepcopy(r[4]), r[5]
                me_now = ag._track(obs)                  # my position on THIS screen
                perc.tracker.pos, perc.tracker.miss = r[1], 0
                ag.gone = r[5]
                O.append(obs); P.append(prev_obs); ME.append(np.array(me_now if me_now else (-1, -1), np.float32))
                PME.append(np.array(r[1] if r[1] else (-1, -1), np.float32))
                Q.append(q); T.append(k); RID.append(rid); SID.append(si)
                if len(Q) % 100 == 0:
                    print(len(Q), f"{time.perf_counter() - t0:.0f}s", flush=True)
                if len(Q) >= n_roots:
                    break
            me = ag._track(obs)
            from ragnarok.agent_v71.policy import ego_view
            _, _, prev = ego_view(obs, perc.bg, me, prev)
            slot = base if rng.random() > mix else int(rng.choice(choices))
            prev_obs = obs
            obs, over = env.step(ag.adapter.to_button(slot))
            ev = hud.step(obs)
            if over or ag._dead(ev):
                break
    np.savez_compressed(out, obs=np.array(O, np.uint8), prev=np.array(P, np.uint8), me=np.array(ME), q=np.array(Q),
                        t=np.array(T), rid=np.array(RID), sid=np.array(SID), pme=np.array(PME), mask=ag.mask, base=base, bg=perc.bg,
                        counters=np.array(perc.counters or [], np.int32).reshape(-1, 4), game=game, seed=seed)
    print("saved", out, len(Q), f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
