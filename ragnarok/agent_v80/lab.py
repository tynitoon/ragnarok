"""The laboratory (docs/ARC4_V80_DESIGN.md §3): savestate branches recorded object by object.

From a root (a savestate on a trajectory of the agent's base policy) every available slot is held
`hold` steps, then the base slot for `horizon` steps. Each branch is recorded step by step: the
tracked objects, my position, the slot pressed, and what followed (my score event, my death
signal, and ALE's game-over flag `over`). `over` without a death signal is CENSORING, never a
label (design §3). Sibling branches of one root differ only by my action: their differences near me
or my shots and soon after the divergence are interventions (causal cone), the rest observations.
Agent side only: screen bytes, the agent's HUD events and its death signal; no RAM, no true score.
"""
import copy

import numpy as np

from ragnarok.agent_v80.objects import Parser, Tracker

# one row per visible or coasting track: id cx cy vx vy area colour x0 y0 x1 y1 missed
COLS = ("id", "cx", "cy", "vx", "vy", "area", "colour", "x0", "y0", "x1", "y1", "missed")


def rows(tracker):
    if not tracker.tracks:
        return np.zeros((0, len(COLS)), np.float64)
    return np.array([[t["id"], t["cx"], t["cy"], t["vx"], t["vy"], t["area"], t["colour"], *t["box"], t["missed"]]
                     for t in tracker.tracks.values()], np.float64)


def make_parser(ag, gauges=()):
    return Parser(ag.perc.bg, boxes=list(ag.perc.counters or []) + list(gauges))


def roots_from(ag, env, states, base, rng, parser, n_roll=4, length=300, every=6, mix=0.15, seen=None, rid0=0):
    """States my base policy visits, with 15 % random slots for diversity, deduplicated by screen
    (the emulator is deterministic; pass one `seen` set across calls). Each root is
    (savestate, my tracker position, previous screen, screen, HUD reader, vanish count, object
    tracker, rollout id); the object tracker is snapshotted BEFORE it sees the root screen, so the
    branch's first step gives every object its real velocity (review v80, item 1)."""
    perc, out = ag.perc, []
    seen = set() if seen is None else seen
    choices = np.nonzero(ag.mask)[0]
    for i in range(n_roll):
        rid = rid0 + i
        st, mpos, _, obs, hud0, gone0 = states[int(rng.integers(len(states)))][:6]
        env.restore(st)
        perc.tracker.pos, perc.tracker.miss = mpos, 0
        hud, ag.gone = copy.deepcopy(hud0), gone0
        trk, prev_obs = Tracker(), None
        for k in range(length):
            if k % every == 0 and k >= 4:               # a few steps of tracking first: velocities
                key = hash(obs.tobytes())
                if key not in seen:
                    seen.add(key)
                    out.append((env.clone(), perc.tracker.pos, prev_obs, obs, copy.deepcopy(hud), ag.gone,
                                copy.deepcopy(trk), rid))
            trk.step(parser.parse(obs))
            ag._track(obs)
            slot = base if rng.random() > mix else int(rng.choice(choices))
            prev_obs = obs
            obs, over = env.step(ag.adapter.to_button(slot))
            ev = hud.step(obs)
            if over or ag._dead(ev):
                break
    return out


def _me(ag, me):
    """My position on this screen, or NaN when my tracker found me nowhere (no stale position)."""
    if me is None or ag.perc.tracker.miss > 0:
        return (np.nan, np.nan)
    return (float(me[0]), float(me[1]))


def trace(ag, env, root, slot, base, parser, hold=8, horizon=56):
    """One branch. Returns dict: objs[t] (rows at step t, before acting), me[t], slot[t], score[t],
    death[t] (outcomes of step t), over (bool), end (steps run). Frames are charged by env."""
    st, mpos, _, obs, hud0, gone0, trk0 = root[:7]
    env.restore(st)
    ag.perc.tracker.pos, ag.perc.tracker.miss = mpos, 0
    hud, ag.gone = copy.deepcopy(hud0), gone0
    trk = copy.deepcopy(trk0)
    objs, me_l, sl, sc, de = [], [], [], [], []
    over = False
    for k in range(hold + horizon):
        me = ag._track(obs)
        trk.step(parser.parse(obs))
        objs.append(rows(trk))
        me_l.append(_me(ag, me))
        s = slot if k < hold else base
        sl.append(s)
        obs, over = env.step(ag.adapter.to_button(s))
        ev = hud.step(obs)
        sc.append(bool(ag.s_idx is not None and ag.s_idx in ev))
        dead = ag._dead(ev)
        de.append(bool(dead))
        if dead or over:
            break
    trk.step(parser.parse(obs))
    objs.append(rows(trk))                           # the frame after the last action
    me = ag._track(obs)
    me_l.append(_me(ag, me))
    return dict(objs=objs, me=np.array(me_l, np.float32), slot=np.array(sl, np.int16),
                score=np.array(sc, bool), death=np.array(de, bool), over=bool(over and not any(de)),
                end=len(sl))


def siblings(ag, env, root, base, parser, hold=8, horizon=56):
    """Every available slot from one root: the unit of intervention."""
    return {int(c): trace(ag, env, root, int(c), base, parser, hold, horizon) for c in np.nonzero(ag.mask)[0]}


def branch_value(tr, dp=2.0):
    """My own events minus dp if my own death signal fired; `over` alone counts nothing."""
    return float(tr["score"].sum()) - dp * float(tr["death"].any())


def best_constant(ag, env, states, rng, parser, hold=8, horizon=56, n=16, dp=2.0):
    """The agent's own best constant slot, valued without the `over` penalty (design §3)."""
    sel = [states[int(i)] for i in rng.choice(len(states), size=min(n, len(states)), replace=False)]
    vals = {}
    for c in np.nonzero(ag.mask)[0]:
        v = []
        for s in sel:
            st, mpos, prev, obs, hud0, gone0 = s[:6]
            v.append(branch_value(trace(ag, env, (st, mpos, None, obs, hud0, gone0, Tracker()), int(c), int(c),
                                        parser, hold, horizon), dp))
        vals[int(c)] = float(np.mean(v))
    best = max(vals.values())
    return min(c for c in vals if vals[c] == best), vals
