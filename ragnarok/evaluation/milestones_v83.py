"""v83 milestones and climbs (docs/ARC4_V83_PROTOCOL.md §3), evaluator side.

Ladder M_k = B + 2^k u, k = 0..10. T(M) = billed frames (entry included) at the explorer epoch's first
clone with true cumulative score >= M (records from senses_v83.Trace), confirmed by replaying its
actions from the epoch's reset base (a mismatch is an Interruption). Climb C = T(M*) - T(M1) from the
arm's own M1; late climb from M2. Censoring always against the carried arm; floor 5,000 frames.
"""
import pickle
import zlib

import numpy as np

from ragnarok.agent_v71.env import AtariEnv

K = 10
FLOOR = 5000


def ladder(B, u):
    return [B + (2 ** k) * u for k in range(K + 1)]


def first_T(records, m):
    """records: [(billed, true), ...] in billed order (new bests only)."""
    for b, t in records:
        if t >= m:
            return b
    return None


def first_index(records, m):
    for i, (b, t) in enumerate(records):
        if t >= m:
            return i
    return None


def replay(game, seed, base_state, actions):
    """True score after `actions` from the pickled+zlib base state, on a fresh emulator (unbilled) that
    first plays one NOOP step (senses_v83.Trace.replay; build notes 21)."""
    env = AtariEnv(game, seed=seed)
    env.step(0)
    env.ale.restoreState(pickle.loads(zlib.decompress(bytes(base_state))))
    env._true_score = 0.0
    for a in actions:
        env.step(int(a))
    return env.true_score()


def confirm(game, seed, npz, records, idx):
    """Replay record `idx` of a room_job npz; returns (replayed, recorded, ok)."""
    off = npz["rec_offsets"]
    acts = npz["rec_actions"][off[idx]:off[idx + 1]]
    got = replay(game, seed, npz["base_state"], acts)
    return got, records[idx][1], bool(got == records[idx][1])


def top_rung(records, lad):
    """Highest k with M_k reached (-1 if none)."""
    best = max((t for _, t in records), default=-np.inf)
    ks = [k for k, m in enumerate(lad) if best >= m]
    return max(ks) if ks else -1


def climb(records, lad, k_star, cap, carried, k_from=1):
    """C = T(M*) - T(M_k_from), censored against the carried arm, floored at FLOOR."""
    t1, ts = first_T(records, lad[k_from]), first_T(records, lad[k_star])
    if ts is not None and t1 is not None:
        c = ts - t1
    elif carried:
        c = cap
    elif t1 is not None:
        c = cap - t1
    else:
        c = FLOOR
    return max(FLOOR, c)
