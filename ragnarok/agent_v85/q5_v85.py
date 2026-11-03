"""Q5, the macro process injected into V2s as process 4 (docs/ARC4_V85_PROTOCOL_EN.md §4.4, A.6, A.7).

A library is a list of (macro, weight); a macro is a run-length sequence ((slot, steps), ...) of canonical
slots (K*mac and family M: each slot held 4 steps; K*mac-exact: 1 step). Q5 draws macros with probability
0.9 * weight / total + 0.1 / N (count-weighted with a 10% uniform mixture) and chains them to the end of the
rollout. Slots are resolved on the run's own adapter (`ex.slots`): an unavailable slot falls back to its
movement class without extra, then to NOOP (priv slots without a match go to NOOP).

Streams (A.6): `first_slot` draws one macro from P_{e,j} and returns its first resolved slot, with no side
effect. Inside the rollout (in-rollout draws from X_e), the call at t = 1 opens the rollout: the first macro
is drawn from X_e among the macros whose first slot is the slot played at t = 0, with the same weights;
later macros are drawn from X_e. Drawing a macro, then its first slot, has the same law as drawing the first
slot, then the macro given that slot; this split keeps `propose` side-effect free (choice, documented).
With an empty library `len(q5) == 0` and V2s keeps its 4-process bandit (byte-identical, G-build test 6).
The position inside the current macro is per rollout and is not part of `to_state` (checkpoints are taken
at expansion boundaries).
"""
import numpy as np

from ragnarok.agent_v83.adapter import SLOTS

NOOP = SLOTS.index("0,0,0")
UNIFORM = 0.1


def resolve(slot, avail):
    """The playable slot: itself if available, else its movement class without extra, else NOOP."""
    if slot in avail:
        return int(slot)
    name = SLOTS[slot]
    if not name.startswith("priv"):
        base = SLOTS.index(name[:name.rindex(",")] + ",0")
        if base in avail:
            return base
    return NOOP


class Q5:
    def __init__(self, library=()):
        self.set_library(library)

    def set_library(self, library):
        """library: [(((slot, steps), ...), weight), ...]; weights > 0."""
        self.lib = [(tuple((int(s), int(n)) for s, n in m), float(w)) for m, w in library]
        self._cache = None
        self._seq, self._pos = None, 0

    def __len__(self):
        return len(self.lib)

    def _tables(self, ex):
        key = tuple(ex.slots)
        if self._cache is None or self._cache[0] != key:
            avail = set(key)
            seqs = [np.array([resolve(s, avail) for s, n in m for _ in range(n)], np.int64) for m, _ in self.lib]
            w = np.array([w for _, w in self.lib], np.float64)
            p = (1.0 - UNIFORM) * w / w.sum() + UNIFORM / len(w)
            self._cache = (key, seqs, np.array([q[0] for q in seqs], np.int64), p / p.sum())
        return self._cache[1:]

    def first_slot(self, ex, i, rng):
        seqs, _, p = self._tables(ex)
        return int(seqs[int(rng.choice(len(p), p=p))][0])

    def next_slot(self, ex, t, slot, rng):
        seqs, first, p = self._tables(ex)
        if t == 1:
            cand = np.flatnonzero(first == int(slot))
            if not len(cand):
                cand = np.arange(len(p))
            pc = p[cand] / p[cand].sum()
            self._seq, self._pos = seqs[int(cand[int(rng.choice(len(cand), p=pc))])], 0
        self._pos += 1
        if self._pos >= len(self._seq):
            self._seq, self._pos = seqs[int(rng.choice(len(p), p=p))], 0
        return int(self._seq[self._pos])

    def to_state(self):
        return dict(lib=list(self.lib))

    def from_state(self, d):
        self.set_library(d["lib"])
