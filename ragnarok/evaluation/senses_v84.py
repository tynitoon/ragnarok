"""v84 senses and trace (docs/ARC4_V84_PROTOCOL_EN.md §3.2, §5, §7.2). Evaluator side.

Same agent API as v83's SenseEnv (step -> (obs, rise, lives, end); clone/restore/free; billed; sibling),
plus the ladder quantity: the cumulative SENSED rise of a trajectory, sum over its agent steps of
max(0, r_step), kept with every handle (restored with it), and the lineage depth (agent steps since the
reset). Trace84 records, per epoch, every clone that raises the best cumulative sensed rise: (billed,
cum_rise, true score, depth, handle). Replay (note 21 of v83: one NOOP before restoring) recomputes the
cumulative sensed rise and the true score from the reset base.
"""
import pickle
import zlib

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.evaluation.senses_v83 import LOAD_FRAMES, SenseEnv, Spent, Trace   # noqa: F401


class Trace84(Trace):
    """Records new bests of the cumulative sensed rise (the v84 ladder), with true score and depth."""

    def on_clone(self, h, parent, since, billed, true, cum=0.0, depth=0):
        self.lineage[h] = (parent, since)
        base = parent if parent < 0 else self.base_of[parent]
        self.base_of[h] = base
        if cum > self.best[base]:
            self.best[base] = cum
            self.by_base[base].append((int(billed), float(cum), int(h), float(true), int(depth)))
            if base == self.cur and self.stop_at is not None and cum >= self.stop_at:
                self.stopped = True

    def replay84(self, h):
        """(cumulative sensed rise, true score, depth) at handle h, replayed from its reset base."""
        base, acts = self.actions(h)
        env = AtariEnv(self.game, seed=self.seed)
        env.step(0)
        env.ale.restoreState(pickle.loads(zlib.decompress(self.bases[base])))
        env._true_score = 0.0
        cum = 0.0
        for a in acts:
            before = env._true_score
            env.step(a)
            cum += max(0.0, env._true_score - before)
        return cum, env.true_score(), len(acts)

    def first_reaching(self, m):
        return next((r for r in self.records if r[1] >= m), None)


class SenseEnv84(SenseEnv):
    """SenseEnv that keeps the cumulative sensed rise and the lineage depth with every handle."""

    def __init__(self, game, seed, cap=None, trace=None, _ledger=None):
        self._cum, self._depth = 0.0, 0
        super().__init__(game, seed, cap=cap, trace=trace, _ledger=_ledger)
        self._meta = {}

    def reset(self):
        out = super().reset()
        self._cum, self._depth = 0.0, 0
        return out

    def step(self, a):
        obs, rise, lives, end = super().step(a)
        self._cum += rise
        self._depth += 1
        return obs, rise, lives, end

    def clone(self):
        h = self._next
        self._next += 1
        st = zlib.compress(pickle.dumps(self._e.ale.cloneState()), 1)
        ob = zlib.compress(self._e.obs().tobytes(), 1)
        self._store[h] = (st, ob, self._e._true_score, self._e.ale.lives())
        self._meta[h] = (self._cum, self._depth)
        if self._trace is not None:
            self._trace.on_clone(h, self._anchor, bytes(self._since), self.billed, self._e._true_score,
                                 self._cum, self._depth)
        self._anchor, self._since = h, bytearray()
        return h

    def restore(self, h):
        out = super().restore(h)
        self._cum, self._depth = self._meta[h]
        return out

    def free(self, h):
        super().free(h)
        self._meta.pop(h, None)

    def sibling(self, offset):
        return SenseEnv84(self._game, self._seed + int(offset), _ledger=self._led)
