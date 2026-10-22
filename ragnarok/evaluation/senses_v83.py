"""v83 senses (docs/ARC4_V83_PROTOCOL.md §0-§1): the only door between the agent and the emulator.

The evaluator builds a SenseEnv and hands it to the agent. `step(a) -> (obs, rise, lives, end)`:
rise = max(0, ALE reward summed over the 4 frames of the step), lives = ale.lives(), end = game over
(an episode boundary only, never a label). `clone()` returns an opaque int handle (zlib-compressed
ALE state and max-pooled obs); `restore(h)` bills 60 frames; `free(h)` drops a stored state.

The billed ledger counts every emulated frame of the family (probes and the adapter's sibling
emulator included) and never decreases; at `cap` billed frames, or once the evaluator's trace asks
for an early stop, `step` and `restore` raise Spent. The true cumulative score exists only in the
evaluator trace (Trace). This module lives outside `ragnarok/agent_v83/`, which never imports
`ragnarok.evaluation` (tests/test_isolation_v83.py).
"""
import pickle
import zlib

import numpy as np

from ragnarok.agent_v71.env import AtariEnv

LOAD_FRAMES = 60          # ARC4_VISION 2.5: fixed charge per savestate load (AtariEnv.restore)


class Spent(Exception):
    """The billed-frame cap is reached (or the evaluator stopped the run): the run is over."""


class _Ledger:
    def __init__(self, cap):
        self.cap = cap
        self.envs = []

    @property
    def billed(self):
        return sum(e.frames for e in self.envs)


class Trace:
    """Evaluator only. At every clone of the traced emulator that raises the best true cumulative
    score so far, records (billed frames, true score, handle). The lineage of every handle (parent
    handle or reset base, actions since) allows a replay from the base, outside the ledger."""

    def __init__(self, stop_at=None):
        self.stop_at = stop_at
        self.stopped = False
        self.records = []          # (billed, true, handle), billed increasing
        self.best = -np.inf
        self.lineage = {}          # handle -> (parent, actions since parent)
        self.bases = {}            # negative id -> pickled ALE state right after a reset
        self.game = self.seed = None

    def on_reset(self, base, state):
        self.bases[base] = state

    def on_clone(self, h, parent, since, billed, true):
        self.lineage[h] = (parent, since)
        if true > self.best:
            self.best = true
            self.records.append((int(billed), float(true), int(h)))
            if self.stop_at is not None and true >= self.stop_at:
                self.stopped = True

    def actions(self, h):
        """(base id, actions from that reset to handle h)."""
        segs = []
        while h > 0:
            h, seg = self.lineage[h]
            segs.append(seg)
        return h, b"".join(reversed(segs))

    def replay(self, h):
        """True cumulative score at handle h, replayed from its reset base on a fresh emulator."""
        base, acts = self.actions(h)
        env = AtariEnv(self.game, seed=self.seed)
        env.ale.restoreState(pickle.loads(zlib.decompress(self.bases[base])))
        env._true_score = 0.0
        for a in acts:
            env.step(a)
        return env.true_score()

    def first_reaching(self, m):
        """(billed, true, handle) of the first clone whose true cumulative score is >= m, or None."""
        return next((r for r in self.records if r[1] >= m), None)


class SenseEnv:
    def __init__(self, game, seed, cap=None, trace=None, _ledger=None):
        self._e = AtariEnv(game, seed=seed)
        self._game, self._seed = game, int(seed)
        self._led = _ledger or _Ledger(cap)
        self._led.envs.append(self._e)
        self._trace = trace
        if trace is not None:
            trace.game, trace.seed = game, int(seed)
        self._store = {}
        self._next = 1
        self._nbase = 0
        self._shape = self._e.obs().shape
        self.n_actions = self._e.n_actions
        self._new_base()

    # ---- agent API ---------------------------------------------------------------------------
    @property
    def billed(self):
        return self._led.billed

    def reset(self):
        self._check()
        self._e.reset()
        self._new_base()
        return self._e.obs(), self._e.ale.lives()

    def obs(self):
        return self._e.obs()

    def step(self, a):
        self._check()
        a = int(a)
        before = self._e._true_score
        obs, end = self._e.step(a)
        self._since.append(a)
        return obs, max(0.0, self._e._true_score - before), self._e.ale.lives(), end

    def clone(self):
        h = self._next
        self._next += 1
        st = zlib.compress(pickle.dumps(self._e.ale.cloneState()), 1)
        ob = zlib.compress(self._e.obs().tobytes(), 1)
        self._store[h] = (st, ob, self._e._true_score, self._e.ale.lives())
        if self._trace is not None:
            self._trace.on_clone(h, self._anchor, bytes(self._since), self.billed, self._e._true_score)
        self._anchor, self._since = h, bytearray()
        return h

    def restore(self, h):
        """Back to handle h (60 billed frames). Returns (obs, lives) there."""
        self._check()
        st, ob, true, lives = self._store[h]
        self._e.ale.restoreState(pickle.loads(zlib.decompress(st)))
        o = np.frombuffer(zlib.decompress(ob), np.uint8).reshape(self._shape)
        self._e._prev, self._e._last = o.copy(), o.copy()     # obs() = max(prev, last) = o
        self._e._true_score = true
        self._e.loads += 1
        self._e.frames += LOAD_FRAMES
        self._anchor, self._since = h, bytearray()
        return o.copy(), lives

    def free(self, h):
        self._store.pop(h, None)

    def sibling(self, offset):
        """The same game on another emulator seed (seed + offset), billed to the same ledger, untraced."""
        return SenseEnv(self._game, self._seed + int(offset), _ledger=self._led)

    # ---- private -----------------------------------------------------------------------------
    def _new_base(self):
        self._nbase -= 1
        if self._trace is not None:
            self._trace.on_reset(self._nbase, zlib.compress(pickle.dumps(self._e.ale.cloneState()), 1))
        self._anchor, self._since = self._nbase, bytearray()

    def _check(self):
        if self._led.cap is not None and self._led.billed >= self._led.cap:
            raise Spent("billed cap")
        if self._trace is not None and self._trace.stopped:
            raise Spent("evaluator stop")
