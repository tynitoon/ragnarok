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
    """Evaluator only. Each reset opens an epoch (its base: the ALE state right after the reset). At
    every clone of the traced emulator that raises the best true cumulative score of its epoch, records
    (billed frames, true score, handle). The lineage of every handle (parent handle or base, actions
    since) allows a replay outside the ledger. A replay starts from the handle's ANCHOR: the first clone
    on its lineage taken after episode frame 0 (its state is kept), else from the base. In some games
    (Q*bert) a state saved at episode frame 0 does not restore faithfully (build notes 21), so a replay
    from the reset base can leave the live trajectory; from frame 1 on, restores are exact. Milestones are read on the LAST epoch,
    the explorer's tree (it resets once, after entry; entry-phase states do not count). The run script
    sets `stop_at` only after entry."""

    def __init__(self, stop_at=None):
        self.stop_at = stop_at
        self.stopped = False
        self.by_base = {}          # base -> [(billed, true, handle)], billed increasing
        self.best = {}
        self.lineage = {}          # handle -> (parent, actions since parent)
        self.base_of = {}
        self.frame = {}            # handle -> episode frame number at the clone
        self.anchors = {}          # handle -> (pickled state, true score): first clones after frame 0
        self.bases = {}            # negative id -> pickled ALE state right after a reset
        self.cur = None
        self.game = self.seed = None
        self.rise_billed = []      # billed frames and sizes of the first 1000 sensed rises (all epochs)
        self.rise_sizes = []
        self.n_rises = 0

    @property
    def records(self):
        return self.by_base.get(self.cur, [])

    def on_reset(self, base, state):
        self.bases[base] = state
        self.by_base[base], self.best[base], self.cur = [], -np.inf, base

    def on_clone(self, h, parent, since, billed, true, state=None, frame=None):
        self.lineage[h] = (parent, since)
        self.frame[h] = frame
        if state is not None and frame and (parent < 0 or not self.frame.get(parent)):
            self.anchors[h] = (state, float(true))
        base = parent if parent < 0 else self.base_of[parent]
        self.base_of[h] = base
        if true > self.best[base]:
            self.best[base] = true
            self.by_base[base].append((int(billed), float(true), int(h)))
            if base == self.cur and self.stop_at is not None and true >= self.stop_at:
                self.stopped = True

    def on_rise(self, billed, rise):
        self.n_rises += 1
        if len(self.rise_billed) < 1000:
            self.rise_billed.append(int(billed))
            self.rise_sizes.append(float(rise))

    def start(self, h):
        """(start state, true score there, actions from there to handle h): from h's anchor if its
        lineage has one, else from its reset base."""
        segs, x = [], h
        while x > 0:
            if x in self.anchors:
                st, true = self.anchors[x]
                return st, true, b"".join(reversed(segs))
            x, seg = self.lineage[x]
            segs.append(seg)
        return self.bases[x], 0.0, b"".join(reversed(segs))

    def replay(self, h):
        """True cumulative score at handle h, replayed from its start state on a fresh emulator."""
        st, true, acts = self.start(h)
        env = AtariEnv(self.game, seed=self.seed)
        env.ale.restoreState(pickle.loads(zlib.decompress(st)))
        env._true_score = true
        for a in acts:
            env.step(a)
        return env.true_score()

    def first_reaching(self, m):
        """(billed, true, handle) of the explorer epoch's first clone with true score >= m, or None."""
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
        rise = max(0.0, self._e._true_score - before)
        if rise > 0 and self._trace is not None:
            self._trace.on_rise(self.billed, rise)
        return obs, rise, self._e.ale.lives(), end

    def clone(self):
        h = self._next
        self._next += 1
        st = zlib.compress(pickle.dumps(self._e.ale.cloneState()), 1)
        ob = zlib.compress(self._e.obs().tobytes(), 1)
        self._store[h] = (st, ob, self._e._true_score, self._e.ale.lives())
        if self._trace is not None:
            self._trace.on_clone(h, self._anchor, bytes(self._since), self.billed, self._e._true_score,
                                 st, self._e.ale.getEpisodeFrameNumber())
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
