"""v85 senses (docs/ARC4_V85_PROTOCOL_EN.md A.5, A.7, §13.3). Evaluator side.

SenseEnv85 = SenseEnv84 plus:
- `start_from(blob, S, cap)`: reset, one NOOP, restore the shared start s1 (billed 60); then cum = S(s1),
  depth = 0, trace base = the s1 state, clock start = billed now, cap = clock start + cap.
- `sandbox()`: a context in which rollouts are never billed and never traced (frames go to
  `oracle_frames`), `clone()` raises, the cap is not checked; on exit the ALE state, frame and load
  counters, anchor and since, cum, depth, true score and frames are restored exactly.
- `to_state()` / `from_state()` for checkpoints.
Trace85 records, relative to the clock start, (billed, cum, handle, true, depth, oracle frames) at every
clone that raises the best cumulative sensed rise, and `replay85` replays a handle's actions from s1 in
the same context as the live start (fresh emulator, reset, one NOOP, restore s1).
"""
import contextlib
import pickle
import zlib

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.evaluation.senses_v84 import LOAD_FRAMES, SenseEnv84, Spent, Trace84   # noqa: F401


def make_blob(env):
    """The start blob of an AtariEnv at its current state: ALE state, max-pooled obs, true score, lives."""
    return pickle.dumps(dict(state=zlib.compress(pickle.dumps(env.ale.cloneState()), 1),
                             obs=zlib.compress(env.obs().tobytes(), 1), shape=env.obs().shape,
                             true=float(env._true_score), lives=int(env.ale.lives())))


class Trace85(Trace84):
    def __init__(self, stop_at=None):
        super().__init__(stop_at)
        self.start_blob, self.start_S = None, 0.0

    def on_clone(self, h, parent, since, billed, true, cum=0.0, depth=0, oracle=0):
        self.lineage[h] = (parent, since)
        base = parent if parent < 0 else self.base_of[parent]
        self.base_of[h] = base
        if cum > self.best[base]:
            self.best[base] = cum
            self.by_base[base].append((int(billed), float(cum), int(h), float(true), int(depth), int(oracle)))
            if base == self.cur and self.stop_at is not None and cum >= self.stop_at:
                self.stopped = True

    def replay85(self, h):
        """(cum, true, depth) at handle h, replayed from s1 like the live start (reset, NOOP, restore)."""
        base, acts = self.actions(h)
        b = pickle.loads(self.start_blob)
        env = AtariEnv(self.game, seed=self.seed)
        env.step(0)
        env.ale.restoreState(pickle.loads(zlib.decompress(b["state"])))
        env._true_score = b["true"]
        cum = self.start_S
        for a in acts:
            before = env._true_score
            env.step(a)
            cum += max(0.0, env._true_score - before)
        return cum, env.true_score(), len(acts)

    def to_state(self):
        return {k: getattr(self, k) for k in ("stop_at", "stopped", "by_base", "best", "lineage", "base_of", "bases",
                                              "cur", "game", "seed", "rise_billed", "rise_sizes", "n_rises",
                                              "start_blob", "start_S")}

    def from_state(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class SenseEnv85(SenseEnv84):
    def __init__(self, game, seed, cap=None, trace=None, _ledger=None):
        self._sandbox = False
        self.clock_start = 0
        self.oracle_frames = 0
        super().__init__(game, seed, cap=cap, trace=trace, _ledger=_ledger)

    # ---------------------------------------------------------------- start
    def start_from(self, blob, S, cap=4_000_000):
        b = pickle.loads(blob)
        self._e.reset()
        self._e.step(0)
        self._e.ale.restoreState(pickle.loads(zlib.decompress(b["state"])))
        o = np.frombuffer(zlib.decompress(b["obs"]), np.uint8).reshape(b["shape"])
        self._e._prev, self._e._last = o.copy(), o.copy()
        self._e._true_score = b["true"]
        self._e.loads += 1
        self._e.frames += LOAD_FRAMES
        self._cum, self._depth = float(S), 0
        self._nbase -= 1
        if self._trace is not None:
            self._trace.on_reset(self._nbase, b["state"])
            self._trace.start_blob, self._trace.start_S = blob, float(S)
        self._anchor, self._since = self._nbase, bytearray()
        self.clock_start = self.billed
        self.oracle_frames = 0
        self._led.cap = self.clock_start + int(cap)
        return o.copy(), b["lives"]

    @property
    def billed_since_start(self):
        return self.billed - self.clock_start

    # ---------------------------------------------------------------- sandbox
    @contextlib.contextmanager
    def sandbox(self):
        """Unbilled, untraced look-ahead; everything restored on exit (A.7)."""
        assert not self._sandbox, "nested sandbox"
        e = self._e
        saved = dict(state=e.ale.cloneState(), prev=e._prev.copy(), last=e._last.copy(), true=e._true_score,
                     frames=e.frames, loads=e.loads, anchor=self._anchor, since=bytearray(self._since),
                     cum=self._cum, depth=self._depth, trace=self._trace)
        self._sandbox, self._trace = True, None
        try:
            yield self
        finally:
            self.oracle_frames += e.frames - saved["frames"]
            e.ale.restoreState(saved["state"])
            e._prev, e._last, e._true_score = saved["prev"], saved["last"], saved["true"]
            e.frames, e.loads = saved["frames"], saved["loads"]
            self._anchor, self._since = saved["anchor"], saved["since"]
            self._cum, self._depth = saved["cum"], saved["depth"]
            self._trace, self._sandbox = saved["trace"], False

    def _check(self):
        if self._sandbox:
            return
        super()._check()

    def reset(self):
        if self._sandbox:
            raise RuntimeError("reset() is forbidden inside the sandbox")
        return super().reset()

    def free(self, h):
        if self._sandbox:
            raise RuntimeError("free() is forbidden inside the sandbox")
        super().free(h)

    def clone(self):
        if self._sandbox:
            raise RuntimeError("clone() is forbidden inside the sandbox")
        h = self._next
        self._next += 1
        st = zlib.compress(pickle.dumps(self._e.ale.cloneState()), 1)
        ob = zlib.compress(self._e.obs().tobytes(), 1)
        self._store[h] = (st, ob, self._e._true_score, self._e.ale.lives())
        self._meta[h] = (self._cum, self._depth)
        if self._trace is not None:
            self._trace.on_clone(h, self._anchor, bytes(self._since), self.billed - self.clock_start,
                                 self._e._true_score, self._cum, self._depth, self.oracle_frames)
        self._anchor, self._since = h, bytearray()
        return h

    # ---------------------------------------------------------------- checkpoints
    def to_state(self):
        """§13.3, at an expansion boundary: no live ALE state and no RNG state (the next expansion restores)."""
        e = self._e
        return dict(frames=e.frames, loads=e.loads, store=dict(self._store), meta=dict(self._meta), next=self._next,
                    nbase=self._nbase, anchor=self._anchor, since=bytes(self._since), cum=self._cum,
                    depth=self._depth, clock_start=self.clock_start, oracle_frames=self.oracle_frames, cap=self._led.cap)

    def from_state(self, d):
        """On a fresh SenseEnv85: reset, one NOOP (after-step context, as in a live run), then the counters."""
        e = self._e
        e.reset()
        e.step(0)
        e.frames, e.loads = d["frames"], d["loads"]
        self._store, self._meta = dict(d["store"]), dict(d["meta"])
        self._next, self._nbase, self._anchor, self._since = d["next"], d["nbase"], d["anchor"], bytearray(d["since"])
        self._cum, self._depth = d["cum"], d["depth"]
        self.clock_start, self.oracle_frames, self._led.cap = d["clock_start"], d["oracle_frames"], d["cap"]
