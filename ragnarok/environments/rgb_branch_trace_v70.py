"""Prospective paid branch RGB traces, using the unchanged cold engine.

Private engine adapter only: no new RPC or training runner yet. Every trace
entry is a real image already paid by the base engine. There are no native
state/RAM/reward/lives fields or fictitious frames after a terminal.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, ROM_SPECS


@dataclass(frozen=True)
class RGBBranchTrace:
    raw: np.ndarray
    pooled: np.ndarray
    endpoint: np.ndarray


class _RecordingDeque(deque):
    def __init__(self, frames, sink):
        super().__init__(frames, maxlen=2)
        self.sink = sink

    def append(self, frame):
        super().append(frame)
        # The base _advance pays the native act before appending its RGB.
        pooled = np.maximum(self[-2], self[-1]) if len(self) == 2 else self[-1]
        self.sink.append((frame.copy(), pooled.copy()))


class _RGBTraceColdEngine(_ColdPixelEngine):
    def __init__(self, config):
        if config.game not in ROM_SPECS or config.mode != 'lab':
            raise ValueError('RGB branch traces require a declared cold training game in lab mode')
        self._recording = None
        self._trace_failure = None
        super().__init__(config)

    def _guard(self):
        if self._trace_failure is not None:
            raise ValueError('Trajectory stopped after a paid RGB trace failure')

    def _paid_call(self, method, *args):
        self._guard()
        before = self._budget.charged_frames
        try:
            return method(*args)
        except BaseException as error:
            if self._budget.charged_frames != before:
                self._trace_failure = dict(error_type=type(error).__name__, error=str(error))
            raise

    def _advance(self, action, frames, *, experimental):
        if self._recording is None:
            return super()._advance(action, frames, experimental=experimental)
        sink = []
        self._frames = _RecordingDeque(self._frames, sink)
        try:
            endpoint = super()._advance(action, frames, experimental=experimental)
            shape = (0, *endpoint.shape)
            raw = np.stack([p[0] for p in sink]) if sink else np.empty(shape, dtype=np.uint8)
            pooled = np.stack([p[1] for p in sink]) if sink else np.empty(shape, dtype=np.uint8)
            self._recording.append(RGBBranchTrace(raw, pooled, endpoint.copy()))
            return endpoint
        finally:
            self._frames = deque(self._frames, maxlen=2)

    def branch_pair_trace(self, action, control=0, frames=4):
        self._guard()
        if self._recording is not None:
            raise ValueError('Nested trace recording is unavailable')
        before = self._budget.charged_frames
        self._recording = []
        try:
            endpoints = super().branch_pair(action, control, frames)
            traces = tuple(self._recording)
            if len(traces) != 2 or any(not np.array_equal(t.endpoint, e)
                                      for t, e in zip(traces, endpoints, strict=True)):
                raise RuntimeError('RGB trace endpoints differ from the paid branch outputs')
            return traces
        except BaseException as error:
            if self._budget.charged_frames != before:
                self._trace_failure = dict(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self._recording = None

    def step(self, action):
        return self._paid_call(super().step, action)

    def branch_pair(self, action, control=0, frames=4):
        return self._paid_call(super().branch_pair, action, control, frames)

    def clone(self):
        self._guard()
        return super().clone()

    def restore(self, token):
        return self._paid_call(super().restore, token)
