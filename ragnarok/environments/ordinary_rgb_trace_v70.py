"""Prospective ordinary RGB trace adapter, retaining the existing paid engine.

The start screen is captured after any construction and before the requested
acts. Only images cross this interface: no terminal/reset flag or native clock.
"""
from dataclasses import dataclass

import numpy as np

from ragnarok.environments.rgb_branch_trace_v70 import _RGBTraceColdEngine


@dataclass(frozen=True)
class OrdinaryRGBTrace:
    start_pair: np.ndarray  # Two wrapper RGB buffers; cold initialization duplicates its first screen.
    raw: np.ndarray  # One actual image after each paid act; may have0..4 rows.
    pooled: np.ndarray
    endpoint: np.ndarray


def validate_trace(trace):
    if (not isinstance(trace,OrdinaryRGBTrace) or not isinstance(trace.start_pair,np.ndarray)
            or trace.start_pair.dtype!=np.uint8 or trace.start_pair.ndim!=4
            or trace.start_pair.shape[0]!=2 or trace.start_pair.shape[-1]!=3
            or min(trace.start_pair.shape[1:3])<1):
        raise ValueError('An explicit two-buffer RGB start is required')
    shape=trace.start_pair.shape[1:]
    if (not isinstance(trace.raw,np.ndarray) or trace.raw.dtype!=np.uint8 or trace.raw.ndim!=4
            or trace.raw.shape[1:]!=shape or not 0<=len(trace.raw)<=4
            or not isinstance(trace.pooled,np.ndarray) or trace.pooled.dtype!=np.uint8
            or trace.pooled.shape!=trace.raw.shape or not isinstance(trace.endpoint,np.ndarray)
            or trace.endpoint.dtype!=np.uint8 or trace.endpoint.shape!=shape):
        raise ValueError('Ordinary trace contains only0..4 actual raw/pooled RGB frames')
    last=trace.start_pair[-1]
    for raw,pooled in zip(trace.raw,trace.pooled,strict=True):
        if not np.array_equal(np.maximum(last,raw),pooled):
            raise ValueError('Pooled trace does not match the actual adjacent RGB buffers')
        last=raw
    endpoint=trace.pooled[-1] if len(trace.raw) else np.maximum(*trace.start_pair)
    if not np.array_equal(endpoint,trace.endpoint):
        raise ValueError('Ordinary trace endpoint differs from its observed history')
    return len(trace.raw)


def endpoint_pair(trace):
    validate_trace(trace)
    if len(trace.raw)>=2:return trace.raw[-2:].copy()
    if len(trace.raw)==1:return np.stack((trace.start_pair[-1],trace.raw[-1]))
    return trace.start_pair.copy()


class _OrdinaryRGBTraceEngine(_RGBTraceColdEngine):
    def __init__(self,config):
        if config.frame_skip!=4:raise ValueError('The ordinary trace draft fixes action repeat4')
        self._ordinary_starts=None
        super().__init__(config)

    def observe_pair(self):
        self._guard()
        return np.stack(tuple(self._frames))

    def _advance(self,action,frames,*,experimental):
        if self._ordinary_starts is not None:
            if experimental:raise ValueError('An ordinary trace cannot silently contain experimental acts')
            self._ordinary_starts.append(self.observe_pair())
        return super()._advance(action,frames,experimental=experimental)

    def step_trace(self,action):
        self._guard()
        if self._recording is not None or self._ordinary_starts is not None or self._saves:
            raise ValueError('An ordinary trace requires no nested recording or manual snapshot session')
        before=self._budget.charged_frames
        self._recording=[]
        self._ordinary_starts=[]
        try:
            endpoint=super().step(action)
            if len(self._recording)!=1 or len(self._ordinary_starts)!=1:
                raise ValueError('Exactly one ordinary advance must supply its start and trace')
            trace=self._recording[0]
            if not np.array_equal(trace.endpoint,endpoint):
                raise ValueError('Ordinary endpoint differs from the unchanged engine')
            result=OrdinaryRGBTrace(self._ordinary_starts[0],trace.raw,trace.pooled,trace.endpoint)
            validate_trace(result)
            return result
        except BaseException as error:
            self._record_failure(error,before)
            raise
        finally:
            self._recording=None
            self._ordinary_starts=None


@dataclass(frozen=True)
class RGBChronologyStep:
    epoch: int
    start_frame: int
    end_frame: int
    visual_discontinuity: bool


class RGBChronology:
    """Own delivered-act clock and pixel-derived continuity, not a death detector.

Bootstrap/restore charges never pretend to be observed frames. A changed start
screen cuts this local memory; an invisible native reset cannot be inferred.
"""
    def __init__(self,initial_raw):
        if (not isinstance(initial_raw,np.ndarray) or initial_raw.dtype!=np.uint8
                or initial_raw.ndim!=3 or initial_raw.shape[-1]!=3 or min(initial_raw.shape[:2])<1):
            raise ValueError('Initial actual RGB screen required')
        self._last_raw=initial_raw.copy()
        self._epoch=self._frame=0

    def accept(self,trace):
        length=validate_trace(trace)
        if trace.start_pair.shape[1:]!=self._last_raw.shape:
            raise ValueError('Resolution changes require a fresh chronology')
        # Compare actual current screens only. Resetting a pooling buffer must
        # not by itself manufacture a native-reset annotation for the learner.
        discontinuity=not np.array_equal(self._last_raw,trace.start_pair[-1])
        epoch=self._epoch+int(discontinuity)
        start=0 if discontinuity else self._frame
        end=start+length
        if end>2**53:raise ValueError('Delivered frame clock exceeds exact numeric range')
        last=(trace.raw[-1] if length else trace.start_pair[-1]).copy()
        result=RGBChronologyStep(epoch,start,end,discontinuity)
        self._epoch,self._frame,self._last_raw=epoch,end,last
        return result
