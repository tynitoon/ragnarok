"""Prospective single owner of RGB operations and their causal recording metadata.

This is not a learner or native runner. A caller must relinquish other access
to the supplied pool; Python object privacy is a contract, not a security wall.
"""
from dataclasses import dataclass
import copy

import numpy as np

from ragnarok.environments.ordinary_rgb_trace_v70 import (
    RGBChronology,RGBChronologyStep,OrdinaryRGBTrace,endpoint_pair,validate_trace)
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace


@dataclass(frozen=True)
class RGBAnchor:
    token: tuple[str,int,int]  # Own session, operation sequence, worker.
    epoch: int
    frame: int  # Delivered-act image clock, not charged/bootstrap time.
    pair: np.ndarray


@dataclass(frozen=True)
class StepExperience:
    operation_id: tuple[str,int]
    worker: int
    requested_action: int  # Own command log only; never the parser's effect input.
    effect_assumption: np.ndarray
    chronology: RGBChronologyStep
    trace: OrdinaryRGBTrace


@dataclass(frozen=True)
class BranchExperience:
    operation_id: tuple[str,int]
    worker: int
    anchor: RGBAnchor
    requested_action: int
    effect_assumption: np.ndarray
    control_action: int
    requested_frames: int
    intervention: RGBBranchTrace
    control: RGBBranchTrace


def _pair(value):
    if (not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=4
            or value.shape[0]!=2 or value.shape[-1]!=3 or min(value.shape[1:3])<1):
        raise ValueError('Two own RGB buffers required')
    result=value.copy()
    result.flags.writeable=False
    return result


def _effect(value):
    if value is None:value=np.zeros(12,dtype=np.float32)
    if (not isinstance(value,np.ndarray) or value.shape!=(12,)
            or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all()
            or (value<0).any() or (value>1).any()):
        raise ValueError('Own12-effect assumption in[0,1] required')
    result=value.astype(np.float32,copy=True)
    result.flags.writeable=False
    return result


def _branch_trace(trace,start,frames):
    shape=start.shape[1:]
    if (not isinstance(trace,RGBBranchTrace) or not isinstance(trace.raw,np.ndarray)
            or trace.raw.dtype!=np.uint8 or trace.raw.ndim!=4 or trace.raw.shape[1:]!=shape
            or not 0<=len(trace.raw)<=frames or not isinstance(trace.pooled,np.ndarray)
            or trace.pooled.dtype!=np.uint8 or trace.pooled.shape!=trace.raw.shape
            or not isinstance(trace.endpoint,np.ndarray) or trace.endpoint.dtype!=np.uint8
            or trace.endpoint.shape!=shape):
        raise ValueError('Paid branch must have only its actual requested RGB prefix')
    previous=start[-1]
    for raw,pooled in zip(trace.raw,trace.pooled,strict=True):
        if not np.array_equal(np.maximum(previous,raw),pooled):
            raise ValueError('Branch RGB history does not start from the owned anchor')
        previous=raw
    endpoint=trace.pooled[-1] if len(trace.raw) else np.maximum(*start)
    if not np.array_equal(endpoint,trace.endpoint):raise ValueError('Branch endpoint/history mismatch')


class OwnedRGBCollector:
    """Bounded outstanding records; every paid call gets an own monotonically new ID.

An operation can be taken once as a NEW experience. A learner may later reuse
that recorded sample for declared updates, without calling it new emulation.
"""
    def __init__(self,pool,workers,*,session_id,max_pending=64):
        if type(workers) is not int or not 1<=workers<=16:
            raise ValueError('Declare1..16 owned workers')
        if not isinstance(session_id,str) or not session_id:
            raise ValueError('A fresh own recording session ID is required')
        if type(max_pending) is not int or not 1<=max_pending<=4096:
            raise ValueError('Declare pending operation capacity in1..4096')
        pairs=[_pair(pool.observe_pair(i)) for i in range(workers)]
        self._pool,self._workers,self._session=pool,workers,session_id
        self._max_pending=max_pending
        self._sequence=0
        self._stopped=False
        self._failed_operation=None
        self._failure=None
        self._pending={}
        self._chronologies=[RGBChronology(p[-1]) for p in pairs]
        self._anchors=[RGBAnchor((session_id,0,i),0,0,p) for i,p in enumerate(pairs)]

    def _guard(self):
        if self._stopped:raise ValueError('RGB collector stopped after an earlier operation failure or closure')

    def _worker(self,index):
        if type(index) is not int or not 0<=index<self._workers:raise ValueError('Unknown owned worker')

    def current_anchor(self,index=0):
        self._guard();self._worker(index)
        return copy.deepcopy(self._anchors[index])

    def _reserve(self):
        self._guard()
        if len(self._pending)>=self._max_pending:
            raise ValueError('Pending RGB experience capacity reached before any new command')
        self._sequence+=1
        return self._session,self._sequence

    def _fail(self,operation,error):
        # Stop before allocating an error description: an OOM in logging must
        # never leave the collector able to issue another paid command.
        self._stopped=True
        self._failed_operation=operation
        self._failure=dict(operation_id=operation,error_type=type(error).__name__,error=str(error))

    def step(self,actions,*,effects=None):
        self._guard()
        actions=list(actions)
        if len(actions)!=self._workers or any(type(a) is not int or not 0<=a<18 for a in actions):
            raise ValueError('One own integer button command in0..17 per worker required')
        if effects is None:effects=[None]*self._workers
        effects=list(effects)
        if len(effects)!=self._workers:raise ValueError('One own effect assumption per worker required')
        assumptions=[_effect(value) for value in effects]
        operation=self._reserve()
        try:
            traces=self._pool.step(actions)
            if len(traces)!=self._workers:raise ValueError('Incomplete paid worker batch')
            chronologies=copy.deepcopy(self._chronologies)
            anchors=[];records=[]
            for i,(action,effect,trace,clock) in enumerate(zip(actions,assumptions,traces,chronologies,strict=True)):
                validate_trace(trace)
                timing=clock.accept(trace)
                anchors.append(RGBAnchor((self._session,operation[1],i),timing.epoch,timing.end_frame,_pair(endpoint_pair(trace))))
                records.append(StepExperience(operation,i,action,effect,timing,copy.deepcopy(trace)))
            pending={**self._pending,operation:tuple(records)}
            self._anchors,self._chronologies,self._pending=anchors,chronologies,pending
            return operation
        except BaseException as error:
            self._fail(operation,error)
            raise

    def branch(self,action,*,expected_anchor,worker=0,frames=4,effect=None):
        self._guard();self._worker(worker)
        if type(action) is not int or not 0<=action<18 or type(frames) is not int or not 1<=frames<=64:
            raise ValueError('Own branch command and horizon1..64 required')
        anchor=self._anchors[worker]
        if expected_anchor!=anchor.token:
            raise ValueError('Stale or foreign anchor refused before any branch command')
        assumption=_effect(effect)
        operation=self._reserve()
        try:
            before=_pair(self._pool.observe_pair(worker))
            if not np.array_equal(before,anchor.pair):raise ValueError('Owned anchor RGB changed outside the collector')
            pair=self._pool.branch_pair_trace(action,control=0,frames=frames,index=worker)
            if not isinstance(pair,tuple) or len(pair)!=2:raise ValueError('One paid operation must return both branches')
            for trace in pair:_branch_trace(trace,anchor.pair,frames)
            after=_pair(self._pool.observe_pair(worker))
            if not np.array_equal(after,anchor.pair):raise ValueError('Paid branch did not restore the owned RGB anchor')
            record=BranchExperience(operation,worker,copy.deepcopy(anchor),action,assumption,0,frames,
                                    copy.deepcopy(pair[0]),copy.deepcopy(pair[1]))
            self._pending={**self._pending,operation:(record,)}
            return operation
        except BaseException as error:
            self._fail(operation,error)
            raise

    def take(self,operation):
        # Already collected successes remain available after a later failure.
        if operation not in self._pending:raise ValueError('Unknown or already consumed RGB operation')
        records=copy.deepcopy(self._pending[operation])
        del self._pending[operation]
        return records

    def report(self,index=0):
        self._worker(index)
        return dict(collector_stopped=self._stopped,failed_operation=self._failed_operation,
                    collector_failure=copy.deepcopy(self._failure),worker=self._pool.report(index))

    def close(self):
        self._stopped=True
        self._pool.close()
