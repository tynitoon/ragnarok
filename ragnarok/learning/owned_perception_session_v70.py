"""Prospective temporal perception intake behind owned RGB/P/C operations.

The clock counts delivered ordinary act images, not native bootstrap/restore
charges. Visual discontinuities alone reset this memory. No policy or optimizer.
"""
from dataclasses import dataclass
import copy
from functools import wraps
from threading import Lock

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.environments.ordinary_rgb_trace_v70 import endpoint_pair
from ragnarok.learning.owned_pixel_objectives_v70 import OwnedPixelObjectives,OwnedPrediction
from ragnarok.learning.perception_stream_v70 import PerceptionStream,PerceptionFrame


@dataclass(frozen=True)
class OwnedPerceptionBatch:
    operation_id: tuple[str,int]
    prediction: tuple[OwnedPrediction,...]
    observations: tuple[PerceptionFrame|None,...]


def _serialized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        if not self._operation_lock.acquire(blocking=False):
            raise ValueError('Another temporal perception operation is in progress')
        try:return method(self,*args,**kwargs)
        finally:self._operation_lock.release()
    return call


class OwnedPerceptionSession:
    """Exclusive synchronous owner of collector, parser and per-worker memories.

Temporal observations use the UNKNOWN (zero) forthcoming effect hypothesis;
their tracker motion is the parser's own uncertain prediction under that input.
P/C separately use the already recorded chosen effect via OwnedPixelObjectives.
No raw button or hidden native event supplies a temporal feature.
"""
    def __init__(self,collector,parser,*,workers,max_tracks=512,max_groups=512):
        self._operation_lock=Lock()
        if (not isinstance(collector,OwnedRGBCollector) or type(workers) is not int
                or workers!=collector._workers):
            raise ValueError('The explicit worker count must match the exclusively owned collector')
        self._objectives=OwnedPixelObjectives(collector,parser)
        self._parser,self._workers=parser,workers
        self._max_tracks,self._max_groups=max_tracks,max_groups
        self._stopped=False
        self._pending=None
        streams=[]
        for i in range(workers):
            anchor=collector.current_anchor(i)
            stream=self._new_stream(anchor.pair.shape[1:3],f'{anchor.token[0]}/worker/{i}',anchor.epoch)
            # Seed RGB memory once. No initial autograd graph is retained by
            # PerceptionStream or presented as an additional paid observation.
            stream.observe(*anchor.pair,native_frame=anchor.frame)
            streams.append(stream)
        self._streams=streams

    def _new_stream(self,shape,stream_id,epoch):
        return PerceptionStream(self._parser,shape,stream_id=stream_id,epoch=epoch,
                                max_tracks=self._max_tracks,max_groups=self._max_groups)

    def _guard(self):
        if self._stopped:raise ValueError('Temporal perception owner stopped after failure or closure')

    @_serialized
    def current_anchor(self,worker=0):
        self._guard()
        return self._objectives.current_anchor(worker)

    @_serialized
    def step(self,actions,*,effects=None):
        self._guard()
        try:
            self._pending=self._objectives.step(actions,effects=effects)
            if len(self._pending)!=self._workers:raise ValueError('Incomplete owned perception batch')
            candidates=[];observations=[]
            operation=self._pending[0].record.operation_id
            for i,(prediction,previous) in enumerate(zip(self._pending,self._streams,strict=True)):
                record=prediction.record
                clock=record.chronology
                if record.worker!=i or record.operation_id!=operation:
                    raise ValueError('One ordered owned operation must supply the temporal batch')
                if clock.visual_discontinuity:
                    if clock.epoch!=previous.epoch+1 or clock.start_frame!=0:
                        raise ValueError('Visual memory reset must start the next own epoch at zero')
                    candidate=self._new_stream(record.trace.start_pair.shape[1:3],previous.stream_id,clock.epoch)
                    observed=candidate.observe(*record.trace.start_pair,native_frame=0)
                else:
                    if clock.epoch!=previous.epoch or clock.start_frame!=previous._tracker._frame:
                        raise ValueError('Delivered-image chronology must continue this own memory')
                    # The reviewed observe implementation forks its tracker and
                    # builds fresh history before commit. A shallow shell lets
                    # us postpone ALL workers' commits without cloning weights
                    # or copying an optimizer graph. No saved entry is mutated.
                    candidate=copy.copy(previous)
                    observed=None
                if clock.end_frame>clock.start_frame:
                    observed=candidate.observe(*endpoint_pair(record.trace),native_frame=clock.end_frame)
                # T0 without a visual discontinuity produces NO temporal target
                # and never retries observe at an already occupied timestamp.
                candidates.append(candidate);observations.append(observed)
            result=OwnedPerceptionBatch(operation,self._pending,tuple(observations))
            self._streams=candidates
            self._pending=None
            return result
        except BaseException:
            self._stopped=True
            raise

    @_serialized
    def branch(self,action,*,expected_anchor,worker=0,frames=4,effect=None):
        self._guard()
        try:
            # Branch time never enters the live tracker or its identity memory.
            return self._objectives.branch(action,expected_anchor=expected_anchor,
                                           worker=worker,frames=frames,effect=effect)
        except BaseException:
            self._stopped=True
            raise

    @_serialized
    def failure_records(self):
        if self._pending is not None:return copy.deepcopy(tuple(p.record for p in self._pending))
        return self._objectives.failure_records()

    @_serialized
    def report(self,worker=0):
        """External diagnostic metadata only, not a model observation."""
        return dict(temporal_owner_stopped=self._stopped,temporal_batch_retained=self._pending is not None,
                    objectives=self._objectives.report(worker))

    @_serialized
    def close(self):
        self._stopped=True
        self._objectives.close()
