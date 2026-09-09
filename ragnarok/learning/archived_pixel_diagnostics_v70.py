"""Prospective CPU objective/gradient diagnostics on attested own RGB clips.

The caller must seal archive provenance. No emulator, optimizer or weight update
exists here. Summing losses with unit coefficients only exercises their graphs;
it does not define the future training recipe or demonstrate useful learning.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from threading import Lock
from weakref import WeakKeyDictionary

import numpy as np
import torch

from ragnarok.environments.owned_rgb_collector_v70 import StepExperience,_pair,_effect,_branch_trace
from ragnarok.environments.ordinary_rgb_trace_v70 import validate_trace,endpoint_pair
from ragnarok.learning.parser_v70 import CounterfactualParser,max_pool_rgb
from ragnarok.learning.perception_stream_v70 import PerceptionStream
from ragnarok.learning.pixel_prediction_v70 import prediction_terms
from ragnarok.learning.visual_instances_v70 import DensePrediction,extract_instances
from ragnarok.learning.counterfactual_targets_v70 import component_targets,controllability_loss
from ragnarok.learning.gradient_guard_v70 import admit_adam_gradients


# A parser is dedicated for its lifetime, including after a stopped diagnostic.
# Weak keys do not keep otherwise unreachable networks alive. This registry
# prevents two diagnostic owners; callers must also keep outside code away from
# the owned network and must not construct aliases sharing its parameter storage.
_OWNED_PARSERS=WeakKeyDictionary()
_OWNERSHIP_LOCK=Lock()


def weight_sha256(parser):
    digest=hashlib.sha256()
    for name,value in parser.state_dict().items():
        if value.device.type!='cpu':raise ValueError('Archived diagnostics are CPU only')
        array=value.detach().contiguous().numpy()
        digest.update(json.dumps([name,list(array.shape),str(array.dtype)],separators=(',',':')).encode()+b'\n')
        digest.update(array.tobytes())
    return digest.hexdigest()


class ArchivedPixelDiagnostics:
    """Exclusive owner of one CPU parser and one discontinuous archived stream.

Only contiguous archive indices may share identity history. A collection gap
starts fresh tracking at its actual captured source anchor, without inventing
intermediate observations. RGB-derived epochs are preserved as bookkeeping.
"""
    def __init__(self,parser,*,stream_id):
        if (not isinstance(parser,CounterfactualParser) or not isinstance(stream_id,str) or not stream_id
                or any(p.device.type!='cpu' or p.dtype!=torch.float32 or not torch.isfinite(p).all()
                       or p.grad is not None
                       for p in parser.parameters())):
            raise ValueError('Own finite float32 CPU parser without gradients and explicit archive stream required')
        self._parser,self._stream_id=parser,stream_id
        self._lock=Lock();self._stopped=False
        self._initial_weights=weight_sha256(parser)
        self._index=None;self._end=None;self._epoch=None;self._pair=None;self._stream=None
        self._experiments=set()
        # Claim only after all other constructor allocations have succeeded.
        with _OWNERSHIP_LOCK:
            if parser in _OWNED_PARSERS:raise ValueError('Parser already owned by an archived diagnostic')
            _OWNED_PARSERS[parser]=True

    def _enter(self):
        if not self._lock.acquire(blocking=False):raise ValueError('Another archived diagnostic is active')
        if self._stopped:
            self._lock.release();raise ValueError('Archived diagnostics stopped after failure')

    def _infer(self,pair,effect):
        previous=torch.tensor(pair[0]).permute(2,0,1)[None]
        current=torch.tensor(pair[1]).permute(2,0,1)[None]
        source=max_pool_rgb(previous,current)
        output=self._parser(source,torch.tensor(effect,dtype=torch.float32)[None])
        for name in ('embedding','foreground_logits','displacement','controllability_logits','solidity_logits'):
            if not torch.isfinite(getattr(output,name)).all():raise ValueError('Nonfinite archived parser head')
        return source,output

    def _backward(self,losses):
        self._parser.zero_grad(set_to_none=True)
        try:
            if any(not torch.isfinite(value) for value in losses.values()):raise ValueError('Nonfinite diagnostic term')
            total=sum(losses.values())
            if not torch.isfinite(total):raise ValueError('Nonfinite diagnostic sum')
            total.backward()
            admission=admit_adam_gradients(self._parser,max_norm=1.)
            if weight_sha256(self._parser)!=self._initial_weights:
                raise ValueError('Diagnostic unexpectedly modified model weights')
            return dict(losses={name:float(value.detach()) for name,value in losses.items()},
                diagnostic_sum=float(total.detach()),gradient=asdict(admission),weights_unchanged=True)
        finally:
            # A late failure must not leave the preceding backward available to
            # an accidental consumer. The diagnostic remains stopped on error.
            self._parser.zero_grad(set_to_none=True)

    def step(self,index,record):
        self._enter()
        try:
            if (type(index) is not int or index<0 or self._index is not None and index<=self._index
                    or not isinstance(record,StepExperience) or record.worker!=0):
                raise ValueError('Strictly increasing archive indices and one own worker record required')
            record=copy.deepcopy(record)
            trace=record.trace;length=validate_trace(trace);effect=_effect(record.effect_assumption)
            timing=record.chronology
            if (type(timing.epoch) is not int or timing.epoch<0 or type(timing.start_frame) is not int
                    or type(timing.end_frame) is not int or not 0<=timing.start_frame<=timing.end_frame<=2**53
                    or timing.end_frame-timing.start_frame!=length or type(timing.visual_discontinuity) is not bool):
                raise ValueError('Recorded delivered-image chronology must match its actual RGB prefix')
            contiguous=self._index is not None and index==self._index+1
            changed=False
            if contiguous:
                changed=not np.array_equal(self._pair[-1],trace.start_pair[-1])
                expected_epoch=self._epoch+int(changed)
                expected_start=0 if changed else self._end
                if (timing.epoch,timing.start_frame,timing.visual_discontinuity)!=(expected_epoch,expected_start,changed):
                    raise ValueError('Contiguous archived chronology disagrees with its current raw screens')
            restart=not contiguous or changed
            stream=self._stream
            if restart:
                stream=PerceptionStream(self._parser,trace.start_pair.shape[1:3],
                    stream_id=f'{self._stream_id}/clip/{index}',epoch=timing.epoch,max_tracks=512,max_groups=512)
                # An archived source is a reused observation, never a newly paid
                # frame or an additional diagnostic target in the returned row.
                stream.observe(*trace.start_pair,native_frame=timing.start_frame)
            source,output=self._infer(trace.start_pair,effect)
            zero=output.displacement.reshape(-1)[0]*0+output.foreground_logits.reshape(-1)[0]*0
            if length==4:
                target=torch.tensor(trace.endpoint).permute(2,0,1)[None]
                terms=prediction_terms(source,target,output)
                p,foreground=terms['image_l1'],terms['foreground_bce']
            else:p=foreground=zero
            observed=stream.observe(*endpoint_pair(trace),native_frame=timing.end_frame) if length else None
            losses=dict(prediction=p,foreground=foreground,
                grouping_pull=observed.grouping.pull if observed else zero,
                grouping_separation=observed.grouping.separation if observed else zero,
                identity=observed.identity.loss if observed else zero)
            diagnostics=self._backward(losses)
            result=dict(index=index,actual_frames=length,prediction_eligible=length==4,
                history_restarted=restart,archive_epoch=timing.epoch,delivered_end_frame=timing.end_frame,
                observed_instances=len(observed.tracking.observed) if observed else 0,
                motion_groups=observed.grouping.groups if observed else 0,
                identity_anchors=observed.identity.supervised_anchors if observed else 0,
                identity_shared_ids=observed.identity.shared_visible_ids if observed else 0,**diagnostics)
            self._index,self._end,self._epoch,self._pair,self._stream=index,timing.end_frame,timing.epoch,endpoint_pair(trace),stream
            return result
        except BaseException:
            self._stopped=True
            raise
        finally:self._lock.release()

    def branch(self,operation_id,*,start_pair,effect,intervention,control,requested_frames):
        """One archived paid pair; no clock advance or claim of new experience."""
        self._enter()
        try:
            if (not isinstance(operation_id,tuple) or len(operation_id)!=2
                    or not isinstance(operation_id[0],str) or type(operation_id[1]) is not int
                    or operation_id[1]<1 or operation_id in self._experiments
                    or type(requested_frames) is not int or not 1<=requested_frames<=64):
                raise ValueError('A unique archived operation and explicit branch horizon required')
            pair=_pair(start_pair);effect=_effect(effect)
            intervention,control=copy.deepcopy((intervention,control))
            for trace in (intervention,control):_branch_trace(trace,pair,requested_frames)
            source,output=self._infer(pair,effect)
            def detached(name):return getattr(output,name).detach().numpy()[0]
            dense=DensePrediction(*(detached(name) for name in (
                'embedding','foreground_logits','displacement','controllability_logits','solidity_logits')))
            own=extract_instances(dense,source[0].permute(1,2,0).numpy())
            targets=component_targets(own,dense.displacement,intervention,control,horizon_frames=4,image_kind='pooled')
            loss=controllability_loss(output.controllability_logits[0,0],targets)
            result=dict(operation_id=list(operation_id),own_instances=len(own),
                common_prefix_frames=targets.common_prefix_frames,
                supervised_pixels=int(np.count_nonzero(targets.weights)),
                positive_pixels=int(np.count_nonzero((targets.weights>0)&(targets.values>0))),
                **self._backward({'controllability':loss}))
            self._experiments.add(operation_id)
            return result
        except BaseException:
            self._stopped=True
            raise
        finally:self._lock.release()
