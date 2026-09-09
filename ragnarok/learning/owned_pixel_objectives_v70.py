"""Prospective P/C intake from exclusively owned paid RGB operations.

No optimizer, policy, native runner or external labels. Forward passes here are
retrospective: their pixels cannot be used to reselect the action already sent.
"""
from dataclasses import dataclass
import copy
from functools import wraps
from threading import Lock

import torch

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector,StepExperience,BranchExperience
from ragnarok.learning.parser_v70 import CounterfactualParser,ParserOutput,max_pool_rgb
from ragnarok.learning.pixel_prediction_v70 import prediction_terms
from ragnarok.learning.visual_instances_v70 import DensePrediction,extract_instances
from ragnarok.learning.counterfactual_targets_v70 import (
    ControllabilityTargets,component_targets,controllability_loss)


@dataclass(frozen=True)
class OwnedPrediction:
    record: StepExperience
    source_rgb: torch.Tensor
    effect: torch.Tensor
    output: ParserOutput
    eligible: bool
    reason: str
    image_l1: torch.Tensor
    foreground_bce: torch.Tensor


@dataclass(frozen=True)
class OwnedControllability:
    record: BranchExperience
    source_rgb: torch.Tensor
    effect: torch.Tensor
    output: ParserOutput
    targets: ControllabilityTargets
    loss: torch.Tensor


def _serialized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        if not self._operation_lock.acquire(blocking=False):
            raise ValueError('Another pixel objective operation is in progress')
        try:return method(self,*args,**kwargs)
        finally:self._operation_lock.release()
    return call


class OwnedPixelObjectives:
    """One exclusive collector owner, fixed four-frame P/C and pooled C images.

Methods execute and consume one operation themselves: there is no intake of
caller-supplied records, anchors with substituted pixels, or old predictions.
Model weights and the collector must not be modified concurrently elsewhere.
Any step/branch failure stops this owner. Diagnostic/access errors alone do not.
Paid records surviving a processing failure stay
available through failure_records(); an unsuccessful take stays in collector.
This is an in-memory diagnostic contract, not durable training storage.
"""
    prediction_horizon_frames=4

    def __init__(self,collector,parser):
        self._operation_lock=Lock()
        if not isinstance(collector,OwnedRGBCollector) or not isinstance(parser,CounterfactualParser):
            raise ValueError('An exclusively owned RGB collector and own parser are required')
        self._collector,self._parser=collector,parser
        self._stopped=False
        self._operation=None
        self._records=None

    def _guard(self):
        if self._stopped:raise ValueError('Pixel objective owner stopped after an earlier failure or closure')

    @_serialized
    def current_anchor(self,worker=0):
        self._guard()
        return self._collector.current_anchor(worker)

    def _infer(self,pair,assumption):
        device=next(self._parser.parameters()).device
        previous=torch.tensor(pair[0],device=device).permute(2,0,1)[None]
        current=torch.tensor(pair[1],device=device).permute(2,0,1)[None]
        source=max_pool_rgb(previous,current)
        effect=torch.tensor(assumption,dtype=torch.float32,device=device)[None]
        output=self._parser(source,effect)
        # Check even heads not used by a particular objective. A finite loss
        # must not hide an already invalid parser state.
        for name in ('embedding','foreground_logits','displacement','controllability_logits','solidity_logits'):
            if not torch.isfinite(getattr(output,name)).all():
                raise ValueError('Nonfinite own parser output')
        return source,effect,output

    def _prediction(self,record):
        source,effect,output=self._infer(record.trace.start_pair,record.effect_assumption)
        length=len(record.trace.raw)
        if length<4:
            zero=output.displacement.reshape(-1)[0]*0+output.foreground_logits.reshape(-1)[0]*0
            reason='no_delivered_action_image' if length==0 else 'truncated_prediction_horizon'
            return OwnedPrediction(record,source,effect,output,False,reason,zero,zero)
        target=torch.tensor(record.trace.endpoint,device=source.device).permute(2,0,1)[None]
        terms=prediction_terms(source,target,output)
        if not torch.isfinite(terms['image_l1']) or not torch.isfinite(terms['foreground_bce']):
            raise ValueError('Nonfinite P loss; paid data retained but no eligible transition returned')
        return OwnedPrediction(record,source,effect,output,True,'complete_prediction_horizon',
                               terms['image_l1'],terms['foreground_bce'])

    @_serialized
    def step(self,actions,*,effects=None):
        """P from the actual departure after any unseen cold construction.

Construction/restore charges are not an observed duration. The record has
its delivered-image chronology; this module does not run I or infer deaths.
"""
        self._guard()
        try:
            self._operation=self._collector.step(actions,effects=effects)
            self._records=self._collector.take(self._operation)
            result=tuple(self._prediction(record) for record in self._records)
            self._records=self._operation=None
            return result
        except BaseException:
            self._stopped=True
            raise

    @_serialized
    def branch(self,action,*,expected_anchor,worker=0,frames=4,effect=None):
        """C at frame4 of one owned paid pair, including a longer pair's prefix.

Shorter common prefixes remain unknown. Frames beyond4 supply no extra C
sample in this draft. Source components/flow are recomputed with THIS effect.
"""
        self._guard()
        try:
            self._operation=self._collector.branch(action,expected_anchor=expected_anchor,
                                                   worker=worker,frames=frames,effect=effect)
            self._records=self._collector.take(self._operation)
            record,=self._records
            source,chosen,output=self._infer(record.anchor.pair,record.effect_assumption)
            def detached(name):return getattr(output,name).detach().cpu().numpy()[0]
            dense=DensePrediction(*(detached(name) for name in (
                'embedding','foreground_logits','displacement','controllability_logits','solidity_logits')))
            pixels=source[0].permute(1,2,0).cpu().numpy()
            own=extract_instances(dense,pixels)
            targets=component_targets(own,dense.displacement,record.intervention,record.control,
                                      horizon_frames=4,image_kind='pooled')
            loss=controllability_loss(output.controllability_logits[0,0],targets)
            if not torch.isfinite(loss):raise ValueError('Nonfinite C loss; paid pair retained')
            result=OwnedControllability(record,source,chosen,output,targets,loss)
            self._records=self._operation=None
            return result
        except BaseException:
            self._stopped=True
            raise

    @_serialized
    def failure_records(self):
        """Detached record copies only, with no stale graph or new paid command."""
        return copy.deepcopy(self._records)

    @_serialized
    def report(self,worker=0):
        """External accounting/status; never feed this dictionary to the parser."""
        return dict(objective_owner_stopped=self._stopped,processing_operation=self._operation,
                    records_retained=self._records is not None,collector=self._collector.report(worker))

    @_serialized
    def close(self):
        self._stopped=True
        self._collector.close()
