"""Draft assembly of our pixel parser, tracker and own G/I targets for one stream.

No emulator calls, training loop or external labels. The caller owns provenance
of the two last raw frames and the clock measured in actual ordinary frames.
"""
from dataclasses import dataclass
import copy

import numpy as np
import torch

from ragnarok.learning.parser_v70 import CounterfactualParser,ParserOutput,max_pool_rgb
from ragnarok.learning.visual_instances_v70 import DensePrediction,extract_instances
from ragnarok.learning.bounded_tracker_v70 import BoundedInstanceTracker
from ragnarok.learning.instance_tracker_v70 import TrackerFrame
from ragnarok.learning.grouping_v70 import grouping_targets,grouping_loss,GroupingLoss
from ragnarok.learning.identity_v70 import IdentityObservation,IdentityLoss,identity_loss
from ragnarok.learning.pixel_prediction_v70 import prediction_terms


@dataclass(frozen=True)
class PerceptionFrame:
    stream_id: str
    epoch: int
    native_frame: int
    prediction_horizon_frames: int
    pooled_rgb: torch.Tensor
    effect: torch.Tensor
    output: ParserOutput
    tracking: TrackerFrame
    grouping: GroupingLoss
    identity: IdentityLoss


@dataclass(frozen=True)
class TransitionPrediction:
    eligible: bool
    actual_frames: int
    reason: str
    image_l1: torch.Tensor
    foreground_bce: torch.Tensor


@dataclass(frozen=True)
class _IdentityMemory:
    pooled_rgb: torch.Tensor
    tracking: TrackerFrame


class PerceptionStream:
    """One ordinary stream or one lab fork; never share a tracker between them.

The unconditioned architecture does not take a horizon feature, so this draft
fixes its predicted interval to4 actual frames. Truncated transitions do not
supply P supervision. I uses exact8-frame endpoints, not two arbitrary steps.
"""
    prediction_horizon_frames=4

    def __init__(self,parser,image_shape,*,stream_id,epoch=0,max_tracks=512,max_groups=512):
        if not isinstance(parser,CounterfactualParser):
            raise ValueError('Own counterfactual parser required')
        if not isinstance(stream_id,str) or not stream_id or type(epoch) is not int or epoch<0:
            raise ValueError('An explicit own stream and reset epoch are required')
        if type(max_groups) is not int or not 1<=max_groups<=4096:
            raise ValueError('Declare motion group capacity in1..4096')
        self.parser,self.stream_id,self.epoch=parser,stream_id,epoch
        self._tracker=BoundedInstanceTracker(image_shape,max_tracks=max_tracks)
        self.max_groups=max_groups
        self._history={}

    def observe(self,previous_rgb,current_rgb,*,native_frame,effect=None):
        """Observe without retaining autograd graphs or updating any model weights.

`effect` describes the chosen forthcoming four-frame action, not a raw button
number. Zero explicitly means unknown. A future collector must attest the
action, image alignment, stream and epoch; these arguments alone cannot do so.
"""
        shape=(*self._tracker.image_shape,3)
        for rgb in (previous_rgb,current_rgb):
            if not isinstance(rgb,np.ndarray) or rgb.shape!=shape or rgb.dtype!=np.uint8:
                raise ValueError('The last two own raw uint8 RGB frames must match the stream shape')
        if effect is None: effect=np.zeros(12,dtype=np.float32)
        if (not isinstance(effect,np.ndarray) or effect.shape!=(12,)
                or not np.issubdtype(effect.dtype,np.floating) or not np.isfinite(effect).all()
                or (effect<0).any() or (effect>1).any()):
            raise ValueError('Own12-effect features in[0,1] required; raw button IDs are not features')
        device=next(self.parser.parameters()).device
        def tensor(rgb):
            return torch.tensor(rgb,device=device).permute(2,0,1)[None]
        pooled=max_pool_rgb(tensor(previous_rgb),tensor(current_rgb))
        chosen=torch.tensor(effect,dtype=torch.float32,device=device)[None]
        output=self.parser(pooled,chosen)
        def detached(name): return getattr(output,name).detach().cpu().numpy()[0]
        dense=DensePrediction(*(detached(name) for name in (
            'embedding','foreground_logits','displacement','controllability_logits','solidity_logits')))
        own=extract_instances(dense,pooled[0].permute(1,2,0).cpu().numpy())
        # All changes, including identity history, are prepared before commit.
        candidate=self._tracker.fork()
        tracked=candidate.update(own,frame=native_frame,prediction_horizon_frames=4)
        targets=grouping_targets(dense.foreground_logits[0],dense.displacement,
                                  prediction_horizon_frames=4,max_groups=self.max_groups)
        grouping=grouping_loss(output.embedding[0],targets)
        previous=self._history.get(native_frame-8)
        if previous is None:
            identity=IdentityLoss(output.embedding.sum()*0,0,0,0,len(tracked.observed))
        else:
            # Recompute f with current weights. Saved observations contain only
            # uint8 pixels and own detached tracking, not stale computation graphs.
            earlier=self.parser(previous.pooled_rgb).embedding[0]
            identity=identity_loss(
                IdentityObservation(self.stream_id,self.epoch,previous.tracking,earlier),
                IdentityObservation(self.stream_id,self.epoch,tracked,output.embedding[0]))
        history={t:value for t,value in self._history.items() if native_frame-t<=8}
        history[native_frame]=_IdentityMemory(pooled.detach().clone(),copy.deepcopy(tracked))
        result=PerceptionFrame(self.stream_id,self.epoch,native_frame,4,pooled,chosen,output,
                               tracked,grouping,identity)
        self._tracker,self._history=candidate,history
        return result


def transition_prediction(source,following):
    """P on a complete four-frame transition only; incomplete futures are unknown."""
    if (not isinstance(source,PerceptionFrame) or not isinstance(following,PerceptionFrame)
            or source.stream_id!=following.stream_id or source.epoch!=following.epoch
            or source.prediction_horizon_frames!=4 or following.prediction_horizon_frames!=4
            or source.pooled_rgb.shape!=following.pooled_rgb.shape
            or source.pooled_rgb.device!=following.pooled_rgb.device):
        raise ValueError('P requires aligned endpoints from the same own stream and epoch')
    elapsed=following.native_frame-source.native_frame
    if type(elapsed) is not int or not 1<=elapsed<=4:
        raise ValueError('Expected one ordinary transition of1..4 actual frames')
    if (not torch.isfinite(source.output.displacement).all()
            or not torch.isfinite(source.output.foreground_logits).all()):
        raise ValueError('Source prediction fields must be finite before any P target')
    if elapsed<4:
        # Indexing preserves a zero gradient to each tensor without an overflowing
        # reduction of individually finite values before multiplying by zero.
        zero=(source.output.displacement.reshape(-1)[0]*0
              +source.output.foreground_logits.reshape(-1)[0]*0)
        return TransitionPrediction(False,elapsed,'truncated_prediction_horizon',zero,zero)
    result=prediction_terms(source.pooled_rgb,following.pooled_rgb,source.output)
    if not torch.isfinite(result['image_l1']) or not torch.isfinite(result['foreground_bce']):
        raise ValueError('Nonfinite P loss; no eligible training transition may be returned')
    return TransitionPrediction(True,elapsed,'complete_prediction_horizon',result['image_l1'],result['foreground_bce'])
