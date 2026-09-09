"""Prospective temporal stream preserving both own RGB screens for motion.

No emulator, optimizer or oracle. This separate intake explicitly requires the
ordered parser; it does not change the architecture attached to earlier runs.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch

from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import max_pool_rgb
from ragnarok.learning.perception_stream_v70 import PerceptionStream,PerceptionFrame
from ragnarok.learning.visual_instances_v70 import DensePrediction,extract_instances
from ragnarok.learning.instance_tracker_v70 import TrackerFrame
from ragnarok.learning.grouping_v70 import grouping_targets,grouping_loss
from ragnarok.learning.identity_v70 import IdentityObservation,IdentityLoss,identity_loss


@dataclass(frozen=True)
class _OrderedIdentityMemory:
    stream_id: str
    epoch: int
    previous_rgb: torch.Tensor
    current_rgb: torch.Tensor
    tracking: TrackerFrame


class OrderedPerceptionStream(PerceptionStream):
    """Explicit ordered-input contract with the same four-frame P/eight-frame I.

The caller still attests provenance and delivered-image time. Trackers remain
private to one stream/epoch. Stored history contains two detached uint8 screens
per timestamp and detached own tracking, never graphs or stale embeddings.
"""
    def __init__(self,parser,image_shape,*,stream_id,epoch=0,max_tracks=512,max_groups=512):
        if type(parser) is not OrderedMotionParser:
            raise ValueError('This intake requires the explicit ordered-motion parser')
        if len(image_shape)!=2 or any(type(x) is not int or x<8 for x in image_shape):
            raise ValueError('Ordered parser spatial dimensions must both be at least8')
        super().__init__(parser,image_shape,stream_id=stream_id,epoch=epoch,
                         max_tracks=max_tracks,max_groups=max_groups)

    def observe(self,previous_rgb,current_rgb,*,native_frame,effect=None):
        shape=(*self._tracker.image_shape,3)
        for rgb in (previous_rgb,current_rgb):
            if not isinstance(rgb,np.ndarray) or rgb.shape!=shape or rgb.dtype!=np.uint8:
                raise ValueError('Own ordered raw uint8 RGB screens must match the stream shape')
        if effect is None:effect=np.zeros(12,np.float32)
        if (not isinstance(effect,np.ndarray) or effect.shape!=(12,)
                or not np.issubdtype(effect.dtype,np.floating) or not np.isfinite(effect).all()
                or (effect<0).any() or (effect>1).any()):
            raise ValueError('Only own12-effect hypotheses in[0,1] may condition displacement')
        device=next(self.parser.parameters()).device
        def tensor(rgb):return torch.tensor(rgb,device=device).permute(2,0,1)[None]
        previous,current=tensor(previous_rgb),tensor(current_rgb)
        pooled=max_pool_rgb(previous,current)
        chosen=torch.tensor(effect,dtype=torch.float32,device=device)[None]
        output=self.parser(previous,current,chosen)
        names=('embedding','foreground_logits','displacement','controllability_logits','solidity_logits')
        if any(not torch.isfinite(getattr(output,name)).all() for name in names):
            raise ValueError('Every ordered parser head must be finite before temporal targets')
        dense=DensePrediction(*(getattr(output,name).detach().cpu().numpy()[0] for name in names))
        own=extract_instances(dense,pooled[0].permute(1,2,0).cpu().numpy())
        candidate=self._tracker.fork()
        tracked=candidate.update(own,frame=native_frame,prediction_horizon_frames=4)
        targets=grouping_targets(dense.foreground_logits[0],dense.displacement,
                                  prediction_horizon_frames=4,max_groups=self.max_groups)
        grouping=grouping_loss(output.embedding[0],targets)
        earlier=self._history.get(native_frame-8)
        if earlier is None:
            identity=IdentityLoss(output.embedding.sum()*0,0,0,0,len(tracked.observed))
        else:
            # Use exactly the earlier pair with CURRENT weights. Only f enters
            # I; recomputing the full parser introduces no effect/flow target.
            embedding=self.parser(earlier.previous_rgb,earlier.current_rgb).embedding[0]
            identity=identity_loss(
                IdentityObservation(earlier.stream_id,earlier.epoch,earlier.tracking,embedding),
                IdentityObservation(self.stream_id,self.epoch,tracked,output.embedding[0]))
        history={t:value for t,value in self._history.items() if native_frame-t<=8}
        history[native_frame]=_OrderedIdentityMemory(self.stream_id,self.epoch,
            previous.detach().clone(),current.detach().clone(),copy.deepcopy(tracked))
        result=PerceptionFrame(self.stream_id,self.epoch,native_frame,4,pooled,chosen,output,
                               tracked,grouping,identity)
        self._tracker,self._history=candidate,history
        return result
