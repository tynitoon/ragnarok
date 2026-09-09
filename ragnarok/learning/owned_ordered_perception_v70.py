"""Explicit ordered-parser adapters around the reviewed owned P/C/G/I flow.

Paid command, record retention, lock and stop semantics are inherited unchanged.
Only model intake and temporal-stream construction differ from the old owners.
"""
from threading import Lock

import torch

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.ordered_perception_stream_v70 import OrderedPerceptionStream
from ragnarok.learning.parser_v70 import max_pool_rgb
from ragnarok.learning.owned_pixel_objectives_v70 import OwnedPixelObjectives
from ragnarok.learning.owned_perception_session_v70 import OwnedPerceptionSession


def _contract(collector,parser):
    if not isinstance(collector,OwnedRGBCollector) or type(parser) is not OrderedMotionParser:
        raise ValueError('Exclusively owned collector and explicit ordered-motion parser required before commands')
    parameters=list(parser.parameters())
    if (not parameters or any(p.dtype!=torch.float32 or p.device!=parameters[0].device
                              or not torch.isfinite(p).all() for p in parameters)):
        raise ValueError('Finite float32 ordered parser on one device required before commands')
    for i in range(collector._workers):
        if min(collector.current_anchor(i).pair.shape[1:3])<8:
            raise ValueError('Every own RGB anchor must support the ordered parser before commands')


class OwnedOrderedPixelObjectives(OwnedPixelObjectives):
    """The old P/C horizons and losses, with an explicit ordered source pair."""
    def __init__(self,collector,parser):
        _contract(collector,parser)
        super().__init__(collector,parser)

    def _infer(self,pair,assumption):
        device=next(self._parser.parameters()).device
        def tensor(rgb):return torch.tensor(rgb,device=device).permute(2,0,1)[None]
        previous,current=tensor(pair[0]),tensor(pair[1])
        source=max_pool_rgb(previous,current)
        effect=torch.tensor(assumption,dtype=torch.float32,device=device)[None]
        output=self._parser(previous,current,effect)
        for name in ('embedding','foreground_logits','displacement','controllability_logits','solidity_logits'):
            if not torch.isfinite(getattr(output,name)).all():raise ValueError('Nonfinite own ordered parser output')
        return source,effect,output


class OwnedOrderedPerceptionSession(OwnedPerceptionSession):
    """Same owned live clock/branch isolation, explicitly bound to ordered intake.

As for the initial owner, neither collector nor model may be accessed or changed
concurrently elsewhere. Constructor validation and initial inference perform no
new action; caller still owns cleanup if construction fails.
"""
    def __init__(self,collector,parser,*,workers,max_tracks=512,max_groups=512):
        _contract(collector,parser)
        if type(workers) is not int or workers!=collector._workers:
            raise ValueError('Explicit worker count must match the exclusively owned collector')
        self._operation_lock=Lock()
        self._objectives=OwnedOrderedPixelObjectives(collector,parser)
        self._parser,self._workers=parser,workers
        self._max_tracks,self._max_groups=max_tracks,max_groups
        self._stopped=False;self._pending=None
        streams=[]
        for i in range(workers):
            anchor=collector.current_anchor(i)
            stream=self._new_stream(anchor.pair.shape[1:3],f'{anchor.token[0]}/worker/{i}',anchor.epoch)
            stream.observe(*anchor.pair,native_frame=anchor.frame)
            streams.append(stream)
        self._streams=streams

    def _new_stream(self,shape,stream_id,epoch):
        return OrderedPerceptionStream(self._parser,shape,stream_id=stream_id,epoch=epoch,
            max_tracks=self._max_tracks,max_groups=self._max_groups)
