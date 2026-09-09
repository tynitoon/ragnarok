"""Own ordered pixels -> motion plus bounded collision priorities; no trainer.

Composition intentionally prevents old parser owners from accepting this new
output contract. No future image, mask, classical flow or label is an input.
"""
from dataclasses import dataclass

import torch
from torch import nn

from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import ParserOutput, _Block, max_pool_rgb


@dataclass(frozen=True)
class PriorityParserOutput:
    visual: ParserOutput
    log_priority: torch.Tensor


class PriorityMotionParser(nn.Module):
    """New explicit contract, not a subclass substitutable for an old owner.

The existing ordered model is initialized first, unchanged. A separate head
consumes the six ordered RGB channels, its32D appearance embedding and its own
two displacement channels. Effects enter priorities only through that predicted
displacement. The zero final layer initially gives uniform priorities exactly.
"""

    def __init__(self):
        super().__init__()
        self.visual = OrderedMotionParser()
        self.priority = nn.Sequential(_Block(40, 32), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.priority[-1].weight)
        nn.init.zeros_(self.priority[-1].bias)

    def forward(self, previous, current, effect=None):
        max_pool_rgb(previous, current)  # Validate the ordered uint8 pair.
        visual = self.visual(previous, current, effect)
        for name in ('embedding', 'foreground_logits', 'displacement',
                     'controllability_logits', 'solidity_logits'):
            value = getattr(visual, name)
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Finite float32 own visual heads required before priority prediction')
        ordered = torch.cat((previous, current), dim=1).float() / 255
        inputs = torch.cat((ordered, visual.embedding, visual.displacement), dim=1)
        logits = self.priority(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite own priority logits')
        bounded = 20 * torch.tanh(logits / 20)
        return PriorityParserOutput(visual=visual, log_priority=bounded)
