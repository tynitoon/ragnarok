"""Prospective motion head using the ordered pair of own observed RGB screens.

Appearance/foreground/control/solidity still use the pooled view. The extra
motion branch learns from six raw RGB channels, with no computed flow or labels.
This is a separate architecture candidate, not the parser of earlier results.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ragnarok.learning.parser_v70 import CounterfactualParser,ParserOutput,_Block,max_pool_rgb


class OrderedMotionParser(CounterfactualParser):
    """Explicit previous/current inputs; the former pooled-only API is refused."""
    def __init__(self):
        super().__init__()
        self.ordered_motion=_Block(6,32)
        self.displacement=nn.Sequential(nn.Conv2d(96,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,2,1))
        nn.init.normal_(self.displacement[-1].weight,std=1e-3)
        nn.init.zeros_(self.displacement[-1].bias)

    def forward(self,previous,current,effect=None):
        pooled=max_pool_rgb(previous,current)
        if pooled.shape[0]==0 or min(pooled.shape[-2:])<8:
            raise ValueError('Nonempty ordered BCHW RGB pairs of height/width at least8 required')
        rgb=pooled.float()/255
        if effect is None:effect=rgb.new_zeros((rgb.shape[0],12))
        if (not isinstance(effect,torch.Tensor) or effect.shape!=(rgb.shape[0],12)
                or not effect.is_floating_point() or effect.device!=rgb.device
                or not torch.isfinite(effect).all() or (effect<0).any() or (effect>1).any()):
            raise ValueError('Only finite Bx12 own effect hypotheses in[0,1] may condition motion')
        native=self.pyramid(rgb)
        half=self.pyramid(F.interpolate(rgb,size=(rgb.shape[-2]//2,rgb.shape[-1]//2),mode='area'))
        half=F.interpolate(half,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        features=self.fusion(torch.cat((native,half),dim=1))
        motion=self.ordered_motion(torch.cat((previous,current),dim=1).float()/255)
        context=self.effect(effect.to(dtype=features.dtype))[:,:,None,None].expand(-1,-1,*features.shape[-2:])
        return ParserOutput(F.normalize(self.embedding(features),dim=1,eps=1e-6),
            self.foreground(features),self.displacement(torch.cat((features,motion,context),dim=1)),
            self.controllability(features),self.solidity(features))
