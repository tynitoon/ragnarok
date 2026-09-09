"""Draft counterfactual visual parser: pixels in, learned dense heads out.

No environment, scorer, reference extractor, palette or object-label imports.
There is no training runner or pretrained checkpoint in this module.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ParserOutput:
    embedding: torch.Tensor
    foreground_logits: torch.Tensor
    displacement: torch.Tensor
    controllability_logits: torch.Tensor
    solidity_logits: torch.Tensor


def max_pool_rgb(previous, current):
    """G7 only: the actual last two uint8 RGB frames, supplied by the caller."""
    if (not isinstance(previous,torch.Tensor) or not isinstance(current,torch.Tensor)
            or previous.dtype != torch.uint8 or current.dtype != torch.uint8
            or previous.ndim != 4 or previous.shape != current.shape or previous.shape[1] != 3
            or previous.device != current.device):
        raise ValueError('Two equal BCHW uint8 RGB tensors on one device required')
    return torch.maximum(previous,current)


class _Block(nn.Sequential):
    def __init__(self, inputs, outputs, *, stride=1):
        super().__init__(nn.Conv2d(inputs,outputs,3,stride=stride,padding=1,bias=False),
                         nn.GroupNorm(8,outputs),nn.SiLU(),
                         nn.Conv2d(outputs,outputs,3,padding=1,bias=False),
                         nn.GroupNorm(8,outputs),nn.SiLU())


class _SharedPyramidUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1,self.down2,self.down3 = _Block(3,32,stride=2),_Block(32,64,stride=2),_Block(64,96,stride=2)
        self.up2,self.up1,self.up0 = _Block(160,64),_Block(96,32),_Block(35,32)

    def forward(self, rgb):
        first=self.down1(rgb)
        second=self.down2(first)
        low=self.down3(second)
        def join(value,skip):
            return torch.cat((F.interpolate(value,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),dim=1)
        value=self.up2(join(low,second))
        value=self.up1(join(value,first))
        return self.up0(join(value,rgb))


class CounterfactualParser(nn.Module):
    """Shared native/half-scale network, with twelve fixed semantic effect slots.

The effect vector conditions only displacement. All-zero means unknown effect;
mapping babbling buttons to this vocabulary is explicitly not solved here.
Displacement is forward motion in input pixels, x then y, per observed interval.
"""
    def __init__(self):
        super().__init__()
        self.pyramid=_SharedPyramidUNet()
        self.fusion=nn.Sequential(nn.Conv2d(64,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.embedding=nn.Conv2d(32,32,1)
        self.foreground=nn.Conv2d(32,1,1)
        self.controllability=nn.Conv2d(32,1,1)
        self.solidity=nn.Conv2d(32,1,1)
        self.effect=nn.Sequential(nn.Linear(12,32),nn.SiLU())
        self.displacement=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,2,1))
        nn.init.normal_(self.displacement[-1].weight,std=1e-3)
        nn.init.zeros_(self.displacement[-1].bias)

    def forward(self, pooled_rgb, effect=None):
        if (not isinstance(pooled_rgb,torch.Tensor) or pooled_rgb.dtype != torch.uint8
                or pooled_rgb.ndim != 4 or pooled_rgb.shape[1] != 3 or pooled_rgb.shape[0] == 0
                or min(pooled_rgb.shape[-2:]) < 8):
            raise ValueError('Expected nonempty BCHW uint8 pooled RGB with height/width at least8')
        rgb=pooled_rgb.float()/255
        if effect is None:
            effect=rgb.new_zeros((rgb.shape[0],12))
        if (not isinstance(effect,torch.Tensor) or effect.shape != (rgb.shape[0],12)
                or not effect.is_floating_point() or effect.device != rgb.device
                or not torch.isfinite(effect).all()):
            raise ValueError('Expected finite Bx12 effect features on the image device')
        native=self.pyramid(rgb)
        half=self.pyramid(F.interpolate(rgb,size=(rgb.shape[-2]//2,rgb.shape[-1]//2),mode='area'))
        half=F.interpolate(half,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        features=self.fusion(torch.cat((native,half),dim=1))
        context=self.effect(effect.to(dtype=features.dtype))[:,:,None,None].expand(-1,-1,*features.shape[-2:])
        return ParserOutput(F.normalize(self.embedding(features),dim=1,eps=1e-6),
            self.foreground(features),self.displacement(torch.cat((features,context),dim=1)),
            self.controllability(features),self.solidity(features))
