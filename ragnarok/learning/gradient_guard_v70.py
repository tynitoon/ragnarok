"""Prospective float32 gradient admission before an Adam update, not a trainer.

No backward or optimizer step is performed here. All candidate gradients are
prepared before assignment; weights and optimizer moments are never modified.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GradientAdmission:
    tensors: int
    elements: int
    global_norm_before: float
    global_norm_after: float
    max_norm: float
    scale: float
    clipped: bool


def _scaled_gradient(gradient,scale):
    # Float64 multiplication avoids losing a tiny scaling factor on conversion
    # to float32 before it meets an otherwise representable extreme gradient.
    return (gradient.detach().double()*scale).to(dtype=torch.float32)


def admit_adam_gradients(model,*,max_norm=1.0):
    """Validate then globally clip dense float32 gradients using float64 sums.

The norm1 default is a prospective optimization choice, not a frozen ARC4 gate
or proof of stable learning. A failure must prevent the caller's optimizer step.
The caller must exclusively own model/gradients throughout this call and step.
An all-zero admitted gradient does not certify that a useful target existed.
"""
    if not isinstance(model,torch.nn.Module):raise ValueError('An own torch model is required')
    if type(max_norm) not in (float,int) or not math.isfinite(max_norm) or not 0<max_norm<=1:
        raise ValueError('Declare a finite positive global norm bound at most1 for this Adam draft')
    gradients=[];squares=[]
    for parameter in model.parameters():
        if parameter.dtype!=torch.float32 or not torch.isfinite(parameter).all():
            raise ValueError('Finite float32 model parameters required before Adam admission')
        gradient=parameter.grad
        if gradient is None:continue
        if (gradient.dtype!=torch.float32 or gradient.layout!=torch.strided
                or gradient.device!=parameter.device or not torch.isfinite(gradient).all()):
            raise ValueError('Finite dense float32 gradients required; do not update after rejection')
        gradients.append((parameter,gradient))
        squares.append(float(gradient.detach().double().square().sum()))
    if not gradients:raise ValueError('No gradient tensors supplied to Adam admission')
    norm=math.sqrt(math.fsum(squares))
    if not math.isfinite(norm):raise ValueError('Nonfinite global gradient norm')
    scale=max_norm/norm if norm>max_norm else 1.0
    candidates=[(parameter,_scaled_gradient(gradient,scale)) for parameter,gradient in gradients]
    after=math.sqrt(math.fsum(float(gradient.double().square().sum()) for _,gradient in candidates))
    if not math.isfinite(after) or after>max_norm*(1+4*torch.finfo(torch.float32).eps):
        raise ValueError('Candidate gradients exceed the declared norm bound')
    # Allocate even the receipt before assigning the prepared gradients.
    result=GradientAdmission(len(gradients),sum(g.numel() for _,g in gradients),norm,after,
                             float(max_norm),scale,norm>max_norm)
    for parameter,gradient in candidates:parameter.grad=gradient
    return result
