"""Draft C targets from own paid branch RGB and own predicted future positions.

An absent future, an out-of-frame prediction or absent own motion is unknown,
with zero training weight. No RAM/control truth or classical flow is used.
"""
from dataclasses import dataclass

import numpy as np

from ragnarok.environments.rgb_trace_difference_v70 import difference
from ragnarok.learning.visual_instances_v70 import VisualInstance


@dataclass(frozen=True)
class ControllabilityTargets:
    values: np.ndarray
    weights: np.ndarray
    instances: tuple[dict,...]
    horizon_frames: int
    common_prefix_frames: int
    image_kind: str


def component_targets(instances, displacement, intervention, control, *, horizon_frames, image_kind, motion_threshold=.25):
    """Project source pixels with the model's forward field for this exact horizon.

Branches must be the two outputs of ONE paid branch_pair_trace call. Pixel
arrays cannot certify that common-anchor provenance; the collector must do so.
"""
    if type(horizon_frames) is not int or not 1<=horizon_frames<=64:
        raise ValueError('Declare the predicted horizon in1..64 actual branch frames')
    if not np.isfinite(motion_threshold) or motion_threshold<0:
        raise ValueError('Finite nonnegative own-motion threshold required')
    observed=difference(intervention,control,image_kind=image_kind)
    height,width=observed.first_changed_frame.shape
    if (not isinstance(displacement,np.ndarray) or displacement.shape!=(2,height,width)
            or not np.issubdtype(displacement.dtype,np.floating) or not np.isfinite(displacement).all()
            or (np.abs(displacement)>np.finfo(np.float32).max).any()):
        raise ValueError('Own finite float32-representable forward displacement must match the image')
    instances=list(instances)
    occupied=set()
    identities=set()
    for instance in instances:
        if (not isinstance(instance,VisualInstance) or type(instance.frame_index) is not int
                or instance.frame_index<0 or instance.frame_index in identities):
            raise ValueError('Unique own-instance frame indices required')
        identities.add(instance.frame_index)
        pixels=instance.source_pixels
        if (not isinstance(pixels,np.ndarray) or pixels.ndim!=1 or not len(pixels)
                or not np.issubdtype(pixels.dtype,np.integer) or (pixels<0).any() or (pixels>=height*width).any()):
            raise ValueError('Own source pixels must belong to this frame')
        members=set(map(int,pixels))
        if len(members)!=len(pixels) or occupied&members: raise ValueError('Duplicate own source pixels')
        occupied.update(members)
    flow=displacement.astype(np.float64).reshape(2,-1)
    values=np.zeros(height*width,dtype=np.float32)
    weights=np.zeros_like(values)
    common=len(observed.changed)
    rows=[]
    changed=None if common<horizon_frames else observed.changed[horizon_frames-1]
    for instance in instances:
        pixels=instance.source_pixels
        row=dict(frame_index=instance.frame_index,source_pixels=len(pixels),eligible=False,target=None)
        if changed is None:
            row['reason']='incomplete_common_horizon'
        elif np.linalg.norm(flow[:,pixels],axis=0).mean()<=motion_threshold:
            row['reason']='no_own_predicted_motion'
        else:
            y,x=np.divmod(pixels,width)
            x=x+flow[0,pixels]
            y=y+flow[1,pixels]
            # Clip only for safe integer conversion; outside weights remain zero.
            x=np.clip(x,-2,width+1);y=np.clip(y,-2,height+1)
            x0,y0=np.floor(x),np.floor(y)
            dx,dy=x-x0,y-y0
            evidence=np.zeros(len(pixels),dtype=np.float64)
            coverage=np.zeros_like(evidence)
            for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                xx,yy=x0+ox,y0+oy
                valid=(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
                ix,iy=np.clip(xx,0,width-1).astype(np.int64),np.clip(yy,0,height-1).astype(np.int64)
                coverage+=weight*valid
                evidence+=weight*valid*changed[iy,ix]
            visible=coverage>0
            if not visible.any():
                row['reason']='predicted_support_outside_frame'
            else:
                target=float(np.mean(evidence[visible]/coverage[visible]))
                selected=pixels[visible]
                values[selected]=target
                weights[selected]=1
                row.update(eligible=True,target=target,projected_source_pixels=int(visible.sum()),
                           reason='own_motion_and_complete_observation')
        rows.append(row)
    values=values.reshape(height,width);weights=weights.reshape(height,width)
    values.flags.writeable=weights.flags.writeable=False
    return ControllabilityTargets(values,weights,tuple(rows),horizon_frames,common,image_kind)


def controllability_loss(logits, targets):
    """Unbatched own C-head loss. Unknown components contribute exactly no gradient."""
    import torch
    from torch.nn import functional as F
    if (not isinstance(targets,ControllabilityTargets) or not isinstance(logits,torch.Tensor)
            or logits.shape!=targets.values.shape or not logits.is_floating_point()
            or targets.weights.shape!=targets.values.shape or not torch.isfinite(logits).all()
            or (logits.abs()>torch.finfo(torch.float32).max).any()):
        raise ValueError('Finite source-aligned C logits and own branch targets required')
    values=torch.tensor(targets.values,dtype=torch.float32,device=logits.device)
    weights=torch.tensor(targets.weights,dtype=torch.float32,device=logits.device)
    if (not torch.isfinite(values).all() or not torch.isfinite(weights).all()
            or (values<0).any() or (values>1).any() or (weights<0).any() or (weights>1).any()):
        raise ValueError('Targets and weights must be finite in[0,1]')
    loss=F.binary_cross_entropy_with_logits(logits.float(),values,reduction='none')
    # Finite float32 losses can overflow a float32 sum before division.
    return ((loss*weights).sum(dtype=torch.float64)/weights.sum(dtype=torch.float64).clamp_min(1)).float()
