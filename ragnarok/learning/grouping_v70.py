"""Draft common-fate targets from the parser's own foreground and motion only."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GroupingTargets:
    labels: np.ndarray  # H,W; -1 is unknown/background, 0..count-1 own groups.
    count: int
    prediction_horizon_frames: int
    motion_tolerance_pixels: float


@dataclass(frozen=True)
class GroupingLoss:
    pull: torch.Tensor
    separation: torch.Tensor
    groups: int
    foreground_pixels: int


def validate_embedding(embedding):
    if (not isinstance(embedding,torch.Tensor) or embedding.ndim!=3 or embedding.shape[0]!=32
            or min(embedding.shape[1:])<1 or not embedding.is_floating_point()
            or not torch.isfinite(embedding).all() or (embedding.abs()>1).any()):
        raise ValueError('Own finite32-channel embedding with components in[-1,1] required')
    return embedding.float()


def grouping_targets(foreground_logits,displacement,*,prediction_horizon_frames,
                     motion_tolerance_pixels=.5,max_groups=512):
    """Four-neighbour connectedness through locally agreeing forward displacement.

Targets are detached: the loss trains f, not its own m/d target generators.
Agreement is local, so gradual changes can chain across a group's full extent.
"""
    if (not isinstance(foreground_logits,np.ndarray) or foreground_logits.ndim!=2
            or min(foreground_logits.shape)<1 or not np.issubdtype(foreground_logits.dtype,np.floating)
            or not np.isfinite(foreground_logits).all()):
        raise ValueError('Finite own H,W foreground logits required')
    height,width=foreground_logits.shape
    if (not isinstance(displacement,np.ndarray) or displacement.shape!=(2,height,width)
            or not np.issubdtype(displacement.dtype,np.floating) or not np.isfinite(displacement).all()
            or (np.abs(displacement)>np.finfo(np.float32).max).any()):
        raise ValueError('Own float32-representable forward displacement required')
    if type(prediction_horizon_frames) is not int or not 1<=prediction_horizon_frames<=64:
        raise ValueError('Declare the own displacement horizon in1..64 frames')
    if (not np.isfinite(motion_tolerance_pixels) or motion_tolerance_pixels<0
            or type(max_groups) is not int or not 1<=max_groups<=4096):
        raise ValueError('Nonnegative finite motion tolerance and group capacity1..4096 required')
    flow=displacement.astype(np.float64)
    foreground=foreground_logits>0
    horizontal=np.linalg.norm(flow[:,:,1:]-flow[:,:,:-1],axis=0)<=motion_tolerance_pixels
    vertical=np.linalg.norm(flow[:,1:,:]-flow[:,:-1,:],axis=0)<=motion_tolerance_pixels
    labels=np.full((height,width),-1,dtype=np.int32)
    count=0
    for start in np.flatnonzero(foreground):
        y,x=divmod(int(start),width)
        if labels[y,x]>=0: continue
        if count==max_groups:
            raise ValueError('Own motion group capacity exceeded; no silent group truncation')
        pending=[(y,x)]
        labels[y,x]=count
        while pending:
            yy,xx=pending.pop()
            neighbours=[]
            if xx>0 and horizontal[yy,xx-1]: neighbours.append((yy,xx-1))
            if xx+1<width and horizontal[yy,xx]: neighbours.append((yy,xx+1))
            if yy>0 and vertical[yy-1,xx]: neighbours.append((yy-1,xx))
            if yy+1<height and vertical[yy,xx]: neighbours.append((yy+1,xx))
            for ny,nx in neighbours:
                if foreground[ny,nx] and labels[ny,nx]<0:
                    labels[ny,nx]=count
                    pending.append((ny,nx))
        count+=1
    labels.flags.writeable=False
    return GroupingTargets(labels,count,prediction_horizon_frames,float(motion_tolerance_pixels))


def grouping_loss(embedding,targets,*,margin=1.):
    """Equal-weight group variance and pairwise squared hinge, returned separately."""
    features=validate_embedding(embedding)
    if (not isinstance(targets,GroupingTargets) or not isinstance(targets.labels,np.ndarray)
            or targets.labels.shape!=features.shape[1:]
            or not np.issubdtype(targets.labels.dtype,np.integer)
            or type(targets.count) is not int or not 0<=targets.count<=4096
            or not np.isfinite(margin) or not 0<margin<=2):
        raise ValueError('Own aligned grouping targets and finite margin in(0,2] required')
    labels=targets.labels.reshape(-1)
    if ((labels<-1).any() or (labels>=targets.count).any()
            or not np.array_equal(np.unique(labels[labels>=0]),np.arange(targets.count))):
        raise ValueError('Group labels must be contiguous with -1 as the only unknown')
    selected=np.flatnonzero(labels>=0)
    zero=features.sum()*0
    if not len(selected): return GroupingLoss(zero,zero,0,0)
    indices=torch.tensor(selected,device=features.device)
    groups=torch.tensor(labels[selected],dtype=torch.long,device=features.device)
    pixels=features.reshape(32,-1)[:,indices].T
    counts=torch.bincount(groups,minlength=targets.count).to(pixels.dtype)
    means=pixels.new_zeros((targets.count,32)).index_add(0,groups,pixels)/counts[:,None]
    squared=(pixels-means[groups]).square().sum(dim=1)
    per_group=squared.new_zeros(targets.count).index_add(0,groups,squared)/counts
    separation=F.relu(margin-torch.pdist(means)).square().mean() if targets.count>1 else zero
    return GroupingLoss(per_group.mean(),separation,targets.count,len(selected))
