"""Eight-frame contrastive identity from visible observations of our own tracker."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ragnarok.learning.grouping_v70 import validate_embedding
from ragnarok.learning.instance_tracker_v70 import TrackerFrame,TrackedObservation


@dataclass(frozen=True)
class IdentityObservation:
    stream_id: str  # Collector-owned; distinct for each live environment/lab fork.
    epoch: int  # Changes when own tracking memory is reset, even if IDs restart.
    tracking: TrackerFrame
    embedding: torch.Tensor  # Recomputed own head with gradient, not saved descriptors.


@dataclass(frozen=True)
class IdentityLoss:
    loss: torch.Tensor
    shared_visible_ids: int
    supervised_anchors: int
    previous_candidates: int
    current_candidates: int


def _visible_features(observation):
    if (not isinstance(observation,IdentityObservation) or not isinstance(observation.stream_id,str)
            or not observation.stream_id or type(observation.epoch) is not int or observation.epoch<0
            or not isinstance(observation.tracking,TrackerFrame)
            or type(observation.tracking.frame) is not int or not 0<=observation.tracking.frame<=2**53):
        raise ValueError('Own stream, reset epoch and tracked frame are required')
    features=validate_embedding(observation.embedding)
    height,width=features.shape[1:]
    occupied=set()
    descriptors={}
    if len(observation.tracking.observed)>4096:
        raise ValueError('At most4096 observed instances per identity frame')
    for row in observation.tracking.observed:
        if (not isinstance(row,TrackedObservation) or type(row.track_id) is not int
                or row.track_id<0 or row.track_id in descriptors):
            raise ValueError('Unique nonnegative own track IDs required')
        pixels=row.instance.source_pixels
        if (not isinstance(pixels,np.ndarray) or pixels.ndim!=1 or not len(pixels)
                or not np.issubdtype(pixels.dtype,np.integer) or (pixels<0).any() or (pixels>=height*width).any()):
            raise ValueError('Observed own source pixels must match this embedding frame')
        members=set(map(int,pixels))
        if len(members)!=len(pixels) or occupied&members:
            raise ValueError('Identity observations may not duplicate source pixels')
        occupied.update(members)
        index=torch.tensor(pixels.astype(np.int64),device=features.device)
        descriptors[row.track_id]=features.reshape(32,-1)[:,index].mean(dim=1)
    ids=sorted(descriptors)
    matrix=torch.stack([descriptors[i] for i in ids]) if ids else features.new_empty((0,32))
    return features,ids,F.normalize(matrix,dim=1,eps=1e-6)


def identity_loss(previous,current,*,temperature=.1):
    """Symmetric InfoNCE, exact lag8; coasting is never a visible positive/negative.

Same numeric ID in a different stream/reset epoch is not a correspondence.
These tokens and endpoint alignment must be enforced by the future collector.
"""
    left,left_ids,left_features=_visible_features(previous)
    right,right_ids,right_features=_visible_features(current)
    if (previous.stream_id!=current.stream_id or previous.epoch!=current.epoch
            or current.tracking.frame-previous.tracking.frame!=8):
        raise ValueError('Identity requires the same stream/reset epoch at exactly8 frames lag')
    if left.device!=right.device or left.shape!=right.shape:
        raise ValueError('Identity endpoint embeddings must share device and image shape')
    if not np.isfinite(temperature) or not 1e-4<=temperature<=1:
        raise ValueError('Finite contrastive temperature in[1e-4,1] required')
    shared=sorted(set(left_ids)&set(right_ids))
    zero=(left.sum()+right.sum())*0
    terms=[]
    anchors=0
    if shared:
        left_positions=torch.tensor([left_ids.index(i) for i in shared],device=left.device)
        right_positions=torch.tensor([right_ids.index(i) for i in shared],device=right.device)
        # A one-candidate softmax has no contrastive evidence, so that direction
        # is explicitly unsupervised, not counted as a successful identity target.
        if len(left_ids)>=2:
            logits=right_features[right_positions]@left_features.T/temperature
            terms.append(F.cross_entropy(logits,left_positions))
            anchors+=len(shared)
        if len(right_ids)>=2:
            logits=left_features[left_positions]@right_features.T/temperature
            terms.append(F.cross_entropy(logits,right_positions))
            anchors+=len(shared)
    loss=torch.stack(terms).mean() if terms else zero
    return IdentityLoss(loss,len(shared),anchors,len(left_ids),len(right_ids))
