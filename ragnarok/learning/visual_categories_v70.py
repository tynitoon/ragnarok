"""Per-entry visual codebook from own instance descriptors, never a carried prior."""
from dataclasses import dataclass
import copy

import numpy as np

from ragnarok.learning.instance_tracker_v70 import TrackerFrame,TrackedObservation
from ragnarok.learning.visual_instances_v70 import VisualInstance


@dataclass(frozen=True)
class CategoryAssignment:
    track_id: int
    category_id: int
    created: bool


@dataclass
class _Category:
    center: np.ndarray
    observations: int


def visual_features(instance):
    if (not isinstance(instance,VisualInstance) or not isinstance(instance.descriptor,np.ndarray)
            or instance.descriptor.dtype!=np.float32 or instance.descriptor.shape!=(49,)
            or not np.isfinite(instance.descriptor).all()):
        raise ValueError('Own finite49-float visual instance required')
    values=instance.descriptor.astype(np.float64)
    if ((values[2:4]<=0).any() or (values[2:4]>1).any() or (np.abs(values[7:39])>1).any()
            or (values[41:49]<0).any() or (values[41:49]>1).any()
            or not np.isclose(values[41:49].sum(),1.,rtol=0,atol=1e-6)):
        raise ValueError('Own size fractions, appearance and normalized RGB histogram required')
    return np.concatenate((values[7:39],np.log(values[2:4]),values[41:49]))


class VisualCategoryBook:
    """Draft nearest-centroid Euclidean rule on [f32, log width/height, RGB8].

Create fresh at each game entry in every arm. This local codebook is NOT a
cross-game skill or semantic identity. Behaviour-based merges are not provided.
"""
    def __init__(self,*,distance_threshold=.6,max_categories=64,max_observations_per_frame=512):
        if not np.isfinite(distance_threshold) or not 0<distance_threshold<=100:
            raise ValueError('Finite positive visual distance threshold at most100 required')
        if type(max_categories) is not int or not 1<=max_categories<=64:
            raise ValueError('Declare category capacity in1..64')
        if type(max_observations_per_frame) is not int or not 1<=max_observations_per_frame<=4096:
            raise ValueError('Declare per-frame observed capacity in1..4096')
        self.distance_threshold=float(distance_threshold)
        self.max_categories=max_categories
        self.max_observations_per_frame=max_observations_per_frame
        self._frame=None
        self._categories=[]

    def observe(self,tracking):
        if (not isinstance(tracking,TrackerFrame) or type(tracking.frame) is not int
                or not 0<=tracking.frame<=2**53 or (self._frame is not None and tracking.frame<=self._frame)
                or len(tracking.observed)>self.max_observations_per_frame):
            raise ValueError('Increasing own observed frame within the declared capacity required')
        rows=[]
        ids=set();pixels=set()
        for row in tracking.observed:
            if (not isinstance(row,TrackedObservation) or type(row.track_id) is not int
                    or row.track_id<0 or row.track_id in ids):
                raise ValueError('Unique own observed track IDs required')
            features=visual_features(row.instance)
            source=row.instance.source_pixels
            if (not isinstance(source,np.ndarray) or source.ndim!=1 or not len(source)
                    or not np.issubdtype(source.dtype,np.integer) or (source<0).any()):
                raise ValueError('Nonempty own source pixel indices required')
            members=set(map(int,source))
            if len(members)!=len(source) or members&pixels:
                raise ValueError('Own observed instances must not duplicate pixels')
            pixels.update(members);ids.add(row.track_id)
            rows.append((int(source.min()),row.track_id,features))
        # Geometry order avoids dependence on caller/category/track numbering.
        rows.sort(key=lambda row:row[0])
        candidate=copy.deepcopy(self._categories)
        assignments=[]
        for _,track_id,features in rows:
            distances=np.array([np.linalg.norm(features-c.center) for c in candidate])
            nearest=int(distances.argmin()) if len(distances) else None
            created=bool(nearest is None or distances[nearest]>self.distance_threshold)
            if created:
                if len(candidate)==self.max_categories:
                    raise ValueError('Visual category capacity exceeded; no forced merge or eviction')
                nearest=len(candidate)
                candidate.append(_Category(features.copy(),1))
            else:
                category=candidate[nearest]
                count=category.observations+1
                if count>2**53:raise ValueError('Category observation count exceeds exact numeric range')
                category.center=category.center+(features-category.center)/count
                category.observations=count
            assignments.append(CategoryAssignment(track_id,nearest,created))
        result=tuple(assignments)
        self._categories,self._frame=candidate,tracking.frame
        return result

    def export_local(self):
        """Detached JSON-compatible local diagnostics, excluded from lineage export."""
        return dict(kind='own_per_entry_visual_codebook',carried_across_games=False,
                    frame=self._frame,distance_threshold=self.distance_threshold,
                    max_categories=self.max_categories,categories=[
                        dict(category_id=i,center=c.center.tolist(),observations=c.observations)
                        for i,c in enumerate(self._categories)])
