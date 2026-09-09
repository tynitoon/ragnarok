"""Own-prediction tracking; visible observations remain distinct from coasting.

The caller must supply an observation clock and the displacement prediction's
horizon in the same frame unit. This module never queries emulator time/state.
"""
from collections import deque
from dataclasses import dataclass
import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

from ragnarok.learning.visual_instances_v70 import VisualInstance


@dataclass(frozen=True)
class TrackedObservation:
    track_id: int
    instance: VisualInstance
    first_seen: int
    reappeared: bool


@dataclass(frozen=True)
class CoastedTrack:
    track_id: int
    predicted_center: tuple[float,float]  # Screen fractions; may leave the screen.
    missing_frames: int


@dataclass(frozen=True)
class TrackerFrame:
    frame: int
    observed: tuple[TrackedObservation,...]
    coasting: tuple[CoastedTrack,...]
    expired_ids: tuple[int,...]


@dataclass
class _Track:
    first_seen: int
    last_seen: int
    descriptor: np.ndarray
    center_pixels: np.ndarray
    velocity: np.ndarray
    history: deque
    missed: bool = False


class InstanceTracker:
    def __init__(self, image_shape, *, memory_frames=4, coast_frames=30, max_distance_pixels=8., appearance_weight=8.):
        if (not isinstance(image_shape,tuple) or len(image_shape)!=2
                or any(type(v) is not int or v<1 for v in image_shape)):
            raise ValueError('A fixed(height,width) image shape is required per tracker')
        if (type(memory_frames) is not int or type(coast_frames) is not int
                or not 0<memory_frames<=coast_frames):
            raise ValueError('Require0<memory_frames<=coast_frames')
        if (not np.isfinite(max_distance_pixels) or max_distance_pixels<=0
                or not np.isfinite(appearance_weight) or appearance_weight<0):
            raise ValueError('Finite positive distance and nonnegative appearance weight required')
        self.image_shape=image_shape
        self.memory_frames,self.coast_frames=memory_frames,coast_frames
        self.max_distance_pixels,self.appearance_weight=float(max_distance_pixels),float(appearance_weight)
        self._frame=None
        self._next_id=0
        self._tracks={}

    def fork(self):
        """Independent own-agent memory for a laboratory branch, without native state."""
        return copy.deepcopy(self)

    def _predict(self, track, frame):
        return track.center_pixels+track.velocity*(frame-track.last_seen)

    def _embedding(self, track, frame):
        recent=[d[7:39] for t,d in track.history if frame-t<=self.memory_frames]
        return np.mean(recent,axis=0) if recent else track.descriptor[7:39]

    def update(self, instances, *, frame, prediction_horizon_frames):
        # Commit only a complete observation update. An allocation error cannot
        # leave advanced tracks behind a stale clock and make a retry double-count.
        candidate=self.fork()
        result=candidate._update(instances,frame=frame,prediction_horizon_frames=prediction_horizon_frames)
        self._frame,self._next_id,self._tracks=candidate._frame,candidate._next_id,candidate._tracks
        return result

    def _update(self, instances, *, frame, prediction_horizon_frames):
        if (type(frame) is not int or not 0<=frame<=2**53
                or (self._frame is not None and frame<=self._frame)):
            raise ValueError('Observation frame must strictly increase in the declared clock')
        if type(prediction_horizon_frames) is not int or not 1<=prediction_horizon_frames<=2**53:
            raise ValueError('Positive prediction horizon in the same frame unit required')
        instances=list(instances)
        height,width=self.image_shape
        occupied=set()
        for instance in instances:
            if (not isinstance(instance,VisualInstance) or not isinstance(instance.descriptor,np.ndarray)
                    or instance.descriptor.shape!=(49,) or instance.descriptor.dtype!=np.float32
                    or not np.isfinite(instance.descriptor).all()):
                raise ValueError('Finite own-parser49-float instances required')
            x,y,w,h=instance.box
            pixels=instance.source_pixels
            if (any(type(v) is not int for v in instance.box) or not 0<=x<x+w<=width
                    or not 0<=y<y+h<=height or not isinstance(pixels,np.ndarray) or pixels.ndim!=1 or len(pixels)==0
                    or not np.issubdtype(pixels.dtype,np.integer) or (pixels<0).any() or (pixels>=height*width).any()):
                raise ValueError('Instance geometry must belong to the current image')
            members=set(map(int,pixels))
            if len(members)!=len(pixels) or occupied&members:
                raise ValueError('Own instances must not duplicate source pixels')
            occupied.update(members)
            py,px=np.divmod(pixels,width)
            if (int(px.min()),int(py.min()),int(px.max()-px.min()+1),int(py.max()-py.min()+1))!=instance.box:
                raise ValueError('Source pixels must have the declared tight box')
            if not np.isclose(instance.descriptor[4],len(pixels)/(height*width),rtol=1e-5,atol=1e-7):
                raise ValueError('Source pixels and descriptor area disagree')
            if not np.allclose(instance.descriptor[:4],[(x+w/2)/width,(y+h/2)/height,w/width,h/height],rtol=1e-5,atol=1e-7):
                raise ValueError('Instance descriptor/box disagree')
            if (np.abs(instance.descriptor[7:39])>1).any():
                raise ValueError('Expected normalized own appearance embedding')
        # Stable geometry order, independent of the caller's list/category numbering.
        instances.sort(key=lambda i:int(i.source_pixels.min()))
        expired=tuple(sorted(i for i,t in self._tracks.items() if frame-t.last_seen>self.coast_frames))
        ids=sorted(i for i in self._tracks if i not in expired)
        matches={}
        if ids and instances:
            centers=np.array([(i.box[0]+i.box[2]/2,i.box[1]+i.box[3]/2) for i in instances],dtype=float)
            predicted=np.stack([self._predict(self._tracks[i],frame) for i in ids])
            distance=np.linalg.norm(predicted[:,None]-centers[None],axis=2)
            old=np.stack([self._embedding(self._tracks[i],frame) for i in ids])
            new=np.stack([i.descriptor[7:39] for i in instances])
            norms=np.linalg.norm(old,axis=1)[:,None]*np.linalg.norm(new,axis=1)[None]
            similarity=np.divide(old@new.T,norms,out=np.ones_like(norms),where=norms>1e-12)
            appearance=1-np.clip(similarity,-1,1)
            # Common positive scaling preserves the ordering of admissible costs,
            # while keeping all dummies finite even for extreme finite parameters.
            cost_scale=max(self.max_distance_pixels,self.appearance_weight,1.)
            pair_cost=distance/cost_scale+(self.appearance_weight/cost_scale)*appearance
            unmatched=4.*(len(ids)+1)  # Every admissible pair cost is at most3.
            tolerance=np.array([1e-12+np.finfo(np.float32).eps*np.linalg.norm(
                self._tracks[i].velocity*(frame-self._tracks[i].last_seen)) for i in ids])[:,None]
            admissible=distance<=self.max_distance_pixels+tolerance
            cost=np.full((len(ids),len(instances)+len(ids)),unmatched)
            cost[:,:len(instances)]=np.where(admissible,pair_cost,unmatched*2)
            rows,columns=linear_sum_assignment(cost)
            matches={int(column):ids[int(row)] for row,column in zip(rows,columns)
                     if column<len(instances) and admissible[row,column]}
        for track_id in expired: del self._tracks[track_id]
        observed=[]
        matched_ids=set()
        for position,instance in enumerate(instances):
            descriptor=instance.descriptor.astype(np.float64,copy=True)
            center=np.array([instance.box[0]+instance.box[2]/2,instance.box[1]+instance.box[3]/2],dtype=float)
            velocity=descriptor[5:7]*[width,height]/prediction_horizon_frames
            if position in matches:
                track_id=matches[position]
                track=self._tracks[track_id]
                reappeared=track.missed
                track.last_seen=frame
                track.descriptor=descriptor
                track.center_pixels=center
                track.velocity=velocity
                track.missed=False
                track.history.append((frame,descriptor.copy()))
                while track.history and frame-track.history[0][0]>self.memory_frames: track.history.popleft()
            else:
                track_id=self._next_id
                self._next_id+=1
                track=_Track(frame,frame,descriptor,center,velocity,
                             deque([(frame,descriptor.copy())]))
                self._tracks[track_id]=track
                reappeared=False
            matched_ids.add(track_id)
            observed.append(TrackedObservation(track_id,instance,track.first_seen,reappeared))
        coasting=[]
        for track_id in sorted(self._tracks):
            if track_id not in matched_ids:
                track=self._tracks[track_id]
                track.missed=True
                coasting.append(CoastedTrack(track_id,tuple(map(float,self._predict(track,frame)/[width,height])),frame-track.last_seen))
        self._frame=frame
        return TrackerFrame(frame,tuple(observed),tuple(coasting),expired)
