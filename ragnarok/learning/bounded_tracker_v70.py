"""Capacity wrapper preserving the independently reviewed tracker implementation."""
from ragnarok.learning.instance_tracker_v70 import InstanceTracker


class BoundedInstanceTracker(InstanceTracker):
    def __init__(self,image_shape,*,max_tracks=512,**kwargs):
        if type(max_tracks) is not int or not 1<=max_tracks<=4096:
            raise ValueError('Declared track capacity must be in1..4096')
        self.max_tracks=max_tracks
        super().__init__(image_shape,**kwargs)

    def _update(self,instances,*,frame,prediction_horizon_frames):
        instances=list(instances)
        if len(instances)>self.max_tracks or len(self._tracks)>self.max_tracks:
            raise ValueError('Track/observation capacity exceeded before assignment')
        result=super()._update(instances,frame=frame,prediction_horizon_frames=prediction_horizon_frames)
        if len(self._tracks)>self.max_tracks:
            # The reviewed update operates on a private candidate. Raising here
            # discards it, including new identities, observations and expirations.
            raise ValueError('Track capacity exceeded including coasting identities')
        return result
