"""Prospective exact last-input cache per external visible kind, with bounded state."""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_temporal_groups_v70 import REGISTRY,VisibleEnvelopeGroup


class CachedVisibleGroups:
    def __init__(self,game):
        self.game=game
        self.roi,self.kinds,_=REGISTRY[game]
        self._cache={}

    def group(self,objects):
        objects=list(objects)
        for o in objects:
            if not isinstance(o,SceneObject) or o.kind not in self.kinds or not o.instance_id:
                raise ValueError('Declared external source scene objects required')
            x,y,w,h=o.box
            if (any(type(v) is not int for v in o.box) or not self.roi[0]<=x<x+w<=self.roi[2]
                    or not self.roi[1]<=y<y+h<=self.roi[3]):
                raise ValueError('Source rectangle outside its calibrated integer ROI')
        candidate={}
        result=[]
        for kind in self.kinds:
            own=tuple(dict.fromkeys(o for o in objects if o.kind==kind))
            cached=self._cache.get(kind)
            if cached is not None and cached[0]==own:
                groups=cached[1]
            elif not own:groups=()
            else:
                mask=np.zeros((self.roi[3],self.roi[2]),dtype=bool)
                for o in own:
                    x,y,w,h=o.box;mask[y:y+h,x:x+w]=True
                labels,_=ndimage.label(mask)
                members={}
                for o in own:
                    x,y,_,_=o.box
                    members.setdefault(int(labels[y,x]),set()).add(o.instance_id)
                groups=tuple(VisibleEnvelopeGroup(ObjectLabel(kind,(xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start)),
                    frozenset(members[label])) for label,(ys,xs) in enumerate(ndimage.find_objects(labels),1))
            candidate[kind]=(own,groups)
            result.extend(groups)
        self._cache=candidate
        return result
