"""Exact four-connected filled envelopes with external render-slot membership.

The raster keeps the sealed component definition and scan order. Membership is
used only to carry the scorer's raw-frame movement eligibility to a pooled
component. It is neither a learned instance nor a physical identity guarantee.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.scene_perception_v70 import SceneObject,ScenePerceptionScorer

REGISTRY={
    'ms_pacman':((0,0,160,171),('player','ghost','fruit','power_pill','pill'),('player','ghost','fruit')),
    'asteroids':((0,16,160,194),('player','asteroid','player_missile'),('player','asteroid','player_missile')),
}


@dataclass(frozen=True)
class VisibleEnvelopeGroup:
    label:ObjectLabel
    members:frozenset[str]


def group_visible_sources(objects,*,game):
    roi,kinds,_=REGISTRY[game];objects=list(objects)
    for o in objects:
        if not isinstance(o,SceneObject) or o.kind not in kinds or not o.instance_id:
            raise ValueError('Declared external source scene objects required')
        x,y,w,h=o.box
        if (any(type(v) is not int for v in o.box) or not roi[0]<=x<x+w<=roi[2]
                or not roi[1]<=y<y+h<=roi[3]):
            raise ValueError('Source rectangle outside its calibrated integer ROI')
    groups=[]
    for kind in kinds:
        own=list(dict.fromkeys(o for o in objects if o.kind==kind))
        mask=np.zeros((roi[3],roi[2]),dtype=bool)
        for o in own:
            x,y,w,h=o.box;mask[y:y+h,x:x+w]=True
        labels,_=ndimage.label(mask)
        members={}
        for o in own:
            x,y,_,_=o.box
            members.setdefault(int(labels[y,x]),set()).add(o.instance_id)
        for label,(ys,xs) in enumerate(ndimage.find_objects(labels),1):
            box=(xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start)
            groups.append(VisibleEnvelopeGroup(ObjectLabel(kind,box),frozenset(members[label])))
    return groups


class VisibleTemporalScorer(ScenePerceptionScorer):
    def __init__(self,game):
        roi,self.kinds,self.mobile=REGISTRY[game]
        super().__init__(roi=roi)

    def score(self,boxes,*,frame_truths=None):
        if frame_truths is None:return super().score(boxes)
        groups=list(frame_truths)
        if any(not isinstance(g,VisibleEnvelopeGroup) for g in groups):
            raise ValueError('External visible envelope groups required')
        objects=[SceneObject(f'component:{i}',g.label.kind,g.label.box,g.label.kind in self.mobile)
            for i,g in enumerate(groups)]
        raw_moving=self._moving
        self._moving={f'component:{i}' for i,g in enumerate(groups) if g.members&raw_moving}
        try:return super().score(boxes,frame_truths=objects)
        finally:self._moving=raw_moving

    def report(self):
        report=super().report()
        for kind in self.kinds:
            report['per_type'].setdefault(kind,dict(visible=0,eligible=0,recalled=0,recall_moving=None))
        return report
