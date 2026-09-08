"""Instance-aware judge: object identity is distinct from semantic category.

This evaluator is never an agent observation. Static structural changes (such
as a brick run breaking) participate in precision without becoming translation.
"""
from dataclasses import dataclass

import numpy as np

from ragnarok.evaluation.perception_v70 import PerceptionScorer, match_centers


@dataclass(frozen=True)
class SceneObject:
    instance_id: str
    kind: str
    box: tuple
    can_move: bool = True

    @property
    def center(self):
        x,y,w,h = self.box
        return x+w/2,y+h/2


class ScenePerceptionScorer(PerceptionScorer):
    def _objects(self, objects):
        objects=list(objects)
        if any(not isinstance(o,SceneObject) or not o.instance_id or not o.kind or type(o.can_move) is not bool for o in objects):
            raise ValueError("Expected declared scene identities, categories and motion eligibility")
        if len({o.instance_id for o in objects})!=len(objects):
            raise ValueError("Scene instance IDs must be unique in a frame")
        boxes=np.array([o.box for o in objects],dtype=float).reshape(-1,4)
        self._centers(boxes)  # finite, positive geometry validation before history mutation
        left,top,right,bottom=self.roi
        return [o for o in objects if left<=o.center[0]<right and top<=o.center[1]<bottom]

    def observe_truth(self, truths, *, frame, episode):
        if type(frame) is not int or type(episode) is not int or frame<0 or episode<0:
            raise ValueError("Frame and episode must be nonnegative integers")
        if self._frame is not None and (frame!=self._frame+1 or episode not in (self._episode,self._episode+1)):
            raise ValueError("Scorer requires consecutive raw frames and monotone episode IDs")
        visible=self._objects(truths)
        positions={o.instance_id:o.center for o in visible if o.can_move}
        if episode!=self._episode:
            self._history.clear()
        self._history.append(positions)
        identities=set().union(*(old.keys() for old in self._history))
        self._moving={key for key in identities if len({old[key] for old in self._history if key in old})>1}
        self._visible=visible
        self._frame,self._episode=frame,episode
        self._observed_frames+=1

    def score(self, boxes, *, frame_truths=None):
        visible=None if frame_truths is None else self._objects(frame_truths)
        self._score_centers(self._centers(boxes),visible=visible)

    def _score_centers(self, centers, *, visible=None):
        if self._frame is None or self._frame==self._scored_frame:
            raise ValueError("Observe a new truth frame before scoring")
        visible=self._visible if visible is None else visible
        moving=self._moving & {o.instance_id for o in visible if o.can_move}
        pairs=match_centers(np.array([o.center for o in visible],dtype=float).reshape(-1,2),centers,tolerance=self.tolerance)
        matched={visible[i].instance_id for i,_ in pairs}
        self._frames+=1
        self._predictions+=len(centers)
        self._matches+=len(pairs)
        self._eligible+=len(moving)
        self._recalled+=len(moving & matched)
        for obj in visible:
            stat=self._per_type.setdefault(obj.kind,dict(visible=0,eligible=0,recalled=0))
            stat["visible"]+=1
            stat["eligible"]+=int(obj.instance_id in moving)
            stat["recalled"]+=int(obj.instance_id in moving and obj.instance_id in matched)
        self._scored_frame=self._frame


def breakout_scene_objects(ram):
    # The sealed geometry decoder is reused unchanged, without semantic input to the reference.
    from ragnarok.evaluation.breakout_v70 import decode_breakout_ram
    return [SceneObject(o.kind if o.kind in ("player","ball") else f"{o.kind}:{o.box[0]}",
                        o.kind,o.box,can_move=o.kind in ("player","ball")) for o in decode_breakout_ram(ram)]
