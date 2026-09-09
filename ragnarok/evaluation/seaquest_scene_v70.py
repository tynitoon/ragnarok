"""External Seaquest scoring of raw slots and temporal spatial components."""
from ragnarok.evaluation.seaquest_v70 import KINDS,decode_seaquest_ram
from ragnarok.evaluation.scene_perception_v70 import SceneObject,ScenePerceptionScorer


def seaquest_scene_objects(ram):
    return [SceneObject(f"{o.kind}:{o.slot}",o.kind,o.box) for o in decode_seaquest_ram(ram)]


class SeaquestSceneScorer(ScenePerceptionScorer):
    def __init__(self):
        super().__init__(roi=(0,33,160,165))

    def score(self,boxes,*,frame_truths=None):
        if frame_truths is None:
            return super().score(boxes)
        groups=list(frame_truths)
        objects=[SceneObject(f"component:{i}",g.label.kind,g.label.box) for i,g in enumerate(groups)]
        raw_moving=self._moving
        # Distinct components may have exactly the same member slots. Count
        # each spatial component once, with its own frame-local identity.
        self._moving={f"component:{i}" for i,g in enumerate(groups) if g.members & raw_moving}
        try:
            return super().score(boxes,frame_truths=objects)
        finally:
            self._moving=raw_moving

    def report(self):
        result=super().report()
        for kind in KINDS:
            result["per_type"].setdefault(kind,dict(visible=0,eligible=0,recalled=0,recall_moving=None))
        return result
