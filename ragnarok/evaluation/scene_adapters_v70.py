"""Temporal scene identities for experimenter scoring, outside sealed decoders."""
from ragnarok.evaluation.fast_pooled_breakout_v70 import pooled_breakout_ram_intervals
from ragnarok.evaluation.scene_perception_v70 import SceneObject


def pooled_breakout_scene_objects(previous,current):
    return [SceneObject(o.kind if o.kind in ("player","ball") else f"{o.kind}:{o.box[0]}",
                        o.kind,o.box,can_move=o.kind in ("player","ball"))
            for o in pooled_breakout_ram_intervals(previous,current)]
