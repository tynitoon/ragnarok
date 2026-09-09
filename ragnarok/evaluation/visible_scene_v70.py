"""External visible-render scene keys; never features for the reference or agent."""
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import source_ram_objects as decode_visible_ram
from ragnarok.evaluation.pooled_asteroids_wrap_v70 import source_ram_objects as decode_visible_asteroids_ram


def mspacman_visible_scene(current,previous):
    return [SceneObject(f'{o.kind}:{o.slot}',o.kind,o.box,
                        can_move=o.kind in ('player','ghost','fruit'))
            for o in decode_visible_ram(current,previous)]


def asteroids_visible_scene(current,previous):
    return [SceneObject(f'{o.kind}:{o.slot}',o.kind,o.box,can_move=True)
            for o in decode_visible_asteroids_ram(current,previous)]
