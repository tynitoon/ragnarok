"""Prospective external V2 judge: gather just the eight pixels of each pill.

The previous integral-image candidate was slower and is preserved separately.
No reference geometry, actor occlusion, learned feature or native state changes.
"""
import numpy as np

from ragnarok.evaluation.mspacman_render_v70 import power_pill_render_boxes
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_support,player_list_render_box
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_support,fruit_render_boxes
from ragnarok.evaluation.mspacman_maze_visible_v70 import wrapped_ghost_supports,support_boxes
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import returning_eye_box,eye_objects
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.fast_visible_mspacman_v70 import _PILL_TABLES,_unoccluded_pills


def _indices(labels):
    result=[]
    for label in labels:
        x,y,w,h=label.box
        if (w,h)!=(4,2) or not 0<=x<x+w<=160 or not 0<=y<y+h<=171:
            raise ValueError('Frozen nominal pill geometry must be4x2 inside160x171')
        result.append([(y+dy)*160+x+dx for dy in range(2) for dx in range(4)])
    array=np.array(result,dtype=np.intp).reshape(-1,8)
    array.flags.writeable=False
    return array


_PILL_INDICES=tuple(_indices(table[0]) for table in _PILL_TABLES)


def _nominal_and_power(current,occupied):
    if (not isinstance(occupied,np.ndarray) or occupied.dtype!=np.bool_
            or occupied.shape!=(171,160)):
        raise ValueError('Frozen actor occupancy must be a bool171x160 raster')
    table_index=0 if current[0]==0 else 1
    labels,offsets,bits=_PILL_TABLES[table_index]
    selected=np.flatnonzero(current[offsets]&bits)
    hidden=occupied.reshape(-1)[_PILL_INDICES[table_index][selected]]
    covered=hidden.any(axis=1)
    result=[]
    for index,blocked in zip(selected.tolist(),covered.tolist(),strict=True):
        obj=labels[index]
        if blocked:result.extend(_unoccluded_pills([obj],occupied))
        else:result.append(obj)
    # Blinking power pills have their frozen general geometry and clipping.
    result.extend(_unoccluded_pills(power_pill_render_boxes(current),occupied))
    return result


def source_ram_objects(current, previous):
    current, previous = np.asarray(current), np.asarray(previous)
    if any(r.shape != (128,) or r.dtype != np.uint8 for r in (current, previous)):
        raise ValueError('Current and actual previous RAM must be uint8[128]')
    actors = player_list_support(current, previous) | fruit_support(current, previous)
    occupied = actors.copy()
    ghosts = list(wrapped_ghost_supports(current, previous))
    for _, mask in ghosts:
        occupied |= mask
    result = player_list_render_box(current, previous) + fruit_render_boxes(current, previous)
    result += _nominal_and_power(current, occupied)
    for entry, mask in ghosts:
        box = returning_eye_box(entry, previous)
        if box is not None:
            result.extend(eye_objects(box, actors, f'draw:{entry.index}'))
        else:
            y = entry.box[1]
            if 0 <= y < 171 and y % 12 == 1:
                mask[y, :] = False
            result.extend(support_boxes(mask, 'ghost', f'draw:{entry.index}'))
    return result


def mspacman_visible_scene(current, previous):
    return [SceneObject(f'{o.kind}:{o.slot}', o.kind, o.box,
                        can_move=o.kind in ('player', 'ghost', 'fruit'))
            for o in source_ram_objects(current, previous)]
