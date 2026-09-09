"""CPU-only implementation of the frozen V2 external source-frame judge.

This changes computation, not geometry: pill occlusion still uses the legacy
ghost supports, while the final ghost labels use V2 rows/eyes. No RGB input,
agent features, learning or mutable trajectory cache is introduced here.
"""
import numpy as np
from ragnarok.evaluation.mspacman_v70 import GRID1, GRID2, SlottedLabel
from ragnarok.evaluation.mspacman_render_v70 import power_pill_render_boxes
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_support, player_list_render_box
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_support, fruit_render_boxes
from ragnarok.evaluation.mspacman_maze_visible_v70 import wrapped_ghost_supports, support_boxes
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import returning_eye_box, eye_objects
from ragnarok.evaluation.scene_perception_v70 import SceneObject


def _pill_table(grid):
    labels, offsets, bits = [], [], []
    # Preserve the insertion order of the frozen decoder's per-row dictionary.
    columns = [(17, 0, 16), (0, 0, 64)]
    for j in range(8):
        columns.append(((4-(j >> 1)) if j & 1 else (12-(j >> 1)), 1, 1 << j))
        columns.append(((13+(j >> 1)) if j & 1 else (5+(j >> 1)), 2, 1 << j))
    for row in range(14):
        for col, byte, bit in columns:
            if grid[row][col]:
                x = 8+8*col if col <= 8 else 12+8*col
                labels.append(SlottedLabel('pill', (x, 7+12*row, 4, 2), f'{row}:{col}'))
                offsets.append(59+3*row+byte)
                bits.append(bit)
    offsets, bits = np.array(offsets, dtype=np.intp), np.array(bits, dtype=np.uint8)
    offsets.flags.writeable = bits.flags.writeable = False
    return tuple(labels), offsets, bits


_PILL_TABLES = (_pill_table(GRID1), _pill_table(GRID2))


def _nominal_pills(current):
    labels, offsets, bits = _PILL_TABLES[0 if current[0] == 0 else 1]
    return [labels[i] for i in np.flatnonzero(current[offsets] & bits)]


def _unoccluded_pills(pills, occupied):
    result = []
    for obj in pills:
        x, y, w, h = obj.box
        hidden = occupied[y:y+h, x:x+w]
        if not hidden.any():
            result.append(obj)
            continue
        ys, xs = np.nonzero(~hidden)
        if len(xs):
            left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            result.append(SlottedLabel(obj.kind, (x+left, y+top, right-left+1, bottom-top+1), obj.slot))
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
    result += _unoccluded_pills(_nominal_pills(current) + power_pill_render_boxes(current), occupied)
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
