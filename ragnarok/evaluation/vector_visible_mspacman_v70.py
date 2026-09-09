"""Prospective exact external judge adapter; frozen source geometry is unchanged."""
import numpy as np

from ragnarok.evaluation import fast_visible_mspacman_v70 as original


def unoccluded_pills(pills,occupied):
    """Integral-image quick admission, original clipping for overlapping supports."""
    pills=list(pills)
    if not pills:return []
    # This helper consumes the same validated/internal geometry as the original.
    # Signed sums make rectangle inclusion-exclusion safe at every image edge.
    summed=np.pad(occupied.cumsum(axis=0,dtype=np.int64).cumsum(axis=1,dtype=np.int64),((1,0),(1,0)))
    boxes=np.array([o.box for o in pills],dtype=np.intp)
    x,y,w,h=boxes.T
    overlap=summed[y+h,x+w]-summed[y,x+w]-summed[y+h,x]+summed[y,x]
    result=pills.copy()
    for i in np.flatnonzero(overlap):
        clipped=original._unoccluded_pills([pills[i]],occupied)
        result[i]=clipped[0] if clipped else None
    return [o for o in result if o is not None]


def source_ram_objects(current,previous):
    current,previous=np.asarray(current),np.asarray(previous)
    if any(r.shape!=(128,) or r.dtype!=np.uint8 for r in (current,previous)):
        raise ValueError('Current and actual previous RAM must be uint8[128]')
    actors=original.player_list_support(current,previous)|original.fruit_support(current,previous)
    occupied=actors.copy()
    ghosts=list(original.wrapped_ghost_supports(current,previous))
    for _,mask in ghosts:occupied|=mask
    result=original.player_list_render_box(current,previous)+original.fruit_render_boxes(current,previous)
    result+=unoccluded_pills(original._nominal_pills(current)+original.power_pill_render_boxes(current),occupied)
    for entry,mask in ghosts:
        box=original.returning_eye_box(entry,previous)
        if box is not None:result.extend(original.eye_objects(box,actors,f'draw:{entry.index}'))
        else:
            y=entry.box[1]
            if 0<=y<171 and y%12==1:mask[y,:]=False
            result.extend(original.support_boxes(mask,'ghost',f'draw:{entry.index}'))
    return result


def mspacman_visible_scene(current,previous):
    return [original.SceneObject(f'{o.kind}:{o.slot}',o.kind,o.box,
        can_move=o.kind in ('player','ghost','fruit')) for o in source_ram_objects(current,previous)]
