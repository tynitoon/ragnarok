"""Temporal V1: predict the blank first ghost row at y=1 mod12.

Observed on all1557matching render entries of development17421. This is a
separate RAM-side rendering hypothesis, not a threshold/topology change. The
sealed raw V3 and temporal V0 remain unchanged. Reserved17422 stays closed.
"""
from ragnarok.evaluation.pooled_mspacman_v70 import (
    KINDS, ROI, _masks, _add, _objects, MsPacmanPooledPixelJudge,
)
from ragnarok.evaluation.mspacman_maze_visible_v70 import (
    decode_visible_ram, wrapped_ghost_supports, support_boxes,
)

REVISION='mspacman_two_raw_spatial_envelopes_blank_ghost_row_v1'


def source_ram_objects(current,previous):
    result=[o for o in decode_visible_ram(current,previous) if o.kind!='ghost']
    for entry,mask in wrapped_ghost_supports(current,previous):
        y=entry.box[1]
        if 0<=y<ROI[3] and y%12==1:
            mask[y,:]=False
        result.extend(support_boxes(mask,'ghost',f'draw:{entry.index}'))
    return result


def pooled_mspacman_ram_objects(before_previous,previous,current):
    masks=_masks()
    for now,before in ((previous,before_previous),(current,previous)):
        for obj in source_ram_objects(now,before):
            _add(masks,obj)
    return _objects(masks)
