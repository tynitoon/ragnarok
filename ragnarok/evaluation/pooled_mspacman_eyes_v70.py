"""Temporal V2 adds the visible eyes of returning ghosts on the RAM side.

On development17421, all221 matching source entries have exactly this support,
including player/fruit occlusion. They remain ghosts. RGB interpretation, other
categories, spatial unions, matching and criteria are unchanged. Logical RAM
states are external rendering evidence only; no agent receives these objects.
"""
import numpy as np
from ragnarok.evaluation.pooled_mspacman_rows_v70 import (
    KINDS,ROI,_masks,_add,_objects,MsPacmanPooledPixelJudge,
)
from ragnarok.evaluation.mspacman_maze_visible_v70 import (
    decode_visible_ram,wrapped_ghost_supports,support_boxes,
)
from ragnarok.evaluation.mspacman_v70 import SlottedLabel
from ragnarok.evaluation.mspacman_player_list_v70 import player_list_support
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_support

REVISION='mspacman_two_raw_spatial_envelopes_returning_eyes_v2'
EYE_ROWS=(66,165,165,165,66)


def returning_eye_box(entry,previous):
    """Require a unique exact allocation/renderer match, not just small RGB size.

High state nibbles4/6 and the pointer offset47 were observed on two returns.
A coordinate coincidence with state5 also occurs and must remain a normal
ghost. Low state/draw nibbles do not supply identities to the output.
"""
    if entry.colour_code!=1:return None
    r=np.asarray(previous)
    if r.shape!=(128,) or r.dtype!=np.uint8:raise ValueError('RAM must be uint8[128]')
    x,y,_,_=entry.box;y+=47
    matches=[i for i in range(4) if int(r[1+i])>>4 in (4,6)
        and x==(int(r[6+i])-12)%160 and y==int(r[12+i])+2]
    if len(matches)>1:raise ValueError('Ambiguous returning-eye allocation')
    return (x,y,8,5) if matches else None


def eye_objects(box,occluded,slot):
    x,y,_,_=box
    mask=np.zeros((ROI[3],ROI[2]),dtype=bool)
    for dy,bits in enumerate(EYE_ROWS):
        if 0<=y+dy<ROI[3]:
            for dx in range(8):
                if bits&(1<<(7-dx)):mask[y+dy,(x+dx)%160]=True
    mask &= ~occluded
    # Both eyes are one render entry despite the gap between them. Split only
    # at the physical horizontal wrap, just as for a normal eight-pixel sprite.
    bands=[(x,min(x+8,160))]
    if x+8>160:bands.append((0,x+8-160))
    result=[]
    for i,(left,right) in enumerate(bands):
        yy,xx=np.nonzero(mask[:,left:right])
        if len(xx):
            result.append(SlottedLabel('ghost',(left+int(xx.min()),int(yy.min()),
                int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)),f'{slot}:{i}'))
    return result


def source_ram_objects(current,previous):
    result=[o for o in decode_visible_ram(current,previous) if o.kind!='ghost']
    occluded=None
    for entry,mask in wrapped_ghost_supports(current,previous):
        box=returning_eye_box(entry,previous)
        if box is not None:
            if occluded is None:
                occluded=player_list_support(current,previous)|fruit_support(current,previous)
            result.extend(eye_objects(box,occluded,f'draw:{entry.index}'))
        else:
            y=entry.box[1]
            if 0<=y<ROI[3] and y%12==1:mask[y,:]=False
            result.extend(support_boxes(mask,'ghost',f'draw:{entry.index}'))
    return result


def pooled_mspacman_ram_objects(before_previous,previous,current):
    masks=_masks()
    for now,before in ((previous,before_previous),(current,previous)):
        for obj in source_ram_objects(now,before):_add(masks,obj)
    return _objects(masks)
