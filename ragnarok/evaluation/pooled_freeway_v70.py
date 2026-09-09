"""Two-source Freeway geometry, separate from the sealed raw calibration."""
import numpy as np

from ragnarok.evaluation.freeway_v70 import (
    KINDS, CHICKEN_COLOUR, CAR_COLOURS, decode_freeway_ram, screen_freeway_objects,
)
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="freeway_two_raw_box_envelopes_v0"


def _envelopes(groups):
    result=[]
    for kind in KINDS:
        boxes=groups[kind]
        if not boxes:
            continue
        left,top=min(b[0] for b in boxes),min(b[1] for b in boxes)
        right,bottom=max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)
        result.append(ObjectLabel(kind,(left,top,right-left,bottom-top)))
    return result


def pooled_freeway_ram_objects(previous,current):
    groups={kind:[] for kind in KINDS}
    for ram in (previous,current):
        for obj in decode_freeway_ram(ram):
            groups[obj.kind].append(obj.box)
    return _envelopes(groups)


def pooled_freeway_screen_objects(previous,current):
    groups={kind:[] for kind in KINDS}
    for rgb,other in ((previous,current),(current,previous)):
        for obj in screen_freeway_objects(rgb):
            x,y,w,h=obj.box
            a,b=rgb[y:y+h,x:x+w],other[y:y+h,x:x+w]
            colour=CAR_COLOURS[int(obj.kind.split("_")[1])] if obj.kind.startswith("car_") else CHICKEN_COLOUR
            support=np.all(a==colour,axis=2)&(np.all(a==b,axis=2)|np.any(a>b,axis=2))
            ys,xs=np.nonzero(support)
            if len(xs):
                left,top=int(xs.min()),int(ys.min())
                groups[obj.kind].append((x+left,y+top,int(xs.max())-left+1,int(ys.max())-top+1))
    return _envelopes(groups)
