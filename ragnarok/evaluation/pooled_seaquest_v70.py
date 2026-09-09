"""Two-source spatial envelopes, distinct from the sealed raw Seaquest judge.

These labels describe contributing raw sources, not recoverable identities from
the maximum RGB alone. Same-kind touching envelopes are spatial unions.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.seaquest_v70 import (
    KINDS, PLAYER_COLOURS, SHARK_COLOURS, SUBMARINE_COLOUR, BLUE,
    decode_seaquest_ram, screen_seaquest_objects,
)

REVISION="seaquest_two_raw_spatial_envelopes_v0"
COLOURS={"player":PLAYER_COLOURS,"player_missile":PLAYER_COLOURS,
         "shark":SHARK_COLOURS,"submarine":(SUBMARINE_COLOUR,),
         "diver":(BLUE,),"enemy_missile":(BLUE,)}


def _masks():
    return {kind:np.zeros((210,160),dtype=bool) for kind in KINDS}


def _objects(masks):
    result=[]
    for kind in KINDS:
        labels,_=ndimage.label(masks[kind])
        for y,x in ndimage.find_objects(labels):
            result.append(ObjectLabel(kind,(x.start,y.start,x.stop-x.start,y.stop-y.start)))
    return result


def pooled_seaquest_ram_objects(previous,current):
    masks=_masks()
    for ram in (previous,current):
        for obj in decode_seaquest_ram(ram):
            x,y,w,h=obj.box
            masks[obj.kind][y:y+h,x:x+w]=True
    return _objects(masks)


def pooled_seaquest_screen_objects(previous,current):
    masks=_masks()
    for rgb,other in ((previous,current),(current,previous)):
        for obj in screen_seaquest_objects(rgb):
            x,y,w,h=obj.box
            own,competing=rgb[y:y+h,x:x+w],other[y:y+h,x:x+w]
            palette=np.any(np.stack([np.all(own==c,axis=2) for c in COLOURS[obj.kind]]),axis=0)
            contributing=palette&(np.all(own==competing,axis=2)|np.any(own>competing,axis=2))
            yy,xx=np.nonzero(contributing)
            if len(xx):
                masks[obj.kind][y+int(yy.min()):y+int(yy.max())+1,
                                x+int(xx.min()):x+int(xx.max())+1]=True
    return _objects(masks)
