"""External component-window geometry of two actual Asteroids source images."""
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.asteroids_v70 import KINDS, ROI, PLAYER_COLOUR, BLUE, ASTEROID_COLOURS
from ragnarok.evaluation.asteroids_visible_v70 import decode_visible_asteroids_ram, screen_visible_asteroids_objects

REVISION='asteroids_two_raw_spatial_envelopes_v0'
COLOURS={'player':(PLAYER_COLOUR,), 'player_missile':(PLAYER_COLOUR,BLUE), 'asteroid':ASTEROID_COLOURS}


def _masks():
    return {kind:np.zeros((210,160),dtype=bool) for kind in KINDS}


def _add(masks,obj):
    x,y,w,h=obj.box
    if obj.kind not in masks or not (ROI[0]<=x<x+w<=ROI[2] and ROI[1]<=y<y+h<=ROI[3]):
        raise ValueError('Source box outside the declared category/ROI')
    masks[obj.kind][y:y+h,x:x+w]=True


def _objects(masks):
    result=[]
    for kind in KINDS:
        labels,_=ndimage.label(masks[kind])
        for y,x in ndimage.find_objects(labels):
            result.append(ObjectLabel(kind,(x.start,y.start,x.stop-x.start,y.stop-y.start)))
    return result


def pooled_asteroids_ram_objects(before_previous,previous,current):
    masks=_masks()
    for now,before in ((previous,before_previous),(current,previous)):
        for obj in decode_visible_asteroids_ram(now,before):
            _add(masks,obj)
    return _objects(masks)


def pooled_asteroids_screen_objects(previous,current):
    masks=_masks()
    for rgb,other in ((previous,current),(current,previous)):
        for obj in screen_visible_asteroids_objects(rgb):
            x,y,w,h=obj.box
            own,competing=rgb[y:y+h,x:x+w],other[y:y+h,x:x+w]
            palette=np.any(np.stack([np.all(own==c,axis=2) for c in COLOURS[obj.kind]]),axis=0)
            contributing=palette&(np.all(own==competing,axis=2)|np.any(own>competing,axis=2))
            yy,xx=np.nonzero(contributing)
            if len(xx):
                box=(x+int(xx.min()),y+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
                _add(masks,ObjectLabel(obj.kind,box))
    return _objects(masks)
