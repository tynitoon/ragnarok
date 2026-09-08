"""Temporal Breakout judge, separate from the sealed raw calibration."""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.breakout_v70 import KINDS, COLOURS, decode_breakout_ram, screen_breakout_objects
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="breakout_two_raw_spatial_union_v0"


def _boxes(kind,mask):
    if kind in ("player","ball"):
        ys,xs=np.nonzero(mask)
        if not len(xs):
            return []
        x,y=int(xs.min()),int(ys.min())
        return [ObjectLabel(kind,(x,y,int(xs.max())-x+1,int(ys.max())-y+1))]
    labels,_=ndimage.label(mask)
    return [ObjectLabel(kind,(x.start,y.start,x.stop-x.start,y.stop-y.start))
            for y,x in ndimage.find_objects(labels)]


def pooled_breakout_ram_objects(previous,current):
    masks={kind:np.zeros((210,160),dtype=bool) for kind in KINDS}
    for ram in (previous,current):
        for obj in decode_breakout_ram(ram):
            x,y,w,h=obj.box
            masks[obj.kind][y:y+h,x:x+w]=True
    return [obj for kind in KINDS for obj in _boxes(kind,masks[kind])]


def _pixel_masks(rgb):
    masks={kind:np.zeros((210,160),dtype=bool) for kind in KINDS}
    # Only the independently calibrated RGB checker; no RAM or generic reference.
    for obj in screen_breakout_objects(rgb):
        x,y,w,h=obj.box
        colour=COLOURS[int(obj.kind.split("_")[1])] if obj.kind.startswith("bricks_") else COLOURS[0]
        masks[obj.kind][y:y+h,x:x+w] |= np.all(rgb[y:y+h,x:x+w]==colour,axis=2)
    return masks


def pooled_breakout_screen_objects(previous,current):
    a,b=_pixel_masks(previous),_pixel_masks(current)
    same=np.all(previous==current,axis=2)
    survives_a=same | np.any(previous>current,axis=2)
    survives_b=same | np.any(current>previous,axis=2)
    return [obj for kind in KINDS for obj in _boxes(kind,(a[kind]&survives_a)|(b[kind]&survives_b))]
