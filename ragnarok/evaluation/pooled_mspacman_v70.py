"""Independent two-source visible envelopes, distinct from sealed raw judges.

RGB receives only the two images and its own causal history. RAM receives three
adjacent snapshots. Neither instrument is an agent feature or learning target.
"""
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.pong_v70 import ObjectLabel
from ragnarok.evaluation.mspacman_v70 import KINDS, ROI, PALETTES
from ragnarok.evaluation.mspacman_ghosts_v70 import GHOST_COLOURS
from ragnarok.evaluation.mspacman_maze_visible_v70 import (
    MAZE_COLOURS, MsPacmanPixelJudge, decode_visible_ram,
)

REVISION = 'mspacman_two_raw_spatial_envelopes_v0'
COLOURS = dict(PALETTES, ghost=GHOST_COLOURS, pill=MAZE_COLOURS, power_pill=MAZE_COLOURS)


def _masks():
    return {kind:np.zeros((ROI[3],ROI[2]),dtype=bool) for kind in KINDS}


def _add(masks, obj):
    x,y,w,h = obj.box
    if obj.kind not in masks or not (0 <= x < x+w <= 160 and 0 <= y < y+h <= 171):
        raise ValueError('Source envelope outside declared category/ROI')
    masks[obj.kind][y:y+h,x:x+w] = True


def _objects(masks):
    result = []
    for kind in KINDS:
        labels,_ = ndimage.label(masks[kind])  # four-connected, no wrap or diagonal joins
        for y,x in ndimage.find_objects(labels):
            result.append(ObjectLabel(kind,(x.start,y.start,x.stop-x.start,y.stop-y.start)))
    return result


def pooled_mspacman_ram_objects(before_previous, previous, current):
    masks = _masks()
    for now,before in ((previous,before_previous),(current,previous)):
        for obj in decode_visible_ram(now,before):
            _add(masks,obj)
    return _objects(masks)


class MsPacmanPooledPixelJudge:
    def __init__(self):
        self.raw = MsPacmanPixelJudge()

    def step(self, previous, current):
        masks = _masks()
        for rgb,other in ((previous,current),(current,previous)):
            for obj in self.raw.step(rgb):
                x,y,w,h = obj.box
                own,competing = rgb[y:y+h,x:x+w],other[y:y+h,x:x+w]
                palette = np.any(np.stack([np.all(own==c,axis=2) for c in COLOURS[obj.kind]]),axis=0)
                contributing = palette & (np.all(own==competing,axis=2) | np.any(own>competing,axis=2))
                yy,xx = np.nonzero(contributing)
                if len(xx):
                    box = (x+int(xx.min()),y+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
                    _add(masks,ObjectLabel(obj.kind,box))
        return _objects(masks)
