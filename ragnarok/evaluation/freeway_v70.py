"""Experimenter-only Freeway geometry and independent RGB calibration checker.

RAM coordinates and palette facts: OCAtari MIT, commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/freeway.py and vision/freeway.py.
License retained in docs/licenses/OCAtari-MIT.txt. Baseline not yet calibrated.
"""
import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="freeway_raw_wrapped_visible_cars_lane4_phase_v2"
CHICKEN_COLOUR=(252,252,84)
CAR_COLOURS=((167,26,26),(180,231,117),(105,105,15),(228,111,111),(24,26,167),
             (162,98,33),(84,92,214),(184,50,50),(135,183,84),(210,210,64))
KINDS=("player","chicken_2")+tuple(f"car_{i}" for i in range(10))
ROI=(0,22,160,201)


def _clipped(kind,x,y,w,h):
    left,top,right,bottom=max(ROI[0],x),max(ROI[1],y),min(ROI[2],x+w),min(ROI[3],y+h)
    return ObjectLabel(kind,(left,top,right-left,bottom-top)) if right>left and bottom>top else None


def decode_freeway_ram(ram):
    if not isinstance(ram,np.ndarray) or ram.dtype!=np.uint8 or ram.shape!=(128,):
        raise ValueError("Expected128 uint8 RAM bytes")
    objects=[_clipped("player",44,193-int(ram[14]),6,8),
             _clipped("chicken_2",108,193-int(ram[15]),6,8)]
    for i in range(10):
        # The fastest left-moving lane is empirically one pixel ahead of r-3 in
        # development, compatible with a one-frame phase; internal draw order is unproven.
        x=(int(ram[117-i])-3+int(i==4))%160
        # Development shows horizontal wrap and an eight-pixel left render blank.
        # This projects visible sprite extent, with one-pixel timing/corner residuals.
        left,right=max(8,x),min(160,x+8)
        if right>left:
            objects.append(ObjectLabel(f"car_{i}",(left,28+16*i,right-left,8)))
    return [o for o in objects if o is not None]


def _colour_box(rgb,kind,colour,roi):
    left,top,right,bottom=roi
    ys,xs=np.nonzero(np.all(rgb[top:bottom,left:right]==colour,axis=2))
    if not len(xs):
        return []
    x,y=int(xs.min()),int(ys.min())
    return [ObjectLabel(kind,(left+x,top+y,int(xs.max())-x+1,int(ys.max())-y+1))]


def screen_freeway_objects(rgb):
    """Extent of each semantic colour in its fixed lane/half, with no RAM access.

Disconnected sprite pixels belong to one rendered object. HUD and road borders
are outside the prescribed regions; no candidate box locates the pixel search.
"""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(210,160,3):
        raise ValueError("Expected raw210x160RGB")
    objects=[]
    for kind,left in (("player",0),("chicken_2",80)):
        mask=np.all(rgb[22:201,left:left+80]==CHICKEN_COLOUR,axis=2)
        labels,_=ndimage.label(mask,np.ones((3,3),dtype=np.uint8))
        for y,x in ndimage.find_objects(labels):
            w,h=x.stop-x.start,y.stop-y.start
            # Independent RGB sprite-size range from the pinned vision detector.
            # The baseline full-colour extent incorrectly included yellow road dashes.
            if 4<=w<=10 and 5<=h<=11:
                objects.append(ObjectLabel(kind,(left+x.start,22+y.start,w,h)))
    for i,colour in enumerate(CAR_COLOURS):
        objects+=_colour_box(rgb,f"car_{i}",colour,(0,22+16*i,160,38+16*i))
    return objects
