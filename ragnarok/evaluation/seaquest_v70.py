"""Development RAM geometry and independent RGB checker for Seaquest actors.

Coordinate/formation/palette facts adapted from OCAtari MIT commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/seaquest.py and vision/seaquest.py.
License: docs/licenses/OCAtari-MIT.txt. This baseline is not yet calibrated.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="seaquest_raw_visible_actor_baseline_v0"
KINDS=("player","shark","submarine","diver","enemy_missile","player_missile")
ROI=(0,33,160,165)
PLAYER_COLOURS=((187,187,53),(236,236,236))
SHARK_COLOURS=((92,186,92),(198,108,58),(160,171,79),(72,160,72),(198,89,179))
SUBMARINE_COLOUR=(170,170,170)
BLUE=(66,72,200)


@dataclass(frozen=True)
class SlottedLabel(ObjectLabel):
    # A RAM allocation slot, not an assertion of continuous identity across respawns.
    slot: str


def _clipped(kind,slot,x,y,w,h):
    left,top,right,bottom=max(ROI[0],x),max(ROI[1],y),min(ROI[2],x+w),min(ROI[3],y+h)
    if right>left and bottom>top:
        return SlottedLabel(kind,(left,top,right-left,bottom-top),slot)
    return None


def decode_seaquest_ram(ram):
    if not isinstance(ram,np.ndarray) or ram.dtype!=np.uint8 or ram.shape!=(128,):
        raise ValueError("Expected128 uint8 RAM bytes")
    r=list(map(int,ram))
    objects=[]
    if not 0<r[105]<15:
        objects.append(_clipped("player","player",r[70],r[97]+33,16,11))
    for lane in range(4):
        submarine=3<r[89+lane]%8<7
        for slot in range(3):
            if (r[36+lane]>>(2-slot))&1:
                x=(r[30+lane]+16*slot)%256
                kind="submarine" if submarine else "shark"
                y=141-24*lane+(0 if submarine else r[93]-4)
                objects.append(_clipped(kind,f"enemy:{lane}:{slot}",x,y,8,11 if submarine else 7))
    for i in range(4):
        lane=3-i
        if 0<r[74-i]<160:
            missile=3<r[89+lane]%8<7
            kind="enemy_missile" if missile else "diver"
            objects.append(_clipped(kind,f"blue:{lane}",r[74-i]+(3 if missile else 0),
                                    141-24*lane+(4 if missile else 0),6 if missile else 8,4 if missile else 11))
    if r[60]>=2 and r[118]<160:
        objects.append(_clipped("submarine","surface",r[118],45,8,11))
    if 0<r[103]<160:
        objects.append(_clipped("player_missile","player_missile",r[103],r[97]+40,8,1))
    return [o for o in objects if o is not None]


def _components(rgb,colours):
    left,top,right,bottom=ROI
    mask=np.zeros((bottom-top,right-left),dtype=bool)
    for colour in colours:
        mask |= np.all(rgb[top:bottom,left:right]==colour,axis=2)
    labels,_=ndimage.label(mask,np.ones((3,3),dtype=np.uint8))
    return [(left+x.start,top+y.start,x.stop-x.start,y.stop-y.start)
            for y,x in ndimage.find_objects(labels)]


def screen_seaquest_objects(rgb):
    """Palette/shape connected components only; no RAM or candidate search boxes.

The actor ROI excludes score, life and oxygen HUD. Fragments, occlusions and
unseen types remain explicit calibration concerns rather than inferred labels.
"""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(210,160,3):
        raise ValueError("Expected raw210x160RGB")
    groups={kind:[] for kind in KINDS}
    for box in _components(rgb,PLAYER_COLOURS):
        if box[3]>6:
            groups["player"].append(box)
        elif box[3]==1 and box[2]>=4:
            groups["player_missile"].append(box)
    groups["shark"]=_components(rgb,SHARK_COLOURS)
    groups["submarine"]=_components(rgb,(SUBMARINE_COLOUR,))
    for box in _components(rgb,(BLUE,)):
        groups["diver" if box[3]>5 else "enemy_missile"].append(box)
    return [ObjectLabel(kind,box) for kind in KINDS for box in groups[kind]]
