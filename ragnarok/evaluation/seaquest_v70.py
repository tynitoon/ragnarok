"""Development RAM geometry and independent RGB checker for Seaquest actors.

Coordinate/formation/palette facts adapted from OCAtari MIT commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/seaquest.py and vision/seaquest.py.
License: docs/licenses/OCAtari-MIT.txt. This baseline is not yet calibrated.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="seaquest_raw_development_sprite_support_v1"
KINDS=("player","shark","submarine","diver","enemy_missile","player_missile")
ROI=(0,33,160,165)
PLAYER_COLOURS=((187,187,53),(236,236,236))
SHARK_COLOURS=((92,186,92),(198,108,58),(160,171,79),(72,160,72),(198,89,179))
SUBMARINE_COLOUR=(170,170,170)
BLUE=(66,72,200)

# Empirical sprite support from development17311 only. These masks belong to
# the external RAM evaluator, never to the agent or the generic RGB reference.
# Full sprites have animation variants; boundaries must use occupied pixels,
# rather than intersecting an 8xN rectangle with the screen edge.
SHARK_SUPPORT=(
    ("....#...","....##..","##.#####",".#######",".#######","##.##...","...#...."),
    ("....#...","#...##..","##.#####",".#######",".######.","##.##.##","#..#...#"),
)
DIVER_SUPPORT=(
    ("......#.","......#.","....##..","....####","...##...","####....","..#.....",".#......","#......."),
    ("......#.","......#.","....##..","....####","...##...","..##....",".##.....","#.......","........"),
)
PLAYER_SUPPORT=("....##..........",)+("....####........",)*4+(
    "############..##","################","################",
    "################","..##############","..............##")


def _sprite_box(kind,slot,x,y,rows,*,flip=False,surface=False):
    points=[]
    for dy,row in enumerate(rows):
        for dx,pixel in enumerate(row[::-1] if flip else row):
            px,py=x+dx,y+dy
            if (pixel=="#" and 8<=px<160 and ROI[1]<=py<ROI[3]
                    and not (surface and 53<=py<57)):
                points.append((px,py))
    if not points:
        return None
    xs,ys=zip(*points)
    return SlottedLabel(kind,(min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1),slot)


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
    if r[105]==0 or (r[105]>=15 and r[105]%2==1):
        objects.append(_sprite_box("player","player",r[70],r[97]+33,PLAYER_SUPPORT,
                                   flip=r[86]==0,surface=True))
    for lane in range(4):
        submarine=3<r[89+lane]%8<7
        for slot in range(3):
            if (r[36+lane]>>(2-slot))&1:
                x=(r[30+lane]+16*slot)%256
                kind="submarine" if submarine else "shark"
                y=141-24*lane+(0 if submarine else r[93]-4)
                if submarine:
                    objects.append(_clipped(kind,f"enemy:{lane}:{slot}",x,y,8,11))
                else:
                    rows=SHARK_SUPPORT[int(r[89+lane]%8!=0)]
                    objects.append(_sprite_box(kind,f"enemy:{lane}:{slot}",x,y,rows,
                                               flip=bool(r[89+lane]&8)))
    for i in range(4):
        lane=3-i
        if 0<r[74-i]<160:
            missile=3<r[89+lane]%8<7
            kind="enemy_missile" if missile else "diver"
            if missile:
                objects.append(_clipped(kind,f"blue:{lane}",r[74-i]+3,145-24*lane,6,4))
            else:
                rows=DIVER_SUPPORT[int(r[89+lane]%8==1)]
                objects.append(_sprite_box(kind,f"blue:{lane}",r[74-i],142-24*lane,rows,
                                           flip=bool(r[89+lane]&8)))
    if r[60]>=2 and r[118]<160:
        objects.append(_clipped("submarine","surface",r[118],45,8,11))
    if 0<r[103]<160:
        objects.append(_clipped("player_missile","player_missile",r[103],r[97]+40,8,1))
    return [o for o in objects if o is not None]


def _components(rgb,colours,*,vertical_gap=0):
    left,top,right,bottom=ROI
    mask=np.zeros((bottom-top,right-left),dtype=bool)
    for colour in colours:
        mask |= np.all(rgb[top:bottom,left:right]==colour,axis=2)
    if vertical_gap:
        # Group pieces separated by the four-row surface stripe. Only original
        # support contributes to the returned boxes, never filled pixels.
        connected=ndimage.binary_closing(mask,structure=np.ones((vertical_gap+1,1),bool)) | mask
    else:
        connected=mask
    labels,_=ndimage.label(connected,np.ones((3,3),dtype=np.uint8))
    labels=np.where(mask,labels,0)
    return [(left+x.start,top+y.start,x.stop-x.start,y.stop-y.start)
            for item in ndimage.find_objects(labels) if item is not None for y,x in (item,)]


def screen_seaquest_objects(rgb):
    """Palette/shape connected components only; no RAM or candidate search boxes.

The actor ROI excludes score, life and oxygen HUD. Fragments, occlusions and
unseen types remain explicit calibration concerns rather than inferred labels.
"""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(210,160,3):
        raise ValueError("Expected raw210x160RGB")
    groups={kind:[] for kind in KINDS}
    for box in _components(rgb,PLAYER_COLOURS,vertical_gap=4):
        if box[3]>6:
            groups["player"].append(box)
        elif box[3]==1 and box[2]>=4:
            groups["player_missile"].append(box)
    groups["shark"]=_components(rgb,SHARK_COLOURS)
    groups["submarine"]=_components(rgb,(SUBMARINE_COLOUR,))
    for box in _components(rgb,(BLUE,)):
        if box[2]>2:
            groups["diver" if box[3]>5 else "enemy_missile"].append(box)
    return [ObjectLabel(kind,box) for kind in KINDS for box in groups[kind]]
