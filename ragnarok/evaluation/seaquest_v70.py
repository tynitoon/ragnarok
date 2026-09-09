"""Development RAM geometry and independent RGB checker for Seaquest actors.

Coordinate/formation/palette facts adapted from OCAtari MIT commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/seaquest.py and vision/seaquest.py.
License: docs/licenses/OCAtari-MIT.txt. This baseline is not yet calibrated.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="seaquest_raw_development_fragments_and_contacts_v2"
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
    if 0<r[103]<160 and not 53<=r[97]+40<57:
        objects.append(_sprite_box("player_missile","player_missile",r[103],r[97]+40,("########",)))
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


def _merge_fragments(boxes,max_width,max_height):
    """Join support fragments only when their complete union fits one sprite.

No RAM boxes are consulted. Shark formations have eight blank pixels between
full sprites, so two actors cannot fit inside an eight-pixel-wide union.
"""
    boxes=sorted(boxes)
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j]
                x,y=min(a[0],b[0]),min(a[1],b[1])
                right,bottom=max(a[0]+a[2],b[0]+b[2]),max(a[1]+a[3],b[1]+b[3])
                if right-x<=max_width and bottom-y<=max_height:
                    boxes[i]=(x,y,right-x,bottom-y)
                    boxes.pop(j)
                    changed=True
                    break
            if changed:
                break
    return sorted(boxes,key=lambda b:(b[1],b[0]))


def _player_and_projectiles(rgb):
    players,missiles=[],[]
    boxes=_merge_fragments(_components(rgb,PLAYER_COLOURS,vertical_gap=4),16,11)
    for box in boxes:
        x,y,w,h=box
        if h>6:
            if w>16:
                part=rgb[y:y+h,x:x+w]
                mask=np.any(np.stack([np.all(part==c,axis=2) for c in PLAYER_COLOURS]),axis=0)
                splits=[]
                for row in range(h):
                    without=mask.copy()
                    without[row]=False
                    yy,xx=np.where(without)
                    if len(xx) and xx.max()-xx.min()+1<=16 and yy.max()-yy.min()+1>=7:
                        left,right=int(xx.min()),int(xx.max())+1
                        outside=np.flatnonzero(mask[row] & ((np.arange(w)<left)|(np.arange(w)>=right)))
                        if len(outside):
                            splits.append((row,left,right,outside))
                if len(splits)==1:
                    row,left,right,outside=splits[0]
                    support=mask[:,left:right]
                    yy,xx=np.where(support)
                    players.append((x+left+int(xx.min()),y+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)))
                    # A horizontal projectile is eight pixels wide before
                    # screen clipping. Its exposed end anchors the contact.
                    if int(outside.max())<left:
                        mx=x+int(outside.min())
                    else:
                        mx=x+int(outside.max())-7
                    ml,mr=max(8,mx),min(160,mx+8)
                    if mr>ml:
                        missiles.append((ml,y+row,mr-ml,1))
                    continue
            players.append(box)
        elif h==1 and w<=8 and y>=57:
            missiles.append(box)
    return players,missiles


def screen_seaquest_objects(rgb):
    """Palette/shape connected components only; no RAM or candidate search boxes.

The actor ROI excludes score, life and oxygen HUD. Fragments, occlusions and
unseen types remain explicit calibration concerns rather than inferred labels.
"""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(210,160,3):
        raise ValueError("Expected raw210x160RGB")
    groups={kind:[] for kind in KINDS}
    groups["player"],groups["player_missile"]=_player_and_projectiles(rgb)
    groups["shark"]=_merge_fragments(_components(rgb,SHARK_COLOURS),8,7)
    groups["submarine"]=_components(rgb,(SUBMARINE_COLOUR,))
    for box in _components(rgb,(BLUE,)):
        if box[2]>2:
            groups["diver" if box[3]>5 else "enemy_missile"].append(box)
    return [ObjectLabel(kind,box) for kind in KINDS for box in groups[kind]]
