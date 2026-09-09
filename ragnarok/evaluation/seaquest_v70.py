"""Development RAM geometry and independent RGB checker for Seaquest actors.

Coordinate/formation/palette facts adapted from OCAtari MIT commit
467accd5f3da379d83705b008094ffb787eec8a1, ram/seaquest.py and vision/seaquest.py.
License: docs/licenses/OCAtari-MIT.txt. This baseline is not yet calibrated.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="seaquest_raw_development_visible_fragments_v5"
KINDS=("player","shark","submarine","diver","enemy_missile","player_missile")
ROI=(0,33,160,165)
PLAYER_COLOURS=((187,187,53),(236,236,236))
SHARK_COLOURS=((92,186,92),(198,108,58),(160,171,79),(72,160,72),(198,89,179))
SUBMARINE_COLOUR=(170,170,170)
BLUE=(66,72,200)

# Empirical supports: sharks/divers initially from17311; player/submarine
# refinements from17311/17331/17341, and surface patrol from17361. These belong
# to the external evaluator, never to the agent or generic RGB reference.
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
PLAYER_ANIMATION={
    164: PLAYER_SUPPORT[:5]+("############..##","################","##..##########..",
                            "################","..##############","................"),
    178: PLAYER_SUPPORT[:5]+("############..##","##############..","##..############",
                            "################","..############..","..............##"),
    192: PLAYER_SUPPORT[:5]+("############....","################","##..############",
                            "##############..","..##############","..............##"),
}
SUBMARINE_SUPPORT=(
    ("..##....",)*4+("######.#","########","#.#####.","########",".#######","........","........"),
    ("..##....",)*4+("######.#","#######.","#.######","########",".######.",".......#","........"),
    ("..##....",)*4+("######..","########","#.######","#######.",".#######",".......#","........"),
)
ENEMY_MISSILE_SUPPORT=("..####","......","......","####..")
SURFACE_ANIMATION={
    164: ("..#.....",)+("..##....",)*4+("######.#","########","#.#####."),
    178: ("..#.....",)+("..##....",)*4+("######.#","#######.","#.######"),
    192: ("..#.....",)+("..##....",)*4+("######..","########","#.######"),
}


def _sprite_points(x,y,rows,*,flip=False,surface=False):
    points=set()
    for dy,row in enumerate(rows):
        for dx,pixel in enumerate(row[::-1] if flip else row):
            px,py=x+dx,y+dy
            if (pixel=="#" and 8<=px<160 and ROI[1]<=py<ROI[3]
                    and not (surface and 53<=py<57)):
                points.add((px,py))
    return points


def _points_box(kind,slot,points):
    if not points:
        return None
    xs,ys=zip(*points)
    return SlottedLabel(kind,(min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1),slot)


def _sprite_box(kind,slot,x,y,rows,*,flip=False,surface=False):
    return _points_box(kind,slot,_sprite_points(x,y,rows,flip=flip,surface=surface))


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
    player_points=set()
    projectile_points=set()
    if 0<r[103]<160 and not 53<=r[97]+40<57:
        projectile_points=_sprite_points(r[103],r[97]+40,("########",))
    if r[105]==0 or (r[105]>=15 and r[105]%2==1):
        player_points=_sprite_points(r[70],r[97]+33,PLAYER_ANIMATION.get(r[24],PLAYER_SUPPORT),
                                      flip=r[86]==0,surface=True)
        objects.append(_points_box("player","player",player_points))
    foreground=[set() for _ in range(4)]
    for lane in range(4):
        submarine=3<r[89+lane]%8<7
        for slot in range(3):
            if (r[36+lane]>>(2-slot))&1:
                x=(r[30+lane]+16*slot)%256
                kind="submarine" if submarine else "shark"
                y=141-24*lane+(0 if submarine else r[93]-4)
                if submarine:
                    rows=SUBMARINE_SUPPORT[r[89+lane]%8-4]
                    points=_sprite_points(x,y,rows,flip=not bool(r[89+lane]&8))
                else:
                    rows=SHARK_SUPPORT[int(r[89+lane]%8!=0)]
                    points=_sprite_points(x,y,rows,flip=bool(r[89+lane]&8))
                foreground[lane] |= points
                objects.append(_points_box(kind,f"enemy:{lane}:{slot}",points-player_points))
    for i in range(4):
        lane=3-i
        if 0<r[74-i]<160:
            missile=3<r[89+lane]%8<7
            kind="enemy_missile" if missile else "diver"
            if missile:
                # The two strokes keep the same orientation in both travel
                # directions; edge clipping and occlusion make this observable.
                points=_sprite_points(r[74-i]+2,145-24*lane,ENEMY_MISSILE_SUPPORT)
            else:
                rows=DIVER_SUPPORT[int(r[89+lane]%8==1)]
                points=_sprite_points(r[74-i],142-24*lane,rows,flip=bool(r[89+lane]&8))
            objects.append(_points_box(kind,f"blue:{lane}",points-player_points-projectile_points-foreground[lane]))
    if r[60]>=2 and r[118]<160:
        points=_sprite_points(r[118],45,SURFACE_ANIMATION.get(r[24],SURFACE_ANIMATION[164]),surface=True)
        objects.append(_points_box("submarine","surface",points-player_points-projectile_points))
    objects.append(_points_box("player_missile","player_missile",projectile_points))
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
        if h>1:
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
    # A projectile can fill a one-pixel hole inside the player's silhouette
    # without widening its component. Require an exact sprite match outside
    # the projectile row and a unique eight-pixel interval explaining all
    # additional palette pixels. Shape templates alone never supply a box.
    for px,py,pw,ph in players:
        if pw!=16 or ph not in (10,11) or py<57 or py+11>ROI[3]:
            continue
        row_y=py+7
        if any(my==row_y for _,my,_,_ in missiles):
            continue
        part=rgb[py:py+11,px:px+16]
        actual=np.any(np.stack([np.all(part==c,axis=2) for c in PLAYER_COLOURS]),axis=0)
        full_row=np.any(np.stack([np.all(rgb[row_y]==c,axis=1) for c in PLAYER_COLOURS]),axis=0)
        hypotheses=set()
        for rows in PLAYER_ANIMATION.values():
            for flip in (False,True):
                mask=np.array([[v=="#" for v in (row[::-1] if flip else row)] for row in rows])
                compare=np.arange(11)!=7
                if not np.array_equal(mask[compare],actual[compare]):
                    continue
                body={px+i for i,v in enumerate(mask[7]) if v}
                if not all(full_row[v] for v in body):
                    continue
                residue=set(map(int,np.flatnonzero(full_row[8:160])+8))-body
                if not residue:
                    continue
                for left in range(max(1,min(residue)-7),min(159,max(residue))+1):
                    strip=set(range(max(8,left),min(160,left+8)))
                    if strip-body==residue and all(full_row[v] for v in strip):
                        hypotheses.add((max(8,left),row_y,min(160,left+8)-max(8,left),1))
        if len(hypotheses)==1:
            missiles.append(hypotheses.pop())
    return players,missiles


def _blue_actors(rgb):
    divers,missiles=[],[]
    for base in (69,93,117,141):
        # Bubbles are above the body band. The lane contains at most one blue
        # actor, whose disconnected projectile segments must stay one object.
        yy,xx=np.where(np.all(rgb[base+1:base+10]==BLUE,axis=2))
        if not len(xx):
            continue
        box=(int(xx.min()),base+1+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
        x,y,w,h=box
        if y<base+4 or y+h>base+8:
            divers.append(box)
        elif w>=3:
            missiles.append(box)
        else:
            # A tiny edge remnant is ambiguous by shape. Visible grey actors
            # in this lane provide RGB context, never the RAM lane-type flag.
            grey=np.any(np.all(rgb[base:base+11]==SUBMARINE_COLOUR,axis=2))
            (missiles if grey else divers).append(box)
    return divers,missiles


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
    grey=_components(rgb,(SUBMARINE_COLOUR,))
    groups["submarine"]=_merge_fragments([box for box in grey if box[1]>=57],8,11)+_merge_fragments(
        [box for box in grey if box[1]<57],8,8)
    groups["diver"],groups["enemy_missile"]=_blue_actors(rgb)
    return [ObjectLabel(kind,box) for kind in KINDS for box in groups[kind]]
