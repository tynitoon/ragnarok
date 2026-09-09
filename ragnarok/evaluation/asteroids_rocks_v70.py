"""Development-only rock raster hypotheses, separate from nominal baseline."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.asteroids_v70 import ASTEROID_ROWS,ASTEROID_COLOURS,BLUE,x_position,SlottedLabel
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION='asteroids_rock_raster_development_v1'
# Dominant complete glyphs observed on17511; each row occupies two scanlines.
GLYPHS=(
    (16,(768,2016,2040,8184,8190,32766,65535,32766,65535,32766,4092,4092,1008,240)),
    (16,(4080,16380,16383,4095,16383,65535,65532,65532,32766,32766,8190,4092,4092,1008)),
    (8,(24,60,126,255,255,126,12)),
    (4,(6,15,15,2)),
)


@dataclass(frozen=True)
class RockEntry:
    slot:int
    glyph:int
    x:int
    y:int
    top_clip:int


def rock_entries(ram):
    r=np.asarray(ram)
    if r.shape!=(128,) or r.dtype!=np.uint8:raise ValueError('RAM must be uint8[128]')
    r=r.astype(int)
    if not r[57]&1:return []
    result=[]
    for row in ASTEROID_ROWS:
        if not r[row+18] or r[row]==224:continue
        size=int(r[row+36])&127
        glyph=(int(bool(size&16)) if size<32 else 2 if size<48 else 3)
        yy=int(r[row]) if r[row]<128 else int(r[row])-256
        small=glyph>=2;cohort=int(row>=12)
        result.append(RockEntry(row,glyph,(x_position(r[row+18])-int(small))%160,
                                24+2*yy+cohort-2*int(small),18+cohort))
    return result


def rock_support(entry):
    mask=np.zeros((210,160),dtype=bool)
    width,rows=GLYPHS[entry.glyph]
    for row,bits in enumerate(rows):
        for repeat in range(2):
            y=entry.y+2*row+repeat
            if entry.top_clip<=y<194:
                for dx in range(width):
                    if bits&(1<<(width-1-dx)):mask[y,(entry.x+dx)%160]=True
    return mask


def _bands(xs,ys):
    cols=np.unique(xs)
    result=[]
    for group in np.split(cols,np.where(np.diff(cols)>1)[0]+1):
        if not len(group):continue
        left,right=int(group[0]),int(group[-1])+1
        subset=ys[(xs>=left)&(xs<right)]
        result.append((left,int(subset.min()),right-left,int(subset.max()-subset.min()+1)))
    return result


def decode_rocks(ram):
    result=[]
    for entry in rock_entries(ram):
        ys,xs=np.where(rock_support(entry))
        for i,box in enumerate(_bands(xs,ys)):
            result.append(SlottedLabel('asteroid',box,f'{entry.slot}:{i}'))
    return result


def _rock_phase(view):
    if any(np.any(np.all(view==colour,axis=2)) for colour in ASTEROID_COLOURS if colour!=BLUE):
        return True
    labels,_=ndimage.label(np.all(view==BLUE,axis=2),np.ones((3,3),dtype=np.uint8))
    return any(sx.stop-sx.start>2 or sy.stop-sy.start>3 for sy,sx in ndimage.find_objects(labels))


def screen_rocks(rgb):
    """RGB-only horizontal torus connectivity, then one box per visible band.

    Joining across the screen edge groups a rock's disconnected narrow edge
    fragments through its opposite-side body. Interior disconnected fragments
    remain in the output; no retrospective state or RAM chooses a grouping.
    """
    rgb=np.asarray(rgb)
    if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8:raise ValueError('RGB must be uint8[210,160,3]')
    view=rgb[16:194]
    if not _rock_phase(view):return []
    result=[]
    for colour in ASTEROID_COLOURS:
        labels,count=ndimage.label(np.all(view==colour,axis=2),np.ones((3,3),dtype=np.uint8))
        if not count:continue
        parents=list(range(count+1))
        def root(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]];i=parents[i]
            return i
        for y in range(len(labels)):
            left=int(labels[y,0])
            if left:
                for yy in range(max(0,y-1),min(len(labels),y+2)):
                    right=int(labels[yy,-1])
                    if right:parents[root(right)]=root(left)
        mapping=np.asarray([root(i) for i in range(count+1)],dtype=np.int32)
        groups=mapping[labels]
        for group in np.unique(mapping[1:]):
            ys,xs=np.where(groups==group)
            for box in _bands(xs,ys+16):result.append(ObjectLabel('asteroid',box))
    return result
