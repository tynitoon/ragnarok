"""Uncalibrated functional Ms. Pac-Man hypotheses, development only.

RAM geometry/grid facts and palette attribution: MIT OCAtari, revision
467accd5f3da379d83705b008094ffb787eec8a1, ram/mspacman.py and vision/mspacman.py.
License retained in docs/licenses/OCAtari-MIT.txt. Mutable source object lifecycle
is not reused. Nominal boxes are hypotheses to confront with independent RGB.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from ragnarok.evaluation.pong_v70 import ObjectLabel

REVISION="mspacman_raw_nominal_v0"
KINDS=("player","ghost","fruit","power_pill","pill")
ROI=(0,0,160,171)
PALETTES={"player":((210,164,74),),
          "ghost":((180,122,48),(84,184,153),(198,89,179),(200,72,72),(66,114,194)),
          "fruit":((184,50,50),(162,162,42),(198,108,58),(110,156,66))}
PILL_COLOUR=(228,111,111)
GRID1=tuple(tuple(map(int,row)) for row in (
    '111101111111101111',
    '100101000000101001',
    '111111111111111111',
    '010100100001001010',
    '110111111111111011',
    '010000100001000010',
    '011111100001111110',
    '010000100001000010',
    '110111111111111011',
    '010101010010101010',
    '111101011110101111',
    '100101110011101001',
    '100100010010001001',
    '111111111111111111',
))
GRID2=tuple(tuple(map(int,row)) for row in (
    '111111111111111111',
    '100001000000100001',
    '110111011110111011',
    '010101010010101010',
    '010101111111101010',
    '111100100001001111',
    '100111100001111001',
    '100100100001001001',
    '100101111111101001',
    '111111000000111111',
    '001001011110100100',
    '111101110011101111',
    '100101000000101001',
    '111101111111101111',
))


@dataclass(frozen=True)
class SlottedLabel(ObjectLabel):
    slot: str


def decode_mspacman_ram(ram):
    ram=np.asarray(ram)
    if ram.shape!=(128,) or ram.dtype!=np.uint8:
        raise ValueError("RAM must be uint8[128]")
    r=ram.astype(int)
    result=[]
    def add(kind,slot,x,y,w,h):
        left,top=max(ROI[0],x),max(ROI[1],y)
        right,bottom=min(ROI[2],x+w),min(ROI[3],y+h)
        if left<right and top<bottom:
            result.append(SlottedLabel(kind,(int(left),int(top),int(right-left),int(bottom-top)),str(slot)))
    add("player",0,r[10]-13,r[16]+1,9,10)
    if r[47]>0:
        for i in range(min(4,r[19]+1)):
            add("ghost",i,r[6+i]-13,r[12+i]+1,9,10)
    if r[11]>0 and r[17]>0:
        add("fruit",0,r[11]-13,r[17]+1,9,10)
    for bit,x,y in ((4,148,146),(8,148,14),(16,8,147),(32,8,15)):
        if r[117]&bit:
            add("power_pill",bit,x,y,4,7)
    grid=GRID1 if r[0]==0 else GRID2
    for row in range(14):
        values=r[59+3*row:62+3*row]
        active={17:bool(values[0]&16),0:bool(values[0]&64)}
        for j in range(8):
            active[(4-(j>>1)) if j&1 else (12-(j>>1))]=bool(values[1]&(1<<j))
            active[(13+(j>>1)) if j&1 else (5+(j>>1))]=bool(values[2]&(1<<j))
        for col,present in active.items():
            if present and grid[row][col]:
                x=(8+8*col) if col<=8 else (12+8*col)
                add("pill",f"{row}:{col}",x,7+12*row,4,2)
    return result


def screen_mspacman_objects(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(210,160,3) or rgb.dtype!=np.uint8:
        raise ValueError("RGB must be uint8[210,160,3]")
    view=rgb[ROI[1]:ROI[3],ROI[0]:ROI[2]]
    result=[]
    def components(mask):
        labels,_=ndimage.label(mask,np.ones((3,3),dtype=np.uint8))
        return [(s[1].start+ROI[0],s[0].start+ROI[1],s[1].stop-s[1].start,s[0].stop-s[0].start)
                for s in ndimage.find_objects(labels)]
    for kind,colours in PALETTES.items():
        # All components per colour, including several frightened ghosts.
        for colour in colours:
            result.extend(ObjectLabel(kind,box) for box in components(np.all(view==colour,axis=2)))
    for box in components(np.all(view==PILL_COLOUR,axis=2)):
        result.append(ObjectLabel("power_pill" if box[3]>2 else "pill",box))
    return result
