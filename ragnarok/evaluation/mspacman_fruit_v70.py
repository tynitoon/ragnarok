"""External RAM-only fruit renderer learned on development17431, not reserved.

The special render list interleaves fruit and player. Logical allocation alone
does not imply visible fruit. Two observed types are calibrated candidates;
unobserved types fail explicitly. Glyphs are development pixel measurements,
never agent labels or learned-agent parameters.
"""
import numpy as np
from ragnarok.evaluation.mspacman_render_v70 import _ram, _clipped
from ragnarok.evaluation.mspacman_support_v70 import add_support

REVISION="mspacman_fruit_list_development_v1"
FRUIT_ROWS=(
    (96,152,20,18,38,79,111,246,240,96),
    (0,0,60,118,94,122,110,60,52,24),
)


def fruit_render_entries(current_ram,previous_ram=None):
    current=_ram(current_ram)
    r=current if previous_ram is None else _ram(previous_ram)
    if int(r[34]) not in (0,1):
        raise ValueError("Uncalibrated special render-list length")
    entries=[]
    for i in range(int(r[34])+1):
        kind=int(r[45+i])>>4
        if kind==7:
            continue
        if kind not in range(len(FRUIT_ROWS)):
            raise ValueError("Uncalibrated fruit type")
        pointer=256*int(r[53+i])+int(r[51+i])+int(r[17])
        if pointer!=0xDF14+22*kind:
            raise ValueError("Uncalibrated fruit glyph pointer")
        entries.append((kind,int(r[11])-12,int(r[17])+1))
    if len(entries)>1:
        raise ValueError("Uncalibrated multiple fruit entries")
    return entries


def fruit_render_boxes(current_ram,previous_ram=None):
    result=[]
    for kind,x,y in fruit_render_entries(current_ram,previous_ram):
        # Strawberry glyph has two empty top rows and one empty side column.
        dx,dy,w,h=(0,0,8,10) if kind==0 else (1,2,6,8)
        result.extend(_clipped('fruit',0,x+dx,y+dy,x+dx+w,y+dy+h))
    return result


def fruit_support(current_ram,previous_ram=None):
    canvas=np.zeros((171,160),dtype=bool)
    for kind,x,y in fruit_render_entries(current_ram,previous_ram):
        add_support(canvas,FRUIT_ROWS[kind],x,y)
    return canvas
