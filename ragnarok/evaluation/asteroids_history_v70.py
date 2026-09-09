"""Development renderer chronology from adjacent RAM snapshots, never RGB.

The first P0 object is positioned before its ordinary horizontal update. A
collision flag can also be committed after the frame's solid glyph was loaded.
Only contiguous same-episode snapshots may be supplied by the external scorer.
"""
import numpy as np
from ragnarok.evaluation.asteroids_v70 import ASTEROID_ROWS,x_position
from ragnarok.evaluation.asteroids_occlusion_v70 import decode_occluded_rocks

REVISION='asteroids_rock_ram_chronology_development_v4'


def rendered_rock_ram(ram,previous=None):
    r=np.asarray(ram)
    if r.shape!=(128,) or r.dtype!=np.uint8:raise ValueError('Expected uint8 RAM[128]')
    out=r.copy()
    if previous is None or not int(r[57])&1:return out
    p=np.asarray(previous)
    if p.shape!=(128,) or p.dtype!=np.uint8:raise ValueError('Expected previous uint8 RAM[128]')
    if int(p[57])&1:raise ValueError('Previous snapshot must be the adjacent opposite phase')
    for row in ASTEROID_ROWS:
        if not r[row+18] or r[row]==224 or not p[row+18] or p[row]==224:continue
        old,new=int(p[row+36])&127,int(r[row+36])&127
        # The newly set destruction bit follows that frame's solid rendering.
        if old<64<=new:out[row+36]=p[row+36]
    if r[3]!=224 and p[3]!=224 and r[21] and p[21]:
        y=int(r[3]); y-=256 if y>=128 else 0
        old_y=int(p[3]);old_y-=256 if old_y>=128 else 0
        dx=(x_position(r[21])-x_position(p[21])+80)%160-80
        if abs(y-old_y)<=1 and abs(dx)<=1 and int(out[39])==int(p[39]):
            out[21]=p[21]
    return out


def decode_history_rocks(ram,previous=None):
    return decode_occluded_rocks(rendered_rock_ram(ram,previous))
