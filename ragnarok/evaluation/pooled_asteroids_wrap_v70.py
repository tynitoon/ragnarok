"""Temporal V1: the already modeled top wrap also offsets explosion pointers.

Only the RAM-side recognition of wrapped explosion glyphs is extended. The
raw and temporal V0 sources, RGB decoder, ROI and component definition remain
unchanged. Any other unseen pointer still raises instead of hiding a frame.
"""
import numpy as np
from ragnarok.evaluation.pooled_asteroids_v70 import (
    KINDS, ROI, _masks, _add, _objects, pooled_asteroids_screen_objects,
)
from ragnarok.evaluation.asteroids_player_roi_v70 import EXPLOSION_POINTER, decode_player_missile_ram
from ragnarok.evaluation.asteroids_history_v70 import decode_history_rocks

REVISION='asteroids_two_raw_spatial_envelopes_wrapped_explosion_v1'


def source_ram_objects(current,previous):
    ram=np.asarray(current)
    if ram.shape!=(128,) or ram.dtype!=np.uint8:
        raise ValueError('Expected uint8 RAM[128]')
    y=int(ram[74]);y-=256 if y>=128 else 0
    corrected=ram
    if (not int(ram[57])&1 and int(ram[110])==223 and -5<=y<0
            and int(ram[109])+5+y in EXPLOSION_POINTER):
        corrected=ram.copy()
        corrected[109]=int(ram[109])+5+y
    return decode_player_missile_ram(corrected)+decode_history_rocks(ram,previous)


def pooled_asteroids_ram_objects(before_previous,previous,current):
    masks=_masks()
    for now,before in ((previous,before_previous),(current,previous)):
        for obj in source_ram_objects(now,before):
            _add(masks,obj)
    return _objects(masks)
