"""Optimization must equal the sealed raster judge, including arbitrary bitmaps."""
import numpy as np

from ragnarok.evaluation.fast_pooled_breakout_v70 import pooled_breakout_ram_intervals
from ragnarok.evaluation.pooled_breakout_v70 import pooled_breakout_ram_objects


def test_interval_union_equals_raster_for_random_ram_and_boundary_positions():
    rng=np.random.default_rng(71270)
    for index in range(256):
        a,b=rng.integers(0,256,(2,128),dtype=np.uint8)
        # Exercise every possible player coordinate, varied clipping, full/empty rows.
        a[72],b[72]=index,255-index
        if index%8==0:
            a[:36]=0
        if index%8==1:
            b[:36]=255
        assert pooled_breakout_ram_intervals(a,b)==pooled_breakout_ram_objects(a,b)
        assert pooled_breakout_ram_intervals(b,a)==pooled_breakout_ram_objects(a,b)
