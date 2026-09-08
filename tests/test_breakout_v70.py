"""Boundary geometry and independent assignment checks for the Breakout judge."""
import numpy as np
import pytest

from ragnarok.evaluation.breakout_v70 import decode_breakout_ram, screen_breakout_objects
from ragnarok.evaluation.pong_v70 import ObjectLabel
from scripts.calibrate_breakout_v70 import compare_boxes


@pytest.mark.parametrize("y,height",[(190,4),(191,4),(192,3),(193,2),(194,1),(195,0),(196,0)])
def test_ball_uses_declared_empirical_clip_before_ram_coordinate_clears(y,height):
    ram=np.zeros(128,dtype=np.uint8)
    ram[99],ram[101]=60,y-9
    balls=[o.box for o in decode_breakout_ram(ram) if o.kind=="ball"]
    assert balls==([(11,y,2,height)] if height else [])


def test_ram_unsigned_subtraction_and_shape_are_explicit():
    assert decode_breakout_ram(np.zeros(128,dtype=np.uint8))==[]
    with pytest.raises(ValueError):
        decode_breakout_ram(np.zeros(128,dtype=np.int64))


def test_visible_brick_run_matching_preserves_cardinality_before_distance():
    a=[ObjectLabel("bricks_0",(10,57,8,6)),ObjectLabel("bricks_0",(12,57,8,6))]
    b=[ObjectLabel("bricks_0",(11,57,8,6)),ObjectLabel("bricks_0",(9,57,8,6))]
    assert compare_boxes(a,b)==(2,2,0)
    assert compare_boxes(a,[])==(0,0,0)


def test_screen_checker_ignores_hud_and_permanent_boundary_walls():
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    rgb[5:15,20:60]=(200,72,72)
    rgb[189:193,:8]=(200,72,72)
    rgb[189:193,152:]=(200,72,72)
    rgb[189:193,50:66]=(200,72,72)
    rgb[150:154,80:82]=(200,72,72)
    objects={o.kind:o.box for o in screen_breakout_objects(rgb)}
    assert objects=={"player":(50,189,16,4),"ball":(80,150,2,4)}
