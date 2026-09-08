import numpy as np

from ragnarok.evaluation.breakout_v70 import COLOURS, decode_breakout_ram, screen_breakout_objects
from ragnarok.evaluation.pooled_breakout_v70 import pooled_breakout_ram_objects, pooled_breakout_screen_objects


def test_raw_identity_and_frame_permutation_preserve_canonical_types():
    a=np.zeros((210,160,3),dtype=np.uint8)
    a[189:193,50:66]=COLOURS[0]
    a[140:144,90:92]=COLOURS[0]
    a[87:93,8:40]=COLOURS[5]
    b=a.copy()
    b[140:144,90:92]=0
    b[144:148,94:96]=COLOURS[0]
    assert pooled_breakout_screen_objects(a,a)==screen_breakout_objects(a)
    assert pooled_breakout_screen_objects(a,b)==pooled_breakout_screen_objects(b,a)
    ram=np.zeros(128,dtype=np.uint8)
    ram[72],ram[99],ram[101]=100,100,120
    assert pooled_breakout_ram_objects(ram,ram)==decode_breakout_ram(ram)


def test_same_colour_different_types_survive_exact_rgb_ties():
    a=np.zeros((210,160,3),dtype=np.uint8)
    b=a.copy()
    a[189:193,50:52]=COLOURS[0]  # compact ball in sourceA
    b[189:193,45:61]=COLOURS[0]  # wide paddle in sourceB
    assert {o.kind for o in pooled_breakout_screen_objects(a,b)}=={"player","ball"}


def test_hybrid_max_colour_remains_attributable_and_white_can_remove_source():
    a=np.zeros((210,160,3),dtype=np.uint8)
    b=a.copy()
    a[140:144,50:52]=COLOURS[0]
    b[140:144,50:52]=(142,142,142)
    objects=pooled_breakout_screen_objects(a,b)
    assert [o.box for o in objects if o.kind=="ball"]==[(50,140,2,4)]
    assert tuple(np.maximum(a,b)[140,50])==(200,142,142)
    b[140:144,50:52]=255
    assert pooled_breakout_screen_objects(a,b)==[]


def test_brick_intervals_merge_in_the_pooled_window():
    a=np.zeros((210,160,3),dtype=np.uint8)
    b=a.copy()
    a[87:93,8:16]=COLOURS[5]
    b[87:93,16:24]=COLOURS[5]
    assert [o.box for o in pooled_breakout_screen_objects(a,b)]==[(8,87,16,6)]
