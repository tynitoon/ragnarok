import numpy as np
from ragnarok.evaluation.asteroids_rocks_v70 import RockEntry,rock_support,screen_rocks,rock_entries


def test_torus_joins_separated_edge_fragments_via_opposite_body():
    mask=rock_support(RockEntry(12,0,145,89,19))
    rgb=np.zeros((210,160,3),dtype=np.uint8);rgb[mask]=(214,214,214)
    boxes=[o.box for o in screen_rocks(rgb)]
    assert sorted(boxes)==[(0,101,1,6),(145,89,15,28)]


def test_signed_top_exit_is_not_the_inactive_sentinel():
    ram=np.zeros(128,dtype=np.uint8);ram[57]=1;ram[3]=255;ram[21]=71;ram[39]=12
    entry=rock_entries(ram)[0]
    assert (entry.x,entry.y,entry.top_clip)==(36,22,18)
    ram[3]=224
    assert rock_entries(ram)==[]
    ram[3]=255;ram[57]=2
    assert rock_entries(ram)==[]


def test_blue_missiles_do_not_invent_a_rock_frame():
    rgb=np.zeros((210,160,3),dtype=np.uint8);rgb[100:102,80]=(117,181,239)
    assert screen_rocks(rgb)==[]
    rgb[40:44,30:34]=(180,122,48)
    assert sorted(o.box for o in screen_rocks(rgb))==[(30,40,4,4),(80,100,1,2)]
