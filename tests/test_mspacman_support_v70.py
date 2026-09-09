import numpy as np

from ragnarok.evaluation.mspacman_support_v70 import actor_support,player_support
from ragnarok.evaluation.mspacman_visible_v70 import decode_visible_ram


def state():
    ram=np.zeros(128,dtype=np.uint8)
    ram[10],ram[16],ram[55],ram[56]=100,90,2,3
    ram[33]=255
    return ram


def test_ghost_eye_holes_remain_transparent_and_reflection_is_current():
    prior=state();prior[33]=0;prior[35]=0x42;prior[41]=0;prior[47]=161
    current=prior.copy();current[30]=16
    mask=actor_support(current,prior)
    # Decoded body starts at(20,20). Recorded normal right/left eye holes
    # expose the background or a pill; using a filled rectangle would fail.
    assert mask[20,22] and not mask[22,22]
    assert not mask[29,20] and mask[29,21]
    current[30]=0
    reflected=actor_support(current,prior)
    assert reflected[29,20] and not reflected[29,21]
    assert np.array_equal(mask[20:30,20:28][:,::-1],reflected[20:30,20:28])


def test_same_episode_predecessor_and_cold_fallback_are_explicit():
    prior=state();current=state();current[10]=140
    assert np.array_equal(player_support(current,prior),player_support(prior))
    assert not np.array_equal(player_support(current,prior),player_support(current))


def test_fully_occluded_pill_is_removed_but_surviving_pixels_keep_the_slot():
    ram=state();ram[59]=64  # First row, first column: x8,y7,w4,h2.
    ram[33]=0;ram[35]=0x30;ram[41]=0;ram[47]=178  # Body x0,y3; pill beyond it.
    visible=[o for o in decode_visible_ram(ram) if o.kind=="pill"]
    assert [o.box for o in visible]==[(8,7,4,2)]
    # Put the normal body's solid middle rows across the pill. Reflection
    # does not fill eye holes; this is genuinely its solid body section.
    ram[35]=0x41;ram[47]=179  # 12-4=8, y=2: pill is rows5..6, both fully occupied.
    assert not any(o.kind=="pill" for o in decode_visible_ram(ram))
    ram[47]=182  # y=-1: alternating feet leave two pixels of the pill at y8.
    remaining=[o for o in decode_visible_ram(ram) if o.kind=="pill"]
    assert [(o.slot,o.box) for o in remaining]==[("0:0",(9,8,3,1))]
