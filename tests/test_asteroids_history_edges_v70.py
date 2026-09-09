"""Chronology and clipped-glyph witnesses, with prior evidence preserved."""
import numpy as np
import pytest

from tests.test_asteroids_deblend_v70 import frame
from ragnarok.evaluation.asteroids_history_v70 import rendered_rock_ram,decode_history_rocks
from ragnarok.evaluation.asteroids_deblend_edges_v70 import screen_boundary_deblended_rocks,deblend_boundary_component
from scripts.calibrate_breakout_v70 import compare_boxes


def test_first_p0_uses_preupdate_x_without_mutating_or_delaying_other_positions():
    _,previous=frame(530);_,ram=frame(531)
    before=ram.copy();old=previous.copy();effective=rendered_rock_ram(ram,previous)
    assert int(effective[21])==234 and int(ram[21])==250
    assert np.array_equal(effective[22:39],ram[22:39])
    assert np.array_equal(ram,before) and np.array_equal(previous,old)


def test_new_destruction_flag_follows_already_rendered_solid_glyph():
    rgb,ram=frame(213);_,previous=frame(212)
    effective=rendered_rock_ram(ram,previous)
    assert int(ram[39])==92 and int(effective[39])==24
    assert (33,114,16,28) in {o.box for o in decode_history_rocks(ram,previous)}


def test_nonadjacent_phase_rejected_and_cold_fallback_explicit():
    _,ram=frame(531)
    with pytest.raises(ValueError,match='opposite phase'):
        rendered_rock_ram(ram,ram)
    assert np.array_equal(rendered_rock_ram(ram,None),ram)


@pytest.mark.parametrize('number',[2193,3329,3509,5797,7023,7203])
def test_two_clipped_same_colour_glyphs_have_exact_visible_pieces(number):
    rgb,ram=frame(number);_,previous=frame(number-1)
    a=decode_history_rocks(ram,previous);b=screen_boundary_deblended_rocks(rgb)
    assert len(a)==len(b)
    assert compare_boxes(a,b)==(len(a),len(a),len(a))


def test_clipped_component_decomposition_preserves_every_pixel():
    rgb,_=frame(2193)
    # The overlapping grey objects in this archived boundary frame.
    mask=np.all(rgb==(214,214,214),axis=2)
    mask[:16]=False;mask[194:]=False
    from ragnarok.evaluation.asteroids_deblend_v70 import torus_masks
    split=False
    for component in torus_masks(mask):
        pieces,reason=deblend_boundary_component(component)
        total=np.zeros_like(mask)
        for piece in pieces:
            assert not np.any(total&piece)
            total|=piece
        assert np.array_equal(total,component)
        split|=reason=='unique_clipped_pair'
    assert split


def test_roi_outside_pixels_do_not_create_an_object():
    rgb=np.zeros((210,160,3),np.uint8);rgb[194:,30:50]=(214,214,214)
    assert screen_boundary_deblended_rocks(rgb)==[]
