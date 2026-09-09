"""Regression witnesses from immutable development 17511; no emulator."""
import numpy as np
import pytest

from ragnarok.evaluation.asteroids_player_v70 import (
    classify_red_pixels, decode_player_missile_ram, player_ram_pixels,
    screen_player_missile_objects,
)


def ram(**changes):
    result = np.zeros(128,dtype=np.uint8)
    for key,value in {57:212,60:76,73:151,74:43,86:224,87:224,
                      107:8,109:145,110:223,111:43}.items():
        result[key] = value
    for key,value in changes.items(): result[int(key[1:])] = value
    return result


def test_render_scratch_suppresses_birth_frame_210():
    # One-based210: logical y is already43 while the render scratch is absent.
    assert decode_player_missile_ram(ram(r111=224)) == []
    assert [o.box for o in decode_player_missile_ram(ram())] == [(48,104,6,10)]


def test_first_explosion_is_one_player_not_six_missiles():
    # Actual six disconnected red dots in zero-based213.
    observed = {(x,y+dy) for x,y in [(49,104),(52,106),(50,108),
                                    (48,110),(52,110),(50,112)] for dy in (0,1)}
    r = ram(r57=214,r94=129,r107=0,r109=175)
    assert player_ram_pixels(r) == observed
    player,missile,fallback = classify_red_pixels(observed)
    assert player == [(48,104,5,10)]
    assert missile == fallback == []


def test_wrapped_ship_preserves_top_and_bottom_visible_pieces():
    # Zero-based6447: top glyph and bottom five-pixel row are both the ship.
    r = ram(r57=240,r60=24,r73=89,r74=255,r76=105,r109=165,r111=88)
    observed = {(x,y) for y in range(18,22) for x in (52,53,54)}
    observed |= {(53,y) for y in range(22,26)}
    observed |= {(x,194) for x in range(51,56)}
    assert player_ram_pixels(r) == observed
    player,missile,fallback = classify_red_pixels(observed)
    assert sorted(player) == [(51,194,5,1),(52,18,3,8)]
    assert missile == fallback == []


def test_current_blue_shot_can_be_fully_occluded_at_birth():
    # Zero-based4961: RAM missile exists at100,105 but ship hides both pixels.
    r = ram(r57=34,r60=60,r73=34,r74=41,r76=162,r83=2,r84=226,
            r87=44,r107=9,r111=41)
    assert not [o for o in decode_player_missile_ram(r) if o.kind=='player_missile']


def test_missile_x_wrap_is_retained():
    r = ram(r111=224,r83=218,r86=43)
    assert [o.box for o in decode_player_missile_ram(r)] == [(1,103,1,2)]


def test_rgb_checker_uses_colour_phase_and_excludes_hud():
    rgb = np.zeros((210,160,3),dtype=np.uint8)
    rgb[1:10,3:9] = (240,128,128)
    rgb[40:42,7] = (117,181,239)
    assert [o.box for o in screen_player_missile_objects(rgb)] == [(7,40,1,2)]
    rgb[50:54,20:24] = (180,122,48)
    assert screen_player_missile_objects(rgb) == []


def test_unknown_pointer_fails():
    with pytest.raises(ValueError,match='pointer'):
        player_ram_pixels(ram(r109=200))
