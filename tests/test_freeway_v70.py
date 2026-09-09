import numpy as np
import pytest

from ragnarok.evaluation.freeway_v70 import CAR_COLOURS, CHICKEN_COLOUR, decode_freeway_ram, screen_freeway_objects


def test_uint8_arithmetic_is_signed_and_screen_clipped():
    ram=np.zeros(128,dtype=np.uint8)
    ram[117]=1
    ram[116]=161
    objects={o.kind:o.box for o in decode_freeway_ram(ram)}
    assert objects["car_0"]==(158,28,2,8)
    assert objects["car_1"]==(158,44,2,8)
    assert objects["player"]==(44,193,6,8)
    ram[14]=255
    assert "player" not in {o.kind for o in decode_freeway_ram(ram)}
    with pytest.raises(ValueError):
        decode_freeway_ram(ram.astype(int))


def test_pixel_checker_searches_disconnected_sprite_without_ram_or_hud():
    rgb=np.zeros((210,160,3),dtype=np.uint8)
    rgb[30,5]=CAR_COLOURS[0]
    rgb[35,12]=CAR_COLOURS[0]
    rgb[3:13,40:47]=CAR_COLOURS[3]  # score colour, outside playfield
    rgb[190:198,45:52]=CHICKEN_COLOUR
    rgb[102,12:16]=CHICKEN_COLOUR  # road dashes share the chicken colour
    objects={o.kind:o.box for o in screen_freeway_objects(rgb)}
    assert objects=={"player":(45,190,7,8),"car_0":(5,30,8,6)}
    rgb[22:201,:80]=0
    assert screen_freeway_objects(rgb)==[]


def test_car_wrap_disappearance_and_left_render_clip():
    ram=np.zeros(128,dtype=np.uint8)
    ram[117]=3
    assert "car_0" not in {o.kind for o in decode_freeway_ram(ram)}
    ram[117]=4
    assert {o.kind:o.box for o in decode_freeway_ram(ram)}["car_0"]==(8,28,1,8)
