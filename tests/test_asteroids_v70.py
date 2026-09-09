import numpy as np
import pytest

from ragnarok.evaluation.asteroids_v70 import decode_asteroids_ram, screen_asteroids_objects, x_position


def test_nominal_start_and_signed_encoding():
    # Pinned documented ship start: encoded29,41 is screen84,100.
    ram = np.zeros(128, dtype=np.uint8)
    ram[73:75] = (29, 41)
    assert [(o.kind, o.box) for o in decode_asteroids_ram(ram)] == [("player", (84, 100, 6, 10))]
    assert x_position(0) == 64 and x_position(1) == 4
    assert x_position(16) == 81 and x_position(144) == 89
    with pytest.raises(ValueError):
        x_position(-1)


def test_visibility_and_border_clip_keep_allocation():
    ram = np.zeros(128, dtype=np.uint8)
    ram[74] = 224
    ram[3], ram[21], ram[39] = 80, 10, 0
    # Encoded10 -> x157; y184. Nominal16x28 is clipped to3x10.
    labels = decode_asteroids_ram(ram)
    assert [(o.kind, o.box, o.slot) for o in labels] == [("asteroid", (157, 184, 3, 10), "3")]
    assert all(type(value) is int for o in labels for value in o.box)
    ram[3] = 128
    assert decode_asteroids_ram(ram) == []


def test_shared_palette_split_and_hud_exclusion():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[5:15, 68:80] = (184, 50, 50)
    rgb[20:30, 10:16] = (240, 128, 128)
    rgb[40:42, 50] = (240, 128, 128)
    rgb[60:62, 60] = (117, 181, 239)
    rgb[100:108, 80:84] = (117, 181, 239)
    # A tiny asteroid of an unshared colour is retained, not silently dropped.
    rgb[120, 100] = (180, 122, 48)
    actual = {(o.kind, o.box) for o in screen_asteroids_objects(rgb)}
    assert actual == {("player", (10, 20, 6, 10)), ("player_missile", (50, 40, 1, 2)),
                      ("player_missile", (60, 60, 1, 2)), ("asteroid", (80, 100, 4, 8)),
                      ("asteroid", (100, 120, 1, 1))}
