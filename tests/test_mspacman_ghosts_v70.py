import numpy as np
import pytest

from ragnarok.evaluation.mspacman_ghosts_v70 import (
    GHOST_COLOURS, decode_ghost_drawlist, ghost_draw_entries, screen_ghosts,
)


def empty_ram():
    ram = np.zeros(128, dtype=np.uint8)
    ram[33] = 255
    return ram


def test_render_selection_uses_same_episode_previous_snapshot_and_no_dedup():
    prior = empty_ram()
    prior[33] = 1
    prior[35:37], prior[41:43], prior[47:49] = [128, 128], [48, 16], [154, 154]
    current = empty_ram()
    entries = ghost_draw_entries(current, prior)
    assert [(e.index, e.colour_code, e.form_code, e.render_code, e.box) for e in entries] == [
        (0, 3, 0, 48, (11, 27, 8, 10)),
        (1, 1, 0, 16, (11, 27, 8, 10)),
    ]
    assert [o.slot for o in decode_ghost_drawlist(current, prior)] == ["draw:0", "draw:1"]
    assert not decode_ghost_drawlist(current)  # Explicit cold-start fallback.
    current[33] = 3
    assert len(ghost_draw_entries(current, prior)) == 2


def test_gray_and_frightened_use_render_y_and_expose_unclipped_geometry():
    ram = empty_ram()
    ram[33] = 1
    ram[35:37], ram[41:43], ram[47:49] = [6, 6], [65, 85], [201, 35]
    assert [e.box for e in ghost_draw_entries(ram)] == [(72, 3, 8, 10), (72, 169, 8, 10)]
    assert [o.box for o in decode_ghost_drawlist(ram)] == [(72, 3, 8, 10), (72, 169, 8, 2)]


def test_unknown_list_or_colour_fails_and_byte_arithmetic_is_signed():
    ram = empty_ram()
    ram[33] = 4
    with pytest.raises(ValueError, match="length"):
        ghost_draw_entries(ram)
    ram[33], ram[41] = 0, 96
    with pytest.raises(ValueError, match="colour"):
        ghost_draw_entries(ram)
    ram[41], ram[35], ram[47] = 0, 144, 182
    assert ghost_draw_entries(ram)[0].box == (10, -1, 8, 10)
    assert decode_ghost_drawlist(ram)[0].box == (10, 0, 8, 9)


def test_pixel_checker_retains_fragmented_body_multiple_blue_and_gray_not_hud():
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    # Four disconnected islands fit one 8x10 normal body; no RAM supplies its location.
    for y, x in ((30, 20), (30, 27), (39, 20), (39, 27)):
        rgb[y, x] = GHOST_COLOURS[3]
    rgb[70:80, 20:28] = GHOST_COLOURS[5]
    rgb[70:80, 36:44] = GHOST_COLOURS[5]
    rgb[120:130, 60:68] = GHOST_COLOURS[4]
    rgb[173:183, 100:108] = GHOST_COLOURS[4]
    assert sorted(o.box for o in screen_ghosts(rgb)) == [
        (20, 30, 8, 10), (20, 70, 8, 10), (36, 70, 8, 10), (60, 120, 8, 10),
    ]
    # Oversize components remain counted as discrepancies, never dropped by a filter.
    rgb[100:102, 100:120] = GHOST_COLOURS[0]
    assert (100, 100, 20, 2) in [o.box for o in screen_ghosts(rgb)]


def test_input_contracts_reject_wrong_shape_or_dtype():
    with pytest.raises(ValueError, match="RAM"):
        ghost_draw_entries(np.zeros(128))
    with pytest.raises(ValueError, match="RAM"):
        ghost_draw_entries(empty_ram(), np.zeros(127, dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        screen_ghosts(np.zeros((210, 160, 3)))
