from pathlib import Path

import numpy as np
import pytest

from ragnarok.evaluation.mspacman_player_list_v70 import (
    PATTERN_BASE, PATTERN_STRIDE, player_list_entry,
    player_list_render_box, player_list_support,
)
from ragnarok.evaluation.mspacman_support_v70 import PLAYER_YELLOW_ROWS, add_support


def player_ram(phase=0, direction=0, index=0, x=88, y=98):
    ram = np.zeros(128, dtype=np.uint8)
    ram[10], ram[16], ram[34] = x, y, index
    ram[45 + index] = 117
    pointer = PATTERN_BASE + PATTERN_STRIDE * (3 * direction + phase) - y
    ram[51 + index], ram[53 + index] = pointer & 255, pointer >> 8
    ram[55:57] = (255, 255)  # Shared scratch must never select player shape.
    return ram


@pytest.mark.parametrize("phase,direction", [(p, d) for d in range(4) for p in range(3)])
def test_all_twelve_pointers_ignore_fruit_scratch(phase, direction):
    ram = player_ram(phase, direction, index=1)
    entry = player_list_entry(ram)
    assert (entry.index, entry.phase, entry.direction) == (1, phase, direction)
    expected = np.zeros((171, 160), dtype=bool)
    add_support(expected, PLAYER_YELLOW_ROWS[phase][direction], 76, 99)
    assert np.array_equal(player_list_support(ram), expected)
    assert all(type(v) is int for v in player_list_render_box(ram)[0].box)


def test_previous_list_controls_presence_and_cold_fallback_is_explicit():
    current = player_ram(2, 0)
    fruit_only = current.copy(); fruit_only[45] = 4
    assert player_list_entry(current, fruit_only) is None
    assert not player_list_support(current, fruit_only).any()
    assert player_list_render_box(current, fruit_only) == []
    assert player_list_entry(fruit_only, current).phase == 2
    assert player_list_entry(current) is not None
    empty = current.copy(); empty[34] = 255
    with pytest.raises(ValueError, match="length"):
        player_list_entry(current, empty)


@pytest.mark.parametrize("fault", ["length", "duplicate", "colour", "pointer"])
def test_unknown_render_cases_fail_instead_of_hiding_a_player(fault):
    ram = player_ram()
    if fault == "length": ram[34] = 2
    if fault == "duplicate": ram[34], ram[46] = 1, 112
    if fault == "colour": ram[45] = 128
    if fault == "pointer": ram[51] += 1
    with pytest.raises(ValueError): player_list_entry(ram)


def test_unsigned_byte_arithmetic_and_visible_roi_clipping():
    ram = player_ram(2, 3, x=8, y=168)
    entry = player_list_entry(ram)
    assert entry.base_address == PATTERN_BASE + 11 * PATTERN_STRIDE
    assert entry.box == (-4, 169, 8, 10)
    support = player_list_support(ram)
    assert support.dtype == bool and support.shape == (171, 160)
    for label in player_list_render_box(ram):
        x, y, w, h = label.box
        assert 0 <= x < x + w <= 160 and 0 <= y < y + h <= 171
    with pytest.raises(ValueError): player_list_entry(np.zeros(128))


def recorded(number):
    folder = Path(__file__).resolve().parents[1] / "craft_v6_out/v70_mspacman_calibration/coverage_development_v1"
    first = ((number - 1) // 256) * 256 + 1
    path = folder / f"frames_{first:05d}_{first + 255:05d}.npz"
    if not path.is_file(): pytest.skip("Optional immutable development archive absent")
    with np.load(path, allow_pickle=False) as archive:
        return archive["rgb"][number-first].copy(), archive["ram"][number-first].copy()


def test_recorded_fruit_only_and_player_second_slot():
    rgb, current = recorded(1315)
    _, previous = recorded(1314)
    assert not np.any(np.all(rgb[:171] == (210, 164, 74), axis=2))
    assert player_list_render_box(current, previous) == []
    rgb, current = recorded(1501)
    _, previous = recorded(1500)
    assert previous[55] > 2
    entry = player_list_entry(current, previous)
    assert (entry.index, entry.phase, entry.direction) == (1, 0, 1)
    assert np.array_equal(player_list_support(current, previous),
                          np.all(rgb[:171] == (210, 164, 74), axis=2))
