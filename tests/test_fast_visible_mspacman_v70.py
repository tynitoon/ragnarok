"""Ordered equivalence, real renderer edge cases and oracle input immutability."""
import numpy as np
import pytest
from tests.test_mspacman_pooled_rows_v70 import row
from ragnarok.evaluation import fast_visible_mspacman_v70 as fast
from ragnarok.evaluation import pooled_mspacman_eyes_v70 as frozen
from ragnarok.evaluation.mspacman_v70 import decode_mspacman_ram, SlottedLabel
from ragnarok.evaluation.visible_scene_v70 import mspacman_visible_scene


@pytest.mark.parametrize('layout', [0, 1, 255])
def test_pill_mapping_order_every_single_bit_and_full_board(layout):
    ram = np.zeros(128, dtype=np.uint8)
    ram[0] = layout
    for offset in range(59, 101):
        for bit in range(8):
            ram[offset] = 1 << bit
            assert fast._nominal_pills(ram) == [o for o in decode_mspacman_ram(ram) if o.kind == 'pill']
        ram[offset] = 0
    ram[59:101] = 255
    assert fast._nominal_pills(ram) == [o for o in decode_mspacman_ram(ram) if o.kind == 'pill']


def test_occlusion_preserves_visible_envelopes_and_order():
    pill = SlottedLabel('pill', (8, 7, 4, 2), '0:0')
    power = SlottedLabel('power_pill', (148, 14, 4, 7), '8')
    mask = np.zeros((171, 160), dtype=bool)
    assert fast._unoccluded_pills([pill, power], mask) == [pill, power]
    mask[7:9, 8:10] = True
    mask[14:21, 148:152] = True
    assert fast._unoccluded_pills([pill, power], mask) == [SlottedLabel('pill', (10, 7, 2, 2), '0:0')]
    mask[7:9, 10:12] = True
    mask[17, 150] = False
    assert fast._unoccluded_pills([pill, power], mask) == [SlottedLabel('power_pill', (150, 17, 1, 1), '8')]


@pytest.mark.parametrize('index', [1, 246, 453, 1250, 1957, 1958, 1963, 2025, 5245, 7081, 7086, 8192])
def test_real_history_ghost_rows_eyes_occlusion_and_scene_identity(index):
    r = row(index)
    for now, before in [('ram_previous', 'ram_before_previous'), ('ram_current', 'ram_previous')]:
        current, previous = r[now], r[before]
        current_copy, previous_copy = current.copy(), previous.copy()
        expected = frozen.source_ram_objects(current, previous)
        assert fast.source_ram_objects(current, previous) == expected
        assert fast.mspacman_visible_scene(current, previous) == mspacman_visible_scene(current, previous)
        assert np.array_equal(current, current_copy) and np.array_equal(previous, previous_copy)


@pytest.mark.parametrize('previous', [None, np.zeros(128, dtype=np.int64), np.zeros(127, dtype=np.uint8)])
def test_actual_previous_required(previous):
    with pytest.raises(ValueError, match='actual previous'):
        fast.source_ram_objects(np.zeros(128, dtype=np.uint8), previous)
