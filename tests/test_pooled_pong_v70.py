"""Metamorphic tests of pooled attribution, with no RAM-derived visual targets."""
import numpy as np

from ragnarok.evaluation.pong_v70 import PALETTE, screen_calibration_boxes
from ragnarok.evaluation.pooled_pong_v70 import pooled_ram_objects, pooled_screen_objects
from scripts.capture_pong_pooled_v70 import _PairRAMTap


def background():
    return np.full((210,160,3), (144,72,17), dtype=np.uint8)


def test_identical_frames_and_order_symmetry():
    a, b = background(), background()
    a[60:76,140:144] = PALETTE["player"]
    b[70:86,140:144] = PALETTE["player"]
    assert pooled_screen_objects(a,a) == screen_calibration_boxes(a)
    assert pooled_screen_objects(a,b) == pooled_screen_objects(b,a)
    assert pooled_screen_objects(a,b)[0].box == (140,60,4,26)


def test_hybrid_colours_preserve_disjoint_player_extent():
    a, b = background(), background()
    a[50:66,140:144] = PALETTE["player"]
    b[90:106,140:144] = PALETTE["player"]
    assert screen_calibration_boxes(np.maximum(a,b)) == []  # exact palette is the wrong instrument
    assert pooled_screen_objects(a,b)[0].box == (140,50,4,56)


def test_white_occluder_removes_all_or_part_of_the_other_type():
    a, b = background(), background()
    a[60:76,16:20] = PALETTE["enemy"]
    b[60:76,16:20] = PALETTE["ball"]
    assert {o.kind for o in pooled_screen_objects(a,b)} == {"ball"}
    b = background()
    b[64:68,18:20] = PALETTE["ball"]
    objects = {o.kind:o.box for o in pooled_screen_objects(a,b)}
    assert objects["enemy"] == (16,60,4,16)


def test_box_adapter_clips_and_uses_both_raw_times():
    a, b = np.zeros(128,dtype=np.uint8), np.zeros(128,dtype=np.uint8)
    a[51], b[51] = 43, 83
    objects = pooled_ram_objects(a,b)
    assert objects[0].box == (140,34,4,52)
    assert pooled_ram_objects(a,b) == pooled_ram_objects(b,a)


def test_ram_tap_brackets_exactly_one_act_and_copies_mutable_ram():
    class Native:
        def __init__(self):
            self.ram = np.zeros(128,dtype=np.uint8)
        def getRAM(self):
            return self.ram
        def act(self, action):
            self.ram[0] += action
            return 7
    native = Native()
    tap = _PairRAMTap(native)
    assert tap.act(3) == 7
    assert tap.previous[0] == 0 and tap.current[0] == 3
    native.ram[0] = 99
    assert tap.current[0] == 3


def test_temporal_reversal_preserves_order_when_different_types_appear():
    a, b = np.zeros(128,dtype=np.uint8), np.zeros(128,dtype=np.uint8)
    a[49], a[54], b[51] = 100, 100, 109
    assert pooled_ram_objects(a,b) == pooled_ram_objects(b,a)
    assert [obj.kind for obj in pooled_ram_objects(a,b)] == ["player","ball"]
