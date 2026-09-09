import numpy as np
import pytest

from scripts.seaquest_coverage_controller_v2 import CoverageController
from scripts.capture_seaquest_coverage_v2 import capture


def frame(enemy_x=None, diver_x=20, facing=1):
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    x, y = 76, 69
    tip = 10 if facing == 1 else 4
    rgb[y:y+5, x+tip:x+tip+2] = (187, 187, 53)
    rgb[y+5:y+11, x:x+16] = (187, 187, 53)
    rgb[172, 49:112] = (214, 214, 214)
    rgb[70:79, diver_x:diver_x+8] = (66, 72, 200)
    if enemy_x is not None:
        rgb[69:76, enemy_x:enemy_x+8] = (92, 186, 92)
    return rgb


def test_waits_for_diver_without_turning_away_from_incoming_enemy():
    # Already faces the right enemy; a diver on the left must not make the
    # collection controller turn left and expose its back as v1 did.
    rgb = frame(enemy_x=130)
    before = rgb.copy()
    assert CoverageController().action(rgb) == 1
    assert np.array_equal(rgb, before)
    assert CoverageController().action(frame(enemy_x=20, diver_x=130)) == 12
    assert CoverageController().action(frame(enemy_x=20, diver_x=130, facing=-1)) == 1


def test_contacting_projectile_does_not_reverse_facing_or_push_player():
    for start in (89, 91):
        rgb = frame(enemy_x=130)
        rgb[76, start:start+8] = (187, 187, 53)
        assert CoverageController().action(rgb) == 1


def test_reserved_v2_capture_is_disabled_before_any_emulator(tmp_path):
    destination = tmp_path/"reserved"
    with pytest.raises(ValueError, match="Reserved coverage disabled"):
        capture(destination, partition="reserved_validation")
    assert not destination.exists()
