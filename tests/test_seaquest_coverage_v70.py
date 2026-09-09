"""Offline tests: no new game state is sampled while designing coverage."""
import numpy as np
import pytest

from scripts.capture_seaquest_coverage_v70 import CoverageController, capture


def picture(*, player_y=70, oxygen=63, collected=0, diver=True):
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[player_y:player_y+10, 70:86] = (187, 187, 53)
    rgb[172, 49:49+oxygen] = (214, 214, 214)
    for i in range(collected):
        rgb[180, 60+8*i] = (24, 26, 167)
    if diver:
        rgb[94:103, 100:108] = (66, 72, 200)
    return rgb


def test_controller_only_pixels_determinism_and_surface_cycle():
    sequence = [picture(), picture(oxygen=20), picture(player_y=46, oxygen=30),
                picture(player_y=46, oxygen=63), picture(collected=6)]
    a, b = CoverageController(), CoverageController()
    actions = []
    for rgb in sequence:
        before = rgb.copy()
        actions.append(a.action(rgb))
        assert actions[-1] == b.action(rgb)
        assert np.array_equal(before, rgb)
    assert actions == [16, 10, 1, 16, 10]
    assert CoverageController().action(picture(player_y=46, oxygen=1)) == 1
    assert a.action(np.zeros((210, 160, 3), dtype=np.uint8)) == 1
    with pytest.raises(ValueError):
        a.action(np.zeros((128,), dtype=np.uint8))


def test_reserved_capture_cannot_construct_an_engine_without_prior_seal(tmp_path):
    destination = tmp_path/"reserved"
    with pytest.raises(ValueError, match="Reserved coverage disabled"):
        capture(destination, partition="reserved_validation")
    assert not destination.exists()
