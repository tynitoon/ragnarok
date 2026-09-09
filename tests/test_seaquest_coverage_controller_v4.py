import pytest

from scripts.seaquest_coverage_controller_v4 import CoverageController
from scripts.capture_seaquest_coverage_v4 import capture
from tests.test_seaquest_coverage_controller_v2 import frame
from tests.test_seaquest_coverage_controller_v3 import surface_frame


def test_nearby_enemy_overrides_diver_chase_but_distant_one_does_not():
    assert CoverageController().action(frame(enemy_x=115, diver_x=20)) == 1
    assert CoverageController().action(frame(enemy_x=145, diver_x=20)) == 12
    assert CoverageController().action(frame(enemy_x=None, diver_x=130)) == 11


def test_new_oxygen_threshold_and_surface_fix():
    rgb=frame(enemy_x=115)
    rgb[172,49:112]=0
    rgb[172,49:58]=(214,214,214)
    assert CoverageController().action(rgb)==1
    rgb[172,57]=0
    assert CoverageController().action(rgb)==10
    controller=CoverageController()
    controller.phase="ascend"
    assert controller.action(surface_frame(47,8))==10
    assert controller.action(surface_frame(46,8))==1


def test_reserved_v4_stays_disabled(tmp_path):
    with pytest.raises(ValueError,match="Reserved coverage disabled"):
        capture(tmp_path/"reserved",partition="reserved_validation")
