import numpy as np
import pytest

from scripts.seaquest_coverage_controller_v3 import CoverageController
from scripts.capture_seaquest_coverage_v3 import capture


def surface_frame(top, oxygen):
    rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    rgb[top:top+6, 76:88] = (187, 187, 53)
    rgb[172, 49:49+oxygen] = (214, 214, 214)
    return rgb


def test_cropped_centre_does_not_stop_ascent_one_row_below_refill():
    controller = CoverageController()
    controller.phase = "ascend"
    assert controller.action(surface_frame(47, 12)) == 10
    assert controller.phase == "ascend"
    assert controller.action(surface_frame(46, 12)) == 1
    assert controller.phase == "refill"


def test_initial_surface_waits_and_reserved_capture_stays_disabled(tmp_path):
    assert CoverageController().action(surface_frame(46, 1)) == 1
    destination = tmp_path/"reserved"
    with pytest.raises(ValueError, match="Reserved coverage disabled"):
        capture(destination, partition="reserved_validation")
    assert not destination.exists()
