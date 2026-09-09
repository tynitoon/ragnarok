import numpy as np
import pytest

from ragnarok.evaluation.mspacman_render_v70 import player_render_box, power_pill_render_boxes


def test_player_uses_preceding_position_and_animation_together():
    prior = np.zeros(128, dtype=np.uint8)
    prior[10], prior[16], prior[55], prior[56] = 50, 30, 0, 3
    current = prior.copy()
    current[10], current[16], current[55], current[56] = 52, 32, 2, 1
    assert player_render_box(current, prior)[0].box == (40,31,6,10)
    assert player_render_box(current)[0].box == (40,33,8,10)


def test_phase_fields_are_independent_and_power_pill_consumption_is_current():
    current = np.zeros(128, dtype=np.uint8)
    current[117] = 4 | 8 | 16 | 32
    assert {o.slot for o in power_pill_render_boxes(current)} == {"4", "8"}
    current[30] = 8
    current[117] &= 255-16
    assert [(o.slot,o.box) for o in power_pill_render_boxes(current)] == [("32",(8,15,4,7))]


def test_player_rejects_unknown_phase_and_clips_visible_envelope():
    current = np.zeros(128, dtype=np.uint8)
    current[10], current[16], current[55] = 9, 168, 2
    assert player_render_box(current)[0].box == (0,169,5,2)
    current[55] = 3
    with pytest.raises(ValueError, match="Uncalibrated"):
        player_render_box(current)
