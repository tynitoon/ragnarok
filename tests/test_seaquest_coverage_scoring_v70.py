from ragnarok.evaluation.pong_v70 import ObjectLabel
from scripts.calibrate_seaquest_coverage_v70 import accumulate, finish, new_stats


def test_empty_frames_cannot_dilute_missed_positive_actors():
    s = new_stats()
    box = ObjectLabel("submarine", (40, 80, 8, 11))
    for _ in range(9900):
        accumulate(s, [], [])
    for _ in range(50):
        accumulate(s, [box], [box])
        accumulate(s, [box], [])
    finish(s)
    assert s["presence_fraction"] == .995
    assert s["box_fraction"] == 1
    assert s["positive_presence_fraction"] == .5
    assert not s["passed"]


def test_absent_surface_submarine_does_not_pass_vacuously():
    s = new_stats()
    accumulate(s, [], [])
    finish(s)
    assert s["presence_fraction"] == 1
    assert s["positive_presence_fraction"] is None and s["box_fraction"] is None
    assert not s["passed"]
