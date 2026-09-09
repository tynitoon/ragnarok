"""External constructive witnesses; no learned model or optimizer is involved.

The declared toy has an opaque white foreground moving over a stationary black
background. Its physical labels are used ONLY by this test judge, never training.
"""
import torch

from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb, motion_foreground_target

torch.set_num_threads(1)


def opaque_scene(dx):
    source = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    source[:, :, 6:8, 4:7] = 1
    future = torch.zeros_like(source)
    future[:, :, 6:8, 4 + dx:7 + dx] = 1
    foreground_flow = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    foreground_flow[:, 0, 6:8, 4:7] = dx
    return source, future, foreground_flow


def test_correct_opaque_foreground_motion_is_diluted_by_stationary_background():
    for dx in (-4, 4):
        source, future, field = opaque_scene(dx)
        reconstruction, coverage = forward_warp_rgb(source, field)
        destination = future[:, :1] == 1
        assert torch.equal(coverage[destination], torch.full((6,), 2.))
        assert torch.equal(reconstruction[future == 1], torch.full((18,), .5))
        # The physically declared opaque future is NOT a weighted average with
        # the background it covers. The frozen native fitting threshold is strict.
        assert float((reconstruction - future).abs().mean()) == .01171875
        assert not torch.equal(reconstruction, future)


def test_ejecting_background_repairs_rgb_but_marks_empty_source_pixels_as_foreground():
    for dx in (-4, 4):
        source, future, field = opaque_scene(dx)
        true_foreground = source[:, :1] == 1
        vacated_background = future[:, :1] == 1
        assert not (true_foreground & vacated_background).any()
        # Displace the black pixels at the incoming object's future position
        # downward into other black pixels. This motion is artificial in the toy.
        field[:, 1, 6:8, 4 + dx:7 + dx] = 4
        reconstruction, _ = forward_warp_rgb(source, field)
        assert torch.equal(reconstruction, future)
        target = motion_foreground_target(source, future, field)
        assert (target[true_foreground] > .99).all()
        assert (target[vacated_background] > .99).all()
        assert torch.count_nonzero(target) == 12
        assert torch.count_nonzero(true_foreground) == 6
