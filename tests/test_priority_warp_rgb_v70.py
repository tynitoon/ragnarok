"""Constructive renderer/gradient checks only; no learned priorities or updates."""
import pytest
import torch

from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb
from ragnarok.learning.priority_warp_rgb_v70 import priority_warp_rgb
from tests.test_prediction_occlusion_limits_v70 import opaque_scene

torch.set_num_threads(1)


def test_zero_priorities_reproduce_old_rgb_coverage_and_holes():
    image = torch.linspace(0, 1, 3 * 9 * 11).reshape(1, 3, 9, 11)
    field = torch.zeros((1, 2, 9, 11))
    field[:, 0] = .375
    field[:, 1] = -.25
    field[:, :, 2:4, 2:4] = 50
    old_rgb, old_coverage = forward_warp_rgb(image, field)
    result = priority_warp_rgb(image, field, torch.zeros((1, 1, 9, 11)))
    assert torch.equal(result['reconstructed'], old_rgb)
    assert torch.equal(result['coverage'], old_coverage)
    assert torch.equal(result['weighted_coverage'], old_coverage)


def test_declared_foreground_priority_can_resolve_opaque_collision_with_static_background():
    for dx in (-4, 4):
        source, future, physical_field = opaque_scene(dx)
        # Judge-supplied priorities demonstrate representational capacity only.
        # No model receives these labels, and no priority is learned in this test.
        priority = torch.zeros((1, 1, 16, 16))
        priority[source[:, :1] == 1] = 8
        result = priority_warp_rgb(source, physical_field, priority)
        error = float((result['reconstructed'] - future).abs().mean())
        assert error < 1e-5
        assert torch.count_nonzero(physical_field[:, :, source[0, 0] == 0]) == 0
        assert torch.equal(result['coverage'][future[:, :1] == 1], torch.full((6,), 2.))
        # Inverting the priority must actually invert who wins the collision.
        reversed_rgb = priority_warp_rgb(source, physical_field, -priority)['reconstructed']
        assert (reversed_rgb[future == 1] < .001).all()


def test_common_priority_offsets_do_not_attenuate_image_or_change_geometry():
    source, _, field = opaque_scene(4)
    priorities = torch.zeros((1, 1, 16, 16))
    priorities[:, :, 6:8, 4:7] = 4
    normal = priority_warp_rgb(source, field, priorities)
    for offset in (-15, 12):
        shifted = priority_warp_rgb(source, field, priorities + offset)
        assert torch.allclose(normal['reconstructed'], shifted['reconstructed'], rtol=0, atol=2e-7)
        assert torch.equal(normal['coverage'], shifted['coverage'])
    # A near-empty projection also needs an offset-invariant relative floor.
    tiny_field = torch.zeros((1, 2, 1, 1))
    tiny_field[:, 0] = 1 - 2 ** -24
    for offset in (-20, 0, 20):
        rendered = priority_warp_rgb(torch.ones((1, 3, 1, 1)), tiny_field,
                                     torch.full((1, 1, 1, 1), float(offset)))
        assert rendered['normalization_floor_active'].all()
        assert float(rendered['reconstructed'][0, 0, 0, 0]) == pytest.approx(2 ** -24 / 1e-6, rel=2e-7)


def test_both_fields_and_priorities_have_finite_gradients_without_parameter_updates():
    source, future, field = opaque_scene(4)
    field = (field + .125).requires_grad_()
    priority = torch.zeros((1, 1, 16, 16), requires_grad=True)
    before_field, before_priority = field.detach().clone(), priority.detach().clone()
    reconstruction = priority_warp_rgb(source, field, priority)['reconstructed']
    (reconstruction - future).square().mean().backward()
    assert torch.isfinite(field.grad).all() and torch.count_nonzero(field.grad) > 0
    assert torch.isfinite(priority.grad).all() and torch.count_nonzero(priority.grad) > 0
    assert torch.equal(field, before_field) and torch.equal(priority, before_priority)


def test_all_outside_and_extreme_admitted_priorities_remain_finite_in_backward():
    source = torch.ones((1, 3, 2, 2))
    for outside in (False, True):
        field = torch.full((1, 2, 2, 2), 50. if outside else .125, requires_grad=True)
        priority = torch.tensor([[[[-20., 20.], [20., -20.]]]], requires_grad=True)
        result = priority_warp_rgb(source, field, priority)
        result['reconstructed'].sum().backward()
        assert torch.isfinite(field.grad).all() and torch.isfinite(priority.grad).all()
        if outside:
            assert torch.count_nonzero(result['reconstructed']) == 0
            assert torch.count_nonzero(result['coverage']) == 0


def test_invalid_priority_contracts_are_refused_without_clipping():
    source, _, field = opaque_scene(4)
    for priority in (torch.zeros((1, 2, 16, 16)), torch.full((1, 1, 16, 16), 20.1),
                     torch.full((1, 1, 16, 16), float('nan')), torch.zeros((1, 1, 16, 16)).double()):
        with pytest.raises(ValueError, match='log-priorities'):
            priority_warp_rgb(source, field, priority)
