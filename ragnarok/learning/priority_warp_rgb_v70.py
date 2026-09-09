"""Separate priority-weighted forward warp; no network, labels or training.

Operator inspired by Niklaus & Liu, Softmax Splatting, CVPR2020, Eq.13:
https://arxiv.org/abs/2003.05534. This is an independent PyTorch implementation.
The caller must obtain priorities from its own available observations. The
interpolation paper's future-dependent importance estimate is NOT used here.
"""
import torch

from ragnarok.learning.pixel_prediction_v70 import _validate, _coordinates


def priority_warp_rgb(image, displacement, log_priority):
    """Resolve collisions using exp(source log-priority), not equal averaging.

RGB, B2HW displacement and B1HW priorities are finite float32 tensors on one
device. Priorities are explicitly bounded to[-20,20], without silent clipping.
Zero priorities recover the old uniform renderer. Holes remain black.

The denominator floor is relative to the greatest contributing source priority
at each destination. This retains the old1e-6 coverage floor at zero priorities
and avoids inventing attenuation merely from a common log-priority offset.
Returned coverage is geometry, not an assertion of visibility or foreground.
"""
    _validate(image, displacement)
    if (image.dtype != torch.float32 or displacement.dtype != torch.float32
            or not isinstance(log_priority, torch.Tensor)
            or log_priority.dtype != torch.float32
            or log_priority.shape != (image.shape[0], 1, *image.shape[-2:])
            or log_priority.device != image.device
            or not torch.isfinite(log_priority).all() or (log_priority.abs() > 20).any()):
        raise ValueError('Float32 own RGB/flow and finite B1HW log-priorities in[-20,20] required')
    batch, channels, height, width = image.shape
    x, y = _coordinates(image, displacement)
    x, y = x.clamp(-2, width + 1), y.clamp(-2, height + 1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    sums = image.new_zeros((batch, channels, height * width))
    coverage = image.new_zeros((batch, 1, height * width))
    weighted_coverage = torch.zeros_like(coverage)
    maximum_priority = torch.zeros_like(coverage)
    colors = image.reshape(batch, channels, -1)
    priority = log_priority.exp().reshape(batch, 1, -1)
    for ox, oy, geometric in (
            (0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy), (1, 1, dx * dy)):
        xx, yy = x0 + ox, y0 + oy
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        indices = (yy.clamp(0, height - 1).long() * width
                   + xx.clamp(0, width - 1).long()).reshape(batch, 1, -1)
        weights = (geometric * valid).reshape(batch, 1, -1)
        weighted = weights * priority
        sums = sums.scatter_add(2, indices.expand(-1, channels, -1), colors * weighted)
        coverage = coverage.scatter_add(2, indices, weights)
        weighted_coverage = weighted_coverage.scatter_add(2, indices, weighted)
        contributing = torch.where(weights > 0, priority, torch.zeros_like(priority))
        maximum_priority = maximum_priority.scatter_reduce(
            2, indices, contributing, reduce='amax', include_self=True)
    floor = 1e-6 * maximum_priority
    denominator = torch.maximum(weighted_coverage, floor)
    # Empty destinations divide by1, avoiding an inactive infinite derivative.
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    reconstructed = (sums / denominator).reshape_as(image)
    if (not torch.isfinite(reconstructed).all() or not torch.isfinite(weighted_coverage).all()
            or (reconstructed < 0).any() or (reconstructed > 1).any()):
        raise ValueError('Priority reconstruction must remain finite RGB in[0,1]')
    shape = (batch, 1, height, width)
    return dict(
        reconstructed=reconstructed, coverage=coverage.reshape(shape),
        weighted_coverage=weighted_coverage.reshape(shape),
        normalization_floor_active=((weighted_coverage > 0) & (weighted_coverage < floor)).reshape(shape),
    )
