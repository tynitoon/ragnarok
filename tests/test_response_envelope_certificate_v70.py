"""Synthetic external-judge checks: three forward calls in the complete suite.

One forward contains six sparse/empty stationary scenes; two more cover 1x1
and 2x3 images.  Factory/architecture checks perform no forward.  There are no
optimizer steps, emulator calls, archive reads or learner initializations.
"""

import math

import pytest
import torch
from torch import nn

from ragnarok.evaluation.response_envelope_certificate_v70 import make_reference_model
from ragnarok.learning.response_envelope_v70 import (
    ResponseEnvelope,
    anchor_balanced_bce,
    measure,
)

torch.set_num_threads(1)


def _one_forward(inputs):
    model = make_reference_model()
    calls = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with torch.inference_mode():
            result = model(inputs)
    finally:
        hook.remove()
    assert calls == [1]
    assert not result.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    return result


def test_factory_never_forwards_and_preserves_cpu_rng(monkeypatch):
    def forbidden_forward(*_args, **_kwargs):
        raise AssertionError("The certificate factory must not evaluate a scene")

    monkeypatch.setattr(ResponseEnvelope, "forward", forbidden_forward)
    before = torch.get_rng_state().clone()
    model = make_reference_model()
    assert type(model) is ResponseEnvelope
    assert torch.equal(torch.get_rng_state(), before)


def test_exact_architecture_parameter_count_frozen_state_and_sparse_coefficients():
    model = make_reference_model()
    assert type(model) is ResponseEnvelope
    assert sum(parameter.numel() for parameter in model.parameters()) == 10177
    assert len(model.body) == 10
    assert all(not module.training for module in model.modules())
    for parameter in model.parameters():
        assert parameter.device.type == "cpu"
        assert parameter.dtype == torch.float32
        assert not parameter.requires_grad
        assert parameter.grad is None
        assert torch.isfinite(parameter).all()

    for number, dilation in enumerate((1, 2, 4, 8, 16)):
        layer = model.body[2 * number]
        assert type(layer) is nn.Conv2d
        assert type(model.body[2 * number + 1]) is nn.SiLU
        assert layer.in_channels == (6 if number == 0 else 16)
        assert layer.out_channels == 16
        assert layer.kernel_size == (3, 3)
        assert layer.dilation == layer.padding == (dilation, dilation)
        assert layer.padding_mode == "replicate"
        assert layer.stride == (1, 1) and layer.groups == 1
        outer = layer.weight.detach().clone()
        outer[:, :, 1, 1] = 0
        assert torch.count_nonzero(outer) == 0
        assert torch.count_nonzero(layer.weight[2:]) == 0

    first = model.body[0]
    assert first.weight[0, [0, 3], 1, 1].tolist() == [4, 4]
    assert first.weight[1, [0, 3], 1, 1].tolist() == [-4, -4]
    assert first.bias[:2].tolist() == [-4, 4]
    for layer in model.body[2::2]:
        assert layer.weight[:2, :2, 1, 1].tolist() == [[1, -1], [-1, 1]]
        assert torch.count_nonzero(layer.bias) == 0
    assert type(model.head) is nn.Conv2d
    assert model.head.in_channels == 16 and model.head.out_channels == 1
    assert model.head.kernel_size == (1, 1)
    assert model.head.weight[0, :2, 0, 0].tolist() == [2, -2]
    assert torch.count_nonzero(model.head.bias) == 0
    # This total, together with the explicit entries, excludes stray weights.
    assert sum(int(torch.count_nonzero(p)) for p in model.parameters()) == 24


def test_factory_float32_is_independent_of_the_default_dtype():
    previous_dtype = torch.get_default_dtype()
    before = torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float64)
        model = make_reference_model()
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert torch.get_default_dtype() == torch.float64
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_rare_and_empty_stationary_scenes_have_signed_eight_logits_and_exact_masks():
    anchors, height, width = 6, 21, 23
    support = torch.zeros((anchors, height, width), dtype=torch.bool)
    support[1, 9:11, 10:12] = True
    support[2, [0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]] = True
    support[3, [0, height - 1, height // 2, height // 2],
            [width // 2, width // 2, 0, width - 1]] = True
    support[4, [1, 7, 19], [21, 3, 17]] = True
    assert support.sum(dim=(1, 2)).tolist() == [0, 4, 4, 4, 3, 0]
    assert int(support.sum(dim=(1, 2)).max()) / (height * width) < .01

    generator = torch.Generator(device="cpu").manual_seed(110)
    rgb = torch.rand((anchors, 3, height, width), generator=generator)
    rgb[:, 0] = support.float()
    rgb[:, 2][support] = 0
    inputs = torch.cat((rgb, rgb), dim=1)
    before = inputs.clone()
    logits = _one_forward(inputs)
    expected = torch.where(support[:, None], 8., -8.)
    torch.testing.assert_close(logits, expected, rtol=0, atol=2e-6)
    assert torch.equal(inputs, before)
    assert torch.equal(logits[:, 0] >= 0, support)

    cases = [dict(game=f"synthetic_{index // 2}", anchor_index=index % 2)
             for index in range(anchors)]
    stats = measure(logits.sigmoid()[:, 0].numpy(), support.numpy(), cases)
    assert stats["mean_nonempty_iou"] == 1
    assert stats["mean_nonempty_average_precision"] == 1
    assert stats["empty_anchor_false_positive_fractions"] == [0, 0]
    for index, row in enumerate(stats["cases"]):
        assert row["false_positive_pixels"] == 0
        assert row["iou"] == (None if index in (0, 5) else 1)
    bce = anchor_balanced_bce(logits, support[:, None].float())
    assert float(bce) == pytest.approx(math.log1p(math.exp(-8)), abs=2e-7)


@pytest.mark.parametrize("height,width", [(1, 1), (2, 3)])
def test_replicated_padding_even_beyond_the_whole_image_keeps_the_certificate(height, width):
    # These tiny all-negative/all-positive scenes test padding, not rarity.
    # Every location touches a boundary although the last padding radius is 16.
    inputs = torch.linspace(0, 1, 2 * 6 * height * width).reshape(2, 6, height, width)
    inputs[0, [0, 3]] = 0
    inputs[1, [0, 3]] = 1
    before = inputs.clone()
    logits = _one_forward(inputs)
    torch.testing.assert_close(logits[0], torch.full_like(logits[0], -8), rtol=0, atol=2e-6)
    torch.testing.assert_close(logits[1], torch.full_like(logits[1], 8), rtol=0, atol=2e-6)
    assert torch.equal(inputs, before)
