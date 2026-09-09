"""Source-level contracts; no native data or optimizer updates."""
import pytest
import torch

from ragnarok.learning.archived_pixel_diagnostics_v70 import weight_sha256
from ragnarok.learning.blurred_prediction_loss_v70 import blurred_prediction_loss
from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import CounterfactualParser, max_pool_rgb
from ragnarok.learning.priority_motion_parser_v70 import PriorityMotionParser, PriorityParserOutput
from ragnarok.learning.priority_warp_rgb_v70 import priority_warp_rgb
from scripts.check_synthetic_motion_fit_v70 import scenes

torch.set_num_threads(1)


def test_new_contract_preserves_virgin_visual_weights_and_initial_outputs():
    previous, current, _ = scenes()
    torch.manual_seed(764)
    old = OrderedMotionParser()
    torch.manual_seed(764)
    candidate = PriorityMotionParser()
    assert not isinstance(candidate, CounterfactualParser)
    assert weight_sha256(candidate.visual) == weight_sha256(old)
    before = weight_sha256(candidate)
    with torch.no_grad():
        output = candidate(previous, current)
        prior = old(previous, current)
    assert isinstance(output, PriorityParserOutput)
    assert output.log_priority.shape == (2, 1, 16, 16)
    assert torch.count_nonzero(output.log_priority) == 0
    for name in ('embedding', 'foreground_logits', 'displacement',
                 'controllability_logits', 'solidity_logits'):
        assert torch.equal(getattr(output.visual, name), getattr(prior, name))
    assert weight_sha256(candidate) == before


def test_rgb_loss_reaches_new_priority_layer_and_existing_motion_without_updates():
    previous, current, future = scenes()
    torch.manual_seed(764)
    model = PriorityMotionParser()
    before = weight_sha256(model)
    output = model(previous, current)
    prediction = priority_warp_rgb(max_pool_rgb(previous, current).float() / 255,
                                   output.visual.displacement, output.log_priority)
    loss = blurred_prediction_loss(prediction['reconstructed'], future.float() / 255)['loss']
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.count_nonzero(model.priority[-1].weight.grad) > 0
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.visual.ordered_motion.parameters())
    # Zero initialization of the final priority layer delays trunk gradients;
    # this test does not claim the whole new branch already receives a signal.
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.priority[0].parameters())
    assert weight_sha256(model) == before


def test_large_priority_logits_are_bounded_and_invalid_visual_outputs_refused():
    previous, current, _ = scenes()
    model = PriorityMotionParser()
    with torch.no_grad():
        model.priority[-1].bias.fill_(1e12)  # Declared numerical fixture, not training.
        output = model(previous, current)
    assert torch.equal(output.log_priority, torch.full_like(output.log_priority, 20))
    with torch.no_grad():
        model.visual.foreground.bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='visual heads'):
        model(previous, current)


def test_future_is_not_a_forward_argument_and_invalid_observations_are_rejected():
    previous, current, future = scenes()
    model = PriorityMotionParser()
    with pytest.raises(TypeError):
        model(previous, current, future=future)
    with pytest.raises(ValueError, match='uint8'):
        model(previous.float(), current.float())
    with pytest.raises(ValueError, match='effect'):
        model(previous, current, torch.full((2, 12), 1.1))
