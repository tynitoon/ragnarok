"""Initialization and gradient contracts on tiny invented inputs, no training fit."""
import torch

from ragnarok.learning.direct_response_envelope_v70 import DirectResponseEnvelope
from ragnarok.learning.response_envelope_v70 import ResponseEnvelope, anchor_balanced_bce


def test_same_contextual_weights_rng_and_initial_function_with_only_seven_new_zeros():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(764)
        original = ResponseEnvelope()
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(764)
        candidate = DirectResponseEnvelope()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert sum(value.numel() for value in candidate.parameters()) == 10184
        for key, value in original.state_dict().items():
            assert torch.equal(value, candidate.contextual.state_dict()[key])
        assert not candidate.direct_weight.any() and not candidate.direct_bias.any()
        inputs = torch.arange(180, dtype=torch.float32).reshape(2, 6, 3, 5) / 255
        with torch.inference_mode():
            reference, combined = original(inputs), candidate(inputs)
            assert torch.equal(reference, combined)
            contextual, direct = candidate.evaluation_components
            assert torch.equal(contextual, reference) and not direct.any()
            assert torch.equal(contextual + direct, combined)


def test_direct_rgb_gradient_exists_when_zero_contextual_path_has_none():
    with torch.random.fork_rng(devices=[]):
        model = DirectResponseEnvelope()
        with torch.no_grad():
            for value in model.contextual.parameters(): value.zero_()
        inputs = torch.zeros(2, 6, 1, 1)
        inputs[0, (0, 3)] = 1
        targets = torch.tensor([1., 0.]).reshape(2, 1, 1, 1)
        loss = anchor_balanced_bce(model(inputs), targets)
        assert model.evaluation_components is None
        loss.backward()
        expected = torch.zeros(1, 6, 1, 1)
        expected[0, (0, 3)] = -.25
        assert torch.equal(model.direct_weight.grad, expected)
        assert model.direct_bias.grad.item() == 0
        assert all(value.grad is not None and not value.grad.any() for value in model.contextual.parameters())
