"""Synthetic model/metric invariants; no Atari archive is opened."""
import numpy as np
import pytest
import torch
import json
from torch.nn import functional as F

from ragnarok.learning.response_envelope_v70 import (ResponseEnvelope, anchor_balanced_bce,
    average_precision, measure, feasibility_verdict)


def test_constant_input_has_no_spatial_coordinate_code_and_sizes_share_weights():
    torch.set_num_threads(1)
    torch.manual_seed(764)
    model = ResponseEnvelope()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    for h, w in ((13, 17), (19, 23)):
        output = model(torch.zeros(2, 6, h, w))
        assert output.shape == (2, 1, h, w)
        torch.testing.assert_close(output, output[:, :, :1, :1].expand_as(output), atol=1e-7, rtol=0)
    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == shapes


def test_balanced_loss_keeps_empty_anchors_and_has_finite_gradient():
    logits = torch.tensor([[[[2.0, -1.0], [.2, 1.0]]], [[[1.0, -2.0], [.3, -.2]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    pixels = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    expected = (.5 * (pixels[0, 0, 0, 0] + pixels[0][target[0] == 0].mean()) + pixels[1].mean()) / 2
    loss = anchor_balanced_bce(logits, target)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[1].abs().sum() > 0


def test_average_precision_groups_ties_without_using_label_order():
    labels = np.array([True, False, True, False])
    assert average_precision(np.ones(4), labels) == .5
    assert average_precision(np.ones(4), labels[::-1]) == .5
    assert average_precision(np.array([.8, .2, .8, .1]), labels) == 1
    assert average_precision(np.zeros(4), np.zeros(4, bool)) is None


def test_empty_targets_are_not_counted_as_perfect_iou():
    targets = np.array([[[True, False]], [[False, False]]])
    cases = [dict(game='a', anchor_index=0), dict(game='b', anchor_index=0)]
    scores = np.array([[[.9, .1]], [[.8, .2]]])
    measured = measure(scores, targets, cases)
    assert measured['mean_nonempty_iou'] == 1
    assert measured['cases'][1]['iou'] is None
    assert measured['cases'][1]['average_precision'] is None
    assert measured['empty_anchor_false_positive_fractions'] == [.5]


def test_one_bad_empty_anchor_or_incomplete_fit_cannot_pass():
    def entry(iou, empty):
        return dict(mean_nonempty_iou=iou, empty_anchor_false_positive_fractions=empty)
    values = dict(visual=entry(.5, [0.0, .002]), constant=entry(.2, []),
                  initial=entry(.1, []), zero=entry(0.0, []))
    done = {'visual': 256, 'constant': 256}
    assert not feasibility_verdict(values, done)['passed']
    values['visual']['empty_anchor_false_positive_fractions'] = [0.0, 0.0]
    assert feasibility_verdict(values, done)['passed']
    assert not feasibility_verdict(values, {'visual': 256, 'constant': 255})['passed']


def test_full_native_batch_has_finite_backward_for_resource_sizing():
    # One synthetic forward/backward, not fitting real or synthetic tasks.
    torch.set_num_threads(1)
    torch.manual_seed(764)
    model = ResponseEnvelope()
    inputs = torch.zeros(12, 6, 210, 160)
    targets = torch.zeros(12, 1, 210, 160)
    targets[:10, :, 100:104, 70:74] = 1
    loss = anchor_balanced_bce(model(inputs), targets)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_probability_does_not_enter_a_score(value):
    with pytest.raises(ValueError):
        measure(np.array([[[value]]]), np.ones((1, 1, 1), bool), [dict(game='a', anchor_index=0)])


def runner_fixture(monkeypatch, tmp_path):
    from scripts import fit_response_envelope_v70 as fit
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'FOLDER', tmp_path / 'fit')
    fit.FOLDER.mkdir()
    monkeypatch.setattr(fit, 'PREFLIGHT', fit.FOLDER / 'preflight_v0.json')
    monkeypatch.setattr(fit, 'TARGET_RESULT', tmp_path / 'targets.json')
    monkeypatch.setattr(fit, 'SPEC', dict(fit.SPEC, updates_per_arm=2))
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'synthetic': True})
    monkeypatch.setattr(fit, 'proofs', lambda: proof)
    monkeypatch.setattr(fit, 'committed', lambda path: None)
    fit.write_json(fit.PREFLIGHT, dict(kind='response_envelope_fit_independent_preflight', passed=True, **proof))
    cases = [dict(game=f'fixture_{i // 2}', anchor_index=i % 2) for i in range(12)]
    fit.write_json(fit.TARGET_RESULT, dict(cases=cases, inherited_collection_charged_frames=123))
    inputs = torch.zeros(12, 6, 3, 4)
    labels = torch.zeros(12, 1, 3, 4)
    labels[:10, :, 1, 1] = 1
    monkeypatch.setattr(fit, 'load_dataset', lambda result: (inputs, labels, labels[:, 0].numpy().astype(bool)))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Satisfy the population guard without running the full FCN in scheduling tests.
            self.values = torch.nn.Parameter(torch.zeros(10177))
        def forward(self, x):
            return x[:, :1] * self.values[0] + self.values[1] + self.values[2:].sum() * 0
    monkeypatch.setattr(fit, 'ResponseEnvelope', Tiny)
    return fit


def test_runner_keeps_paired_schedule_and_all_final_controls(monkeypatch, tmp_path):
    fit = runner_fixture(monkeypatch, tmp_path)
    assert fit.run() == 0
    result = json.loads((fit.FOLDER / 'fit_v0.json').read_text())
    assert result['execution_status'] == 'complete'
    assert result['optimizer_updates'] == 4 and result['completed_updates'] == {'visual': 2, 'constant': 2}
    assert set(result['metrics']) == {'visual', 'constant', 'initial', 'zero'}
    events = [json.loads(line) for line in (fit.FOLDER / 'fit_v0_progress.jsonl').read_text().splitlines()]
    assert [(e['arm'], e['update']) for e in events if e['event'] == 'update_complete'] == [
        ('visual', 1), ('constant', 1), ('visual', 2), ('constant', 2)]
    before = (fit.FOLDER / 'fit_v0.json').read_bytes()
    with pytest.raises(FileExistsError):
        fit.run()
    assert (fit.FOLDER / 'fit_v0.json').read_bytes() == before


def test_runner_timeout_counts_completed_step_and_preserves_partial_weights(monkeypatch, tmp_path):
    fit = runner_fixture(monkeypatch, tmp_path)
    original = fit.update
    state = {'stepped': False}
    def update(*args):
        result = original(*args)
        state['stepped'] = True
        return result
    def check(deadline):
        if state['stepped']:
            raise TimeoutError('synthetic work deadline')
    monkeypatch.setattr(fit, 'update', update)
    monkeypatch.setattr(fit, 'check_time', check)
    assert fit.run() == 1
    result = json.loads((fit.FOLDER / 'fit_v0.json').read_text())
    assert result['optimizer_updates'] == 1 and result['completed_updates'] == {'visual': 1, 'constant': 0}
    assert result['verdict']['complete'] is False
    assert (fit.FOLDER / 'final_weights.pt').exists()
    assert 'final_predictions.npz' not in result['artifacts']


def test_preflight_rejection_precedes_dataset_and_model(monkeypatch, tmp_path):
    fit = runner_fixture(monkeypatch, tmp_path)
    fit.PREFLIGHT.write_text(json.dumps(dict(passed=False)))
    def forbidden(*args):
        pytest.fail('No dataset or model is allowed before valid preflight')
    monkeypatch.setattr(fit, 'load_dataset', forbidden)
    monkeypatch.setattr(fit, 'ResponseEnvelope', forbidden)
    with pytest.raises(ValueError, match='prospective'):
        fit.run()
    assert not (fit.FOLDER / 'fit_v0_attempt.json').exists()


def test_nonfinite_loss_cannot_call_optimizer_step():
    from scripts import fit_response_envelope_v70 as fit
    model = torch.nn.Conv2d(1, 1, 1)
    class NoStep:
        def zero_grad(self, **kwargs):
            model.zero_grad(**kwargs)
        def step(self):
            pytest.fail('Nonfinite loss must stop before optimizer.step')
    with pytest.raises(FloatingPointError):
        fit.update(model, NoStep(), torch.full((1, 1, 2, 2), float('nan')), torch.zeros(1, 1, 2, 2))
