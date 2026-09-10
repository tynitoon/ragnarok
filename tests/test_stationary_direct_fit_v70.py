"""Direct-path integration and failure tests use tiny fixtures, no real dataset."""
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts import fit_stationary_direct_v70 as fit


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'FOLDER', tmp_path / 'fit')
    fit.FOLDER.mkdir()
    monkeypatch.setattr(fit, 'PREFLIGHT', fit.FOLDER / 'preflight.json')
    monkeypatch.setattr(fit, 'CONTROL_RESULT', tmp_path / 'control.json')
    monkeypatch.setattr(fit, 'INITIAL_MANIFEST', tmp_path / 'initial_identity.json')
    monkeypatch.setattr(fit, 'SPEC', dict(fit.SPEC, updates_per_arm=2, expected_training_forward_batches=4))
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'fixture': True})
    monkeypatch.setattr(fit, 'proofs', lambda: proof)
    monkeypatch.setattr(fit, 'committed', lambda _: None)
    fit.write_json(fit.PREFLIGHT, dict(kind='stationary_direct_independent_preflight', passed=True, **proof))
    cases = [dict(game=f'synthetic_pair_{i // 2}', anchor_index=i % 2) for i in range(12)]
    fit.write_json(fit.CONTROL_RESULT, dict(cases=cases))
    inputs, labels = torch.zeros(12, 6, 2, 3), torch.zeros(12, 1, 2, 3)
    inputs[:, 0, 0, 0] = 1
    labels[:, :, 0, 0] = 1
    inputs[6:8] = labels[6:8] = 0
    targets = labels[:, 0].numpy().astype(bool)
    monkeypatch.setattr(fit, 'load_dataset', lambda _: (inputs, labels, targets))
    class Contextual(nn.Module):
        def __init__(self):
            super().__init__()
            self.values = nn.Parameter(torch.zeros(10177))
        def forward(self, x):
            return x[:, :1] * self.values[0] + self.values[1] + self.values[2:].sum() * 0
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.contextual = Contextual()
            self.direct_weight = nn.Parameter(torch.zeros(1, 6, 1, 1))
            self.direct_bias = nn.Parameter(torch.zeros(1))
            self.evaluation_components = None
        def forward(self, x):
            contextual, direct = self.contextual(x), F.conv2d(x, self.direct_weight, self.direct_bias)
            self.evaluation_components = (contextual, direct) if not torch.is_grad_enabled() else None
            return contextual + direct
    monkeypatch.setattr(fit, 'ResponseEnvelope', Tiny)
    zeros = np.zeros(targets.shape, np.float32)
    probability = np.full(targets.shape, .5, np.float32)
    measured = fit.measure(probability, targets, cases)
    measured['balanced_bce'] = float(fit.anchor_balanced_bce(torch.zeros_like(labels), labels))
    identity = dict(state_arrays={'values': fit.signature(np.zeros(10177, np.float32))},
                     initial_logits=fit.signature(zeros), initial_predictions=fit.signature(probability),
                     initial_metrics=measured)
    fit.write_json(fit.INITIAL_MANIFEST, identity)


def result():
    return json.loads((fit.FOLDER / 'fit_v0.json').read_text())


def test_components_are_saved_from_only_three_evaluation_calls_with_exact_initial_identity(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    assert fit.run() == 0
    saved = result()
    assert saved['execution_status'] == 'complete' and saved['initial_identity_verified']
    assert saved['optimizer_updates'] == 4
    assert saved['model_forward_entries'] == saved['model_forward_returns'] == {
        'training': {'visual': 2, 'constant': 2}, 'evaluation': {'visual': 2, 'constant': 1}}
    assert set(saved['component_metrics']) == {'initial', 'visual', 'constant'}
    for arm in ('initial', 'visual', 'constant'):
        with np.load(fit.FOLDER / (arm + '_logits.npz')) as total, np.load(fit.FOLDER / (arm + '_components_logits.npz')) as parts:
            np.testing.assert_array_equal(parts['contextual_logits'] + parts['direct_logits'], total['logits'])
        assert set(saved['component_metrics'][arm]) == {'contextual', 'direct'}
    prior = (fit.FOLDER / 'fit_v0.json').read_bytes()
    with pytest.raises(FileExistsError): fit.run()
    assert (fit.FOLDER / 'fit_v0.json').read_bytes() == prior


def test_parameter_identity_mismatch_stops_before_any_forward_or_update(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    identity = json.loads(fit.INITIAL_MANIFEST.read_text())
    identity['state_arrays']['values']['sha256'] = 'wrong'
    fit.INITIAL_MANIFEST.write_text(json.dumps(identity))
    assert fit.run() == 1
    saved = result()
    assert not saved['initial_identity_verified'] and saved['optimizer_updates'] == 0
    assert saved['model_forward_returns']['evaluation'] == {'visual': 0, 'constant': 0}
    assert (fit.FOLDER / 'final_weights.pt').exists()


def test_initial_output_mismatch_preserves_unique_components_before_stopping(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    identity = json.loads(fit.INITIAL_MANIFEST.read_text())
    identity['initial_logits']['sha256'] = 'wrong'
    fit.INITIAL_MANIFEST.write_text(json.dumps(identity))
    assert fit.run() == 1
    saved = result()
    assert not saved['initial_identity_verified'] and saved['optimizer_updates'] == 0
    assert saved['model_forward_returns']['evaluation'] == {'visual': 1, 'constant': 0}
    for kind in ('logits', 'predictions', 'components_logits', 'components_predictions'):
        assert (fit.FOLDER / ('initial_' + kind + '.npz')).exists()


def test_metric_rejection_preserves_all_returned_contributions(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    def reject(*args): raise ValueError('Injected metric rejection')
    monkeypatch.setattr(fit, 'measure', reject)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 0 and saved['metrics'] == {}
    for kind in ('logits', 'predictions', 'components_logits', 'components_predictions'):
        assert (fit.FOLDER / ('initial_' + kind + '.npz')).exists()


def test_closed_journal_hash_failure_keeps_payload_and_weights_but_invalidates_success(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    original = fit.sha
    def broken(path):
        if path.name == 'fit_v0_progress.jsonl': raise OSError('Injected closed journal hash failure')
        return original(path)
    monkeypatch.setattr(fit, 'sha', broken)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 4 and not saved['passed']
    assert saved['progress_sha256'] is None and saved['scientific_payload_sha256'] is not None
    assert len(saved['component_metrics']) == 3 and (fit.FOLDER / 'final_weights.pt').exists()
