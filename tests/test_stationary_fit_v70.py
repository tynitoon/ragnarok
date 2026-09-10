"""Fault and isolation witnesses with fixtures only, never the real control or FCN."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from scripts import fit_stationary_control_v70 as fit


def runner_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'FOLDER', tmp_path / 'fit')
    fit.FOLDER.mkdir()
    monkeypatch.setattr(fit, 'PREFLIGHT', fit.FOLDER / 'preflight_v0.json')
    monkeypatch.setattr(fit, 'CONTROL_RESULT', tmp_path / 'control.json')
    monkeypatch.setattr(fit, 'SPEC', dict(fit.SPEC, updates_per_arm=2, expected_training_forward_batches=4))
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'fixture': True})
    monkeypatch.setattr(fit, 'proofs', lambda: proof)
    monkeypatch.setattr(fit, 'committed', lambda path: None)
    fit.write_json(fit.PREFLIGHT, dict(kind='stationary_fit_independent_preflight', passed=True, **proof))
    cases = [dict(game=f'synthetic_pair_{i // 2}', anchor_index=i % 2) for i in range(12)]
    fit.write_json(fit.CONTROL_RESULT, dict(cases=cases))
    inputs = torch.zeros(12, 6, 2, 3)
    labels = torch.zeros(12, 1, 2, 3)
    labels[:, :, 0, 0] = 1
    labels[6:8] = 0
    monkeypatch.setattr(fit, 'load_dataset', lambda result: (inputs, labels, labels[:, 0].numpy().astype(bool)))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.zeros(10177))
        def forward(self, x):
            return x[:, :1] * self.values[0] + self.values[1] + self.values[2:].sum() * 0
    monkeypatch.setattr(fit, 'ResponseEnvelope', Tiny)
    return proof


def result():
    return json.loads((fit.FOLDER / 'fit_v0.json').read_text())


def test_imports_never_load_the_certificate_or_builder():
    assert 'ragnarok.evaluation.response_envelope_certificate_v70' not in sys.modules
    assert 'scripts.build_stationary_control_v70' not in sys.modules


def test_loader_decodes_only_authorized_members_and_keeps_observed_labels(monkeypatch):
    past = np.zeros((12, 2, 210, 160, 3), np.uint8)
    targets = np.zeros((12, 210, 160), bool)
    for index, count in enumerate([1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2]):
        targets[index].reshape(-1)[:count] = True
    accessed = []
    class Archive:
        files = ['past_pairs', 'observed_response_envelopes', 'noop_futures', 'action_futures']
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, name):
            accessed.append(name)
            if name == 'past_pairs': return past
            if name == 'observed_response_envelopes': return targets
            pytest.fail('The learner must not decode a future branch')
    monkeypatch.setattr(fit.np, 'load', lambda *args, **kwargs: Archive())
    control = dict(artifacts={fit.CONTROL_DATA.name: dict(arrays={
        'past_pairs': fit.signature(past), 'observed_response_envelopes': fit.signature(targets)})},
        cases=[dict(game=f'synthetic_pair_{i // 2}', anchor_index=i % 2) for i in range(12)])
    inputs, labels, observed = fit.load_dataset(control)
    assert accessed == ['past_pairs', 'observed_response_envelopes']
    assert inputs.shape == (12, 6, 210, 160) and not inputs.any()
    # Positive labels survive zero RGB: the loader cannot replace observations with a color rule.
    np.testing.assert_array_equal(observed, targets)
    assert int(labels.sum()) == 847


def test_complete_schedule_counts_training_and_evaluation_forwards_and_refuses_retry(monkeypatch, tmp_path):
    runner_fixture(monkeypatch, tmp_path)
    assert fit.run() == 0
    saved = result()
    assert saved['execution_status'] == 'complete' and saved['optimizer_updates'] == 4
    assert saved['model_forward_entries'] == saved['model_forward_returns'] == {
        'training': {'visual': 2, 'constant': 2}, 'evaluation': {'visual': 2, 'constant': 1}}
    assert saved['model_forward_count_exact'] and saved['optimizer_update_count_exact']
    assert set(saved['metrics']) == {'initial', 'zero', 'visual', 'constant'}
    events = [json.loads(line) for line in (fit.FOLDER / 'fit_v0_progress.jsonl').read_text().splitlines()]
    assert [(row['arm'], row['update']) for row in events if row['event'] == 'update_complete'] == [
        ('visual', 1), ('constant', 1), ('visual', 2), ('constant', 2)]
    prior = (fit.FOLDER / 'fit_v0.json').read_bytes()
    with pytest.raises(FileExistsError):
        fit.run()
    assert (fit.FOLDER / 'fit_v0.json').read_bytes() == prior


def test_failed_preflight_precedes_data_and_model(monkeypatch, tmp_path):
    runner_fixture(monkeypatch, tmp_path)
    fit.PREFLIGHT.write_text(json.dumps(dict(passed=False)))
    def forbidden(*args): pytest.fail('Nothing may be constructed before preflight')
    monkeypatch.setattr(fit, 'load_dataset', forbidden)
    monkeypatch.setattr(fit, 'ResponseEnvelope', forbidden)
    with pytest.raises(ValueError, match='prospective'):
        fit.run()
    assert not (fit.FOLDER / 'fit_v0_attempt.json').exists()


def test_partial_optimizer_mutation_is_uncertain_and_preserved(monkeypatch, tmp_path):
    runner_fixture(monkeypatch, tmp_path)
    class PartialOptimizer:
        def __init__(self, parameters, **kwargs): self.parameters = list(parameters)
        def zero_grad(self, **kwargs):
            for value in self.parameters: value.grad = None
        def step(self):
            with torch.no_grad(): self.parameters[0][0] += 7
            raise RuntimeError('injected failure after mutation')
    monkeypatch.setattr(fit.torch.optim, 'Adam', PartialOptimizer)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 0 and saved['optimizer_updates_upper_bound'] == 1
    assert not saved['optimizer_update_count_exact'] and saved['pending_optimizer_arm'] == 'visual'
    assert saved['model_forward_returns']['training'] == {'visual': 1, 'constant': 0}
    weights = torch.load(fit.FOLDER / 'final_weights.pt', weights_only=True)
    assert weights['visual']['values'][0] == 7


def test_journal_failure_after_step_cannot_erase_return_or_weights(monkeypatch, tmp_path):
    runner_fixture(monkeypatch, tmp_path)
    original_open = Path.open
    class BrokenJournal:
        def __init__(self, stream): self.stream, self.broken = stream, False
        def write(self, text):
            if self.broken or json.loads(text)['event'] == 'optimizer_step_return':
                self.broken = True
                raise OSError('injected returned-step journal failure')
            return self.stream.write(text)
        def flush(self): return self.stream.flush()
        def fileno(self): return self.stream.fileno()
        def close(self): return self.stream.close()
    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        return BrokenJournal(stream) if path.name == 'fit_v0_progress.jsonl' and mode == 'x' else stream
    monkeypatch.setattr(Path, 'open', open_path)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == saved['optimizer_updates_upper_bound'] == 1
    assert saved['optimizer_update_count_exact'] and saved['pending_optimizer_arm'] is None
    assert (fit.FOLDER / 'final_weights.pt').exists()


def test_metric_rejection_keeps_unique_evaluation_output(monkeypatch, tmp_path):
    runner_fixture(monkeypatch, tmp_path)
    def reject(*args): raise ValueError('injected metric rejection')
    monkeypatch.setattr(fit, 'measure', reject)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 0
    assert saved['model_forward_returns']['evaluation'] == {'visual': 1, 'constant': 0}
    assert (fit.FOLDER / 'initial_logits.npz').exists()
    assert (fit.FOLDER / 'initial_predictions.npz').exists()
    assert (fit.FOLDER / 'final_weights.pt').exists()


@pytest.mark.parametrize('bad_bce', [float('inf'), float('nan')])
def test_nonfinite_evaluation_bce_stops_before_optimizer_and_preserves_parseable_failure(
        monkeypatch, tmp_path, bad_bce):
    runner_fixture(monkeypatch, tmp_path)
    optimizer_constructions = []
    def forbidden_optimizer(*args, **kwargs):
        optimizer_constructions.append(True)
        pytest.fail('A rejected initial evaluation must stop before optimizer construction')
    monkeypatch.setattr(fit.torch.optim, 'Adam', forbidden_optimizer)
    monkeypatch.setattr(fit, 'anchor_balanced_bce', lambda *_: torch.tensor(bad_bce))
    assert fit.run() == 1
    saved = result()
    assert not optimizer_constructions
    assert saved['execution_status'] == 'incomplete' and not saved['passed']
    assert saved['optimizer_updates'] == saved['optimizer_updates_upper_bound'] == 0
    assert saved['optimizer_update_count_exact']
    assert saved['model_forward_returns'] == {
        'training': {'visual': 0, 'constant': 0}, 'evaluation': {'visual': 1, 'constant': 0}}
    assert saved['metrics'] == {}
    assert any(row['type'] == 'FloatingPointError' for row in saved['failures'])
    for name in ('initial_logits.npz', 'initial_predictions.npz', 'initial_weights.pt', 'final_weights.pt'):
        assert (fit.FOLDER / name).exists()


def test_final_provenance_change_invalidates_an_otherwise_complete_run(monkeypatch, tmp_path):
    proof = runner_fixture(monkeypatch, tmp_path)
    calls = []
    def checked():
        calls.append(None)
        return proof if len(calls) == 1 else dict(proof, runtime={'changed': True})
    monkeypatch.setattr(fit, 'proofs', checked)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 4 and saved['passed'] is False
    assert not saved['final_provenance_verified']
    assert any(row['stage'] == 'final_provenance_or_time' for row in saved['failures'])


def test_forward_exception_remains_an_uncertain_call():
    class Broken(torch.nn.Module):
        def forward(self, inputs): raise RuntimeError('injected forward failure')
    model, phase = Broken(), {'name': 'training'}
    entries, returns = {'training': {'visual': 0}}, {'training': {'visual': 0}}
    fit.attach_forward_counters(model, 'visual', phase, entries, returns)
    with pytest.raises(RuntimeError):
        model(torch.zeros(1))
    assert entries['training']['visual'] == 1 and returns['training']['visual'] == 0


def test_payload_time_and_write_failure_cannot_publish_success(monkeypatch, tmp_path):
    clock_value = [0.0]
    monkeypatch.setattr(fit.time, 'monotonic', lambda: clock_value[0])
    original = fit.write_json
    payload = dict(failures=[], candidate_execution_complete=True, candidate_verdict=dict(passed=True, complete=True))
    payload_path, final_path = tmp_path / 'payload.json', tmp_path / 'result.json'
    def slow_write(path, value):
        original(path, value)
        if path == payload_path: clock_value[0] = 1201.
    monkeypatch.setattr(fit, 'write_json', slow_write)
    saved = fit.finish_result(payload, 0, payload_path, final_path)
    assert saved['execution_status'] == 'incomplete' and not saved['passed']
    assert saved['work_seconds'] == 1201 and saved['scientific_payload_sha256'] == fit.sha(payload_path)
    path2, final2 = tmp_path / 'payload2.json', tmp_path / 'result2.json'
    def interrupted_write(path, value):
        original(path, value)
        if path == path2: raise OSError('injected payload failure')
    monkeypatch.setattr(fit, 'write_json', interrupted_write)
    saved = fit.finish_result(payload, 1201., path2, final2)
    assert path2.exists() and final2.exists() and not saved['passed']
    assert saved['scientific_payload_sha256'] is None
