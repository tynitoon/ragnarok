"""Fault/isolation witnesses use tiny invented fixtures, never the real archive."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import fit_atari_affine_v70 as fit


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'FOLDER', tmp_path / 'fit')
    fit.FOLDER.mkdir()
    monkeypatch.setattr(fit, 'PREFLIGHT', fit.FOLDER / 'preflight_v0.json')
    monkeypatch.setattr(fit, 'TARGET_RESULT', tmp_path / 'control.json')
    monkeypatch.setattr(fit, 'SPEC', dict(fit.SPEC, newton_updates_per_arm=1))
    proof = dict(source_sha256={}, evidence_sha256={}, runtime={'fixture': True})
    monkeypatch.setattr(fit, 'proofs', lambda: proof)
    monkeypatch.setattr(fit, 'committed', lambda path: None)
    fit.write_json(fit.PREFLIGHT, dict(kind='atari_affine_independent_preflight', passed=True, **proof))
    cases = [dict(game=f'synthetic_pair_{i // 2}', anchor_index=i % 2) for i in range(12)]
    fit.write_json(fit.TARGET_RESULT, dict(cases=cases))
    # Non-separable all-zero RGB with observed positive labels: no color rule is injected.
    past = np.zeros((12, 2, 2, 3, 3), np.uint8)
    targets = np.zeros((12, 2, 3), bool)
    targets[:, 0, 0] = True
    targets[6:8] = False
    monkeypatch.setattr(fit, 'load_dataset', lambda result: (past, targets))
    return proof


def result():
    return json.loads((fit.FOLDER / 'fit_v0.json').read_text())


def test_import_never_loads_a_generator_certificate_or_emulator():
    assert not {'scripts.build_stationary_control_v70',
                'ragnarok.evaluation.response_envelope_certificate_v70',
                'ragnarok.environments.stationary_intervention_v70', 'ale_py'} & sys.modules.keys()


def test_loader_decodes_only_past_and_observed_labels(monkeypatch):
    past = np.zeros((12, 2, 210, 160, 3), np.uint8)
    targets = np.zeros((12, 210, 160), bool)
    for index, count in enumerate([144, 296, 344, 184, 127, 127, 0, 0, 109, 184, 78, 54]):
        targets[index].reshape(-1)[:count] = True
    accessed = []
    class Archive:
        files = ['past_pairs', 'observed_response_envelopes']
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key):
            accessed.append(key)
            if key == 'past_pairs': return past
            if key == 'observed_response_envelopes': return targets
            pytest.fail('Future branches must not be decoded')
    monkeypatch.setattr(fit.np, 'load', lambda *args, **kwargs: Archive())
    control = dict(output=dict(arrays={
        'past_pairs': fit.signature(past), 'observed_response_envelopes': fit.signature(targets)}),
        cases=[dict(game=game, anchor_index=i) for game in ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids') for i in range(2)])
    observed_past, observed_targets = fit.load_dataset(control)
    assert accessed == ['past_pairs', 'observed_response_envelopes']
    np.testing.assert_array_equal(observed_past, past)
    np.testing.assert_array_equal(observed_targets, targets)
    assert observed_targets.sum() == 1647 and not observed_past.any()


def test_complete_schedule_separate_costs_zero_initial_and_no_retry(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    assert fit.run() == 0
    saved = result()
    assert saved['execution_status'] == 'complete' and saved['optimizer_updates'] == 2
    assert saved['operation_entries'] == saved['operation_returns'] == fit.expected_counts()
    assert saved['operation_totals'] == dict(linear_prediction=7, objective=7, gradient=2,
        hessian=2, linear_solve=2, parameter_update=2)
    assert saved['operation_count_exact'] and saved['optimizer_update_count_exact']
    assert not saved['passed']
    with np.load(fit.FOLDER / 'initial_state.npz') as data:
        assert len(data.files) == 4 and all(not data[key].any() for key in data.files)
    events = [json.loads(line) for line in (fit.FOLDER / 'fit_v0_progress.jsonl').read_text().splitlines()]
    assert [(row['arm'], row['update']) for row in events if row['event'] == 'update_complete'] == [
        ('newton_visual', 1), ('newton_constant', 1)]
    prior = (fit.FOLDER / 'fit_v0.json').read_bytes()
    with pytest.raises(FileExistsError): fit.run()
    assert (fit.FOLDER / 'fit_v0.json').read_bytes() == prior


def test_failed_preflight_precedes_data_or_calculation(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    fit.PREFLIGHT.write_text(json.dumps(dict(passed=False)))
    def forbidden(*args): pytest.fail('No data or calculation before preflight')
    monkeypatch.setattr(fit, 'load_dataset', forbidden)
    with pytest.raises(ValueError, match='prospective'): fit.run()
    assert not (fit.FOLDER / 'fit_v0_attempt.json').exists()


def test_partial_update_preserves_mutated_state_and_uncertainty(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    def partial(states, arm, candidate):
        states[arm]['coefficients'][0] = 7.
        raise RuntimeError('Injected interruption after mutation')
    monkeypatch.setattr(fit, 'apply_candidate', partial)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 0 and saved['optimizer_updates_upper_bound'] == 1
    assert not saved['optimizer_update_count_exact'] and not saved['operation_count_exact']
    assert saved['pending_update_arm'] == 'newton_visual'
    assert saved['operation_returns']['linear_solve']['newton_visual'] == 1
    with np.load(fit.FOLDER / 'final_state.npz') as data:
        assert data['newton_visual_coefficients'][0] == 7


def broken_journal(monkeypatch, reject):
    original_open = Path.open
    class BrokenJournal:
        def __init__(self, stream): self.stream, self.broken = stream, False
        def write(self, text):
            if self.broken or reject(json.loads(text)):
                self.broken = True
                raise OSError('Injected journal failure after return')
            return self.stream.write(text)
        def flush(self): return self.stream.flush()
        def fileno(self): return self.stream.fileno()
        def close(self): return self.stream.close()
    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        return BrokenJournal(stream) if path.name == 'fit_v0_progress.jsonl' and mode == 'x' else stream
    monkeypatch.setattr(Path, 'open', open_path)


def test_returned_update_is_exact_despite_journal_failure(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    broken_journal(monkeypatch, lambda row: row['event'] == 'operation_return'
                   and row['operation'] == 'parameter_update')
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == saved['optimizer_updates_upper_bound'] == 1
    assert saved['optimizer_update_count_exact'] and saved['pending_update_arm'] is None
    with np.load(fit.FOLDER / 'final_state.npz') as data:
        assert data['newton_visual_step'] == 1 and data['newton_visual_coefficients'][-1] < 0


def test_unique_evaluation_saved_before_return_journal_failure(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    broken_journal(monkeypatch, lambda row: row['event'] == 'operation_return'
                   and row['operation'] == 'linear_prediction' and row['arm'] == 'initial')
    assert fit.run() == 1
    saved = result()
    assert saved['operation_returns']['linear_prediction']['initial'] == 1
    assert saved['optimizer_updates'] == 0 and saved['metrics'] == {}
    assert (fit.FOLDER / 'initial_logits.npz').exists()
    assert (fit.FOLDER / 'final_state.npz').exists()


@pytest.mark.parametrize('stage', ['metric', 'objective'])
def test_bad_evaluation_keeps_logits_and_probabilities_before_rejection(monkeypatch, tmp_path, stage):
    fixture(monkeypatch, tmp_path)
    if stage == 'objective':
        monkeypatch.setattr(fit.affine, 'objective', lambda *_: float('nan'))
    else:
        def reject(*args): raise ValueError('Injected metric rejection')
        monkeypatch.setattr(fit, 'measure', reject)
    assert fit.run() == 1
    saved = result()
    assert saved['execution_status'] == 'incomplete' and saved['optimizer_updates'] == 0
    assert saved['metrics'] == {} and not saved['passed']
    for name in ('initial_logits.npz', 'initial_predictions.npz', 'initial_state.npz', 'final_state.npz'):
        assert (fit.FOLDER / name).exists()


def test_solver_exception_is_counted_without_mutation(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    def broken(*args): raise np.linalg.LinAlgError('Injected solve failure')
    monkeypatch.setattr(fit.affine, 'newton_candidate', broken)
    assert fit.run() == 1
    saved = result()
    assert saved['operation_entries']['linear_solve']['newton_visual'] == 1
    assert saved['operation_returns']['linear_solve']['newton_visual'] == 0
    assert not saved['operation_count_exact'] and saved['optimizer_updates'] == 0
    assert saved['optimizer_update_count_exact']


def test_newton_loss_increase_stops_after_counted_update_without_retry(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    def ascent(coefficients, first, second): return coefficients + first
    monkeypatch.setattr(fit.affine, 'newton_candidate', ascent)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 1 and saved['optimizer_update_count_exact']
    assert any('Newton objective increased' in row['message'] for row in saved['failures'])


def test_final_provenance_change_invalidates_completed_math(monkeypatch, tmp_path):
    proof = fixture(monkeypatch, tmp_path)
    calls = []
    def checked():
        calls.append(None)
        return proof if len(calls) == 1 else dict(proof, runtime={'changed': True})
    monkeypatch.setattr(fit, 'proofs', checked)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 2 and not saved['final_provenance_verified']
    assert not saved['passed'] and saved['execution_status'] == 'incomplete'


def test_late_payload_invalidates_completed_math(monkeypatch, tmp_path):
    payload = dict(failures=[], candidate_execution_complete=True, candidate_verdict=dict(passed=True))
    clock_value = [0.]
    monkeypatch.setattr(fit.time, 'monotonic', lambda: clock_value[0])
    original = fit.write_json
    def slow_write(path, value):
        original(path, value)
        if path.name == 'late_payload.json': clock_value[0] = 121.
    monkeypatch.setattr(fit, 'write_json', slow_write)
    saved = fit.finish_result(payload, 0, tmp_path / 'late_payload.json', tmp_path / 'late_result.json')
    assert saved['work_seconds'] == 121 and saved['execution_status'] == 'incomplete'
    assert not saved['passed'] and (tmp_path / 'late_payload.json').exists()


def test_payload_write_failure_cannot_publish_scientific_success(monkeypatch, tmp_path):
    payload = dict(failures=[], candidate_execution_complete=True,
                   candidate_verdict=dict(passed=True))
    original = fit.write_json
    def broken(path, value):
        original(path, value)
        if path.name == 'payload.json': raise OSError('Injected payload closure failure')
    monkeypatch.setattr(fit, 'write_json', broken)
    saved = fit.finish_result(payload, fit.time.monotonic(), tmp_path / 'payload.json', tmp_path / 'result.json')
    assert saved['execution_status'] == 'incomplete' and not saved['passed']
    assert saved['scientific_payload_sha256'] is None and (tmp_path / 'payload.json').exists()


def test_journal_hash_failure_keeps_complete_states_and_metrics(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    original = fit.sha
    def reject(path):
        if path.name == 'fit_v0_progress.jsonl': raise OSError('Injected closed journal hash failure')
        return original(path)
    monkeypatch.setattr(fit, 'sha', reject)
    assert fit.run() == 1
    saved = result()
    assert saved['optimizer_updates'] == 2 and saved['optimizer_update_count_exact']
    assert set(saved['metrics']) == {'initial', 'zero', 'newton_visual', 'newton_constant'}
    assert saved['progress_sha256'] is None and (fit.FOLDER / 'final_state.npz').exists()
    assert any(row['stage'] == 'trace_hash_fit_v0_progress.jsonl' for row in saved['failures'])
