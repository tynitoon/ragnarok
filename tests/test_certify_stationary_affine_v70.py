"""Small made-up coefficient fixtures, never saved training weights or pixels."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import certify_stationary_affine_v70 as judge


def fixture(monkeypatch, tmp_path):
    folder = tmp_path / 'family'
    folder.mkdir()
    monkeypatch.setattr(judge, 'FOLDER', folder)
    monkeypatch.setattr(judge, 'PREFLIGHT', folder / 'preflight.json')
    monkeypatch.setattr(judge, 'FIT', tmp_path / 'fit')
    monkeypatch.setattr(judge, 'RESULT', tmp_path / 'fit.json')
    monkeypatch.setattr(judge, 'committed', lambda _: None)
    proof = dict(source_sha256={}, evidence_sha256={}, environment={'fixture': True})
    monkeypatch.setattr(judge, 'proofs', lambda: proof)
    judge.PREFLIGHT.write_text(json.dumps(dict(kind='stationary_affine_family_independent_preflight', passed=True, **proof)))
    judge.RESULT.write_text(json.dumps(dict(artifacts={'initial_state.npz': {}, 'final_state.npz': {}})))
    initial = {arm + '_coefficients': np.zeros(7) for arm in judge.ARMS[1:]}
    final = {key: value.copy() for key, value in initial.items()}
    final['newton_visual_coefficients'][:] = [1., 0., 0., 1., 0., 0., -1.]
    final['newton_constant_coefficients'][-1] = -1.
    monkeypatch.setattr(judge, 'decode_state', lambda path, artifact: initial if path.name.startswith('initial') else final)
    return proof


def result():
    return json.loads((judge.FOLDER / 'family_v0.json').read_text())


def test_all_five_sets_reported_in_order_with_exact_costs_and_no_retry(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    assert judge.run() == 0
    saved = result()
    assert saved['execution_status'] == 'complete' and saved['passed']
    assert list(saved['certificates']) == list(judge.ARMS)
    assert saved['operation_entries'] == saved['operation_returns'] == dict(state_archive=2, certificate=5)
    assert not saved['certificates']['adam_visual']['certified']
    original = (judge.FOLDER / 'family_v0.json').read_bytes()
    with pytest.raises(FileExistsError): judge.run()
    assert (judge.FOLDER / 'family_v0.json').read_bytes() == original


def test_failed_preflight_stops_before_any_state_decode(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    judge.PREFLIGHT.write_text(json.dumps(dict(passed=False)))
    def forbidden(*args): pytest.fail('No coefficients before exact preflight')
    monkeypatch.setattr(judge, 'decode_state', forbidden)
    assert judge.run() == 1
    saved = result()
    assert not saved['passed'] and saved['operation_entries'] == dict(state_archive=0, certificate=0)
    assert (judge.FOLDER / 'family_v0_attempt.json').exists()


def test_failed_archive_remains_uncertain_without_certificate(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    def broken(*args): raise ValueError('Injected array signature failure')
    monkeypatch.setattr(judge, 'decode_state', broken)
    assert judge.run() == 1
    saved = result()
    assert saved['operation_entries']['state_archive'] == 1 and saved['operation_returns']['state_archive'] == 0
    assert not saved['operation_counts_exact'] and saved['certificates'] == {}


def test_failed_second_certificate_keeps_first_and_uncertain_call(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    original, calls = judge.exact_family_bounds, []
    def broken(coefficients):
        calls.append(None)
        if len(calls) == 2: raise ValueError('Injected certificate failure')
        return original(coefficients)
    monkeypatch.setattr(judge, 'exact_family_bounds', broken)
    assert judge.run() == 1
    saved = result()
    assert list(saved['certificates']) == ['initial']
    assert saved['operation_entries']['certificate'] == 2 and saved['operation_returns']['certificate'] == 1
    assert not saved['passed'] and not saved['operation_counts_exact']


def test_returned_certificate_survives_journal_failure(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    original_open = Path.open
    class Broken:
        def __init__(self, stream): self.stream = stream
        def write(self, data):
            if json.loads(data)['event'] == 'certificate_return': raise OSError('Injected returned journal failure')
            return self.stream.write(data)
        def flush(self): return self.stream.flush()
        def fileno(self): return self.stream.fileno()
        def close(self): return self.stream.close()
    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        return Broken(stream) if path.name == 'family_v0_progress.jsonl' and mode == 'xb' else stream
    monkeypatch.setattr(Path, 'open', open_path)
    assert judge.run() == 1
    saved = result()
    assert list(saved['certificates']) == ['initial']
    assert saved['operation_returns']['certificate'] == saved['operation_entries']['certificate'] == 1
    assert saved['operation_counts_exact'] and not saved['passed']


def test_late_payload_cannot_publish_success(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    clock_value = [0.]
    monkeypatch.setattr(judge.time, 'monotonic', lambda: clock_value[0])
    original = judge.sha
    def slow_hash(path):
        digest = original(path)
        if path.name == 'family_v0_payload.json': clock_value[0] = 61.
        return digest
    monkeypatch.setattr(judge, 'sha', slow_hash)
    assert judge.run() == 1
    saved = result()
    assert len(saved['certificates']) == 5 and saved['work_seconds'] == 61
    assert saved['execution_status'] == 'incomplete' and not saved['passed']
    assert saved['scientific_payload_sha256'] is not None
