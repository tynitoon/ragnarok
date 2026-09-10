"""Certificate metric and publication witnesses; no neural forward or dataset generation."""
import json
import math

import numpy as np
import pytest
import torch

from scripts import build_stationary_control_v70 as builder


def _certificate_tensors():
    targets = np.zeros((3, 2, 3), dtype=bool)
    targets[0, 0, 0] = True
    logits = torch.from_numpy(np.where(targets[:, None], 8., -8.).astype(np.float32))
    cases = [dict(game=f'synthetic_{index}', anchor_index=0) for index in range(3)]
    return logits, targets, cases


def test_signed_eight_tensors_satisfy_the_unchanged_metrics_and_bce():
    result = builder.certificate_result(*_certificate_tensors())
    assert result['passed'] is True
    assert result['maximum_logit_absolute_error'] == 0
    # The protocol allows float32 BCE cancellation at the large negative logit.
    assert result['balanced_bce'] == pytest.approx(
        math.log1p(math.exp(-8)), abs=builder.SPEC['maximum_bce_absolute_error'])
    assert result['metrics']['mean_nonempty_iou'] == 1
    assert result['metrics']['empty_anchor_false_positive_fractions'] == [0, 0]


def test_wrong_reference_decision_is_a_failed_certificate_without_changing_the_threshold():
    logits, targets, cases = _certificate_tensors()
    logits[1, 0, 0, 0] = 8
    result = builder.certificate_result(logits, targets, cases)
    assert result['passed'] is False
    assert result['metrics']['empty_anchor_false_positive_fractions'][0] == pytest.approx(1 / 6)
    assert result['maximum_logit_absolute_error'] == 16


def test_nonfinite_reference_cannot_produce_a_positive_result():
    logits, targets, cases = _certificate_tensors()
    logits[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Finite float32'):
        builder.certificate_result(logits, targets, cases)


def test_returned_output_survives_a_broken_journal(tmp_path):
    logits, targets, cases = _certificate_tensors()
    path = tmp_path / 'returned.npz'
    def save_logits(value):
        with path.open('xb') as stream:
            np.savez_compressed(stream, **value)
    def broken_log(event):
        raise OSError('injected return journal failure')
    with pytest.raises(OSError, match='return journal failure'):
        builder.preserve_and_certify(logits, targets, cases, save_logits,
                                     broken_log, builder.time.monotonic() + 60)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['reference_logits'], logits[:, 0].numpy())


def test_nonfinite_returned_output_is_saved_before_metric_rejection(tmp_path):
    logits, targets, cases = _certificate_tensors()
    logits[0, 0, 0, 0] = float('nan')
    path = tmp_path / 'returned.npz'
    events = []
    def save_logits(value):
        with path.open('xb') as stream:
            np.savez_compressed(stream, **value)
    with pytest.raises(ValueError, match='Finite float32'):
        builder.preserve_and_certify(logits, targets, cases, save_logits,
                                     events.append, builder.time.monotonic() + 60)
    assert events == ['reference_forward_return']
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['reference_logits'], logits[:, 0].numpy())


def test_slow_scientific_payload_is_counted_before_the_final_verdict(monkeypatch, tmp_path):
    clock_value = [0.0]
    monkeypatch.setattr(builder.time, 'monotonic', lambda: clock_value[0])
    original = builder.write_json
    payload_path, final_path = tmp_path / 'payload.json', tmp_path / 'result.json'
    def slow_write(path, value):
        original(path, value)
        if path == payload_path:
            clock_value[0] = 301.0
    monkeypatch.setattr(builder, 'write_json', slow_write)
    payload = dict(errors=[], certificate=dict(passed=True), candidate_execution_status='complete')
    result = builder.finish_result(payload, 0.0, payload_path, final_path)
    assert json.loads(payload_path.read_text())['candidate_execution_status'] == 'complete'
    assert result['execution_status'] == 'incomplete' and result['passed'] is False
    assert result['work_seconds'] == 301.0
    assert result['scientific_payload_sha256'] == builder.sha(payload_path)


def test_payload_write_failure_preserves_it_and_publishes_failure(monkeypatch, tmp_path):
    original = builder.write_json
    payload_path, final_path = tmp_path / 'payload.json', tmp_path / 'result.json'
    def interrupted_write(path, value):
        original(path, value)
        if path == payload_path:
            raise OSError('injected payload close failure')
    monkeypatch.setattr(builder, 'write_json', interrupted_write)
    result = builder.finish_result(dict(errors=[], certificate=dict(passed=True)),
                                    builder.time.monotonic(), payload_path, final_path)
    assert payload_path.exists() and final_path.exists()
    assert result['execution_status'] == 'incomplete' and result['passed'] is False
    assert result['scientific_payload_sha256'] is None
    assert result['errors'][0]['stage'] == 'payload'


def test_complete_reference_miss_remains_a_scientific_failure(tmp_path):
    result = builder.finish_result(dict(errors=[], certificate=dict(passed=False)),
                                    builder.time.monotonic(), tmp_path / 'payload.json', tmp_path / 'result.json')
    assert result['execution_status'] == 'complete' and result['passed'] is False
    assert result['errors'] == []
