"""Fixed-array and receipt witnesses; no generated family, network or saved fit is used."""
import json

import numpy as np
import pytest

from scripts import validate_stationary_control_v70 as validation


def _draws():
    old_rgb = np.zeros((2, 2, 2, 3, 3), dtype=np.uint8)
    old_rgb[1] = 5
    new_rgb = old_rgb + 1
    target = np.zeros((2, 2, 3), bool)
    target[0, 0, 0] = True
    return new_rgb, target, old_rgb, target.copy()


def _measures():
    def entry(iou, empty):
        return dict(mean_nonempty_iou=iou, empty_anchor_false_positive_fractions=empty)
    return dict(visual=entry(.4, [.001, 0.]), constant=entry(.1, []),
                initial=entry(.2, []), zero=entry(0., []))


def test_unfulfilled_condition_prevents_any_array_model_or_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(validation, 'FIT_RESULT_SHA', None)
    monkeypatch.setattr(validation, 'FOLDER', tmp_path)
    def forbidden(*args, **kwargs): pytest.fail('No scientific input or calculation before the condition')
    for name in ('generate_screens', 'collect_interventions', 'ResponseEnvelope'):
        monkeypatch.setattr(validation, name, forbidden)
    monkeypatch.setattr(validation.np, 'load', forbidden)
    monkeypatch.setattr(validation.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='No reviewed successful fit'):
        validation.run()
    assert list(tmp_path.iterdir()) == []


def test_same_support_with_new_rgb_is_allowed_and_reported_without_exclusion():
    compared = validation.compare_draws(*_draws())
    assert compared['admissible'] is True
    assert compared['cases'][0]['exact_development_past_matches'] == []
    assert compared['cases'][0]['exact_development_mask_matches'] == [0]
    assert compared['cases'][0]['corresponding_support_iou'] == 1
    assert compared['cases'][1]['corresponding_support_iou'] is None


def test_exact_rgb_match_against_another_development_case_rejects_the_whole_draw():
    new_rgb, target, old_rgb, old_target = _draws()
    new_rgb[0] = old_rgb[1]
    compared = validation.compare_draws(new_rgb, target, old_rgb, old_target)
    assert compared['admissible'] is False
    assert compared['cases'][0]['exact_development_past_matches'] == [1]


def test_same_scientific_clauses_without_fictitious_optimizer_steps():
    measures = _measures()
    assert validation.validation_verdict(measures)['passed'] is True
    measures['visual']['empty_anchor_false_positive_fractions'][1] = .00101
    assert validation.validation_verdict(measures)['passed'] is False


def test_complete_duplicate_screen_assessment_is_explicitly_inconclusive(tmp_path):
    payload = dict(errors=[], comparison=dict(admissible=False), candidate_execution_complete=True, metrics={},
                   model_forward_entries=dict(initial=0, visual=0, constant=0), optimizer_updates=0)
    result = validation.finish_result(payload, validation.time.monotonic(), tmp_path / 'payload.json', tmp_path / 'result.json')
    assert result['execution_status'] == 'complete' and result['assessment_status'] == 'inconclusive'
    assert result['passed'] is False and result['optimizer_updates'] == 0
    assert json.loads((tmp_path / 'result.json').read_text())['verdict']['reason'] == 'Repeated development RGB input'


def test_payload_closure_over_budget_overrides_a_successful_candidate(monkeypatch, tmp_path):
    original = validation.write_json
    clock_value = [0.]
    monkeypatch.setattr(validation.time, 'monotonic', lambda: clock_value[0])
    payload_path, final_path = tmp_path / 'payload.json', tmp_path / 'result.json'
    def slow_write(path, value):
        original(path, value)
        if path == payload_path: clock_value[0] = 181.
    monkeypatch.setattr(validation, 'write_json', slow_write)
    payload = dict(errors=[], comparison=dict(admissible=True), candidate_execution_complete=True, metrics=_measures())
    result = validation.finish_result(payload, 0., payload_path, final_path)
    assert json.loads(payload_path.read_text())['candidate_verdict']['passed'] is True
    assert result['execution_status'] == 'incomplete' and result['assessment_status'] == 'inconclusive'
    assert result['passed'] is False and result['work_seconds'] == 181.
    assert result['scientific_payload_sha256'] == validation.sha(payload_path)


def test_payload_failure_preserves_output_and_cannot_publish_success(monkeypatch, tmp_path):
    original = validation.write_json
    payload_path, final_path = tmp_path / 'payload.json', tmp_path / 'result.json'
    def fail_write(path, value):
        original(path, value)
        if path == payload_path: raise OSError('injected payload failure')
    monkeypatch.setattr(validation, 'write_json', fail_write)
    payload = dict(errors=[], comparison=dict(admissible=True), candidate_execution_complete=True, metrics=_measures())
    result = validation.finish_result(payload, validation.time.monotonic(), payload_path, final_path)
    assert payload_path.exists() and final_path.exists() and result['passed'] is False
    assert result['scientific_payload_sha256'] is None
