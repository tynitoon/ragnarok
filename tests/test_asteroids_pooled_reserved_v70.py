"""The reserved executable keeps a durable single attempt and rejects unsealed construction."""
import json
import pytest
from scripts import validate_asteroids_pooled_v70 as runner


def test_unsealed_prerequisite_cannot_construct_or_burn_the_reserved_seed(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'DESTINATION',tmp_path/'raw')
    monkeypatch.setattr(runner,'ATTEMPT',tmp_path/'capture_attempt.json')
    def reject():raise ValueError('not sealed')
    monkeypatch.setattr(runner,'check',reject)
    monkeypatch.setattr(runner,'_ColdPixelEngine',lambda _:pytest.fail('constructed emulator'))
    with pytest.raises(ValueError,match='not sealed'):runner.capture()
    assert not runner.ATTEMPT.exists() and not runner.DESTINATION.exists()


def test_failed_constructor_burns_the_single_reserved_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'DESTINATION',tmp_path/'raw')
    monkeypatch.setattr(runner,'ATTEMPT',tmp_path/'capture_attempt.json')
    monkeypatch.setattr(runner,'check',lambda:dict(proof='checked'))
    monkeypatch.setattr(runner,'git',lambda *args:'sealed-commit')
    def fail(config):
        assert json.loads(runner.ATTEMPT.read_text())['status']=='started'
        raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(runner,'_ColdPixelEngine',fail)
    with pytest.raises(RuntimeError,match='synthetic'):runner.capture()
    record=json.loads(runner.ATTEMPT.read_text())
    assert record['status']=='failed' and record['accounting_status']=='constructor_did_not_return_cost_unknown_no_retry'
    with pytest.raises(ValueError,match='earlier attempt'):runner.capture()


def test_score_attempt_never_collides_with_capture_attempt():
    assert runner.RESULT.with_name(runner.RESULT.stem+'_attempt.json')!=runner.ATTEMPT


def test_capture_does_not_use_the_development_seed():
    assert runner.SPEC['development_seed']!=runner.SPEC['reserved_seed']
