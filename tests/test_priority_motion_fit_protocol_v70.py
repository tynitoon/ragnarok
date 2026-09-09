"""Prospective fitting checks; no real optimizer update."""
import json

import pytest
import torch

from scripts import check_priority_motion_fit_v70 as run

torch.set_num_threads(1)


def test_initial_visual_identity_and_rounding_contract_without_updates():
    protocol = json.loads((run.ROOT / run.PROTOCOL).read_text())
    assert protocol['spec'] == run.SPEC and protocol['gate_result'] is None
    torch.manual_seed(764)
    model = run.PriorityMotionParser()
    before = run.base.weight_sha256(model)
    first = run.initial_measurement(model, run.base.scenes())
    assert first['log_priority_abs_max'] == 0
    assert run.SPEC['native_fit_threshold'] == .01171875
    assert run.SPEC['updates'] == 256 and run.SPEC['foreground_or_control_objective_used'] is False
    assert run.base.weight_sha256(model) == before
    assert all(p.grad is None for p in model.parameters())


def destinations(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'FOLDER', tmp_path)
    monkeypatch.setattr(run, 'ATTEMPT', tmp_path / 'attempt.json')
    monkeypatch.setattr(run, 'OUTPUT', tmp_path / 'result.json')


def test_missing_preflight_prevents_optimizer_and_attempt(monkeypatch, tmp_path):
    destinations(monkeypatch, tmp_path)
    def fail():
        raise ValueError('Missing independent preflight')
    monkeypatch.setattr(run, 'preflight', fail)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('No optimizer before preflight'))
    with pytest.raises(ValueError, match='Missing independent'):
        run.run()
    assert not run.ATTEMPT.exists()


def test_failed_initial_identity_stops_before_optimizer(monkeypatch, tmp_path):
    destinations(monkeypatch, tmp_path)
    monkeypatch.setattr(run, 'preflight', lambda: {'synthetic_fixture': True})
    def fail(*args):
        raise ValueError('Wrong virgin visual identity')
    monkeypatch.setattr(run, 'initial_measurement', fail)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('No optimizer after initial mismatch'))
    with pytest.raises(ValueError, match='virgin visual'):
        run.run()
    attempt = json.loads(run.ATTEMPT.read_text())
    assert attempt['completed_updates'] == 0 and attempt['optimizer_step_in_progress'] is None


def test_uncertain_step_is_retained_and_never_retried(monkeypatch, tmp_path):
    destinations(monkeypatch, tmp_path)
    monkeypatch.setattr(run, 'preflight', lambda: {'synthetic_fixture': True})
    calls = []
    class FailedOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)
        def zero_grad(self, **kwargs):
            for parameter in self.parameters:
                parameter.grad = None
        def step(self):
            calls.append(1)
            raise MemoryError('Synthetic step failure without a real parameter update')
    monkeypatch.setattr(torch.optim, 'Adam', FailedOptimizer)
    with pytest.raises(MemoryError, match='Synthetic step failure'):
        run.run()
    attempt = json.loads(run.ATTEMPT.read_text())
    assert attempt['completed_updates'] == 0 and attempt['optimizer_step_in_progress'] == 1
    assert not run.OUTPUT.exists()
    with pytest.raises(ValueError, match='no retry'):
        run.run()
    assert calls == [1]
