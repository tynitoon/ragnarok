"""Prospective diagnostic contracts; no real optimizer updates."""
import json

import pytest
import torch

from scripts import check_free_displacement_fit_v70 as run

torch.set_num_threads(1)


def test_initial_fields_use_no_rng_or_target_and_reconstruct_the_common_source():
    protocol = json.loads((run.ROOT / run.PROTOCOL).read_text())
    assert protocol['spec'] == run.SPEC and protocol['gate_result'] is None
    rng_before = torch.get_rng_state().clone()
    model = run.FreeDisplacements()
    assert torch.equal(rng_before, torch.get_rng_state())
    assert model.displacement.shape == (2, 2, 16, 16)
    assert model.displacement.dtype == torch.float32
    assert model.displacement.device.type == 'cpu' and torch.count_nonzero(model.displacement) == 0
    assert sum(p.numel() for p in model.parameters()) == 1024
    previous, current, future = run.base.scenes()
    source = run.base.max_pool_rgb(previous, current).float() / 255
    target = future.float() / 255
    row = run.measure(model, source, target, 0)
    assert row['image_l1'] == .046875 and row['reconstruction_difference'] == 0
    assert run.SPEC['native_fit_threshold'] == run.prior.SPEC['native_fit_threshold'] == .01171875
    terms, reconstructed, coverage = run.objective(model, source, target)
    assert torch.equal(reconstructed, source) and torch.equal(coverage, torch.ones_like(coverage))
    before = run.base.weight_sha256(model)
    terms['loss'].backward()
    assert torch.isfinite(model.displacement.grad).all()
    assert torch.count_nonzero(model.displacement.grad) > 0 and target.grad is None
    assert run.base.weight_sha256(model) == before


def destinations(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'FOLDER', tmp_path)
    monkeypatch.setattr(run, 'ATTEMPT', tmp_path / 'attempt.json')
    monkeypatch.setattr(run, 'OUTPUT', tmp_path / 'result.json')


def test_missing_preflight_prevents_field_creation_optimizer_and_attempt(monkeypatch, tmp_path):
    destinations(monkeypatch, tmp_path)
    def fail():
        raise ValueError('Missing independent preflight')
    monkeypatch.setattr(run, 'preflight', fail)
    monkeypatch.setattr(run, 'FreeDisplacements', lambda: pytest.fail('No field before preflight'))
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('No optimizer before preflight'))
    with pytest.raises(ValueError, match='Missing independent'):
        run.run()
    assert not run.ATTEMPT.exists()


def test_uncertain_optimizer_call_is_retained_and_never_retried(monkeypatch, tmp_path):
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
            raise MemoryError('Synthetic step failure without parameter update')
    monkeypatch.setattr(torch.optim, 'Adam', FailedOptimizer)
    with pytest.raises(MemoryError, match='Synthetic step failure'):
        run.run()
    record = json.loads(run.ATTEMPT.read_text())
    assert record['status'] == 'failed' and record['completed_updates'] == 0
    assert record['optimizer_step_in_progress'] == 1 and not run.OUTPUT.exists()
    with pytest.raises(ValueError, match='no retry'):
        run.run()
    assert calls == [1]


def test_elapsed_work_bound_prevents_first_step_and_preserves_attempt(monkeypatch, tmp_path):
    destinations(monkeypatch, tmp_path)
    monkeypatch.setattr(run, 'preflight', lambda: {'synthetic_fixture': True})
    times = iter((0, run.SPEC['max_work_seconds'] + 1))
    monkeypatch.setattr(run.time, 'monotonic', lambda: next(times))
    class UnusedOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def step(self):
            pytest.fail('Elapsed work bound must prevent every step')
    monkeypatch.setattr(torch.optim, 'Adam', UnusedOptimizer)
    with pytest.raises(ValueError, match='work ceiling'):
        run.run()
    record = json.loads(run.ATTEMPT.read_text())
    assert record['completed_updates'] == 0 and record['optimizer_step_in_progress'] is None
    assert record['status'] == 'failed' and not run.OUTPUT.exists()
