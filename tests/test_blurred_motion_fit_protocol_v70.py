"""Prospective candidate protocol checks; no real optimizer update."""
import json

import pytest
import torch

from scripts import check_blurred_motion_fit_v70 as run

torch.set_num_threads(1)


def test_candidate_keeps_original_native_criterion_budget_and_immutable_baseline():
    protocol=json.loads((run.ROOT/run.PROTOCOL).read_text())
    assert protocol['spec']==run.SPEC and protocol['gate_result'] is None
    assert run.SPEC['native_fit_threshold']==run.base.SPEC['ordered_fit_threshold']
    assert run.SPEC['updates']==run.base.SPEC['updates_per_arm']==256
    assert run.SPEC['baseline_reexecuted'] is False
    assert run.baseline()['arm']=='ordered'


def test_candidate_measurement_preserves_native_values_and_reports_filtered_loss_separately():
    torch.manual_seed(764)
    with torch.device('cpu'):model=run.base.OrderedMotionParser()
    before=run.base.weight_sha256(model)
    row=run.measure(model,run.base.scenes(),0);old=run.baseline()['measurements'][0]
    assert {key:row[key] for key in old}==old
    assert row['candidate_objective']!=row['image_l1']
    # Filtered objective reduces the batch at once; the preserved metric first
    # reduces each scene. Float32 summation order need not be bitwise identical.
    assert row['scale_l1']['1']==pytest.approx(row['image_l1'],abs=1e-8,rel=0)
    assert run.base.weight_sha256(model)==before and all(p.grad is None for p in model.parameters())


def destinations(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'FOLDER',tmp_path)
    monkeypatch.setattr(run,'ATTEMPT',tmp_path/'attempt.json');monkeypatch.setattr(run,'OUTPUT',tmp_path/'result.json')


def test_unreviewed_candidate_cannot_create_an_optimizer_or_attempt(monkeypatch,tmp_path):
    destinations(monkeypatch,tmp_path)
    def fail():raise ValueError('Missing independent candidate preflight')
    monkeypatch.setattr(run,'preflight',fail)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('No optimizer before preflight'))
    with pytest.raises(ValueError,match='Missing independent'):run.run()
    assert not run.ATTEMPT.exists()


def test_changed_initial_weight_identity_stops_before_optimizer(monkeypatch,tmp_path):
    destinations(monkeypatch,tmp_path);old=run.baseline();old['initial_weight_sha256']='wrong synthetic fixture identity'
    monkeypatch.setattr(run,'baseline',lambda:old)
    monkeypatch.setattr(run,'preflight',lambda:{'synthetic_fixture':True})
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('No optimizer after initial mismatch'))
    with pytest.raises(ValueError,match='identical virgin'):run.run()
    record=json.loads(run.ATTEMPT.read_text())
    assert record['status']=='failed' and record['completed_updates']==0
    assert record['optimizer_step_in_progress'] is None


def test_uncertain_optimizer_call_is_retained_and_not_retried(monkeypatch,tmp_path):
    destinations(monkeypatch,tmp_path);monkeypatch.setattr(run,'preflight',lambda:{'synthetic_fixture':True})
    calls=[]
    class FailedOptimizer:
        def __init__(self,parameters,**kwargs):self.parameters=list(parameters)
        def zero_grad(self,**kwargs):
            for p in self.parameters:p.grad=None
        def step(self):
            calls.append(1)
            raise MemoryError('Synthetic step failure without real weight update')
    monkeypatch.setattr(torch.optim,'Adam',FailedOptimizer)
    with pytest.raises(MemoryError,match='Synthetic step failure'):run.run()
    record=json.loads(run.ATTEMPT.read_text())
    assert record['status']=='failed' and record['completed_updates']==0 and record['optimizer_step_in_progress']==1
    with pytest.raises(ValueError,match='no retry'):run.run()
    assert calls==[1] and not run.OUTPUT.exists()
