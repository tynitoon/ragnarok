"""Prospective fitting protocol checks; no optimizer update is executed here."""
import json

import pytest
import torch

from scripts import check_synthetic_motion_fit_v70 as run
from ragnarok.learning.parser_v70 import CounterfactualParser,max_pool_rgb

torch.set_num_threads(1)


def test_announced_protocol_keeps_two_scene_fit_distinct_from_generalization():
    protocol=json.loads((run.ROOT/run.PROTOCOL).read_text())
    assert protocol['spec']==run.SPEC and protocol['gate_result'] is None
    assert run.SPEC['updates_per_arm']==256 and run.SPEC['batch_size']==2
    assert run.SPEC['independent_generalization_test'] is False
    a,b,target=run.scenes();pooled=max_pool_rgb(a,b)
    assert torch.equal(pooled[0],pooled[1]) and not torch.equal(a[0],a[1])
    assert (target[0].float()/255-target[1].float()/255).abs().mean()==.046875


def test_measurement_checks_pooled_lower_bound_without_modifying_weights():
    torch.manual_seed(764);model=CounterfactualParser().cpu();before=run.weight_sha256(model)
    row=run.measure(model,'pooled',run.scenes(),0)
    assert row['image_l1']>=.0234375-1e-7 and row['reconstruction_difference']==0
    assert run.weight_sha256(model)==before and all(p.grad is None for p in model.parameters())


def test_failed_preflight_precedes_attempt_and_any_optimizer(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ATTEMPT',tmp_path/'attempt.json');monkeypatch.setattr(run,'OUTPUT',tmp_path/'result.json')
    def fail():raise ValueError('Synthetic missing independent preflight')
    monkeypatch.setattr(run,'preflight',fail)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('No optimizer before preflight'))
    with pytest.raises(ValueError,match='missing independent'):run.run()
    assert not run.ATTEMPT.exists()


def test_optimizer_failure_records_uncertain_step_without_retry(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'FOLDER',tmp_path)
    monkeypatch.setattr(run,'ATTEMPT',tmp_path/'attempt.json');monkeypatch.setattr(run,'OUTPUT',tmp_path/'result.json')
    monkeypatch.setattr(run,'preflight',lambda:{'synthetic_fixture':True})
    calls=[]
    class FailedOptimizer:
        def __init__(self,parameters,**kwargs):self.parameters=list(parameters)
        def zero_grad(self,**kwargs):
            for p in self.parameters:p.grad=None
        def step(self):
            calls.append('uncertain_step')
            # Fault injection only. No real optimizer/parameter update occurs.
            raise MemoryError('synthetic optimizer interruption')
    monkeypatch.setattr(torch.optim,'Adam',FailedOptimizer)
    with pytest.raises(MemoryError,match='optimizer interruption'):run.run()
    record=json.loads(run.ATTEMPT.read_text())
    assert record['status']=='failed' and record['completed_updates']=={'pooled':0}
    assert record['optimizer_step_in_progress']=={'arm':'pooled','update':1} and not run.OUTPUT.exists()
    with pytest.raises(ValueError,match='no retry'):run.run()
    assert calls==['uncertain_step']
