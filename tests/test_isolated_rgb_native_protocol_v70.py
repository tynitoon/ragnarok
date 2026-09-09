"""Prospective runner failure/provenance checks; synthetic engines only."""
from dataclasses import fields
import hashlib
import json
import time

import numpy as np
import pytest

from scripts import check_isolated_rgb_native_v70 as run
from ragnarok.environments.isolated_rgb_pool_v70 import IsolatedWorkerError
from tests.test_isolated_rgb_worker_v70 import subject


def test_fixed_protocol_and_paid_costs_are_explicit():
    protocol=json.loads((run.ROOT/run.PROTOCOL).read_text())
    assert protocol['spec']==run.SPEC and protocol['child_runtime']==run.CHILD
    base=json.loads((run.ROOT/run.BASE_RESULT).read_text())
    assert sum(c['accounting']['charged_frames'] for c in base['cases'])==7454
    assert sum(c['decisions'] for c in base['cases'])==1266
    assert len([c for c in base['cases'] if c['branch']])==6
    assert protocol['agent_training_allowed'] is False and protocol['gate_result'] is None


@pytest.mark.parametrize('later',[None,{'charged_frames':0}])
def test_unknown_or_regressed_report_never_erases_acknowledged_paid_frames(later):
    reports=run.Reports();reports.begin('complete');reports.begin('failed');reports.begin('constructor_unknown')
    reports.remember('complete',dict(accounting={'charged_frames':852},failure=None))
    reports.remember('failed',dict(accounting={'charged_frames':601},failure={'error':'paid'}))
    reports.remember('failed',dict(accounting=later,failure={'error':'report failed'}))
    result=reports.collect()
    assert result['known_charged_frames']==1453 and result['constructor_count']==3
    assert result['uncertain_engine_reports']==['failed','constructor_unknown']
    assert result['accounting_complete'] is False


def fixture(monkeypatch,tmp_path,*,damage=False,paid_failure=False):
    witness=subject();trace=witness.step_trace(0);digest=hashlib.sha256();run.trace_digest(digest,trace)
    arrays={f's000_{f.name}':getattr(trace,f.name).copy() for f in fields(trace)}
    if damage:arrays['s000_raw'][0,0,0,0]^=1
    path=tmp_path/'synthetic.npz';np.savez(path,**arrays)
    case=dict(name='fixture',game='pong',seed=1,limit=108000,decisions=1,branch_after=None)
    baseline=dict(name='fixture',passed=True,decisions=1,length_counts={'4':1},final_epoch=0,
        final_delivered_frame=4,ordered_trace_sha256=digest.hexdigest(),selected_trace_archive=str(path),
        selected_steps=[dict(index=0,action=0,operation_id=['owned-ordinary-v0/fixture',1],epoch=0,
            start_frame=0,end_frame=4,visual_discontinuity=False,actual_frames=4,effect_assumption=[0.]*12)],
        branch=None,accounting=witness.accounting())
    instances=[]
    class Pool:
        def __init__(self,*args,**kwargs):
            self.engine=subject();self._peers=[];self.closed=False;instances.append(self)
        def observe_pair(self,index=0):return self.engine.observe_pair()
        def step(self,actions):
            if paid_failure:
                self.engine._budget.record_play(1)
                raise IsolatedWorkerError('Paid synthetic failure',reports={0:self.report(0)})
            return [self.engine.step_trace(actions[0])]
        def report(self,index):return dict(accounting=self.engine.accounting(),failure=None)
        def close(self):self.closed=True
    monkeypatch.setattr(run,'IsolatedRGBPool',Pool)
    return case,baseline,instances


def test_synthetic_fidelity_pass_keeps_costs_and_closes(monkeypatch,tmp_path):
    case,baseline,instances=fixture(monkeypatch,tmp_path);reports=run.Reports()
    row=run.run_case(case,baseline,reports,time.monotonic()+10)
    assert row['ordered_trace_sha256']==baseline['ordered_trace_sha256']
    assert reports.collect()['known_charged_frames']==604 and instances[0].closed


def test_single_changed_archived_pixel_fails_after_retaining_costs_and_closing(monkeypatch,tmp_path):
    case,baseline,instances=fixture(monkeypatch,tmp_path,damage=True);reports=run.Reports()
    with pytest.raises(ValueError,match='RGB differs'):run.run_case(case,baseline,reports,time.monotonic()+10)
    assert reports.collect()['known_charged_frames']==604 and instances[0].closed


def test_paid_failure_preserves_report_and_stops_without_retry(monkeypatch,tmp_path):
    case,baseline,instances=fixture(monkeypatch,tmp_path,paid_failure=True);reports=run.Reports()
    with pytest.raises(IsolatedWorkerError,match='Paid synthetic'):run.run_case(case,baseline,reports,time.monotonic()+10)
    assert reports.collect()['known_charged_frames']==601 and instances[0].closed


def test_existing_attempt_refuses_before_preflight_or_constructor(monkeypatch,tmp_path):
    marker=tmp_path/'attempt.json';marker.write_text('{"status":"failed"}')
    monkeypatch.setattr(run,'ATTEMPT',marker)
    monkeypatch.setattr(run,'preflight',lambda:pytest.fail('No preflight after existing attempt'))
    with pytest.raises(ValueError,match='no retry'):run.run()
    assert marker.read_text()=='{"status":"failed"}'


def test_deadline_refuses_constructor(monkeypatch,tmp_path):
    case,baseline,instances=fixture(monkeypatch,tmp_path);reports=run.Reports()
    with pytest.raises(ValueError,match='ceiling'):run.run_case(case,baseline,reports,time.monotonic()-1)
    assert not instances and reports.collect()['constructor_count']==0
