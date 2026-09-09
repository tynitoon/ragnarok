"""Synthetic provenance, terminal chronology and retained paid failure; no ALE."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ragnarok.environments.ordinary_rgb_trace_v70 import _OrdinaryRGBTraceEngine
from scripts import check_owned_ordinary_native_v70 as check
from tests.test_rgb_branch_trace_wiring_v70 import synthetic as base_synthetic

WITNESS_CLASS=check._WitnessEngine


def synthetic(cls,cfg):
    obj=base_synthetic(cls,cfg)
    obj._ordinary_starts=None
    obj._witness_raw=None;obj.last_trace=None;obj._native_wrapper=None
    obj._witness_incomplete=False
    if cls is WITNESS_CLASS:
        obj.set_native_wrapper(lambda native:check._RGBReadTap(native,obj))
        cold=obj._cold_start
        def recreated(*,experimental):
            cold(experimental=experimental)
            obj._ale=obj._native_wrapper(obj._ale)
        obj._cold_start=recreated
    return obj


class SyntheticPool:
    def __init__(self,configs):
        self.engine=synthetic(_OrdinaryRGBTraceEngine,configs[0]);self.closed=False
        self._processes=[SimpleNamespace(is_alive=lambda:not self.closed)]
    def observe_pair(self,index=0):return self.engine.observe_pair()
    def step(self,actions):return [self.engine.step_trace(actions[0])]
    def branch_pair_trace(self,action,*,control,frames,index):return self.engine.branch_pair_trace(action,control,frames)
    def report(self,index):return dict(accounting=self.engine.accounting(),failure=self.engine._trace_failure)
    def close(self):self.closed=True


def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'ROOT',tmp_path);monkeypatch.setattr(check,'FOLDER',tmp_path)
    for field in ('OUTPUT','ATTEMPT','PREFLIGHT'):monkeypatch.setattr(check,field,tmp_path/(field+'.json'))
    monkeypatch.setattr(check,'preflight',lambda:{'synthetic':True})
    cls=check._WitnessEngine
    monkeypatch.setattr(check,'_WitnessEngine',lambda cfg:synthetic(cls,cfg))
    monkeypatch.setattr(check,'OrdinaryRGBPool',SyntheticPool)


@pytest.mark.parametrize('length',[1,2,3])
def test_actual_short_prefix_cold_recreation_and_archived_chronology(monkeypatch,tmp_path,length):
    setup(monkeypatch,tmp_path)
    result=check.run_case(check.CASES[5+length],check._Reports())
    assert result['passed'] and result['length_counts']=={4:129,length:1}
    assert result['final_epoch']==1 and result['final_delivered_frame']==8
    assert result['accounting']['charged_frames']==2*132+516+length
    archive=np.load(tmp_path/result['selected_trace_archive'],allow_pickle=False)
    assert len(archive.files)==32
    assert archive['s127_raw'].shape[0]==length and archive['s128_raw'].shape[0]==4
    assert not np.array_equal(archive['s127_raw'][-1],archive['s128_start_pair'][-1])
    assert len(result['selected_steps'])==8


def test_branch_never_ages_ordinary_clock_and_all_costs_are_compared(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    result=check.run_case(check.CASES[0],check._Reports())
    assert result['passed'] and result['length_counts']=={4:146}
    assert result['branch']['anchor_frame']==576 and result['final_delivered_frame']==584
    assert result['accounting']['charged_frames']==132+584+136
    assert result['accounting']['restores']==2
    with np.load(tmp_path/result['selected_trace_archive'],allow_pickle=False) as archive:
        assert len(archive.files)==39 and archive['branch_start_pair'].shape==(2,210,160,3)
        for name in ('intervention','control'):
            assert archive[f'branch_{name}_raw'].shape==(8,210,160,3)
            np.testing.assert_array_equal(archive[f'branch_{name}_endpoint'],archive[f'branch_{name}_pooled'][-1])


def test_second_constructor_failure_records_first_known_cost_and_unknown_peer(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    def fail(_):raise RuntimeError('synthetic worker constructor loss')
    monkeypatch.setattr(check,'OrdinaryRGBPool',fail)
    with pytest.raises(RuntimeError,match='constructor loss'):check.run()
    attempt=json.loads(check.ATTEMPT.read_text())
    assert attempt['known_charged_frames']==132 and attempt['constructor_count']==2
    assert attempt['constructor_without_return']=='pong/worker' and not attempt['accounting_complete']
    assert attempt['status']=='failed' and attempt['no_retry']
    with pytest.raises(ValueError,match='no retry'):check.run()


def test_corrupted_middle_raw_frame_stops_and_preserves_both_paid_reports(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    original=SyntheticPool.step
    def corrupt(self,actions):
        trace=original(self,actions)[0]
        raw=trace.raw.copy();raw[1,0,0,0]^=1
        # Keep internal trace recurrence coherent, so the independent witness
        # must catch the corruption instead of the generic trace validator.
        pooled=[];previous=trace.start_pair[-1]
        for row in raw:pooled.append(np.maximum(previous,row));previous=row
        return [replace(trace,raw=raw,pooled=np.stack(pooled),endpoint=pooled[-1])]
    monkeypatch.setattr(SyntheticPool,'step',corrupt)
    with pytest.raises(ValueError,match='raw mismatch'):check.run()
    attempt=json.loads(check.ATTEMPT.read_text())
    assert attempt['known_charged_frames']==272 and attempt['accounting_complete']
    assert len(attempt['engine_reports'])==2 and attempt['completed_cases']==[]


def test_known_report_survives_peer_error_without_reading_first_again():
    reports=check._Reports();calls=[]
    def known():
        calls.append('known')
        if len(calls)>1:raise MemoryError('lost on second read')
        return dict(accounting={'charged_frames':852})
    def lost():raise MemoryError('peer report')
    for name,fn in [('known',known),('lost',lost)]:reports.begin(name);reports.register(name,fn)
    assert reports.read('known')['accounting']['charged_frames']==852
    result=reports.collect()
    assert result['known_charged_frames']==852 and not result['accounting_complete'] and calls==['known']
    assert reports.collect()==result and calls==['known']


def test_returned_but_incomplete_worker_report_never_becomes_complete_globally():
    reports=check._Reports();reports.begin('failed')
    reports.register('failed',lambda:dict(accounting={'charged_frames':132},failure={'accounting_complete':False}))
    result=reports.collect()
    assert result['known_charged_frames']==132 and result['uncertain_engine_reports']==['failed']
    assert result['accounting_complete'] is False


@pytest.mark.parametrize('operation',['step','branch_pair'])
def test_witness_native_elapsed_failure_keeps_known_lower_bound_and_forbids_reuse(operation):
    witness=synthetic(WITNESS_CLASS,check.config(check.CASES[0]))
    witness._budget.record_play(468,agent_steps=0)  # Admit the prospective branch before its first act.
    native=witness._ale.native;act=native.act
    def unexpected_duration(action):
        result=act(action);native.frame+=1
        return result
    native.act=unexpected_duration
    with pytest.raises(RuntimeError,match='one real ALE frame'):getattr(witness,operation)(1)
    report=witness.diagnostic_report()
    assert report['failure']['accounting_complete'] is False
    assert report['accounting']['charged_frames']==(600 if operation=='step' else 660)
    for command in (witness.step,witness.branch_pair):
        with pytest.raises(ValueError,match='stopped'):command(1)
    assert witness.diagnostic_report()==report


def test_missing_preflight_refuses_before_native_or_attempt(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    def fail():raise ValueError('no independent review')
    monkeypatch.setattr(check,'preflight',fail)
    monkeypatch.setattr(check,'_WitnessEngine',lambda _:pytest.fail('native before review'))
    with pytest.raises(ValueError,match='no independent'):check.run()
    assert not check.ATTEMPT.exists()
