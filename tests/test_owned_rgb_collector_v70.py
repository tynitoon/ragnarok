"""Causal ownership/recording with synthetic engines and no learner or ROM."""
from dataclasses import replace
from threading import Event,Thread

import numpy as np
import pytest

from ragnarok.environments import owned_rgb_collector_v70 as collecting
from tests.test_ordinary_rgb_trace_v70 import subject


class SyntheticPool:
    def __init__(self,workers=1):
        self.engines=[subject() for _ in range(workers)]
        self.calls=[]
        self.closed=False
    def observe_pair(self,index=0):return self.engines[index].observe_pair()
    def step(self,actions):
        self.calls.append(('step',tuple(actions)))
        return [engine.step_trace(action) for engine,action in zip(self.engines,actions,strict=True)]
    def branch_pair_trace(self,action,*,control,frames,index):
        self.calls.append(('branch',index,action,control,frames))
        return self.engines[index].branch_pair_trace(action,control,frames)
    def report(self,index):return dict(accounting=self.engines[index]._budget.report())
    def close(self):self.closed=True


def collector(workers=1,**kwargs):
    pool=SyntheticPool(workers)
    return collecting.OwnedRGBCollector(pool,workers,session_id='synthetic-session',**kwargs),pool


def test_one_batch_records_exact_actions_effect_assumptions_and_cannot_be_taken_twice():
    own,pool=collector(2)
    effect=np.zeros(12,np.float32);effect[3]=1
    operation=own.step([1,0],effects=[effect,None])
    effect[:]=0
    result=own.take(operation)
    assert len(result)==2 and [r.requested_action for r in result]==[1,0]
    assert result[0].effect_assumption[3]==1 and result[1].effect_assumption.sum()==0
    assert all(r.chronology.end_frame==4 for r in result)
    assert own.current_anchor(0).token!=own.current_anchor(1).token
    with pytest.raises(ValueError,match='already consumed'):own.take(operation)
    assert pool.calls==[('step',(1,0))]


def test_branch_is_bound_to_one_current_anchor_and_does_not_advance_live_clock():
    own,pool=collector()
    anchor=own.current_anchor()
    operation=own.branch(1,expected_anchor=anchor.token,frames=8)
    record,=own.take(operation)
    assert record.anchor.token==anchor.token and record.requested_frames==8 and record.control_action==0
    assert len(record.intervention.raw)==len(record.control.raw)==8
    assert own.current_anchor().token==anchor.token and own.current_anchor().frame==0
    assert pool.report(0)['accounting']['charged_frames']==736
    own.step([0])
    calls=list(pool.calls)
    with pytest.raises(ValueError,match='Stale'):own.branch(1,expected_anchor=anchor.token)
    assert pool.calls==calls


def test_pending_capacity_prevents_spending_and_take_allocation_failure_keeps_record(monkeypatch):
    own,pool=collector(max_pending=1)
    operation=own.step([1])
    before=pool.report(0)
    with pytest.raises(ValueError,match='capacity'):own.step([0])
    assert pool.report(0)==before
    original=collecting.copy.deepcopy
    def fail(value):raise MemoryError('synthetic consumption copy')
    monkeypatch.setattr(collecting.copy,'deepcopy',fail)
    with pytest.raises(MemoryError):own.take(operation)
    monkeypatch.setattr(collecting.copy,'deepcopy',original)
    assert len(own.take(operation))==1
    assert own.step([0])!=operation


def test_mutating_public_anchor_or_taken_record_cannot_rewrite_owned_current_pixels():
    own,pool=collector()
    anchor=own.current_anchor()
    anchor.pair[:]=255
    assert np.array_equal(own.current_anchor().pair,pool.observe_pair())
    operation=own.step([1]);record,=own.take(operation)
    record.trace.raw[:]=255
    assert np.array_equal(own.current_anchor().pair,pool.observe_pair())


def test_malformed_paid_batch_stops_without_rewinding_charges_or_committing_partial_anchors():
    own,pool=collector(2)
    previous=[own.current_anchor(i) for i in range(2)]
    original=pool.step
    def malformed(actions):return original(actions)[:1]
    pool.step=malformed
    with pytest.raises(ValueError,match='Incomplete'):own.step([1,0])
    assert all(pool.report(i)['accounting']['charged_frames']==604 for i in range(2))
    assert [a.token for a in own._anchors]==[a.token for a in previous]
    with pytest.raises(ValueError,match='stopped'):own.step([0,0])
    assert own.report()['collector_failure']['operation_id']==('synthetic-session',1)


def test_stale_external_pixel_mutation_is_detected_before_branch_but_reports_remain():
    own,pool=collector()
    anchor=own.current_anchor()
    pool.engines[0]._frames[-1][:]=255  # Violation of exclusive pool ownership.
    with pytest.raises(ValueError,match='outside'):own.branch(1,expected_anchor=anchor.token)
    assert pool.calls==[] and own.report()['worker']['accounting']['charged_frames']==600


def test_bad_branch_prefix_after_payment_stops_and_retains_previous_success_for_take():
    own,pool=collector()
    previous=own.step([1])
    anchor=own.current_anchor()
    original=pool.branch_pair_trace
    def malformed(*args,**kwargs):
        left,right=original(*args,**kwargs)
        return replace(left,pooled=np.zeros_like(left.pooled)),right
    pool.branch_pair_trace=malformed
    with pytest.raises(ValueError,match='history'):own.branch(1,expected_anchor=anchor.token)
    assert own.report()['worker']['accounting']['charged_frames']==732
    assert len(own.take(previous))==1
    with pytest.raises(ValueError,match='stopped'):own.branch(1,expected_anchor=anchor.token)


def test_invalid_effect_is_refused_before_any_command_and_owned_close_forwards():
    own,pool=collector()
    with pytest.raises(ValueError,match='effect'):own.step([1],effects=[np.full(12,np.inf)])
    assert pool.calls==[]
    own.close()
    assert pool.closed


def test_even_error_description_allocation_failure_cannot_reopen_paid_commands(monkeypatch):
    own,pool=collector()
    original=pool.step
    def malformed(actions):
        original(actions)
        return []
    pool.step=malformed
    def fail(*args,**kwargs):raise MemoryError('synthetic error description allocation')
    monkeypatch.setattr(collecting,'dict',fail,raising=False)
    with pytest.raises(MemoryError):own.step([1])
    assert own._stopped and own._failed_operation==('synthetic-session',1)
    with pytest.raises(ValueError,match='stopped'):own.step([1])
    monkeypatch.delattr(collecting,'dict')
    report=own.report()
    assert report['collector_stopped'] and report['collector_failure'] is None
    assert report['worker']['accounting']['charged_frames']==604


def test_concurrent_calls_cannot_invert_paid_replies_or_read_partial_state():
    own,pool=collector()
    entered,release=Event(),Event()
    original=pool.step
    def held(actions):
        result=original(actions)
        entered.set()
        assert release.wait(5)
        return result
    pool.step=held
    results=[];errors=[]
    def run():
        try:results.append(own.step([1]))
        except BaseException as error:errors.append(error)
    worker=Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        for call in (lambda:own.step([0]),lambda:own.branch(1,expected_anchor=('synthetic-session',0,0)),
                     lambda:own.take(('synthetic-session',1)),own.current_anchor,own.report,own.close):
            with pytest.raises(ValueError,match='in progress'):call()
        assert pool.calls==[('step',(1,))]
    finally:
        release.set();worker.join(5)
    assert not worker.is_alive() and not errors and len(results)==1
    record,=own.take(results[0])
    assert record.chronology.epoch==0 and record.chronology.end_frame==4
    np.testing.assert_array_equal(own.current_anchor().pair,pool.observe_pair())
    assert own.step([0])==('synthetic-session',2)
    assert own.current_anchor().frame==8 and own.current_anchor().epoch==0


def test_reentrant_callback_is_rejected_without_deadlock_or_second_charge():
    own,pool=collector()
    original=pool.step
    def callback(actions):
        with pytest.raises(ValueError,match='in progress'):own.step([0])
        return original(actions)
    pool.step=callback
    operation=own.step([1])
    assert len(own.take(operation))==1
    assert pool.calls==[('step',(1,))] and own.report()['worker']['accounting']['charged_frames']==604
