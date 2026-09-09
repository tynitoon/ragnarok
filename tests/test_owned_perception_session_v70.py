"""Delivered-image chronology and atomic G/I intake, all synthetic CPU."""
from dataclasses import replace
from threading import Event,Thread

import numpy as np
import pytest
import torch

from ragnarok.learning import owned_perception_session_v70 as temporal
from tests.test_owned_pixel_objectives_v70 import owner as objective_owner,ScriptedParser,reset_engine

torch.set_num_threads(1)


class IslandParser(ScriptedParser):
    """Two synthetic own visual components, with no relation to game labels."""
    def forward(self,rgb,effect=None):
        if effect is None:effect=torch.zeros((len(rgb),12),device=rgb.device)
        output=super().forward(rgb,effect)
        _,_,height,width=rgb.shape
        left=torch.zeros((height,width),dtype=torch.bool,device=rgb.device)
        right=left.clone()
        left[3:6,3:6]=True;right[9:12,15:18]=True
        foreground=torch.where(left|right,self.scalar+1,self.scalar*0-2)[None,None]
        first=torch.where(left,self.scalar*0+1,self.scalar)
        second=torch.where(left,self.scalar,self.scalar*0+1)
        rest=(self.scalar*0).expand(30,height,width)
        embedding=torch.nn.functional.normalize(torch.cat((first[None],second[None],rest),0),dim=0)[None]
        return replace(output,foreground_logits=foreground,embedding=embedding)


def session(workers=1,terminal=999,credit=600):
    unused,pool,parser=objective_owner(workers,parser=IslandParser(),terminal=terminal,credit=credit)
    own=temporal.OwnedPerceptionSession(unused._collector,parser,workers=workers)
    return own,pool,parser


def test_two_complete_steps_use_exact_eight_delivered_images_and_current_weight_identity():
    own,pool,parser=session()
    first=own.step([1],effects=[np.ones(12,np.float32)])
    frame=first.observations[0]
    assert frame.native_frame==4 and frame.epoch==0 and frame.effect.sum().item()==0
    assert first.prediction[0].effect.sum().item()==12  # P uses recorded action hypothesis.
    second=own.step([0])
    later=second.observations[0]
    assert later.native_frame==8 and later.identity.supervised_anchors==4
    assert later.identity.previous_candidates==later.identity.current_candidates==2
    later.identity.loss.backward()
    assert torch.isfinite(parser.scalar.grad) and parser.scalar.grad.abs()>0
    history=own._streams[0]._history
    assert set(history)=={0,4,8}
    assert all(v.pooled_rgb.dtype==torch.uint8 and not v.pooled_rgb.requires_grad for v in history.values())
    assert pool.report(0)['accounting']['charged_frames']==608
    before=history[8].tracking.observed[0].instance.source_pixels.copy()
    later.tracking.observed[0].instance.source_pixels.flags.writeable=True
    later.tracking.observed[0].instance.source_pixels[:]=0
    np.testing.assert_array_equal(history[8].tracking.observed[0].instance.source_pixels,before)


def test_irregular_prefixes_are_not_implicitly_two_calls_per_identity_interval():
    own,pool,parser=session()
    original=pool.step
    lengths=iter([1,3,4])
    def prefixes(actions):
        # Synthetic native stop boundary is set by the pool fixture before
        # its advance; no such annotation is exposed to the collector/model.
        pool.engines[0]._ale.terminal=pool.engines[0]._ale.frame+next(lengths)
        return original(actions)
    pool.step=prefixes
    first=own.step([1]);second=own.step([0]);third=own.step([0])
    assert [b.observations[0].native_frame for b in (first,second,third)]==[1,4,8]
    assert first.observations[0].identity.supervised_anchors==0
    assert second.observations[0].identity.supervised_anchors==0
    assert third.observations[0].identity.supervised_anchors==4
    assert [b.prediction[0].eligible for b in (first,second,third)]==[False,False,True]
    assert own._streams[0].epoch==0


def test_zero_prefix_with_identical_start_has_no_new_temporal_observation_or_graph():
    own,pool,parser=session(terminal=0)
    def cold(*,experimental):
        pool.engines[0]._budget.record_play(132,agent_steps=0)
        reset_engine(pool.engines[0],terminal=0)
    pool.engines[0]._cold_start=cold
    prior_history=own._streams[0]._history
    first=own.step([0]);second=own.step([0])
    assert first.observations==second.observations==(None,)
    assert not first.prediction[0].eligible and not second.prediction[0].eligible
    assert own._streams[0]._history is prior_history and set(prior_history)=={0}
    assert own.current_anchor().frame==0 and own.current_anchor().epoch==0
    assert first.operation_id!=second.operation_id
    assert pool.report(0)['accounting']['charged_frames']==864


@pytest.mark.parametrize('new_terminal',[0,999])
def test_visual_cold_departure_starts_fresh_epoch_without_old_identity_or_bootstrap_clock(new_terminal):
    own,pool,parser=session(terminal=1)
    own.step([1])
    old=own._streams[0]
    def cold(*,experimental):
        pool.engines[0]._budget.record_play(132,agent_steps=0)
        reset_engine(pool.engines[0],terminal=new_terminal)
    pool.engines[0]._cold_start=cold
    result=own.step([0])
    frame=result.observations[0]
    assert frame.epoch==1 and frame.native_frame==(0 if new_terminal==0 else 4)
    assert frame.identity.supervised_anchors==0
    assert own._streams[0] is not old and old.epoch==0 and old._tracker._frame==1
    assert all(v.epoch==1 for v in own._streams[0]._history.values())
    assert pool.report(0)['accounting']['charged_frames']==733+(0 if new_terminal==0 else 4)


def test_branch_pair_does_not_age_ordinary_tracker_or_identity_memory():
    own,pool,parser=session(credit=1200)
    own.step([1])
    before=own._streams[0]
    history=before._history
    result=own.branch(1,expected_anchor=own.current_anchor().token,frames=8)
    assert result.targets.horizon_frames==4
    assert own._streams[0] is before and before._history is history and before._tracker._frame==4
    assert own.current_anchor().frame==4 and own.report()['temporal_owner_stopped'] is False


def test_late_second_worker_failure_keeps_all_previous_memories_and_paid_records(monkeypatch):
    own,pool,parser=session(workers=2)
    previous=list(own._streams)
    histories=[s._history for s in previous]
    original=temporal.PerceptionStream.observe
    def fail_second(self,*args,**kwargs):
        if self.stream_id.endswith('/1') and kwargs['native_frame']==4:
            raise MemoryError('synthetic second-worker temporal allocation')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(temporal.PerceptionStream,'observe',fail_second)
    with pytest.raises(MemoryError):own.step([1,0])
    assert all(a is b for a,b in zip(own._streams,previous,strict=True))
    assert all(s._history is h and s._tracker._frame==0 for s,h in zip(previous,histories,strict=True))
    records=own.failure_records()
    assert len(records)==2 and [r.worker for r in records]==[0,1]
    assert own.report()['temporal_batch_retained']
    with pytest.raises(ValueError,match='stopped'):own.step([1,0])
    assert pool.calls==[('step',(1,0))]
    assert all(pool.report(i)['accounting']['charged_frames']==604 for i in range(2))


def test_final_batch_allocation_failure_cannot_commit_any_memory(monkeypatch):
    own,pool,parser=session()
    previous=own._streams
    def fail(*args):raise MemoryError('synthetic final batch')
    monkeypatch.setattr(temporal,'OwnedPerceptionBatch',fail)
    with pytest.raises(MemoryError):own.step([1])
    assert own._streams is previous and previous[0]._tracker._frame==0
    assert len(own.failure_records())==1 and own.report()['temporal_owner_stopped']


def test_temporal_owner_lock_spans_its_post_objective_work(monkeypatch):
    own,pool,parser=session()
    entered,release=Event(),Event()
    original=temporal.PerceptionStream.observe
    def held(self,*args,**kwargs):
        entered.set()
        assert release.wait(5)
        with pytest.raises(ValueError,match='in progress'):own.current_anchor()
        return original(self,*args,**kwargs)
    monkeypatch.setattr(temporal.PerceptionStream,'observe',held)
    results=[];errors=[]
    def run():
        try:results.append(own.step([1]))
        except BaseException as error:errors.append(error)
    thread=Thread(target=run);thread.start()
    try:
        assert entered.wait(5)
        for call in (lambda:own.step([0]),own.current_anchor,own.report,own.failure_records,own.close):
            with pytest.raises(ValueError,match='in progress'):call()
        assert pool.calls==[('step',(1,))]
    finally:
        release.set();thread.join(5)
    assert not thread.is_alive() and not errors and len(results)==1
    assert own._streams[0]._tracker._frame==4 and not own.report()['temporal_owner_stopped']
