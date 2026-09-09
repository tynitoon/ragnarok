"""Synthetic paid-operation to own P/C graph integration; no ROM or GPU."""
from collections import deque
from dataclasses import replace
from threading import Event,Thread

import numpy as np
import pytest
import torch

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.learning.parser_v70 import CounterfactualParser,ParserOutput
from ragnarok.learning import owned_pixel_objectives_v70 as objectives
from ragnarok.learning.pixel_prediction_v70 import prediction_terms
from ragnarok.learning.counterfactual_targets_v70 import component_targets
from tests.test_owned_rgb_collector_v70 import SyntheticPool
from tests.test_rgb_branch_trace_v70 import PixelNative

torch.set_num_threads(1)


class SmallNative(PixelNative):
    def getScreenRGB(self):
        image=np.zeros((16,24,3),np.uint8)
        image[4:8,self.frame%20:(self.frame%20)+3]=self.value
        return image


def reset_engine(engine,terminal=999):
    engine._ale=SmallNative(terminal)
    engine._frames=deque([engine._ale.getScreenRGB(),engine._ale.getScreenRGB()],maxlen=2)
    engine._last_action=0


class ScriptedParser(CounterfactualParser):
    """An own synthetic prediction, deliberately independent of native truth."""
    def __init__(self):
        super().__init__()
        self.scalar=torch.nn.Parameter(torch.tensor(.4))
        self.seen=[]
    def forward(self,rgb,effect=None):
        self.seen.append((rgb.detach().clone(),effect.detach().clone()))
        batch,_,height,width=rgb.shape
        base=self.scalar.expand(batch,1,height,width)
        embedding=torch.cat((base*0+1,(base*0).expand(batch,31,height,width)),1)
        displacement=torch.cat((base*0+effect[:,0,None,None,None]+1,base*0),1)
        return ParserOutput(embedding,base+1,displacement,base,base*0)


def owner(workers=1,parser=None,terminal=999,credit=600):
    pool=SyntheticPool(workers)
    for engine in pool.engines:
        reset_engine(engine,terminal)
        if credit>600:engine._budget.record_play(credit-600,agent_steps=0)
    collector=OwnedRGBCollector(pool,workers,session_id='synthetic-objectives')
    parser=ScriptedParser() if parser is None else parser
    return objectives.OwnedPixelObjectives(collector,parser),pool,parser


def test_real_parser_receives_exact_source_effect_and_p_backpropagates_without_weight_updates():
    torch.manual_seed(714)
    own,pool,parser=owner(parser=CounterfactualParser().cpu())
    before={name:p.detach().clone() for name,p in parser.named_parameters()}
    effect=np.zeros(12,np.float32);effect[3]=1
    result,=own.step([1],effects=[effect])
    assert result.eligible and result.record.chronology.end_frame==4
    np.testing.assert_array_equal(result.source_rgb[0].permute(1,2,0).numpy(),
                                  np.maximum(*result.record.trace.start_pair))
    torch.testing.assert_close(result.effect,torch.tensor(effect)[None])
    endpoint=torch.tensor(result.record.trace.endpoint).permute(2,0,1)[None]
    direct=prediction_terms(result.source_rgb,endpoint,result.output)
    torch.testing.assert_close(result.image_l1,direct['image_l1'],rtol=0,atol=0)
    torch.testing.assert_close(result.foreground_bce,direct['foreground_bce'],rtol=0,atol=0)
    (result.image_l1+result.foreground_bce).backward()
    # A black source has zero trunk features: the head weight gradient is
    # exactly zero, while its bias still receives genuine BCE supervision.
    assert parser.foreground.bias.grad.abs().sum()>0
    assert parser.foreground.weight.grad.abs().sum()==0
    for name,p in parser.named_parameters():
        torch.testing.assert_close(p.detach(),before[name],rtol=0,atol=0)
        assert p.device.type=='cpu' and (p.grad is None or torch.isfinite(p.grad).all())
    with pytest.raises(ValueError,match='already consumed'):own._collector.take(result.record.operation_id)
    assert pool.calls==[('step',(1,))]


@pytest.mark.parametrize('terminal',[0,1,2,3])
def test_actual_prefix_zero_to_three_remains_unknown_with_finite_zero_gradient(terminal):
    own,pool,parser=owner(terminal=terminal)
    if terminal==0:
        # A cold construction may itself leave a no-act prefix. Account it,
        # without handing its hidden native terminal annotation to the learner.
        def cold(*,experimental):
            pool.engines[0]._budget.record_play(132,agent_steps=0)
            reset_engine(pool.engines[0],terminal=0)
        pool.engines[0]._cold_start=cold
    result,=own.step([1])
    assert not result.eligible and len(result.record.trace.raw)==terminal
    assert result.record.chronology.end_frame==terminal
    assert result.reason==('no_delivered_action_image' if terminal==0 else 'truncated_prediction_horizon')
    (result.image_l1+result.foreground_bce).backward()
    assert parser.scalar.grad.item()==0
    assert pool.report(0)['accounting']['charged_frames']==600+terminal+(132 if terminal==0 else 0)


def test_cold_departure_is_actual_retrospective_source_and_effect_cannot_change_during_payment():
    own,pool,parser=owner(terminal=1)
    own.step([1])
    previous=own.current_anchor().pair.copy()
    def cold(*,experimental):
        pool.engines[0]._budget.record_play(132,agent_steps=0)
        reset_engine(pool.engines[0])
    pool.engines[0]._cold_start=cold
    effect=np.zeros(12,np.float32);effect[0]=.75
    original=pool.step
    def during(actions):
        effect[:]=0
        return original(actions)
    pool.step=during
    result,=own.step([0],effects=[effect])
    assert result.record.chronology.visual_discontinuity and result.record.chronology.epoch==1
    assert result.record.chronology.start_frame==0 and result.record.chronology.end_frame==4
    assert not np.array_equal(previous,result.record.trace.start_pair)
    assert result.effect[0,0].item()==.75 and parser.seen[-1][1][0,0].item()==.75
    assert torch.count_nonzero(result.source_rgb)==0
    assert pool.report(0)['accounting']['charged_frames']==737


def test_branch_uses_one_paid_anchor_effect_and_exact_fourth_pooled_frame_of_longer_trace(monkeypatch):
    own,pool,parser=owner()
    anchor=own.current_anchor()
    effect=np.zeros(12,np.float32);effect[0]=.5
    seen=[]
    original=objectives.component_targets
    def capture(instances,displacement,left,right,**kwargs):
        seen.append((instances,displacement.copy(),left,right,kwargs))
        return original(instances,displacement,left,right,**kwargs)
    monkeypatch.setattr(objectives,'component_targets',capture)
    result=own.branch(1,expected_anchor=anchor.token,frames=8,effect=effect)
    instances,flow,left,right,kwargs=seen[0]
    assert kwargs==dict(horizon_frames=4,image_kind='pooled') and np.all(flow[0]==1.5)
    short=lambda t:replace(t,raw=t.raw[:4],pooled=t.pooled[:4],endpoint=t.pooled[3])
    direct=component_targets(instances,flow,short(left),short(right),**kwargs)
    np.testing.assert_array_equal(result.targets.values,direct.values)
    np.testing.assert_array_equal(result.targets.weights,direct.weights)
    assert result.targets.weights.sum()>0
    result.loss.backward()
    assert torch.isfinite(parser.scalar.grad) and parser.scalar.grad.abs()>0
    assert own.current_anchor().token==anchor.token and own.current_anchor().frame==0
    assert pool.calls==[('branch',0,1,0,8)]
    assert pool.report(0)['accounting']['charged_frames']==736


def test_short_branch_unknown_and_mutating_previous_result_cannot_supply_a_new_source():
    own,pool,parser=owner(credit=1200)
    result=own.branch(1,expected_anchor=own.current_anchor().token,frames=2)
    assert np.count_nonzero(result.targets.weights)==0 and result.loss.item()==0
    result.loss.backward()
    assert parser.scalar.grad.item()==0
    result.record.anchor.pair[:]=255
    result.source_rgb.fill_(255)
    next_result=own.branch(1,expected_anchor=own.current_anchor().token)
    assert torch.count_nonzero(next_result.source_rgb)==0


def test_postpayment_parser_failure_keeps_entire_batch_and_never_reissues_commands(monkeypatch):
    own,pool,parser=owner(workers=2)
    original=parser.forward
    def fail_second(*args):
        if len(parser.seen)==1:raise MemoryError('synthetic second-worker inference')
        return original(*args)
    monkeypatch.setattr(parser,'forward',fail_second)
    with pytest.raises(MemoryError):own.step([1,0])
    records=own.failure_records()
    assert len(records)==2 and own.report()['records_retained']
    records[0].trace.raw[:]=255
    assert not np.array_equal(own.failure_records()[0].trace.raw,records[0].trace.raw)
    with pytest.raises(ValueError,match='stopped'):own.step([1,0])
    assert pool.calls==[('step',(1,0))]
    assert all(pool.report(i)['accounting']['charged_frames']==604 for i in range(2))


def test_extreme_finite_p_loss_is_refused_and_nonfinite_unused_head_is_also_refused(monkeypatch):
    own,pool,parser=owner()
    original=parser.forward
    def extreme(*args):
        output=original(*args)
        return replace(output,foreground_logits=output.foreground_logits*0+torch.finfo(torch.float32).max)
    monkeypatch.setattr(parser,'forward',extreme)
    with pytest.raises(ValueError,match='Nonfinite P loss'):own.step([1])
    assert len(own.failure_records())==1 and pool.report(0)['accounting']['charged_frames']==604
    own,pool,parser=owner()
    original=parser.forward
    def bad_unused(*args):
        output=original(*args)
        return replace(output,solidity_logits=output.solidity_logits*float('nan'))
    monkeypatch.setattr(parser,'forward',bad_unused)
    with pytest.raises(ValueError,match='parser output'):own.step([1])
    assert own.report()['objective_owner_stopped']


def test_take_failure_preserves_pending_paid_data_and_invalid_command_never_spends(monkeypatch):
    own,pool,parser=owner()
    original=own._collector.take
    def fail(*args):raise MemoryError('synthetic take')
    monkeypatch.setattr(own._collector,'take',fail)
    with pytest.raises(MemoryError):own.step([1])
    report=own.report()
    assert report['objective_owner_stopped'] and not report['records_retained']
    assert len(original(report['processing_operation']))==1
    own,pool,parser=owner()
    with pytest.raises(ValueError):own.step([18])
    assert pool.calls==[] and own.report()['objective_owner_stopped']
    own.close()
    assert pool.closed


def test_owner_lock_spans_inference_after_collector_released_its_own_lock(monkeypatch):
    own,pool,parser=owner()
    entered,release=Event(),Event()
    original=parser.forward
    def held(*args):
        entered.set()
        assert release.wait(5)
        # Same-thread reentrancy is refused too, without stopping this owner.
        with pytest.raises(ValueError,match='in progress'):own.step([0])
        return original(*args)
    monkeypatch.setattr(parser,'forward',held)
    results=[];errors=[]
    def run():
        try:results.append(own.step([1]))
        except BaseException as error:errors.append(error)
    thread=Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        for call in (lambda:own.step([0]),own.current_anchor,own.report,own.failure_records,own.close):
            with pytest.raises(ValueError,match='in progress'):call()
        assert pool.calls==[('step',(1,))]
    finally:
        release.set();thread.join(5)
    assert not thread.is_alive() and not errors and len(results)==1
    assert not own.report()['objective_owner_stopped']
    assert own.current_anchor().frame==4
