"""Ordered model contract before synthetic commands, plus owned P/C/G/I wiring."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import CounterfactualParser
from ragnarok.learning.owned_ordered_perception_v70 import (
    OwnedOrderedPixelObjectives,OwnedOrderedPerceptionSession)
from tests.test_owned_rgb_collector_v70 import SyntheticPool
from tests.test_owned_pixel_objectives_v70 import reset_engine

torch.set_num_threads(1)


def fixture(workers=1):
    pool=SyntheticPool(workers)
    for engine in pool.engines:reset_engine(engine)
    collector=OwnedRGBCollector(pool,workers,session_id='synthetic-ordered-owner')
    torch.manual_seed(764);parser=OrderedMotionParser().cpu()
    return collector,pool,parser


@pytest.mark.parametrize('owner',[OwnedOrderedPixelObjectives,OwnedOrderedPerceptionSession])
def test_wrong_model_rejected_before_any_paid_command(owner):
    collector,pool,_=fixture();wrong=CounterfactualParser().cpu()
    kwargs={'workers':1} if owner is OwnedOrderedPerceptionSession else {}
    with pytest.raises(ValueError,match='before commands'):owner(collector,wrong,**kwargs)
    assert pool.calls==[] and pool.report(0)['accounting']['charged_frames']==600
    collector.close()


def test_recorded_order_and_effect_reach_prediction_but_not_temporal_effect():
    collector,pool,parser=fixture();seen=[];forward=parser.forward
    def capture(previous,current,effect=None):
        seen.append((previous.detach().clone(),current.detach().clone(),None if effect is None else effect.detach().clone()))
        return forward(previous,current,effect)
    parser.forward=capture
    own=OwnedOrderedPerceptionSession(collector,parser,workers=1)
    try:
        effect=np.zeros(12,np.float32);effect[2]=.75
        batch=own.step([1],effects=[effect]);prediction=batch.prediction[0]
        pair=prediction.record.trace.start_pair
        np.testing.assert_array_equal(seen[1][0][0].permute(1,2,0).numpy(),pair[0])
        np.testing.assert_array_equal(seen[1][1][0].permute(1,2,0).numpy(),pair[1])
        assert seen[1][2][0,2]==.75 and batch.observations[0].effect.sum()==0
        later=own.step([0]);assert later.observations[0].native_frame==8
        assert set(own._streams[0]._history)=={0,4,8}
        assert pool.calls==[('step',(1,)),('step',(0,))]
    finally:own.close()


def test_counterfactual_pair_keeps_live_memory_and_exact_costs():
    collector,pool,parser=fixture();own=OwnedOrderedPerceptionSession(collector,parser,workers=1)
    try:
        own.step([1]);anchor=own.current_anchor();history=own._streams[0]._history
        # This synthetic controller exposes only buttons0/1.
        branch=own.branch(1,expected_anchor=anchor.token,frames=8)
        assert len(branch.record.intervention.raw)==len(branch.record.control.raw)==8
        assert own._streams[0]._history is history and own.current_anchor().frame==4
        assert pool.report(0)['accounting']['charged_frames']==740
        assert torch.isfinite(branch.loss)
    finally:own.close()


def test_late_nonfinite_head_retains_paid_record_and_blocks_further_commands():
    collector,pool,parser=fixture();own=OwnedOrderedPerceptionSession(collector,parser,workers=1)
    before=own._streams;forward=parser.forward
    def invalid(*args,**kwargs):
        out=forward(*args,**kwargs)
        return replace(out,solidity_logits=torch.full_like(out.solidity_logits,float('nan')))
    parser.forward=invalid
    try:
        with pytest.raises(ValueError,match='Nonfinite'):own.step([1])
        retained=own.failure_records()
        assert len(retained)==1 and len(retained[0].trace.raw)==4
        assert own._streams is before and pool.report(0)['accounting']['charged_frames']==604
        with pytest.raises(ValueError,match='stopped'):own.step([0])
        assert pool.calls==[('step',(1,))]
    finally:own.close()
