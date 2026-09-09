"""Synthetic ordered temporal intake, without native images or optimization."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ragnarok.learning import ordered_perception_stream_v70 as module
from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import CounterfactualParser
from ragnarok.learning.perception_stream_v70 import transition_prediction
from ragnarok.learning.gradient_guard_v70 import admit_adam_gradients

torch.set_num_threads(1)


def rgb(x):
    value=np.zeros((16,24,3),np.uint8);value[4:6,x:x+2]=255
    return value


def stream(seed=764):
    torch.manual_seed(seed)
    return module.OrderedPerceptionStream(OrderedMotionParser().cpu(),(16,24),stream_id='synthetic-ordered')


def test_constructor_refuses_pooled_parser_before_any_inference():
    old=CounterfactualParser().cpu()
    old.forward=lambda *args:pytest.fail('Constructor must not infer with a rejected model')
    with pytest.raises(ValueError,match='explicit ordered-motion'):
        module.OrderedPerceptionStream(old,(16,24),stream_id='wrong')


def test_reversing_actual_input_order_changes_only_motion_even_through_temporal_intake():
    left,right=stream(),stream()
    a=left.observe(rgb(4),rgb(5),native_frame=0)
    b=right.observe(rgb(5),rgb(4),native_frame=0)
    assert torch.equal(a.pooled_rgb,b.pooled_rgb)
    assert not torch.equal(a.output.displacement,b.output.displacement)
    for name in ('embedding','foreground_logits','controllability_logits','solidity_logits'):
        assert torch.equal(getattr(a.output,name),getattr(b.output,name))


def test_identity_recomputes_exact_eight_frame_pair_and_memory_owns_both_screens():
    own=stream();a,b=rgb(4),rgb(5)
    first=own.observe(a,b,native_frame=0)
    saved=(own._history[0].previous_rgb.clone(),own._history[0].current_rgb.clone())
    a[:]=0;b[:]=0;first.pooled_rgb.zero_()
    calls=[];original=own.parser.forward
    def recording(previous,current,effect=None):
        calls.append((previous.detach().clone(),current.detach().clone()))
        return original(previous,current,effect)
    own.parser.forward=recording
    own.observe(rgb(6),rgb(7),native_frame=3)
    assert len(calls)==1
    own.observe(rgb(8),rgb(9),native_frame=8)
    assert len(calls)==3 and all(torch.equal(x,y) for x,y in zip(calls[-1],saved))
    own.observe(rgb(9),rgb(10),native_frame=12)
    assert set(own._history)=={8,12}
    for item in own._history.values():
        assert item.previous_rgb.dtype==item.current_rgb.dtype==torch.uint8
        assert not item.previous_rgb.requires_grad and not item.current_rgb.requires_grad


def test_late_identity_failure_keeps_entire_prior_state(monkeypatch):
    own=stream();own.observe(rgb(4),rgb(5),native_frame=0)
    tracker,history=own._tracker,own._history
    def fail(*args,**kwargs):raise MemoryError('synthetic late I failure')
    monkeypatch.setattr(module,'identity_loss',fail)
    with pytest.raises(MemoryError):own.observe(rgb(8),rgb(9),native_frame=8)
    assert own._tracker is tracker and own._history is history and own._tracker._frame==0


def test_full_prediction_gradient_reaches_ordered_branch_with_weights_unchanged():
    own=stream();before={n:p.detach().clone() for n,p in own.parser.named_parameters()}
    first=own.observe(rgb(4),rgb(5),native_frame=0)
    later=own.observe(rgb(8),rgb(9),native_frame=4)
    loss=transition_prediction(first,later)
    (loss.image_l1+loss.foreground_bce).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in own.parser.ordered_motion.parameters())
    admit_adam_gradients(own.parser,max_norm=1.)
    assert all(torch.equal(before[n],p) for n,p in own.parser.named_parameters())
    own.parser.zero_grad(set_to_none=True)


def test_invalid_clock_effect_or_head_never_commits(monkeypatch):
    own=stream();own.observe(rgb(4),rgb(5),native_frame=0)
    tracker,history=own._tracker,own._history
    with pytest.raises(ValueError):own.observe(rgb(4),rgb(5),native_frame=0)
    with pytest.raises(ValueError):own.observe(rgb(4),rgb(5),native_frame=4,effect=np.array([5]))
    original=own.parser.forward
    def nonfinite(*args,**kwargs):
        out=original(*args,**kwargs)
        return replace(out,solidity_logits=torch.full_like(out.solidity_logits,float('nan')))
    monkeypatch.setattr(own.parser,'forward',nonfinite)
    with pytest.raises(ValueError,match='Every ordered parser head'):own.observe(rgb(4),rgb(5),native_frame=4)
    assert own._tracker is tracker and own._history is history
