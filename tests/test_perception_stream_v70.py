"""Assembly and failure-atomicity on synthetic RGB, without an emulator."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ragnarok.learning.parser_v70 import CounterfactualParser
from ragnarok.learning.perception_stream_v70 import PerceptionStream,transition_prediction
from ragnarok.learning import perception_stream_v70 as stream_module

torch.set_num_threads(1)


def stream(seed=708):
    torch.manual_seed(seed)
    model=CounterfactualParser().cpu()
    return PerceptionStream(model,(16,24),stream_id='synthetic-live0')


def rgb(offset=0):
    value=np.zeros((16,24,3),np.uint8)
    value[4:8,4+offset:7+offset]=[255,100,25]
    return value


def test_full_assembly_pools_actual_frames_and_backpropagates_without_training_weights():
    own=stream()
    parameters={name:p.detach().clone() for name,p in own.parser.named_parameters()}
    first=own.observe(rgb(),rgb(1),native_frame=0)
    later=own.observe(rgb(1),rgb(2),native_frame=4)
    expected=np.maximum(rgb(),rgb(1))
    np.testing.assert_array_equal(first.pooled_rgb[0].permute(1,2,0).numpy(),expected)
    prediction=transition_prediction(first,later)
    assert prediction.eligible and prediction.actual_frames==4
    (prediction.image_l1+prediction.foreground_bce+first.grouping.pull+first.grouping.separation).backward()
    assert own.parser.displacement[-1].weight.grad.abs().sum()>0
    assert own.parser.foreground.weight.grad.abs().sum()>0
    for name,p in own.parser.named_parameters():
        torch.testing.assert_close(p.detach(),parameters[name],rtol=0,atol=0)
        assert p.device.type=='cpu' and (p.grad is None or torch.isfinite(p.grad).all())


def test_identity_uses_exact_native_lag_and_retains_no_graph_or_return_alias():
    own=stream()
    first=own.observe(rgb(),rgb(),native_frame=0)
    original=own._history[0].pooled_rgb.clone()
    first.pooled_rgb.zero_()
    torch.testing.assert_close(own._history[0].pooled_rgb,original,rtol=0,atol=0)
    calls=[]
    original_forward=own.parser.forward
    def count(*args,**kwargs):
        calls.append(1)
        return original_forward(*args,**kwargs)
    own.parser.forward=count
    own.observe(rgb(),rgb(),native_frame=3)
    assert len(calls)==1  # Three real frames, no implicit one-step/four-frame clock.
    eighth=own.observe(rgb(),rgb(),native_frame=8)
    assert len(calls)==3  # Current plus exact t0 recomputation, not t3.
    assert eighth.identity.previous_candidates==len(own._history[0].tracking.observed)
    own.observe(rgb(),rgb(),native_frame=12)
    assert 0 not in own._history and set(own._history)=={8,12}
    assert all(v.pooled_rgb.dtype==torch.uint8 and not v.pooled_rgb.requires_grad for v in own._history.values())


def test_truncated_p_is_unknown_with_zero_gradient_and_stream_reset_is_rejected():
    own=stream()
    first=own.observe(rgb(),rgb(),native_frame=0)
    short=own.observe(rgb(),rgb(1),native_frame=2)
    first.output.displacement.retain_grad()
    loss=transition_prediction(first,short)
    assert not loss.eligible and loss.actual_frames==2
    (loss.image_l1+loss.foreground_bce).backward()
    assert torch.count_nonzero(first.output.displacement.grad)==0
    for wrong in (replace(short,stream_id='other'),replace(short,epoch=1),replace(short,native_frame=8)):
        with pytest.raises(ValueError): transition_prediction(first,wrong)


def test_late_grouping_failure_leaves_tracker_and_history_unchanged(monkeypatch):
    own=stream()
    initial=own.observe(rgb(),rgb(),native_frame=0)
    prior_tracker,prior_history=own._tracker,own._history
    original=stream_module.grouping_loss
    def fail(*args,**kwargs): raise MemoryError('synthetic late objective allocation')
    monkeypatch.setattr(stream_module,'grouping_loss',fail)
    with pytest.raises(MemoryError): own.observe(rgb(),rgb(),native_frame=4)
    assert own._tracker is prior_tracker and own._history is prior_history
    assert own._tracker._frame==0 and set(own._history)=={0}
    monkeypatch.setattr(stream_module,'grouping_loss',original)
    retried=own.observe(rgb(),rgb(),native_frame=4)
    assert [r.track_id for r in initial.tracking.observed]==[r.track_id for r in retried.tracking.observed]


def test_late_identity_failure_and_caller_pixel_mutation_cannot_corrupt_memory(monkeypatch):
    own=stream()
    pixels=rgb()
    initial=own.observe(pixels,pixels,native_frame=0)
    pixels[:]=0
    row=initial.tracking.observed[0]
    saved=own._history[0].tracking.observed[0].instance.source_pixels.copy()
    row.instance.source_pixels.flags.writeable=True
    row.instance.source_pixels[:]=0
    np.testing.assert_array_equal(own._history[0].tracking.observed[0].instance.source_pixels,saved)
    prior_tracker,prior_history=own._tracker,own._history
    def fail(*args,**kwargs): raise MemoryError('synthetic identity failure')
    monkeypatch.setattr(stream_module,'identity_loss',fail)
    with pytest.raises(MemoryError): own.observe(rgb(),rgb(),native_frame=8)
    assert own._tracker is prior_tracker and own._history is prior_history


def test_observation_clock_and_effect_domain_are_checked_without_committing():
    own=stream()
    own.observe(rgb(),rgb(),native_frame=0)
    with pytest.raises(ValueError): own.observe(rgb(),rgb(),native_frame=0)
    with pytest.raises(ValueError): own.observe(rgb(),rgb(),native_frame=4,effect=np.array([3]))
    with pytest.raises(ValueError): own.observe(rgb(),rgb(),native_frame=4,effect=np.full(12,2.,np.float64))
    assert own._tracker._frame==0 and set(own._history)=={0}


def test_extreme_finite_fields_cannot_make_unknown_p_nan_or_eligible_p_infinite():
    own=stream()
    first=own.observe(rgb(),rgb(),native_frame=0)
    later=own.observe(rgb(),rgb(1),native_frame=4)
    extreme=torch.finfo(torch.float32).max
    output=replace(first.output,
        displacement=torch.full_like(first.output.displacement,extreme,requires_grad=True),
        foreground_logits=torch.full_like(first.output.foreground_logits,extreme,requires_grad=True))
    source=replace(first,output=output)
    unknown=transition_prediction(source,replace(later,native_frame=2))
    assert not unknown.eligible and unknown.image_l1.item()==unknown.foreground_bce.item()==0
    (unknown.image_l1+unknown.foreground_bce).backward()
    assert torch.count_nonzero(output.displacement.grad)==torch.count_nonzero(output.foreground_logits.grad)==0
    with pytest.raises(ValueError,match='Nonfinite P loss'): transition_prediction(source,later)
