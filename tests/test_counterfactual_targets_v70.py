"""Synthetic paid-trace-shaped inputs and own predicted positions, no Atari."""
import numpy as np
import pytest
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.learning.visual_instances_v70 import extract_instances
from ragnarok.learning.counterfactual_targets_v70 import component_targets,controllability_loss
from tests.test_visual_instances_v70 import scene


def trace(count=4,changed=None):
    frames=np.zeros((count,10,12,3),dtype=np.uint8)
    if changed is not None and count:
        y,x=changed
        frames[-1,y,x]=255
    endpoint=frames[-1].copy() if count else np.zeros((10,12,3),dtype=np.uint8)
    return RGBBranchTrace(frames,frames.copy(),endpoint)


def objects():
    prediction,rgb=scene()
    prediction.foreground_logits[:,3,2]=10
    prediction.displacement[0,3,2]=6
    return extract_instances(prediction,rgb),prediction.displacement


def test_changed_pixel_is_matched_at_predicted_future_position_not_old_position():
    own,flow=objects()
    result=component_targets(own,flow,trace(changed=(3,8)),trace(),horizon_frames=4,image_kind='pooled')
    assert result.values[3,2]==1 and result.weights[3,2]==1
    assert result.weights.sum()==1 and result.instances[0]['eligible']
    old=component_targets(own,flow,trace(changed=(3,2)),trace(),horizon_frames=4,image_kind='pooled')
    assert old.values[3,2]==0 and old.weights[3,2]==1


@pytest.mark.parametrize('left,right',[(0,4),(1,1),(3,4)])
def test_truncated_horizon_is_unknown_instead_of_negative(left,right):
    own,flow=objects()
    result=component_targets(own,flow,trace(left),trace(right),horizon_frames=4,image_kind='raw')
    assert result.weights.sum()==0 and result.instances[0]['target'] is None
    assert result.instances[0]['reason']=='incomplete_common_horizon'


def test_absent_predicted_motion_and_outside_support_are_not_supervised():
    own,flow=objects()
    stationary=component_targets(own,np.zeros_like(flow),trace(),trace(),horizon_frames=4,image_kind='pooled')
    outside=component_targets(own,np.full_like(flow,np.finfo(np.float32).max),trace(),trace(),horizon_frames=4,image_kind='pooled')
    assert stationary.weights.sum()==outside.weights.sum()==0
    assert stationary.instances[0]['reason']=='no_own_predicted_motion'
    assert outside.instances[0]['reason']=='predicted_support_outside_frame'


def test_fractional_projection_uses_bilinear_evidence_and_prefix_is_explicit():
    own,flow=objects()
    flow[0,3,2]=5.5
    result=component_targets(own,flow,trace(changed=(3,8)),trace(),horizon_frames=4,image_kind='pooled')
    assert result.values[3,2]==.5
    assert result.horizon_frames==result.common_prefix_frames==4
    assert not result.values.flags.writeable


def test_unknown_targets_have_zero_gradient_but_positive_own_target_trains_c_head():
    torch=pytest.importorskip('torch')
    own,flow=objects()
    target=component_targets(own,flow,trace(changed=(3,8)),trace(),horizon_frames=4,image_kind='pooled')
    logits=torch.zeros((10,12),requires_grad=True,device='cpu')
    loss=controllability_loss(logits,target)
    loss.backward()
    assert logits.grad[3,2]<0 and torch.count_nonzero(logits.grad)==1
    unknown=component_targets(own,flow,trace(1),trace(1),horizon_frames=4,image_kind='pooled')
    logits.grad.zero_()
    assert controllability_loss(logits,unknown).item()==0
    controllability_loss(logits,unknown).backward()
    assert torch.count_nonzero(logits.grad)==0
