"""Own synthetic motion/identity hypotheses; no emulator or calibration images."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ragnarok.learning.grouping_v70 import grouping_targets,grouping_loss
from ragnarok.learning.identity_v70 import IdentityObservation,identity_loss
from ragnarok.learning.instance_tracker_v70 import InstanceTracker
from tests.test_instance_tracker_v70 import observations

torch.set_num_threads(1)


def test_grouping_uses_motion_and_connectedness_without_appearance_targets():
    foreground=np.array([[1,1,1,-1,1]],dtype=np.float32)
    flow=np.zeros((2,1,5),dtype=np.float32)
    flow[0,0,2]=2
    target=grouping_targets(foreground,flow,prediction_horizon_frames=4)
    np.testing.assert_array_equal(target.labels,[[0,0,1,-1,2]])
    assert target.count==3 and not target.labels.flags.writeable
    assert target.prediction_horizon_frames==4


def test_local_agreement_can_chain_across_larger_motion_difference():
    foreground=np.ones((1,4),dtype=np.float32)
    flow=np.zeros((2,1,4),dtype=np.float32)
    flow[0,0]=[0,.5,1,1.5]
    assert grouping_targets(foreground,flow,prediction_horizon_frames=4).count==1


def test_equal_group_weight_and_background_gradient_have_scalar_witness():
    target=grouping_targets(np.array([[1,1,-1,1]],np.float32),np.zeros((2,1,4),np.float32),prediction_horizon_frames=4)
    features=torch.zeros((32,1,4),device='cpu')
    features[0,0,:2]=torch.tensor([1.,-1.])
    features[1,0,3]=1
    features.requires_grad_()
    result=grouping_loss(features,target)
    assert result.pull.item()==.5 and result.separation.item()==0
    (result.pull+result.separation).backward()
    assert torch.count_nonzero(features.grad[:,:,2])==0
    assert features.grad[0,0,0]>0 and features.grad[0,0,1]<0
    permuted=replace(target,labels=np.where(target.labels<0,-1,1-target.labels))
    assert grouping_loss(features,permuted).pull.item()==result.pull.item()


def test_separation_pushes_distinct_groups_but_exact_collapse_is_stationary():
    target=grouping_targets(np.array([[1,-1,1]],np.float32),np.zeros((2,1,3),np.float32),prediction_horizon_frames=4)
    features=torch.zeros((32,1,3),device='cpu')
    features[0,0,2]=.5
    features.requires_grad_()
    result=grouping_loss(features,target)
    assert result.separation.item()==.25
    result.separation.backward()
    assert features.grad[0,0,0]>0 and features.grad[0,0,2]<0
    collapsed=torch.zeros_like(features,requires_grad=True)
    collapse_loss=grouping_loss(collapsed,target).separation
    assert collapse_loss.item()==1
    collapse_loss.backward()
    assert torch.count_nonzero(collapsed.grad)==0  # Known ambiguity, no hidden tie-breaking teacher.


def test_empty_foreground_is_unknown_and_capacity_refuses_without_truncation():
    foreground=-np.ones((2,2),np.float32)
    flow=np.zeros((2,2,2),np.float32)
    target=grouping_targets(foreground,flow,prediction_horizon_frames=4)
    features=torch.rand((32,2,2),requires_grad=True,device='cpu')
    losses=grouping_loss(features,target)
    (losses.pull+losses.separation).backward()
    assert losses.groups==0 and torch.count_nonzero(features.grad)==0
    foreground[:]=[[1,-1],[-1,1]]
    with pytest.raises(ValueError,match='capacity'):
        grouping_targets(foreground,flow,prediction_horizon_frames=4,max_groups=1)


def test_motion_targets_handle_extreme_finite_fields_and_validate_before_cast():
    flow=np.full((2,1,2),np.finfo(np.float32).max,np.float32)
    flow[:,:,1]*=-1
    assert grouping_targets(np.ones((1,2),np.float32),flow,prediction_horizon_frames=4).count==2
    huge=np.full((2,1,2),np.finfo(np.float64).max,np.float64)
    with pytest.raises(ValueError,match='representable'):
        grouping_targets(np.ones((1,2),np.float32),huge,prediction_horizon_frames=4)


def identity_pair(count=2):
    tracker=InstanceTracker((32,64))
    initial=tracker.update(observations([2,20][:count],[1,1][:count]),frame=0,prediction_horizon_frames=4)
    future=tracker.update(observations([4,22][:count],[1,1][:count]),frame=8,prediction_horizon_frames=4)
    def endpoint(frame):
        features=torch.zeros((32,32,64),device='cpu')
        for row in frame.observed:
            features.reshape(32,-1)[row.track_id,torch.tensor(row.instance.source_pixels)]=1
        return IdentityObservation('ordinary-env0',0,frame,features.requires_grad_())
    return endpoint(initial),endpoint(future)


def test_identity_prefers_true_own_correspondence_with_gradients_at_both_endpoints():
    earlier,later=identity_pair()
    good=identity_loss(earlier,later)
    swapped=replace(later,embedding=later.embedding[[1,0,*range(2,32)]])
    wrong=identity_loss(earlier,swapped)
    assert good.loss.item()<.001 and wrong.loss.item()>9
    assert good.shared_visible_ids==2 and good.supervised_anchors==4
    good.loss.backward()
    for endpoint in (earlier,later):
        assert torch.isfinite(endpoint.embedding.grad).all()
        assert torch.count_nonzero(endpoint.embedding.grad)>0
        assert torch.count_nonzero(endpoint.embedding.grad[:,:5])==0


def test_identity_order_and_consistent_id_renaming_do_not_change_loss():
    earlier,later=identity_pair()
    def rename(endpoint):
        rows=tuple(replace(row,track_id=101-row.track_id) for row in reversed(endpoint.tracking.observed))
        return replace(endpoint,tracking=replace(endpoint.tracking,observed=rows))
    torch.testing.assert_close(identity_loss(earlier,later).loss,identity_loss(rename(earlier),rename(later)).loss,rtol=0,atol=0)


@pytest.mark.parametrize('change',['stream','epoch','lag'])
def test_identity_rejects_cross_stream_reset_or_wrong_horizon(change):
    earlier,later=identity_pair()
    if change=='stream': later=replace(later,stream_id='laboratory-fork')
    if change=='epoch': later=replace(later,epoch=1)
    if change=='lag': later=replace(later,tracking=replace(later.tracking,frame=4))
    with pytest.raises(ValueError,match='exactly8'):
        identity_loss(earlier,later)


def test_one_object_has_no_contrastive_target_and_coasting_is_not_evidence():
    earlier,later=identity_pair(1)
    one=identity_loss(earlier,later)
    assert one.shared_visible_ids==1 and one.supervised_anchors==0 and one.loss.item()==0
    tracker=InstanceTracker((32,64))
    tracker.update(observations([2]),frame=0,prediction_horizon_frames=4)
    missing=tracker.update([],frame=8,prediction_horizon_frames=4)
    assert len(missing.coasting)==1
    hidden=replace(later,tracking=missing)
    result=identity_loss(earlier,hidden)
    assert result.shared_visible_ids==result.current_candidates==result.supervised_anchors==0
    result.loss.backward()
    assert torch.count_nonzero(later.embedding.grad)==0


def test_unknown_new_ids_are_not_forced_to_match_and_duplicate_ids_fail():
    earlier,later=identity_pair()
    unmatched=replace(later,tracking=replace(later.tracking,observed=tuple(replace(row,track_id=row.track_id+10) for row in later.tracking.observed)))
    result=identity_loss(earlier,unmatched)
    assert result.shared_visible_ids==result.supervised_anchors==0 and result.loss.item()==0
    duplicate=replace(later,tracking=replace(later.tracking,observed=(later.tracking.observed[0],)*2))
    with pytest.raises(ValueError,match='Unique'):
        identity_loss(earlier,duplicate)
