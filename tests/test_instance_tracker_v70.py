"""Synthetic positions/predictions; no game identity or emulator clock access."""
import numpy as np
import pytest
from ragnarok.learning.visual_instances_v70 import extract_instances
from ragnarok.learning.instance_tracker_v70 import InstanceTracker
from tests.test_visual_instances_v70 import scene


def observations(positions,velocities=None,identities=None):
    prediction,rgb=scene(32,64)
    velocities=velocities or [0]*len(positions)
    identities=identities or list(range(len(positions)))
    prediction.embedding[:]=0
    for x,dx,identity in zip(positions,velocities,identities):
        prediction.foreground_logits[:,10:12,x:x+2]=10
        prediction.embedding[identity,10:12,x:x+2]=1
        prediction.displacement[0,10:12,x:x+2]=dx
    return extract_instances(prediction,rgb)


def test_predicted_motion_preserves_id_even_when_displacement_exceeds_match_radius():
    tracker=InstanceTracker((32,64))
    first=tracker.update(observations([2],[12]),frame=0,prediction_horizon_frames=4)
    next_frame=tracker.update(observations([14],[12]),frame=4,prediction_horizon_frames=4)
    assert first.observed[0].track_id==next_frame.observed[0].track_id
    assert not next_frame.coasting and not next_frame.observed[0].reappeared


def test_hungarian_uses_appearance_at_crossing_and_is_input_order_invariant():
    a,b=InstanceTracker((32,64)),InstanceTracker((32,64))
    initial=observations([10,20],identities=[0,1])
    a.update(initial,frame=0,prediction_horizon_frames=4)
    b.update(list(reversed(initial)),frame=0,prediction_horizon_frames=4)
    crossed=observations([14,17],identities=[1,0])
    left=a.update(crossed,frame=4,prediction_horizon_frames=4)
    right=b.update(list(reversed(crossed)),frame=4,prediction_horizon_frames=4)
    assert [(o.instance.box,o.track_id) for o in left.observed]==[(o.instance.box,o.track_id) for o in right.observed]
    assert [o.track_id for o in left.observed]==[1,0]


def test_missing_object_coasts_without_becoming_visible_and_returns_with_same_id():
    tracker=InstanceTracker((32,64))
    first=tracker.update(observations([10],[2]),frame=0,prediction_horizon_frames=4)
    gap=tracker.update([],frame=4,prediction_horizon_frames=4)
    assert not gap.observed and len(gap.coasting)==1
    assert gap.coasting[0].missing_frames==4
    np.testing.assert_allclose(gap.coasting[0].predicted_center,[(11+2)/64,11/32])
    returned=tracker.update(observations([14],[2]),frame=8,prediction_horizon_frames=4)
    assert returned.observed[0].track_id==first.observed[0].track_id and returned.observed[0].reappeared


def test_expiry_after30_frames_cannot_be_reported_as_reappearance():
    tracker=InstanceTracker((32,64))
    tracker.update(observations([10]),frame=0,prediction_horizon_frames=4)
    assert len(tracker.update([],frame=30,prediction_horizon_frames=4).coasting)==1
    fresh=tracker.update(observations([10]),frame=31,prediction_horizon_frames=4)
    assert fresh.expired_ids==(0,) and fresh.observed[0].track_id==1
    assert fresh.observed[0].first_seen==31 and not fresh.observed[0].reappeared


def test_far_away_candidate_gets_new_id_instead_of_forced_match():
    tracker=InstanceTracker((32,64))
    tracker.update(observations([2]),frame=0,prediction_horizon_frames=4)
    moved=tracker.update(observations([40]),frame=4,prediction_horizon_frames=4)
    assert moved.observed[0].track_id==1 and moved.coasting[0].track_id==0


def test_branch_fork_and_input_mutation_cannot_change_ordinary_tracking_memory():
    tracker=InstanceTracker((32,64))
    initial=observations([2],[4])
    tracker.update(initial,frame=0,prediction_horizon_frames=4)
    branch=tracker.fork()
    branch.update(observations([40]),frame=4,prediction_horizon_frames=4)
    assert tracker._frame==0 and tracker._next_id==1
    initial[0].descriptor.flags.writeable=True
    initial[0].descriptor[:]=0
    real=tracker.update(observations([6],[4]),frame=4,prediction_horizon_frames=4)
    assert real.observed[0].track_id==0 and len(tracker._tracks[0].history)==2


def test_invalid_timeline_or_duplicate_pixels_cannot_advance_tracker():
    tracker=InstanceTracker((32,64))
    one=observations([2])
    tracker.update(one,frame=0,prediction_horizon_frames=4)
    with pytest.raises(ValueError,match='strictly increase'): tracker.update(one,frame=0,prediction_horizon_frames=4)
    with pytest.raises(ValueError,match='duplicate'): tracker.update(one+one,frame=4,prediction_horizon_frames=4)
    assert tracker._frame==0 and tracker._next_id==1
