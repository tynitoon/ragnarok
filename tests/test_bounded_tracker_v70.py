"""Bounded coasting, no silent eviction, atomic capacity refusal."""
import pytest
from ragnarok.learning.bounded_tracker_v70 import BoundedInstanceTracker
from tests.test_instance_tracker_v70 import observations


def test_coasting_counts_toward_capacity_without_losing_original_identity_or_clock():
    tracker=BoundedInstanceTracker((32,64),max_tracks=1)
    tracker.update(observations([2]),frame=0,prediction_horizon_frames=4)
    with pytest.raises(ValueError,match='coasting'):
        tracker.update(observations([40]),frame=4,prediction_horizon_frames=4)
    assert tracker._frame==0 and tracker._next_id==1 and len(tracker._tracks)==1
    assert tracker._tracks[0].last_seen==0
    returned=tracker.update(observations([2]),frame=4,prediction_horizon_frames=4)
    assert returned.observed[0].track_id==0


def test_expired_track_frees_capacity_and_fork_keeps_the_limit():
    tracker=BoundedInstanceTracker((32,64),max_tracks=1)
    tracker.update(observations([2]),frame=0,prediction_horizon_frames=4)
    branch=tracker.fork()
    assert branch.max_tracks==1
    result=branch.update(observations([40]),frame=31,prediction_horizon_frames=4)
    assert result.expired_ids==(0,) and result.observed[0].track_id==1
    assert tracker._frame==0


def test_too_many_observations_refuse_before_hungarian_allocation(monkeypatch):
    import ragnarok.learning.instance_tracker_v70 as core
    tracker=BoundedInstanceTracker((32,64),max_tracks=1)
    tracker.update(observations([2]),frame=0,prediction_horizon_frames=4)
    monkeypatch.setattr(core,'linear_sum_assignment',lambda *a,**k:pytest.fail('Assignment called over capacity'))
    with pytest.raises(ValueError,match='before assignment'):
        tracker.update(observations([10,20]),frame=4,prediction_horizon_frames=4)
    assert tracker._frame==0


@pytest.mark.parametrize('capacity',[0,True,4097])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError): BoundedInstanceTracker((32,64),max_tracks=capacity)
