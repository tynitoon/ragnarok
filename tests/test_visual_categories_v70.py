"""Synthetic own descriptors; no game names, semantic types or palette truth."""
from dataclasses import replace
import json

import numpy as np
import pytest

from ragnarok.learning.instance_tracker_v70 import InstanceTracker
from ragnarok.learning.visual_categories_v70 import VisualCategoryBook,visual_features
from tests.test_instance_tracker_v70 import observations


def frame(positions=(2,20),*,identities=None):
    return InstanceTracker((32,64)).update(observations(list(positions),identities=identities),frame=0,prediction_horizon_frames=4)


def test_same_appearance_and_relative_size_reuse_category_without_using_position_or_track_id():
    tracked=frame(identities=[0,0])
    book=VisualCategoryBook()
    result=book.observe(tracked)
    assert [r.category_id for r in result]==[0,0]
    assert [r.created for r in result]==[True,False]
    later=replace(tracked,frame=4,observed=tuple(replace(row,track_id=row.track_id+100) for row in reversed(tracked.observed)))
    assert [r.category_id for r in book.observe(later)]==[0,0]
    assert book.export_local()['categories'][0]['observations']==4


def test_different_own_features_create_categories_and_order_is_geometric():
    tracked=frame()
    first,second=VisualCategoryBook(),VisualCategoryBook()
    a=first.observe(tracked)
    b=second.observe(replace(tracked,observed=tuple(reversed(tracked.observed))))
    assert a==b and [r.category_id for r in a]==[0,1]
    assert first.export_local()==second.export_local()
    assert len(visual_features(tracked.observed[0].instance))==42


def test_capacity_failure_is_atomic_even_after_an_existing_centroid_was_updated():
    tracked=frame()
    book=VisualCategoryBook(max_categories=1)
    book.observe(replace(tracked,observed=(tracked.observed[0],)))
    before=book.export_local()
    with pytest.raises(ValueError,match='capacity'): book.observe(replace(tracked,frame=4))
    assert book.export_local()==before
    book.observe(replace(tracked,frame=4,observed=(tracked.observed[0],)))
    assert book.export_local()['categories'][0]['observations']==2


def test_coasting_does_not_update_categories_and_local_export_is_detached():
    tracker=InstanceTracker((32,64))
    book=VisualCategoryBook()
    book.observe(tracker.update(observations([2]),frame=0,prediction_horizon_frames=4))
    missing=tracker.update([],frame=4,prediction_horizon_frames=4)
    assert missing.coasting and book.observe(missing)==()
    result=book.export_local()
    assert result['carried_across_games'] is False and result['categories'][0]['observations']==1
    json.dumps(result,allow_nan=False)
    result['categories'][0]['center'][0]=1000
    assert book.export_local()['categories'][0]['center'][0]==1
    assert VisualCategoryBook().export_local()['categories']==[]


def test_log_size_and_histogram_domains_are_explicit():
    instance=frame().observed[0].instance
    for index,value in ((2,0),(3,-1),(41,-.1)):
        descriptor=instance.descriptor.copy();descriptor[index]=value
        with pytest.raises(ValueError):visual_features(replace(instance,descriptor=descriptor))
    descriptor=instance.descriptor.copy();descriptor[41:49]=0
    with pytest.raises(ValueError):visual_features(replace(instance,descriptor=descriptor))
    tiny=instance.descriptor.copy();tiny[2:4]=np.nextafter(np.float32(0),np.float32(1))
    assert np.isfinite(visual_features(replace(instance,descriptor=tiny))).all()
