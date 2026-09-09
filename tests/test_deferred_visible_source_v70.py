"""Lazy geometry must neither invent frame0 objects nor omit a needed source."""
from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine
from ragnarok.evaluation.deferred_visible_source_v70 import DeferredVisibleSourceEngine
from ragnarok.evaluation.scene_perception_v70 import SceneObject


def setup(cls, reject_zero=False):
    obj = cls.__new__(cls)
    obj.loads = 1
    obj.engine = SimpleNamespace(accounting=lambda:dict(cold_loads=obj.loads))
    obj._loads = 1
    obj._phase = 'ordinary'
    obj._episode = obj._ordinary_frames = obj._lab_decoded = 0
    obj._source_labels = deque(maxlen=2)
    obj.calls, obj.truth = [],[]
    def decoder(current,previous):
        values = int(current[0]),int(previous[0])
        obj.calls.append(values)
        if reject_zero and values[0]==0:
            raise ValueError('unknown cold frame graphics bank')
        return [SceneObject(str(values[0]),'player',(1+values[0],2,3,4))]
    obj._decoder = decoder
    obj.scorer = SimpleNamespace(observe_truth=lambda labels,**kwargs:obj.truth.append((list(labels),kwargs)))
    return obj


def tap(frame):
    values = [max(0,frame-i) for i in (2,1,0)]
    arrays = [np.full(128,value,dtype=np.uint8) for value in values]
    return SimpleNamespace(before_previous=arrays[0],previous=arrays[1],current=arrays[2],frame_numbers=tuple(values))


def sources(obj):
    return [list(scene) for scene in obj._source_labels]


def test_unsupported_unscored_frame0_does_not_block_complete_repeat():
    old = setup(VisibleReferenceEngine,reject_zero=True)
    with pytest.raises(ValueError,match='graphics bank'):old._on_frame(tap(1))
    new = setup(DeferredVisibleSourceEngine,reject_zero=True)
    for frame in range(1,5):new._on_frame(tap(frame))
    assert [scene[0].instance_id for scene in sources(new)] == ['3','4']
    assert new.calls == [(1,0),(2,1),(3,2),(4,3)]
    assert len(new.truth)==4 and new._ordinary_frames==4


def test_needed_frame0_keeps_original_unknown_bank_guard():
    obj = setup(DeferredVisibleSourceEngine,reject_zero=True)
    obj._on_frame(tap(1))
    with pytest.raises(ValueError,match='graphics bank'):sources(obj)
    assert obj.calls == [(1,0),(0,0)]


@pytest.mark.parametrize('frames',[1,2,3,4])
def test_successful_scored_geometry_and_motion_history_equal_frozen_version(frames):
    old,new = setup(VisibleReferenceEngine),setup(DeferredVisibleSourceEngine)
    for frame in range(1,frames+1):
        old._on_frame(tap(frame));new._on_frame(tap(frame))
    assert sources(new)==sources(old) and new.truth==old.truth
    assert new._ordinary_frames==old._ordinary_frames==frames


def test_deferred_ram_is_copied_and_successful_result_is_decoded_only_once():
    obj = setup(DeferredVisibleSourceEngine)
    first = tap(1)
    obj._on_frame(first)
    first.previous[:]=99;first.before_previous[:]=98
    assert sources(obj)[0][0].instance_id=='0'
    assert sources(obj)[0][0].instance_id=='0'
    assert obj.calls==[(1,0),(0,0)]


def test_lab_frames_do_not_replace_ordinary_sources_or_movement_history():
    obj = setup(DeferredVisibleSourceEngine)
    for frame in range(1,5):obj._on_frame(tap(frame))
    before = sources(obj),list(obj.truth)
    obj._phase = 'lab'
    obj._on_frame(tap(5));obj._on_frame(tap(6))
    assert (sources(obj),obj.truth)==before and obj._lab_decoded==2
    obj._phase = 'ordinary'
    obj.loads=2
    obj._on_frame(tap(1))
    assert [scene[0].instance_id for scene in sources(obj)]==['0','1']
    assert obj._episode==1 and obj._ordinary_frames==5
