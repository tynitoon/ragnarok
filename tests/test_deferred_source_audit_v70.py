"""Source-cache replay boundaries and retained guard failures, without ALE."""
import numpy as np
import pytest
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from scripts.check_deferred_visible_source_v70 import audit_rows


def rows(lengths):
    for episode,length in enumerate(lengths):
        for frame in range(1,length+1):
            yield dict(episode=episode,frame_in_episode=frame,
                       ram=np.full(128,frame,dtype=np.uint8),
                       ram_previous=np.full(128,frame-1,dtype=np.uint8))


def decoder(current,previous):
    if current[0] == 0: raise ValueError('unsupported actual cold scene')
    return [SceneObject('player','player',(int(current[0]),20,2,3))]


def test_every_raw_history_and_partial_group_including_archive_end_is_consumed():
    result=audit_rows(rows([4,6,3]),decoder,expected_frames=13,expected_episodes=3)
    assert result['consumed_groups']==4 and result['group_lengths']=={'4':2,'2':1,'3':1}
    assert result['callback_decoder_calls']==13 and result['consumed_source_objects']==8
    assert result['movement_history_exactly_equal'] and result['group_membership_exactly_equal']


def test_required_unsupported_cold_source_fails_instead_of_omitting_one_frame_episode():
    with pytest.raises(ValueError,match='actual cold'):
        audit_rows(rows([4,1,4]),decoder,expected_frames=9,expected_episodes=3)


def test_incomplete_archive_fails_even_when_observed_prefix_is_equal():
    with pytest.raises(ValueError,match='Complete declared'):
        audit_rows(rows([4]),decoder,expected_frames=8,expected_episodes=1)
