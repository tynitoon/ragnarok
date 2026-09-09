"""Native-frame alignment without downstream or terminal assumptions."""
import numpy as np
import pytest
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.environments.rgb_trace_difference_v70 import difference


def clip(raw, anchor=None):
    anchor = np.zeros(raw.shape[1:], dtype=np.uint8) if anchor is None else anchor
    pooled = np.maximum(np.concatenate([anchor[None], raw[:-1]]), raw) if len(raw) else raw.copy()
    return RGBBranchTrace(raw.copy(), pooled, pooled[-1].copy() if len(raw) else anchor.copy())


def test_first_actual_change_is_distinct_from_pooled_persistence():
    raw = np.zeros((4,8,9,3), dtype=np.uint8)
    raw[0,2,3] = 80
    raw[2,5,6] = 90
    control = clip(np.zeros_like(raw))
    intervention = clip(raw)
    a = difference(intervention, control)
    b = difference(intervention, control, image_kind='pooled')
    assert a.first_changed_frame[2,3] == b.first_changed_frame[2,3] == 1
    assert a.first_changed_frame[5,6] == 3
    assert np.count_nonzero(a.first_changed_frame) == 2
    assert list(a.changed[:,2,3]) == [True,False,False,False]
    assert list(b.changed[:,2,3]) == [True,True,False,False]
    assert np.array_equal(intervention.raw,raw)


@pytest.mark.parametrize('counts', [(4,1),(0,4),(4,0),(0,0)])
def test_terminal_tails_are_unpaired_not_fabricated_divergences(counts):
    left, right = counts
    a = clip(np.zeros((left,8,9,3),dtype=np.uint8))
    b = clip(np.zeros((right,8,9,3),dtype=np.uint8))
    result = difference(a,b)
    assert result.changed.shape == (min(counts),8,9)
    assert not result.changed.any() and not result.first_changed_frame.any()
    assert result.unpaired_intervention_frames == left-min(counts)
    assert result.unpaired_control_frames == right-min(counts)


def test_shared_anchor_can_delay_pooled_change_without_rewriting_raw_change_time():
    raw = np.zeros((2,8,9,3), dtype=np.uint8)
    altered = raw.copy();altered[0,2,3] = 20
    anchor = np.zeros((8,9,3),dtype=np.uint8);anchor[2,3] = 30
    a,b = clip(altered,anchor),clip(raw,anchor)
    assert difference(a,b).first_changed_frame[2,3] == 1
    assert difference(a,b,image_kind='pooled').first_changed_frame[2,3] == 2


def test_bad_geometry_dtype_or_endpoint_cannot_supply_a_target():
    a = clip(np.zeros((2,8,9,3),dtype=np.uint8))
    b = clip(np.zeros((2,7,9,3),dtype=np.uint8))
    with pytest.raises(ValueError,match='geometry'):difference(a,b)
    b = RGBBranchTrace(a.raw.astype(float),a.pooled,a.endpoint)
    with pytest.raises(ValueError,match='uint8'):difference(a,b)
    b = RGBBranchTrace(a.raw,a.pooled,np.ones_like(a.endpoint))
    with pytest.raises(ValueError,match='endpoint'):difference(a,b)
