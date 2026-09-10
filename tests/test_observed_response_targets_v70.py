"""Past/future boundary and occlusion witnesses; no network, ROM or data file."""
from dataclasses import fields, replace

import numpy as np
import pytest

from ragnarok.environments.owned_rgb_collector_v70 import BranchExperience, RGBAnchor
from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.learning.observed_response_targets_v70 import (
    observed_response_targets, prediction_input,
)
from tests.test_counterfactual_occlusion_limits_v70 import (
    _assert_same_observations, _changed_occluder_world, _hidden_subject_world,
    _own_hypothesis,
)


def trace(frames, pair):
    raw = np.asarray(frames, dtype=np.uint8).reshape((-1,) + pair.shape[1:])
    previous = np.concatenate((pair[-1:], raw[:-1]), axis=0) if len(raw) else raw
    pooled = np.maximum(previous, raw)
    endpoint = pooled[-1].copy() if len(raw) else np.maximum(*pair)
    return RGBBranchTrace(raw, pooled, endpoint)


def record_for(anchor, intervention, control, *, requested=4, effect=None):
    pair = np.stack((anchor, anchor))
    return BranchExperience(
        ('fixture', 2), 0, RGBAnchor(('fixture', 1, 0), 0, 4, pair),
        1, np.zeros(12, np.float32) if effect is None else effect, 0, requested,
        intervention, control,
    )


def build(record, instances=(), *, image_kind='raw'):
    past = prediction_input(record.anchor.pair, record.effect_assumption)
    return observed_response_targets(record, past, instances, image_kind=image_kind)


def assert_unknown(target, reason):
    assert target.value is None and target.weight == 0
    assert target.reason == reason


def test_observed_flash_survives_and_raw_pooling_have_different_horizons():
    anchor = np.zeros((3, 4, 3), np.uint8)
    pair = np.stack((anchor, anchor))
    flash = anchor.copy()
    flash[1, 2] = (255, 17, 9)
    record = record_for(anchor, trace([flash, anchor], pair), trace([anchor, anchor], pair), requested=2)
    raw, pooled = (build(record, image_kind=kind) for kind in ('raw', 'pooled'))
    expected = np.zeros((3, 4), bool)
    expected[1, 2] = True
    np.testing.assert_array_equal(raw.horizons[0].rgb_changed, expected)
    assert raw.horizons[1].rgb_changed is not None
    assert not raw.horizons[1].rgb_changed.any()
    for horizon in pooled.horizons:
        np.testing.assert_array_equal(horizon.rgb_changed, expected)
    # No source instance has been guessed from that screen-coordinate response.
    assert all(not horizon.controllability for horizon in raw.horizons)
    assert_unknown(raw.horizons[0].foreground, 'correspondence_not_validated')


@pytest.mark.parametrize('left_count,right_count,requested', [
    (2, 0, 3), (0, 2, 3), (2, 1, 3), (1, 2, 3), (0, 0, 1), (64, 64, 64),
])
def test_asymmetric_and_empty_prefixes_preserve_unknown_future(left_count, right_count, requested):
    anchor, _, _ = _changed_occluder_world(subject_is_controlled=False)
    pair = np.stack((anchor, anchor))
    own, _ = _own_hypothesis(anchor)
    record = record_for(anchor, trace([anchor] * left_count, pair),
                        trace([anchor] * right_count, pair), requested=requested)
    result = build(record, own)
    assert result.common_prefix_frames == min(left_count, right_count)
    assert result.unpaired_intervention_frames == max(0, left_count - right_count)
    assert result.unpaired_control_frames == max(0, right_count - left_count)
    assert len(result.horizons) == requested
    assert result.source_shape == anchor.shape[:2]
    for offset, horizon in enumerate(result.horizons, 1):
        assert horizon.frame_offset == offset
        left, right = offset <= left_count, offset <= right_count
        assert horizon.intervention_observed is left and horizon.control_observed is right
        if left and right:
            assert horizon.rgb_changed is not None and not horizon.rgb_changed.any()
        else:
            assert horizon.rgb_changed is None
        assert_unknown(horizon.foreground, 'correspondence_not_validated' if left else 'missing_intervention_horizon')
        assert_unknown(horizon.controllability[0].target,
                       'correspondence_not_validated' if left and right else 'incomplete_common_horizon')


@pytest.mark.parametrize('image_kind', ['raw', 'pooled'])
@pytest.mark.parametrize('world_builder,has_response', [
    (_changed_occluder_world, True), (_hidden_subject_world, False),
])
def test_opposite_causal_worlds_remain_unknown_without_discarding_rgb(world_builder, has_response, image_kind):
    worlds = [world_builder(subject_is_controlled=controlled) for controlled in (False, True)]
    _assert_same_observations(*worlds)
    results = []
    for anchor, traces, _judge_only_states in worlds:
        own, _own_motion_unused = _own_hypothesis(anchor)
        # Physical states, true control and predicted motion do not enter this API.
        results.append(build(record_for(anchor, *traces), own, image_kind=image_kind))
    for left, right in zip(results[0].horizons, results[1].horizons, strict=True):
        np.testing.assert_array_equal(left.rgb_changed, right.rgb_changed)
        for horizon in (left, right):
            assert_unknown(horizon.foreground, 'correspondence_not_validated')
            assert len(horizon.controllability) == 1
            assert_unknown(horizon.controllability[0].target, 'correspondence_not_validated')
    assert bool(results[0].horizons[-1].rgb_changed.any()) is has_response


def test_past_and_outputs_are_immutable_snapshots_with_no_future_or_command_field():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    own, _ = _own_hypothesis(anchor)
    own = [replace(item, source_pixels=item.source_pixels.copy()) for item in own]
    record = record_for(anchor, *traces)
    pair, effect = record.anchor.pair.copy(), record.effect_assumption.copy()
    past = prediction_input(pair, effect)
    pair[:] = 13
    effect[:] = 1
    np.testing.assert_array_equal(past.pair, record.anchor.pair)
    np.testing.assert_array_equal(past.effect, record.effect_assumption)
    for keyword in ('future', 'requested_action', 'game', 'mask'):
        assert not hasattr(past, keyword)
        with pytest.raises(TypeError):
            prediction_input(record.anchor.pair, **{keyword: 1})
    result = observed_response_targets(record, past, own, image_kind='raw')
    change = result.horizons[-1].rgb_changed
    pixels = result.horizons[-1].controllability[0].source_pixels
    expected_change, expected_pixels = change.copy(), pixels.copy()
    own[0].source_pixels[:] = 0
    record.intervention.raw[:] = 0
    record.anchor.pair[:] = 0
    record.effect_assumption[:] = 1
    np.testing.assert_array_equal(change, expected_change)
    np.testing.assert_array_equal(pixels, expected_pixels)
    np.testing.assert_array_equal(past.pair, np.stack((anchor, anchor)))
    assert past.effect.sum() == 0
    for array in (past.pair, past.effect, change, pixels):
        with pytest.raises(ValueError):
            array.flags.writeable = True
        with pytest.raises(ValueError):
            array.flat[0] = 0
    assert {item.name for item in fields(result.origin)} == {
        'operation_id', 'anchor_token', 'worker', 'epoch', 'frame',
    }


def test_array_metadata_mutation_does_not_change_inputs_or_other_horizons():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    own, _ = _own_hypothesis(anchor)
    record = record_for(anchor, *traces)
    past = prediction_input(record.anchor.pair, record.effect_assumption)
    result = observed_response_targets(record, past, own, image_kind='raw')
    getters = (
        lambda: past.pair, lambda: past.effect,
        lambda: result.horizons[0].rgb_changed,
        lambda: result.horizons[0].controllability[0].source_pixels,
    )
    for get_array in getters:
        expected = get_array().copy()
        altered = get_array()
        altered.shape = (altered.size,)
        altered.dtype = np.uint8 if altered.dtype.itemsize > 1 else np.uint16
        assert altered.dtype != expected.dtype
        fresh = get_array()
        assert fresh.dtype == expected.dtype and fresh.shape == expected.shape
        np.testing.assert_array_equal(fresh, expected)
    for horizon in result.horizons:
        np.testing.assert_array_equal(horizon.controllability[0].source_pixels, own[0].source_pixels)
    # The stable past input still matches the recorded observation after all
    # the attempted metadata mutations; no network forward is involved.
    observed_response_targets(record, past, own, image_kind='raw')


def test_different_future_and_command_keep_identical_predictive_inputs():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    record = record_for(anchor, *traces)
    past = prediction_input(record.anchor.pair, record.effect_assumption)
    first = observed_response_targets(record, past, image_kind='raw')
    second = observed_response_targets(replace(record, intervention=record.control, requested_action=17),
                                       past, image_kind='raw')
    assert first.origin == second.origin
    assert first.horizons[-1].rgb_changed.any() and not second.horizons[-1].rgb_changed.any()
    np.testing.assert_array_equal(past.pair, record.anchor.pair)
    np.testing.assert_array_equal(past.effect, record.effect_assumption)
    other = prediction_input(record.anchor.pair, record.effect_assumption)
    np.testing.assert_array_equal(past.pair, other.pair)
    np.testing.assert_array_equal(past.effect, other.effect)


@pytest.mark.parametrize('field_name,value', [
    ('operation_id', ('foreign', 2)), ('operation_id', ('fixture', 1)),
    ('operation_id', ('fixture', True)), ('worker', 1), ('worker', True),
    ('requested_frames', 0), ('requested_frames', 65), ('requested_frames', True),
    ('control_action', 1), ('requested_action', 18),
])
def test_malformed_record_metadata_is_refused(field_name, value):
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    with pytest.raises(ValueError):
        build(replace(record_for(anchor, *traces), **{field_name: value}))


def test_mismatched_past_effect_and_anchor_clock_are_refused():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    record = record_for(anchor, *traces)
    for pair, effect in ((np.zeros_like(record.anchor.pair), record.effect_assumption),
                         (record.anchor.pair, np.ones(12, np.float32))):
        with pytest.raises(ValueError, match='past and effect'):
            observed_response_targets(record, prediction_input(pair, effect), image_kind='raw')
    for bad_anchor in (replace(record.anchor, epoch=-1), replace(record.anchor, frame=-1),
                       replace(record.anchor, token=('fixture', 1, True))):
        with pytest.raises(ValueError, match='anchor clocks'):
            build(replace(record, anchor=bad_anchor))
    with pytest.raises(ValueError):
        build(record, image_kind='endpoint')


def test_malformed_rgb_prefix_pooling_or_empty_endpoint_is_refused():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    record = record_for(anchor, *traces)
    bad_traces = [
        replace(record.intervention, pooled=np.zeros_like(record.intervention.pooled)),
        replace(record.intervention, endpoint=np.zeros_like(anchor)),
        replace(record.intervention, raw=record.intervention.raw.astype(np.float32)),
        trace([anchor] * 5, record.anchor.pair),
        replace(trace([], record.anchor.pair), endpoint=np.zeros_like(anchor)),
    ]
    for bad_trace in bad_traces:
        with pytest.raises(ValueError):
            build(replace(record, intervention=bad_trace))


def test_invalid_or_overlapping_own_supports_are_refused():
    anchor, traces, _ = _changed_occluder_world(subject_is_controlled=False)
    record = record_for(anchor, *traces)
    own, _ = _own_hypothesis(anchor)
    item = own[0]
    bad_sets = [
        (item, item), (item, replace(item, frame_index=1)),
        (replace(item, frame_index=True),),
        (replace(item, source_pixels=np.array([], np.int64)),),
        (replace(item, source_pixels=np.array([1, 1], np.int64)),),
        (replace(item, source_pixels=np.array([-1], np.int64)),),
        (replace(item, source_pixels=np.array([anchor.shape[0] * anchor.shape[1]], np.int64)),),
        (replace(item, source_pixels=np.array([1.0])),),
    ]
    for invalid in bad_sets:
        with pytest.raises(ValueError):
            build(record, invalid)
