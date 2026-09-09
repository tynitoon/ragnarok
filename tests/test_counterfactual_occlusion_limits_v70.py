"""External toy judges expose C's unresolved correspondence/occlusion ambiguity.

Physical identities, causal states and layer order exist only in these judges.
The unchanged helper receives its ordinary synthetic own-instance hypothesis,
its own-field-shaped hypothesis, and RGBBranchTrace values. No model, optimizer,
native engine or oracle argument is involved; these are not collected traces.
"""
import numpy as np
import pytest

from ragnarok.environments.rgb_branch_trace_v70 import RGBBranchTrace
from ragnarok.learning.counterfactual_targets_v70 import component_targets
from ragnarok.learning.visual_instances_v70 import DensePrediction, extract_instances


HEIGHT, WIDTH = 10, 12
SOURCE_Y, SOURCE_X, PROJECTED_X = 3, 2, 6
WHITE, RED, GREEN = (255, 255, 255), (255, 0, 0), (0, 255, 0)


def _trace(raw, anchor):
    """Use actual adjacent-frame MAX and the public synthetic trace data shape."""
    raw = np.asarray(raw, dtype=np.uint8)
    previous = np.concatenate((anchor[None], raw[:-1]), axis=0)
    pooled = np.maximum(previous, raw)
    return RGBBranchTrace(raw.copy(), pooled, pooled[-1].copy())


def _own_hypothesis(anchor):
    """Fixed toy head outputs; identical in both causally opposite worlds.

No head is learned or inferred here. This grants one source component and a
four-pixel prediction, without giving the helper any world identity/layer flag.
"""
    one = (1, HEIGHT, WIDTH)
    embedding = np.zeros((32, HEIGHT, WIDTH), dtype=np.float32)
    embedding[0] = 1
    foreground = np.full(one, -10, dtype=np.float32)
    foreground[0, SOURCE_Y, SOURCE_X] = 10
    displacement = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)
    displacement[0, SOURCE_Y, SOURCE_X] = PROJECTED_X - SOURCE_X
    prediction = DensePrediction(
        embedding, foreground, displacement,
        np.zeros(one, dtype=np.float32), np.zeros(one, dtype=np.float32),
    )
    instances = extract_instances(prediction, anchor)
    assert len(instances) == 1
    np.testing.assert_array_equal(
        instances[0].source_pixels, [SOURCE_Y * WIDTH + SOURCE_X],
    )
    return instances, displacement


def _changed_occluder_world(*, subject_is_controlled):
    """Judge A moves autonomously to x6; either A or front object B changes.

Uncontrolled world: B covers A at frame4 and B changes RED -> GREEN under
intervention; A's own physical state is identical in both branches.
Controlled world: A is in front and changes to RED/GREEN at frame4; B stays
RED. Every observed RGB frame is identical to the uncontrolled world's frame.
The intervention affects appearance here, not A's already-autonomous motion.
"""
    anchor = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    anchor[SOURCE_Y, SOURCE_X] = WHITE
    anchor[SOURCE_Y, PROJECTED_X] = RED
    traces, subject_states = [], []
    for intervention in (True, False):
        frames, states = [], []
        for offset in range(1, 5):
            x = SOURCE_X + offset
            response_color = GREEN if intervention else RED
            a_color = response_color if subject_is_controlled and offset == 4 else WHITE
            b_color = response_color if not subject_is_controlled and offset == 4 else RED
            frame = np.zeros_like(anchor)
            if subject_is_controlled:
                frame[SOURCE_Y, PROJECTED_X] = b_color
                frame[SOURCE_Y, x] = a_color
            else:
                frame[SOURCE_Y, x] = a_color
                frame[SOURCE_Y, PROJECTED_X] = b_color
            frames.append(frame)
            states.append((x, SOURCE_Y, a_color))
        traces.append(_trace(frames, anchor))
        subject_states.append(tuple(states))
    return anchor, tuple(traces), tuple(subject_states)


def _hidden_subject_world(*, subject_is_controlled):
    """Judge A moves under an opaque static strip, so its causal states are hidden.

Intervention trajectory: x3,4,5,6 in both worlds. Control trajectory: x3,3,3,3
when A is controlled, x3,4,5,6 when autonomous. A's predicted intervention
endpoint x6 is physically correct in both worlds, but never visible there.
"""
    anchor = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    anchor[SOURCE_Y, SOURCE_X] = WHITE
    anchor[SOURCE_Y, 3:7] = RED
    traces, subject_states = [], []
    for intervention in (True, False):
        frames, states = [], []
        for offset in range(1, 5):
            x = 3 if subject_is_controlled and not intervention else SOURCE_X + offset
            frame = np.zeros_like(anchor)
            frame[SOURCE_Y, x] = WHITE
            frame[SOURCE_Y, 3:7] = RED  # Opaque B is in front of A in this judge.
            frames.append(frame)
            states.append((x, SOURCE_Y, WHITE))
        traces.append(_trace(frames, anchor))
        subject_states.append(tuple(states))
    return anchor, tuple(traces), tuple(subject_states)


def _assert_same_observations(first, second):
    np.testing.assert_array_equal(first[0], second[0])
    for left, right in zip(first[1], second[1], strict=True):
        for name in ('raw', 'pooled', 'endpoint'):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def _evaluate(world, image_kind):
    anchor, traces, _judge_only_states = world
    instances, displacement = _own_hypothesis(anchor)
    # No physical identity, true motion, layer order or visibility enters this API.
    return component_targets(
        instances, displacement, traces[0], traces[1],
        horizon_frames=4, image_kind=image_kind,
    )


def _assert_supervised(result, expected):
    assert result.horizon_frames == result.common_prefix_frames == 4
    assert result.weights.sum() == 1
    assert result.weights[SOURCE_Y, SOURCE_X] == 1
    assert result.values[SOURCE_Y, SOURCE_X] == expected
    assert len(result.instances) == 1
    row = result.instances[0]
    assert row['eligible'] is True
    assert row['target'] == expected
    assert row['projected_source_pixels'] == 1
    assert row['reason'] == 'own_motion_and_complete_observation'


@pytest.mark.parametrize('image_kind', ['raw', 'pooled'])
def test_other_object_at_correct_destination_produces_false_positive(image_kind):
    uncontrolled = _changed_occluder_world(subject_is_controlled=False)
    controlled = _changed_occluder_world(subject_is_controlled=True)
    assert uncontrolled[2][0] == uncontrolled[2][1]
    assert controlled[2][0] != controlled[2][1]
    assert all(states[-1][0] == PROJECTED_X for states in uncontrolled[2])
    _assert_same_observations(uncontrolled, controlled)
    # In one world the difference belongs to B; in the other it belongs to A.
    # Correct endpoint + complete traces cannot disambiguate these equal inputs.
    for world in (uncontrolled, controlled):
        _assert_supervised(_evaluate(world, image_kind), 1)


@pytest.mark.parametrize('image_kind', ['raw', 'pooled'])
def test_hidden_controlled_subject_produces_false_negative(image_kind):
    controlled = _hidden_subject_world(subject_is_controlled=True)
    uncontrolled = _hidden_subject_world(subject_is_controlled=False)
    assert controlled[2][0] != controlled[2][1]
    assert uncontrolled[2][0] == uncontrolled[2][1]
    assert controlled[2][0][-1][0] == PROJECTED_X
    assert controlled[2][1][-1][0] == 3
    _assert_same_observations(controlled, uncontrolled)
    for trace in controlled[1]:
        np.testing.assert_array_equal(trace.raw[:, SOURCE_Y, PROJECTED_X], [RED] * 4)
    # No branch pixel reveals A's response behind B. The existing target is
    # nonetheless an admissible zero rather than unknown (weight0/targetNone).
    for world in (controlled, uncontrolled):
        _assert_supervised(_evaluate(world, image_kind), 0)
