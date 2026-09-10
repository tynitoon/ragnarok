"""Observed branch RGB responses, without claiming source-object attribution.

This prospective interface does not replace the frozen historical C objectives.
It supplies no admissible m/C target until a correspondence mechanism exists.
Recording ownership is a caller contract, not a guarantee derived from pixels.
"""
from dataclasses import dataclass, field

import numpy as np

from ragnarok.environments.owned_rgb_collector_v70 import (
    BranchExperience, RGBAnchor, _branch_trace, _effect, _pair,
)
from ragnarok.environments.rgb_trace_difference_v70 import difference
from ragnarok.learning.visual_instances_v70 import VisualInstance


def _snapshot(array):
    # A read-only owning ndarray can have its writeable flag re-enabled. Bytes
    # backing also prevents that route, while breaking all caller-buffer aliases.
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True, eq=False)
class PredictionInput:
    """Only data available before the branch; no command IDs or future images."""

    pair: np.ndarray
    effect: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'pair', _snapshot(_pair(self.pair)))
        object.__setattr__(self, 'effect', _snapshot(_effect(self.effect)))


def prediction_input(pair, effect=None):
    """Snapshot two own past RGB frames and an own twelve-effect assumption."""
    return PredictionInput(pair, effect)


@dataclass(frozen=True, slots=True)
class UnknownTarget:
    """No numeric label. Weight zero denotes unavailable supervision."""

    reason: str
    value: None = field(default=None, init=False)
    weight: float = field(default=0.0, init=False)


@dataclass(frozen=True, slots=True, eq=False)
class InstanceAttribution:
    frame_index: int
    source_pixels: np.ndarray
    target: UnknownTarget


@dataclass(frozen=True, slots=True, eq=False)
class HorizonResponse:
    frame_offset: int  # One-based native offset after the recorded anchor.
    intervention_observed: bool
    control_observed: bool
    rgb_changed: np.ndarray | None  # Screen coordinates, never instance labels.
    foreground: UnknownTarget  # Uniform status over the entire source_shape.
    controllability: tuple[InstanceAttribution, ...]


@dataclass(frozen=True, slots=True)
class ResponseOrigin:
    """Recording metadata only; none of these fields are model inputs."""

    operation_id: tuple[str, int]
    anchor_token: tuple[str, int, int]
    worker: int
    epoch: int
    frame: int


@dataclass(frozen=True, slots=True, eq=False)
class ObservedResponseTargets:
    origin: ResponseOrigin
    source_shape: tuple[int, int]
    requested_frames: int
    image_kind: str
    common_prefix_frames: int
    unpaired_intervention_frames: int
    unpaired_control_frames: int
    horizons: tuple[HorizonResponse, ...]


def _origin(record):
    if not isinstance(record, BranchExperience) or not isinstance(record.anchor, RGBAnchor):
        raise ValueError('One owned BranchExperience and its anchor are required')
    operation, token = record.operation_id, record.anchor.token
    if (not isinstance(operation, tuple) or len(operation) != 2
            or not isinstance(operation[0], str) or not operation[0]
            or type(operation[1]) is not int or operation[1] < 1
            or not isinstance(token, tuple) or len(token) != 3
            or token[0] != operation[0] or type(token[1]) is not int
            or not 0 <= token[1] < operation[1]
            or type(record.worker) is not int or not 0 <= record.worker < 16
            or type(token[2]) is not int or token[2] != record.worker
            or type(record.anchor.epoch) is not int or record.anchor.epoch < 0
            or type(record.anchor.frame) is not int or record.anchor.frame < 0):
        raise ValueError('Consistent own session, operation, worker and anchor clocks required')
    if (type(record.requested_frames) is not int or not 1 <= record.requested_frames <= 64
            or type(record.requested_action) is not int or not 0 <= record.requested_action < 18
            or type(record.control_action) is not int or record.control_action != 0):
        raise ValueError('An owned Atari branch request with horizon 1..64 and NOOP control is required')
    return ResponseOrigin(operation, token, record.worker, record.anchor.epoch, record.anchor.frame)


def _supports(instances, shape):
    height, width = shape
    occupied, identities, supports = set(), set(), []
    for index, instance in enumerate(instances):
        if index >= 4096:
            raise ValueError('At most 4096 own source instances are supported')
        if (not isinstance(instance, VisualInstance) or type(instance.frame_index) is not int
                or instance.frame_index < 0 or instance.frame_index in identities):
            raise ValueError('Unique own-instance frame indices required')
        pixels = instance.source_pixels
        if (not isinstance(pixels, np.ndarray) or pixels.ndim != 1 or not len(pixels)
                or not np.issubdtype(pixels.dtype, np.integer)
                or (pixels < 0).any() or (pixels >= height * width).any()):
            raise ValueError('Nonempty integer source pixels within the anchor image required')
        members = set(map(int, pixels))
        if len(members) != len(pixels) or occupied & members:
            raise ValueError('Own instance supports must be disjoint, without duplicate pixels')
        occupied.update(members)
        identities.add(instance.frame_index)
        supports.append((instance.frame_index, _snapshot(pixels.astype(np.int64))))
    return tuple(supports)


def observed_response_targets(record, prediction, instances=(), *, image_kind):
    """Build post-observation targets from one paid branch-pair recording.

    The owner must have made `prediction` and its source instances before the
    branch. Equality checks catch mismatches, not fabricated chronology or a
    fabricated record. This stateless transform neither pays nor repeats frames.

    m is unknown across every source pixel, even when there are no instances.
    C is unknown per own source instance. No loss function is exposed: `None`
    values and zero weights must not be counted as learned negative examples.
    """
    origin = _origin(record)
    if not isinstance(prediction, PredictionInput):
        raise ValueError('An own past-only PredictionInput is required')
    pair, effect = _pair(record.anchor.pair), _effect(record.effect_assumption)
    if (not np.array_equal(prediction.pair, pair)
            or not np.array_equal(prediction.effect, effect)):
        raise ValueError('Prediction past and effect must match this branch recording')
    _branch_trace(record.intervention, pair, record.requested_frames)
    _branch_trace(record.control, pair, record.requested_frames)
    observed = difference(record.intervention, record.control, image_kind=image_kind)
    shape = tuple(pair.shape[1:3])
    supports = _supports(instances, shape)
    left_count, right_count = len(record.intervention.raw), len(record.control.raw)
    common = len(observed.changed)
    horizons = []
    for offset in range(1, record.requested_frames + 1):
        left, right = offset <= left_count, offset <= right_count
        change = _snapshot(observed.changed[offset - 1]) if left and right else None
        motion = UnknownTarget('correspondence_not_validated' if left else 'missing_intervention_horizon')
        control = UnknownTarget('correspondence_not_validated' if left and right else 'incomplete_common_horizon')
        attributions = tuple(InstanceAttribution(index, pixels, control) for index, pixels in supports)
        horizons.append(HorizonResponse(offset, left, right, change, motion, attributions))
    return ObservedResponseTargets(
        origin, shape, record.requested_frames, image_kind, common,
        observed.unpaired_intervention_frames, observed.unpaired_control_frames,
        tuple(horizons),
    )
