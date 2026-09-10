"""Independent synthetic witnesses for the external local-equivalence judge.

No real image, NPZ, model, emulator, optimizer or frozen scientific module is
used. Integer convolutions test graph topology, not a floating-backend bitwise
contract. The proposed diagnostic module is intentionally the only local import.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
from numbers import Integral
import random

import numpy as np
import pytest

from scripts import diagnose_response_envelope_equivalence_v70 as diagnostic


def _clip(value, lower, upper):
    return min(upper, max(lower, value))


def _raw_patch(past, y, x, radius):
    """Reference extraction; past has the ordered shape [2, H, W, 3]."""
    padded = np.pad(past, ((0, 0), (radius, radius), (radius, radius), (0, 0)),
                    mode='edge')
    side = 2 * radius + 1
    return np.ascontiguousarray(padded[:, y:y + side, x:x + side, :])


def _reference_keys(past_pairs, radius):
    anchors, _, height, width, _ = past_pairs.shape
    keys = {}
    for a, y, x in np.ndindex(anchors, height, width):
        # Do not call the implementation's edge_key in this partition oracle.
        topology = (min(y, radius), min(height - 1 - y, radius),
                    min(x, radius), min(width - 1 - x, radius))
        keys[(a, y, x)] = (topology, _raw_patch(past_pairs[a], y, x, radius).tobytes())
    return keys


def _assert_partition(past_pairs, group_ids, representatives, radius):
    shape = (past_pairs.shape[0], past_pairs.shape[2], past_pairs.shape[3])
    assert group_ids.dtype == np.int32 and group_ids.shape == shape
    assert representatives.dtype == np.int32
    assert representatives.ndim == 2 and representatives.shape[1] == 3
    groups = len(representatives)
    assert set(group_ids.reshape(-1).tolist()) == set(range(groups))
    for group, coordinate in enumerate(representatives):
        index = tuple(int(value) for value in coordinate)
        assert all(0 <= value < limit for value, limit in zip(index, shape, strict=True))
        assert group_ids[index] == group
    keys = _reference_keys(past_pairs, radius)
    # Equality of partitions, not equality of incidental numeric group IDs.
    for left, right in product(keys, repeat=2):
        assert bool(group_ids[left] == group_ids[right]) == (keys[left] == keys[right])


_DILATIONS = (1, 2, 4, 8, 16)
_KERNEL = np.array([[0, 1, -1], [2, 3, 1], [-1, 0, 2]], dtype=np.int64)


def _full_integer_network(image):
    """Whole-image implementation: materialize each layer's replicated padding."""
    value = np.asarray(image, dtype=np.int64)
    height, width = value.shape
    for level, dilation in enumerate(_DILATIONS, 1):
        padded = np.pad(value, dilation, mode='edge')
        following = np.full_like(value, level)
        for ky, kx in np.ndindex(3, 3):
            following += (_KERNEL[ky, kx]
                          * padded[ky * dilation:ky * dilation + height,
                                   kx * dilation:kx * dilation + width])
        value = np.maximum(following, 0)
    return value


def _local_integer_graph(image, y, x):
    """Independent recursive graph using only a patch and the proposed key."""
    radius = sum(_DILATIONS)
    height, width = image.shape
    top, bottom, left, right = diagnostic.edge_key(y, x, height, width, radius)
    padded = np.pad(image, radius, mode='edge')
    patch = padded[y:y + 2 * radius + 1, x:x + 2 * radius + 1]

    @lru_cache(None)
    def node(level, dy, dx):
        if level == 0:
            return int(patch[radius + dy, radius + dx])
        total = level
        dilation = _DILATIONS[level - 1]
        for ky, kx in np.ndindex(3, 3):
            qy = _clip(dy + (ky - 1) * dilation, -top, bottom)
            qx = _clip(dx + (kx - 1) * dilation, -left, right)
            total += int(_KERNEL[ky, kx]) * node(level - 1, qy, qx)
        return max(total, 0)

    return node(len(_DILATIONS), 0, 0)


def _two_replicated_sums(row):
    value = list(row)
    for _ in range(2):
        value = [sum(value[_clip(x + offset, 0, len(value) - 1)]
                     for offset in (-1, 0, 1)) for x in range(len(value))]
    return value


def _brute_iou(positives, negatives):
    """Enumerate all binary group decisions; no ratio sorting in this oracle."""
    positives, negatives = list(map(int, positives)), list(map(int, negatives))
    actual = sum(positives)
    if actual == 0:
        return None, {(0, 0, 0)}
    best = Fraction(-1)
    optimal = set()
    for choices in product((0, 1), repeat=len(positives)):
        numerator = sum(p * take for p, take in zip(positives, choices, strict=True))
        denominator = actual + sum(n * take for n, take in zip(negatives, choices, strict=True))
        score = Fraction(numerator, denominator)
        witness = (numerator, denominator, sum(choices))
        if score > best:
            best, optimal = score, {witness}
        elif score == best:
            optimal.add(witness)
    return best, optimal


def _macro_fixture():
    group_ids = np.array([[[0, 0, 0, 0, 0]],
                          [[1, 2, 2, 2, 2]],
                          [[3, 3, 3, 3, 3]]], dtype=np.int32)
    targets = np.array([[[True, False, False, False, False]]] * 3, dtype=bool)
    cases = [dict(game='alpha', anchor_index=0),
             dict(game='alpha', anchor_index=1),
             dict(game='beta', anchor_index=0)]
    return group_ids, targets, cases


def test_clipped_edge_distances_preserve_every_reachable_clamp():
    radius = 31
    for length in range(1, 81):
        for position in range(length):
            vertical = diagnostic.edge_key(position, 0, length, 1, radius)
            horizontal = diagnostic.edge_key(0, position, 1, length, radius)
            distances = (min(position, radius), min(length - 1 - position, radius))
            assert vertical == (*distances, 0, 0)
            assert horizontal == (0, 0, *distances)
            for offset in range(-radius, radius + 1):
                real = _clip(position + offset, 0, length - 1) - position
                assert real == _clip(offset, -vertical[0], vertical[1])
                assert real == _clip(offset, -horizontal[2], horizontal[3])
    assert diagnostic.edge_key(31, 31, 100, 100) == (31, 31, 31, 31)
    assert diagnostic.edge_key(32, 32, 100, 100) == (31, 31, 31, 31)
    assert diagnostic.edge_key(30, 31, 100, 100) != (31, 31, 31, 31)


def test_five_successive_paddings_match_the_independent_local_graph():
    for height, width in ((1, 1), (2, 3), (9, 11), (67, 71)):
        image = ((np.arange(height * width, dtype=np.int64).reshape(height, width) * 7) % 11) - 5
        complete = _full_integer_network(image)
        positions = {(0, 0), (0, width - 1), (height - 1, 0),
                     (height - 1, width - 1), (height // 2, width // 2)}
        positions.update((y, x) for y, x in ((30, 31), (31, 31), (32, 32))
                         if y < height and x < width)
        for y, x in positions:
            assert _local_integer_graph(image, y, x) == int(complete[y, x])


def test_equal_raw_patches_with_different_topology_can_produce_13_and_12():
    first, second = [1, 2, 2, 2, 2], [1, 1, 1, 2, 2]
    past = np.empty((2, 2, 1, 5, 3), dtype=np.uint8)
    past[0] = np.array(first, dtype=np.uint8)[None, None, :, None]
    past[1] = np.array(second, dtype=np.uint8)[None, None, :, None]
    assert np.array_equal(_raw_patch(past[0], 0, 0, 31),
                          _raw_patch(past[1], 0, 2, 31))
    assert _two_replicated_sums(first)[0] == 13
    assert _two_replicated_sums(second)[2] == 12
    assert diagnostic.edge_key(0, 0, 1, 5) != diagnostic.edge_key(0, 2, 1, 5)
    groups, representatives, _ = diagnostic.partition_patches(past)
    assert groups[0, 0, 0] != groups[1, 0, 2]
    _assert_partition(past, groups, representatives, 31)


def test_partition_equals_full_patch_and_topology_equality_without_mutation():
    rng = np.random.default_rng(1701)
    first = rng.integers(0, 3, size=(2, 4, 5, 3), dtype=np.uint8)
    changed = first.copy()
    changed[1, 2, 3, 0] += 7
    past = np.stack((first, changed, first))
    before = past.copy()
    groups, representatives, collision_groups = diagnostic.partition_patches(past, radius=1)
    _assert_partition(past, groups, representatives, 1)
    assert np.array_equal(groups[0], groups[2])
    assert collision_groups == 0
    assert np.array_equal(past, before)


def test_partition_preserves_the_order_of_the_two_rgb_images():
    past = np.zeros((3, 2, 1, 1, 3), dtype=np.uint8)
    past[0, 0, 0, 0] = [1, 2, 3]
    past[0, 1, 0, 0] = [4, 5, 6]
    past[1] = past[0, ::-1]
    past[2] = past[0]
    groups, representatives, _ = diagnostic.partition_patches(past, radius=1)
    assert groups[0, 0, 0] == groups[2, 0, 0]
    assert groups[0, 0, 0] != groups[1, 0, 0]
    _assert_partition(past, groups, representatives, 1)


def test_forced_digest_collisions_are_resolved_by_full_comparison():
    past = np.zeros((4, 2, 1, 1, 3), dtype=np.uint8)
    for index, value in enumerate((1, 2, 1, 3)):
        past[index] = value
    groups, representatives, collisions = diagnostic.partition_patches(
        past, radius=1, digest=lambda data: '0' * 64)
    _assert_partition(past, groups, representatives, 1)
    assert len(representatives) == 3
    # All keys have the same topology and digest: two additional unequal groups.
    assert collisions == 2


def test_anchor_iou_bound_matches_all_binary_subsets_not_a_greedy_oracle():
    rng = random.Random(9150)
    examples = [([2, 1, 0], [0, 2, 0]), ([1, 1], [4, 0]),
                ([0, 0], [0, 3]), ([2, 4, 1], [2, 4, 1])]
    for _ in range(100):
        size = rng.randrange(1, 8)
        examples.append(([rng.randrange(5) for _ in range(size)],
                         [rng.randrange(6) for _ in range(size)]))
    for positives, negatives in examples:
        expected, witnesses = _brute_iou(positives, negatives)
        result = diagnostic.anchor_iou_bound(np.array(positives, dtype=np.int64),
                                             np.array(negatives, dtype=np.int64))
        values = tuple(result[key] for key in ('numerator', 'denominator', 'selected_groups'))
        assert all(isinstance(value, Integral) and not isinstance(value, bool) for value in values)
        assert values in witnesses
        if expected is None:
            assert result['iou'] is None
        else:
            assert result['iou'] == pytest.approx(float(expected), abs=1e-15)


def test_empty_anchor_has_no_iou_and_is_not_a_perfect_positive_case():
    result = diagnostic.anchor_iou_bound(np.zeros(3, dtype=np.int64),
                                         np.array([1, 4, 2], dtype=np.int64))
    assert result['iou'] is None
    assert result['numerator'] == result['denominator'] == result['selected_groups'] == 0


def test_global_bce_uses_anchor_balancing_and_the_empty_anchor_weight():
    groups = np.array([[[0, 1, 1, 1]], [[0, 0, 2, 2]]], dtype=np.int32)
    targets = np.array([[[True, False, False, False]], [[False] * 4]], dtype=bool)
    cases = [dict(game='positive', anchor_index=0), dict(game='empty', anchor_index=0)]
    result = diagnostic.summarize_partition(groups, targets, cases)
    # Group 0: positive mass 1/(2*2*1)=1/4, negative mass 2/(2*4)=1/4.
    # Groups 1 and 2 are pure negative and have infimum zero.
    assert result['balanced_bce_lower_bound'] == pytest.approx(0.5 * math.log(2), abs=1e-12)
    assert result['cases'][1]['iou_upper_bound'] is None
    assert result['cases'][1]['target_pixels'] == 0
    assert 'empty' not in result['iou_upper_bound_by_game']


def test_bce_groups_are_global_even_when_independent_anchor_iou_is_perfect():
    groups = np.zeros((2, 1, 4), dtype=np.int32)
    targets = np.array([[[True] * 4], [[False] * 4]], dtype=bool)
    cases = [dict(game='one', anchor_index=0), dict(game='one', anchor_index=1)]
    result = diagnostic.summarize_partition(groups, targets, cases)
    # Both single-class anchors have total weight 1/2 in the same global group.
    assert result['balanced_bce_lower_bound'] == pytest.approx(math.log(2), abs=1e-12)
    # This upper bound relaxes cross-anchor ties; it does NOT certify specificity.
    assert result['macro_nonempty_iou_upper_bound'] == pytest.approx(1.0)
    assert result['cases'][1]['iou_upper_bound'] is None


def test_iou_means_are_per_anchor_then_game_and_not_micro_aggregation():
    groups, targets, cases = _macro_fixture()
    result = diagnostic.summarize_partition(groups, targets, cases)
    assert [row['iou_upper_bound'] for row in result['cases']] == pytest.approx([0.2, 1.0, 0.2])
    assert result['iou_upper_bound_by_game'] == pytest.approx({'alpha': 0.6, 'beta': 0.2})
    assert result['macro_nonempty_iou_upper_bound'] == pytest.approx(0.4)
    assert result['macro_nonempty_iou_upper_bound'] != pytest.approx((0.2 + 1 + 0.2) / 3)
    # Pooling alpha's pixels before optimization gives 0.5, below its true
    # anchor-mean upper bound 0.6: that micro result is not a safe replacement.
    micro, _ = _brute_iou([1, 1, 0], [4, 0, 4])
    assert micro == Fraction(1, 2)
    assert float(micro) < result['iou_upper_bound_by_game']['alpha']
    for row, case in zip(result['cases'], cases, strict=True):
        assert (row['game'], row['anchor_index']) == (case['game'], case['anchor_index'])
        assert row['target_pixels'] == 1
        assert row['iou_upper_bound'] == pytest.approx(row['bound_numerator'] / row['bound_denominator'])


def test_summary_is_invariant_to_group_id_renumbering():
    groups, targets, cases = _macro_fixture()
    permuted = np.array([3, 2, 0, 1], dtype=np.int32)[groups]
    first = diagnostic.summarize_partition(groups, targets, cases)
    second = diagnostic.summarize_partition(permuted, targets, cases)
    assert second['balanced_bce_lower_bound'] == pytest.approx(first['balanced_bce_lower_bound'], abs=1e-12)
    assert second['macro_nonempty_iou_upper_bound'] == pytest.approx(first['macro_nonempty_iou_upper_bound'])
    assert second['iou_upper_bound_by_game'] == pytest.approx(first['iou_upper_bound_by_game'])
    assert [row['iou_upper_bound'] for row in second['cases']] == pytest.approx(
        [row['iou_upper_bound'] for row in first['cases']])
