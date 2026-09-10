"""Bounded prospective fixtures; no production seeds, optimizer or performance score."""
import math

import numpy as np
import pytest
import torch

from ragnarok.evaluation.photometric_scenes_v70 import STRATA, render, scene
from ragnarok.learning.photometric_correspondence_v70 import (
    PixelAppearanceEncoder, associations_from_features, cycle_loss, encode_clips,
    propose, rgb_reference, transitions,
)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def test_uniform_appearance_has_irreducible_loss_and_no_admission():
    features = torch.zeros(1, 3, 4, 2)
    features[..., 0] = 1
    assert cycle_loss(features).item() == pytest.approx(math.log(4), abs=1e-6)
    result = associations_from_features(features, height=2, width=2)
    assert not result.admitted.any()
    assert (result.destination == -1).all()
    assert (result.best_index == 0).all()  # Exact tie: first index.
    assert torch.allclose(result.round_trip_score, torch.full((1, 2, 2), 1/16))


def test_cycle_tracks_a_permutation_with_returning_indices():
    # Positions are changed, but each of three appearances stays distinct.
    first = torch.eye(3)
    middle = first[[2, 0, 1]]
    last = first[[1, 2, 0]]
    features = torch.stack((first, middle, last))[None]
    result = associations_from_features(features, height=1, width=3)
    assert result.destination.tolist() == [[[2, 0, 1]]]
    assert result.admitted.all()
    assert (result.round_trip_score > 0.99).all()
    assert (result.endpoint_similarity == 1).all()
    assert cycle_loss(features).item() < 0.001
    for matrix in transitions(features):
        assert torch.allclose(matrix.sum(dim=-1), torch.ones(1, 3))


def test_duplicate_source_identities_cannot_be_resolved_by_a_cycle():
    first = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    features = torch.stack((first, first[[2, 0, 1]], first[[1, 2, 0]]))[None]
    result = associations_from_features(features, height=1, width=3)
    assert not result.admitted[0, 0, :2].any()
    assert (result.round_trip_score[0, 0, :2] <= 0.25001).all()
    assert result.admitted[0, 0, 2]


def test_encoder_shared_over_positions_and_one_finite_backward():
    torch.manual_seed(55103)
    encoder = PixelAppearanceEncoder()
    assert sum(p.numel() for p in encoder.parameters()) == 1712
    clips = torch.tensor(np.random.default_rng(55104).integers(21, 223, (1, 3, 2, 2, 3), dtype=np.uint8))
    features = encode_clips(encoder, clips)
    mirrored = encode_clips(encoder, clips.flip(3))
    assert torch.equal(mirrored, features.reshape(1, 3, 2, 2, 16).flip(3).reshape(1, 3, 4, 16))
    assert torch.isfinite(features).all()
    assert torch.allclose(features.norm(dim=-1), torch.ones(1, 3, 4), atol=1e-6)
    loss = cycle_loss(features)
    loss.backward()  # Exactly one backward in this entire fixture module.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in encoder.parameters()) > 0


def test_pooled_guard_never_calls_an_encoder_or_makes_up_scores():
    result = propose(object(), torch.zeros(2, 3, 2, 2, 3, dtype=torch.uint8), image_kind='pooled')
    assert result.computed is False
    assert result.reason == 'temporal_support_not_point_observation'
    assert not result.admitted.any() and (result.destination == -1).all()
    assert result.best_index is result.round_trip_score is result.endpoint_similarity is None


@pytest.mark.parametrize('kind', [None, '', 'mixed', ['raw', 'pooled'], np.array(['raw', 'pooled'])])
def test_missing_or_mixed_time_support_is_rejected(kind):
    with pytest.raises(ValueError):
        propose(None, torch.zeros(1, 3, 2, 2, 3, dtype=torch.uint8), image_kind=kind)


def test_missing_kind_has_no_default_and_black_reference_is_zero():
    clips = torch.zeros(1, 3, 2, 2, 3, dtype=torch.uint8)
    with pytest.raises(TypeError):
        propose(None, clips)
    assert torch.equal(rgb_reference(clips), torch.zeros(1, 3, 4, 3))
    result = propose(None, clips, image_kind='raw')
    assert not result.admitted.any() and torch.isfinite(result.round_trip_score).all()


@pytest.mark.parametrize('shape,dtype', [((1, 2, 2, 2, 3), torch.uint8),
                                       ((1, 3, 2, 2, 3), torch.float32),
                                       ((17, 3, 2, 2, 3), torch.uint8),
                                       ((1, 3, 17, 17, 3), torch.uint8)])
def test_input_bounds_fail_before_computation(shape, dtype):
    with pytest.raises(ValueError):
        propose(None, torch.zeros(shape, dtype=dtype), image_kind='raw')


@pytest.mark.parametrize('stratum', STRATA)
def test_generator_physical_law_and_rendering(stratum):
    rng = np.random.Generator(np.random.PCG64(55105))
    rgb, metadata = scene(rng, stratum=stratum, scene_id='fixture')
    rgb_again, metadata_again = scene(np.random.Generator(np.random.PCG64(55105)),
                                     stratum=stratum, scene_id='fixture')
    assert np.array_equal(rgb, rgb_again) and metadata == metadata_again
    assert rgb.shape == (3, 8, 8, 3) and rgb.dtype == np.uint8
    assert rgb.min() >= 21 and rgb.max() <= 223
    positions = np.asarray(metadata['positions_xy'])
    colors = np.asarray(metadata['quantized_colors'])
    assert len(set(map(tuple, positions[0]))) == 4
    assert metadata['source_pixel'] == positions[0, 0, 1]*8+positions[0, 0, 0]
    assert all(0.65 <= e < 1.75 for e in metadata['exposures'])
    for t in range(3):
        expected = np.broadcast_to(colors[t, 0], (8, 8, 3)).copy()
        for actor, (x, y) in enumerate(positions[t]):
            if 0 <= x < 8 and 0 <= y < 8:
                expected[y, x] = colors[t, actor+1]
        assert np.array_equal(rgb[t], expected)
    if stratum == 'visible_moving':
        assert len(set(map(tuple, positions[:, 0]))) == 3
    if stratum == 'visible_stationary':
        assert (positions[:, 0] == positions[0, 0]).all()
    if stratum == 'occluded':
        assert np.array_equal(positions[2, 0], positions[2, 1])
        x, y = positions[2, 0]
        assert np.array_equal(rgb[2, y, x], colors[2, 2])
    if stratum == 'out_of_frame':
        assert positions[2, 0, 0] == 8
        assert metadata['target_pixel'] is None  # x=8 must not wrap to the next row.
    if stratum == 'ambiguous':
        assert np.array_equal(colors[:, 1], colors[:, 2])
        for t, swapped in enumerate(metadata['ambiguity_swaps_relative_initial'], start=1):
            assert np.array_equal(positions[t, :2], positions[0, [1, 0] if swapped else [0, 1]])
        opposite = positions.copy()
        opposite[1:, :2] = opposite[1:, [1, 0]]
        opposite_rgb, _ = render(metadata['base_colors'], metadata['exposures'], opposite)
        assert np.array_equal(rgb, opposite_rgb)
    if stratum in STRATA[:2]:
        assert metadata['target_pixel'] == positions[2, 0, 1]*8+positions[2, 0, 0]
        assert all(len(set(map(tuple, p))) == 4 for p in positions)
    else:
        assert metadata['target_pixel'] is None


def test_renderer_keeps_quantized_coincidences_and_ties_to_even():
    colors = [[32, 34, 36], [33, 35, 37], [34, 36, 38], [50, 60, 70], [80, 90, 100]]
    positions = [[[0, 0], [1, 0], [2, 0], [3, 0]]] * 3
    rgb, palette = render(colors, [1.5, 0.65, 1.0], positions)
    assert palette[0, 1].tolist() == [50, 52, 56]
    # Color equality is allowed and retained, not rejected by the renderer.
    same_colors = [colors[0]] * 5
    same, _ = render(same_colors, [1.0]*3, positions)
    assert np.array_equal(same, np.broadcast_to(colors[0], same.shape))
