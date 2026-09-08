"""Check optimized modal counting against independent per-pixel Python histories."""
from collections import Counter

import numpy as np
import pytest

from scripts.ref_extract_v70 import ReferenceExtractor


def test_mode_matches_independent_histories_through_palette_growth_and_ties():
    rng = np.random.default_rng(8061)
    palette = rng.integers(0, 256, (39, 3), dtype=np.uint8)
    ref = ReferenceExtractor(max_colours=39)
    histories = [Counter() for _ in range(35)]
    incumbents = [None] * 35
    for t in range(100):
        # Palette grows across two reallocations; every pixel has its own history.
        rgb = palette[rng.integers(0, min(39, 1 + t), (5, 7))]
        for i, colour in enumerate(map(tuple, rgb.reshape(-1, 3).tolist())):
            history = histories[i]
            history[colour] += 1
            if incumbents[i] is None or history[colour] > history[incumbents[i]]:
                incumbents[i] = colour
        ref.observe(rgb)
        expected = np.array(incumbents, dtype=np.uint8).reshape(5, 7, 3)
        assert np.array_equal(ref._mode, ref._packed(expected))
        assert ref._best.ravel().tolist() == [max(h.values()) for h in histories]
        assert ref.frames_seen == t + 1


def test_capacity_rejection_is_atomic_and_queries_do_not_register_colours():
    ref = ReferenceExtractor(max_colours=1)
    black = np.zeros((2, 3, 3), dtype=np.uint8)
    white = black + 255
    ref.update(black)
    old_counts = ref._counts.copy()
    for _ in range(3):
        assert ref.extract(white).tolist() == [[0, 0, 3, 2]]
    with pytest.raises(ValueError, match="capacity"):
        ref.update(white)
    assert np.array_equal(old_counts, ref._counts)
    assert ref.frames_seen == 1 and ref._palette_size == 1
    assert ref._colour_ids[0xFFFFFF] == 0
    assert ref.extract(black).shape == (0, 4)
    ref.reset()
    assert ref._colour_ids is None and ref._counts is None
    assert ref.observe(white).shape == (0, 4)


def test_maximum_palette_and_large_pixel_offsets_do_not_wrap():
    codes = np.arange(4096, dtype=np.uint32).reshape(64, 64)
    rgb = np.stack((codes & 255, codes >> 8, codes * 0), axis=-1).astype(np.uint8)
    ref = ReferenceExtractor(max_colours=4096)
    ref.update(rgb)
    ref.update(np.roll(rgb, 1, axis=1))
    assert np.array_equal(ref._mode, ref._packed(rgb))
    assert np.all(ref._best == 1)
    ref.update(np.roll(rgb, 1, axis=1))
    assert np.array_equal(ref._mode, ref._packed(np.roll(rgb, 1, axis=1)))
    assert np.all(ref._best == 2)


def test_counter_limit_and_resolution_failure_preserve_state():
    ref = ReferenceExtractor()
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    ref.update(rgb)
    with pytest.raises(ValueError, match="Resolution"):
        ref.update(rgb[:, :3])
    assert ref.frames_seen == 1 and np.all(ref._best == 1)
    ref.frames_seen = 1_000_000
    with pytest.raises(ValueError, match="limit"):
        ref.update(rgb)
    assert np.all(ref._best == 1)
