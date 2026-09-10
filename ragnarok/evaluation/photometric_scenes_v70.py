"""Declared procedural scenes; physical identities belong exclusively to the judge.

The learner must read only the separately written RGB arrays. This generator is
never imported by its fitting process. Sampling does not depend on a model.
"""
import numpy as np


STRATA = ('visible_moving', 'visible_stationary', 'occluded', 'out_of_frame', 'ambiguous')
SIDE = 8


def _positions(rng, count, excluded=()):
    candidates = np.setdiff1d(np.arange(SIDE*SIDE), np.asarray(excluded, dtype=int))
    flat = rng.choice(candidates, size=count, replace=False)
    return np.stack((flat % SIDE, flat // SIDE), axis=-1)


def _flat(position):
    x, y = position
    if not (0 <= x < SIDE and 0 <= y < SIDE):
        return None
    return int(y*SIDE+x)


def render(base_colors, exposures, positions_xy):
    """Float64 exposure, ties-to-even quantization, fixed opaque layer order."""
    colors = np.asarray(base_colors, dtype=np.float64)
    exposures = np.asarray(exposures, dtype=np.float64)
    positions = np.asarray(positions_xy, dtype=int)
    if colors.shape != (5, 3) or exposures.shape != (3,) or positions.shape != (3, 4, 2):
        raise ValueError('Five RGB colors, three exposures and three sets of four xy points required')
    quantized = np.rint(exposures[:, None, None] * colors[None])
    if not np.isfinite(quantized).all() or quantized.min() < 0 or quantized.max() > 255:
        raise ValueError('The renderer must not clip colors')
    quantized = quantized.astype(np.uint8)
    frames = np.empty((3, SIDE, SIDE, 3), dtype=np.uint8)
    for frame_index in range(3):
        frames[frame_index] = quantized[frame_index, 0]
        for actor, (x, y) in enumerate(positions[frame_index]):
            if 0 <= x < SIDE and 0 <= y < SIDE:
                frames[frame_index, y, x] = quantized[frame_index, actor+1]
    return frames, quantized


def scene(rng, *, stratum, scene_id):
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError('An explicit PCG64 stream is required')
    if stratum not in STRATA or not isinstance(scene_id, str) or not scene_id:
        raise ValueError('Known stratum and nonempty scene ID required')
    codes = rng.choice(96**3, size=5, replace=False)
    colors = np.stack((codes // 96**2, codes // 96 % 96, codes % 96), axis=-1) + 32
    if stratum == 'ambiguous':
        colors[2] = colors[1]
    exposures = rng.uniform(0.65, 1.75, size=3)
    positions = np.empty((3, 4, 2), dtype=int)
    positions[0] = _positions(rng, 4)
    permutations = []
    for t in (1, 2):
        if stratum == 'ambiguous':
            swapped = bool(rng.integers(0, 2))
            permutations.append(swapped)
            positions[t, :2] = positions[0, [1, 0] if swapped else [0, 1]]
            positions[t, 2:] = _positions(rng, 2, [_flat(p) for p in positions[0, :2]])
        elif stratum == 'occluded' and t == 2:
            positions[t, 1] = _positions(rng, 1)[0]
            positions[t, 0] = positions[t, 1]
            positions[t, 2:] = _positions(rng, 2, [_flat(positions[t, 1])])
        elif stratum == 'out_of_frame' and t == 2:
            positions[t, 0] = (SIDE, int(rng.integers(0, SIDE)))
            positions[t, 1:] = _positions(rng, 3)
        else:
            if stratum == 'visible_stationary':
                positions[t, 0] = positions[0, 0]
            else:
                positions[t, 0] = _positions(rng, 1, [_flat(p) for p in positions[:t, 0]])[0]
            positions[t, 1:] = _positions(rng, 3, [_flat(positions[t, 0])])
    rgb, quantized = render(colors, exposures, positions)
    visible = [True, True, stratum not in ('occluded', 'out_of_frame')]
    target = _flat(positions[2, 0]) if stratum in STRATA[:2] else None
    metadata = dict(scene_id=scene_id, stratum=stratum, source_pixel=_flat(positions[0, 0]),
                    target_pixel=target, query_visible=visible,
                    positions_xy=positions.tolist(), base_colors=colors.tolist(),
                    exposures=exposures.tolist(), quantized_colors=quantized.tolist(),
                    ambiguity_swaps_relative_initial=permutations,
                    # Palette coincidences are reported even if the query is hidden;
                    # query_visible prevents claiming observation of that hidden RGB.
                    query_palette_coincidences=[
                        [other for other in (0, 2, 3, 4)
                         if np.array_equal(quantized[t, 1], quantized[t, other])]
                        for t in range(3)])
    return rgb, metadata


def training_table():
    """Production seed: called once, only inside the bounded generation stage."""
    rng = np.random.Generator(np.random.PCG64(19492))
    pairs = [scene(rng, stratum=STRATA[int(rng.integers(0, 2))],
                   scene_id=f'train_{index:04d}') for index in range(128)]
    return np.stack([p[0] for p in pairs]), [p[1] for p in pairs]


def evaluation_table():
    """Five independent spawned streams; no candidate-dependent rejection."""
    streams = np.random.SeedSequence(19493).spawn(5)
    pairs = []
    for stratum, seed in zip(STRATA, streams, strict=True):
        rng = np.random.Generator(np.random.PCG64(seed))
        pairs.extend(scene(rng, stratum=stratum, scene_id=f'{stratum}_{index:04d}')
                     for index in range(256))
    return np.stack([p[0] for p in pairs]), [p[1] for p in pairs]
