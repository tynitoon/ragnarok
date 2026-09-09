"""Scorer-only development pixel inventory; never starts an emulator."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.evaluation.cold_fixtures_v70 import cold_fixture_frames

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'craft_v6_out/v70_asteroids_calibration'


def main():
    output = BASE / 'development_player_pixels_v0.npz'
    receipt = output.with_suffix('.json')
    if output.exists() or receipt.exists():
        raise ValueError('Preserve existing derived inventories')
    colours = {'red': (240, 128, 128), 'blue': (117, 181, 239)}
    pixels = {key: [] for key in colours}
    rams, episodes = [], []
    for index, (rgb, ram, episode, _) in enumerate(cold_fixture_frames(BASE / 'development')):
        rams.append(ram)
        episodes.append(episode)
        for key, colour in colours.items():
            y, x = np.nonzero(np.all(rgb[16:210] == colour, axis=2))
            pixels[key].append(np.column_stack((np.full(len(x), index), x, y + 16)))
    inventory = np.load(BASE / 'development_palette_inventory_v0.npz', allow_pickle=False)
    assert np.array_equal(np.stack(rams), inventory['ram'])
    assert np.array_equal(episodes, inventory['episode'])
    with output.open('xb') as f:
        np.savez_compressed(f, **{key: np.concatenate(rows).astype(np.int32)
                                 for key, rows in pixels.items()})
    source = Path(__file__).relative_to(ROOT).as_posix()
    data = dict(kind='asteroids_player_pixel_development_inventory', seed=17511,
                frames=index + 1, columns=['zero_based_frame', 'x', 'y'],
                colours=colours, roi=[0, 16, 160, 194],
                payload_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                palette_inventory_sha256=hashlib.sha256((BASE / 'development_palette_inventory_v0.npz').read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((BASE / 'development/manifest.json').read_bytes()).hexdigest(),
                source_sha256={source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()},
                provenance='Every archived development frame; strict reader verifies every shard and complete accounting. RAM and episodes equal the original palette inventory.',
                agent_training_allowed=False, gate_result=None)
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(json.dumps(dict(frames=index + 1, pixel_counts={k: sum(map(len, v)) for k, v in pixels.items()})))


if __name__ == '__main__':
    main()
