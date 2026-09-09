"""Strict offline two-RGB/three-RAM windows; no agent observation interface."""
import hashlib
import json
from pathlib import Path

import numpy as np
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS, CONVENTION

KIND = 'scorer_only_cold_two_rgb_three_ram_fixture'
RGB_KEYS = ('raw_previous', 'raw_current', 'pooled')
RAM_KEYS = ('ram_before_previous', 'ram_previous', 'ram_current')
INDEX_KEYS = ('episode', 'frame_in_episode', 'charged_frame', 'action')
KEYS = (*RGB_KEYS, *RAM_KEYS, *INDEX_KEYS, 'ram_frame_numbers', 'cold_history_fallback')


def triple_pooled_fixture_frames(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('kind') != KIND or manifest.get('game') not in ROM_SPECS
            or manifest.get('reset_convention') != CONVENTION
            or manifest.get('mode') != 0 or manifest.get('action_repeat') != 4
            or any(manifest.get(k) is not False for k in ('agent_training_allowed',
                'reference_training_allowed', 'agent_entry_states_allowed'))):
        raise ValueError('Expected scorer-only cold two-RGB/three-RAM windows')
    bootstrap = ROM_SPECS[manifest['game']][1]
    last_episode, last_local, last_count = 0, 0, bootstrap
    total = ordinary = 0
    seen = set()
    previous = None
    for shard in manifest['shards']:
        path = directory/shard['path']
        resolved = path.resolve()
        if resolved.parent != directory.resolve() or resolved in seen:
            raise ValueError('Shard must be unique and inside the fixture directory')
        seen.add(resolved)
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError('Triple fixture hash mismatch')
        n = shard['decisions']
        if type(n) is not int or not 1 <= n <= 128:
            raise ValueError('Invalid shard length')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(KEYS):
                raise ValueError('Unexpected or missing fixture fields')
            data = {k: archive[k] for k in KEYS}
        for keys, shape, dtype in ((RGB_KEYS, (n,210,160,3), np.uint8),
                                  (RAM_KEYS, (n,128), np.uint8),
                                  (INDEX_KEYS, (n,), np.int64),
                                  (('ram_frame_numbers',), (n,3), np.int64),
                                  (('cold_history_fallback',), (n,), np.bool_)):
            for key in keys:
                if data[key].shape != shape or data[key].dtype != dtype:
                    raise ValueError(f'Malformed triple fixture field: {key}')
        for i in range(n):
            row = {k:data[k][i] for k in KEYS}
            episode, local, count, action = (int(row[k]) for k in INDEX_KEYS)
            if (episode not in (last_episode, last_episode+1) or (total == 0 and episode != 0)
                    or not 0 <= action <= 17):
                raise ValueError('Invalid episode sequence or button')
            reset = episode != last_episode
            advanced = local if reset else local-last_local
            if not 1 <= advanced <= 4 or count-last_count != advanced+bootstrap*reset:
                raise ValueError('Discontinuous frames or uncharged cold load')
            if not np.array_equal(row['ram_frame_numbers'], (max(0,local-2),max(0,local-1),local)):
                raise ValueError('RAM history is not adjacent to the source images')
            if bool(row['cold_history_fallback']) != (local < 2):
                raise ValueError('Undeclared cold history fallback')
            if local == 1 and not np.array_equal(row['ram_before_previous'], row['ram_previous']):
                raise ValueError('Cold fallback must duplicate the initial RAM')
            if not np.array_equal(row['pooled'], np.maximum(row['raw_previous'], row['raw_current'])):
                raise ValueError('Saved observation is not the actual two-frame max')
            if previous is not None and not reset:
                links = []
                if advanced == 1:
                    links = [('ram_before_previous','ram_previous'), ('ram_previous','ram_current'),
                             ('raw_previous','raw_current')]
                elif advanced == 2:
                    links = [('ram_before_previous','ram_current')]
                if any(not np.array_equal(row[a], previous[b]) for a,b in links):
                    raise ValueError('Overlapping window history disagrees')
            total += 1
            ordinary += advanced
            yield dict(index=total, **row)
            previous = row
            last_episode, last_local, last_count = episode, local, count
    account = manifest['accounting']
    if (total != manifest['decisions'] or ordinary != manifest['ordinary_real_frames']
            or last_count != account['charged_frames'] or account['cold_loads'] != last_episode+1
            or manifest['episode_count'] != last_episode+1 or account['restores'] != 0
            or account['experiment_frames'] != 0 or last_count != ordinary+bootstrap*(last_episode+1)):
        raise ValueError('Final fixture counts and complete ordinary/cold ledger disagree')
