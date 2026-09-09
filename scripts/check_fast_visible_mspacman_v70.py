"""Offline ordered source-label equivalence, with no emulator or new seed.

All two-source temporal DEV/RESERVED windows and the fresh raw reference are
fixed inputs. This is neither another calibration nor a learning/gate result.
"""
import hashlib
import json
import time
import numpy as np
from ragnarok.evaluation.fast_visible_mspacman_v70 import source_ram_objects as candidate
from ragnarok.evaluation.pooled_mspacman_eyes_v70 import source_ram_objects as frozen
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.triple_pooled_fixtures_v70 import triple_pooled_fixture_frames
from scripts import visible_reference_v70 as reference
from scripts.temporal_validation_seal_v70 import ROOT, sha, utc, write_new, require_committed

SOURCE_PATHS = tuple(dict.fromkeys((*reference.SOURCE_PATHS,
    'ragnarok/evaluation/fast_visible_mspacman_v70.py',
    'tests/test_fast_visible_mspacman_v70.py', 'scripts/check_fast_visible_mspacman_v70.py')))
BASE = ROOT/'craft_v6_out/v70_mspacman_calibration'
COLLECTIONS = (('development', BASE/'pooled_development_v0', 'temporal'),
               ('reserved', BASE/'pooled_reserved_v2', 'temporal'),
               ('fresh_raw', BASE/'raw_reference_v0', 'raw'))
OUTPUT = ROOT/'craft_v6_out/v70_visible_cpu/fast_mspacman_equivalence_v0.json'


def hashes():
    sources = {p: sha(ROOT/p) for p in SOURCE_PATHS}
    evidence = {}
    for _, directory, _ in COLLECTIONS:
        manifest = directory/'manifest.json'
        for path in [manifest, *(directory/s['path'] for s in json.loads(manifest.read_text(encoding='utf-8'))['shards'])]:
            evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(source_sha256=sources, evidence_sha256=evidence)


def run():
    attempt = OUTPUT.with_name(OUTPUT.stem+'_attempt.json')
    if OUTPUT.exists() or attempt.exists():
        raise ValueError('Preserve earlier equivalence results/attempts; no overwrite')
    proof = hashes()
    for mapping in proof.values():
        for path in mapping:
            require_committed(path)
    record = dict(status='started', started_utc=utc(), **proof, new_emulated_frames=0)
    write_new(attempt, record)
    collections = {}
    try:
        for name, directory, kind in COLLECTIONS:
            counts = dict(windows=0, source_pairs=0, labels=0)
            frozen_seconds = candidate_seconds = 0.0
            digest = hashlib.sha256()
            loader = triple_pooled_fixture_frames if kind == 'temporal' else raw_history_fixture_frames
            for row in loader(directory):
                if kind == 'temporal':
                    pairs = [(row['ram_previous'], row['ram_before_previous']),
                             (row['ram_current'], row['ram_previous'])]
                else:
                    pairs = [(row['ram'], row['ram_previous'])]
                    if int(row['frame_in_episode']) == 1:
                        pairs.insert(0, (row['ram_previous'], row['ram_previous']))
                for source, (current, previous) in enumerate(pairs):
                    copies = current.copy(), previous.copy()
                    tick = time.perf_counter()
                    expected = frozen(current, previous)
                    frozen_seconds += time.perf_counter()-tick
                    if not np.array_equal(current, copies[0]) or not np.array_equal(previous, copies[1]):
                        raise ValueError('Frozen decoder changed archived RAM')
                    tick = time.perf_counter()
                    actual = candidate(current, previous)
                    candidate_seconds += time.perf_counter()-tick
                    if not np.array_equal(current, copies[0]) or not np.array_equal(previous, copies[1]):
                        raise ValueError('Candidate changed archived RAM')
                    if actual != expected:
                        record['first_mismatch'] = dict(collection=name, index=row['index'], source=source,
                            expected=[(o.kind, o.box, o.slot) for o in expected],
                            actual=[(o.kind, o.box, o.slot) for o in actual])
                        raise ValueError(f'Ordered source labels differ: {name}/{row["index"]}/{source}')
                    digest.update(json.dumps([row['index'], source,
                        [(o.kind, o.box, o.slot) for o in actual]], separators=(',', ':')).encode()+b'\n')
                    counts['source_pairs'] += 1
                    counts['labels'] += len(actual)
                counts['windows'] += 1
            if counts['windows'] != 8192:
                raise ValueError('Every fixed archived window must be checked')
            collections[name] = dict(**counts, ordered_label_sha256=digest.hexdigest(),
                frozen_decode_seconds=frozen_seconds, candidate_decode_seconds=candidate_seconds)
            print(json.dumps(dict(collection=name, **collections[name])), flush=True)
        if hashes() != proof:
            raise ValueError('Sources or evidence changed during equivalence check')
        result = dict(kind='offline_ordered_mspacman_source_equivalence', passed=True,
            created_utc=utc(), **proof, collections=collections, new_emulated_frames=0,
            agent_training_allowed=False, gate_result=None,
            limitations=['Same frozen source labels, not a new independent RAM/RGB calibration.',
                'Timing measures only decoder calls on archived RAM, frozen then candidate per pair.',
                'No emulator, RGB reference, IPC, branches or gate throughput measured here.'])
        write_new(OUTPUT, result)
        record.update(status='complete', completed_utc=utc(), result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed', failed_utc=utc(), error_type=type(error).__name__, error=str(error),
                      completed_collections=collections)
        raise
    finally:
        attempt.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
