"""One prospective external grouping comparison on two existing raw archives.

No emulator, new image, agent input, training or full-chain benchmark.
"""
from collections import deque
import hashlib
import json
import os
import sys
from time import perf_counter

import numpy as np

from ragnarok.evaluation.fast_visible_mspacman_v70 import mspacman_visible_scene
from ragnarok.evaluation.visible_scene_v70 import asteroids_visible_scene
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.visible_temporal_groups_v70 import group_visible_sources
from ragnarok.evaluation.rectangle_visible_groups_v70 import rectangle_visible_groups
from scripts import check_fast_visible_mspacman_v70 as prior
from scripts.temporal_validation_seal_v70 import ROOT, sha, utc, runtime, write_new, require_committed

PROTOCOL = 'docs/ARC4_RECTANGLE_GROUPS_OFFLINE_PROTOCOL.json'
FOLDER = ROOT / 'craft_v6_out/v70_rectangle_groups'
SOURCE_REVIEW = FOLDER / 'source_review_v0.json'
PREFLIGHT = FOLDER / 'preflight_v0.json'
ATTEMPT = FOLDER / 'offline_v0_attempt.json'
OUTPUT = FOLDER / 'offline_v0.json'
EARLIER_REVIEW = 'craft_v6_out/v70_visible_cpu/fast_mspacman_equivalence_review_v0.json'
EARLIER_REVIEW_SHA = '945c47fc3e4b83d8063339f02c9968a30b3fc53bed404e3b4f6e92ca633578e4'
COLLECTIONS = (
    ('ms_pacman', 'craft_v6_out/v70_mspacman_calibration/raw_reference_v0', 17471,
     '115ddbf1d72ba2634b943ffa6c46e42d6d3202ca9cf5abed92128a20bfa149f7'),
    ('asteroids', 'craft_v6_out/v70_asteroids_calibration/raw_reference_v0', 17571,
     '80da5c106803fd27ec9778afb76e61b44f37771b92f0829086d2738dd03c5d0e'),
)
NEW_SOURCES = (
    'ragnarok/evaluation/rectangle_visible_groups_v70.py',
    'tests/test_rectangle_visible_groups_v70.py', 'docs/ARC4_RECTANGLE_GROUPS_DRAFT.md',
    'scripts/check_rectangle_groups_v70.py', 'tests/test_rectangle_groups_protocol_v70.py',
    'docs/ARC4_RECTANGLE_GROUPS_OFFLINE_DRAFT.md', PROTOCOL,
    'scripts/__init__.py', 'ragnarok/__init__.py', 'ragnarok/environments/__init__.py',
    'ragnarok/evaluation/visible_temporal_groups_v70.py', 'scripts/temporal_validation_seal_v70.py',
)
SPEC = dict(
    games=['ms_pacman', 'asteroids'], seeds=[17471, 17571], raw_frames_each=8192,
    episodes_each=5, paired_groups_each=8187,
    population='every_consecutive_pair_of_observed_raw_scenes_within_an_episode',
    call_order='global_pair_even_raster_first_odd_rectangles_first',
    equality='complete_ordered_groups_kind_box_and_frozen_member_set',
    max_work_seconds=300, numerical_threads=1, new_emulated_frames=0,
    agent_training_allowed=False, full_chain_timing=False,
)


def current_runtime():
    return dict(**runtime(), executable=sys.executable,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                numerical_threads={key: os.environ.get(key)
                    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def proofs():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    actual_runtime = current_runtime()
    if (protocol.get('kind') != 'prospective_offline_rectangle_groups_v0'
            or protocol.get('spec') != SPEC or protocol.get('runtime') != actual_runtime
            or protocol.get('gate_result') is not None
            or actual_runtime['cuda_visible_devices'] != ''
            or any(value != '1' for value in actual_runtime['numerical_threads'].values())):
        raise ValueError('Exact prospective one-thread offline protocol/runtime required')
    if sha(ROOT / EARLIER_REVIEW) != EARLIER_REVIEW_SHA:
        raise ValueError('The preceding decoder review must remain unchanged')
    earlier = json.loads((ROOT / EARLIER_REVIEW).read_text(encoding='utf-8'))
    reviewed = json.loads(SOURCE_REVIEW.read_text(encoding='utf-8'))
    if earlier.get('passed') is not True or reviewed.get('passed') is not True:
        raise ValueError('Independent original-decoder and candidate-source reviews required')
    if any(reviewed['source_sha256'].get(path) != sha(ROOT / path) for path in NEW_SOURCES[:3]):
        raise ValueError('Current candidate module, tests and documentation must be reviewed')
    sources = {path: sha(ROOT / path) for path in dict.fromkeys((*prior.SOURCE_PATHS, *NEW_SOURCES))}
    evidence = {EARLIER_REVIEW: EARLIER_REVIEW_SHA,
                SOURCE_REVIEW.relative_to(ROOT).as_posix(): sha(SOURCE_REVIEW)}
    for review in (earlier, reviewed):
        for field, target in (('source_sha256', sources), ('evidence_sha256', evidence)):
            for path, expected in review[field].items():
                if sha(ROOT / path) != expected or path in target and target[path] != expected:
                    raise ValueError('Reviewed source/evidence changed or conflicts')
                target[path] = expected
    for game, folder, seed, expected in COLLECTIONS:
        directory = ROOT / folder
        manifest_path = directory / 'manifest.json'
        if sha(manifest_path) != expected:
            raise ValueError('The exact already-observed archive must be retained')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('game'), manifest.get('seed'), manifest.get('frames'),
                manifest.get('episode_count'), manifest.get('agent_training_allowed')) != (
                    game, seed, 8192, 5, False):
            raise ValueError('Declared complete external archive required')
        evidence[manifest_path.relative_to(ROOT).as_posix()] = expected
        for shard in manifest['shards']:
            path = directory / shard['path']
            if path.parent != directory or sha(path) != shard['sha256']:
                raise ValueError('Local exact archived shard required')
            evidence[path.relative_to(ROOT).as_posix()] = shard['sha256']
    for path in {*sources, *evidence}:
        require_committed(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=actual_runtime)


def preflight():
    proof = proofs()
    review = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind') != 'rectangle_groups_independent_preflight'
            or review.get('passed') is not True
            or any(review.get(key) != value for key, value in proof.items())):
        raise ValueError('Exact independent preflight required before reading archive arrays')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof, preflight_review_sha256=sha(PREFLIGHT))


def compare_rows(rows, *, game, decoder, progress, deadline, first_pair=0,
                 expected_frames=8192, expected_episodes=5,
                 original=group_visible_sources, candidate=rectangle_visible_groups):
    recent = deque(maxlen=2)
    frames = episodes = groups = object_count = result_count = 0
    component_seconds = dict(raster=0., rectangles=0.)
    first_calls = dict(raster=0, rectangles=0)
    digest = hashlib.sha256()
    last_episode = None
    for row in rows:
        if perf_counter() > deadline:
            raise ValueError('Offline grouping work ceiling reached')
        episode, local = int(row['episode']), int(row['frame_in_episode'])
        if local == 1:
            if last_episode is not None and episode != last_episode + 1:
                raise ValueError('Consecutive archived episodes required')
            episodes += 1
            recent.clear()
        elif episode != last_episode:
            raise ValueError('Missing declared episode boundary')
        last_episode = episode
        current, previous = row['ram'], row['ram_previous']
        copies = current.copy(), previous.copy()
        source = decoder(current, previous)
        if not np.array_equal(current, copies[0]) or not np.array_equal(previous, copies[1]):
            raise ValueError('Decoder mutated archived RAM')
        recent.append(source)
        frames += 1
        progress.update(game=game, completed_raw_frames=frames, completed_pairs=groups)
        if len(recent) < 2:
            continue  # No invented cold frame0 scene; precisely one omitted pair per episode.
        objects = [*recent[0], *recent[1]]
        snapshot = list(objects)
        calls = [('raster', original), ('rectangles', candidate)]
        if (first_pair + groups) % 2:
            calls.reverse()
        first_calls[calls[0][0]] += 1
        outputs = {}
        for label, function in calls:
            tick = perf_counter()
            outputs[label] = function(objects, game=game)
            component_seconds[label] += perf_counter() - tick
            if objects != snapshot:
                raise ValueError('Grouping mutated source objects')
        if outputs['raster'] != outputs['rectangles']:
            progress['first_mismatch'] = dict(game=game, frame=frames, episode=episode, local=local)
            raise ValueError('Complete ordered group or membership mismatch')
        result = outputs['rectangles']
        event = [frames, episode, local,
                 [(obj.instance_id, obj.kind, obj.box, obj.can_move) for obj in objects],
                 [(group.label.kind, group.label.box, sorted(group.members)) for group in result]]
        digest.update(json.dumps(event, separators=(',', ':')).encode() + b'\n')
        groups += 1
        object_count += len(objects)
        result_count += len(result)
        progress['completed_pairs'] = groups
    if (frames, episodes, groups) != (expected_frames, expected_episodes, expected_frames - expected_episodes):
        raise ValueError('Every declared frame and within-episode pair must be compared')
    return dict(raw_frames=frames, episodes=episodes, paired_groups=groups,
                consumed_source_objects=object_count, returned_groups=result_count,
                first_call_counts=first_calls, component_seconds=component_seconds,
                ordered_objects_groups_sha256=digest.hexdigest(), complete_ordered_groups_exact=True)


def run():
    if ATTEMPT.exists() or OUTPUT.exists():
        raise ValueError('Preserve every offline attempt/result; no retry')
    proof = preflight()
    record = dict(status='started', started_utc=utc(), **proof, new_emulated_frames=0,
                  completed_raw_frames=0, completed_pairs=0)
    write_new(ATTEMPT, record)
    collections = {}
    started = perf_counter()
    deadline = started + SPEC['max_work_seconds']
    try:
        for game, directory, _, _ in COLLECTIONS:
            result = compare_rows(
                raw_history_fixture_frames(ROOT / directory), game=game,
                decoder=mspacman_visible_scene if game == 'ms_pacman' else asteroids_visible_scene,
                progress=record, deadline=deadline,
                first_pair=sum(item['paired_groups'] for item in collections.values()),
            )
            collections[game] = result
            print(json.dumps(dict(game=game, **result)), flush=True)
        work_seconds = perf_counter() - started
        if work_seconds > SPEC['max_work_seconds'] or preflight() != proof:
            raise ValueError('Bounded complete work and unchanged proof required')
        result = dict(kind='offline_rectangle_groups_equivalence', passed=True, created_utc=utc(),
                      **proof, collections=collections, work_seconds=work_seconds,
                      new_emulated_frames=0, agent_training_allowed=False, gate_result=None,
                      limitations=json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))['limitations'])
        write_new(OUTPUT, result)
        record.update(status='complete', completed_utc=utc(), result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed', failed_utc=utc(), error_type=type(error).__name__,
                      error=str(error), completed_collections=collections, no_retry=True)
        raise
    finally:
        ATTEMPT.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(passed=result['passed'], collections=list(result['collections']))))
