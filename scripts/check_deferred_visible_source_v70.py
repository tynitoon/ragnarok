"""Offline source-cache audit on the complete existing Asteroids raw archive."""
from collections import deque
import hashlib
from itertools import chain
import json
from types import SimpleNamespace

from ragnarok.evaluation.deferred_visible_source_v70 import DeferredVisibleSourceEngine
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.visible_scene_v70 import asteroids_visible_scene
from ragnarok.evaluation.visible_temporal_groups_v70 import VisibleTemporalScorer, group_visible_sources
from scripts import bench_visible_reference_v70 as bench
from scripts.temporal_validation_seal_v70 import ROOT, sha, utc, write_new, require_committed

PROTOCOL = 'docs/ARC4_DEFERRED_SOURCE_OFFLINE_PROTOCOL.json'
SOURCE_PATHS = (*bench.SOURCE_PATHS, 'ragnarok/evaluation/deferred_visible_source_v70.py',
                'tests/test_deferred_visible_source_v70.py', 'scripts/diagnose_asteroids_cold_cache_v70.py',
                'scripts/check_deferred_visible_source_v70.py', 'tests/test_deferred_source_audit_v70.py', PROTOCOL)
DIRECTORY = ROOT/'craft_v6_out/v70_asteroids_calibration/raw_reference_v0'
FOLDER = ROOT/'craft_v6_out/v70_asteroids_cpu'
OUTPUT = FOLDER/'deferred_source_consumption_v0.json'
ATTEMPT = FOLDER/'deferred_source_consumption_v0_attempt.json'


def _history(scorer):
    return (list(scorer._history), scorer._moving, scorer._visible,
            scorer._frame, scorer._episode, scorer._observed_frames)


def audit_rows(rows, decoder=asteroids_visible_scene, *, expected_frames=8192, expected_episodes=5, progress=None):
    rows = iter(rows)
    row = next(rows, None)
    subject = DeferredVisibleSourceEngine.__new__(DeferredVisibleSourceEngine)
    loads = [1]
    subject.engine = SimpleNamespace(accounting=lambda: dict(cold_loads=loads[0]))
    subject._phase, subject._loads = 'ordinary', 1
    subject._episode = subject._ordinary_frames = subject._lab_decoded = 0
    subject._source_labels = deque(maxlen=2)
    subject.scorer = VisibleTemporalScorer('asteroids')
    reference = VisibleTemporalScorer('asteroids')
    calls = [0]
    def counted(current, previous):
        calls[0] += 1
        return decoder(current, previous)
    subject._decoder = counted
    recent = deque(maxlen=2)
    previous_previous = None
    digest = hashlib.sha256()
    frames = groups = episodes = labels = 0
    lengths = {}
    while row is not None:
        following = next(rows, None)
        episode, local = int(row['episode']), int(row['frame_in_episode'])
        current, previous = row['ram'], row['ram_previous']
        copies = current.copy(), previous.copy()
        if local == 1:
            loads[0] = episode+1
            episodes += 1
            recent.clear()
            previous_previous = previous
        source = decoder(current, previous)
        recent.append(source)
        subject._on_frame(SimpleNamespace(current=current, previous=previous,
            before_previous=previous_previous, frame_numbers=(max(0,local-2),local-1,local)))
        frames += 1
        reference.observe_truth(source, frame=frames, episode=episode)
        if _history(subject.scorer) != _history(reference):
            raise ValueError(f'Ordinary truth history differs at frame{frames}')
        if not (current == copies[0]).all() or not (previous == copies[1]).all():
            raise ValueError('Archived RAM was mutated')
        boundary = following is None or int(following['episode']) != episode or local % 4 == 0
        if boundary:
            actual = list(chain.from_iterable(subject._source_labels))
            expected = list(chain.from_iterable(recent))
            if len(recent) == 1:
                expected = list(decoder(previous, previous)) + expected
            if actual != expected:
                raise ValueError(f'Ordered consumed scenes differ at frame{frames}')
            actual_groups = group_visible_sources(actual, game='asteroids')
            expected_groups = group_visible_sources(expected, game='asteroids')
            if actual_groups != expected_groups:
                raise ValueError(f'Consumed source membership differs at frame{frames}')
            groups += 1
            length = (local-1)%4+1
            lengths[str(length)] = lengths.get(str(length),0)+1
            labels += len(actual)
            event = dict(frame=frames, episode=episode, local=local,
                objects=[(o.instance_id,o.kind,o.box,o.can_move) for o in actual],
                groups=[(g.label.kind,g.label.box,sorted(g.members)) for g in actual_groups],
                moving=sorted(reference._moving))
            digest.update(json.dumps(event,sort_keys=True,separators=(',',':')).encode())
        if progress is not None: progress.update(ordinary_truth_frames=frames,consumed_groups=groups)
        previous_previous = previous.copy()
        row = following
    if frames != expected_frames or episodes != expected_episodes:
        raise ValueError('Complete declared archive required')
    return dict(ordinary_truth_frames=frames,episodes=episodes,consumed_groups=groups,
                consumed_source_objects=labels,group_lengths=lengths,callback_decoder_calls=calls[0],
                ordered_consumption_sha256=digest.hexdigest(),movement_history_exactly_equal=True,
                ordered_objects_exactly_equal=True,group_membership_exactly_equal=True)


def sources(): return {p:sha(ROOT/p) for p in SOURCE_PATHS}


def evidence():
    manifest = json.loads((DIRECTORY/'manifest.json').read_text(encoding='utf-8'))
    paths = [DIRECTORY/'manifest.json', *(DIRECTORY/s['path'] for s in manifest['shards']),
             FOLDER/'cpu_wiring_v0_attempt.json', FOLDER/'cold_cache_diagnostic_v0.json',
             FOLDER/'deferred_source_review_v0.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def run():
    if OUTPUT.exists() or ATTEMPT.exists():
        raise ValueError('Preserve every earlier offline result/attempt; no retry')
    protocol = json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    source_hashes, proof_hashes = sources(), evidence()
    frozen = json.loads((ROOT/bench.PREFLIGHT_REVIEW).read_text(encoding='utf-8'))
    reviewed = json.loads((FOLDER/'deferred_source_review_v0.json').read_text(encoding='utf-8'))
    failure = json.loads((FOLDER/'cpu_wiring_v0_attempt.json').read_text(encoding='utf-8'))
    if frozen['source_sha256'] != bench.source_hashes(): raise ValueError('Frozen CPU sources changed')
    if reviewed.get('passed') is not True: raise ValueError('Preliminary source review required')
    for path, digest in reviewed['source_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Reviewed prototype changed')
    if (failure.get('status') != 'failed' or failure.get('known_charged_frames') != 265
            or failure.get('error') != 'Unknown player graphics bank'):
        raise ValueError('The original Asteroids failure must remain unchanged')
    for path in (*source_hashes, *proof_hashes): require_committed(path)
    if (protocol['archived_seed'],protocol['archived_frames'],protocol['archived_episodes'],protocol['input']) != (
            17571,8192,5,DIRECTORY.relative_to(ROOT).as_posix()):
        raise ValueError('Unexpected archived population')
    record = dict(status='started',started_utc=utc(),source_sha256=source_hashes,evidence_sha256=proof_hashes,
                  new_emulated_frames=0,agent_training_allowed=False,gate_result=None)
    write_new(ATTEMPT,record)
    try:
        results = audit_rows(raw_history_fixture_frames(DIRECTORY), progress=record)
        if sources()!=source_hashes or evidence()!=proof_hashes:
            raise ValueError('Sources or evidence changed during offline replay')
        result = dict(kind='offline_deferred_source_consumption_equivalence',created_utc=utc(),passed=True,
            source_sha256=source_hashes,evidence_sha256=proof_hashes,**results,new_emulated_frames=0,
            agent_training_allowed=False,gate_result=None,limitations=[
                'Existing17571 frames only, not failed18176 RAM or a new frame0 RGB calibration.',
                'Source-cache grouping is not a native repeat4 rollout; archived actions can change inside a group.',
                'No native execution, benchmark, training, retry or substitute seed; v0 failure remains unchanged.'])
        write_new(OUTPUT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error_type=type(error).__name__,error=str(error))
        raise
    finally:
        ATTEMPT.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    result=run()
    print(json.dumps({k:result[k] for k in ('passed','ordinary_truth_frames','consumed_groups',
                                         'group_lengths','ordered_consumption_sha256')},indent=2))
