"""Offline diagnosis of the retained Asteroids v0 cold-cache failure.

No emulator is constructed; archived RAM and a synthetic callback sequence are
used separately. Neither establishes a new frame-0 RGB/RAM calibration.
"""
from collections import deque
import json
from types import SimpleNamespace
import numpy as np
from ragnarok.evaluation.raw_history_fixtures_v70 import raw_history_fixture_frames
from ragnarok.evaluation.visible_scene_v70 import asteroids_visible_scene
from ragnarok.evaluation.scene_perception_v70 import SceneObject
from ragnarok.evaluation.visible_reference_pool_v70 import VisibleReferenceEngine
from scripts import bench_visible_reference_v70 as bench
from scripts.temporal_validation_seal_v70 import ROOT, sha, utc, write_new, require_committed

SELF = 'scripts/diagnose_asteroids_cold_cache_v70.py'
DIRECTORY = ROOT/'craft_v6_out/v70_asteroids_calibration/raw_reference_v0'
FAILED = ROOT/'craft_v6_out/v70_asteroids_cpu/cpu_wiring_v0_attempt.json'
OUTPUT = ROOT/'craft_v6_out/v70_asteroids_cpu/cold_cache_diagnostic_v0.json'


def synthetic_cache():
    calls = []
    subject = VisibleReferenceEngine.__new__(VisibleReferenceEngine)
    subject.engine = SimpleNamespace(accounting=lambda:dict(cold_loads=1))
    subject._loads = 1
    subject._phase = 'ordinary'
    subject._episode = subject._ordinary_frames = 0
    subject._source_labels = deque(maxlen=2)
    subject.scorer = SimpleNamespace(observe_truth=lambda *args,**kwargs:None)
    def decode(now, previous):
        calls.append([int(now[0]),int(previous[0])])
        return [SceneObject(str(int(now[0])),'player',(1,1,2,2))]
    subject._decoder = decode
    histories = []
    for frame in range(1,5):
        numbers = tuple(max(0,frame-i) for i in (2,1,0))
        older, previous, current = [np.full(128,n,dtype=np.uint8) for n in numbers]
        subject._on_frame(SimpleNamespace(before_previous=older,previous=previous,
            current=current,frame_numbers=numbers))
        histories.append([int(scene[0].instance_id) for scene in subject._source_labels])
    return dict(decoder_pairs=calls, retained_source_frames_after_each_act=histories,
                frame0_absent_after_complete_repeat=histories[-1]==[3,4],
                frame0_needed_for_one_frame_repeat=histories[0]==[0,1])


def run():
    if OUTPUT.exists():
        raise ValueError('Preserve existing diagnostic')
    source_hashes = {**bench.source_hashes(), SELF:sha(ROOT/SELF)}
    preflight = json.loads((ROOT/bench.PREFLIGHT_REVIEW).read_text(encoding='utf-8'))
    if preflight.get('source_sha256') != bench.source_hashes():
        raise ValueError('Frozen CPU sources changed')
    for path in (SELF, FAILED.relative_to(ROOT).as_posix(),
                 (DIRECTORY/'manifest.json').relative_to(ROOT).as_posix()):
        require_committed(path)
    failure = json.loads(FAILED.read_text(encoding='utf-8'))
    if (failure.get('status')!='failed' or failure.get('known_charged_frames')!=265
            or failure.get('error')!='Unknown player graphics bank'):
        raise ValueError('The retained v0 failure must remain unchanged')
    cold = []
    for row in raw_history_fixture_frames(DIRECTORY):
        if int(row['frame_in_episode']) != 1:
            continue
        previous,current = row['ram_previous'],row['ram']
        copies = previous.copy(),current.copy()
        error = None
        try:
            asteroids_visible_scene(previous,previous)
        except ValueError as exc:
            error = str(exc)
        current_scene = asteroids_visible_scene(current,previous)
        if not np.array_equal(previous,copies[0]) or not np.array_equal(current,copies[1]):
            raise ValueError('Archived RAM changed during diagnosis')
        import hashlib
        cold.append(dict(index=row['index'],episode=int(row['episode']),
            frame0_ram_sha256=hashlib.sha256(previous.tobytes()).hexdigest(),
            frame0_fields={str(i):int(previous[i]) for i in (57,109,110,111)},
            frame0_error=error,frame1_scene_objects=len(current_scene)))
    if len(cold)!=5:
        raise ValueError('All five archived cold starts required')
    manifest = json.loads((DIRECTORY/'manifest.json').read_text(encoding='utf-8'))
    paths = [FAILED,DIRECTORY/'manifest.json',*(DIRECTORY/s['path'] for s in manifest['shards'])]
    if source_hashes != {**bench.source_hashes(),SELF:sha(ROOT/SELF)}:
        raise ValueError('Sources changed during diagnosis')
    result = dict(kind='offline_asteroids_cold_cache_failure_diagnosis',created_utc=utc(),
        source_sha256=source_hashes, evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        archived_seed=17571,archived_cold_starts=cold,synthetic_cache=synthetic_cache(),
        known_failed_wiring_charge=265,new_emulated_frames=0,agent_training_allowed=False,gate_result=None,
        limitations=['The failed18176 RAM was not archived; these are existing17571 witnesses of the same guard.',
            'A full repeat discards source frame0 before scoring, but a one-frame repeat requires it.',
            'No source frame0 RGB was calibrated here; never infer an empty scene from the missing bank.',
            'Frozen v0 sources/results remain unchanged; no retry or substitute seed was performed.'])
    write_new(OUTPUT,result)
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(cold_starts=len(result['archived_cold_starts']),
        errors=[r['frame0_error'] for r in result['archived_cold_starts']],
        synthetic_cache=result['synthetic_cache']),indent=2))
