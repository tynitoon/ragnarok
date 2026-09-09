"""One prospectively sealed pair of Ms. Pac-Man raw validation sequences."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import ale_py
import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from scripts.capture_cold_v70 import capture as capture_uniform
from scripts.mspacman_coverage_controller_v70 import CoverageController
from scripts.capture_mspacman_coverage_v70 import SOURCE_PATHS as COVERAGE_SOURCES
from scripts.calibrate_mspacman_maze_v70 import SOURCE_PATHS as CANDIDATE_SOURCES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'craft_v6_out/v70_mspacman_calibration'
PROTOCOL = 'docs/ARC4_MSPACMAN_RAW_VALIDATION_PROTOCOL.json'
SEAL = 'craft_v6_out/v70_mspacman_calibration/raw_validation_seal_v3.json'
SOURCE_PATHS = tuple(dict.fromkeys((*CANDIDATE_SOURCES, *COVERAGE_SOURCES,
    PROTOCOL, 'scripts/validate_mspacman_raw_v70.py',
    'scripts/calibrate_mspacman_reserved_v70.py', 'tests/test_mspacman_raw_validation_v70.py')))
METADATA_PATHS = tuple('craft_v6_out/v70_mspacman_calibration/' + p for p in (
    'development/manifest.json', 'coverage_development_v1/manifest.json',
    'development_visible_v3.json', 'coverage_visible_v3.json',
    'maze_v3_composition_review.json', 'maze_v3_results_review.json',
    'raw_validation_preflight_review_v3.json'))
DESTINATIONS = {'uniform': 'validation_uniform_v3', 'coverage': 'validation_coverage_v3'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def runtime():
    return dict(python=sys.version, versions={p: importlib.metadata.version(p)
        for p in ('ale-py', 'numpy', 'scipy', 'pillow')},
        ale_binary_sha256=sha(ale_py._ale_py.__file__))


def verify_seal():
    seal = json.loads((ROOT/SEAL).read_text(encoding='utf-8'))
    sources = {p: sha(ROOT/p) for p in SOURCE_PATHS}
    metadata = {p: sha(ROOT/p) for p in METADATA_PATHS}
    if (seal.get('source_sha256') != sources or seal.get('metadata_sha256') != metadata
            or seal.get('runtime') != runtime() or seal.get('seeds') != [17412,17432]):
        raise ValueError('Sealed sources, metadata, runtime or seeds differ')
    source_commit = seal['source_commit']
    subprocess.check_call(['git','-C',str(ROOT),'merge-base','--is-ancestor',source_commit,'HEAD'])
    for p in (*SOURCE_PATHS, *METADATA_PATHS, SEAL):
        current = git('hash-object',f'--path={p}',str(ROOT/p))
        if current != git('rev-parse',f'HEAD:{p}'):
            raise ValueError(f'Commit every source and seal before capture: {p}')
        if p in SOURCE_PATHS and current != git('rev-parse',f'{source_commit}:{p}'):
            raise ValueError(f'Source commit does not bind current bytes: {p}')
    seal['_file_sha256'] = sha(ROOT/SEAL)
    return seal


def attempt_path(name):
    return BASE / (DESTINATIONS[name] + '_attempt.json')


def verify_fixture_attempt(directory, manifest, seal):
    names = [name for name, dest in DESTINATIONS.items() if Path(directory).resolve() == (BASE/dest).resolve()]
    if len(names) != 1:
        raise ValueError('Reserved fixture must be its declared destination')
    name = names[0]
    record = json.loads(attempt_path(name).read_text(encoding='utf-8'))
    if (record.get('status') != 'complete' or record.get('name') != name
            or record.get('seed') != manifest.get('seed')
            or record.get('seal_sha256') != seal['_file_sha256']
            or record.get('manifest_sha256') != sha(Path(directory)/'manifest.json')):
        raise ValueError('Reserved capture lacks its completed sealed attempt')
    if manifest.get('accounting',{}).get('ale_binary_sha256') != seal['runtime']['ale_binary_sha256']:
        raise ValueError('Captured emulator binary differs from seal')


def capture_coverage(destination, seal, protocol):
    """Development capture loop unchanged; only seed and declared provenance differ."""
    if destination.exists():
        raise ValueError('Preserve every previous attempt')
    destination.mkdir(parents=True)
    spec = protocol['coverage']
    frames = spec['frames']
    engine = _ColdPixelEngine(EmulatorConfig(game='ms_pacman',seed=spec['seed'],frame_skip=1,
                mode='live',max_episode_frames=protocol['episode_limit'],max_charged_frames=spec['max_charged_frames']))
    controller = CoverageController()
    images,ram,episodes,actions,charges,shards = [],[],[],[],[],[]
    episode = 0
    reset_spent = engine.accounting()['reset_emulated_frames']
    started = time.perf_counter()
    for index in range(1,frames+1):
        if (index-1)%spec['action_repeat']==0:
            action = controller.action(engine.observe(raw=True))
        engine.step(action)
        report = engine.accounting()
        if report['reset_emulated_frames']!=reset_spent:
            episode+=1;reset_spent=report['reset_emulated_frames']
        images.append(engine.observe(raw=True))
        ram.append(engine._ale.getRAM().copy())  # Recorder only; never controller input.
        episodes.append(episode);actions.append(action);charges.append(report['charged_frames'])
        if len(images)==256 or index==frames:
            path=destination/f'frames_{index-len(images)+1:05d}_{index:05d}.npz'
            np.savez_compressed(path,rgb=np.stack(images),ram=np.stack(ram),
                        episode=np.array(episodes,dtype=np.int32),action=np.array(actions,dtype=np.uint8),
                        charged_frame=np.array(charges,dtype=np.int64))
            shards.append(dict(path=path.name,sha256=sha(path),frames=len(images)))
            images,ram,episodes,actions,charges=[],[],[],[],[]
        if index%4096==0:
            print(f"reserved coverage {index}/{frames}; charged={report['charged_frames']}; episodes={episode+1}",flush=True)
    final=engine.accounting()
    if (final['charged_frames']!=frames+132*final['cold_loads'] or final['cold_loads']!=episode+1
            or final['restores'] or final['experiment_frames']):
        raise RuntimeError('Capture ledger differs; preserve incomplete attempt')
    result=dict(kind='scorer_only_cold_raw_development_fixture',game='ms_pacman',mode=0,
                reset_convention=CONVENTION,seed=spec['seed'],frames=frames,
                action_repeat=spec['action_repeat'],action_schedule='instrumentation_rgb_maze_walker_v1',
                raw_every_real_frame=True,episode_limit=protocol['episode_limit'],sticky_probability=.25,
                sticky_implementation='worker-python-random-per-frame',source_commit=git('rev-parse','HEAD'),
                source_sha256=seal['source_sha256'],created_utc=utc(),elapsed_seconds=time.perf_counter()-started,
                accounting=final,shards=shards,versions=runtime()['versions'],
                agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    write_new(destination/'manifest.json',result)
    return result


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def write_new(path, data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def pair_verdict(uniform, coverage):
    # The uniform result is never edited: an absent fruit keeps its null/FAIL.
    expected = {'player','ghost','fruit','power_pill','pill'}
    if set(uniform['per_type']) != expected or set(coverage['per_type']) != expected:
        raise ValueError('All five categories are required in each result')
    fruit = uniform['per_type']['fruit']
    absent = all(fruit[k] == 0 for k in ('positive_frames','ram_objects','pixel_objects','joint_matches'))
    uniform_observed = all(s['passed'] for k,s in uniform['per_type'].items() if k != 'fruit')
    uniform_observed &= absent or fruit['passed']
    return dict(pair_pass=bool(uniform_observed and all(s['passed'] for s in coverage['per_type'].values())),
        uniform_fruit_strictly_absent=absent, uniform_calibration_pass=uniform['calibration_pass'],
        coverage_calibration_pass=coverage['calibration_pass'],
        status='complete_review_required', gate_result=None, agent_training_allowed=False)


def run():
    seal = verify_seal()
    protocol = json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    # Inspect both destinations and every output before the first construction.
    targets = [BASE/d for d in DESTINATIONS.values()]
    targets += [attempt_path(n) for n in DESTINATIONS]
    targets += [BASE/(n+'_reserved_visible_v3.json') for n in DESTINATIONS]
    targets += [BASE/'raw_reserved_pair_v3.json']
    if any(p.exists() for p in targets):
        raise ValueError('The single reserved pair was already attempted; preserve it')
    for name in ('uniform','coverage'):
        verify_seal()
        spec = protocol[name]
        record = dict(kind='mspacman_reserved_capture_attempt',name=name,seed=spec['seed'],
            status='started',started_utc=utc(),seal_sha256=seal['_file_sha256'],source_commit=git('rev-parse','HEAD'))
        write_new(attempt_path(name),record)  # Must precede even cold construction.
        destination = BASE/DESTINATIONS[name]
        try:
            if name == 'uniform':
                result = capture_uniform(destination,game='ms_pacman',seed=spec['seed'],frames=spec['frames'],
                    episode_limit=protocol['episode_limit'])
            else:
                result = capture_coverage(destination,seal,protocol)
            verify_seal()
            ledger = result['accounting']
            if (ledger['charged_frames'] != spec['frames']+132*ledger['cold_loads']
                    or ledger['charged_frames'] > spec['max_charged_frames']
                    or ledger['restores'] or ledger['experiment_frames']):
                raise RuntimeError('Reserved capture ledger differs')
            record.update(status='complete',completed_utc=utc(),manifest_sha256=sha(destination/'manifest.json'),
                          accounting=ledger)
        except BaseException as error:
            record.update(status='failed',completed_utc=utc(),error=repr(error))
            raise
        finally:
            # This live attempt is the sole mutable record. Raw captures are preserved.
            attempt_path(name).write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    from scripts.calibrate_mspacman_reserved_v70 import calibrate
    scores = {}
    for name in ('uniform','coverage'):
        scores[name] = calibrate(BASE/DESTINATIONS[name])
        write_new(BASE/(name+'_reserved_visible_v3.json'),scores[name])
        print(json.dumps(dict(sequence=name,per_type=scores[name]['per_type']),indent=2),flush=True)
    result = pair_verdict(scores['uniform'],scores['coverage'])
    result.update(validation_seal_sha256=seal['_file_sha256'],created_utc=utc(),
        result_sha256={name:sha(BASE/(name+'_reserved_visible_v3.json')) for name in scores})
    write_new(BASE/'raw_reserved_pair_v3.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Read-only seal preflight; no emulator construction')
    args = parser.parse_args()
    if args.check:
        seal = verify_seal()
        print(json.dumps(dict(preflight_pass=True,source_count=len(SOURCE_PATHS),seal_sha256=seal['_file_sha256'])))
    else:
        run()
