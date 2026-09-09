"""One sealed Asteroids raw validation attempt; no new seed or retry."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import ale_py
from scripts.capture_cold_v70 import capture
from scripts.calibrate_asteroids_visible_v70 import SOURCE_PATHS as COMPOSITION_SOURCES

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'craft_v6_out/v70_asteroids_calibration'
PROTOCOL='docs/ARC4_ASTEROIDS_VISIBLE_VALIDATION_PROTOCOL.json'
SEAL='craft_v6_out/v70_asteroids_calibration/visible_validation_seal_v1.json'
DESTINATION='validation_visible_v1'
ATTEMPT='validation_visible_v1_attempt.json'
RESULT='reserved_visible_v1.json'
SOURCE_PATHS=tuple(dict.fromkeys((*COMPOSITION_SOURCES,PROTOCOL,
    'scripts/validate_asteroids_raw_v70.py','scripts/calibrate_asteroids_reserved_v70.py',
    'tests/test_asteroids_raw_validation_v70.py','tests/test_asteroids_history_edges_v70.py',
    'tests/test_asteroids_deblend_v70.py','tests/test_asteroids_player_roi_v70.py',
    'ragnarok/__init__.py','ragnarok/evaluation/__init__.py',
    'ragnarok/environments/__init__.py','scripts/__init__.py')))
METADATA_PATHS=tuple('craft_v6_out/v70_asteroids_calibration/'+p for p in (
    'development/manifest.json','development_visible_v1.json',
    'player_render_roi_development_diagnostic_v2.json','player_missile_roi_correction_review.json',
    'rock_edges_development_v5.json','rock_visible_piece_scope_review.json',
    'visible_composition_preseal_review.json','visible_validation_preflight_review.json'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def runtime():
    return dict(python=sys.version,versions={p:importlib.metadata.version(p)
        for p in ('ale-py','numpy','scipy','pillow')},ale_binary_sha256=sha(ale_py._ale_py.__file__))


def verify_seal():
    seal=json.loads((ROOT/SEAL).read_text(encoding='utf-8'))
    if (seal.get('source_sha256')!={p:sha(ROOT/p) for p in SOURCE_PATHS}
            or seal.get('metadata_sha256')!={p:sha(ROOT/p) for p in METADATA_PATHS}
            or seal.get('runtime')!=runtime() or seal.get('seed')!=17512):
        raise ValueError('Sealed source, metadata, runtime or seed differs')
    source_commit=seal['source_commit']
    subprocess.check_call(['git','-C',str(ROOT),'merge-base','--is-ancestor',source_commit,'HEAD'])
    for p in (*SOURCE_PATHS,*METADATA_PATHS,SEAL):
        current=git('hash-object',f'--path={p}',str(ROOT/p))
        if current!=git('rev-parse',f'HEAD:{p}'):
            raise ValueError(f'Commit all sources and metadata before capture: {p}')
        if p in SOURCE_PATHS and current!=git('rev-parse',f'{source_commit}:{p}'):
            raise ValueError(f'Source commit does not bind current bytes: {p}')
    seal['_file_sha256']=sha(ROOT/SEAL)
    return seal


def verify_fixture_attempt(directory,manifest,seal):
    if Path(directory).resolve()!=(BASE/DESTINATION).resolve():
        raise ValueError('Reserved fixture must use its declared destination')
    attempt=json.loads((BASE/ATTEMPT).read_text(encoding='utf-8'))
    if (attempt.get('status') not in ('captured','scoring','complete')
            or attempt.get('seed')!=17512 or manifest.get('seed')!=17512
            or attempt.get('seal_sha256')!=seal['_file_sha256']
            or attempt.get('manifest_sha256')!=sha(Path(directory)/'manifest.json')
            or manifest.get('accounting',{}).get('ale_binary_sha256')!=seal['runtime']['ale_binary_sha256']):
        raise ValueError('Reserved fixture lacks its sealed completed capture')


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def write_attempt(value):
    # This live attempt is the sole mutable record. All raw data are preserved.
    (BASE/ATTEMPT).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def run():
    seal=verify_seal()
    protocol=json.loads((ROOT/PROTOCOL).read_text(encoding='utf-8'))
    if any((BASE/p).exists() for p in (DESTINATION,ATTEMPT,RESULT)):
        raise ValueError('The single reserved experiment was already attempted')
    record=dict(kind='asteroids_visible_reserved_attempt',status='started',seed=17512,
        source_commit=git('rev-parse','HEAD'),seal_sha256=seal['_file_sha256'],started_utc=utc())
    write_new(BASE/ATTEMPT,record)  # Durable before the first emulator construction.
    try:
        manifest=capture(BASE/DESTINATION,game='asteroids',seed=protocol['seed'],
            frames=protocol['frames'],episode_limit=protocol['episode_limit'])
        verify_seal()
        ledger=manifest['accounting']
        if (ledger['charged_frames']!=protocol['frames']+132*ledger['cold_loads']
                or ledger['charged_frames']>protocol['max_charged_frames']
                or ledger['restores'] or ledger['experiment_frames']):
            raise RuntimeError('Capture accounting differs from the fixed protocol')
        record.update(status='captured',captured_utc=utc(),manifest_sha256=sha(BASE/DESTINATION/'manifest.json'),
                      accounting=ledger)
        write_attempt(record)
        from scripts.calibrate_asteroids_reserved_v70 import calibrate
        record.update(status='scoring');write_attempt(record)
        result=calibrate(BASE/DESTINATION)
        verify_seal()
        write_new(BASE/RESULT,result)
        record.update(status='complete',completed_utc=utc(),result_sha256=sha(BASE/RESULT),
            calibration_pass=result['calibration_pass'],next_action='independent_scientific_review_required')
    except BaseException as error:
        record.update(status='failed',failed_utc=utc(),error=repr(error))
        raise
    finally:
        write_attempt(record)
    print(json.dumps(dict(calibration_pass=result['calibration_pass'],per_type=result['per_type'],
                         next_action=record['next_action']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify committed seal without an emulator')
    args=parser.parse_args()
    if args.check:
        seal=verify_seal()
        print(json.dumps(dict(preflight_pass=True,source_count=len(SOURCE_PATHS),seal_sha256=seal['_file_sha256'])))
    else:run()
