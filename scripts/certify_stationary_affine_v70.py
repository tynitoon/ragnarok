"""A separate, one-shot algebraic judge of the already learned affine rule."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from ragnarok.evaluation.affine_stationary_family_v70 import exact_family_bounds
from scripts.fit_response_envelope_v70 import (
    sha, sha_bytes, signature, committed, configure_runtime, check_time)

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / 'craft_v6_out/v70_stationary_affine_fit'
RESULT, REVIEW = FIT / 'fit_v0.json', FIT / 'fit_review_v0.json'
RESULT_SHA = 'c60b92fea8846af3cbc8bfe9db95bdfc1585ad9c661b276462e9db67f2aa8014'
REVIEW_SHA = '8be644481304ed9393f62fa93232ed5534dc46f5ee5fa2775226b710f0a646f5'
FOLDER = ROOT / 'craft_v6_out/v70_stationary_affine_family'
PREFLIGHT = FOLDER / 'preflight_v0.json'
SOURCES = ('ragnarok/evaluation/affine_stationary_family_v70.py',
           'tests/test_affine_stationary_family_v70.py', 'scripts/certify_stationary_affine_v70.py',
           'tests/test_certify_stationary_affine_v70.py', 'docs/ARC4_STATIONARY_AFFINE_FAMILY_DRAFT.md')
ARMS = ('initial', 'newton_visual', 'newton_constant', 'adam_visual', 'adam_constant')
SPEC = dict(version=0, primary='newton_visual', coefficient_sets=list(ARMS), arithmetic='exact rational float64 coefficients',
            positive_condition='b + (w0+w3) + min(0,w1+w4) > 0',
            negative_condition='b + max(0,w1+w4) + max(0,w2+w5) < 0',
            max_work_seconds=60, cleanup_reserve_seconds=10, max_output_bytes=1000000,
            cpu_threads=1, gpu_used=False, max_state_archives=2, certificates=5, model_products=0,
            gradients=0, optimizer_updates=0, image_decodes=0, new_synthetic_steps=0, new_emulated_frames=0,
            retry=False, claim='Exact learned affine function on the declared synthetic family only; not all floating-point roundings')


def proofs():
    if REVIEW_SHA is None or sha(RESULT) != RESULT_SHA or sha(REVIEW) != REVIEW_SHA:
        raise ValueError('Exact completed independently reviewed affine fit required')
    result, review = (json.loads(path.read_text(encoding='utf-8')) for path in (RESULT, REVIEW))
    if (result.get('kind') != 'stationary_affine_development_fit' or result.get('execution_status') != 'complete'
            or result.get('final_provenance_verified') is not True
            or review.get('kind') != 'stationary_affine_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != RESULT_SHA
            or review.get('source_sha256') != result['source_sha256']):
        raise ValueError('The algebraic judge requires the exact valid prior experiment')
    sources = dict(result['source_sha256'])
    for rel in SOURCES:
        if rel in sources: raise ValueError('No replacement of inherited sources')
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected: raise ValueError('Changed required source: ' + rel)
        committed(ROOT / rel)
    evidence = {}
    for path in (RESULT, REVIEW, FIT / 'initial_state.npz', FIT / 'final_state.npz'):
        digest = sha(path)
        if path.suffix == '.npz' and digest != result['artifacts'][path.name]['sha256']:
            raise ValueError('Frozen learned or initial coefficients changed')
        committed(path)
        evidence[path.relative_to(ROOT).as_posix()] = digest
    environment = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__)
    if (environment != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4'
            or any(name.split('.')[0] == 'ale_py' for name in sys.modules)):
        raise ValueError('Declared exact CPU1 runtime without emulator required')
    return dict(source_sha256=sources, evidence_sha256=evidence, environment=environment, runtime=runtime)


def decode_state(path, artifact):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(artifact['arrays']): raise ValueError('State member population changed')
        arrays = {name: data[name] for name in data.files}
    if any(signature(value) != artifact['arrays'][key] for key, value in arrays.items()):
        raise ValueError('State array signature changed')
    return arrays


def run():
    configure_runtime()
    names = ('family_v0_attempt.json', 'family_v0_progress.jsonl', 'family_v0_payload.json', 'family_v0.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve every prior algebraic attempt; no retry')
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    failures, checked, certificates, byte_counts = [], None, {}, {}
    entries, returns = dict(state_archive=0, certificate=0), dict(state_archive=0, certificate=0)
    journal, provenance, preflight_hash = None, False, None

    def fail(stage, error):
        failures.append(dict(stage=stage, type=type(error).__name__, message=str(error)))

    def write_json(name, value):
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if len(data) + sum(byte_counts.values()) > SPEC['max_output_bytes']:
            raise ValueError('Fixed output byte ceiling exceeded')
        with (FOLDER / name).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        byte_counts[name] = len(data)
        if sha(FOLDER / name) != sha_bytes(data): raise ValueError('JSON readback differs')

    def event(kind, **fields):
        if journal is None: raise OSError('Journal not opened')
        data = (json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started, **fields), allow_nan=False) + '\n').encode('utf-8')
        if len(data) + sum(byte_counts.values()) > SPEC['max_output_bytes']:
            raise ValueError('Fixed output byte ceiling exceeded')
        journal.write(data)
        journal.flush()
        os.fsync(journal.fileno())
        byte_counts[names[1]] = byte_counts.get(names[1], 0) + len(data)

    try:
        write_json(names[0], dict(kind='stationary_affine_family_attempt', created_utc=datetime.now(timezone.utc).isoformat(), protocol=SPEC))
        journal = (FOLDER / names[1]).open('xb')
        checked = proofs()
        preflight_hash = sha(PREFLIGHT)
        preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
        if (preflight.get('kind') != 'stationary_affine_family_independent_preflight'
                or preflight.get('passed') is not True or any(preflight.get(key) != value for key, value in checked.items())):
            raise ValueError('Exact independent prospective family preflight required')
        committed(PREFLIGHT)
        result = json.loads(RESULT.read_text(encoding='utf-8'))
        states = {}
        for name in ('initial_state.npz', 'final_state.npz'):
            check_time(deadline)
            event('archive_enter', name=name)
            entries['state_archive'] += 1
            states[name] = decode_state(FIT / name, result['artifacts'][name])
            returns['state_archive'] += 1
            event('archive_return', name=name)
        initial, final = states['initial_state.npz'], states['final_state.npz']
        if any(value.any() for value in initial.values()): raise ValueError('Initial states must be exactly zero')
        coefficients = {'initial': initial['newton_visual_coefficients']}
        coefficients.update({arm: final[arm + '_coefficients'] for arm in ARMS[1:]})
        for arm in ARMS:
            check_time(deadline)
            event('certificate_enter', arm=arm)
            entries['certificate'] += 1
            measured = exact_family_bounds(coefficients[arm])
            returns['certificate'] += 1
            certificates[arm] = measured
            event('certificate_return', arm=arm, certificate=measured)
    except BaseException as error:
        fail('scientific_work', error)
    finally:
        try:
            if checked is not None and (proofs() != checked or sha(PREFLIGHT) != preflight_hash):
                raise ValueError('Final sources, evidence or preflight changed')
            if checked is None: raise ValueError('Initial provenance did not complete')
            committed(PREFLIGHT)
            check_time(started + SPEC['max_work_seconds'])
            provenance = True
        except BaseException as error:
            fail('final_provenance_or_time', error)
        try:
            event('finished', entries=entries, returns=returns, failures=failures.copy())
        except BaseException as error:
            fail('journal_finished', error)
        if journal is not None:
            try: journal.close()
            except BaseException as error: fail('journal_close', error)
    complete = (not failures and provenance and entries == returns == dict(state_archive=2, certificate=5)
                and list(certificates) == list(ARMS))
    payload = dict(kind='stationary_affine_family_payload', protocol=SPEC, provenance=checked, certificates=certificates,
                   failures=failures.copy(), candidate_complete=complete, operation_entries=entries, operation_returns=returns,
                   operation_counts_exact=entries == returns, final_provenance_verified=provenance,
                   preflight_sha256=preflight_hash, result_sha256=RESULT_SHA, review_sha256=REVIEW_SHA,
                   progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None)
    payload_hash = None
    try:
        write_json(names[2], payload)
        payload_hash = sha(FOLDER / names[2])
    except BaseException as error: fail('scientific_payload', error)
    elapsed = time.monotonic() - started
    if elapsed >= SPEC['max_work_seconds']: fail('scientific_payload_time', TimeoutError('Fixed complete-work ceiling exceeded'))
    complete = complete and not failures and elapsed < SPEC['max_work_seconds']
    result = dict(payload, kind='stationary_affine_family_certificate', failures=failures,
                  execution_status='complete' if complete else 'incomplete',
                  passed=bool(complete and certificates['newton_visual']['certified']),
                  work_seconds=elapsed, scientific_payload_sha256=payload_hash)
    write_json(names[3], result)
    print(json.dumps(dict(execution_status=result['execution_status'], primary_certified=result['passed'],
                        certificates=certificates, work_seconds=elapsed), allow_nan=False))
    return 0 if complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
