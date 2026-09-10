"""One reviewed, bounded derivation from the new action-survey archives only."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / 'craft_v6_out/v70_action_signal_survey/survey_v0.json'
REVIEW = SURVEY.with_name('survey_review_v0.json')
FOLDER = ROOT / 'craft_v6_out/v70_response_envelopes'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_RESPONSE_ENVELOPE_PROTOCOL.json'
SOURCES = ('scripts/build_response_envelopes_v70.py', 'tests/test_response_envelopes_v70.py',
           'docs/ARC4_RESPONSE_ENVELOPE_PROTOCOL.json', 'docs/ARC4_RESPONSE_ENVELOPE_DRAFT.md')
SURVEY_SHA = '877dc75eb44141d89395f009f1325cffd87b7ea13f3c3ea247c7d5f6cd45b61d'
REVIEW_SHA = '3211472c93d6d910b477540fe8384cfdc49bfd112c875a9a3c7b736ee980924f'
GAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
SPEC = dict(version=0, games=list(GAMES), anchors=[10000, 20000], commands=list(range(18)),
            horizons=list(range(1, 9)), shape=[210, 160, 3], max_work_seconds=300,
            max_output_bytes=16000000, max_input_archive_bytes=3700000,
            target='OR over all commands and horizons of any raw RGB difference from own NOOP',
            input='two past RGB images only', dataset_role='development_only',
            new_emulated_frames=0, optimizer_updates=0, gpu_used=False)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def signature(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=sha_bytes(array.tobytes(order='C')))


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    if git('hash-object', f'--path={rel}', str(path)) != git('rev-parse', f'HEAD:{rel}'):
        raise ValueError(f'Uncommitted required input: {rel}')


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def proofs():
    """No archive bytes or arrays are opened by this preflight."""
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4'
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] in ('torch', 'ale_py') for name in sys.modules)):
        raise ValueError('Exact CPU-only parent runtime required, without Torch/ALE imports')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Prospective protocol changed')
    if sha(SURVEY) != SURVEY_SHA or sha(REVIEW) != REVIEW_SHA:
        raise ValueError('Reviewed survey changed')
    survey = json.loads(SURVEY.read_text(encoding='utf-8'))
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if (review['kind'] != 'action_signal_survey_independent_result_review' or review['passed'] is not True
            or review['result_sha256'] != SURVEY_SHA or review['source_sha256'] != survey['source_sha256']
            or survey['execution_status'] != 'complete' or survey['accounting_complete'] is not True
            or survey['acquired_pairs'] != 216 or tuple(c['game'] for c in survey['cases']) != GAMES):
        raise ValueError('Complete independently reviewed survey required')
    sources = dict(survey['source_sha256'])
    for rel in SOURCES:
        committed(ROOT / rel)
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Source changed: {rel}')
    for path in (SURVEY, REVIEW):
        committed(path)
    evidence = {path.relative_to(ROOT).as_posix(): sha(path) for path in (SURVEY, REVIEW)}
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=runtime)


def shapes(shape):
    result = {'branch_start_pair': (2, *shape)}
    for side in ('intervention', 'control'):
        for field in ('raw', 'pooled', 'endpoint'):
            result[f'branch_{side}_{field}'] = (*shape,) if field == 'endpoint' else (8, *shape)
    return result


def decode(data, metadata, shape):
    if len(data) != metadata['bytes'] or sha_bytes(data) != metadata['sha256']:
        raise ValueError('Archive bytes changed')
    expected = shapes(shape)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (len(members) != 7 or {m.filename for m in members} != {key + '.npy' for key in expected}
                or any(m.compress_type != zipfile.ZIP_STORED or m.flag_bits & 1 for m in members)):
            raise ValueError('Exact seven uncompressed, unencrypted arrays required')
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in expected}
    if set(metadata['arrays']) != set(expected):
        raise ValueError('Array signature population differs')
    for key, value in arrays.items():
        if value.dtype != np.uint8 or value.shape != expected[key] or signature(value) != metadata['arrays'][key]:
            raise ValueError(f'Array signature or shape differs: {key}')
    return arrays


def observed_union(arrays, summary):
    if summary['complete_eight_frame_pair'] is not True or summary['common_prefix_frames'] != 8:
        raise ValueError('Truncated or unknown window is not an empty target')
    changes = np.any(arrays['branch_intervention_raw'] != arrays['branch_control_raw'], axis=-1)
    union = np.zeros(changes.shape[1:], dtype=bool)
    if len(summary['horizons']) != 8:
        raise ValueError('All eight horizons required')
    for index, change in enumerate(changes):
        union |= change
        row = summary['horizons'][index]
        if (row['horizon'] != index + 1 or row['observed'] is not True
                or row['changed_pixels'] != int(change.sum())
                or row['cumulative_union_pixels'] != int(union.sum())):
            raise ValueError('Raw counts do not match the actual array coordinates')
    return union


def validate_controls(arrays, baseline, anchor, command):
    if signature(arrays['branch_start_pair']) != anchor['own_anchor']['pair']:
        raise ValueError('Past images differ from the owned anchor')
    for key in ('branch_start_pair', 'branch_control_raw', 'branch_control_pooled', 'branch_control_endpoint'):
        if baseline is not None and not np.array_equal(arrays[key], baseline[key]):
            raise ValueError('Repeated NOOP or past images differ')
    if command == 0:
        for field in ('raw', 'pooled', 'endpoint'):
            if not np.array_equal(arrays[f'branch_intervention_{field}'], arrays[f'branch_control_{field}']):
                raise ValueError('NOOP does not reproduce NOOP')


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Prospective cooperative work ceiling reached; no retry')


def run():
    checked = proofs()
    review = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind') != 'response_envelope_independent_preflight' or review.get('passed') is not True
            or any(review.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact independent prospective preflight required')
    committed(PREFLIGHT)
    paths = {key: FOLDER / name for key, name in dict(attempt='targets_v0_attempt.json',
             progress='targets_v0_progress.jsonl', dataset='targets_v0.npz', result='targets_v0.json').items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Preserve all earlier attempts, journals and outputs')
    # Git stores these archives byte-exact (-text). Build the index without reading their data.
    if git('rev-parse', '--show-object-format') != 'sha1':
        raise ValueError('This checked Git-blob comparison requires the current sha1 repository')
    tree = {}
    for entry in subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-rz', 'HEAD']).split(b'\0'):
        if entry:
            header, name = entry.split(b'\t', 1)
            tree[name.decode('utf-8')] = header.split()[2].decode()
    survey = json.loads(SURVEY.read_text(encoding='utf-8'))
    attempt = dict(kind='observed_response_envelope_attempt', started_utc=datetime.now(timezone.utc).isoformat(),
                   protocol=SPEC, **checked, preflight_sha256=sha(PREFLIGHT))
    write_json(paths['attempt'], attempt)
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds']
    done, pasts, targets, archive_hashes = [], [], [], {}
    archives_read = bytes_read = 0
    output_info = None
    failure = None
    with paths['progress'].open('x', encoding='utf-8', newline='\n') as journal:
        def event(kind, **values):
            journal.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started, **values),
                                     allow_nan=False) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        try:
            for case in survey['cases']:
                if len(case['anchors']) != 2:
                    raise ValueError('Both prospective anchors required')
                for index, anchor in enumerate(case['anchors']):
                    if (anchor['target_delivered_frames'] != SPEC['anchors'][index]
                            or anchor['instrument_valid'] is not True or len(anchor['pairs']) != 18):
                        raise ValueError('Complete valid anchor required')
                    baseline = None
                    target = np.zeros(SPEC['shape'][:2], dtype=bool)
                    for command, pair in enumerate(anchor['pairs']):
                        check_time(deadline)
                        if (pair['command'] != command or pair['status'] != 'acquired'
                                or pair['own_anchor_token'] != anchor['own_anchor']['token']):
                            raise ValueError('Fixed own command/anchor required')
                        metadata = pair['archive']
                        rel = f'craft_v6_out/v70_action_signal_survey/rgb/{case["game"]}_anchor{index}_action{command:02d}.npz'
                        if metadata['path'] != rel or metadata['bytes'] > SPEC['max_input_archive_bytes']:
                            raise ValueError('Only the exact new survey archives are admitted')
                        event('archive_read_started', path=rel)
                        with (ROOT / rel).open('rb') as stream:
                            data = stream.read(SPEC['max_input_archive_bytes'] + 1)
                        archives_read += 1
                        bytes_read += len(data)
                        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                        if blob != tree.get(rel):
                            raise ValueError('Archive differs from its committed Git blob')
                        arrays = decode(data, metadata, tuple(SPEC['shape']))
                        check_time(deadline)
                        validate_controls(arrays, baseline, anchor, command)
                        target |= observed_union(arrays, pair['raw'])
                        if baseline is None:
                            baseline = arrays
                        archive_hashes[rel] = metadata['sha256']
                        event('pair_verified', path=rel, command=command)
                    pasts.append(baseline['branch_start_pair'])
                    targets.append(target)
                    done.append(dict(game=case['game'], anchor_index=index, own_anchor=anchor['own_anchor'],
                                     target_delivered_frames=anchor['target_delivered_frames'],
                                     actual_delivered_frames=anchor['delivered_frames'],
                                     envelope_pixels=int(target.sum()), envelope_signature=signature(target)))
                    event('anchor_complete', **done[-1])
            dataset = dict(past_pairs=np.stack(pasts), observed_response_envelopes=np.stack(targets))
            check_time(deadline)
            buffer = io.BytesIO()
            np.savez(buffer, **dataset)
            data = buffer.getvalue()
            if len(data) > SPEC['max_output_bytes']:
                raise ValueError('Derived output exceeds its fixed storage ceiling')
            with paths['dataset'].open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            output_info = dict(path=paths['dataset'].relative_to(ROOT).as_posix(), bytes=len(data),
                               sha256=sha_bytes(data), arrays={key: signature(value) for key, value in dataset.items()})
            for rel, expected in {**checked['source_sha256'], **checked['evidence_sha256'],
                                  PREFLIGHT.relative_to(ROOT).as_posix(): attempt['preflight_sha256']}.items():
                if sha(ROOT / rel) != expected:
                    raise ValueError(f'Final provenance changed: {rel}')
            check_time(deadline)
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
            event('failed', failure=failure)
        event('finished', passed=failure is None, archives_read=archives_read, bytes_read=bytes_read)
    result = dict(kind='observed_response_envelope_derivation', passed=failure is None,
                  execution_status='complete' if failure is None else 'failed', failure=failure,
                  **checked, protocol=SPEC, preflight_sha256=attempt['preflight_sha256'],
                  attempt_sha256=sha(paths['attempt']), progress_sha256=sha(paths['progress']),
                  cases=done, archives_read=archives_read, bytes_read=bytes_read, archive_sha256=archive_hashes,
                  output=output_info, work_seconds=time.monotonic() - started,
                  inherited_collection_charged_frames=survey['known_charged_frames'],
                  new_emulated_frames=0, optimizer_updates=0, model_inferences=0, gpu_used=False,
                  claim='Derived observed contrasts on twelve development anchors; no learned result')
    write_json(paths['result'], result)
    print(json.dumps(dict(passed=result['passed'], anchors=len(done), archives_read=archives_read,
                         output=output_info, work_seconds=result['work_seconds'])))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        raise SystemExit(run())
    checked = proofs()
    print(json.dumps(dict(check=True, sources=len(checked['source_sha256']), archive_arrays_opened=0)))
