"""One independent algebraic review, using committed JSON only, never NPZ data."""
import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'craft_v6_out/v70_stationary_affine_family/'
FIT = 'craft_v6_out/v70_stationary_affine_fit/'
RESULT_SHA = '129fd18914406329ee872444b8ffe573416e3922fd85f1e21f1f47ed351d5711'
FIT_SHA = 'c60b92fea8846af3cbc8bfe9db95bdfc1585ad9c661b276462e9db67f2aa8014'
FIT_REVIEW_SHA = '8be644481304ed9393f62fa93232ed5534dc46f5ee5fa2775226b710f0a646f5'
PREFLIGHT_SHA = '1206de32d5f644caa448df44bb290a65cbe196b2a0a7c044b9b895f8043483e0'
SOURCE_COMMIT, PREFLIGHT_COMMIT, RESULT_COMMIT = '9b29545', 'ccf1411', '3133ac0'
FIT_COMMIT, FIT_REVIEW_COMMIT = '3ef0cf9', '0015732'
ARMS = ('initial', 'newton_visual', 'newton_constant', 'adam_visual', 'adam_constant')
NEW_SOURCES = {
    'ragnarok/evaluation/affine_stationary_family_v70.py',
    'tests/test_affine_stationary_family_v70.py', 'scripts/certify_stationary_affine_v70.py',
    'tests/test_certify_stationary_affine_v70.py', 'docs/ARC4_STATIONARY_AFFINE_FAMILY_DRAFT.md',
}
MAX_SECONDS = 60


def require(value, message):
    if not value:
        raise ValueError(message)


class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'ale_py', 'ragnarok', 'scripts'}:
            raise RuntimeError('Review forbids numerical experiment import: ' + fullname)
        return None


def exclusive_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


class Review:
    def __init__(self):
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.evidence, self.sources, self.checks = {}, {}, {}
        self.recalculations = 0

    def checkpoint(self):
        require(time.monotonic() - self.started < MAX_SECONDS, 'Fixed review ceiling reached')

    def git(self, *args, input_text=None):
        self.checkpoint()
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, input=input_text,
            timeout=max(.1, MAX_SECONDS - (time.monotonic() - self.started))).strip()

    def path(self, rel):
        path = ROOT / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts
                and path.resolve().is_relative_to(ROOT.resolve()), 'Outside repository')
        require(path.suffix.lower() != '.npz', 'No NPZ file may be reopened by this reviewer')
        return path

    def digest(self, rel):
        self.checkpoint()
        with self.path(rel).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def json(self, rel, expected=None):
        data = self.path(rel).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, 'JSON SHA differs: ' + rel)
        self.evidence[rel] = digest
        return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def committed(self, paths, ref='HEAD'):
        paths = list(paths)
        require(all(not name.endswith('.npz') for name in paths), 'Git hashing of a NPZ is forbidden')
        actual = self.git('hash-object', '--stdin-paths', input_text='\n'.join(paths) + '\n').splitlines()
        expected = self.git('rev-parse', *[ref + ':' + rel for rel in paths]).splitlines()
        require(actual == expected and len(actual) == len(paths), 'Committed blobs differ: ' + ref)

    def date(self, commit):
        return datetime.fromisoformat(self.git('show', '-s', '--format=%cI', commit))

    def check_state_json(self, review, fit):
        states = review['checks']['state_arrays']
        keys = {arm + '_' + key for arm in ARMS[1:] for key in
                (('coefficients', 'step') if arm.startswith('newton_') else
                 ('coefficients', 'step', 'first_moment', 'second_moment'))}
        count = 0
        for name in ('initial_state.npz', 'final_state.npz'):
            state, descriptions = states[name], fit['artifacts'][name]['arrays']
            require(set(state) == set(descriptions) == keys and len(keys) == 12, 'State JSON member population differs')
            for key, value in state.items():
                desc = descriptions[key]
                if key.endswith('_step'):
                    expected = 0 if name.startswith('initial') else (16 if key.startswith('newton_') else 256)
                    require(type(value) is int and value == expected, 'Unexpected saved step')
                    shape, dtype, data = [], 'int64', struct.pack('<q', value)
                else:
                    require(isinstance(value, list) and len(value) == 7
                            and all(type(x) in (float, int) and math.isfinite(x) for x in value), 'Seven finite saved coefficients or moments required')
                    if name.startswith('initial'):
                        require(all(x == 0 for x in value), 'Initial state must be zero')
                    if key.endswith('_second_moment'):
                        require(all(x >= 0 for x in value), 'Negative second moment')
                    shape, dtype, data = [7], 'float64', struct.pack('<7d', *value)
                require(desc == dict(shape=shape, dtype=dtype, sha256=hashlib.sha256(data).hexdigest()),
                        'JSON values do not reconstruct the previously attested state signature: ' + key)
                count += 1
        self.checks['state_json_signatures'] = dict(members_reserialized=count, npz_members_decoded=0,
            method='Little-endian struct serialization of committed JSON values, matched to prior NPZ array signatures')
        return {'initial': states['initial_state.npz']['newton_visual_coefficients'],
                **{arm: states['final_state.npz'][arm + '_coefficients'] for arm in ARMS[1:]}}

    def bounds(self, values, recorded):
        # Independent endpoint enumeration by additions, not the production min/max formula.
        q = [Fraction(*float(value).as_integer_ratio()) for value in values]
        positives = [q[6] + q[0] + q[3], q[6] + q[0] + q[3] + q[1] + q[4]]
        negatives = [q[6], q[6] + q[1] + q[4], q[6] + q[2] + q[5],
                     q[6] + q[1] + q[4] + q[2] + q[5]]
        low, high = min(positives), max(negatives)
        self.recalculations += 1
        for key, expected in (('positive_min_logit', low), ('negative_max_logit', high)):
            row = recorded[key]
            require(row['numerator'] == str(expected.numerator)
                    and row['denominator'] == str(expected.denominator), 'Exact bound fraction differs: ' + key)
            try:
                approximation = float(expected)
            except OverflowError:
                approximation = None
            require(row['approximate'] == approximation, 'Displayed approximation differs')
        require(recorded['positive_strict'] is (low > 0) and recorded['negative_strict'] is (high < 0)
                and recorded['certified'] is (low > 0 and high < 0), 'Strict scientific certificate differs')
        return dict(positive_min=dict(numerator=str(low.numerator), denominator=str(low.denominator)),
                    negative_max=dict(numerator=str(high.numerator), denominator=str(high.denominator)),
                    positive_strict=low > 0, negative_strict=high < 0, certified=low > 0 and high < 0)

    def perform(self):
        own = Path(__file__).resolve().relative_to(ROOT).as_posix()
        self.committed([own])
        self.evidence[own] = self.digest(own)
        self.checks['reviewer_commit'] = self.git('log', '-1', '--format=%H', '--', own)
        preflight = self.json(FOLDER + 'preflight_v0.json', PREFLIGHT_SHA)
        require(preflight['kind'] == 'stationary_affine_family_independent_preflight' and preflight['passed'] is True, 'Invalid independent preflight')
        result = self.json(FOLDER + 'family_v0.json', RESULT_SHA)
        payload = self.json(FOLDER + 'family_v0_payload.json', result['scientific_payload_sha256'])
        attempt = self.json(FOLDER + 'family_v0_attempt.json')
        fit = self.json(FIT + 'fit_v0.json', FIT_SHA)
        fit_review = self.json(FIT + 'fit_review_v0.json', FIT_REVIEW_SHA)
        require(fit['kind'] == 'stationary_affine_development_fit' and fit['execution_status'] == 'complete'
                and fit['final_provenance_verified'] is True and fit_review['kind'] == 'stationary_affine_independent_result_review'
                and fit_review['passed'] is True and fit_review['result_sha256'] == FIT_SHA
                and fit_review['source_sha256'] == fit['source_sha256'], 'Prior reviewed fit is not the exact complete fit')
        require(result['kind'] == 'stationary_affine_family_certificate' and payload['kind'] == 'stationary_affine_family_payload'
                and attempt['kind'] == 'stationary_affine_family_attempt', 'Wrong judgement artefact kinds')
        require(result['execution_status'] == 'complete' and result['candidate_complete'] is True
                and result['failures'] == [] and result['final_provenance_verified'] is True, 'Incomplete judgement')
        require(all(result[key] == value for key, value in payload.items() if key != 'kind'), 'Final result differs from payload')
        require(result['preflight_sha256'] == PREFLIGHT_SHA and result['result_sha256'] == FIT_SHA
                and result['review_sha256'] == FIT_REVIEW_SHA, 'Provenance linkage differs')
        proof = result['provenance']
        require(set(proof) == {'source_sha256', 'evidence_sha256', 'environment', 'runtime'}
                and all(preflight[key] == value for key, value in proof.items()), 'Exact checked preflight maps differ')
        self.sources = proof['source_sha256']
        require(len(self.sources) == 117 and len(fit['source_sha256']) == 112
                and set(self.sources) == set(fit['source_sha256']) | NEW_SOURCES
                and all(self.sources[key] == value for key, value in fit['source_sha256'].items()), 'Source closure differs')
        for rel, expected in self.sources.items():
            require(self.digest(rel) == expected, 'Source SHA differs: ' + rel)
        self.committed(self.sources, SOURCE_COMMIT)
        self.committed(self.sources)
        expected_proofs = {FIT + 'fit_v0.json': FIT_SHA, FIT + 'fit_review_v0.json': FIT_REVIEW_SHA,
                          **{FIT + name: fit['artifacts'][name]['sha256'] for name in ('initial_state.npz', 'final_state.npz')}}
        require(proof['evidence_sha256'] == expected_proofs, 'Four fixed proofs differ')
        npz_evidence = {}
        for name in ('initial_state.npz', 'final_state.npz'):
            rel = FIT + name
            require(fit_review['evidence_sha256'][rel] == expected_proofs[rel], 'Prior review did not attest the exact NPZ hash')
            blob = self.git('rev-parse', FIT_COMMIT + ':' + rel)
            require(blob == self.git('rev-parse', 'HEAD:' + rel), 'Committed NPZ blob changed')
            npz_evidence[rel] = dict(sha256=expected_proofs[rel], committed_blob=blob,
                method='Hash inherited from pinned prior result/review and preflight; Git tree blob identity checked; NPZ not reopened')
        self.checks['sealed_npz_provenance'] = npz_evidence
        self.evidence.update({rel: value['sha256'] for rel, value in npz_evidence.items()})
        protocol = result['protocol']
        require(protocol == attempt['protocol'] and protocol['primary'] == 'newton_visual'
                and protocol['coefficient_sets'] == list(ARMS) and protocol['max_work_seconds'] == 60
                and protocol['cleanup_reserve_seconds'] == 10 and protocol['max_output_bytes'] == 1000000
                and protocol['max_state_archives'] == 2 and protocol['certificates'] == 5
                and protocol['cpu_threads'] == 1 and protocol['gpu_used'] is False and protocol['retry'] is False,
                'Fixed protocol differs')
        for key in ('model_products', 'gradients', 'optimizer_updates', 'image_decodes', 'new_synthetic_steps', 'new_emulated_frames'):
            require(protocol[key] == 0, 'Unexpected judgement computation scope')
        require(result['operation_entries'] == result['operation_returns'] == dict(state_archive=2, certificate=5)
                and result['operation_counts_exact'] is True
                and result['validated_state_array_members'] == {'initial_state.npz': 12, 'final_state.npz': 12}, 'Judgement counts differ')
        require(list(result['certificates']) == list(ARMS) and 0 < result['work_seconds'] < 60, 'Incomplete population or invalid duration')
        coefficients = self.check_state_json(fit_review, fit)
        self.checks['independent_certificates'] = {}
        for arm in ARMS:
            self.checkpoint()
            self.checks['independent_certificates'][arm] = self.bounds(coefficients[arm], result['certificates'][arm])
        require(result['passed'] is self.checks['independent_certificates']['newton_visual']['certified'], 'Primary result changed')
        require(self.checks['independent_certificates']['initial']['positive_min']['numerator'] == '0'
                and not self.checks['independent_certificates']['initial']['certified'], 'Zero initial is not strictly certified')
        journal_rel = FOLDER + 'family_v0_progress.jsonl'
        data = self.path(journal_rel).read_bytes()
        require(data.endswith(b'\n') and hashlib.sha256(data).hexdigest() == result['progress_sha256'], 'Closed journal hash differs')
        self.evidence[journal_rel] = result['progress_sha256']
        events = [json.loads(line) for line in data.splitlines()]
        require(len(events) == 15, 'Exactly 15 judgement journal events required')
        expected = [('archive_' + stage, name) for name in ('initial_state.npz', 'final_state.npz') for stage in ('enter', 'return')]
        expected += [('certificate_' + stage, arm) for arm in ARMS for stage in ('enter', 'return')]
        times = []
        for event, (kind, value) in zip(events[:-1], expected, strict=True):
            require(event['event'] == kind and event['name' if kind.startswith('archive') else 'arm'] == value, 'Journal order differs')
            if kind == 'certificate_return':
                require(event['certificate'] == result['certificates'][value], 'Returned certificate differs from payload')
            times.append(event['elapsed_seconds'])
        last = events[-1]
        require(last['event'] == 'finished' and last['entries'] == last['returns'] == result['operation_returns']
                and last['failures'] == [], 'Incomplete journal closure')
        times.append(last['elapsed_seconds'])
        require(all(math.isfinite(t) and 0 <= t < result['work_seconds'] for t in times) and times == sorted(times), 'Uncharged or nonmonotone events')
        judged_files = [FOLDER + name for name in ('family_v0_attempt.json', 'family_v0_progress.jsonl', 'family_v0_payload.json', 'family_v0.json')]
        sizes = {rel: self.path(rel).stat().st_size for rel in judged_files}
        require(sum(sizes.values()) <= 1000000, 'Judgement one-MB output cap exceeded')
        self.committed(judged_files, RESULT_COMMIT)
        self.committed([FOLDER + 'preflight_v0.json'], PREFLIGHT_COMMIT)
        self.committed([FIT + 'fit_v0.json'], FIT_COMMIT)
        self.committed([FIT + 'fit_review_v0.json'], FIT_REVIEW_COMMIT)
        for earlier, later in ((SOURCE_COMMIT, PREFLIGHT_COMMIT), (PREFLIGHT_COMMIT, RESULT_COMMIT),
                               (FIT_COMMIT, FIT_REVIEW_COMMIT), (FIT_REVIEW_COMMIT, SOURCE_COMMIT)):
            self.git('merge-base', '--is-ancestor', earlier, later)
        began = datetime.fromisoformat(attempt['created_utc'])
        require(self.date(SOURCE_COMMIT) <= self.date(PREFLIGHT_COMMIT) < began
                and self.date(RESULT_COMMIT) + timedelta(seconds=1) >= began + timedelta(seconds=result['work_seconds']), 'Git/attempt chronology differs')
        self.checks['judgement'] = dict(work_seconds=result['work_seconds'], archive_entries=2, archive_returns=2,
            validated_array_members=24, certificate_entries=5, certificate_returns=5, journal_events=15,
            last_event_seconds=times[-1], charged_closure_after_last_event_seconds=result['work_seconds']-times[-1],
            output_file_bytes=sizes, total_output_bytes=sum(sizes.values()), scientific_primary_certified=result['passed'])
        self.checks['timing_boundary'] = 'Frozen runner hashes closed journal and writes/fsyncs/hashes payload before final elapsed time; no clock, fsync or process execution is replayed'
        for rel, expected_sha in {**self.sources, **self.evidence}.items():
            if not rel.endswith('.npz'):
                require(self.digest(rel) == expected_sha, 'Evidence changed during review: ' + rel)
        self.committed([rel for rel in self.evidence if not rel.endswith('.npz')])
        self.checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', action='store_true', required=True)
    parser.parse_args()
    for key, value in dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1').items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, ImportGuard())
    require(not any(name.split('.')[0] in {'numpy', 'torch', 'ale_py', 'ragnarok'} for name in sys.modules),
            'A fresh standard-library-only process is required')
    require(Path(sys.executable).resolve() == Path('C:/Python314/python.exe').resolve()
            and sys.version.split()[0] == '3.14.3' and sys.byteorder == 'little', 'Exact Python and little-endian serialization required')
    receipt_path = ROOT / (FOLDER + 'family_review_v0.json')
    attempt_path = ROOT / (FOLDER + 'family_review_v0_attempt.json')
    require(not receipt_path.exists() and not attempt_path.exists(), 'No review retry; preserve every prior trace')
    review = Review()
    attempt_sha = exclusive_json(attempt_path, dict(kind='stationary_affine_family_review_attempt', created_utc=datetime.now(timezone.utc).isoformat(),
        expected_result_sha256=RESULT_SHA, reviewer_sha256=review.digest(Path(__file__).resolve().relative_to(ROOT).as_posix()),
        max_review_seconds=60, expected_algebraic_recalculations=5, npz_reads=0, no_retry=True))
    failures = []
    try:
        review.perform()
    except BaseException as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    elapsed, cpu = time.monotonic()-review.started, time.process_time()-review.cpu_started
    if elapsed >= MAX_SECONDS:
        failures.append(dict(type='TimeoutError', message='Cumulative review ceiling exceeded'))
    receipt = dict(kind='stationary_affine_family_independent_review', passed=not failures,
        reviewed_utc=datetime.now(timezone.utc).isoformat(), result_sha256=RESULT_SHA,
        source_sha256=review.sources, evidence_sha256=review.evidence,
        review_attempt_sha256=attempt_sha,
        checks=review.checks, failures=failures,
        costs=dict(review_seconds=elapsed, main_process_cpu_seconds=cpu, maximum_seconds=60, cpu_threads=1,
            review_algebraic_recalculations=review.recalculations, judgement_algebraic_calculations=5,
            scalar_endpoint_expressions=6*review.recalculations, exact_extrema_recomputed=2*review.recalculations,
            npz_files_reopened=0, image_decodes=0, model_products=0, gradients=0, optimizer_updates=0,
            new_synthetic_steps=0, new_emulated_frames=0, gpu_used=False,
            scope='All verification through final evidence rehash; final receipt/console excluded; CPU time excludes Git children and is bounded conservatively by the wall ceiling'),
        limits=[
            'Instrument validity is separate from each of the five strict scientific certificates.',
            'Five new algebraic reviews are distinct from the five calculations in the original judgement; no model product was performed.',
            'State bytes are reconstructed only from the already reviewed JSON. NPZ hashes inherit the pinned prior attestations and Git blob identity; current worktree NPZ bytes are not reread.',
            'The claim concerns the exact rational affine function on identical pre-intervention RGB pasts in the declared synthetic family, not all floating-point kernel or sigmoid roundings.',
            'A zero positive minimum alone need not misclassify under >=0.5; strict noncertification must not be equated with every equality being a classification error.',
            'No independent empirical validation, general temporal dependence, Atari perception, CNN repair, game mastery, transfer or accelerated learning is established.',
        ])
    exclusive_json(receipt_path, receipt)
    print(json.dumps(dict(passed=receipt['passed'], failures=failures, review_seconds=elapsed,
        algebraic_recalculations=review.recalculations, receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest()), allow_nan=False))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
