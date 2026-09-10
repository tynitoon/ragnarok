"""Bounded independent audit of saved affine results; never rerun a model.

Commit this verifier before its first scientific read.  It imports no project
module or Torch, performs no feature/parameter product, gradient, Hessian or
optimization, and decodes only the admitted saved arrays.  An exclusive review
intent and a separate failure artifact prohibit silent review retries.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_affine_fit'
CONTROL = ROOT / 'craft_v6_out/v70_stationary_control'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_LOGISTIC_PROTOCOL.json'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PREFLIGHT_SHA = '97268ef10d26571f65febf86eb713f527574dc891f5187e8d016126de824992e'
SOURCE_COMMIT = 'd39e496'
PREFLIGHT_COMMIT = '110a599'
ARMS = ('newton_visual', 'newton_constant', 'adam_visual', 'adam_constant')
OPS = ('linear_prediction', 'objective', 'gradient', 'hessian', 'linear_solve',
       'adam_proposal', 'parameter_update')
EXPECTED_TOTALS = dict(linear_prediction=1093, objective=1093, gradient=544,
                       hessian=32, linear_solve=32, adam_proposal=512, parameter_update=544)
TARGET_COUNTS = [1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2]
MAX_SECONDS = 120
SCALAR_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def write_new(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == data, 'Receipt serialization changed')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def git_date(commit):
    return datetime.fromisoformat(git('show', '-s', '--format=%cI', commit))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def assert_blob(path, commit='HEAD'):
    rel = relative(path)
    actual = git('hash-object', '--path=' + rel, str(path))
    expected = git('rev-parse', commit + ':' + rel)
    require(actual == expected, 'Working file differs from committed blob: ' + rel)


def close_numbers(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': mapping keys')
        for key in expected:
            close_numbers(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': sequence length')
        for index, value in enumerate(expected):
            close_numbers(actual[index], value, label + '/' + str(index))
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and abs(actual - expected) <= SCALAR_TOLERANCE * (1 + abs(expected)), label + ': scalar mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact mismatch')


def signature(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def archive_members(path, expected):
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        require(len(rows) == len(expected), 'NPZ member count differs')
        require({row.filename for row in rows} == {name + '.npy' for name in expected}, 'NPZ member names differ')
        require(all(not row.is_dir() and not row.flag_bits & 1 for row in rows), 'Encrypted or directory NPZ member')
        require(sum(row.file_size for row in rows) <= 64000000, 'Uncompressed archive exceeds the fixed ceiling')


def load_saved(path, descriptions, allowed=None):
    archive_members(path, descriptions)
    selected = tuple(descriptions) if allowed is None else tuple(allowed)
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in selected:
            value = archive[name]
            require(signature(value) == descriptions[name], 'Array signature differs: ' + path.name + '/' + name)
            require(value.dtype.kind in 'biuf' and np.isfinite(value).all(), 'Nonfinite or object array')
            arrays[name] = value
    return arrays


def expected_counts():
    counts = {op: {arm: 0 for arm in ('initial', *ARMS)} for op in OPS}
    counts['linear_prediction']['initial'] = counts['objective']['initial'] = 1
    for arm in ARMS:
        steps = 16 if arm.startswith('newton_') else 256
        counts['linear_prediction'][arm] = counts['objective'][arm] = 2 * steps + 1
        counts['gradient'][arm] = counts['parameter_update'][arm] = steps
        counts['hessian'][arm] = counts['linear_solve'][arm] = steps if steps == 16 else 0
        counts['adam_proposal'][arm] = 0 if steps == 16 else steps
    return counts


def audit_journal(path, work_seconds, checkpoint):
    seen = {op: {arm: 0 for arm in ('initial', *ARMS)} for op in OPS}
    previous_time, lines, newton_increases = 0., 0, 0
    first_loss, last_loss = {}, {}
    maximum_continuity_error, maximum_newton_increase = 0., 0.
    steps = {arm: 0 for arm in ARMS}
    with path.open('r', encoding='utf-8') as stream:
        def consume():
            nonlocal previous_time, lines
            checkpoint()
            text = stream.readline()
            require(bool(text) and text.endswith(chr(10)), 'Truncated journal event')
            row = json.loads(text)
            stamp = row['elapsed_seconds']
            require(isinstance(stamp, (float, int)) and math.isfinite(stamp)
                    and previous_time <= stamp <= work_seconds, 'Nonmonotone or uncharged journal timestamp')
            previous_time = stamp
            lines += 1
            return row

        def operation(kind, arm):
            seen[kind][arm] += 1
            for event in ('operation_enter', 'operation_return'):
                row = consume()
                require(set(row) == {'event', 'elapsed_seconds', 'operation', 'arm', 'ordinal'}, 'Unexpected operation fields')
                require((row['event'], row['operation'], row['arm'], row['ordinal'])
                        == (event, kind, arm, seen[kind][arm]), 'Operation agenda or ordinal differs')

        for kind in ('linear_prediction', 'objective'):
            operation(kind, 'initial')
        for step in range(1, 257):
            for arm in (ARMS if step <= 16 else ARMS[2:]):
                agenda = ['linear_prediction', 'objective', 'gradient']
                agenda += ['hessian', 'linear_solve'] if arm.startswith('newton_') else ['adam_proposal']
                agenda += ['parameter_update', 'linear_prediction', 'objective']
                for kind in agenda:
                    operation(kind, arm)
                row = consume()
                require(set(row) == {'event', 'elapsed_seconds', 'arm', 'update', 'loss_before',
                                    'loss_after', 'gradient_norm', 'step_norm'}, 'Unexpected update fields')
                require((row['event'], row['arm'], row['update']) == ('update_complete', arm, step), 'Update schedule differs')
                for key in ('loss_before', 'loss_after', 'gradient_norm', 'step_norm'):
                    require(isinstance(row[key], (float, int)) and math.isfinite(row[key]) and row[key] >= 0,
                            'Invalid saved loss or norm')
                before, after = row['loss_before'], row['loss_after']
                if arm in last_loss:
                    difference = abs(before - last_loss[arm])
                    maximum_continuity_error = max(maximum_continuity_error, difference)
                    require(difference <= SCALAR_TOLERANCE * (1 + abs(before)), 'Loss discontinuity between successive updates')
                else:
                    first_loss[arm] = before
                if arm.startswith('newton_'):
                    increase = after - before
                    maximum_newton_increase = max(maximum_newton_increase, increase)
                    newton_increases += int(increase > 0)
                    require(increase <= 1e-12 * (1 + abs(before)), 'Frozen Newton non-growth guard violated')
                last_loss[arm] = after
                steps[arm] += 1
        for arm in ARMS:
            for kind in ('linear_prediction', 'objective'):
                operation(kind, arm)
        finished = consume()
        require(set(finished) == {'event', 'elapsed_seconds', 'operation_entries', 'operation_returns',
                                 'pending_update_arm', 'failures'}, 'Unexpected final event fields')
        require(finished['event'] == 'finished' and finished['failures'] == []
                and finished['pending_update_arm'] is None, 'Incomplete journal closure')
        require(finished['operation_entries'] == finished['operation_returns'] == seen == expected_counts(), 'Final journal counters differ')
        require(stream.read() == '', 'Unexpected event after closure')
    require(lines == 8245, 'Unexpected exhaustive journal population')
    return dict(events=lines, exact_schedule=True, operation_counts=seen, updates=steps,
                first_loss=first_loss, last_loss=last_loss,
                maximum_continuity_error=maximum_continuity_error,
                maximum_newton_increase=maximum_newton_increase,
                positive_newton_increases_within_tolerance=newton_increases,
                final_event_seconds=previous_time)


def average_precision(scores, targets):
    positives = int(targets.sum())
    if not positives:
        return None
    # Independent aggregation by unique score, rather than sorted-index endpoints.
    _, inverse, population = np.unique(scores.reshape(-1), return_inverse=True, return_counts=True)
    positive = np.bincount(inverse, weights=targets.reshape(-1).astype(np.float64),
                           minlength=len(population))[::-1]
    population = population[::-1]
    precision = np.cumsum(positive) / np.cumsum(population)
    return float(math.fsum((precision * (positive / positives)).tolist()))


def metrics(probability, targets, cases):
    rows, grouped = [], {}
    for scores, target, case in zip(probability, targets, cases, strict=True):
        predicted = scores >= .5
        actual = int(target.sum())
        tp, fp = int(np.count_nonzero(predicted & target)), int(np.count_nonzero(predicted & ~target))
        union, negatives = int(np.count_nonzero(predicted | target)), int(np.count_nonzero(~target))
        row = dict(game=case['game'], anchor_index=case['anchor_index'], target_pixels=actual,
                   true_positive_pixels=tp, false_positive_pixels=fp,
                   iou=tp / union if actual else None,
                   average_precision=average_precision(scores, target),
                   false_positive_fraction=fp / negatives if negatives else None)
        rows.append(row)
        if actual:
            grouped.setdefault(case['game'], []).append(row)
    games = {name: dict(iou=math.fsum(row['iou'] for row in values) / len(values),
                       average_precision=math.fsum(row['average_precision'] for row in values) / len(values))
             for name, values in grouped.items()}
    return dict(cases=rows, nonempty_by_game=games,
                mean_nonempty_iou=math.fsum(value['iou'] for value in games.values()) / len(games),
                mean_nonempty_average_precision=math.fsum(value['average_precision'] for value in games.values()) / len(games),
                empty_anchor_false_positive_fractions=[row['false_positive_fraction'] for row in rows if not row['target_pixels']])


def balanced_bce(logits, targets):
    losses = np.logaddexp(0., np.where(targets, -logits, logits))
    values = []
    for loss, target in zip(losses, targets, strict=True):
        if target.any() and (~target).any():
            values.append(.5 * float(loss[target].mean()) + .5 * float(loss[~target].mean()))
        else:
            values.append(float(loss.mean()))
    return math.fsum(values) / len(values)


def verdict(recipe, measured):
    visual = measured[recipe + '_visual']
    comparison = max(measured[key]['mean_nonempty_iou']
                     for key in (recipe + '_constant', 'initial', 'zero'))
    advantage = visual['mean_nonempty_iou'] - comparison
    absolute = visual['mean_nonempty_iou'] >= .25
    specificity = len(visual['empty_anchor_false_positive_fractions']) == 2 and all(
        value <= .001 for value in visual['empty_anchor_false_positive_fractions'])
    return dict(passed=absolute and advantage >= .10 and specificity, complete=True,
                minimum_iou_pass=absolute, advantage_over_best_control=advantage,
                minimum_advantage_pass=advantage >= .10, each_empty_anchor_pass=specificity,
                claim='Development-set fitting feasibility only; not independent generalization')


def review(expected_result_sha, result_commit):
    started = time.monotonic()
    def checkpoint():
        require(time.monotonic() - started < MAX_SECONDS - 3, 'Review verification budget exhausted')
    attempt_path, failure_path = FOLDER / 'fit_review_v0_attempt.json', FOLDER / 'fit_review_v0_failure.json'
    receipt_path = FOLDER / 'fit_review_v0.json'
    require(not any(path.exists() for path in (attempt_path, failure_path, receipt_path)), 'Preserve previous review; no retry')
    require(Path(sys.executable).resolve() == Path('C:/Python314/python.exe').resolve(), 'Exact reviewer Python required')
    require(sys.version.split()[0] == '3.14.3' and np.__version__ == '2.4.4', 'Exact reviewer Python/NumPy versions required')
    require({key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
            == dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), 'CPU-only one-thread review required')
    require(not any(name.split('.')[0] in ('torch', 'ale_py', 'ragnarok') for name in sys.modules), 'No Torch/emulator/scientific module imports')
    assert_blob(Path(__file__))
    reviewer_commit = git('log', '-1', '--format=%H', '--', relative(Path(__file__)))
    source_commit, preflight_commit = git('rev-parse', SOURCE_COMMIT), git('rev-parse', PREFLIGHT_COMMIT)
    result_commit = git('rev-parse', result_commit)
    for older, newer in ((source_commit, preflight_commit), (preflight_commit, result_commit), (result_commit, 'HEAD')):
        subprocess.check_call(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', older, newer])
    intent = dict(kind='stationary_affine_independent_review_attempt', status='started',
                  created_utc=datetime.now(timezone.utc).isoformat(), expected_result_sha256=expected_result_sha,
                  result_commit=result_commit, verifier_commit=reviewer_commit,
                  verifier_sha256=digest(Path(__file__)), maximum_verification_seconds=MAX_SECONDS,
                  no_retry=True, optimization_or_model_products_authorized=False)
    write_new(attempt_path, intent)
    evidence = {relative(attempt_path): digest(attempt_path), relative(Path(__file__)): digest(Path(__file__))}
    try:
        checkpoint()
        require(digest(PREFLIGHT) == PREFLIGHT_SHA, 'Frozen preflight SHA differs')
        preflight = read_json(PREFLIGHT)
        require(preflight['kind'] == 'stationary_affine_independent_preflight' and preflight['passed'] is True, 'Independent preflight not passed')
        require(len(preflight['source_sha256']) == 112 and len(preflight['evidence_sha256']) == 5, 'Unexpected frozen proof population')
        for rel, expected in {**preflight['source_sha256'], **preflight['evidence_sha256']}.items():
            checkpoint()
            path = ROOT / rel
            require(digest(path) == expected, 'Frozen SHA differs: ' + rel)
            assert_blob(path, source_commit if rel in preflight['source_sha256'] else result_commit)
            assert_blob(path)
        evidence.update(preflight['evidence_sha256'])
        assert_blob(PREFLIGHT, preflight_commit)
        require(importlib.metadata.version('torch') == preflight['runtime']['torch'], 'Installed Torch metadata changed')

        result_path, payload_path = FOLDER / 'fit_v0.json', FOLDER / 'fit_v0_payload.json'
        fit_attempt_path, journal_path = FOLDER / 'fit_v0_attempt.json', FOLDER / 'fit_v0_progress.jsonl'
        require(digest(result_path) == expected_result_sha, 'Result SHA differs from the supplied closed result')
        result, payload, attempt = read_json(result_path), read_json(payload_path), read_json(fit_attempt_path)
        protocol = read_json(PROTOCOL)
        require(protocol['expected_operations'] == EXPECTED_TOTALS and protocol['primary_recipe'] == 'newton', 'Fixed primary recipe or populations differ')
        require(protocol['parameters'] == 7 and protocol['newton_updates_per_arm'] == 16
                and protocol['adam_updates_per_arm'] == 256 and protocol['max_work_seconds'] == 300
                and protocol['newton']['damping'] == 1.75 and protocol['newton']['objective_penalty'] == 0., 'Prospective solver or budget differs')
        require(attempt['kind'] == 'stationary_affine_fit_attempt', 'Wrong fit attempt kind')
        for item in (attempt, payload, result):
            for key in ('source_sha256', 'evidence_sha256', 'runtime'):
                require(item[key] == preflight[key], 'Attempt/result proof field differs: ' + key)
            require(item['protocol'] == protocol and item['preflight_sha256'] == PREFLIGHT_SHA, 'Attempt/result protocol differs')
        for key, value in payload.items():
            if key not in ('kind', 'claim'):
                require(result[key] == value, 'Result differs from scientific payload: ' + key)
        require(result['kind'] == 'stationary_affine_development_fit' and payload['kind'] == 'stationary_affine_scientific_payload', 'Wrong result/payload kind')
        require(result['execution_status'] == 'complete' and result['failures'] == []
                and result['candidate_execution_complete'] is True and result['final_provenance_verified'] is True, 'Incomplete result requires a separately scoped review')
        require(result['scientific_payload_sha256'] == digest(payload_path)
                and result['attempt_sha256'] == digest(fit_attempt_path)
                and result['progress_sha256'] == digest(journal_path), 'Attempt/payload/journal linkage differs')
        require(0 < result['work_seconds'] < 300 and result['pending_update_arm'] is None
                and result['optimizer_update_count_exact'] is True and result['operation_count_exact'] is True, 'Uncertain or over-budget work')
        require(result['operation_entries'] == result['operation_returns'] == expected_counts(), 'Reported per-arm operations differ')
        require(result['operation_totals'] == EXPECTED_TOTALS and result['optimizer_updates'] == 544
                and result['optimizer_updates_upper_bound'] == 544, 'Reported total operations differ')
        require(result['completed_updates'] == {arm: 16 if arm.startswith('newton_') else 256 for arm in ARMS}, 'Reported update agenda differs')
        require(result['inherited_synthetic_steps'] == 216 and result['new_synthetic_steps'] == 0
                and result['new_initial_screens'] == 0 and result['new_emulated_frames'] == 0 and result['gpu_used'] is False, 'Unexpected new data or compute scope')
        start_utc = datetime.fromisoformat(attempt['created_utc'])
        require(git_date(source_commit) <= git_date(preflight_commit) < start_utc, 'Sources/preflight were not committed before the fit attempt')
        # Git commit dates have one-second resolution, unlike the attempt clock.
        require(git_date(result_commit) + timedelta(seconds=1) >= start_utc + timedelta(seconds=result['work_seconds']), 'Result commit predates declared work completion')

        artifact_names = {'initial_state.npz', 'final_state.npz'} | {
            arm + '_' + kind + '.npz' for arm in ('initial', *ARMS) for kind in ('logits', 'predictions')}
        require(set(result['artifacts']) == artifact_names, 'Unexpected or missing saved scientific arrays')
        require(sum(row['bytes'] for row in result['artifacts'].values()) <= 64000000, 'Scientific binary cap exceeded')
        for name, description in result['artifacts'].items():
            path = FOLDER / name
            require(path.stat().st_size == description['bytes'] and digest(path) == description['sha256'], 'Saved artifact bytes/SHA differ: ' + name)
        # Include every committed local evidence file, including any console/dispatch.
        tracked = git('ls-files', '--', relative(FOLDER)).splitlines()
        for rel in tracked:
            if Path(rel).name.startswith('fit_review_v0'):
                continue
            checkpoint()
            path = ROOT / rel
            assert_blob(path, result_commit)
            assert_blob(path)
            evidence[rel] = digest(path)

        journal = audit_journal(journal_path, result['work_seconds'], checkpoint)
        control = read_json(CONTROL / 'control_v0.json')
        data_description = control['artifacts']['control_v0_dataset.npz']
        data_path = CONTROL / 'control_v0_dataset.npz'
        require(digest(data_path) == data_description['sha256'], 'Control data linkage differs')
        observed = load_saved(data_path, data_description['arrays'], ('past_pairs', 'observed_response_envelopes'))
        past, targets = observed['past_pairs'], observed['observed_response_envelopes']
        require(past.dtype == np.uint8 and past.shape == (12, 2, 210, 160, 3)
                and targets.dtype == bool and targets.shape == (12, 210, 160), 'Control shape/dtype differs')
        require(np.array_equal(past[:, 0], past[:, 1]) and targets.sum(axis=(1, 2)).tolist() == TARGET_COUNTS, 'Stationarity or declared observed label counts differ')
        cases = [dict(game=f'synthetic_pair_{index // 2}', anchor_index=index % 2) for index in range(12)]
        require(result['cases'] == control['cases'] == cases, 'Synthetic case order differs')
        del past, observed

        state_summary = {}
        state_keys = {arm + '_' + key for arm in ARMS for key in
                      (('coefficients', 'step') if arm.startswith('newton_') else
                       ('coefficients', 'step', 'first_moment', 'second_moment'))}
        for name in ('initial_state.npz', 'final_state.npz'):
            checkpoint()
            arrays = load_saved(FOLDER / name, result['artifacts'][name]['arrays'])
            require(set(arrays) == state_keys, 'State member population differs')
            for key, value in arrays.items():
                if key.endswith('_step'):
                    require(value.shape == () and value.dtype == np.int64, 'State iteration dtype/shape differs')
                    expected = 0 if name == 'initial_state.npz' else (16 if key.startswith('newton_') else 256)
                    require(int(value) == expected, 'State iteration differs')
                else:
                    require(value.shape == (7,) and value.dtype == np.float64, 'Seven finite float64 state values required')
                    if key.endswith('_second_moment'):
                        require(np.all(value >= 0), 'Negative second moment')
                    if name == 'initial_state.npz':
                        require(np.count_nonzero(value) == 0, 'Nonzero initialization')
            state_summary[name] = {key: value.tolist() for key, value in arrays.items()}

        recomputed, probability_errors = {}, {}
        for arm in ('initial', *ARMS):
            checkpoint()
            logits_name, probability_name = arm + '_logits.npz', arm + '_predictions.npz'
            saved_logits = load_saved(FOLDER / logits_name, result['artifacts'][logits_name]['arrays'])
            saved_probability = load_saved(FOLDER / probability_name, result['artifacts'][probability_name]['arrays'])
            require(set(saved_logits) == {'logits'} and set(saved_probability) == {'probability'}, 'Wrong evaluation member')
            logits, probability = saved_logits['logits'], saved_probability['probability']
            require(logits.shape == probability.shape == targets.shape
                    and logits.dtype == probability.dtype == np.float64, 'Evaluation shape/dtype differs')
            require(np.all((probability >= 0) & (probability <= 1)), 'Out-of-range probability')
            regenerated = np.exp(-np.logaddexp(0., -logits))
            require(np.array_equal(regenerated, probability), 'Saved probabilities differ from declared sigmoid of saved logits')
            independent = np.empty_like(logits)
            nonnegative = logits >= 0
            independent[nonnegative] = 1 / (1 + np.exp(-logits[nonnegative]))
            exp_negative = np.exp(logits[~nonnegative])
            independent[~nonnegative] = exp_negative / (1 + exp_negative)
            error = float(np.max(np.abs(independent - probability)))
            require(error <= 2e-15, 'Independent stable sigmoid check failed')
            probability_errors[arm] = error
            if arm == 'initial':
                require(np.count_nonzero(logits) == 0 and np.all(probability == .5), 'Common initial prediction is not zero-logit')
            elif arm.endswith('_constant'):
                require(np.all(logits == logits.flat[0]), 'Constant input produced spatial/case variation')
            measured = metrics(regenerated, targets, cases)
            measured['balanced_bce'] = balanced_bce(logits, targets)
            close_numbers(result['metrics'][arm], measured, 'metrics/' + arm)
            recomputed[arm] = measured
        recomputed['zero'] = metrics(np.zeros(targets.shape, np.float64), targets, cases)
        close_numbers(result['metrics']['zero'], recomputed['zero'], 'metrics/zero')
        require(set(result['metrics']) == set(recomputed), 'Metric population differs')
        for arm in ARMS:
            close_numbers(journal['first_loss'][arm], recomputed['initial']['balanced_bce'], 'initial loss/' + arm)
            close_numbers(journal['last_loss'][arm], recomputed[arm]['balanced_bce'], 'final loss/' + arm)
        verdicts = {recipe: verdict(recipe, recomputed) for recipe in ('newton', 'adam')}
        close_numbers(result['verdicts'], verdicts, 'verdicts')
        close_numbers(result['candidate_verdicts'], verdicts, 'candidate verdicts')
        close_numbers(result['verdict'], verdicts['newton'], 'primary verdict')
        require(result['passed'] == verdicts['newton']['passed'], 'Primary status differs')

        for rel, expected in {**preflight['source_sha256'], **evidence}.items():
            checkpoint()
            require(digest(ROOT / rel) == expected, 'Evidence changed during review: ' + rel)
        checkpoint()
        receipt = dict(kind='stationary_affine_independent_result_review', passed=True,
                       reviewed_utc=datetime.now(timezone.utc).isoformat(), result_sha256=expected_result_sha,
                       source_sha256=result['source_sha256'], evidence_sha256=evidence,
                       attempt_sha256=result['attempt_sha256'], preflight_sha256=PREFLIGHT_SHA,
                       scientific_primary_passed=result['passed'], scientific_verdicts=verdicts,
                       verifier_commit=reviewer_commit, source_commit=source_commit,
                       preflight_commit=preflight_commit, result_commit=result_commit,
                       reviewer_runtime=dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                                             torch_imported=False, cpu_threads=1, cuda_visible_devices=''),
                       checks=dict(all_sources_and_evidence_unchanged=True, committed_before_attempt=True,
                                   payload_attempt_journal_links=True, exact_operation_totals=EXPECTED_TOTALS,
                                   exhaustive_journal=journal, state_arrays=state_summary,
                                   observed_target_counts=TARGET_COUNTS, observed_target_pixels=int(targets.sum()),
                                   independent_sigmoid_maximum_errors=probability_errors,
                                   recomputed_metrics=recomputed,
                                   true_positive_totals={arm: sum(row['true_positive_pixels'] for row in recomputed[arm]['cases']) for arm in ARMS},
                                   false_positive_totals={arm: sum(row['false_positive_pixels'] for row in recomputed[arm]['cases']) for arm in ARMS},
                                   scalar_comparison_tolerance=SCALAR_TOLERANCE),
                       costs=dict(fit_work_seconds=result['work_seconds'], fit_parameter_updates=544,
                                  review_verification_seconds=time.monotonic() - started, review_max_seconds=MAX_SECONDS,
                                  optimization_steps=0, feature_parameter_products=0, gradients_recomputed=0,
                                  hessians_recomputed=0, new_images=0, emulated_frames=0, gpu_used=False,
                                  decoded_control_members=['past_pairs', 'observed_response_envelopes'],
                                  decoded_fit_archives=12),
                       limitations=[
                           'Instrumental review PASS is distinct from each predeclared scientific recipe verdict.',
                           'No feature matrix, feature/parameter product, gradient, Hessian, solver or optimizer was reconstructed or executed.',
                           'Saved parameters and moments were checked structurally and for finitude; their entire numerical trajectory was not replayed.',
                           'All 544 saved update summaries and operation entry/return events were checked; the Newton loss guard and objective continuity are checked from preserved scalars.',
                           'The declared sigmoid is reproduced from saved logits; independent unique-score AP and per-anchor BCE reductions can differ by harmless final rounding within the stated fixed tolerance.',
                           'Only the two permitted control members were decoded; future branches and certificate weights/logits were not loaded.',
                           'Twelve synthetic development cases, with identical past frames; no independent generalization, temporal-order skill, CNN repair, reusable perception, game control, transfer or acceleration established.',
                           'A macro PASS can coexist with individual missed targets. Adam is the declared witness; Newton remains primary.',
                           'Git chronology and immutable file hashes establish the available provenance, not absolute forensic guarantees against an unrecorded external process.',
                           'The 120-second review cap covers verification; only final receipt serialization and console reporting follow the recorded verification time.'])
        write_new(receipt_path, receipt)
        print(json.dumps(dict(passed=True, receipt=str(receipt_path), sha256=digest(receipt_path),
                              sources=len(receipt['source_sha256']), evidence=len(evidence),
                              review_seconds=receipt['costs']['review_verification_seconds'],
                              scientific_verdicts=verdicts,
                              true_positive_totals=receipt['checks']['true_positive_totals']), allow_nan=False))
        return 0
    except BaseException as error:
        failure = dict(kind='stationary_affine_independent_review_failure', passed=False,
                       created_utc=datetime.now(timezone.utc).isoformat(), verifier_commit=reviewer_commit,
                       expected_result_sha256=expected_result_sha, review_attempt_sha256=digest(attempt_path),
                       error=dict(type=type(error).__name__, message=str(error)),
                       elapsed_seconds=time.monotonic() - started, evidence_checked=evidence,
                       retry_authorized=False)
        write_new(failure_path, failure)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-result-sha256', required=True)
    parser.add_argument('--result-commit', required=True)
    args = parser.parse_args()
    require(len(args.expected_result_sha256) == 64 and all(ch in '0123456789abcdef' for ch in args.expected_result_sha256), 'Explicit lowercase result SHA required')
    raise SystemExit(review(args.expected_result_sha256, args.result_commit))
