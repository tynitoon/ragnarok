"""One prospective Newton/Adam calibration of the same seven-parameter model."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from ragnarok.learning import affine_response_envelope_v70 as affine
from ragnarok.learning.response_envelope_v70 import measure, feasibility_verdict
from scripts.fit_response_envelope_v70 import (
    sha, sha_bytes, signature, committed, configure_runtime, write_json, check_time)
from scripts.fit_stationary_control_v70 import (
    proofs as stationary_proofs, CONTROL_RESULT, CONTROL_DATA)

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_affine_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_LOGISTIC_PROTOCOL.json'
CNN_RESULT = ROOT / 'craft_v6_out/v70_stationary_fit/fit_v0.json'
CNN_REVIEW = ROOT / 'craft_v6_out/v70_stationary_fit/fit_review_v1.json'
CNN_RESULT_SHA = '8ca241233e8053cedd6149b1d3992dc7cbc944c4442ce298b067dafc03719c35'
CNN_REVIEW_SHA = 'c2fe9c270afa098f43c403b3f4818460bd9540993794049cd7df04f1151daa98'
SOURCES = ('ragnarok/learning/affine_response_envelope_v70.py',
           'tests/test_affine_response_envelope_v70.py',
           'scripts/fit_stationary_affine_v70.py', 'tests/test_stationary_affine_fit_v70.py',
           'docs/ARC4_STATIONARY_LOGISTIC_DRAFT.md', 'docs/ARC4_STATIONARY_LOGISTIC_PROTOCOL.json')
ARMS = ('newton_visual', 'newton_constant', 'adam_visual', 'adam_constant')
OPERATIONS = ('linear_prediction', 'objective', 'gradient', 'hessian',
              'linear_solve', 'adam_proposal', 'parameter_update')
SPEC = dict(
    version=0, primary_recipe='newton', witness_recipe='adam', parameters=7,
    initialization='all seven coefficients and Adam moments zero; no prior weights',
    input='two ordered RGB triples divided by 255, plus bias; shared across pixels and cases',
    dtype='float64', constant_input='all six RGB channels zero, bias one',
    newton_updates_per_arm=16, adam_updates_per_arm=256,
    schedule='k=1..256: Newton visual, constant if k<=16; then Adam visual, constant',
    newton=dict(damping=1.75, full_step=True, objective_penalty=0.,
                loss_increase_tolerance='1e-12 * (1 + abs(loss_before))'),
    adam=dict(lr=.0003, betas=[.9, .999], eps=1e-8, weight_decay=0.,
              gradient_clip='min(1, 1/(norm + 1e-6))'),
    loss='equal anchor mean, equal positive/negative BCE if both exist, full BCE if one class',
    batch='all twelve certified stationary cases in unchanged manifest order',
    input_loading='decode only past_pairs and observed_response_envelopes; no futures or reference',
    evaluation='common initial and four final arms only; full objective also before and after each update',
    controls=['own trained constant', 'common initial', 'zero probability output'],
    probability_threshold=.5, minimum_visual_iou=.25, minimum_advantage=.1,
    maximum_each_empty_false_positive_fraction=.001, required_empty_anchors=2,
    expected_operations=dict(linear_prediction=1093, objective=1093, gradient=544,
                             hessian=32, linear_solve=32, adam_proposal=512, parameter_update=544),
    max_work_seconds=300, cleanup_reserve_seconds=10, max_binary_output_bytes=64000000,
    work_scope='data through scientific payload fsync/hash, final provenance and journal close; final receipt and console excluded',
    cpu_threads=1, gpu_used=False, inherited_synthetic_steps=216,
    new_synthetic_steps=0, new_initial_screens=0, new_emulated_frames=0, retry=False,
    claim='development association fitting only; comparison of two bounded recipes, not equal-compute efficiency or optimizer-only causality')


def proofs():
    """Reuse the certified provenance checks; decode no arrays and build no model."""
    checked = stationary_proofs()
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Prospective affine calibration specification changed')
    if sha(CNN_RESULT) != CNN_RESULT_SHA or sha(CNN_REVIEW) != CNN_REVIEW_SHA:
        raise ValueError('The unchanged reviewed negative CNN fit is required')
    result = json.loads(CNN_RESULT.read_text(encoding='utf-8'))
    review = json.loads(CNN_REVIEW.read_text(encoding='utf-8'))
    if (result['execution_status'] != 'complete' or result['passed'] is not False
            or result['final_provenance_verified'] is not True
            or result['source_sha256'] != checked['source_sha256']
            or review.get('kind') != 'stationary_fit_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != CNN_RESULT_SHA
            or review.get('source_sha256') != result['source_sha256']):
        raise ValueError('The new experiment must preserve the reviewed negative CNN result')
    sources, evidence = dict(checked['source_sha256']), dict(checked['evidence_sha256'])
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('A new calibration cannot replace an earlier source')
        committed(ROOT / rel)
        sources[rel] = sha(ROOT / rel)
    for path in (CNN_RESULT, CNN_REVIEW):
        committed(path)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=checked['runtime'])


def load_dataset(result):
    with np.load(CONTROL_DATA, allow_pickle=False) as data:
        if set(data.files) != {'past_pairs', 'observed_response_envelopes', 'noop_futures', 'action_futures'}:
            raise ValueError('The fixed four-member synthetic archive is required')
        past, targets = data['past_pairs'], data['observed_response_envelopes']
    expected = result['artifacts'][CONTROL_DATA.name]['arrays']
    if (past.shape != (12, 2, 210, 160, 3) or past.dtype != np.uint8
            or targets.shape != (12, 210, 160) or targets.dtype != bool
            or signature(past) != expected['past_pairs']
            or signature(targets) != expected['observed_response_envelopes']
            or targets.sum(axis=(1, 2)).tolist() != [1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2]):
        raise ValueError('Certified input or observed label shape, signature or count changed')
    if [(row['game'], row['anchor_index']) for row in result['cases']] != [
            (f'synthetic_pair_{i // 2}', i % 2) for i in range(12)]:
        raise ValueError('All twelve cases must retain the declared order')
    return past, targets


def operation_counts():
    return {op: {arm: 0 for arm in ('initial', *ARMS)} for op in OPERATIONS}


def expected_counts():
    counts = operation_counts()
    counts['linear_prediction']['initial'] = counts['objective']['initial'] = 1
    for arm in ARMS:
        newton = arm.startswith('newton_')
        steps = SPEC['newton_updates_per_arm' if newton else 'adam_updates_per_arm']
        counts['linear_prediction'][arm] = counts['objective'][arm] = 2 * steps + 1
        counts['gradient'][arm] = counts['parameter_update'][arm] = steps
        counts['hessian'][arm] = counts['linear_solve'][arm] = steps if newton else 0
        counts['adam_proposal'][arm] = 0 if newton else steps
    return counts


def apply_candidate(states, arm, candidate):
    """The only mutable update boundary, including moments and iteration counter."""
    state = states[arm]
    if arm.startswith('newton_'):
        state['coefficients'][:] = candidate
    else:
        coefficients, first, second = candidate
        state['coefficients'][:] = coefficients
        state['first_moment'][:] = first
        state['second_moment'][:] = second
    state['step'] += 1


def finish_result(payload, started, payload_path, result_path):
    failures, payload_hash = list(payload['failures']), None
    try:
        write_json(payload_path, payload)
        payload_hash = sha(payload_path)
    except BaseException as error:
        failures.append(dict(stage='scientific_payload', type=type(error).__name__, message=str(error)))
    elapsed = time.monotonic() - started
    if elapsed >= SPEC['max_work_seconds']:
        failures.append(dict(stage='complete_scientific_payload_time', type='TimeoutError',
                             message='Scientific payload closure exceeded the fixed budget'))
    complete = not failures and payload['candidate_execution_complete'] and elapsed < SPEC['max_work_seconds']
    verdicts = (payload['candidate_verdicts'] if complete else {
        recipe: dict(passed=False, complete=False, reason='Incomplete or failed execution')
        for recipe in ('newton', 'adam')})
    result = dict(payload, kind='stationary_affine_development_fit', failures=failures,
                  execution_status='complete' if complete else 'incomplete',
                  passed=verdicts['newton']['passed'], verdict=verdicts['newton'], verdicts=verdicts,
                  work_seconds=elapsed, scientific_payload_sha256=payload_hash,
                  claim='No independent generalization, CNN repair, game mastery, transfer or acceleration established')
    write_json(result_path, result)
    return result


def run():
    configure_runtime()
    checked = proofs()
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight_hash, preflight = sha_bytes(preflight_bytes), json.loads(preflight_bytes)
    if (preflight.get('kind') != 'stationary_affine_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact committed independent prospective affine preflight required')
    committed(PREFLIGHT)
    names = ['fit_v0_attempt.json', 'fit_v0_progress.jsonl', 'fit_v0_payload.json', 'fit_v0.json',
             'initial_state.npz', 'final_state.npz']
    names += [f'{arm}_{kind}.npz' for arm in ('initial', *ARMS) for kind in ('logits', 'predictions')]
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve all prior affine traces; no retry')
    control = json.loads(CONTROL_RESULT.read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='stationary_affine_fit_attempt',
               created_utc=datetime.now(timezone.utc).isoformat(), protocol=SPEC,
               preflight_sha256=preflight_hash, **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    states, artifacts, metrics, failures = {}, {}, {}, []
    entries, returns, pending = operation_counts(), operation_counts(), {'arm': None}
    journal, final_provenance_verified = None, False

    def fail(stage, error):
        failures.append(dict(stage=stage, type=type(error).__name__, message=str(error)))

    def event(kind, **fields):
        if journal is None:
            raise OSError('The durable journal was not opened')
        journal.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started,
                                     **fields), allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())

    def safe_event(kind, **fields):
        try:
            event(kind, **fields)
        except BaseException as error:
            fail('journal_' + kind, error)

    def save_binary(name, arrays):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        data = buffer.getvalue()
        if len(data) + sum(row['bytes'] for row in artifacts.values()) > SPEC['max_binary_output_bytes']:
            raise ValueError('Fixed binary output ceiling exceeded')
        path = FOLDER / name
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(path) != sha_bytes(data) or path.stat().st_size != len(data):
            raise ValueError('Saved binary differs from its serialized payload')
        artifacts[name] = dict(bytes=len(data), sha256=sha_bytes(data),
                               arrays={key: signature(value) for key, value in arrays.items()})

    def operation(kind, arm, function, preserve=None):
        check_time(deadline)
        event('operation_enter', operation=kind, arm=arm, ordinal=entries[kind][arm] + 1)
        entries[kind][arm] += 1
        if kind == 'parameter_update':
            pending['arm'] = arm
        value = function()
        returns[kind][arm] += 1
        if kind == 'parameter_update':
            pending['arm'] = None
        # Retain the actual unique evaluation result before a fallible journal or metric.
        if preserve is not None:
            preserve(value)
        event('operation_return', operation=kind, arm=arm, ordinal=returns[kind][arm])
        return value

    def snapshot(name):
        arrays = {}
        for arm, state in states.items():
            for key, value in state.items():
                arrays[f'{arm}_{key}'] = np.asarray(value).copy()
        save_binary(name, arrays)

    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        past, targets = load_dataset(control)
        visual = affine.pixel_features(past)
        constant = np.zeros_like(visual)
        constant[:, -1] = 1.
        labels, weight = affine.weighted_labels(targets)
        for features in (visual, constant):
            affine.validate_problem(features, labels, weight)
        inputs = {arm: visual if arm.endswith('_visual') else constant for arm in ARMS}
        states = {arm: dict(coefficients=np.zeros(7, np.float64), step=0) for arm in ARMS}
        for arm in ('adam_visual', 'adam_constant'):
            states[arm].update(first_moment=np.zeros(7, np.float64), second_moment=np.zeros(7, np.float64))
        snapshot('initial_state.npz')

        def evaluate(arm, features, coefficients):
            logits = operation('linear_prediction', arm, lambda: features @ coefficients,
                lambda raw: save_binary(arm + '_logits.npz', dict(logits=raw.reshape(targets.shape))))
            probability = affine.probabilities(logits).reshape(targets.shape)
            save_binary(arm + '_predictions.npz', dict(probability=probability))
            bce = operation('objective', arm, lambda: affine.objective(logits, labels, weight))
            if not np.isfinite(bce):
                raise FloatingPointError('Nonfinite evaluation objective')
            measured = measure(probability, targets, control['cases'])
            measured['balanced_bce'] = bce
            return measured, probability

        metrics['initial'], initial = evaluate('initial', visual, states['newton_visual']['coefficients'])
        metrics['zero'] = measure(np.zeros_like(initial), targets, control['cases'])
        for step in range(1, SPEC['adam_updates_per_arm'] + 1):
            active = ARMS if step <= SPEC['newton_updates_per_arm'] else ARMS[2:]
            for arm in active:
                features, state = inputs[arm], states[arm]
                before = state['coefficients'].copy()
                logits = operation('linear_prediction', arm, lambda: features @ before)
                loss_before = operation('objective', arm, lambda: affine.objective(logits, labels, weight))
                if not np.isfinite(loss_before):
                    raise FloatingPointError('Nonfinite objective before update')
                probability = affine.probabilities(logits)
                first = operation('gradient', arm, lambda: affine.gradient(features, probability, labels, weight))
                if arm.startswith('newton_'):
                    second = operation('hessian', arm, lambda: affine.hessian(features, probability, weight))
                    candidate = operation('linear_solve', arm, lambda: affine.newton_candidate(before, first, second))
                else:
                    candidate = operation('adam_proposal', arm, lambda: affine.adam_candidate(
                        before, first, state['first_moment'], state['second_moment'], state['step'] + 1))
                operation('parameter_update', arm, lambda: apply_candidate(states, arm, candidate))
                logits_after = operation('linear_prediction', arm, lambda: features @ state['coefficients'])
                loss_after = operation('objective', arm, lambda: affine.objective(logits_after, labels, weight))
                if not np.isfinite(loss_after):
                    raise FloatingPointError('Nonfinite objective after counted update')
                event('update_complete', arm=arm, update=state['step'], loss_before=loss_before,
                      loss_after=loss_after, gradient_norm=float(np.linalg.norm(first)),
                      step_norm=float(np.linalg.norm(state['coefficients'] - before)))
                if (arm.startswith('newton_') and loss_after > loss_before + 1e-12 * (1 + abs(loss_before))):
                    raise FloatingPointError('Newton objective increased beyond the fixed numerical tolerance')
        for arm in ARMS:
            metrics[arm], _ = evaluate(arm, inputs[arm], states[arm]['coefficients'])
        check_time(deadline)
    except BaseException as error:
        fail('scientific_work', error)
        safe_event('failed', parameter_update_returns=returns['parameter_update'], pending_update_arm=pending['arm'])
    finally:
        try:
            if states:
                snapshot('final_state.npz')
        except BaseException as error:
            fail('preserve_state', error)
        try:
            if proofs() != checked or sha(PREFLIGHT) != preflight_hash:
                raise ValueError('Final sources, evidence, runtime or preflight changed')
            committed(PREFLIGHT)
            for name, artifact in artifacts.items():
                if sha(FOLDER / name) != artifact['sha256']:
                    raise ValueError('Scientific output changed during fitting')
            check_time(started + SPEC['max_work_seconds'])
            final_provenance_verified = True
        except BaseException as error:
            fail('final_provenance_or_time', error)
        safe_event('finished', operation_entries=entries, operation_returns=returns,
                   pending_update_arm=pending['arm'], failures=failures.copy())
        if journal is not None:
            try:
                journal.close()
            except BaseException as error:
                fail('journal_close', error)

    candidate_complete = (not failures and final_provenance_verified and pending['arm'] is None
                          and entries == returns == expected_counts()
                          and set(metrics) == {'initial', 'zero', *ARMS})
    candidate_verdicts = {}
    for recipe in ('newton', 'adam'):
        recipe_metrics = {key: metrics.get(key if key in ('initial', 'zero') else recipe + '_' + key)
                          for key in ('visual', 'constant', 'initial', 'zero')}
        counts = {key: returns['parameter_update'][recipe + '_' + key] for key in ('visual', 'constant')}
        candidate_verdicts[recipe] = (feasibility_verdict(recipe_metrics, counts, SPEC[recipe + '_updates_per_arm'])
                                     if candidate_complete else dict(passed=False, complete=False, reason='Incomplete execution'))
    totals = {kind: sum(counts.values()) for kind, counts in returns.items()}
    payload = dict(kind='stationary_affine_scientific_payload', **checked, protocol=SPEC,
                   preflight_sha256=preflight_hash, failures=failures, artifacts=artifacts, metrics=metrics,
                   cases=control['cases'], candidate_execution_complete=candidate_complete,
                   candidate_verdicts=candidate_verdicts, operation_entries=entries, operation_returns=returns,
                   operation_totals=totals, operation_count_exact=entries == returns,
                   completed_updates={arm: returns['parameter_update'][arm] for arm in ARMS},
                   pending_update_arm=pending['arm'], optimizer_update_count_exact=pending['arm'] is None,
                   optimizer_updates=totals['parameter_update'],
                   optimizer_updates_upper_bound=totals['parameter_update'] + int(pending['arm'] is not None),
                   final_provenance_verified=final_provenance_verified,
                   attempt_sha256=sha(FOLDER / names[0]),
                   progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None,
                   inherited_synthetic_steps=216, new_synthetic_steps=0, new_initial_screens=0,
                   new_emulated_frames=0, gpu_used=False,
                   claim='Scientific payload only; require final receipt after charged closure')
    result = finish_result(payload, started, FOLDER / 'fit_v0_payload.json', FOLDER / 'fit_v0.json')
    print(json.dumps(dict(execution_status=result['execution_status'], verdicts=result['verdicts'],
          operation_totals=totals, work_seconds=result['work_seconds']), allow_nan=False))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
