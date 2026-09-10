"""One bounded Newton calibration of seven local RGB coefficients on archived Atari targets."""
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
    sha, sha_bytes, signature, committed, configure_runtime, write_json, check_time,
    proofs as target_proofs, TARGET_RESULT, TARGET_DATA)

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_atari_affine_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_ATARI_AFFINE_PROTOCOL.json'
SOURCES = ('ragnarok/learning/affine_response_envelope_v70.py',
           'tests/test_affine_response_envelope_v70.py',
           'scripts/fit_atari_affine_v70.py', 'tests/test_atari_affine_fit_v70.py',
           'docs/ARC4_ATARI_AFFINE_DRAFT.md', 'docs/ARC4_ATARI_AFFINE_PROTOCOL.json')
PRIOR = (
    ('craft_v6_out/v70_response_envelope_fit', 'fit_v0.json', 'fit_review_v0.json',
     '9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a',
     '3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467',
     'response_envelope_fit_independent_result_review', False),
    ('craft_v6_out/v70_stationary_affine_fit', 'fit_v0.json', 'fit_review_v0.json',
     'c60b92fea8846af3cbc8bfe9db95bdfc1585ad9c661b276462e9db67f2aa8014',
     '8be644481304ed9393f62fa93232ed5534dc46f5ee5fa2775226b710f0a646f5',
     'stationary_affine_independent_result_review', True),
    ('craft_v6_out/v70_stationary_direct_fit', 'fit_v0.json', 'fit_review_v0.json',
     'cea34fdcdc5f1de67e6edc595184a6b961770a018700874fc2a3461f5d6b37b5',
     '723596fefb4a094c38053d9cd40b019abd57c5f7f69b202b86f51a21e8bef978',
     'stationary_direct_fit_independent_result_review', False),
)
ARMS = ('newton_visual', 'newton_constant')
OPERATIONS = ('linear_prediction', 'objective', 'gradient', 'hessian', 'linear_solve', 'parameter_update')
SPEC = dict(
    version=0, recipe='newton', parameters=7, initialization='all seven coefficients zero; no prior weights',
    input='two ordered local RGB triples divided by 255, plus bias; shared across pixels and cases',
    dtype='float64', constant_input='all six RGB channels zero, bias one', newton_updates_per_arm=16,
    schedule='k=1..16: Newton visual, then constant',
    newton=dict(damping=1.75, full_step=True, objective_penalty=0.,
                loss_increase_tolerance='1e-12 * (1 + abs(loss_before))'),
    loss='equal anchor mean, equal positive/negative BCE if both exist, full BCE if one class',
    batch='all twelve existing Atari development anchors in unchanged manifest order, including two empty Seaquest anchors',
    input_loading='decode only past_pairs and observed_response_envelopes; no futures, RAM or prior weights',
    evaluation='common initial and two final arms only; full objective also before and after each update',
    controls=['own trained constant', 'common initial', 'zero probability output'],
    probability_threshold=.5, minimum_visual_iou=.25, minimum_advantage=.1,
    maximum_each_empty_false_positive_fraction=.001, required_empty_anchors=2,
    expected_operations=dict(linear_prediction=67, objective=67, gradient=32,
                             hessian=32, linear_solve=32, parameter_update=32),
    max_work_seconds=120, cleanup_reserve_seconds=10, max_binary_output_bytes=64000000,
    work_scope='data through scientific payload fsync/hash, final provenance and journal close; final receipt and console excluded',
    cpu_threads=1, gpu_used=False, new_synthetic_steps=0, new_initial_screens=0, new_emulated_frames=0, retry=False,
    selection='new candidate chosen after earlier results on this development set, not confirmatory evidence',
    claim='sufficiency of this bounded local-linear recipe on observed envelopes only; not necessity of context, mastery or acceleration')


def proofs():
    """Verify the actual target dependencies and attested prior results, without decoding arrays."""
    checked = target_proofs()
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Prospective Atari affine specification changed')
    sources, evidence = dict(checked['source_sha256']), dict(checked['evidence_sha256'])
    affine_sources = None
    for directory, result_name, review_name, result_hash, review_hash, review_kind, passed in PRIOR:
        result_path, review_path = ROOT / directory / result_name, ROOT / directory / review_name
        if sha(result_path) != result_hash or sha(review_path) != review_hash:
            raise ValueError('Previously reviewed result changed')
        result, review = (json.loads(path.read_text(encoding='utf-8')) for path in (result_path, review_path))
        if (result['execution_status'] != 'complete' or result['passed'] is not passed
                or result['final_provenance_verified'] is not True
                or review.get('kind') != review_kind or review.get('passed') is not True
                or review.get('result_sha256') != result_hash or review.get('source_sha256') != result['source_sha256']
                or any(result['source_sha256'].get(k) != v for k, v in checked['source_sha256'].items())):
            raise ValueError('Attested prior provenance or scientific outcome changed')
        if passed:
            affine_sources = result['source_sha256']
        for path in (result_path, review_path):
            committed(path)
            evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('A new candidate cannot replace an inherited source')
        committed(ROOT / rel)
        sources[rel] = sha(ROOT / rel)
    if affine_sources is None or any(sources[rel] != affine_sources.get(rel) for rel in SOURCES[:2]):
        raise ValueError('Use exactly the previously verified affine mathematics and its tests')
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=checked['runtime'])


def load_dataset(result):
    with np.load(TARGET_DATA, allow_pickle=False) as data:
        if set(data.files) != {'past_pairs', 'observed_response_envelopes'}:
            raise ValueError('Only the two declared Atari archive members are allowed')
        past, targets = data['past_pairs'], data['observed_response_envelopes']
    expected = result['output']['arrays']
    if (past.shape != (12, 2, 210, 160, 3) or past.dtype != np.uint8
            or targets.shape != (12, 210, 160) or targets.dtype != bool
            or signature(past) != expected['past_pairs'] or signature(targets) != expected['observed_response_envelopes']
            or targets.sum(axis=(1, 2)).tolist() != [144, 296, 344, 184, 127, 127, 0, 0, 109, 184, 78, 54]):
        raise ValueError('Fixed Atari input or observed-label shape, signature or count changed')
    order = [(game, index) for game in ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
             for index in range(2)]
    if [(row['game'], row['anchor_index']) for row in result['cases']] != order:
        raise ValueError('All twelve Atari anchors must retain their declared order')
    return past, targets


def operation_counts():
    return {op: {arm: 0 for arm in ('initial', *ARMS)} for op in OPERATIONS}


def expected_counts():
    counts = operation_counts()
    counts['linear_prediction']['initial'] = counts['objective']['initial'] = 1
    for arm in ARMS:
        steps = SPEC['newton_updates_per_arm']
        counts['linear_prediction'][arm] = counts['objective'][arm] = 2 * steps + 1
        for kind in ('gradient', 'hessian', 'linear_solve', 'parameter_update'):
            counts[kind][arm] = steps
    return counts


def apply_candidate(states, arm, candidate):
    states[arm]['coefficients'][:] = candidate
    states[arm]['step'] += 1


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
    verdict = payload['candidate_verdict'] if complete else dict(passed=False, complete=False, reason='Incomplete or failed execution')
    result = dict(payload, kind='atari_affine_development_fit', failures=failures,
                  execution_status='complete' if complete else 'incomplete', passed=verdict['passed'], verdict=verdict,
                  work_seconds=elapsed, scientific_payload_sha256=payload_hash,
                  claim='No independent generalization, contextual necessity, physical object identity, game mastery, transfer or acceleration established')
    write_json(result_path, result)
    return result


def run():
    configure_runtime()
    checked = proofs()
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight_hash, preflight = sha_bytes(preflight_bytes), json.loads(preflight_bytes)
    if (preflight.get('kind') != 'atari_affine_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact committed independent prospective affine preflight required')
    committed(PREFLIGHT)
    names = ['fit_v0_attempt.json', 'fit_v0_progress.jsonl', 'fit_v0_payload.json', 'fit_v0.json',
             'initial_state.npz', 'final_state.npz']
    names += [f'{arm}_{kind}.npz' for arm in ('initial', *ARMS) for kind in ('logits', 'predictions')]
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve all prior affine traces; no retry')
    control = json.loads(TARGET_RESULT.read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='atari_affine_fit_attempt',
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
        for step in range(1, SPEC['newton_updates_per_arm'] + 1):
            for arm in ARMS:
                features, state = inputs[arm], states[arm]
                before = state['coefficients'].copy()
                logits = operation('linear_prediction', arm, lambda: features @ before)
                loss_before = operation('objective', arm, lambda: affine.objective(logits, labels, weight))
                if not np.isfinite(loss_before):
                    raise FloatingPointError('Nonfinite objective before update')
                probability = affine.probabilities(logits)
                first = operation('gradient', arm, lambda: affine.gradient(features, probability, labels, weight))
                second = operation('hessian', arm, lambda: affine.hessian(features, probability, weight))
                candidate = operation('linear_solve', arm, lambda: affine.newton_candidate(before, first, second))
                operation('parameter_update', arm, lambda: apply_candidate(states, arm, candidate))
                logits_after = operation('linear_prediction', arm, lambda: features @ state['coefficients'])
                loss_after = operation('objective', arm, lambda: affine.objective(logits_after, labels, weight))
                if not np.isfinite(loss_after):
                    raise FloatingPointError('Nonfinite objective after counted update')
                event('update_complete', arm=arm, update=state['step'], loss_before=loss_before,
                      loss_after=loss_after, gradient_norm=float(np.linalg.norm(first)),
                      step_norm=float(np.linalg.norm(state['coefficients'] - before)))
                if loss_after > loss_before + 1e-12 * (1 + abs(loss_before)):
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

    trace_hashes = {names[0]: None, names[1]: None}
    for name in trace_hashes:
        try:
            trace_hashes[name] = sha(FOLDER / name)
        except BaseException as error:
            fail('trace_hash_' + name, error)
    candidate_complete = (not failures and final_provenance_verified and pending['arm'] is None
                          and entries == returns == expected_counts()
                          and set(metrics) == {'initial', 'zero', *ARMS})
    recipe_metrics = {key: metrics.get(key if key in ('initial', 'zero') else 'newton_' + key)
                      for key in ('visual', 'constant', 'initial', 'zero')}
    counts = {key: returns['parameter_update']['newton_' + key] for key in ('visual', 'constant')}
    candidate_verdict = (feasibility_verdict(recipe_metrics, counts, SPEC['newton_updates_per_arm'])
                         if candidate_complete else dict(passed=False, complete=False, reason='Incomplete execution'))
    totals = {kind: sum(counts.values()) for kind, counts in returns.items()}
    payload = dict(kind='atari_affine_scientific_payload', **checked, protocol=SPEC,
                   preflight_sha256=preflight_hash, failures=failures, artifacts=artifacts, metrics=metrics,
                   cases=control['cases'], candidate_execution_complete=candidate_complete,
                   candidate_verdict=candidate_verdict, operation_entries=entries, operation_returns=returns,
                   operation_totals=totals, operation_count_exact=entries == returns,
                   completed_updates={arm: returns['parameter_update'][arm] for arm in ARMS},
                   pending_update_arm=pending['arm'], optimizer_update_count_exact=pending['arm'] is None,
                   optimizer_updates=totals['parameter_update'],
                   optimizer_updates_upper_bound=totals['parameter_update'] + int(pending['arm'] is not None),
                   final_provenance_verified=final_provenance_verified,
                   attempt_sha256=trace_hashes[names[0]], progress_sha256=trace_hashes[names[1]],
                   new_synthetic_steps=0, new_initial_screens=0,
                   new_emulated_frames=0, gpu_used=False,
                   claim='Scientific payload only; require final receipt after charged closure')
    result = finish_result(payload, started, FOLDER / 'fit_v0_payload.json', FOLDER / 'fit_v0.json')
    print(json.dumps(dict(execution_status=result['execution_status'], verdict=result['verdict'],
          operation_totals=totals, work_seconds=result['work_seconds']), allow_nan=False))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
