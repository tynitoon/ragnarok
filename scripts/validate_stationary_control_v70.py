"""Conditional frozen-weight validation on one new draw of the stationary family."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ragnarok.environments.stationary_intervention_v70 import generate_screens, collect_interventions, new_ledger
from ragnarok.learning.response_envelope_v70 import ResponseEnvelope, measure, feasibility_verdict
from scripts.fit_stationary_control_v70 import evaluate_saved
from scripts.fit_response_envelope_v70 import sha, sha_bytes, signature, committed, configure_runtime, write_json, check_time

ROOT = Path(__file__).resolve().parents[1]
FIT_FOLDER = ROOT / 'craft_v6_out/v70_stationary_fit'
FIT_RESULT, FIT_REVIEW = FIT_FOLDER / 'fit_v0.json', FIT_FOLDER / 'fit_review_v0.json'
FIT_RESULT_SHA = None  # This conditional instrument cannot run before a reviewed successful fit.
FIT_REVIEW_SHA = None
DEVELOPMENT = ROOT / 'craft_v6_out/v70_stationary_control/control_v0_dataset.npz'
FOLDER = ROOT / 'craft_v6_out/v70_stationary_validation'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_VALIDATION_PROTOCOL.json'
SOURCES = ('scripts/validate_stationary_control_v70.py', 'tests/test_stationary_validation_v70.py',
           'docs/ARC4_STATIONARY_VALIDATION_DRAFT.md', 'docs/ARC4_STATIONARY_VALIDATION_PROTOCOL.json')
SPEC = dict(version=0, data_seed=775, rng='NumPy PCG64', height=210, width=160,
            positive_counts=[1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2], horizon=8,
            initial_screens=12, synthetic_step_returns=216, evaluation_order=['initial', 'visual', 'constant'],
            evaluation_batches=3, optimizer_updates=0, new_emulated_frames=0,
            probability_threshold=.5, minimum_visual_iou=.25, minimum_advantage=.1,
            maximum_each_empty_false_positive_fraction=.001,
            duplicate_rule='any exact ordered past RGB pair repeated from development makes the assessment inconclusive; no replacement',
            support_repeats='descriptive only, including fixed corners and two empty masks',
            controls=['initial saved weights', 'trained constant-input weights', 'zero output without a forward'],
            weights='only the reviewed fit final visual/constant and original saved random initialization',
            cpu_threads=1, gpu_used=False, max_work_seconds=180, cleanup_reserve_seconds=10,
            max_binary_output_bytes=64000000, retry=False,
            work_scope='after attempt fsync through generation, input loading, evaluation, all artifacts and payload fsync/hash, final provenance and journal closure; final receipt excluded',
            claim='performance on twelve new RGB draws of the same family only, not transfer or learning speed')


def proofs():
    if FIT_RESULT_SHA is None or FIT_REVIEW_SHA is None:
        raise ValueError('No reviewed successful fit has been frozen for this conditional instrument')
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                   threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                   deterministic=torch.are_deterministic_algorithms_enabled(), default_dtype=str(torch.get_default_dtype()),
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4' or runtime['torch'] != '2.11.0+cu126'
            or runtime['threads'] != 1 or runtime['interop_threads'] != 1 or runtime['default_dtype'] != 'torch.float32'
            or runtime['deterministic'] is not True
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] == 'ale_py' for name in sys.modules)
            or 'ragnarok.evaluation.response_envelope_certificate_v70' in sys.modules):
        raise ValueError('Exact deterministic CPU runtime without emulator or reference required')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Prospective validation specification changed')
    if sha(FIT_RESULT) != FIT_RESULT_SHA or sha(FIT_REVIEW) != FIT_REVIEW_SHA:
        raise ValueError('Reviewed fit provenance changed')
    fit, review = (json.loads(path.read_text(encoding='utf-8')) for path in (FIT_RESULT, FIT_REVIEW))
    if (fit['execution_status'] != 'complete' or fit['passed'] is not True or fit['final_provenance_verified'] is not True
            or fit['completed_updates'] != dict(visual=256, constant=256)
            or fit['optimizer_update_count_exact'] is not True or fit['model_forward_count_exact'] is not True
            or review.get('kind') != 'stationary_fit_independent_result_review' or review.get('passed') is not True
            or review.get('result_sha256') != FIT_RESULT_SHA or review.get('source_sha256') != fit['source_sha256']):
        raise ValueError('Complete successful independently reviewed fit required; otherwise no validation')
    sources = dict(fit['source_sha256'])
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('Do not replace a prior experiment source')
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Required source changed: {rel}')
        committed(ROOT / rel)
    expected_evidence = {FIT_RESULT: FIT_RESULT_SHA, FIT_REVIEW: FIT_REVIEW_SHA,
                         DEVELOPMENT: fit['evidence_sha256'][DEVELOPMENT.relative_to(ROOT).as_posix()],
                         FIT_FOLDER / 'initial_weights.pt': fit['artifacts']['initial_weights.pt']['sha256'],
                         FIT_FOLDER / 'final_weights.pt': fit['artifacts']['final_weights.pt']['sha256']}
    for path, expected in expected_evidence.items():
        if sha(path) != expected:
            raise ValueError('Validation weights or development evidence changed')
        committed(path)
    return dict(source_sha256=sources, evidence_sha256={path.relative_to(ROOT).as_posix(): value for path, value in expected_evidence.items()},
                runtime=runtime)


def compare_draws(past, targets, development_past, development_targets):
    """Full RGB duplicates reject; mask overlap is explicitly only descriptive."""
    if past.shape != development_past.shape or targets.shape != development_targets.shape or len(past) != len(targets):
        raise ValueError('Matched complete family dimensions required for comparison')
    rows = []
    for index in range(len(past)):
        same_rgb = [j for j in range(len(development_past)) if np.array_equal(past[index], development_past[j])]
        same_mask = [j for j in range(len(development_targets)) if np.array_equal(targets[index], development_targets[j])]
        intersection = int((targets[index] & development_targets[index]).sum())
        union = int((targets[index] | development_targets[index]).sum())
        rows.append(dict(case_index=index, exact_development_past_matches=same_rgb,
                         exact_development_mask_matches=same_mask, corresponding_support_intersection=intersection,
                         corresponding_support_union=union, corresponding_support_iou=intersection / union if union else None))
    return dict(admissible=not any(row['exact_development_past_matches'] for row in rows), cases=rows)


def validation_verdict(metrics):
    # Reuse the exact three scientific clauses; no optimizer budget is credited here.
    verdict = feasibility_verdict(metrics, dict(visual=0, constant=0), expected_updates=0)
    verdict['claim'] = SPEC['claim']
    return verdict


def finish_result(payload, started, payload_path, result_path):
    errors = list(payload['errors'])
    admissible = bool(payload['comparison'] and payload['comparison']['admissible'])
    candidate_assessed = not errors and payload['candidate_execution_complete'] and admissible
    candidate_verdict = (validation_verdict(payload['metrics']) if candidate_assessed
                         else dict(passed=False, complete=False, reason='No complete assessment'))
    payload = dict(payload, candidate_verdict=candidate_verdict)
    payload_hash = None
    try:
        write_json(payload_path, payload)
        payload_hash = sha(payload_path)
    except BaseException as error:
        errors.append(dict(stage='scientific_payload', type=type(error).__name__, message=str(error)))
    elapsed = time.monotonic() - started
    if elapsed >= SPEC['max_work_seconds']:
        errors.append(dict(stage='complete_scientific_payload_time', type='TimeoutError', message='Fixed validation budget exhausted'))
    complete = not errors and payload['candidate_execution_complete']
    assessed = complete and admissible
    verdict = (candidate_verdict if assessed
               else dict(passed=False, complete=False, reason='Incomplete execution' if not complete else 'Repeated development RGB input'))
    result = dict(payload, kind='stationary_fixed_weight_validation', errors=errors,
                  execution_status='complete' if complete else 'incomplete',
                  assessment_status='assessed' if assessed else 'inconclusive', passed=verdict['passed'], verdict=verdict,
                  work_seconds=elapsed, scientific_payload_sha256=payload_hash, claim=SPEC['claim'])
    write_json(result_path, result)
    return result


def run():
    configure_runtime()
    checked = proofs()
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight_hash = sha_bytes(preflight_bytes)
    preflight = json.loads(preflight_bytes)
    if (preflight.get('kind') != 'stationary_validation_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact independently reviewed committed preflight required')
    committed(PREFLIGHT)
    names = ('validation_v0_attempt.json', 'validation_v0_progress.jsonl', 'validation_v0_dataset.npz',
             'initial_logits.npz', 'initial_predictions.npz', 'visual_logits.npz', 'visual_predictions.npz',
             'constant_logits.npz', 'constant_predictions.npz', 'validation_v0_payload.json', 'validation_v0.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve every validation trace; no retry')
    fit = json.loads(FIT_RESULT.read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='stationary_validation_attempt', created_utc=datetime.now(timezone.utc).isoformat(),
                                      protocol=SPEC, preflight_sha256=preflight_hash, **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    counts, errors, artifacts, metrics = new_ledger(), [], {}, {}
    entered, returned = dict(initial=0, visual=0, constant=0), dict(initial=0, visual=0, constant=0)
    journal, comparison, final_provenance_verified = None, None, False
    cases = [dict(game=f'synthetic_pair_{i // 2}', anchor_index=i % 2) for i in range(12)]

    def log(event, **fields):
        journal.write(json.dumps(dict(event=event, elapsed_seconds=time.monotonic() - started, **fields), allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())

    def save(name, value):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **value)
        data = buffer.getvalue()
        if len(data) + sum(row['bytes'] for row in artifacts.values()) > SPEC['max_binary_output_bytes']:
            raise ValueError('Fixed binary output ceiling exceeded')
        path = FOLDER / name
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(path) != sha_bytes(data) or path.stat().st_size != len(data):
            raise ValueError('Saved binary differs from the serialized payload')
        artifacts[name] = dict(sha256=sha_bytes(data), bytes=len(data), arrays={key: signature(array) for key, array in value.items()})

    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        log('started')
        check_time(deadline)
        screens = generate_screens(seed=SPEC['data_seed'], height=SPEC['height'], width=SPEC['width'],
                                   positive_counts=SPEC['positive_counts'], ledger=counts, checkpoint=lambda: check_time(deadline))
        log('initial_screens_generated', counts=counts)
        arrays, _ = collect_interventions(screens, horizon=SPEC['horizon'], ledger=counts,
                                           checkpoint=lambda: check_time(deadline),
                                           on_step_return=lambda tally: log('synthetic_step_return', counts=tally))
        save(names[2], arrays)
        targets, past = arrays['observed_response_envelopes'], arrays['past_pairs']
        if (targets.sum(axis=(1, 2)).tolist() != SPEC['positive_counts'] or counts['initial_screens_generated'] != 12
                or counts['synthetic_step_entries'] != 216 or counts['synthetic_steps'] != 216 or counts['synthetic_rgb_frames'] != 216):
            raise ValueError('Fixed family or construction counts changed')
        with np.load(DEVELOPMENT, allow_pickle=False) as old:
            development_past, development_targets = old['past_pairs'], old['observed_response_envelopes']
        comparison = compare_draws(past, targets, development_past, development_targets)
        log('inputs_compared', comparison=comparison)
        check_time(deadline)
        if comparison['admissible']:
            inputs = torch.from_numpy(past.transpose(0, 1, 4, 2, 3).copy().reshape(12, 6, 210, 160)).float() / 255
            labels = torch.from_numpy(targets.copy()).float().unsqueeze(1)
            initial_state = torch.load(FIT_FOLDER / 'initial_weights.pt', map_location='cpu', weights_only=True)
            final_states = torch.load(FIT_FOLDER / 'final_weights.pt', map_location='cpu', weights_only=True)
            if set(final_states) != {'visual', 'constant'}:
                raise ValueError('Exactly the two reviewed trained arms are required')
            states = dict(initial=initial_state, **final_states)
            for arm in SPEC['evaluation_order']:
                check_time(deadline)
                model = ResponseEnvelope().cpu()
                model.load_state_dict(states[arm], strict=True)
                model.requires_grad_(False)
                model.eval()
                if (sum(p.numel() for p in model.parameters()) != 10177
                        or any(p.dtype != torch.float32 or p.device.type != 'cpu' or not torch.isfinite(p).all() for p in model.parameters())):
                    raise ValueError('Finite unchanged CPU float32 model required')
                def forward_enter(module, args): entered[arm] += 1
                def forward_return(module, args, output): returned[arm] += 1
                model.register_forward_pre_hook(forward_enter)
                model.register_forward_hook(forward_return)
                log('evaluation_started', arm=arm)
                model_input = torch.zeros_like(inputs) if arm == 'constant' else inputs
                metrics[arm], probability = evaluate_saved(model, model_input, labels, targets, cases, save, arm)
                if arm == 'initial':
                    metrics['zero'] = measure(np.zeros_like(probability), targets, cases)
                log('evaluation_complete', arm=arm, forward_entries=entered, forward_returns=returned)
                check_time(deadline)
        if proofs() != checked or sha(PREFLIGHT) != preflight_hash:
            raise ValueError('Final provenance differs from the attempt')
        committed(PREFLIGHT)
        for name, artifact in artifacts.items():
            if sha(FOLDER / name) != artifact['sha256']:
                raise ValueError('Scientific artifact changed during validation')
        check_time(deadline)
        final_provenance_verified = True
    except BaseException as error:
        errors.append(dict(stage='validation', type=type(error).__name__, message=str(error)))
    finally:
        if journal is not None:
            try:
                log('finished', errors=errors, counts=counts, forward_entries=entered, forward_returns=returned)
            except BaseException as error:
                errors.append(dict(stage='journal_finished', type=type(error).__name__, message=str(error)))
            try:
                journal.close()
            except BaseException as error:
                errors.append(dict(stage='journal_close', type=type(error).__name__, message=str(error)))
    expected = (dict(initial=1, visual=1, constant=1) if comparison and comparison['admissible']
                else dict(initial=0, visual=0, constant=0))
    complete = (not errors and final_provenance_verified and comparison is not None and entered == returned == expected
                and (not comparison['admissible'] or set(metrics) == {'initial', 'visual', 'constant', 'zero'}))
    payload = dict(kind='stationary_validation_scientific_payload', **checked, protocol=SPEC, preflight_sha256=preflight_hash,
                   errors=errors, counts=counts, comparison=comparison, metrics=metrics, cases=cases, artifacts=artifacts,
                   candidate_execution_complete=complete, final_provenance_verified=final_provenance_verified,
                   model_forward_entries=entered, model_forward_returns=returned, model_forward_count_exact=entered == returned,
                   synthetic_step_count_exact=counts['synthetic_step_entries'] == counts['synthetic_steps'],
                   inherited_synthetic_steps=fit['inherited_synthetic_steps'], new_synthetic_steps=counts['synthetic_steps'],
                   optimizer_updates=0, new_emulated_frames=0, gpu_used=False,
                   attempt_sha256=sha(FOLDER / names[0]), progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None)
    result = finish_result(payload, started, FOLDER / names[-2], FOLDER / names[-1])
    print(json.dumps(dict(execution_status=result['execution_status'], assessment_status=result['assessment_status'],
                          verdict=result['verdict'], work_seconds=result['work_seconds'], counts=counts), allow_nan=False))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
