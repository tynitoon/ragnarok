"""A separate, fixed-budget random-initialization fit on the certified control."""
import argparse
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ragnarok.learning.response_envelope_v70 import (
    ResponseEnvelope, anchor_balanced_bce, measure, feasibility_verdict)
from scripts.fit_response_envelope_v70 import (
    SPEC as ATARI_SPEC, sha, sha_bytes, signature, committed, configure_runtime,
    write_json, check_time, update)

ROOT = Path(__file__).resolve().parents[1]
CONTROL_FOLDER = ROOT / 'craft_v6_out/v70_stationary_control'
CONTROL_RESULT = CONTROL_FOLDER / 'control_v0.json'
CONTROL_REVIEW = CONTROL_FOLDER / 'control_review_v0.json'
CONTROL_DATA = CONTROL_FOLDER / 'control_v0_dataset.npz'
CONTROL_RESULT_SHA = '88a7fcd8124a40ef0d13bd9112713e3a456ba1a78737eaa14e1786bcd69d0202'
CONTROL_REVIEW_SHA = 'dc121b32ed97628e3775ae28f37eb734680c46d8038b725747b58bf7a83cb69b'
FOLDER = ROOT / 'craft_v6_out/v70_stationary_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_FIT_PROTOCOL.json'
SOURCES = ('scripts/fit_stationary_control_v70.py', 'tests/test_stationary_fit_v70.py',
           'docs/ARC4_STATIONARY_FIT_DRAFT.md', 'docs/ARC4_STATIONARY_FIT_PROTOCOL.json')
SPEC = copy.deepcopy(ATARI_SPEC)
SPEC.update(batch='all twelve certified synthetic cases, unchanged manifest order',
            evaluation='initial and final only; per nonempty case, then equal nonempty pair means',
            targets_role='development calibration, not six games or independent validation',
            inherited_synthetic_steps=216, new_synthetic_steps=0, new_initial_screens=0,
            probability_threshold=.5, expected_training_forward_batches=512,
            expected_evaluation_forward_batches=3, retry=False,
            input_loading='decode past_pairs and observed_response_envelopes only; no futures or reference',
            claim='fixed-budget fitting feasibility of the certified easy family only')


def proofs():
    """Hash provenance, without loading arrays, judge code or model parameters."""
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   torch=torch.__version__, threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(),
                   deterministic=torch.are_deterministic_algorithms_enabled(),
                   default_dtype=str(torch.get_default_dtype()),
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4' or runtime['torch'] != '2.11.0+cu126'
            or runtime['threads'] != 1 or runtime['interop_threads'] != 1 or runtime['deterministic'] is not True
            or runtime['default_dtype'] != 'torch.float32'
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] == 'ale_py' for name in sys.modules)
            or 'ragnarok.evaluation.response_envelope_certificate_v70' in sys.modules):
        raise ValueError('Exact deterministic CPU runtime without emulator or certificate module required')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Fixed stationary fitting specification changed')
    if (CONTROL_RESULT_SHA is None or CONTROL_REVIEW_SHA is None
            or sha(CONTROL_RESULT) != CONTROL_RESULT_SHA or sha(CONTROL_REVIEW) != CONTROL_REVIEW_SHA):
        raise ValueError('Exact complete reviewed control is required before fitting')
    result = json.loads(CONTROL_RESULT.read_text(encoding='utf-8'))
    review = json.loads(CONTROL_REVIEW.read_text(encoding='utf-8'))
    if (result['execution_status'] != 'complete' or result['passed'] is not True
            or result['final_provenance_verified'] is not True
            or result['certificate']['passed'] is not True
            or result['optimizer_updates'] != 0 or result['new_emulated_frames'] != 0
            or result['counts']['synthetic_steps'] != 216
            or review.get('kind') != 'stationary_control_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != CONTROL_RESULT_SHA
            or review.get('source_sha256') != result['source_sha256']
            or sha(CONTROL_DATA) != result['artifacts'][CONTROL_DATA.name]['sha256']):
        raise ValueError('Certified and independently reviewed unchanged synthetic construction required')
    sources = dict(result['source_sha256'])
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('A new experiment cannot replace a prior source')
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Required source changed: {rel}')
        committed(ROOT / rel)
    evidence = {}
    for path in (CONTROL_RESULT, CONTROL_REVIEW, CONTROL_DATA):
        committed(path)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=runtime)


def load_dataset(result):
    with np.load(CONTROL_DATA, allow_pickle=False) as data:
        if set(data.files) != {'past_pairs', 'observed_response_envelopes', 'noop_futures', 'action_futures'}:
            raise ValueError('The fixed four-member synthetic archive is required')
        # NPZ member access is lazy: the future branches are never decoded here.
        past = data['past_pairs']
        targets = data['observed_response_envelopes']
    expected = result['artifacts'][CONTROL_DATA.name]['arrays']
    if (past.shape != (12, 2, 210, 160, 3) or past.dtype != np.uint8
            or targets.shape != (12, 210, 160) or targets.dtype != bool
            or signature(past) != expected['past_pairs']
            or signature(targets) != expected['observed_response_envelopes']
            or targets.sum(axis=(1, 2)).tolist() != [1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2]):
        raise ValueError('Certified input or label shape, dtype, signature or count changed')
    order = [(f'synthetic_pair_{index // 2}', index % 2) for index in range(12)]
    if [(row['game'], row['anchor_index']) for row in result['cases']] != order:
        raise ValueError('All twelve synthetic cases must retain the declared order')
    inputs = torch.from_numpy(past.transpose(0, 1, 4, 2, 3).copy().reshape(12, 6, 210, 160)).float() / 255
    labels = torch.from_numpy(targets.copy()).float().unsqueeze(1)
    return inputs, labels, targets


def attach_forward_counters(model, arm, phase, entries, returns):
    """Observe returns without changing the network, its parameters or its inputs."""
    def enter(module, inputs):
        entries[phase['name']][arm] += 1
    def returned(module, inputs, output):
        returns[phase['name']][arm] += 1
    model.register_forward_pre_hook(enter)
    model.register_forward_hook(returned)


def evaluate_saved(model, inputs, labels, targets, cases, save, name):
    """Save the actual returned logits before any numerical rejection or metrics."""
    with torch.inference_mode():
        logits = model(inputs)
    raw = logits[:, 0].numpy().copy()
    save(name + '_logits.npz', dict(logits=raw))
    if not np.isfinite(raw).all():
        raise FloatingPointError('Nonfinite evaluation logits')
    probability = logits.sigmoid()[:, 0].numpy().copy()
    save(name + '_predictions.npz', dict(probability=probability))
    measured = measure(probability, targets, cases)
    bce = float(anchor_balanced_bce(logits, labels))
    if not np.isfinite(bce):
        raise FloatingPointError('Nonfinite evaluation BCE')
    measured['balanced_bce'] = bce
    return measured, probability


def finish_result(payload, started, payload_path, result_path):
    failures = list(payload['failures'])
    payload_hash = None
    try:
        write_json(payload_path, payload)
        payload_hash = sha(payload_path)
    except BaseException as error:
        failures.append(dict(stage='scientific_payload', type=type(error).__name__, message=str(error)))
    elapsed = time.monotonic() - started
    if elapsed >= SPEC['max_work_seconds']:
        failures.append(dict(stage='complete_scientific_payload_time', type='TimeoutError',
                             message='Scientific payload closure exceeded the fixed budget'))
    complete = (not failures and payload['candidate_execution_complete']
                and elapsed < SPEC['max_work_seconds'])
    verdict = (payload['candidate_verdict'] if complete
               else dict(passed=False, complete=False, reason='Incomplete or failed execution'))
    result = dict(payload, kind='stationary_control_development_fit', failures=failures,
                  execution_status='complete' if complete else 'incomplete', passed=verdict['passed'],
                  verdict=verdict, work_seconds=elapsed, scientific_payload_sha256=payload_hash,
                  claim='No independent generalization, physical object identity, game mastery, transfer or acceleration established')
    write_json(result_path, result)
    return result


def run():
    configure_runtime()
    checked = proofs()
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight_hash = sha_bytes(preflight_bytes)
    preflight = json.loads(preflight_bytes)
    if (preflight.get('kind') != 'stationary_fit_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact committed independent prospective fit preflight required')
    committed(PREFLIGHT)
    names = ('fit_v0_attempt.json', 'fit_v0_progress.jsonl', 'fit_v0.json', 'initial_weights.pt',
             'final_weights.pt', 'initial_logits.npz', 'initial_predictions.npz', 'visual_logits.npz',
             'visual_predictions.npz', 'constant_logits.npz', 'constant_predictions.npz', 'fit_v0_payload.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve all prior fit traces; no retry')
    control_result = json.loads(CONTROL_RESULT.read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='stationary_fit_attempt', created_utc=datetime.now(timezone.utc).isoformat(),
                                      protocol=SPEC, preflight_sha256=preflight_hash, **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    models, artifacts, metrics, failures = {}, {}, {}, []
    counts, entries = dict(visual=0, constant=0), dict(visual=0, constant=0)
    forward_entries = {kind: dict(visual=0, constant=0) for kind in ('training', 'evaluation')}
    forward_returns = copy.deepcopy(forward_entries)
    pending, phase = {'arm': None}, {'name': 'evaluation'}
    journal, final_provenance_verified = None, False

    def fail(stage, error):
        failures.append(dict(stage=stage, type=type(error).__name__, message=str(error)))

    def save_binary(name, value, torch_format=False):
        buffer = io.BytesIO()
        if torch_format:
            torch.save(value, buffer)
        else:
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
            raise ValueError('Saved binary differs from its serialized payload')
        artifacts[name] = dict(bytes=len(data), sha256=sha_bytes(data))
        if not torch_format:
            artifacts[name]['arrays'] = {key: signature(array) for key, array in value.items()}

    def event(kind, **fields):
        if journal is None:
            raise OSError('The durable journal was not opened')
        journal.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started, **fields), allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())

    def safe_event(kind, **fields):
        try:
            event(kind, **fields)
        except BaseException as error:
            fail('journal_' + kind, error)

    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        inputs, labels, targets = load_dataset(control_result)
        check_time(deadline)
        torch.manual_seed(SPEC['seed'])
        visual = ResponseEnvelope().cpu()
        models = dict(visual=visual)
        if (sum(value.numel() for value in visual.parameters()) != 10177
                or any(value.device.type != 'cpu' or value.dtype != torch.float32
                       or not torch.isfinite(value).all() for value in visual.parameters())):
            raise ValueError('Fixed architecture parameter population changed')
        models['constant'] = copy.deepcopy(visual)
        if any(not torch.equal(a, b) for a, b in zip(visual.parameters(), models['constant'].parameters(), strict=True)):
            raise ValueError('Both arms must start at identical weights')
        for arm, model in models.items():
            attach_forward_counters(model, arm, phase, forward_entries, forward_returns)
        save_binary('initial_weights.pt', visual.state_dict(), torch_format=True)
        event('evaluation_started', arm='visual', checkpoint='initial')
        metrics['initial'], initial = evaluate_saved(visual, inputs, labels, targets, control_result['cases'], save_binary, 'initial')
        metrics['zero'] = measure(np.zeros_like(initial), targets, control_result['cases'])
        event('evaluation_complete', arm='visual', checkpoint='initial', forward_entries=forward_entries, forward_returns=forward_returns)
        check_time(deadline)
        optimizers = {key: torch.optim.Adam(model.parameters(), lr=.0003, betas=(.9, .999), eps=1e-8,
                      weight_decay=0., amsgrad=False, foreach=False, fused=False) for key, model in models.items()}
        arm_inputs = dict(visual=inputs, constant=torch.zeros_like(inputs))
        phase['name'] = 'training'
        for step in range(SPEC['updates_per_arm']):
            for arm in SPEC['arm_order']:
                check_time(deadline)
                event('update_started', arm=arm, update=step + 1)
                def before_step():
                    event('optimizer_step_enter', arm=arm, update=step + 1)
                    entries[arm] += 1
                    pending['arm'] = arm
                def after_step():
                    counts[arm] += 1
                    pending['arm'] = None
                    event('optimizer_step_return', arm=arm, update=counts[arm],
                          forward_entries=forward_entries, forward_returns=forward_returns)
                loss, norm = update(models[arm], optimizers[arm], arm_inputs[arm], labels, before_step, after_step)
                event('update_complete', arm=arm, update=counts[arm], loss=loss, gradient_norm_before_clip=norm)
                if any(not torch.isfinite(value).all() for value in models[arm].parameters()):
                    raise FloatingPointError('Nonfinite parameter after the counted optimizer step')
                check_time(deadline)
        phase['name'] = 'evaluation'
        for arm in SPEC['arm_order']:
            event('evaluation_started', arm=arm, checkpoint='final')
            metrics[arm], _ = evaluate_saved(models[arm], arm_inputs[arm], labels, targets,
                                             control_result['cases'], save_binary, arm)
            event('evaluation_complete', arm=arm, checkpoint='final', forward_entries=forward_entries, forward_returns=forward_returns)
            check_time(deadline)
    except BaseException as error:
        fail('scientific_work', error)
        safe_event('failed', completed_updates=counts, pending_optimizer_arm=pending['arm'])
    finally:
        try:
            if models:
                save_binary('final_weights.pt', {key: model.state_dict() for key, model in models.items()}, torch_format=True)
        except BaseException as error:
            fail('preserve_weights', error)
        try:
            if proofs() != checked or sha(PREFLIGHT) != preflight_hash:
                raise ValueError('Final source, evidence, runtime or preflight differs from the attempt')
            committed(PREFLIGHT)
            for name, artifact in artifacts.items():
                if sha(FOLDER / name) != artifact['sha256']:
                    raise ValueError('Scientific output changed during fitting')
            check_time(started + SPEC['max_work_seconds'])
            final_provenance_verified = True
        except BaseException as error:
            fail('final_provenance_or_time', error)
        safe_event('finished', completed_updates=counts, step_entries=entries, pending_optimizer_arm=pending['arm'],
                   forward_entries=forward_entries, forward_returns=forward_returns, failures=failures.copy())
        if journal is not None:
            try:
                journal.close()
            except BaseException as error:
                fail('journal_close', error)

    expected_forwards = dict(training=dict(visual=SPEC['updates_per_arm'], constant=SPEC['updates_per_arm']),
                             evaluation=dict(visual=2, constant=1))
    candidate_complete = (not failures and pending['arm'] is None and final_provenance_verified
                          and counts == entries == dict(visual=SPEC['updates_per_arm'], constant=SPEC['updates_per_arm'])
                          and forward_entries == forward_returns == expected_forwards
                          and set(metrics) == {'initial', 'zero', 'visual', 'constant'})
    candidate_verdict = (feasibility_verdict(metrics, counts, SPEC['updates_per_arm']) if candidate_complete
                         else dict(passed=False, complete=False, reason='Incomplete or failed execution'))
    payload = dict(kind='stationary_fit_scientific_payload', **checked, protocol=SPEC,
                   preflight_sha256=preflight_hash, failures=failures, artifacts=artifacts, metrics=metrics,
                   cases=control_result['cases'], candidate_execution_complete=candidate_complete,
                   candidate_verdict=candidate_verdict, completed_updates=counts, optimizer_step_entries=entries,
                   pending_optimizer_arm=pending['arm'], optimizer_update_count_exact=pending['arm'] is None,
                   optimizer_updates=sum(counts.values()), optimizer_updates_upper_bound=sum(counts.values()) + int(pending['arm'] is not None),
                   model_forward_entries=forward_entries, model_forward_returns=forward_returns,
                   model_forward_count_exact=forward_entries == forward_returns,
                   final_provenance_verified=final_provenance_verified,
                   attempt_sha256=sha(FOLDER / names[0]), progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None,
                   inherited_synthetic_steps=216, new_synthetic_steps=0, new_initial_screens=0,
                   new_emulated_frames=0, gpu_used=False, targets_role=SPEC['targets_role'],
                   claim='Scientific payload only; require the final receipt after charged closure')
    result = finish_result(payload, started, FOLDER / 'fit_v0_payload.json', FOLDER / 'fit_v0.json')
    print(json.dumps(dict(execution_status=result['execution_status'], verdict=result['verdict'], completed_updates=counts,
                          optimizer_update_count_exact=result['optimizer_update_count_exact'], work_seconds=result['work_seconds']), allow_nan=False))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
