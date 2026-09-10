"""One prospective image-versus-constant auxiliary fit; no emulator or GPU."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ragnarok.learning.response_envelope_v70 import (
    ResponseEnvelope, anchor_balanced_bce, measure, feasibility_verdict)

ROOT = Path(__file__).resolve().parents[1]
TARGET_FOLDER = ROOT / 'craft_v6_out/v70_response_envelopes'
TARGET_RESULT = TARGET_FOLDER / 'targets_v0.json'
TARGET_REVIEW = TARGET_FOLDER / 'targets_review_v0.json'
TARGET_DATA = TARGET_FOLDER / 'targets_v0.npz'
FOLDER = ROOT / 'craft_v6_out/v70_response_envelope_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_RESPONSE_ENVELOPE_FIT_PROTOCOL.json'
TARGET_RESULT_SHA = 'd3c662ac1c269ff723add3e4417689a9dc57e7aa1745e9a700a7940bf78b6659'
TARGET_REVIEW_SHA = 'd1211d166c2af7a402587c3225f981d273392be9b08277c1da10419d12d825a2'
SOURCES = ('ragnarok/learning/response_envelope_v70.py', 'scripts/fit_response_envelope_v70.py',
           'tests/test_response_envelope_fit_v70.py', 'docs/ARC4_RESPONSE_ENVELOPE_FIT_DRAFT.md',
           'docs/ARC4_RESPONSE_ENVELOPE_FIT_PROTOCOL.json')
SPEC = dict(version=0, seed=764, updates_per_arm=256, arm_order=['visual', 'constant'],
            input='two ordered RGB images, six channels, divided by 255',
            architecture=dict(channels=16, dilations=[1, 2, 4, 8, 16], kernel=3,
                              activation='SiLU', padding='replicate', output_kernel=1),
            optimizer=dict(name='Adam', lr=0.0003, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0,
                           amsgrad=False, foreach=False, fused=False),
            gradient_clip_norm=1.0, batch='all twelve development anchors, unchanged manifest order',
            loss='equal anchor mean, equal positive/negative BCE if both exist, full BCE if one class',
            controls=['trained constant input', 'initial weights on real input', 'zero output'],
            evaluation='initial and final only; per anchor, then equal nonempty game means',
            probability_threshold=0.5, minimum_visual_iou=0.25, minimum_advantage=0.10,
            maximum_each_empty_false_positive_fraction=0.001, required_empty_anchors=2,
            average_precision='non-interpolated precision-recall area with tied scores grouped',
            max_work_seconds=1200, cleanup_reserve_seconds=10, max_binary_output_bytes=64000000,
            work_scope='through scientific payload fsync/hash and journal close; final verdict receipt and console excluded',
            cpu_threads=1, gpu_used=False, new_emulated_frames=0,
            claim='development fitting feasibility only; no independent validation or transfer')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def signature(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=sha_bytes(array.tobytes(order='C')))


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={rel}', str(path)) != git('rev-parse', f'HEAD:{rel}'):
        raise ValueError(f'Uncommitted required file: {rel}')


def configure_runtime():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def proofs():
    """Hash the small derived dataset; do not decode it or construct a network."""
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   torch=torch.__version__, threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4' or runtime['torch'] != '2.11.0+cu126'
            or runtime['threads'] != 1 or runtime['interop_threads'] != 1 or runtime['deterministic'] is not True
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] == 'ale_py' for name in sys.modules)):
        raise ValueError('Exact deterministic CPU-only runtime required')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Prospective fit specification changed')
    if sha(TARGET_RESULT) != TARGET_RESULT_SHA or sha(TARGET_REVIEW) != TARGET_REVIEW_SHA:
        raise ValueError('Fixed target derivation or independent review changed')
    result = json.loads(TARGET_RESULT.read_text(encoding='utf-8'))
    review = json.loads(TARGET_REVIEW.read_text(encoding='utf-8'))
    if (result['passed'] is not True or result['execution_status'] != 'complete' or result['archives_read'] != 216
            or len(result['cases']) != 12 or review.get('kind') != 'response_envelope_independent_result_review'
            or review.get('passed') is not True or review.get('result_sha256') != TARGET_RESULT_SHA
            or review.get('source_sha256') != result['source_sha256']
            or result['output']['path'] != TARGET_DATA.relative_to(ROOT).as_posix()
            or sha(TARGET_DATA) != result['output']['sha256']):
        raise ValueError('Exact reviewed complete twelve-anchor derivation required')
    sources = dict(result['source_sha256'])
    for rel in SOURCES:
        committed(ROOT / rel)
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Required source changed: {rel}')
    evidence = {}
    for path in (TARGET_RESULT, TARGET_REVIEW, TARGET_DATA):
        committed(path)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=runtime)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_dataset(result):
    with np.load(TARGET_DATA, allow_pickle=False) as data:
        if set(data.files) != {'past_pairs', 'observed_response_envelopes'}:
            raise ValueError('Only past images and the declared auxiliary labels are admitted')
        past = data['past_pairs']
        targets = data['observed_response_envelopes']
    if (past.shape != (12, 2, 210, 160, 3) or past.dtype != np.uint8
            or targets.shape != (12, 210, 160) or targets.dtype != bool
            or signature(past) != result['output']['arrays']['past_pairs']
            or signature(targets) != result['output']['arrays']['observed_response_envelopes']):
        raise ValueError('Derived array shape, dtype or signature changed')
    order = [(game, index) for game in ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
             for index in range(2)]
    if [(row['game'], row['anchor_index']) for row in result['cases']] != order:
        raise ValueError('No reordered or excluded anchor is allowed')
    inputs = torch.from_numpy(past.transpose(0, 1, 4, 2, 3).copy().reshape(12, 6, 210, 160)).float() / 255
    labels = torch.from_numpy(targets.copy()).float().unsqueeze(1)
    return inputs, labels, targets


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Fixed work ceiling reached; no extension or retry')


def update(model, optimizer, inputs, labels, before_step=None, after_step=None):
    optimizer.zero_grad(set_to_none=True)
    loss = anchor_balanced_bce(model(inputs), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    values = float(loss.detach()), float(norm)
    if before_step is not None:
        before_step()
    optimizer.step()
    if after_step is not None:
        after_step()
    return values


@torch.no_grad()
def predict(model, inputs):
    result = torch.sigmoid(model(inputs)).squeeze(1).cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite prediction')
    return result


def run():
    checked = proofs()
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (preflight.get('kind') != 'response_envelope_fit_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact independent prospective fit review required')
    committed(PREFLIGHT)
    names = ('fit_v0_attempt.json', 'fit_v0_progress.jsonl', 'fit_v0.json', 'initial_weights.pt',
             'final_weights.pt', 'initial_predictions.npz', 'final_predictions.npz', 'fit_v0_payload.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve every previous fit trace and checkpoint')
    target_result = json.loads(TARGET_RESULT.read_text(encoding='utf-8'))
    attempt = dict(kind='response_envelope_fit_attempt', created_utc=datetime.now(timezone.utc).isoformat(),
                   protocol=SPEC, **checked, preflight_sha256=sha(PREFLIGHT))
    write_json(FOLDER / names[0], attempt)
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    models, artifacts, metrics = {}, {}, {}
    counts, entries = dict(visual=0, constant=0), dict(visual=0, constant=0)
    pending = {'arm': None}
    inference_batches = 0
    failures = []
    journal = None

    def fail(stage, error):
        failures.append(dict(stage=stage, type=type(error).__name__, message=str(error)))

    def save_binary(name, value, torch_format=False):
        buffer = io.BytesIO()
        if torch_format:
            torch.save(value, buffer)
        else:
            np.savez(buffer, **value)
        data = buffer.getvalue()
        if len(data) + sum(row['bytes'] for row in artifacts.values()) > SPEC['max_binary_output_bytes']:
            raise ValueError('Binary artifact storage ceiling reached')
        with (FOLDER / name).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts[name] = dict(bytes=len(data), sha256=sha_bytes(data))

    def event(kind, **values):
        if journal is None:
            raise OSError('The durable journal was not opened')
        journal.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started, **values),
                                 allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())

    def safe_event(kind, **values):
        try:
            event(kind, **values)
        except BaseException as error:
            fail('journal_' + kind, error)

    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        inputs, labels, targets = load_dataset(target_result)
        check_time(deadline)
        torch.manual_seed(SPEC['seed'])
        visual = ResponseEnvelope().cpu()
        models = dict(visual=visual)
        if sum(value.numel() for value in visual.parameters()) != 10177:
            raise ValueError('Fixed architecture parameter population changed')
        models['constant'] = copy.deepcopy(visual)
        if any(not torch.equal(left, right) for left, right in
               zip(models['visual'].parameters(), models['constant'].parameters(), strict=True)):
            raise ValueError('The two arms must start at the same weights')
        save_binary('initial_weights.pt', visual.state_dict(), torch_format=True)
        initial = predict(visual, inputs)
        inference_batches += 1
        metrics['initial'] = measure(initial, targets, target_result['cases'])
        metrics['zero'] = measure(np.zeros_like(initial), targets, target_result['cases'])
        save_binary('initial_predictions.npz', dict(initial=initial))
        check_time(deadline)
        optimizers = {key: torch.optim.Adam(model.parameters(), lr=0.0003, betas=(.9, .999), eps=1e-8,
                      weight_decay=0.0, amsgrad=False, foreach=False, fused=False) for key, model in models.items()}
        arm_inputs = dict(visual=inputs, constant=torch.zeros_like(inputs))
        for step in range(SPEC['updates_per_arm']):
            for arm in SPEC['arm_order']:
                check_time(deadline)
                event('update_started', arm=arm, update=step + 1)
                def before_step():
                    # This durable entry must succeed before a mutation can be attempted.
                    event('optimizer_step_enter', arm=arm, update=step + 1)
                    entries[arm] += 1
                    pending['arm'] = arm
                def after_step():
                    # Preserve the certain return even if its journal write fails next.
                    counts[arm] += 1
                    pending['arm'] = None
                    event('optimizer_step_return', arm=arm, update=counts[arm])
                loss, norm = update(models[arm], optimizers[arm], arm_inputs[arm], labels, before_step, after_step)
                event('update_complete', arm=arm, update=counts[arm], loss=loss, gradient_norm_before_clip=norm)
                if any(not torch.isfinite(value).all() for value in models[arm].parameters()):
                    raise FloatingPointError('Nonfinite parameter after the counted optimizer step')
                check_time(deadline)
        predictions = {}
        for arm in SPEC['arm_order']:
            predictions[arm] = predict(models[arm], arm_inputs[arm])
            inference_batches += 1
            metrics[arm] = measure(predictions[arm], targets, target_result['cases'])
            check_time(deadline)
        save_binary('final_predictions.npz', predictions)
        check_time(deadline)
    except BaseException as error:
        fail('scientific_work', error)
        safe_event('failed', failure=failures[0], completed_updates=counts, pending_optimizer_arm=pending['arm'])
    finally:
        # Independent of journal health, including catchable interruptions.
        try:
            if models:
                save_binary('final_weights.pt', {key: model.state_dict() for key, model in models.items()}, torch_format=True)
        except BaseException as error:
            fail('preserve_weights', error)
        try:
            for rel, expected in {**checked['source_sha256'], **checked['evidence_sha256'],
                                  PREFLIGHT.relative_to(ROOT).as_posix(): attempt['preflight_sha256']}.items():
                if sha(ROOT / rel) != expected:
                    raise ValueError(f'Final provenance changed: {rel}')
            check_time(started + SPEC['max_work_seconds'])
        except BaseException as error:
            fail('final_provenance_or_time', error)
        safe_event('finished', completed_updates=counts, step_entries=entries, pending_optimizer_arm=pending['arm'], failures=failures.copy())
        if journal is not None:
            try:
                journal.flush()
                os.fsync(journal.fileno())
            except BaseException as error:
                fail('journal_flush', error)
            finally:
                try:
                    journal.close()
                except BaseException as error:
                    fail('journal_close', error)

    complete_candidate = not failures and pending['arm'] is None and set(metrics) == {'initial', 'zero', 'visual', 'constant'}
    candidate_verdict = (feasibility_verdict(metrics, counts, SPEC['updates_per_arm']) if complete_candidate
                         else dict(passed=False, complete=False, reason='Incomplete or failed execution'))
    progress_hash = sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None
    payload = dict(kind='response_envelope_fit_scientific_payload', candidate_verdict=candidate_verdict,
                   failures=failures.copy(), protocol=SPEC, **checked, preflight_sha256=attempt['preflight_sha256'],
                   completed_updates=counts, optimizer_step_entries=entries, pending_optimizer_arm=pending['arm'],
                   optimizer_update_count_exact=pending['arm'] is None, optimizer_updates=sum(counts.values()),
                   optimizer_updates_upper_bound=sum(counts.values()) + int(pending['arm'] is not None),
                   model_inference_batches=inference_batches, metrics=metrics, artifacts=artifacts,
                   attempt_sha256=sha(FOLDER / names[0]), progress_sha256=progress_hash,
                   inherited_collection_charged_frames=target_result['inherited_collection_charged_frames'],
                   new_emulated_frames=0, gpu_used=False, targets_role='twelve development anchors, no validation split',
                   claim='This payload is not a completed verdict; require the subsequent duration-checked receipt')
    write_json(FOLDER / 'fit_v0_payload.json', payload)
    payload_hash = sha(FOLDER / 'fit_v0_payload.json')
    work_seconds = time.monotonic() - started
    if work_seconds >= SPEC['max_work_seconds']:
        fail('complete_scientific_payload_time', TimeoutError('Scientific payload closure exceeded the fixed budget'))
    complete = complete_candidate and not failures
    verdict = candidate_verdict if complete else dict(passed=False, complete=False, reason='Incomplete or failed execution')
    # This small attestation and console are prospectively outside the measured scientific work.
    result = dict(payload, kind='response_envelope_development_fit', execution_status='complete' if complete else 'incomplete',
                  passed=verdict['passed'], verdict=verdict, failure=dict(errors=failures) if failures else None,
                  failures=failures.copy(),
                  work_seconds=work_seconds, scientific_payload_sha256=payload_hash,
                  claim='No independent generalization, object identity, game mastery, transfer or acceleration established')
    write_json(FOLDER / names[2], result)
    print(json.dumps(dict(execution_status=result['execution_status'], verdict=verdict, completed_updates=counts,
                         optimizer_update_count_exact=result['optimizer_update_count_exact'], work_seconds=work_seconds)))
    return 0 if complete else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    configure_runtime()
    if args.run:
        raise SystemExit(run())
    checked = proofs()
    print(json.dumps(dict(check=True, sources=len(checked['source_sha256']), models_constructed=0)))
