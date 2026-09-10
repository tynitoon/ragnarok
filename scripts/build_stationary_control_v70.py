"""One synthetic dataset and external constructive certificate, with no fitting."""
import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from ragnarok.environments.stationary_intervention_v70 import generate_screens, collect_interventions, new_ledger
from ragnarok.evaluation.response_envelope_certificate_v70 import make_reference_model
from ragnarok.learning.response_envelope_v70 import anchor_balanced_bce, measure
from scripts.fit_response_envelope_v70 import (
    sha, sha_bytes, signature, committed, configure_runtime, write_json, check_time)

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_control'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_CONTROL_PROTOCOL.json'
PREFLIGHT = FOLDER / 'preflight_v0.json'
EQUIVALENCE = ROOT / 'craft_v6_out/v70_response_envelope_equivalence/equivalence_v0.json'
FIXED = {
    'craft_v6_out/v70_response_envelope_fit/fit_v0.json': '9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a',
    'craft_v6_out/v70_response_envelope_fit/fit_review_v0.json': '3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467',
    'craft_v6_out/v70_response_envelope_equivalence/equivalence_v0.json': '4ceb0eb9609a7ff218250ea6c5c870e0a0335c912593228f3b9c2d22f0586029',
    'craft_v6_out/v70_response_envelope_equivalence/equivalence_review_v0.json': 'b973a59059530ca5249592d6b75c5f592da2440a3ee295768a316870ecd98f0a',
}
SOURCES = ('ragnarok/environments/stationary_intervention_v70.py',
           'tests/test_stationary_intervention_v70.py',
           'ragnarok/evaluation/response_envelope_certificate_v70.py',
           'tests/test_response_envelope_certificate_v70.py',
           'scripts/build_stationary_control_v70.py', 'tests/test_build_stationary_control_v70.py',
           'docs/ARC4_STATIONARY_CONTROL_DRAFT.md', 'docs/ARC4_STATIONARY_CONTROL_PROTOCOL.json')
SPEC = dict(version=0, role='synthetic construction and external representability certificate only',
            data_seed=774, rng='NumPy PCG64', height=210, width=160,
            positive_counts=[1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2],
            actions=[0, 1], past_noop_steps=2, horizon=8, expected_synthetic_steps=216,
            expected_initial_screens=12,
            target='union of observed RGB differences between owned action1 and NOOP branches',
            reference='fixed two-channel central-coefficient SiLU certificate; never learner weights',
            required_parameters=10177, reference_forward_batches=1, probability_threshold=.5,
            expected_logit_magnitude=8.0, maximum_logit_absolute_error=1e-5,
            maximum_bce_absolute_error=1e-6, max_work_seconds=300, cleanup_reserve_seconds=10,
            max_binary_output_bytes=64000000, cpu_threads=1, gpu_used=False,
            optimizer_updates=0, agent_model_inferences=0, new_emulated_frames=0,
            work_scope='after attempt fsync through payload fsync/hash, journal close and final provenance; final receipt excluded',
            retry=False, claim='explicit reference representability only; no learned ability or transfer')


def proofs():
    """No data generation, NPZ loading, reference construction or model call."""
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   torch=torch.__version__, threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(),
                   deterministic=torch.are_deterministic_algorithms_enabled(),
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4' or runtime['torch'] != '2.11.0+cu126'
            or runtime['threads'] != 1 or runtime['interop_threads'] != 1 or runtime['deterministic'] is not True
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] == 'ale_py' for name in sys.modules)):
        raise ValueError('Exact deterministic CPU runtime required')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Fixed construction specification changed')
    for rel, expected in FIXED.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Prior reviewed evidence changed: {rel}')
        committed(ROOT / rel)
    previous = json.loads(EQUIVALENCE.read_text(encoding='utf-8'))
    sources = dict(previous['source_sha256'])
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('A new control must not replace a previous source')
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Required source changed: {rel}')
        committed(ROOT / rel)
    return dict(source_sha256=sources, evidence_sha256=FIXED.copy(), runtime=runtime)


def certificate_result(logits, targets, cases):
    if (logits.device.type != 'cpu' or logits.dtype != torch.float32
            or logits.shape != (len(targets), 1, *targets.shape[1:])
            or targets.dtype != bool or not torch.isfinite(logits).all()):
        raise ValueError('Finite float32 CPU logits and exact target dimensions required')
    probability = logits.sigmoid()[:, 0].numpy()
    measured = measure(probability, targets, cases)
    expected = np.where(targets, 8., -8.).astype(np.float32)
    maximum_error = float(np.abs(logits[:, 0].numpy() - expected).max())
    bce = float(anchor_balanced_bce(logits, torch.from_numpy(targets.copy()).float().unsqueeze(1)))
    exact_bce = math.log1p(math.exp(-8))
    classification = all(row['false_positive_pixels'] == 0
                         and (row['iou'] == 1 and row['average_precision'] == 1 if row['target_pixels'] else row['iou'] is None)
                         for row in measured['cases'])
    passed = (classification and maximum_error <= SPEC['maximum_logit_absolute_error']
              and abs(bce - exact_bce) <= SPEC['maximum_bce_absolute_error'])
    return dict(passed=passed, metrics=measured, maximum_logit_absolute_error=maximum_error,
                balanced_bce=bce, analytic_balanced_bce=exact_bce,
                claim='finite explicit judge solution, not a learned model')


def finish_result(payload, started, payload_path, result_path):
    """Publish a verdict only after all scientific payload I/O is charged."""
    errors = list(payload['errors'])
    payload_hash = None
    try:
        write_json(payload_path, payload)
        payload_hash = sha(payload_path)
    except BaseException as error:
        errors.append(dict(stage='payload', type=type(error).__name__, message=str(error)))
    elapsed = time.monotonic() - started
    status = 'complete' if not errors and elapsed <= SPEC['max_work_seconds'] else 'incomplete'
    certificate = payload['certificate']
    passed = bool(status == 'complete' and certificate is not None and certificate['passed'])
    result = dict(payload, kind='stationary_control_result', errors=errors, execution_status=status,
                  work_seconds=elapsed, scientific_payload_sha256=payload_hash, passed=passed)
    write_json(result_path, result)
    return result


def run():
    configure_runtime()
    checked = proofs()
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight_hash = sha_bytes(preflight_bytes)
    preflight = json.loads(preflight_bytes)
    if (preflight.get('kind') != 'stationary_control_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact committed independent preflight required')
    committed(PREFLIGHT)
    names = ('control_v0_attempt.json', 'control_v0_progress.jsonl', 'control_v0_dataset.npz',
             'control_v0_reference_weights.pt', 'control_v0_reference_logits.npz',
             'control_v0_payload.json', 'control_v0.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve every earlier control trace; no retry')
    write_json(FOLDER / names[0], dict(kind='stationary_control_attempt',
               created_utc=datetime.now(timezone.utc).isoformat(), protocol=SPEC,
               preflight_sha256=preflight_hash, **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    artifacts, errors, certificate, counts = {}, [], None, new_ledger()
    journal = None
    entered = returned = 0
    final_provenance_verified = False
    cases = [dict(game=f'synthetic_pair_{index // 2}', anchor_index=index % 2)
             for index in range(len(SPEC['positive_counts']))]
    def log(event, **fields):
        journal.write(json.dumps(dict(event=event, seconds=time.monotonic() - started, **fields), allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
    def save_binary(name, value, weights=False):
        buffer = io.BytesIO()
        if weights:
            torch.save(value, buffer)
        else:
            np.savez_compressed(buffer, **value)
        data = buffer.getvalue()
        if len(data) + sum(row['bytes'] for row in artifacts.values()) > SPEC['max_binary_output_bytes']:
            raise ValueError('Fixed binary storage ceiling exceeded')
        path = FOLDER / name
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(path) != sha_bytes(data) or path.stat().st_size != len(data):
            raise ValueError('Saved binary differs from the serialized payload')
        artifacts[name] = dict(sha256=sha_bytes(data), bytes=len(data))
        if not weights:
            artifacts[name]['arrays'] = {key: signature(array) for key, array in value.items()}
    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        log('started')
        check_time(deadline)
        screens = generate_screens(seed=SPEC['data_seed'], height=SPEC['height'], width=SPEC['width'],
                                   positive_counts=SPEC['positive_counts'], ledger=counts,
                                   checkpoint=lambda: check_time(deadline))
        log('initial_screens_generated', counts=counts)
        arrays, _ = collect_interventions(screens, horizon=SPEC['horizon'], ledger=counts,
                                           checkpoint=lambda: check_time(deadline),
                                           on_step_return=lambda tally: log('synthetic_step_return', counts=tally))
        targets = arrays['observed_response_envelopes']
        observed = targets.sum(axis=(1, 2)).tolist()
        if (observed != SPEC['positive_counts'] or counts['synthetic_steps'] != SPEC['expected_synthetic_steps']
                or counts['synthetic_rgb_frames'] != SPEC['expected_synthetic_steps']
                or counts['synthetic_step_entries'] != SPEC['expected_synthetic_steps']
                or counts['initial_screens_generated'] != SPEC['expected_initial_screens']
                or not np.array_equal(arrays['past_pairs'][:, 0], arrays['past_pairs'][:, 1])):
            raise ValueError('Generated observations do not match the declared family and accounting')
        check_time(deadline)
        log('synthetic_collection_complete', counts=counts, target_counts=observed)
        save_binary(names[2], arrays)
        check_time(deadline)
        model = make_reference_model()
        if (sum(p.numel() for p in model.parameters()) != SPEC['required_parameters']
                or any(p.requires_grad or p.device.type != 'cpu' or p.dtype != torch.float32
                       or not torch.isfinite(p).all() for p in model.parameters())
                or any(module.training for module in model.modules())):
            raise ValueError('Only the frozen explicit CPU judge model is allowed')
        save_binary(names[3], model.state_dict(), weights=True)
        inputs = torch.from_numpy(arrays['past_pairs'].transpose(0, 1, 4, 2, 3).copy()
                                  .reshape(len(cases), 6, SPEC['height'], SPEC['width'])).float() / 255
        check_time(deadline)
        log('reference_forward_enter')
        entered += 1
        with torch.inference_mode():
            logits = model(inputs)
        returned += 1
        log('reference_forward_return')
        check_time(deadline)
        certificate = certificate_result(logits, targets, cases)
        save_binary(names[4], dict(reference_logits=logits[:, 0].numpy().copy()))
        if entered != SPEC['reference_forward_batches'] or returned != entered:
            raise ValueError('Exactly one returned reference forward required')
        check_time(deadline)
        if proofs() != checked or sha(PREFLIGHT) != preflight_hash:
            raise ValueError('Final provenance or preflight differs from the attempt')
        committed(PREFLIGHT)
        for name, artifact in artifacts.items():
            if sha(FOLDER / name) != artifact['sha256']:
                raise ValueError('Scientific artifact changed during construction')
        check_time(deadline)
        final_provenance_verified = True
        log('certificate_complete', passed=certificate['passed'])
    except BaseException as error:
        errors.append(dict(stage='construction', type=type(error).__name__, message=str(error)))
    finally:
        if journal is not None:
            try:
                log('finished', errors=errors)
            except BaseException as error:
                errors.append(dict(stage='journal_finished', type=type(error).__name__, message=str(error)))
            try:
                journal.close()
            except BaseException as error:
                errors.append(dict(stage='journal_close', type=type(error).__name__, message=str(error)))
    payload = dict(kind='stationary_control_scientific_payload', **checked, protocol=SPEC,
                   preflight_sha256=preflight_hash, attempt_sha256=sha(FOLDER / names[0]),
                   progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None,
                   errors=errors, certificate=certificate, artifacts=artifacts, counts=counts, cases=cases,
                   synthetic_step_count_exact=counts['synthetic_step_entries'] == counts['synthetic_steps'],
                   reference_forward_entries=entered, reference_forward_returns=returned,
                   reference_forward_count_exact=entered == returned,
                   final_provenance_verified=final_provenance_verified,
                   candidate_execution_status='complete' if not errors else 'incomplete',
                   optimizer_updates=0, agent_model_inferences=0, new_emulated_frames=0, gpu_used=False,
                   claim=SPEC['claim'])
    result = finish_result(payload, started, FOLDER / names[5], FOLDER / names[6])
    print(json.dumps(dict(execution_status=result['execution_status'], passed=result['passed'],
                          work_seconds=result['work_seconds'], counts=counts,
                          reference_forward_entries=entered, reference_forward_returns=returned,
                          errors=result['errors'], certificate=certificate), allow_nan=False))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
