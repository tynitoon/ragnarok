"""A fixed-budget fit of a zero-initialized RGB shortcut beside the original CNN."""
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
    anchor_balanced_bce, measure, feasibility_verdict)
from ragnarok.learning.direct_response_envelope_v70 import DirectResponseEnvelope as ResponseEnvelope
from scripts.fit_stationary_affine_v70 import proofs as affine_proofs
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
FOLDER = ROOT / 'craft_v6_out/v70_stationary_direct_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
PROTOCOL = ROOT / 'docs/ARC4_STATIONARY_DIRECT_PROTOCOL.json'
INITIAL_MANIFEST = ROOT / 'docs/ARC4_STATIONARY_DIRECT_INITIAL_IDENTITY.json'
AFFINE_RESULT = ROOT / 'craft_v6_out/v70_stationary_affine_fit/fit_v0.json'
AFFINE_REVIEW = ROOT / 'craft_v6_out/v70_stationary_affine_fit/fit_review_v0.json'
AFFINE_RESULT_SHA = 'c60b92fea8846af3cbc8bfe9db95bdfc1585ad9c661b276462e9db67f2aa8014'
AFFINE_REVIEW_SHA = '8be644481304ed9393f62fa93232ed5534dc46f5ee5fa2775226b710f0a646f5'
BASELINE_INITIAL = ROOT / 'craft_v6_out/v70_stationary_fit/initial_weights.pt'
SOURCES = ('ragnarok/learning/direct_response_envelope_v70.py', 'tests/test_direct_response_envelope_v70.py',
           'scripts/fit_stationary_direct_v70.py', 'tests/test_stationary_direct_fit_v70.py',
           'docs/ARC4_STATIONARY_DIRECT_PATH_DRAFT.md', 'docs/ARC4_STATIONARY_DIRECT_PROTOCOL.json',
           'docs/ARC4_STATIONARY_DIRECT_INITIAL_IDENTITY.json')
SPEC = copy.deepcopy(ATARI_SPEC)
SPEC.update(batch='all twelve certified synthetic cases, unchanged manifest order',
            evaluation='initial and final only; per nonempty case, then equal nonempty pair means',
            targets_role='development calibration, not six games or independent validation',
            inherited_synthetic_steps=216, new_synthetic_steps=0, new_initial_screens=0,
            probability_threshold=.5, expected_training_forward_batches=512,
            expected_evaluation_forward_batches=3, retry=False,
            input_loading='decode past_pairs and observed_response_envelopes only; no futures or reference',
            claim='fixed-budget feasibility of the augmented CNN on the same development family only',
            parameters=10184, direct_path='six RGB coefficients plus bias, all initially zero',
            initialization_identity='contextual parameter signatures, input, total logits/probabilities and metrics match archived CNN initial before any update',
            component_evaluation='save contextual and direct logits/probabilities in the existing three evaluations; descriptive removal metrics only',
            component_limits='global clipping and subsequent CNN gradients change; component removals do not identify the cause of learning')
SPEC['architecture']['direct_path'] = dict(input_channels=6, output_channels=1, kernel=1,
                                         bias=True, initialization='all zero without RNG consumption')


def proofs():
    """Reuse the reviewed calibration lineage, without decoding old weights or arrays."""
    checked = affine_proofs()
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Fixed direct-path fitting specification changed')
    if sha(AFFINE_RESULT) != AFFINE_RESULT_SHA or sha(AFFINE_REVIEW) != AFFINE_REVIEW_SHA:
        raise ValueError('Exact reviewed positive affine calibration required')
    result, review = (json.loads(path.read_text(encoding='utf-8')) for path in (AFFINE_RESULT, AFFINE_REVIEW))
    if (result['execution_status'] != 'complete' or result['source_sha256'] != checked['source_sha256']
            or any(result['verdicts'][recipe]['passed'] is not True for recipe in ('newton', 'adam'))
            or review.get('kind') != 'stationary_affine_independent_result_review' or review.get('passed') is not True
            or review.get('result_sha256') != AFFINE_RESULT_SHA or review.get('source_sha256') != result['source_sha256']):
        raise ValueError('Unchanged independently reviewed prior calibration required')
    manifest = json.loads(INITIAL_MANIFEST.read_text(encoding='utf-8'))
    if (manifest['kind'] != 'stationary_direct_path_initial_identity'
            or manifest['baseline_result_sha256'] != '8ca241233e8053cedd6149b1d3992dc7cbc944c4442ce298b067dafc03719c35'
            or manifest['baseline_review_sha256'] != 'c2fe9c270afa098f43c403b3f4818460bd9540993794049cd7df04f1151daa98'
            or sha(BASELINE_INITIAL) != manifest['baseline_initial_weights_sha256']
            or any(checked['evidence_sha256'].get(key) != value for key, value in manifest['evidence_sha256'].items())):
        raise ValueError('Frozen random-initial identity manifest differs from the historical experiment')
    sources, evidence = dict(checked['source_sha256']), dict(checked['evidence_sha256'])
    for rel in SOURCES:
        if rel in sources: raise ValueError('New fitting sources cannot replace the prior lineage')
        committed(ROOT / rel)
        sources[rel] = sha(ROOT / rel)
    for path in (AFFINE_RESULT, AFFINE_REVIEW, BASELINE_INITIAL):
        committed(path)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=checked['runtime'])


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
    """Save the summed output first, then both contributions from that same call."""
    with torch.inference_mode():
        logits = model(inputs)
    raw = logits[:, 0].numpy().copy()
    save(name + '_logits.npz', dict(logits=raw))
    components = model.evaluation_components
    if components is None or len(components) != 2:
        raise ValueError('Both contributions from the unique evaluation must be present')
    contextual, direct = components
    component_raw = dict(contextual_logits=contextual[:, 0].numpy().copy(), direct_logits=direct[:, 0].numpy().copy())
    save(name + '_components_logits.npz', component_raw)
    if not np.isfinite(raw).all() or any(not np.isfinite(value).all() for value in component_raw.values()):
        raise FloatingPointError('Nonfinite evaluation logits')
    if not torch.equal(contextual + direct, logits):
        raise ValueError('The saved contributions must sum exactly to the returned output')
    if name == 'initial' and torch.count_nonzero(direct).item() != 0:
        raise ValueError('The initial direct contribution must be exactly zero')
    probability = logits.sigmoid()[:, 0].numpy().copy()
    save(name + '_predictions.npz', dict(probability=probability))
    component_probability = dict(contextual_probability=contextual.sigmoid()[:, 0].numpy().copy(),
                                 direct_probability=direct.sigmoid()[:, 0].numpy().copy())
    save(name + '_components_predictions.npz', component_probability)
    measured = measure(probability, targets, cases)
    bce = float(anchor_balanced_bce(logits, labels))
    if not np.isfinite(bce):
        raise FloatingPointError('Nonfinite evaluation BCE')
    measured['balanced_bce'] = bce
    descriptions = {}
    for key, value in (('contextual', contextual), ('direct', direct)):
        row = measure(component_probability[key + '_probability'], targets, cases)
        loss = float(anchor_balanced_bce(value, labels))
        if not np.isfinite(loss): raise FloatingPointError('Nonfinite descriptive component BCE')
        row['balanced_bce'] = loss
        descriptions[key] = row
    return measured, probability, descriptions


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
    result = dict(payload, kind='stationary_direct_path_development_fit', failures=failures,
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
    if (preflight.get('kind') != 'stationary_direct_independent_preflight' or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('Exact committed independent prospective fit preflight required')
    committed(PREFLIGHT)
    names = ('fit_v0_attempt.json', 'fit_v0_progress.jsonl', 'fit_v0.json', 'initial_weights.pt',
             'final_weights.pt', 'initial_logits.npz', 'initial_predictions.npz', 'visual_logits.npz',
             'visual_predictions.npz', 'constant_logits.npz', 'constant_predictions.npz', 'fit_v0_payload.json')
    names += tuple(f'{arm}_components_{kind}.npz' for arm in ('initial', 'visual', 'constant')
                   for kind in ('logits', 'predictions'))
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve all prior fit traces; no retry')
    control_result = json.loads(CONTROL_RESULT.read_text(encoding='utf-8'))
    identity = json.loads(INITIAL_MANIFEST.read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='stationary_direct_fit_attempt', created_utc=datetime.now(timezone.utc).isoformat(),
                                      protocol=SPEC, preflight_sha256=preflight_hash, **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    models, artifacts, metrics, component_metrics, failures = {}, {}, {}, {}, []
    counts, entries = dict(visual=0, constant=0), dict(visual=0, constant=0)
    forward_entries = {kind: dict(visual=0, constant=0) for kind in ('training', 'evaluation')}
    forward_returns = copy.deepcopy(forward_entries)
    pending, phase = {'arm': None}, {'name': 'evaluation'}
    journal, final_provenance_verified = None, False
    initial_identity_verified = False

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
        if (sum(value.numel() for value in visual.parameters()) != 10184
                or any(value.device.type != 'cpu' or value.dtype != torch.float32
                       or not torch.isfinite(value).all() for value in visual.parameters())):
            raise ValueError('Fixed architecture parameter population changed')
        generated_state = {key: signature(value.detach().numpy()) for key, value in visual.contextual.state_dict().items()}
        if generated_state != identity['state_arrays'] or visual.direct_weight.any() or visual.direct_bias.any():
            raise ValueError('Generated initial contextual weights or direct zeros differ from the declared identity')
        models['constant'] = copy.deepcopy(visual)
        if any(not torch.equal(a, b) for a, b in zip(visual.parameters(), models['constant'].parameters(), strict=True)):
            raise ValueError('Both arms must start at identical weights')
        for arm, model in models.items():
            attach_forward_counters(model, arm, phase, forward_entries, forward_returns)
        save_binary('initial_weights.pt', visual.state_dict(), torch_format=True)
        event('evaluation_started', arm='visual', checkpoint='initial')
        metrics['initial'], initial, component_metrics['initial'] = evaluate_saved(
            visual, inputs, labels, targets, control_result['cases'], save_binary, 'initial')
        if (artifacts['initial_logits.npz']['arrays']['logits'] != identity['initial_logits']
                or artifacts['initial_predictions.npz']['arrays']['probability'] != identity['initial_predictions']
                or metrics['initial'] != identity['initial_metrics']):
            raise ValueError('Unique initial output or measures differ from the historical CNN initial')
        initial_identity_verified = True
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
            metrics[arm], _, component_metrics[arm] = evaluate_saved(models[arm], arm_inputs[arm], labels, targets,
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

    trace_hashes = {names[0]: None, names[1]: None}
    for name in trace_hashes:
        try:
            trace_hashes[name] = sha(FOLDER / name)
        except BaseException as error:
            fail('trace_hash_' + name, error)
    expected_forwards = dict(training=dict(visual=SPEC['updates_per_arm'], constant=SPEC['updates_per_arm']),
                             evaluation=dict(visual=2, constant=1))
    candidate_complete = (not failures and pending['arm'] is None and final_provenance_verified
                          and initial_identity_verified and set(component_metrics) == {'initial', 'visual', 'constant'}
                          and counts == entries == dict(visual=SPEC['updates_per_arm'], constant=SPEC['updates_per_arm'])
                          and forward_entries == forward_returns == expected_forwards
                          and set(metrics) == {'initial', 'zero', 'visual', 'constant'})
    candidate_verdict = (feasibility_verdict(metrics, counts, SPEC['updates_per_arm']) if candidate_complete
                         else dict(passed=False, complete=False, reason='Incomplete or failed execution'))
    payload = dict(kind='stationary_direct_fit_scientific_payload', **checked, protocol=SPEC,
                   preflight_sha256=preflight_hash, failures=failures, artifacts=artifacts, metrics=metrics,
                   component_metrics=component_metrics, initial_identity_verified=initial_identity_verified,
                   cases=control_result['cases'], candidate_execution_complete=candidate_complete,
                   candidate_verdict=candidate_verdict, completed_updates=counts, optimizer_step_entries=entries,
                   pending_optimizer_arm=pending['arm'], optimizer_update_count_exact=pending['arm'] is None,
                   optimizer_updates=sum(counts.values()), optimizer_updates_upper_bound=sum(counts.values()) + int(pending['arm'] is not None),
                   model_forward_entries=forward_entries, model_forward_returns=forward_returns,
                   model_forward_count_exact=forward_entries == forward_returns,
                   final_provenance_verified=final_provenance_verified,
                   attempt_sha256=trace_hashes[names[0]], progress_sha256=trace_hashes[names[1]],
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
