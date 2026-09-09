"""Prospective P-only fit of the separately reviewed priority/motion candidate.

No old run is replayed. Image fitting is not a foreground, agency or Atari gate.
"""
import json
import time

import torch

from scripts import check_free_displacement_fit_v70 as previous_run
from ragnarok.learning.priority_motion_parser_v70 import PriorityMotionParser
from ragnarok.learning.priority_warp_rgb_v70 import priority_warp_rgb

base = previous_run.base
ROOT = base.ROOT
PROTOCOL = 'docs/ARC4_PRIORITY_MOTION_FIT_PROTOCOL.json'
FOLDER = ROOT / 'craft_v6_out/v70_priority_motion_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
ATTEMPT = FOLDER / 'fit_v0_attempt.json'
OUTPUT = FOLDER / 'fit_v0.json'
FREE_RESULT = 'craft_v6_out/v70_free_displacement_fit/fit_v0.json'
FREE_SHA = 'c4e72945902e06430e2cff84bf604d9860f6aaef110bba2a12fd4775f41b9e68'
REVIEWS = (
    'craft_v6_out/v70_free_displacement_fit/fit_review_v0.json',
    'craft_v6_out/v70_parser_synthetic/priority_warp_review_v0.json',
    'craft_v6_out/v70_parser_synthetic/priority_motion_review_v0.json',
)
CANDIDATE_SOURCES = (
    'ragnarok/learning/priority_warp_rgb_v70.py', 'tests/test_priority_warp_rgb_v70.py',
    'docs/ARC4_PRIORITY_WARP_DRAFT.md', 'ragnarok/learning/priority_motion_parser_v70.py',
    'tests/test_priority_motion_parser_v70.py', 'docs/ARC4_PRIORITY_MOTION_DRAFT.md',
)
NEW_SOURCES = (PROTOCOL, 'docs/ARC4_PRIORITY_MOTION_FIT_DRAFT.md',
               'scripts/check_priority_motion_fit_v70.py', 'tests/test_priority_motion_fit_protocol_v70.py')
SPEC = dict(
    arm='ordered_with_predicted_priority', parser_seed=764, updates=256, batch_size=2,
    objective='mean_rgb_l1_box_widths_1_3_7_priority_reconstruction_only',
    learning_rate=.0003, betas=[.9, .999], epsilon=1e-8, weight_decay=0.,
    max_gradient_norm=1., measure_at=[0, 1, 8, 32, 128, 256], max_work_seconds=300,
    native_fit_threshold=.01171875, initial_native_absolute_tolerance=2e-7,
    initial_visual_weights_exact=True, initial_log_priorities_exact_zero=True,
    final_diagnostic_arrays_retained=True, baseline_reexecuted=False,
    foreground_or_control_objective_used=False,
    cpu_threads=1, emulated_frames=0, gpu_used=False, independent_generalization_test=False,
)


def proofs():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    previous = previous_run.proofs()
    if (protocol.get('kind') != 'prospective_priority_motion_fit_v0'
            or protocol.get('spec') != SPEC or protocol.get('runtime') != previous['runtime']
            or protocol.get('gate_result') is not None):
        raise ValueError('Exact prospective priority-motion CPU protocol required')
    if base.sha(ROOT / FREE_RESULT) != FREE_SHA:
        raise ValueError('The disclosed preceding free-field result changed')
    sources, evidence = dict(previous['source_sha256']), dict(previous['evidence_sha256'])
    for index, path in enumerate(REVIEWS):
        review = json.loads((ROOT / path).read_text(encoding='utf-8'))
        if review.get('passed') is not True:
            raise ValueError('All independent result and candidate-source reviews are required')
        if index == 0 and review['evidence_sha256'].get(FREE_RESULT) != FREE_SHA:
            raise ValueError('The exact free-field result must be independently reviewed')
        if index > 0 and any(review['source_sha256'].get(p) != base.sha(ROOT / p)
                             for p in CANDIDATE_SOURCES[(index - 1) * 3:index * 3]):
            raise ValueError('The complete current priority candidate must be independently reviewed')
        evidence[path] = base.sha(ROOT / path)
        for field, destination in (('source_sha256', sources), ('evidence_sha256', evidence)):
            for item, expected in review[field].items():
                if (base.sha(ROOT / item) != expected
                        or item in destination and destination[item] != expected):
                    raise ValueError('Reviewed source or evidence changed or conflicts')
                destination[item] = expected
    for path in NEW_SOURCES:
        sources[path] = base.sha(ROOT / path)
    for path in {*sources, *evidence}:
        base.require_committed(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=previous['runtime'])


def preflight():
    proof = proofs()
    review = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind') != 'priority_motion_fit_independent_preflight'
            or review.get('passed') is not True
            or any(review.get(key) != value for key, value in proof.items())):
        raise ValueError('Exact independent preflight required before any candidate update')
    base.require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof, preflight_review_sha256=base.sha(PREFLIGHT))


def predict(model, data):
    previous, current, future = data
    output = model(previous, current)  # UNKNOWN effect; no future in this call.
    source, target = base.max_pool_rgb(previous, current).float() / 255, future.float() / 255
    rendered = priority_warp_rgb(source, output.visual.displacement, output.log_priority)
    candidate = previous_run.prior.blurred_prediction_loss(rendered['reconstructed'], target)
    return output, source, target, rendered, candidate


def measure(model, data, updates):
    with torch.no_grad():
        output, _, target, rendered, candidate = predict(model, data)
        reconstruction = rendered['reconstructed']
        errors = (reconstruction - target).abs().mean(dim=(1, 2, 3))
        if not torch.isfinite(errors).all():
            raise ValueError('Nonfinite candidate native measurements')
        return dict(
            updates=updates, image_l1=float(errors.mean()), scene_l1=errors.tolist(),
            reconstruction_difference=float((reconstruction[0] - reconstruction[1]).abs().mean()),
            candidate_objective=float(candidate['loss']),
            scale_l1={str(k): float(v) for k, v in candidate['scale_l1'].items()},
            log_priority_abs_max=float(output.log_priority.abs().max()),
            log_priority_mean_per_scene=output.log_priority.mean(dim=(1, 2, 3)).tolist(),
            reconstruction_sum_per_scene=reconstruction.sum(dim=(1, 2, 3)).tolist(),
            displacement_abs_max=float(output.visual.displacement.abs().max()),
        )


def initial_measurement(model, data):
    original = previous_run.prior.baseline()
    if base.weight_sha256(model.visual) != original['initial_weight_sha256']:
        raise ValueError('Exact virgin visual-model weights required')
    first = measure(model, data, 0)
    old = original['measurements'][0]
    compared = [(first[k], old[k]) for k in ('image_l1', 'reconstruction_difference')]
    compared.extend(zip(first['scene_l1'], old['scene_l1'], strict=True))
    if (first['log_priority_abs_max'] != 0
            or any(abs(left - right) > SPEC['initial_native_absolute_tolerance'] for left, right in compared)):
        raise ValueError('Zero initial priorities and prospectively bounded native rounding required')
    return first


def final_arrays(model, data):
    with torch.no_grad():
        output, source, target, rendered, _ = predict(model, data)
        return dict(
            layout='BCHW', image_and_field_dtype='float32', weighted_coverage_dtype='float64',
            source_rgb=source.tolist(), target_rgb=target.tolist(),
            displacement=output.visual.displacement.tolist(), log_priority=output.log_priority.tolist(),
            reconstruction_rgb=rendered['reconstructed'].tolist(),
            coverage=rendered['coverage'].tolist(), weighted_coverage=rendered['weighted_coverage'].tolist(),
        )


def run():
    if ATTEMPT.exists() or OUTPUT.exists():
        raise ValueError('Preserve every priority-model attempt/result; no retry')
    proof = preflight()
    FOLDER.mkdir(parents=True, exist_ok=True)
    record = dict(status='started', started_utc=base.utc(), **proof,
                  completed_updates=0, optimizer_step_in_progress=None, measurements=[])
    base.write_new(ATTEMPT, record)
    work_start = time.monotonic()
    deadline = work_start + SPEC['max_work_seconds']
    try:
        data = base.scenes()
        torch.manual_seed(SPEC['parser_seed'])
        with torch.device('cpu'):
            model = PriorityMotionParser()
        initial_weights = base.weight_sha256(model)
        initial_visual_weights = base.weight_sha256(model.visual)
        rows = record['measurements']
        rows.append(initial_measurement(model, data))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=SPEC['learning_rate'], betas=tuple(SPEC['betas']),
            eps=SPEC['epsilon'], weight_decay=SPEC['weight_decay'], foreach=False, fused=False,
        )
        for update in range(1, SPEC['updates'] + 1):
            if time.monotonic() > deadline:
                raise ValueError('Priority-model work ceiling reached')
            optimizer.zero_grad(set_to_none=True)
            _, _, _, _, candidate = predict(model, data)
            loss = candidate['loss']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite priority-model objective')
            loss.backward()
            base.admit_adam_gradients(model, max_norm=SPEC['max_gradient_norm'])
            record['optimizer_step_in_progress'] = update
            optimizer.step()
            record['completed_updates'] = update
            record['optimizer_step_in_progress'] = None
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite updated priority-model parameters')
            if update in SPEC['measure_at']:
                rows.append(measure(model, data, update))
        optimizer.zero_grad(set_to_none=True)
        arrays = final_arrays(model, data)
        work_seconds = time.monotonic() - work_start
        if work_seconds > SPEC['max_work_seconds'] or preflight() != proof:
            raise ValueError('Bounded priority fit and unchanged proof required')
        result = dict(
            kind='priority_motion_fit_result', execution_complete=True, created_utc=base.utc(),
            **proof, optimizer_updates=256, parameter_count=sum(p.numel() for p in model.parameters()),
            initial_weight_sha256=initial_weights, initial_visual_weight_sha256=initial_visual_weights,
            final_weight_sha256=base.weight_sha256(model), measurements=rows,
            final_arrays=arrays, work_seconds=work_seconds,
            native_fit_below_declared_threshold=rows[-1]['image_l1'] < SPEC['native_fit_threshold'],
            new_emulated_frames=0, gpu_used=False, gate_result=None,
            limitations=json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))['limitations'],
        )
        base.write_new(OUTPUT, result)
        record.update(status='complete', completed_utc=base.utc(), result_sha256=base.sha(OUTPUT))
        return result
    except BaseException as error:
        record.update(status='failed', failed_utc=base.utc(), error_type=type(error).__name__,
                      error=str(error), no_retry=True)
        raise
    finally:
        ATTEMPT.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = run()
    print(json.dumps(dict(execution_complete=True, final_native_image_l1=result['measurements'][-1]['image_l1'],
                          native_fit_below_declared_threshold=result['native_fit_below_declared_threshold'])))
