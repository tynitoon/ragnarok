"""One prospective diagnostic: fit independent dense fields, without a network.

Scene-specific parameters make this an optimization witness, never a learned
perception system. Old fitting runs, their losses and their results are retained.
"""
import json
import time

import torch

from scripts import check_blurred_motion_fit_v70 as prior
from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb

base = prior.base
ROOT = base.ROOT
PROTOCOL = 'docs/ARC4_FREE_DISPLACEMENT_FIT_PROTOCOL.json'
FOLDER = ROOT / 'craft_v6_out/v70_free_displacement_fit'
PREFLIGHT = FOLDER / 'preflight_v0.json'
ATTEMPT = FOLDER / 'fit_v0_attempt.json'
OUTPUT = FOLDER / 'fit_v0.json'
PRIOR_RESULT = 'craft_v6_out/v70_blurred_motion_fit/fit_v0.json'
PRIOR_SHA = '598226a743bfb404e5068ff74648c9564036b4f5e4be28f8644a2b20beef7bfc'
PRIOR_REVIEW = 'craft_v6_out/v70_blurred_motion_fit/fit_review_v0.json'
NEW_SOURCES = (
    PROTOCOL, 'docs/ARC4_FREE_DISPLACEMENT_FIT_DRAFT.md',
    'scripts/check_free_displacement_fit_v70.py',
    'tests/test_free_displacement_fit_protocol_v70.py',
    'docs/ARC4_MOTION_BOOTSTRAP_REVIEW_20260910.md',
)
SPEC = dict(
    arm='independent_free_dense_fields', field_shape=[2, 2, 16, 16],
    initialization='exact_zero_without_rng', parameter_dtype='float32',
    independent_parameters_per_scene=True, parameter_count=1024,
    updates=256, batch_size=2, objective='mean_rgb_l1_box_widths_1_3_7',
    learning_rate=.05, betas=[.9, .999], epsilon=1e-8, weight_decay=0.,
    max_gradient_norm=1., measure_at=[0, 1, 8, 32, 128, 256],
    max_work_seconds=300, success_metric='native_image_l1',
    native_fit_threshold=.01171875, final_diagnostic_arrays_retained=True,
    baseline_reexecuted=False, cpu_threads=1, emulated_frames=0, gpu_used=False,
    independent_generalization_test=False,
)


class FreeDisplacements(torch.nn.Module):
    """No image, direction, identity or target enters this initialization."""

    def __init__(self):
        super().__init__()
        self.displacement = torch.nn.Parameter(
            torch.zeros(SPEC['field_shape'], dtype=torch.float32, device='cpu')
        )


def proofs():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    previous = prior.proofs()
    if (protocol.get('kind') != 'prospective_free_displacement_fit_v0'
            or protocol.get('spec') != SPEC
            or protocol.get('runtime') != previous['runtime']
            or protocol.get('gate_result') is not None):
        raise ValueError('Exact prospective free-field CPU protocol required')
    if base.sha(ROOT / PRIOR_RESULT) != PRIOR_SHA:
        raise ValueError('The disclosed negative multiscale network result changed')
    result = json.loads((ROOT / PRIOR_RESULT).read_text(encoding='utf-8'))
    if (result.get('execution_complete') is not True
            or result.get('optimizer_updates') != 256
            or result.get('native_fit_below_declared_threshold') is not False):
        raise ValueError('This diagnostic is conditional on the completed negative network fit')
    review = json.loads((ROOT / PRIOR_REVIEW).read_text(encoding='utf-8'))
    if (review.get('passed') is not True
            or review['evidence_sha256'].get(PRIOR_RESULT) != PRIOR_SHA):
        raise ValueError('Independent review of the precise preceding result required')
    sources = dict(previous['source_sha256'])
    evidence = dict(previous['evidence_sha256'])
    evidence[PRIOR_REVIEW] = base.sha(ROOT / PRIOR_REVIEW)
    for field, destination in (('source_sha256', sources), ('evidence_sha256', evidence)):
        for path, expected in review[field].items():
            if (base.sha(ROOT / path) != expected
                    or path in destination and destination[path] != expected):
                raise ValueError('Reviewed source or evidence changed or conflicts')
            destination[path] = expected
    for path in NEW_SOURCES:
        sources[path] = base.sha(ROOT / path)
    for path in {*sources, *evidence}:
        base.require_committed(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=previous['runtime'])


def preflight():
    proof = proofs()
    review = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (review.get('kind') != 'free_displacement_fit_independent_preflight'
            or review.get('passed') is not True
            or any(review.get(key) != value for key, value in proof.items())):
        raise ValueError('Exact independent preflight required before any field update')
    base.require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof, preflight_review_sha256=base.sha(PREFLIGHT))


def objective(model, source, target):
    reconstruction, coverage = forward_warp_rgb(source, model.displacement)
    candidate = prior.blurred_prediction_loss(reconstruction, target)
    return candidate, reconstruction, coverage


def measure(model, source, target, updates):
    with torch.no_grad():
        candidate, reconstruction, coverage = objective(model, source, target)
        errors = (reconstruction - target).abs().mean(dim=(1, 2, 3))
        if not torch.isfinite(errors).all() or not torch.isfinite(coverage).all():
            raise ValueError('Nonfinite free-field measurement')
        return dict(
            updates=updates, image_l1=float(errors.mean()), scene_l1=errors.tolist(),
            reconstruction_difference=float((reconstruction[0] - reconstruction[1]).abs().mean()),
            candidate_objective=float(candidate['loss']),
            scale_l1={str(k): float(v) for k, v in candidate['scale_l1'].items()},
            reconstruction_sum_per_scene=reconstruction.sum(dim=(1, 2, 3)).tolist(),
            displacement_abs_max=float(model.displacement.abs().max()),
            coverage_min=float(coverage.min()), coverage_max=float(coverage.max()),
            uncovered_fraction=float((coverage == 0).float().mean()),
        )


def final_arrays(model, source, target):
    with torch.no_grad():
        _, reconstruction, coverage = objective(model, source, target)
        return dict(
            layout='BCHW', dtype='float32', source_rgb=source.tolist(), target_rgb=target.tolist(),
            displacement=model.displacement.tolist(), reconstruction_rgb=reconstruction.tolist(),
            coverage=coverage.tolist(),
        )


def run():
    if ATTEMPT.exists() or OUTPUT.exists():
        raise ValueError('Preserve every field attempt/result; no retry')
    proof = preflight()
    FOLDER.mkdir(parents=True, exist_ok=True)
    record = dict(status='started', started_utc=base.utc(), **proof,
                  completed_updates=0, optimizer_step_in_progress=None, measurements=[])
    base.write_new(ATTEMPT, record)
    work_start = time.monotonic()
    deadline = work_start + SPEC['max_work_seconds']
    try:
        previous, current, future = base.scenes()
        source = base.max_pool_rgb(previous, current).float() / 255
        target = future.float() / 255
        model = FreeDisplacements()
        initial_weights = base.weight_sha256(model)
        rows = record['measurements']
        rows.append(measure(model, source, target, 0))
        if rows[0]['image_l1'] != .046875 or rows[0]['reconstruction_difference'] != 0:
            raise ValueError('Exact zero-field initial reconstruction required')
        optimizer = torch.optim.Adam(
            model.parameters(), lr=SPEC['learning_rate'], betas=tuple(SPEC['betas']),
            eps=SPEC['epsilon'], weight_decay=SPEC['weight_decay'], foreach=False, fused=False,
        )
        for update in range(1, SPEC['updates'] + 1):
            if time.monotonic() > deadline:
                raise ValueError('Free-field work ceiling reached')
            optimizer.zero_grad(set_to_none=True)
            candidate, _, _ = objective(model, source, target)
            loss = candidate['loss']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite free-field objective')
            loss.backward()
            base.admit_adam_gradients(model, max_norm=SPEC['max_gradient_norm'])
            record['optimizer_step_in_progress'] = update
            optimizer.step()
            record['completed_updates'] = update
            record['optimizer_step_in_progress'] = None
            if not torch.isfinite(model.displacement).all():
                raise ValueError('Nonfinite updated free fields')
            if update in SPEC['measure_at']:
                rows.append(measure(model, source, target, update))
        optimizer.zero_grad(set_to_none=True)
        arrays = final_arrays(model, source, target)
        work_seconds = time.monotonic() - work_start
        if work_seconds > SPEC['max_work_seconds'] or preflight() != proof:
            raise ValueError('Bounded field fit and unchanged proof required')
        result = dict(
            kind='free_displacement_fit_result', execution_complete=True, created_utc=base.utc(),
            **proof, optimizer_updates=256, parameter_count=sum(p.numel() for p in model.parameters()),
            initial_weight_sha256=initial_weights, final_weight_sha256=base.weight_sha256(model),
            measurements=rows, final_arrays=arrays, work_seconds=work_seconds,
            native_fit_below_declared_threshold=rows[-1]['image_l1'] < SPEC['native_fit_threshold'],
            preceding_network_result_sha256=PRIOR_SHA,
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
    print(json.dumps(dict(execution_complete=True,
                          final_native_image_l1=result['measurements'][-1]['image_l1'],
                          native_fit_below_declared_threshold=result['native_fit_below_declared_threshold'])))
