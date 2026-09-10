"""Generation and learning run in distinct processes, under one parent's deadline.

The fitting entry point never imports the scene generator or opens judge files.
The control fixture definitions generate observations only; no emulator is made.
"""
import argparse
import io
import json
import os
import traceback

from scripts.photometric_common_v70 import (
    ROOT, OUT, PROTOCOL, check_time, elapsed, read_json, sha, utc,
    ensure_space, verify_sources, write_bytes, write_json,
)


def save_npz(name, **arrays):
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    write_bytes(OUT / name, stream.getvalue())


def generate(intent):
    import numpy as np
    from ragnarok.evaluation.photometric_scenes_v70 import training_table, evaluation_table
    from tests.test_counterfactual_occlusion_limits_v70 import (
        _changed_occluder_world, _hidden_subject_world,
    )
    check_time(intent)
    write_json(OUT / 'generation_train_started_v0.json', dict(
        started_utc=utc(), completed_clips_lower=0, in_flight_clips_upper=128,
        images_per_clip=3, interruption='generation inside this unreturned table has unknown progress'))
    train_rgb, train_judge = training_table()
    save_npz('train_rgb.npz', rgb=train_rgb)
    write_json(OUT / 'train_judge.json', train_judge)
    write_json(OUT / 'generation_train_saved_v0.json', dict(
        completed_clips=128, completed_utc=utc(), rgb_sha256=sha(OUT / 'train_rgb.npz'),
        judge_sha256=sha(OUT / 'train_judge.json')))
    check_time(intent)
    write_json(OUT / 'generation_eval_started_v0.json', dict(
        started_utc=utc(), completed_clips_lower=128, in_flight_clips_upper=1280,
        images_per_clip=3, interruption='generation inside this unreturned table has unknown progress'))
    eval_rgb, eval_judge = evaluation_table()
    save_npz('eval_rgb.npz', rgb=eval_rgb)
    write_json(OUT / 'eval_judge.json', eval_judge)
    write_json(OUT / 'generation_eval_saved_v0.json', dict(
        completed_clips=1408, completed_utc=utc(), rgb_sha256=sha(OUT / 'eval_rgb.npz'),
        judge_sha256=sha(OUT / 'eval_judge.json')))
    check_time(intent)
    raw, pooled, controls = [], [], []
    for family, builder in (('changed_occluder', _changed_occluder_world),
                            ('hidden_subject', _hidden_subject_world)):
        write_json(OUT / (family+'_generation_started_v0.json'), dict(
            started_utc=utc(), completed_selected_views_lower=len(controls)*6,
            in_flight_selected_views_upper=12))
        anchor, traces, _judge_only_states = builder(subject_is_controlled=False)
        for branch_index, trace in enumerate(traces):
            raw.append(np.stack((anchor, trace.raw[1], trace.raw[3])))
            pooled.append(np.stack((anchor, trace.pooled[1], trace.pooled[3])))
            controls.append(dict(family=family, branch_index=branch_index,
                                 source_pixel=3*12+2, target_pixel=None))
        save_npz(family+'_raw_rgb.npz', rgb=np.stack(raw[-2:]))
        save_npz(family+'_pooled_rgb.npz', rgb=np.stack(pooled[-2:]))
        write_json(OUT / (family+'_generation_saved_v0.json'), dict(
            completed_utc=utc(), completed_selected_views=len(controls)*6, controls=controls[-2:]))
        check_time(intent)
    save_npz('controls_raw_rgb.npz', rgb=np.stack(raw))
    save_npz('controls_pooled_rgb.npz', rgb=np.stack(pooled))
    write_json(OUT / 'controls_judge.json', controls)
    names = ('train_rgb.npz', 'eval_rgb.npz', 'controls_raw_rgb.npz',
             'controls_pooled_rgb.npz', 'train_judge.json', 'eval_judge.json', 'controls_judge.json')
    verify_sources(intent['source_sha256'])
    check_time(intent)
    write_json(OUT / 'generation_v0.json', dict(
        completed_utc=utc(), file_sha256={n: sha(OUT / n) for n in names},
        train_clips=len(train_rgb), eval_clips=len(eval_rgb),
        new_family_rgb_frames=(len(train_rgb)+len(eval_rgb))*3,
        historical_clips=8, historical_selected_views=24, new_emulated_frames=0,
        model_calls=0, optimizer_updates=0, elapsed_seconds=elapsed(intent)))


def fit(intent):
    # Importing the learner costs time inside the common production deadline.
    import numpy as np
    import torch
    from ragnarok.learning import photometric_correspondence_v70 as method
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError('CUDA must be unavailable to this CPU experiment')
    check_time(intent)
    spec = read_json(ROOT / PROTOCOL)
    generation = read_json(OUT / 'generation_v0.json')

    def rgb_only(name, expected_shape):
        if sha(OUT / name) != generation['file_sha256'][name]:
            raise ValueError('RGB archive changed')
        with np.load(OUT / name, allow_pickle=False) as archive:
            if archive.files != ['rgb']:
                raise ValueError('Learner archives must contain only the RGB table')
            array = archive['rgb']
        if array.dtype != np.uint8 or array.shape != expected_shape:
            raise ValueError('Unexpected observed RGB shape or dtype')
        return torch.from_numpy(array.copy())

    counts = dict(encoder_calls=0, encoded_images=0, reference_association_calls=0,
                  explicit_bmm_calls=0, similarity_bmm_calls=0, walk_bmm_calls=0,
                  backward_calls=0, gradient_clip_calls=0, optimizer_steps_started=0,
                  optimizer_steps_completed=0, objective_calls=0, pooled_guard_calls=0,
                  training_clip_presentations=0)
    by_phase = {}
    phase = 'training'

    def event(kind, **details):
        payload = json.dumps(dict(kind=kind, phase=phase, elapsed_seconds=elapsed(intent),
                                  counts=dict(counts), **details), allow_nan=False)+'\n'
        ensure_space(len(payload.encode('utf-8')))
        log.write(payload)
        log.flush()
        os.fsync(log.fileno())

    def encoder_hook(_module, arguments, _output):
        counts['encoder_calls'] += 1
        counts['encoded_images'] += len(arguments[0])
        row = by_phase.setdefault(phase, dict(encoder_calls=0, encoded_images=0, explicit_bmm_calls=0))
        row['encoder_calls'] += 1
        row['encoded_images'] += len(arguments[0])

    original_bmm = torch.bmm

    def counted_bmm(left, right, *args, **kwargs):
        counts['explicit_bmm_calls'] += 1
        role = 'walk_bmm_calls' if left.shape[-2] == left.shape[-1] else 'similarity_bmm_calls'
        counts[role] += 1
        row = by_phase.setdefault(phase, dict(encoder_calls=0, encoded_images=0, explicit_bmm_calls=0))
        row['explicit_bmm_calls'] += 1
        return original_bmm(left, right, *args, **kwargs)

    def save_state(name, model):
        stream = io.BytesIO()
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, stream)
        write_bytes(OUT / name, stream.getvalue())

    def proposal_arrays(proposals):
        # Copies are detached snapshots, never reused views into Torch buffers.
        keys = ('destination', 'admitted', 'best_index', 'round_trip_score', 'endpoint_similarity')
        return {key: np.concatenate([getattr(p, key).detach().cpu().numpy().copy()
                                     for p in proposals], axis=0) for key in keys}

    def infer(model, clips, *, kind, name):
        replies = []
        for start in range(0, len(clips), spec['eval_batch_size']):
            check_time(intent)
            event('inference_batch_start', archive=name, start=start,
                  clips=min(spec['eval_batch_size'], len(clips)-start), image_kind=kind,
                  interrupted_batch_bounds=dict(encoder_calls=0 if model is None or kind == 'pooled' else 1,
                                                explicit_bmm_calls=0 if kind == 'pooled' else 4))
            reply = method.propose(model, clips[start:start+spec['eval_batch_size']], image_kind=kind)
            if model is None and reply.computed:
                counts['reference_association_calls'] += 1
            if not reply.computed:
                counts['pooled_guard_calls'] += 1
            replies.append(reply)
            if kind == 'raw':
                # Preserve every completed batch before any further model call.
                batch_name = f'{name}_batch_{start:04d}.npz'
                save_npz(batch_name, **proposal_arrays([reply]))
                event('inference_batch_saved', archive=batch_name, start=start,
                      artifact_sha256=sha(OUT / batch_name))
        if kind == 'raw':
            arrays = proposal_arrays(replies)
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise FloatingPointError('Nonfinite raw proposal')
            save_npz(name+'.npz', **arrays)
        else:
            # Pooled scores are absent, never zero/NaN stand-ins for a computation.
            serialized = []
            for reply in replies:
                if reply.computed or any(getattr(reply, key) is not None for key in
                                         ('best_index', 'round_trip_score', 'endpoint_similarity')):
                    raise ValueError('Pooled guard fabricated a computed result')
                for dest, admitted in zip(reply.destination, reply.admitted, strict=True):
                    serialized.append(dict(destination=dest.cpu().tolist(), admitted=admitted.cpu().tolist(),
                                           best_index=None, round_trip_score=None, endpoint_similarity=None,
                                           computed=False, reason=reply.reason))
            write_json(OUT / (name+'.json'), serialized)
        event('inference_table_saved', archive=name+('.npz' if kind == 'raw' else '.json'))

    model = None
    status, failure, cleanup_failures, initial_restored = 'failed', None, [], False
    log = (OUT / 'updates_v0.jsonl').open('x', encoding='utf-8', newline='\n')
    torch.bmm = counted_bmm
    try:
        train = rgb_only('train_rgb.npz', (128, 3, 8, 8, 3))
        torch.manual_seed(spec['initialization_seed'])
        model = method.PixelAppearanceEncoder().cpu()
        model.register_forward_hook(encoder_hook)
        if sum(p.numel() for p in model.parameters()) != spec['parameters']:
            raise ValueError('Parameter count differs from declaration')
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        save_state('initial_state.pt', model)
        optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'],
                                     betas=tuple(spec['adam_betas']), eps=spec['adam_eps'],
                                     weight_decay=spec['weight_decay'])
        for step in range(spec['updates']):
            check_time(intent)
            start = (step % 32)*4
            event('training_step_start', step=step+1, train_clip_start=start,
                  interrupted_step_bounds=dict(encoder_calls=1, encoded_images=12, explicit_bmm_calls=7,
                                               backward_calls=1, gradient_clip_calls=1, optimizer_steps_completed=1))
            optimizer.zero_grad(set_to_none=True)
            counts['training_clip_presentations'] += 4
            loss = method.cycle_loss(method.encode_clips(model, train[start:start+4]))
            counts['objective_calls'] += 1
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite cycle loss')
            counts['backward_calls'] += 1
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite or missing gradient')
            counts['gradient_clip_calls'] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_norm_clip'],
                                                 error_if_nonfinite=True)
            counts['optimizer_steps_started'] += 1
            event('optimizer_step_start', step=step+1)
            optimizer.step()
            counts['optimizer_steps_completed'] += 1
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite updated parameter')
            event('training_step_complete', step=step+1, train_clip_start=start, batch_clips=4,
                  loss_before_update=float(loss.detach()), gradient_norm_before_clip=float(norm))
            if (step+1) % 128 == 0:
                print(json.dumps(dict(phase='training', updates=step+1, elapsed_seconds=elapsed(intent))), flush=True)
        os.fsync(log.fileno())
        save_state('final_state.pt', model)
        model.eval()
        # Evaluation observations are decoded only after all 512 updates.
        evaluation = rgb_only('eval_rgb.npz', (1280, 3, 8, 8, 3))
        raw_controls = rgb_only('controls_raw_rgb.npz', (4, 3, 10, 12, 3))
        pooled_controls = rgb_only('controls_pooled_rgb.npz', (4, 3, 10, 12, 3))
        for representation in ('final', 'initial', 'rgb_reference'):
            phase = representation
            if representation == 'initial':
                model.load_state_dict(initial, strict=True)
                if not all(torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
                    raise ValueError('Initial parameters did not restore exactly')
                initial_restored = True
            active = None if representation == 'rgb_reference' else model
            infer(active, evaluation, kind='raw', name=representation+'_eval')
            infer(active, raw_controls, kind='raw', name=representation+'_controls_raw')
            infer(active, pooled_controls, kind='pooled', name=representation+'_controls_pooled')
        expected = dict(encoder_calls=674, encoded_images=13848, reference_association_calls=81,
                        explicit_bmm_calls=4556, similarity_bmm_calls=1510, walk_bmm_calls=3046,
                        backward_calls=512, gradient_clip_calls=512, optimizer_steps_started=512,
                        optimizer_steps_completed=512, objective_calls=512, pooled_guard_calls=3,
                        training_clip_presentations=2048)
        if counts != expected:
            raise ValueError(f'Operation accounting differs: {counts}')
        verify_sources(intent['source_sha256'])
        check_time(intent)
        status = 'complete'
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        if model is not None:
            try:
                save_state('partial_or_last_state.pt', model)
            except BaseException as cleanup_error:
                cleanup_failures.append(f'partial_checkpoint: {type(cleanup_error).__name__}: {cleanup_error}')
        traceback.print_exc()
    finally:
        torch.bmm = original_bmm
        for name, operation in (('flush', log.flush), ('fsync', lambda: os.fsync(log.fileno())), ('close', log.close)):
            try:
                operation()
            except BaseException as cleanup_error:
                cleanup_failures.append(f'{name}: {type(cleanup_error).__name__}: {cleanup_error}')
        if cleanup_failures:
            status = 'failed'
        try:
            write_json(OUT / 'fit_worker_v0.json', dict(
                execution_status=status, failure=failure, cleanup_failures=cleanup_failures,
                completed_utc=utc(), counts=counts, by_phase=by_phase, elapsed_seconds=elapsed(intent),
                matrix_accounting=spec['matrix_accounting'],
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                initial_restoration_policy='exact parameter copies; no optimizer or buffer restoration',
                initial_restored_exactly=initial_restored,
                counter_semantics=dict(encoder_calls='returned forward hooks', explicit_bmm_calls='entered calls',
                                       backward_calls='entered calls', gradient_clip_calls='entered calls',
                                       optimizer_steps='separate started and completed counters',
                                       interrupted_work='last durable entry records upper bounds; never infer completion from entry'),
                judge_files_opened=[], m_c_supervision_authorized=False, me_identified=False), emergency=True)
        except BaseException:
            traceback.print_exc()
            raise  # The parent still owns the attempt and records this nonzero exit.
    if status != 'complete':
        raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'fit'))
    args = parser.parse_args()
    intent = read_json(OUT / 'fit_v0_attempt.json')
    verify_sources(intent['source_sha256'])
    check_time(intent)
    if args.stage == 'generate':
        generate(intent)
    else:
        fit(intent)


if __name__ == '__main__':
    main()
