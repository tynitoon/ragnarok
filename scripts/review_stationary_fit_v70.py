"""Independent arithmetic review of the CLOSED stationary fit; never run a model.

Prepared prospectively while the fit was active. Do not invoke until the parent
announces that the result AND console are closed and committed. The two required
CLI attestations bind a particular result and closing commit, not an active run.
This review accepts a complete execution, regardless of its scientific verdict.
An incomplete execution requires a separate explicit audit; it is never silently
treated as a scientific failure or as a validated instrument by this program.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'craft_v6_out/v70_stationary_fit'
CONTROL = 'craft_v6_out/v70_stationary_control/'
PROTOCOL = 'docs/ARC4_STATIONARY_FIT_PROTOCOL.json'
SOURCE_COMMIT = '0e5ede9'
PREFLIGHT_COMMIT = 'c04a9c5'
LAUNCH_COMMIT = 'ee6f32c'
CONTROL_RESULT_SHA = '88a7fcd8124a40ef0d13bd9112713e3a456ba1a78737eaa14e1786bcd69d0202'
CONTROL_REVIEW_SHA = 'dc121b32ed97628e3775ae28f37eb734680c46d8038b725747b58bf7a83cb69b'
OLD_INITIAL = 'craft_v6_out/v70_response_envelope_fit/initial_weights.pt'
OLD_INITIAL_SHA = '0964241c0b88e01532b9876c49f5999f86c4f74a8b5d786825f135bc5197c2e4'
OLD_FIT = 'craft_v6_out/v70_response_envelope_fit/fit_v0.json'
OLD_FIT_SHA = '9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a'
OLD_REVIEW = 'craft_v6_out/v70_response_envelope_fit/fit_review_v0.json'
OLD_REVIEW_SHA = '3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467'
MAX_REVIEW_SECONDS = 300
# Numerical audit tolerances, not changes to any scientific gate or threshold.
METRIC_ATOL = 1e-12
BCE_ATOL = 1e-6
BCE_RTOL = 1e-6
SIGMOID_ATOL = 1e-7
COUNTS = [1, 4, 16, 64, 128, 256, 0, 0, 336, 32, 8, 2]
NEW_SOURCES = {
    'scripts/fit_stationary_control_v70.py', 'tests/test_stationary_fit_v70.py',
    'docs/ARC4_STATIONARY_FIT_DRAFT.md', PROTOCOL,
}
np = None
torch = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_exclusive(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class NoExperimentImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ale_py', 'ragnarok', 'scripts'}:
            raise RuntimeError('Independent reviewer forbids experiment imports: ' + fullname)
        return None


def forbidden(*args, **kwargs):
    raise RuntimeError('No construction, model forward or optimizer call is allowed in this review')


def configure_review_runtime():
    global np, torch
    for key, value in dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONDONTWRITEBYTECODE='1').items():
        os.environ[key] = value
    require(not any(name.split('.')[0] in {'ale_py', 'ragnarok', 'scripts'} for name in sys.modules),
            'A fresh process without experiment or emulator imports is required')
    sys.meta_path.insert(0, NoExperimentImports())
    import numpy
    import torch as torch_module
    np, torch = numpy, torch_module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.nn.Module.__init__ = forbidden
    torch.nn.Module.__call__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    torch.optim.Optimizer.step = forbidden
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   torch=torch.__version__, threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(),
                   deterministic=torch.are_deterministic_algorithms_enabled(),
                   default_dtype=str(torch.get_default_dtype()),
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    require(Path(sys.executable).resolve() == Path('C:/Python314/python.exe').resolve(), 'Wrong Python executable')
    require(runtime['version'] == '3.14.3' and runtime['numpy'] == '2.4.4'
            and runtime['torch'] == '2.11.0+cu126' and runtime['default_dtype'] == 'torch.float32',
            'Wrong numerical runtime')
    return runtime


def array_signature(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha_bytes(value.tobytes(order='C')))


def average_precision_independent(scores, target):
    """A histogram over distinct scores; no sorted-pixel endpoint implementation."""
    total_positive = int(np.count_nonzero(target))
    if not total_positive:
        return None
    _, groups, totals = np.unique(scores.reshape(-1), return_inverse=True, return_counts=True)
    positive = np.bincount(groups, weights=target.reshape(-1).astype(np.int64), minlength=len(totals))[::-1]
    precision = np.cumsum(positive) / np.cumsum(totals[::-1])
    return float(np.dot(positive, precision) / total_positive)


def metrics_independent(probabilities, targets, cases):
    rows, paired = [], {}
    for scores, target, case in zip(probabilities, targets, cases, strict=True):
        predicted = scores >= .5
        actual = int(np.count_nonzero(target))
        tp = int(np.count_nonzero(predicted[target]))
        fp = int(np.count_nonzero(predicted[~target]))
        negatives = target.size - actual
        row = dict(game=case['game'], anchor_index=case['anchor_index'], target_pixels=actual,
                   true_positive_pixels=tp, false_positive_pixels=fp,
                   iou=tp / (actual + fp) if actual else None,
                   average_precision=average_precision_independent(scores, target),
                   false_positive_fraction=fp / negatives if negatives else None)
        rows.append(row)
        if actual:
            paired.setdefault(case['game'], []).append(row)
    means = {key: {metric: math.fsum(row[metric] for row in items) / len(items)
                   for metric in ('iou', 'average_precision')} for key, items in paired.items()}
    return dict(cases=rows, nonempty_by_game=means,
                mean_nonempty_iou=math.fsum(row['iou'] for row in means.values()) / len(means),
                mean_nonempty_average_precision=math.fsum(row['average_precision'] for row in means.values()) / len(means),
                empty_anchor_false_positive_fractions=[row['false_positive_fraction'] for row in rows if not row['target_pixels']])


def bce_independent(logits, targets):
    """Float64 stable softplus with one-class and equal-class anchor handling."""
    per_anchor = []
    for logit, target in zip(logits, targets, strict=True):
        losses = np.logaddexp(0., np.where(target, -logit.astype(np.float64), logit.astype(np.float64)))
        classes = [float(losses[mask].mean()) for mask in (target, ~target) if mask.any()]
        per_anchor.append(math.fsum(classes) / len(classes))
    return math.fsum(per_anchor) / len(per_anchor), per_anchor


def same_metrics(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path + ': keys differ')
        for key in expected:
            same_metrics(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ': length differs')
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            same_metrics(a, b, path + '[' + str(index) + ']')
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and abs(actual - expected) <= METRIC_ATOL, path + ': numerical mismatch')
    else:
        require(actual == expected and type(actual) is type(expected), path + ': mismatch')


def verdict_independent(metrics):
    visual = metrics['visual']
    advantage = visual['mean_nonempty_iou'] - max(metrics[key]['mean_nonempty_iou'] for key in ('initial', 'constant', 'zero'))
    absolute = visual['mean_nonempty_iou'] >= .25
    empty = visual['empty_anchor_false_positive_fractions']
    specificity = len(empty) == 2 and all(value <= .001 for value in empty)
    return dict(passed=absolute and advantage >= .10 and specificity, complete=True,
                minimum_iou_pass=absolute, advantage_over_best_control=advantage,
                minimum_advantage_pass=advantage >= .10, each_empty_anchor_pass=specificity,
                claim='Development-set fitting feasibility only; not independent generalization')


def validate_journal(events, result):
    """Exact event grammar plus every recorded forward entry/return snapshot."""
    require(len(events) == 2055, 'Complete execution requires exactly 2055 journal events')
    elapsed = [row['elapsed_seconds'] for row in events]
    require(all(math.isfinite(t) and 0 <= t < result['work_seconds'] for t in elapsed)
            and elapsed == sorted(elapsed), 'Journal chronology or final charged duration differs')
    cursor = 0
    forward = dict(training=dict(visual=0, constant=0), evaluation=dict(visual=0, constant=0))

    def take(name, **fields):
        nonlocal cursor
        event = events[cursor]
        cursor += 1
        require(event['event'] == name and all(event.get(key) == value for key, value in fields.items()),
                'Journal event mismatch at ' + str(cursor - 1))
        return event

    def snapshot(event):
        require(event['forward_entries'] == event['forward_returns'] == forward,
                'Forward entry/return snapshot mismatch at ' + str(cursor - 1))

    take('evaluation_started', arm='visual', checkpoint='initial')
    forward['evaluation']['visual'] += 1
    snapshot(take('evaluation_complete', arm='visual', checkpoint='initial'))
    for step in range(1, 257):
        for arm in ('visual', 'constant'):
            take('update_started', arm=arm, update=step)
            take('optimizer_step_enter', arm=arm, update=step)
            forward['training'][arm] += 1
            snapshot(take('optimizer_step_return', arm=arm, update=step))
            row = take('update_complete', arm=arm, update=step)
            require(math.isfinite(row['loss']) and row['loss'] >= 0
                    and math.isfinite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0,
                    'Nonfinite or negative training scalar')
    for arm in ('visual', 'constant'):
        take('evaluation_started', arm=arm, checkpoint='final')
        forward['evaluation'][arm] += 1
        snapshot(take('evaluation_complete', arm=arm, checkpoint='final'))
    final = take('finished', completed_updates=dict(visual=256, constant=256),
                 step_entries=dict(visual=256, constant=256), pending_optimizer_arm=None, failures=[])
    snapshot(final)
    require(cursor == len(events) and result['model_forward_entries'] == result['model_forward_returns'] == forward,
            'Final forward totals differ')
    return dict(events=len(events), event_counts=dict(Counter(row['event'] for row in events)),
                optimizer_entries=512, optimizer_returns=512, update_completions=512,
                training_forward_entries=512, training_forward_returns=512,
                evaluation_forward_entries=3, evaluation_forward_returns=3,
                last_event_seconds=elapsed[-1], charged_after_last_event_seconds=result['work_seconds'] - elapsed[-1])


class Review:
    def __init__(self, result_sha, closed_commit, prior=None):
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        self.result_sha, self.closed_commit = result_sha, closed_commit
        self.evidence, self.sources, self.checks = {}, {}, {}
        self.decoded_arrays, self.checkpoints = [], []
        if prior is not None:
            prior_rel = prior.resolve().relative_to(ROOT).as_posix()
            record = self.json_file(prior_rel)
            require(record['kind'] == 'stationary_fit_independent_result_review' and record['passed'] is False
                    and record['result_sha256'] == result_sha and record['decoded_arrays'] == []
                    and record['decoded_checkpoints'] == [], 'Only an explicitly authorized pre-array review failure can be continued')
            seconds, cpu = record['review_seconds'], record['review_main_process_cpu_seconds']
            require(math.isfinite(seconds) and 0 <= seconds < MAX_REVIEW_SECONDS
                    and math.isfinite(cpu) and 0 <= cpu <= seconds, 'Invalid prior review costs')
            self.started -= seconds
            self.cpu_started -= cpu
            prior_attempt = prior.with_name('fit_review_v0_attempt.json').resolve().relative_to(ROOT).as_posix()
            self.evidence_file(prior_attempt, record['review_attempt_sha256'])
            self.checks['prior_review'] = dict(receipt=prior_rel, seconds=seconds, main_process_cpu_seconds=cpu,
                                             decoded_arrays=0, decoded_checkpoints=0,
                                             failures=record['failures'], cost_included_in_current_ceiling=True)

    def time_check(self):
        require(time.monotonic() - self.started < MAX_REVIEW_SECONDS, 'Review conservative wall/CPU ceiling reached')

    def git(self, *args, input_text=None):
        self.time_check()
        return subprocess.check_output(['git', '-C', str(ROOT), *args], input=input_text, text=True,
                                       timeout=max(.1, MAX_REVIEW_SECONDS - (time.monotonic() - self.started))).strip()

    def safe_path(self, rel):
        path = ROOT / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts
                and path.resolve().is_relative_to(ROOT.resolve()), 'Path escapes repository')
        return path

    def evidence_file(self, rel, expected=None):
        self.time_check()
        path = self.safe_path(rel)
        digest = sha_file(path)
        require(expected is None or digest == expected, 'Evidence hash mismatch: ' + rel)
        require(rel not in self.evidence or self.evidence[rel] == digest, 'Evidence changed since first read: ' + rel)
        self.evidence[rel] = digest
        return path

    def json_file(self, rel, expected=None):
        path = self.evidence_file(rel, expected)
        data = path.read_bytes()
        require(sha_bytes(data) == self.evidence[rel], 'Evidence changed during read: ' + rel)
        return json.loads(data)

    def committed(self, mapping, ref='HEAD'):
        paths = list(mapping)
        # Repository-relative paths preserve Git's per-path CRLF/attributes
        # conversion. Absolute Windows paths bypass it in --stdin-paths.
        actual = self.git('hash-object', '--stdin-paths', input_text='\n'.join(paths) + '\n').splitlines()
        expected = self.git('rev-parse', *[ref + ':' + rel for rel in paths]).splitlines()
        require(actual == expected and len(actual) == len(paths), 'Git blob mismatch against ' + ref)

    def array(self, rel, key, signature):
        path = self.evidence_file(rel)
        with np.load(path, allow_pickle=False) as archive:
            require(archive.files == [key], 'Unexpected prediction archive members: ' + rel)
            value = archive[key]
        self.decoded_arrays.append(dict(path=rel, key=key))
        require(array_signature(value) == signature, 'Array signature mismatch: ' + rel)
        require(value.shape == (12, 210, 160) and value.dtype == np.float32 and np.isfinite(value).all(),
                'Invalid saved float32 array: ' + rel)
        self.time_check()
        return value

    def state(self, rel):
        path = self.evidence_file(rel)
        value = torch.load(path, map_location='cpu', weights_only=True)
        self.checkpoints.append(rel)
        self.time_check()
        return value

    def weights(self, initial, final, recipe):
        shapes = {}
        for index in (0, 2, 4, 6, 8):
            shapes['body.' + str(index) + '.weight'] = (16, 6 if index == 0 else 16, 3, 3)
            shapes['body.' + str(index) + '.bias'] = (16,)
        shapes.update({'head.weight': (1, 16, 1, 1), 'head.bias': (1,)})
        require(set(final) == {'visual', 'constant'}, 'Final checkpoint must retain both arms')
        for state in (initial, recipe, final['visual'], final['constant']):
            require(set(state) == set(shapes), 'State keys differ from the fixed architecture')
            require(sum(value.numel() for value in state.values()) == 10177, 'Parameter population differs')
            for key, shape in shapes.items():
                value = state[key]
                require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                        and value.dtype == torch.float32 and value.device.type == 'cpu'
                        and np.isfinite(value.numpy()).all(), 'Invalid archived tensor: ' + key)
        require(all(np.array_equal(initial[key].numpy(), recipe[key].numpy()) for key in shapes),
                'Initial weights differ from the independently reviewed seed-764 recipe archive')
        return dict(parameters_per_arm=10177, state_tensors_per_arm=len(shapes),
                    initial_recipe_bitwise_equal=True, recipe_archive_sha256=OLD_INITIAL_SHA,
                    changed_final_elements={arm: sum(int(np.count_nonzero(initial[key].numpy() != state[key].numpy()))
                                                     for key in shapes) for arm, state in final.items()},
                    initial_constant_equality_evidence='Frozen deepcopy and equality assertion; only one initial state was archived',
                    update_replay_performed=False)

    def perform(self):
        prefix = FOLDER.relative_to(ROOT).as_posix() + '/'
        result_rel = prefix + 'fit_v0.json'
        result = self.json_file(result_rel, self.result_sha)
        require(result['kind'] == 'stationary_control_development_fit', 'Wrong result kind')
        self.committed({result_rel: self.result_sha}, self.closed_commit)
        for previous, later in ((SOURCE_COMMIT, PREFLIGHT_COMMIT), (PREFLIGHT_COMMIT, LAUNCH_COMMIT),
                                (LAUNCH_COMMIT, self.closed_commit)):
            self.git('merge-base', '--is-ancestor', previous, later)
        runtime = configure_review_runtime()
        self.checks['runtime'] = runtime
        require(runtime == result['runtime'], 'Review and recorded fit runtime differ')
        protocol = self.json_file(PROTOCOL)
        require(protocol == result['protocol'] and protocol['seed'] == 764 and protocol['updates_per_arm'] == 256
                and protocol['probability_threshold'] == .5 and protocol['minimum_visual_iou'] == .25
                and protocol['minimum_advantage'] == .1 and protocol['maximum_each_empty_false_positive_fraction'] == .001
                and protocol['max_work_seconds'] == 1200 and protocol['retry'] is False,
                'The fixed prospective protocol differs')
        self.sources = result['source_sha256']
        control = self.json_file(CONTROL + 'control_v0.json', CONTROL_RESULT_SHA)
        control_review = self.json_file(CONTROL + 'control_review_v0.json', CONTROL_REVIEW_SHA)
        require(control_review['kind'] == 'stationary_control_independent_result_review'
                and control_review['passed'] is True and control_review['result_sha256'] == CONTROL_RESULT_SHA
                and control_review['source_sha256'] == control['source_sha256'], 'Control review provenance differs')
        require(len(self.sources) == 106 and set(self.sources) == set(control['source_sha256']) | NEW_SOURCES
                and all(self.sources[key] == value for key, value in control['source_sha256'].items()),
                'Exact 102 inherited plus four new sources required')
        for rel, digest in self.sources.items():
            require(sha_file(self.safe_path(rel)) == digest, 'Source changed: ' + rel)
        self.committed(self.sources, SOURCE_COMMIT)
        evidence = result['evidence_sha256']
        require(evidence == {CONTROL + 'control_v0.json': CONTROL_RESULT_SHA,
                             CONTROL + 'control_review_v0.json': CONTROL_REVIEW_SHA,
                             CONTROL + 'control_v0_dataset.npz': control['artifacts']['control_v0_dataset.npz']['sha256']},
                'Exactly three fixed control proofs required')
        for rel, digest in evidence.items():
            self.evidence_file(rel, digest)
        preflight = self.json_file(prefix + 'preflight_v0.json', result['preflight_sha256'])
        require(preflight['kind'] == 'stationary_fit_independent_preflight' and preflight['passed'] is True
                and all(preflight[key] == result[key] for key in ('source_sha256', 'evidence_sha256', 'runtime')),
                'Preflight differs from the actual result closure')
        self.committed({prefix + 'preflight_v0.json': result['preflight_sha256']}, PREFLIGHT_COMMIT)
        attempt = self.json_file(prefix + 'fit_v0_attempt.json', result['attempt_sha256'])
        require(attempt['kind'] == 'stationary_fit_attempt'
                and all(attempt[key] == result[key] for key in ('protocol', 'preflight_sha256', 'source_sha256', 'evidence_sha256', 'runtime')),
                'Attempt does not bind the preflight and protocol')
        payload = self.json_file(prefix + 'fit_v0_payload.json', result['scientific_payload_sha256'])
        require(payload['kind'] == 'stationary_fit_scientific_payload', 'Wrong scientific payload kind')
        require(all(result.get(key) == value for key, value in payload.items() if key not in {'kind', 'claim'}),
                'Final receipt changed a payload field')
        require(result['execution_status'] == 'complete' and result['candidate_execution_complete'] is True
                and result['failures'] == [] and result['final_provenance_verified'] is True,
                'Execution incomplete or failed; this complete-run review cannot issue an instrument PASS')
        require(result['optimizer_updates'] == result['optimizer_updates_upper_bound'] == 512
                and result['completed_updates'] == result['optimizer_step_entries'] == dict(visual=256, constant=256)
                and result['pending_optimizer_arm'] is None and result['optimizer_update_count_exact'] is True
                and result['model_forward_count_exact'] is True, 'Optimizer or forward accounting is not exact')
        require(result['inherited_synthetic_steps'] == 216 and result['new_synthetic_steps'] == 0
                and result['new_initial_screens'] == 0 and result['new_emulated_frames'] == 0 and result['gpu_used'] is False,
                'Unexpected environment or GPU work')
        require(0 < result['work_seconds'] < 1200, 'Final charged fit duration violates the fixed ceiling')
        progress_path = self.evidence_file(prefix + 'fit_v0_progress.jsonl', result['progress_sha256'])
        progress_bytes = progress_path.read_bytes()
        require(progress_bytes.endswith(b'\n') and sha_bytes(progress_bytes) == result['progress_sha256'],
                'Incomplete or changed journal bytes')
        self.checks['journal'] = validate_journal([json.loads(line) for line in progress_bytes.splitlines()], result)
        expected_artifacts = {'initial_weights.pt', 'final_weights.pt'} | {
            arm + suffix for arm in ('initial', 'visual', 'constant') for suffix in ('_logits.npz', '_predictions.npz')}
        require(set(result['artifacts']) == expected_artifacts, 'Unexpected missing or additional binary artefacts')
        for name, row in result['artifacts'].items():
            path = self.evidence_file(prefix + name, row['sha256'])
            require(path.stat().st_size == row['bytes'], 'Binary size mismatch: ' + name)
        binary_bytes = sum(row['bytes'] for row in result['artifacts'].values())
        require(binary_bytes <= 64000000, 'Binary ceiling exceeded')
        self.checks['binary_bytes'] = binary_bytes
        self.checks['launch'] = self.launch(prefix, result, attempt)
        cases = result['cases']
        require(cases == control['cases'] and [(row['game'], row['anchor_index']) for row in cases]
                == [('synthetic_pair_' + str(i // 2), i % 2) for i in range(12)], 'Case order changed')
        data_rel = CONTROL + 'control_v0_dataset.npz'
        with np.load(self.safe_path(data_rel), allow_pickle=False) as archive:
            require(set(archive.files) == {'past_pairs', 'noop_futures', 'action_futures', 'observed_response_envelopes'},
                    'Control archive members differ')
            targets = archive['observed_response_envelopes']
        self.decoded_arrays.append(dict(path=data_rel, key='observed_response_envelopes'))
        require(targets.shape == (12, 210, 160) and targets.dtype == bool
                and targets.sum(axis=(1, 2)).tolist() == COUNTS
                and array_signature(targets) == control['artifacts']['control_v0_dataset.npz']['arrays']['observed_response_envelopes'],
                'Target signature, order or counts differ')
        metrics, bces, sigmoid_errors = {}, {}, {}
        for arm in ('initial', 'visual', 'constant'):
            pred_name, logit_name = arm + '_predictions.npz', arm + '_logits.npz'
            probability = self.array(prefix + pred_name, 'probability', result['artifacts'][pred_name]['arrays']['probability'])
            logits = self.array(prefix + logit_name, 'logits', result['artifacts'][logit_name]['arrays']['logits'])
            require(np.all((probability >= 0) & (probability <= 1)), 'Probability out of range')
            sigmoid = np.exp(-np.logaddexp(0., -logits.astype(np.float64)))
            sigmoid_errors[arm] = float(np.max(np.abs(sigmoid - probability)))
            require(sigmoid_errors[arm] <= SIGMOID_ATOL, 'Saved probability is inconsistent with saved logits')
            metrics[arm] = metrics_independent(probability, targets, cases)
            recorded = {key: value for key, value in result['metrics'][arm].items() if key != 'balanced_bce'}
            same_metrics(recorded, metrics[arm], arm)
            bce, per_anchor = bce_independent(logits, targets)
            reported_bce = result['metrics'][arm]['balanced_bce']
            require(math.isfinite(reported_bce) and abs(reported_bce - bce) <= BCE_ATOL + BCE_RTOL * abs(bce),
                    'Balanced BCE mismatch beyond the prospectively fixed audit tolerance')
            bces[arm] = dict(independent_float64=bce, recorded_float32=reported_bce,
                            absolute_error=abs(reported_bce - bce), per_anchor_float64=per_anchor)
            if arm == 'constant':
                require(np.all(logits == logits.flat[0]) and np.all(probability == probability.flat[0]),
                        'The constant-input FCN control is not constant across every pixel and case')
            self.time_check()
        metrics['zero'] = metrics_independent(np.zeros(targets.shape, dtype=np.float32), targets, cases)
        same_metrics(result['metrics']['zero'], metrics['zero'], 'zero')
        require(set(result['metrics']) == set(metrics), 'All four controls and exactly 48 metric rows required')
        verdict = verdict_independent(metrics)
        same_metrics(result['candidate_verdict'], verdict, 'candidate_verdict')
        same_metrics(result['verdict'], verdict, 'verdict')
        require(result['passed'] is verdict['passed'], 'Published scientific flag differs')
        self.checks['metrics'] = dict(rows_recomputed=48, recomputed=metrics, balanced_bce=bces,
                                     sigmoid_max_abs_errors=sigmoid_errors, verdict=verdict,
                                     threshold=.5, metric_atol=METRIC_ATOL, bce_atol=BCE_ATOL, bce_rtol=BCE_RTOL,
                                     sigmoid_atol=SIGMOID_ATOL,
                                     zero_bce=None, zero_bce_reason='Not reported by fit; exact probability zero has infinite BCE on positive targets')
        old_fit = self.json_file(OLD_FIT, OLD_FIT_SHA)
        old_review = self.json_file(OLD_REVIEW, OLD_REVIEW_SHA)
        require(old_review['kind'] == 'response_envelope_fit_independent_result_review'
                and old_review['passed'] is True and old_review['result_sha256'] == OLD_FIT_SHA
                and old_review['source_sha256'] == old_fit['source_sha256']
                and old_fit['artifacts']['initial_weights.pt']['sha256'] == OLD_INITIAL_SHA
                and old_fit['protocol']['seed'] == 764
                and old_fit['protocol']['architecture'] == protocol['architecture'], 'Archived random recipe provenance differs')
        self.evidence_file(OLD_INITIAL, OLD_INITIAL_SHA)
        self.checks['weights'] = self.weights(self.state(prefix + 'initial_weights.pt'),
                                              self.state(prefix + 'final_weights.pt'), self.state(OLD_INITIAL))
        self.checks['timing'] = dict(fit_work_seconds=result['work_seconds'],
                                    payload_hash_verified_before_final_receipt=True,
                                    duration_boundary='Frozen finish_result serializes, fsyncs and hashes payload before measuring; OS flush itself is not independently replayed')
        reviewer_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
        self.evidence_file(reviewer_rel)
        self.committed(self.evidence)
        self.committed(self.sources)
        for rel, digest in {**self.sources, **self.evidence}.items():
            self.time_check()
            require(sha_file(self.safe_path(rel)) == digest, 'Source or evidence changed during review: ' + rel)
        self.time_check()
        return result

    def launch(self, prefix, result, attempt):
        """Check committed WMI receipts; never query or interact with a live PID."""
        intent_rel, dispatch_rel, console_rel = [prefix + 'launch_v0/' + name for name in ('intent.json', 'dispatch.json', 'console.log')]
        intent = self.json_file(intent_rel)
        dispatch = self.json_file(dispatch_rel)
        console_path = self.evidence_file(console_rel)
        console = [json.loads(line) for line in console_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        require(len(console) == 2, 'Exactly one launch record and one final console record required')
        require(intent['kind'] == 'stationary_fit_wmi_launch_intent' and dispatch['kind'] == 'stationary_fit_wmi_dispatch',
                'Wrong WMI receipt kind')
        require(dispatch['return_value'] == 0 and dispatch['intent_sha256'] == self.evidence[intent_rel]
                and dispatch['preflight_sha256'] == result['preflight_sha256'] and dispatch['show_window'] == 0,
                'WMI dispatch failed or intent changed')
        require(console[0]['pid'] == dispatch['pid'] and console[0]['detached_from_job'] is True
                and console[0]['mode'] == 'fixed_stationary_control_fit',
                'Detached worker identity mismatch')
        require(intent['preflight_sha256'] == result['preflight_sha256']
                and intent['show_window'] == 0 and intent['gpu_allowed'] is False
                and intent['maximum_scientific_work_seconds'] == 1200 and intent['updates_per_arm'] == 256
                and intent['new_attempt_only'] is True and intent['module'] == 'scripts.fit_stationary_control_v70'
                and intent['console'] == console_rel and Path(intent['cwd']).resolve() == ROOT.resolve()
                and 'C:\\Python314\\python.exe' in intent['command_line']
                and "runpy.run_module('scripts.fit_stationary_control_v70',run_name='__main__')" in intent['command_line'],
                'WMI launch configuration differs')
        require(self.git('rev-parse', intent['source_commit']) == self.git('rev-parse', SOURCE_COMMIT)
                and self.git('rev-parse', intent['independent_preflight_commit']) == self.git('rev-parse', PREFLIGHT_COMMIT),
                'Launch names a different source/preflight commit')
        for key in ('execution_status', 'verdict', 'completed_updates', 'optimizer_update_count_exact', 'work_seconds'):
            require(console[-1][key] == result[key], 'Closed console does not match result: ' + key)
        created = datetime.fromisoformat(intent['created_utc'].replace('Z', '+00:00'))
        started = datetime.fromisoformat(attempt['created_utc'].replace('Z', '+00:00'))
        require(created <= started, 'Attempt predates launch intent')
        # Dispatch observation can legitimately occur after the worker attempt starts.
        observed = datetime.fromisoformat(dispatch['created_utc'].replace('Z', '+00:00'))
        require(created <= observed, 'Dispatch observation predates its intent')
        self.committed({intent_rel: self.evidence[intent_rel], dispatch_rel: self.evidence[dispatch_rel]}, LAUNCH_COMMIT)
        self.committed({console_rel: self.evidence[console_rel]}, self.closed_commit)
        return dict(pid=dispatch['pid'], intent_utc=created.isoformat(), attempt_utc=started.isoformat(),
                    dispatch_observed_utc=observed.isoformat(), detached_from_job=True,
                    final_console_equals_result=True, closed_commit=self.closed_commit,
                    live_process_queries=0, os_flush_or_process_exit_replayed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-closed-result', action='store_true', required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--closed-commit', required=True, help='Commit containing the immutable result AND final console')
    parser.add_argument('--prior-review-receipt', type=Path,
                        help='Explicitly authorized continuation after a preserved pre-array review failure; its costs remain charged')
    parser.add_argument('--review-version', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    require(len(args.result_sha256) == 64 and all(c in '0123456789abcdef' for c in args.result_sha256), 'An exact lowercase result SHA256 is required')
    require((args.review_version == 0) == (args.prior_review_receipt is None),
            'Version one requires the preserved version-zero receipt; version zero cannot hide a prior review')
    if args.review_version == 1:
        require(args.prior_review_receipt.resolve() == (FOLDER / 'fit_review_v0.json').resolve(),
                'The authorized continuation must retain the exact version-zero receipt path')
    output = FOLDER / ('fit_review_v' + str(args.review_version) + '.json')
    attempt_path = FOLDER / ('fit_review_v' + str(args.review_version) + '_attempt.json')
    require(not output.exists() and not attempt_path.exists(), 'Preserve prior review traces; no automatic retry')
    require(sha_file(FOLDER / 'fit_v0.json') == args.result_sha256, 'Closed result hash does not match authorization')
    review = Review(args.result_sha256, args.closed_commit, args.prior_review_receipt)
    script_sha = sha_file(Path(__file__))
    write_exclusive(attempt_path, dict(kind='stationary_fit_independent_review_attempt',
                                      created_utc=datetime.now(timezone.utc).isoformat(),
                                      result_sha256=args.result_sha256, closed_commit=args.closed_commit,
                                      reviewer_sha256=script_sha, max_review_seconds=MAX_REVIEW_SECONDS,
                                      review_version=args.review_version, prior_review=review.checks.get('prior_review'),
                                      scope='Saved targets, predictions, logits and archived weights only; no simulation or model execution'))
    failures = []
    try:
        review.perform()
    except BaseException as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    elapsed, cpu = time.monotonic() - review.started, time.process_time() - review.cpu_started
    if elapsed >= MAX_REVIEW_SECONDS:
        failures.append(dict(type='TimeoutError', message='Review conservative wall/CPU ceiling exceeded'))
    receipt = dict(kind='stationary_fit_independent_result_review', passed=not failures,
                   created_utc=datetime.now(timezone.utc).isoformat(), result_sha256=args.result_sha256,
                   review_version=args.review_version,
                   source_sha256=review.sources, evidence_sha256=review.evidence,
                   review_attempt_sha256=sha_file(attempt_path), checks=review.checks, failures=failures,
                   decoded_arrays=review.decoded_arrays, decoded_checkpoints=review.checkpoints,
                   review_seconds=elapsed, review_main_process_cpu_seconds=cpu,
                   budget=dict(max_seconds=MAX_REVIEW_SECONDS, cpu_threads=1,
                               scope='All verification through final rehash; final receipt and console excluded',
                               accounting='Wall ceiling also bounds serial subprocess CPU; main-process CPU reported separately'),
                   new_model_constructions=0, new_model_forwards=0, new_optimizer_updates=0,
                   new_synthetic_steps=0, new_initial_screens=0, new_emulated_frames=0, gpu_used=False,
                   limits=[
                       'Instrument validity is separate from the frozen scientific criterion.',
                       'All twelve cases are development; no independent validation, physical objects, mastery, transfer or acceleration is established.',
                       'Saved outputs and weights are checked without recomputing any network or optimizer trajectory.',
                       'Initial arm equality is supported by the frozen deepcopy/assertion; only one initial checkpoint exists.',
                       'The old random initialization archive is read only by this reviewer, never supplied to the learner.',
                       'Failed or unfinished checks cannot produce a positive instrument receipt.',
                       'No pixel arrays from future branches or Atari are decoded by this review.',
                   ])
    write_exclusive(output, receipt)
    print(json.dumps(dict(passed=receipt['passed'], failures=failures, review_seconds=elapsed,
                          result_sha256=args.result_sha256, receipt_sha256=sha_file(output)), allow_nan=False))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
