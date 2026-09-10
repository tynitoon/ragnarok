"""Prospective paid action/RGB survey. No learner and no implicit native run.

Only run() constructs the unchanged isolated transport, behind committed proofs
and an independent preflight. Tests replace that constructor with RGB fixtures.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS
from ragnarok.environments.isolated_rgb_pool_v70 import IsolatedRGBPool, IsolatedWorkerError
from ragnarok.environments.owned_rgb_collector_v70 import OwnedRGBCollector
from scripts.check_branch_signal_inventory_v70 import summarize, validate_arrays

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'docs/ARC4_ACTION_SIGNAL_SURVEY_PROTOCOL.json'
FOLDER = ROOT / 'craft_v6_out/v70_action_signal_survey'
PREFLIGHT = FOLDER / 'preflight_v0.json'
ATTEMPT = FOLDER / 'survey_v0_attempt.json'
PROGRESS = FOLDER / 'survey_v0_progress.jsonl'
OUTPUT = FOLDER / 'survey_v0.json'
ARCHIVES = FOLDER / 'rgb'
GAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids')
SHAPE = (210, 160, 3)
SPEC = dict(games=list(GAMES), emulator_seed=19390, command_seed=19391,
            anchors=[10000, 20000], commands=list(range(18)), control=0, horizon=8,
            frame_skip=4, ordinary_command_hold_decisions=4, sticky_probability=.25,
            max_episode_frames=108000, max_snapshots=64, experiment_fraction=.2,
            max_charged_frames_per_game=30000, max_charged_frames_total=180000,
            max_experiment_charge_per_pair=136, max_experiment_charge_total=29376,
            planned_pairs_per_game=36, planned_pairs_total=216,
            max_work_seconds=900, max_archive_bytes=1000000000, peer_timeout_seconds=30,
            cold_costs={g: 128 if g == 'seaquest' else 132 for g in GAMES},
            cpu_threads=1, model_inferences=0, optimizer_updates=0)
CHILD = dict(executable=r'F:\dev\ragnarok\.venv-arc4\Scripts\python.exe',
             python='3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]',
             numpy='2.4.4', ale_py='0.12.1',
             ale_binary_sha256='b7e29d7c2662be5a7d49d9e7151637a7842547ecc149b2b2c6771b9e02159d7c',
             numerical_threads={k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
             cuda_visible_devices='')
PREREQUISITES = (
    ('craft_v6_out/v70_isolated_rgb_native/wiring_review_v0.json',
     'e93d2932c0ad5973558f51729dc189c99523ec5efbc08fce11fd7a93c06ca287',
     'craft_v6_out/v70_isolated_rgb_native/wiring_v0.json',
     '81e366e2ec82dc9f3b888429a8c7c878216d067cfad3cec7b46e4b7de75951cb'),
    ('craft_v6_out/v70_owned_ordinary_native/wiring_review_v0.json',
     '8b4105ce479e7d990293a89ce8ea3789506600729f3e1f721dbf59653f04a668',
     'craft_v6_out/v70_owned_ordinary_native/wiring_v0.json',
     '5db50e4ee27fedf0ec77a62fab9c13efd309749f7a127e47d73ec4f56ae886af'),
    ('craft_v6_out/v70_branch_signal_inventory/inventory_review_v0.json',
     '4ca40b162fd77df1ded9eca6fc41f497c01c3b4fa409bbc8884d8053e0ed8b6d',
     'craft_v6_out/v70_branch_signal_inventory/inventory_v0.json',
     'a2403c7b07ef3389a8fb89e1524e2bbaa5b28f31199b38e434402aa8d3ed020e'))
NEW_SOURCES = ('scripts/survey_action_signal_v70.py', 'tests/test_action_signal_survey_v70.py',
               PROTOCOL, 'docs/ARC4_ACTION_SIGNAL_SURVEY_DRAFT.md')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def local_path(name):
    if not isinstance(name, str) or Path(name).is_absolute() or '\\' in name or '..' in Path(name).parts:
        raise ValueError('Canonical repository-relative path required')
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Proof path outside repository')
    return path


def require_committed(name):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={name}', str(local_path(name))) != git('rev-parse', f'HEAD:{name}'):
        raise ValueError('Source/evidence is not its committed blob: ' + name)


def source_closure(entries):
    """AST only, including parent initializers and explicit child entrypoint.

    Third-party imports are not executed. The child is launched by module-name
    string, so its entrypoint is an explicit seed rather than an inferred import.
    """
    pending, found = list(entries), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = local_path(name)
        if not path.is_file():
            raise ValueError('Missing local closure source: ' + name)
        found.add(name)
        for parent in Path(name).parents:
            init = (parent / '__init__.py').as_posix()
            if init != '__init__.py' and (ROOT / init).is_file() and init not in found:
                pending.append(init)
        module = name[:-3].replace('/', '.')
        package = module.rsplit('.', 1)[0] if not module.endswith('.__init__') else module[:-9]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    bits = package.split('.')
                    base = '.'.join(bits[:len(bits) - node.level + 1] + ([base] if base else []))
                candidates = [base] + [base + '.' + alias.name for alias in node.names if alias.name != '*']
            for candidate in candidates:
                if candidate.split('.')[0] not in ('ragnarok', 'scripts', 'tests'):
                    continue
                stem = candidate.replace('.', '/')
                for relative in (stem + '.py', stem + '/__init__.py'):
                    if (ROOT / relative).is_file() and relative not in found:
                        pending.append(relative)
    return sorted(found)


def runtime():
    return dict(executable=str(Path(sys.executable).resolve()), python=sys.version, numpy=np.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                numerical_threads={key: os.environ.get(key) for key in CHILD['numerical_threads']})


def proofs():
    """Real proof hashing is permitted only after the new files are committed.

    This never constructs a worker, calls an old proof function, or loads NPZ
    members. The prospective independent reviewer must call it before run().
    """
    for name in NEW_SOURCES:
        require_committed(name)
    protocol, current = read_json(ROOT / PROTOCOL), runtime()
    if (protocol.get('kind') != 'prospective_action_signal_survey_v0' or protocol.get('spec') != SPEC
            or protocol.get('parent_runtime') != current or protocol.get('child_runtime') != CHILD
            or protocol.get('agent_training_allowed') is not False or protocol.get('gate_result') is not None
            or current['cuda_visible_devices'] != ''
            or any(value != '1' for value in current['numerical_threads'].values())
            or any(key.split('.')[0] in ('torch', 'ale_py', 'tensorflow') for key in sys.modules)
            or SPEC['cold_costs'] != {game: ROM_SPECS[game][1] for game in GAMES}):
        raise ValueError('Exact frozen CPU/child runtime, population and cold costs required')
    sources, evidence = {}, {}

    def merge(destination, name, expected):
        if (not isinstance(expected, str) or len(expected) != 64 or sha(local_path(name)) != expected
                or name in sources and sources[name] != expected or name in evidence and evidence[name] != expected):
            raise ValueError('Proof mismatch/conflict: ' + name)
        destination[name] = expected

    for review_path, review_sha, result_path, result_sha in PREREQUISITES:
        merge(evidence, review_path, review_sha)
        review = read_json(local_path(review_path))
        if review.get('passed') is not True or review.get('result_sha256') != result_sha:
            raise ValueError('Exact successful independent historical review required')
        for field, destination in (('source_sha256', sources), ('evidence_sha256', evidence)):
            if not isinstance(review.get(field), dict):
                raise ValueError('Separate historical proof maps required')
            for name, value in review[field].items():
                merge(destination, name, value)
        if evidence.get(result_path) != result_sha:
            raise ValueError('Historical result absent from reviewed evidence')
        merge(evidence, result_path, result_sha)
        if read_json(local_path(result_path)).get('passed') is not True:
            raise ValueError('Historical result did not pass its execution contract')
    entries = [NEW_SOURCES[0], NEW_SOURCES[1], 'scripts/serve_ordinary_rgb_v70.py']
    closure = source_closure(entries)
    for name in (*closure, *NEW_SOURCES):
        merge(sources, name, sha(local_path(name)))
    for name in sorted(set(sources) | set(evidence)):
        require_committed(name)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=current,
                child_runtime=CHILD, local_source_closure=closure)


def preflight():
    proof = proofs()
    review = read_json(PREFLIGHT)
    if (review.get('kind') != 'action_signal_survey_independent_preflight' or review.get('passed') is not True
            or any(review.get(key) != value for key, value in proof.items())):
        raise ValueError('Exact independent committed preflight required before any worker')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof, preflight_review_sha256=sha(PREFLIGHT))


class ResourceCeiling(RuntimeError):
    """Prospectively declared global time/archive stop, with no retry."""


def command_stream():
    generator = random.Random(SPEC['command_seed'])
    while True:
        action = generator.randrange(18)
        for _ in range(SPEC['ordinary_command_hold_decisions']):
            yield action


def config(game):
    if game not in GAMES:
        raise ValueError('Only the six declared development games')
    return EmulatorConfig(game=game, seed=SPEC['emulator_seed'], frame_skip=SPEC['frame_skip'],
                          sticky_probability=SPEC['sticky_probability'], mode='lab',
                          experiment_fraction=SPEC['experiment_fraction'],
                          max_charged_frames=SPEC['max_charged_frames_per_game'],
                          max_snapshots=SPEC['max_snapshots'], max_episode_frames=SPEC['max_episode_frames'])


def healthy_account(report):
    account = report.get('accounting')
    if (not isinstance(account, dict) or report.get('failure') is not None
            or report.get('accounting_complete') is False or account.get('accounting_complete') is False):
        raise ValueError('Complete healthy external ledger required before another command')
    names = ('ordinary_frames', 'experiment_frames', 'restores', 'restore_charges', 'charged_frames')
    if (any(type(account.get(key)) is not int or account[key] < 0 for key in names)
            or account['restore_charges'] != 60 * account['restores']
            or account['charged_frames'] != account['ordinary_frames'] + account['experiment_frames'] + account['restore_charges']
            or account.get('max_charged_frames') != SPEC['max_charged_frames_per_game']
            or account.get('experiment_fraction') != SPEC['experiment_fraction']
            or account['charged_frames'] > SPEC['max_charged_frames_per_game']
            or 5 * (account['experiment_frames'] + account['restore_charges']) > account['charged_frames']):
        raise ValueError('External accounting differs from the frozen additive budget')
    return account


def admission(report, game, *, branch):
    account = healthy_account(report)
    reserve = 2 * SPEC['horizon'] + 120 if branch else SPEC['frame_skip'] + SPEC['cold_costs'][game]
    balance = SPEC['max_charged_frames_per_game'] - account['charged_frames']
    experiment = account['experiment_frames'] + account['restore_charges']
    quota_ok = not branch or 5 * (experiment + reserve) <= account['charged_frames'] + reserve
    return dict(admitted=balance >= reserve and quota_ok, remaining_charged_capacity=balance,
                reserved_maximum_charge=reserve, capacity_margin_after_reservation=balance - reserve,
                quota_admitted=quota_ok, reason=None if balance >= reserve and quota_ok else
                ('insufficient_maximum_reservation' if balance < reserve else 'experimental_quota_reservation'),
                actual_ceiling_exhaustion_claimed=False)


def remember(case, report, *, uncertain=False):
    report = copy.deepcopy(report)
    old = case.get('worker_report', {}).get('accounting')
    new = report.get('accounting')
    if old is not None and (new is None or any(new.get(key, -1) < old.get(key, 0) for key in
            ('charged_frames', 'ordinary_frames', 'experiment_frames', 'restores', 'restore_charges'))):
        report['accounting'] = old
        uncertain = True
    if (new is None or report.get('accounting_complete') is False
            or report.get('failure') is not None or new.get('accounting_complete') is False):
        uncertain = True
    case['accounting_uncertain'] |= uncertain
    if case['accounting_uncertain']:
        report['accounting_complete'] = False
    case['worker_report'] = report


def array_signature(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def same_array(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def branch_arrays(record):
    arrays = {'branch_start_pair': record.anchor.pair}
    for side in ('intervention', 'control'):
        for field in ('raw', 'pooled', 'endpoint'):
            arrays[f'branch_{side}_{field}'] = getattr(getattr(record, side), field)
    return arrays


def control_equal(arrays, baseline):
    return all(same_array(arrays[key], baseline[key]) for key in
               ('branch_start_pair', 'branch_control_raw', 'branch_control_pooled', 'branch_control_endpoint'))


class Archives:
    """Exclusive uncompressed RGB archives; pessimistic reservation before pay.

    NPY/ZIP metadata for seven arrays is bounded by a generous 64 KiB reserve.
    Actual bytes are checked too. A partial file is preserved on I/O failure.
    """
    def __init__(self):
        self.bytes = 0
        self.paths = []

    def reserve(self):
        images = 2 + 2 * (2 * SPEC['horizon'] + 1)
        maximum = images * int(np.prod(SHAPE)) + 65536
        if self.bytes + maximum > SPEC['max_archive_bytes']:
            raise ResourceCeiling('archive_storage_reservation')
        return maximum

    def save(self, name, arrays):
        maximum = self.reserve()
        ARCHIVES.mkdir(parents=True, exist_ok=True)
        path = ARCHIVES / name
        self.paths.append(path)
        created = False
        try:
            with path.open('xb') as stream:
                created = True
                # NumPy's ZIP factory recognizes a stream by its read method,
                # even for a write-only archive. No read is performed here.
                class Bounded:
                    def read(inner, *args): return stream.read(*args)
                    def write(inner, data):
                        if stream.tell() + len(data) > maximum:
                            raise ResourceCeiling('archive_actual_bytes_exceed_reservation')
                        return stream.write(data)
                    def tell(inner): return stream.tell()
                    def seek(inner, *args): return stream.seek(*args)
                    def flush(inner): return stream.flush()
                try:
                    np.savez(Bounded(), **arrays)
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())
        finally:
            if created:
                self.bytes += path.stat().st_size
        return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size,
                    arrays={key: array_signature(value) for key, value in arrays.items()})


def planned_cases():
    return [dict(game=game, started=False, execution_status='not_reached', stop_reason=None,
                 accounting_uncertain=False, worker_report={}, ordinary_decisions=0,
                 ordinary_delivered_frames=0, ordinary_trace_sha256=None,
                 anchors=[dict(target_delivered_frames=target, reached=False, delivered_frames=None,
                               overshoot=None, own_anchor=None, instrument_valid=None,
                               pairs=[dict(command=action, status='not_reached', reason='not_yet_reached',
                                           raw=None, pooled=None, archive=None) for action in SPEC['commands']])
                          for target in SPEC['anchors']]) for game in GAMES]


def finish_missing(case, reason):
    for anchor in case['anchors']:
        for row in anchor['pairs']:
            if row['status'] == 'not_reached':
                row['reason'] = reason


def error_metadata(error):
    return dict(error_type=type(error).__name__, error=str(error))


def run_case(case, archives, event, deadline):
    """One unchanged pool and collector, no next game after an instrument error."""
    pool = owner = None
    commands = command_stream()
    digest = hashlib.sha256()
    game = case['game']

    def report():
        value = pool.report(0)
        remember(case, value)
        healthy_account(case['worker_report'])
        return value

    def censor(check):
        case.update(execution_status='censored', stop_reason='budget_reservation', budget_stop=check)
        finish_missing(case, 'budget_reservation')
        event('game_censored', game=game, admission=check, delivered_frames=case['ordinary_delivered_frames'])

    try:
        deadline()
        case.update(started=True, execution_status='running')
        event('before_constructor', game=game, known_charged_frames=0, cost_if_interrupted='unknown')
        pool = IsolatedRGBPool([config(game)], python_executable=CHILD['executable'], expected_runtime=CHILD,
                               timeout=SPEC['peer_timeout_seconds'])
        report()
        owner = OwnedRGBCollector(pool, 1, session_id='action-signal-survey-v0/' + game, max_pending=1)
        if owner.current_anchor().pair.shape != (2, *SHAPE):
            raise ValueError('Native RGB geometry differs from the fixed survey')
        event('constructor_complete', game=game, report=case['worker_report'])
        deadline()
        for anchor_index, anchor_row in enumerate(case['anchors']):
            target = anchor_row['target_delivered_frames']
            while case['ordinary_delivered_frames'] < target:
                deadline()
                check = admission(case['worker_report'], game, branch=False)
                if not check['admitted']:
                    censor(check)
                    return
                action = next(commands)
                event('before_step', game=game, decision=case['ordinary_decisions'], command=action,
                      admission=check, known_report=case['worker_report'])
                operation = owner.step([action])
                record, = owner.take(operation)
                case['ordinary_decisions'] += 1
                case['ordinary_delivered_frames'] += len(record.trace.raw)
                for field in ('start_pair', 'raw', 'pooled', 'endpoint'):
                    value = getattr(record.trace, field)
                    digest.update(json.dumps([field, list(value.shape), str(value.dtype)], separators=(',', ':')).encode())
                    digest.update(value.tobytes(order='C'))
                case['ordinary_trace_sha256'] = digest.hexdigest()
                report()
                event('step_complete', game=game, operation_id=operation, command=action,
                      actual_frames=len(record.trace.raw), delivered_frames=case['ordinary_delivered_frames'],
                      epoch=record.chronology.epoch, epoch_frame=record.chronology.end_frame,
                      trace_sha256=case['ordinary_trace_sha256'], report=case['worker_report'])
                deadline()
            anchor = owner.current_anchor()
            anchor_row.update(reached=True, delivered_frames=case['ordinary_delivered_frames'],
                              overshoot=case['ordinary_delivered_frames'] - target,
                              own_anchor=dict(token=list(anchor.token), epoch=anchor.epoch, frame=anchor.frame,
                                              pair=array_signature(anchor.pair)), instrument_valid=None)
            if not 0 <= anchor_row['overshoot'] < SPEC['frame_skip']:
                raise ValueError('Cumulative delivered-image anchor overshoot exceeds repeat')
            baseline = None
            for row in anchor_row['pairs']:
                deadline()
                check = admission(case['worker_report'], game, branch=True)
                if not check['admitted']:
                    censor(check)
                    return
                archives.reserve()  # Refuse before paying if the full pair cannot be stored.
                row.update(status='started', reason=None)
                event('before_pair', game=game, target=target, command=row['command'],
                      anchor=anchor_row['own_anchor'], admission=check, known_report=case['worker_report'])
                operation = owner.branch(row['command'], expected_anchor=anchor.token, frames=SPEC['horizon'])
                record, = owner.take(operation)
                arrays = branch_arrays(record)
                # These RGB have already been paid and received. Persist them
                # before any report RPC or journal write can fail. Accounting
                # uncertainty must not erase an available visual observation.
                row['archive'] = archives.save(f'{game}_anchor{anchor_index}_action{row["command"]:02d}.npz', arrays)
                row.update(operation_id=list(operation), own_anchor_token=list(record.anchor.token),
                           requested_frames=record.requested_frames, control_action=record.control_action,
                           effect_assumption=record.effect_assumption.tolist())
                report()
                event('pair_received', game=game, target=target, command=row['command'], operation_id=operation,
                      lengths=[len(record.intervention.raw), len(record.control.raw)], report=case['worker_report'])
                lengths = validate_arrays(arrays, shape=SHAPE)
                row['lengths'] = lengths
                for kind in ('raw', 'pooled'):
                    row[kind] = summarize(arrays[f'branch_intervention_{kind}'], arrays[f'branch_control_{kind}'], shape=SHAPE)
                if (record.anchor.token != anchor.token or not same_array(record.anchor.pair, anchor.pair)
                        or record.requested_action != row['command'] or record.control_action != 0
                        or record.requested_frames != SPEC['horizon'] or np.any(record.effect_assumption)):
                    raise ValueError('Own command/anchor metadata changed')
                if baseline is None:
                    if row['command'] != 0 or not all(same_array(arrays[f'branch_intervention_{field}'],
                            arrays[f'branch_control_{field}']) for field in ('raw', 'pooled', 'endpoint')):
                        anchor_row['instrument_valid'] = False
                        raise ValueError('Paid NOOP/NOOP repetition differs')
                    baseline = arrays
                    anchor_row['instrument_valid'] = True
                elif not control_equal(arrays, baseline):
                    anchor_row['instrument_valid'] = False
                    raise ValueError('Repeated NOOP control or its anchor/prefix differs')
                row.update(status='acquired', control_consistent=True)
                event('pair_archived', game=game, target=target, command=row['command'], archive=row['archive'])
                deadline()
        case['execution_status'] = 'complete'
    except BaseException as error:
        case.update(execution_status='censored' if isinstance(error, ResourceCeiling) else 'failed',
                    stop_reason=str(error) if isinstance(error, ResourceCeiling) else 'instrument_failure',
                    failure=error_metadata(error))
        if isinstance(error, IsolatedWorkerError):
            remember(case, error.known_worker_reports.get(0, dict(accounting=None, failure=error_metadata(error))), uncertain=True)
        elif pool is None and case['started']:
            case['accounting_uncertain'] = True
        for anchor in case['anchors']:
            for row in anchor['pairs']:
                if row['status'] == 'started':
                    row.update(status='failed', reason=case['stop_reason'])
        finish_missing(case, case['stop_reason'])
        raise
    finally:
        if pool is not None:
            try:
                remember(case, pool.report(0))
            except BaseException as error:
                case['accounting_uncertain'] = True
                case['final_report_error'] = error_metadata(error)
            try:
                (owner if owner is not None else pool).close()
                case['transport_closed'] = True
            except BaseException as error:
                case['transport_closed'] = False
                case['close_error'] = error_metadata(error)
                raise
        if case['started']:
            event('game_closed', game=game, execution_status=case['execution_status'],
                  report=case['worker_report'], accounting_uncertain=case['accounting_uncertain'])


def accounting(cases):
    started = [case for case in cases if case['started']]
    uncertain = [case['game'] for case in started if case['accounting_uncertain']
                 or not isinstance(case['worker_report'].get('accounting'), dict)]
    known = sum(case['worker_report']['accounting']['charged_frames'] for case in started
                if isinstance(case['worker_report'].get('accounting'), dict))
    return dict(constructor_attempts=len(started), known_charged_frames=known,
                uncertain_cost_games=uncertain, accounting_complete=not uncertain,
                known_charge_is_lower_bound=bool(uncertain))


def run():
    if any(path.exists() for path in (ATTEMPT, PROGRESS, OUTPUT, ARCHIVES)):
        raise ValueError('One exclusive survey only; no retry or overwrite')
    proof = preflight()
    FOLDER.mkdir(parents=True, exist_ok=True)
    cases, archives = planned_cases(), Archives()
    attempt = dict(kind='action_signal_survey_attempt', status='started', started_utc=utc(),
                   **proof, cases=cases, current_game=None, current_phase='initializing')
    write_json(ATTEMPT, attempt, exclusive=True)
    began = time.monotonic()
    status, failure = 'complete', None
    sequence = 0

    def deadline():
        if time.monotonic() - began > SPEC['max_work_seconds']:
            raise ResourceCeiling('global_work_seconds')

    with PROGRESS.open('x', encoding='utf-8') as journal:
        def event(phase, **data):
            nonlocal sequence
            sequence += 1
            attempt.update(current_phase=phase, current_game=data.get('game'))
            json.dump(dict(sequence=sequence, utc=utc(), phase=phase, **data), journal, allow_nan=False)
            journal.write('\n')
            journal.flush()
            os.fsync(journal.fileno())

        try:
            for case in cases:
                deadline()
                run_case(case, archives, event, deadline)
                if case['accounting_uncertain'] or case.get('transport_closed') is not True:
                    raise ValueError('Unknown final accounting/closure prevents the next game')
                if case['execution_status'] == 'censored':
                    status = 'censored'
            deadline()
        except BaseException as error:
            status = 'censored' if isinstance(error, ResourceCeiling) else 'failed'
            failure = error_metadata(error)
            for case in cases:
                finish_missing(case, str(error) if isinstance(error, ResourceCeiling) else 'global_instrument_stop')
                if not case['started']:
                    case['stop_reason'] = 'global_resource_stop' if isinstance(error, ResourceCeiling) else 'global_instrument_stop'
        finally:
            attempt['work_seconds'] = time.monotonic() - began
    # Every worker has been closed before proof hashing resumes. No model/NPZ load.
    try:
        if preflight() != proof:
            raise ValueError('Final sources/proofs/preflight differ')
    except BaseException as error:
        status = 'failed'
        attempt['final_proof_failure'] = error_metadata(error)
    accounts = accounting(cases)
    if accounts['known_charged_frames'] > SPEC['max_charged_frames_total'] or not accounts['accounting_complete']:
        status = 'failed'
    result = dict(kind='action_signal_survey', passed=status != 'failed', execution_status=status,
                  created_utc=utc(), **proof, **accounts, cases=cases, failure=failure,
                  planned_pairs=SPEC['planned_pairs_total'],
                  acquired_pairs=sum(row['status'] == 'acquired' for case in cases for anchor in case['anchors'] for row in anchor['pairs']),
                  archive_bytes=archives.bytes, work_seconds=attempt['work_seconds'],
                  progress_sha256=sha(PROGRESS), progress_records=sequence,
                  model_inferences=0, optimizer_updates=0, gpu_used=False, agent_training_allowed=False,
                  descriptive_signal_threshold=None, gate_result=None,
                  limitations=read_json(ROOT / PROTOCOL)['limitations'])
    write_json(OUTPUT, result, exclusive=True)
    attempt.update(status=status, completed_utc=utc(), result_sha256=sha(OUTPUT), failure=failure,
                   cases=cases, **accounts, current_game=None, current_phase='closed',
                   progress_sha256=sha(PROGRESS), archive_bytes=archives.bytes)
    write_json(ATTEMPT, attempt)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Proofs/preflight only; no constructor or array load')
    args = parser.parse_args()
    if args.check:
        checked = preflight()
        print(json.dumps(dict(check=True, sources=len(checked['source_sha256']), evidence=len(checked['evidence_sha256']))))
    else:
        result = run()
        print(json.dumps({key: result[key] for key in ('passed', 'execution_status', 'acquired_pairs', 'known_charged_frames')}))
        sys.exit(0 if result['passed'] else 1)
