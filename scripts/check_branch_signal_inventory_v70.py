"""Inventory only the RGB signal in six already-paid, reviewed branch pairs.

No emulator, model, Torch, ordinary image or object extractor is imported/read.
Real NPZ access is behind the committed independent preflight in run().
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'docs/ARC4_BRANCH_SIGNAL_INVENTORY_PROTOCOL.json'
FOLDER = ROOT / 'craft_v6_out/v70_branch_signal_inventory'
PREFLIGHT = FOLDER / 'preflight_v0.json'
ATTEMPT = FOLDER / 'inventory_v0_attempt.json'
OUTPUT = FOLDER / 'inventory_v0.json'
NATIVE = 'craft_v6_out/v70_owned_ordinary_native/wiring_v0.json'
DIAGNOSTIC = 'craft_v6_out/v70_archived_pixel_diagnostics/cpu_v0.json'
PINNED_RESULTS = {
    NATIVE: '5db50e4ee27fedf0ec77a62fab9c13efd309749f7a127e47d73ec4f56ae886af',
    DIAGNOSTIC: '610ef8a238449939290efa6ad3cbdb2bfe886cbfc5efddf032113540fd85971d',
}
PINNED_REVIEWS = {
    'craft_v6_out/v70_owned_ordinary_native/wiring_review_v0.json':
        '8b4105ce479e7d990293a89ce8ea3789506600729f3e1f721dbf59653f04a668',
    'craft_v6_out/v70_archived_pixel_diagnostics/cpu_review_v0.json':
        '574e1881b67bb1e5e07c8f05c9433533eb09abdbd30507a37465cb5082eac9aa',
}
NEW_SOURCES = (
    'scripts/__init__.py', 'tests/__init__.py',
    'scripts/check_branch_signal_inventory_v70.py',
    'tests/test_branch_signal_inventory_v70.py', PROTOCOL,
    'docs/ARC4_BRANCH_SIGNAL_INVENTORY_DRAFT.md',
)
NAMES = ('pong', 'breakout', 'freeway', 'seaquest', 'ms_pacman', 'asteroids',
         'pong_prefix_1', 'pong_prefix_2', 'pong_prefix_3')
SHAPE = (210, 160, 3)
BRANCH_KEYS = ('branch_start_pair',) + tuple(
    f'branch_{side}_{field}' for side in ('intervention', 'control')
    for field in ('raw', 'pooled', 'endpoint'))
ORDINARY_KEYS = tuple(f's{index:03d}_{field}' for index in (0, 1, 2, 3, 126, 127, 128, 129)
                      for field in ('start_pair', 'raw', 'pooled', 'endpoint'))
SPEC = dict(case_names=list(NAMES), present_pairs=6, absent_pairs=3, horizons=list(range(1, 9)),
            image_kinds=['raw', 'pooled'], geometry=list(SHAPE), pixels_per_frame=33600,
            intervention_frames=8, control_frames=8, requested_action=5, control_action=0,
            anchor_frame=576, anchor_epoch=0, operation_index=145,
            max_work_seconds=300, cpu_threads=1, model_inferences=0,
            optimizer_updates=0, new_emulated_frames=0)


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


def local_path(path):
    if not isinstance(path, str) or Path(path).is_absolute() or '\\' in path:
        raise ValueError('Canonical relative evidence path required')
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or '..' in Path(path).parts:
        raise ValueError('Evidence must remain within the repository')
    return resolved


def require_committed(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={path}', str(local_path(path))) != git('rev-parse', f'HEAD:{path}'):
        raise ValueError('Current source/evidence differs from its committed blob')


def runtime():
    return dict(executable=str(Path(sys.executable).resolve()), python=sys.version, numpy=np.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                numerical_threads={key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def expected_branch(name):
    return dict(operation_id=['owned-ordinary-v0/' + name, 145], intervention_frames=8,
                control_frames=8, anchor_frame=576, anchor_epoch=0, requested_action=5,
                control_action=0, requested_frames=8, effect_assumption=[0.] * 12)


def validate_population(native, diagnostic):
    cases = native.get('cases', [])
    if (native.get('kind') != 'owned_ordinary_native_wiring' or native.get('passed') is not True
            or native.get('accounting_complete') is not True
            or native.get('known_charged_frames') != 14908 or native.get('constructor_count') != 18
            or [case.get('name') for case in cases] != list(NAMES)):
        raise ValueError('Exact nine-case reviewed native population required')
    for index, case in enumerate(cases):
        name = case['name']
        if (case.get('passed') is not True or 'branch' not in case
                or case['branch'] != (expected_branch(name) if index < 6 else None)
                or case.get('selected_trace_archive') !=
                f'craft_v6_out/v70_owned_ordinary_native/{name}_rgb_v0.npz'):
            raise ValueError('Exact branch metadata, absent cases and archive names required')
    rows = diagnostic.get('cases', [])
    if (diagnostic.get('kind') != 'archived_pixel_cpu_numerical_diagnostics'
            or diagnostic.get('passed') is not True or diagnostic.get('ordinary_records') != 72
            or diagnostic.get('branch_pairs') != 6 or diagnostic.get('optimizer_updates') != 0
            or diagnostic.get('new_emulated_frames') != 0
            or [row.get('case') for row in rows] != list(NAMES)):
        raise ValueError('Exact preserved zero-C diagnostic required')
    for index, row in enumerate(rows):
        branch = row.get('branch')
        if index >= 6:
            if branch is not None:
                raise ValueError('Absent diagnostic branch required')
        elif (not isinstance(branch, dict) or branch.get('operation_id') != expected_branch(NAMES[index])['operation_id']
              or branch.get('common_prefix_frames') != 8 or branch.get('supervised_pixels') != 0
              or branch.get('positive_pixels') != 0 or branch.get('own_instances') != (12, 12, 42, 14, 50, 20)[index]):
            raise ValueError('Preserved C supervision counts changed')
    return cases


def proofs():
    """Read bytes/JSON/Git only; never opens an NPZ array or imports a model."""
    protocol, current = read_json(ROOT / PROTOCOL), runtime()
    if (protocol.get('kind') != 'prospective_branch_signal_inventory_v0' or protocol.get('spec') != SPEC
            or protocol.get('runtime') != current or protocol.get('agent_training_allowed') is not False
            or protocol.get('gate_result') is not None or current['cuda_visible_devices'] != ''
            or any(value != '1' for value in current['numerical_threads'].values())
            or any(key.split('.')[0] in ('torch', 'ale_py', 'tensorflow') for key in sys.modules)):
        raise ValueError('Exact CPU-only inventory protocol/runtime and no model/emulator imports required')
    sources, evidence = {}, {}

    def merge(destination, path, value):
        if (not isinstance(value, str) or len(value) != 64 or sha(local_path(path)) != value
                or path in sources and sources[path] != value
                or path in evidence and evidence[path] != value):
            raise ValueError('Source/evidence hash changed or conflicting maps')
        destination[path] = value

    for path, value in PINNED_REVIEWS.items():
        merge(evidence, path, value)
        review = read_json(local_path(path))
        if review.get('passed') is not True:
            raise ValueError('Passing independent historical reviews required')
        for field, destination in (('source_sha256', sources), ('evidence_sha256', evidence)):
            if not isinstance(review.get(field), dict):
                raise ValueError('Explicit historical source/evidence maps required')
            for p, h in review[field].items():
                merge(destination, p, h)
        # result_sha256 is a string in these reviews, not another path/hash map.
        result_path = NATIVE if 'owned_ordinary_native/' in path else DIAGNOSTIC
        if review.get('result_sha256') != PINNED_RESULTS[result_path]:
            raise ValueError('Historical review must attest its exact result')
    for path, value in PINNED_RESULTS.items():
        if evidence.get(path) != value:
            raise ValueError('Pinned result absent from reviewed evidence')
        merge(evidence, path, value)
    cases = validate_population(read_json(ROOT / NATIVE), read_json(ROOT / DIAGNOSTIC))
    for case in cases:
        path, value = case['selected_trace_archive'], case['selected_trace_sha256']
        if evidence.get(path) != value:
            raise ValueError('All nine exact archives must belong to reviewed evidence')
        merge(evidence, path, value)
    for path in NEW_SOURCES:
        merge(sources, path, sha(local_path(path)))
    for path in sorted(set(sources) | set(evidence)):
        require_committed(path)
    return dict(source_sha256=sources, evidence_sha256=evidence, runtime=current)


def preflight():
    proof = proofs()
    review = read_json(PREFLIGHT)
    if (review.get('kind') != 'branch_signal_inventory_independent_preflight' or review.get('passed') is not True
            or any(review.get(key) != value for key, value in proof.items())):
        raise ValueError('Exact independent preflight required before real NPZ access')
    require_committed(PREFLIGHT.relative_to(ROOT).as_posix())
    return dict(**proof, preflight_review_sha256=sha(PREFLIGHT))


def validate_arrays(arrays, *, shape=None):
    """Pure array contract, also usable with small synthetic RGB fixtures."""
    shape = SHAPE if shape is None else shape
    if set(arrays) != set(BRANCH_KEYS):
        raise ValueError('Only seven explicit branch arrays required')
    start = arrays['branch_start_pair']
    if (not isinstance(start, np.ndarray) or start.dtype != np.uint8 or start.shape != (2, *shape)
            or len(shape) != 3 or shape[-1] != 3 or min(shape) < 1):
        raise ValueError('Two native uint8 RGB anchor frames required')
    lengths = []
    for side in ('intervention', 'control'):
        raw, pooled, endpoint = (arrays[f'branch_{side}_{field}'] for field in ('raw', 'pooled', 'endpoint'))
        if (any(not isinstance(a, np.ndarray) or a.dtype != np.uint8 for a in (raw, pooled, endpoint))
                or raw.ndim != 4 or raw.shape[1:] != shape or not 0 <= len(raw) <= 8
                or pooled.shape != raw.shape or endpoint.shape != shape):
            raise ValueError('Only actual zero-to-eight uint8 RGB frames and native geometry allowed')
        previous = start[-1]
        for image, maximum in zip(raw, pooled, strict=True):
            if not np.array_equal(np.maximum(previous, image), maximum):
                raise ValueError('Pooled RGB must be the real adjacent maximum from the anchor')
            previous = image
        expected = pooled[-1] if len(raw) else np.maximum(start[0], start[1])
        if not np.array_equal(expected, endpoint):
            raise ValueError('Endpoint differs from the adjacent RGB history')
        lengths.append(len(raw))
    return lengths


def summarize(intervention, control, *, shape=None):
    """Count True changes, not channel differences or nonzero RGB intensities."""
    shape = SHAPE if shape is None else shape
    common = min(len(intervention), len(control))
    denominator = int(shape[0] * shape[1])
    union = np.zeros(shape[:2], dtype=bool)
    first = np.zeros(shape[:2], dtype=np.uint8)
    rows = []
    for horizon in range(1, 9):
        observed = horizon <= common
        row = dict(horizon=horizon, observed=observed, pixels_per_frame=denominator,
                   observed_pixel_denominator=denominator if observed else None,
                   changed_pixels=None, changed_fraction=None, cumulative_union_pixels=None,
                   cumulative_union_fraction=None)
        if observed:
            changed = np.any(intervention[horizon - 1] != control[horizon - 1], axis=-1)
            first[(first == 0) & changed] = horizon
            union |= changed
            count, cumulative = int(np.count_nonzero(changed == True)), int(np.count_nonzero(union))
            row.update(changed_pixels=count, changed_fraction=count / denominator,
                       cumulative_union_pixels=cumulative, cumulative_union_fraction=cumulative / denominator)
        rows.append(row)
    histogram = [dict(first_changed_frame=horizon,
                      pixels=int(np.count_nonzero(first == horizon)) if common and horizon <= common else None)
                 for horizon in range(9)]
    return dict(common_prefix_frames=common, unpaired_intervention_frames=len(intervention) - common,
                unpaired_control_frames=len(control) - common, horizons=rows,
                first_changed_histogram=histogram,
                first_changed_zero_meaning='No change observed in the common prefix; not a statement beyond it',
                all_zero_over_observed_prefix=bool(not union.any()) if common else None,
                complete_eight_frame_pair=common == 8)


def absent_case(case):
    empty = np.empty((0, *SHAPE), dtype=np.uint8)
    return dict(case=case['name'], pair_present=False, reason='No branch pair in the fixed native case',
                archive_sha256=case['selected_trace_sha256'], loaded_array_keys=[],
                raw=summarize(empty, empty), pooled=summarize(empty, empty))


def read_case(case, proof):
    """Only called by run after preflight; no ordinary NPZ member is materialized."""
    path = local_path(case['selected_trace_archive'])
    expected = proof['evidence_sha256'].get(case['selected_trace_archive'])
    if expected != case['selected_trace_sha256'] or sha(path) != expected:
        raise ValueError('Archive changed before reading')
    present = case['branch'] is not None
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = set(ORDINARY_KEYS) | (set(BRANCH_KEYS) if present else set())
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != expected_keys:
            raise ValueError('Archive members differ from the exact native inventory')
        if present:
            arrays = {key: archive[key] for key in BRANCH_KEYS}
            if validate_arrays(arrays) != [8, 8]:
                raise ValueError('Frozen six branches require eight actual frames on each side')
            row = dict(case=case['name'], pair_present=True, metadata=case['branch'],
                       archive_sha256=expected, loaded_array_keys=list(BRANCH_KEYS))
            for kind in ('raw', 'pooled'):
                row[kind] = summarize(arrays[f'branch_intervention_{kind}'], arrays[f'branch_control_{kind}'])
        else:
            row = absent_case(case)
    if sha(path) != expected:
        raise ValueError('Archive changed during reading')
    return row


def run():
    if ATTEMPT.exists() or OUTPUT.exists():
        raise ValueError('One inventory attempt only; no retry or overwrite')
    proof = preflight()
    FOLDER.mkdir(parents=True, exist_ok=True)
    attempt = dict(kind='branch_signal_inventory_attempt', status='started', started_utc=utc(), **proof,
                   completed_cases=[], current_case=None, current_phase='initializing',
                   model_inferences=0, optimizer_updates=0, new_emulated_frames=0)
    write_json(ATTEMPT, attempt, exclusive=True)
    began = time.perf_counter()

    def deadline():
        if time.perf_counter() - began > SPEC['max_work_seconds']:
            raise TimeoutError('Declared inventory work ceiling exceeded; no retry')

    try:
        cases = validate_population(read_json(ROOT / NATIVE), read_json(ROOT / DIAGNOSTIC))
        for case in cases:
            attempt.update(current_case=case['name'], current_phase='read_branch_arrays')
            deadline()
            row = read_case(case, proof)
            deadline()
            attempt['completed_cases'].append(row)
        attempt.update(current_case=None, current_phase='final_proofs', work_seconds=time.perf_counter() - began)
        deadline()
        if preflight() != proof:
            raise ValueError('Sources/proofs/runtime/preflight changed during inventory')
        result = dict(kind='paid_branch_signal_inventory', passed=True, created_utc=utc(), **proof,
                      cases=attempt['completed_cases'], present_pairs=6, absent_pairs=3,
                      materialized_arrays=42, ordinary_arrays_read=0, common_paired_frames=48,
                      model_inferences=0, optimizer_updates=0, new_emulated_frames=0,
                      gpu_used=False, agent_training_allowed=False, gate_result=None,
                      preserved_C_supervised_pixels=0, preserved_C_own_instances=150,
                      all_zero_pairs={kind: [row['case'] for row in attempt['completed_cases']
                                           if row['pair_present'] and row[kind]['all_zero_over_observed_prefix']]
                                      for kind in ('raw', 'pooled')},
                      work_seconds=attempt['work_seconds'], limitations=read_json(ROOT / PROTOCOL)['limitations'])
        write_json(OUTPUT, result, exclusive=True)
        attempt.update(status='complete', completed_utc=utc(), result_sha256=sha(OUTPUT), current_phase='complete')
        return result
    except BaseException as error:
        attempt.update(status='failed', failed_utc=utc(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        attempt.setdefault('work_seconds', time.perf_counter() - began)
        write_json(ATTEMPT, attempt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate committed preflight; no NPZ arrays')
    args = parser.parse_args()
    if args.check:
        checked = preflight()
        print(json.dumps(dict(check=True, sources=len(checked['source_sha256']), evidence=len(checked['evidence_sha256']))))
    else:
        result = run()
        print(json.dumps(dict(passed=result['passed'], present_pairs=6, absent_pairs=3,
                              all_zero_pairs=result['all_zero_pairs'])))
