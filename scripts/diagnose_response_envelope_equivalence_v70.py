"""Post-result external diagnostic of exact local ambiguities; never a learner."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / 'craft_v6_out/v70_response_envelopes'
FIT_DIR = ROOT / 'craft_v6_out/v70_response_envelope_fit'
FOLDER = ROOT / 'craft_v6_out/v70_response_envelope_equivalence'
PROTOCOL = ROOT / 'docs/ARC4_RESPONSE_ENVELOPE_EQUIVALENCE_PROTOCOL.json'
PREFLIGHT = FOLDER / 'preflight_v0.json'
TARGET_DATA = TARGET_DIR / 'targets_v0.npz'
FIXED = {
    'craft_v6_out/v70_response_envelopes/targets_v0.json': 'd3c662ac1c269ff723add3e4417689a9dc57e7aa1745e9a700a7940bf78b6659',
    'craft_v6_out/v70_response_envelopes/targets_review_v0.json': 'd1211d166c2af7a402587c3225f981d273392be9b08277c1da10419d12d825a2',
    'craft_v6_out/v70_response_envelopes/targets_v0.npz': '3221b9bae8d2c4f3af771f7af6ee08a66733d2e5c6744c917be601d88c31e03f',
    'craft_v6_out/v70_response_envelope_fit/fit_v0.json': '9472d1153ea590b50fb3bd0a8f37baf5a989392604ade6bde60501a5155bad4a',
    'craft_v6_out/v70_response_envelope_fit/fit_review_v0.json': '3df95a49c616d980c940b9ae20260a8e5c912560eecf07c6e0a8ad7f122d0467',
}
SOURCES = ('scripts/diagnose_response_envelope_equivalence_v70.py',
           'tests/test_response_envelope_equivalence_v70.py',
           'docs/ARC4_RESPONSE_ENVELOPE_EQUIVALENCE_DRAFT.md',
           'docs/ARC4_RESPONSE_ENVELOPE_EQUIVALENCE_PROTOCOL.json')
SPEC = dict(version=0, role='post-result external diagnostic; no new training or validation',
            radius=31, past_shape=[12, 2, 210, 160, 3], target_shape=[12, 210, 160],
            grouping='ordered raw RGB patch plus top/bottom/left/right distances clipped to radius',
            collision_resolution='full patch byte equality within each topology/SHA256 bucket',
            iou_bound='exact per-anchor subset optimum relaxed independently; equal nonempty game means',
            bce_bound='global equivalence groups; actual equal-anchor balanced class masses; natural log',
            max_work_seconds=300, cleanup_reserve_seconds=10, max_binary_output_bytes=16000000,
            work_scope='after attempt fsync through scientific payload fsync/hash and journal close; final receipt excluded',
            cpu_threads=1, gpu_used=False, new_emulated_frames=0, model_inferences=0, optimizer_updates=0,
            retry=False, claim='bounds for the declared mathematical local operator; no capacity or convergence conclusion')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def signature(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=sha_bytes(array.tobytes(order='C')))


def edge_key(y, x, height, width, radius=31):
    if radius < 0 or not (0 <= y < height and 0 <= x < width):
        raise ValueError('A valid pixel and nonnegative radius are required')
    return (min(y, radius), min(height - 1 - y, radius),
            min(x, radius), min(width - 1 - x, radius))


def partition_patches(past_pairs, checkpoint=lambda: None,
                      digest=lambda value: hashlib.sha256(value).digest(), radius=31):
    past = np.asarray(past_pairs)
    if (past.dtype != np.uint8 or past.ndim != 5 or past.shape[1] != 2
            or past.shape[-1] != 3 or min(past.shape) < 1 or radius < 0):
        raise ValueError('Ordered uint8 pairs with shape [anchor,2,height,width,3] required')
    anchors, _, height, width, _ = past.shape
    rgb = past.transpose(0, 2, 3, 1, 4).reshape(anchors, height, width, 6)
    padded = np.pad(rgb, ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode='edge')
    size = 2 * radius + 1
    group_ids = np.empty((anchors, height, width), dtype=np.int32)
    representatives, buckets, collisions = [], {}, 0
    topology = [[edge_key(y, x, height, width, radius) for x in range(width)] for y in range(height)]
    for anchor in range(anchors):
        for y in range(height):
            checkpoint()
            for x in range(width):
                raw = padded[anchor, y:y + size, x:x + size].tobytes(order='C')
                key = (topology[y][x], digest(raw))
                bucket = buckets.get(key)
                chosen = None
                if bucket is not None:
                    for group in bucket:
                        ra, ry, rx = representatives[group]
                        other = padded[ra, ry:ry + size, rx:rx + size].tobytes(order='C')
                        if raw == other:
                            chosen = group
                            break
                if chosen is None:
                    chosen = len(representatives)
                    representatives.append((anchor, y, x))
                    if bucket is None:
                        buckets[key] = [chosen]
                    else:
                        collisions += 1
                        bucket.append(chosen)
                group_ids[anchor, y, x] = chosen
    checkpoint()
    return group_ids, np.asarray(representatives, dtype=np.int32), collisions


def anchor_iou_bound(positive_counts, negative_counts):
    positive, negative = np.asarray(positive_counts), np.asarray(negative_counts)
    if (positive.ndim != 1 or positive.shape != negative.shape
            or positive.dtype.kind not in 'iu' or negative.dtype.kind not in 'iu'
            or (positive < 0).any() or (negative < 0).any()):
        raise ValueError('Nonnegative integer group counts of matching shape required')
    total = sum(int(value) for value in positive)
    if total == 0:
        return dict(iou=None, numerator=0, denominator=0, selected_groups=0)
    pure, mixed = [], []
    for p, n in zip(positive, negative, strict=True):
        p, n = int(p), int(n)
        if p:
            (pure if n == 0 else mixed).append((p, n))
    ordered = pure + sorted(mixed, key=lambda row: Fraction(row[0], row[1]), reverse=True)
    best_num, best_den, best_count, tp, fp = 0, total, 0, 0, 0
    for count, (p, n) in enumerate(ordered, 1):
        tp, fp = tp + p, fp + n
        denominator = total + fp
        if tp * best_den > best_num * denominator:
            best_num, best_den, best_count = tp, denominator, count
    return dict(iou=best_num / best_den, numerator=best_num,
                denominator=best_den, selected_groups=best_count)


def summarize_partition(group_ids, targets, cases):
    ids, labels = np.asarray(group_ids), np.asarray(targets)
    if (ids.ndim != 3 or ids.dtype.kind not in 'iu' or min(ids.shape) < 1 or (ids < 0).any()
            or ids.shape != labels.shape or labels.dtype != bool or len(cases) != len(ids)):
        raise ValueError('Complete integer partition and boolean labels required')
    groups = int(ids.max()) + 1
    if not np.array_equal(np.unique(ids), np.arange(groups)):
        raise ValueError('Group IDs must be dense from zero with no missing group')
    anchors, height, width = ids.shape
    pixels = height * width
    positive_mass, negative_mass = np.zeros(groups), np.zeros(groups)
    global_positive, global_negative = np.zeros(groups, dtype=np.int64), np.zeros(groups, dtype=np.int64)
    rows, grouped = [], {}
    for mapping, target, case in zip(ids, labels, cases, strict=True):
        positive = np.bincount(mapping[target], minlength=groups)
        negative = np.bincount(mapping[~target], minlength=groups)
        p, n = int(target.sum()), int((~target).sum())
        global_positive += positive
        global_negative += negative
        if p and n:
            positive_mass += positive / (2 * anchors * p)
            negative_mass += negative / (2 * anchors * n)
        else:
            positive_mass += positive / (anchors * pixels)
            negative_mass += negative / (anchors * pixels)
        bound = anchor_iou_bound(positive, negative)
        conflict = (positive > 0) & (negative > 0)
        rows.append(dict(game=case['game'], anchor_index=case['anchor_index'], target_pixels=p,
                         iou_upper_bound=bound['iou'], bound_numerator=bound['numerator'],
                         bound_denominator=bound['denominator'], selected_groups=bound['selected_groups'],
                         within_anchor_conflicting_groups=int(conflict.sum()),
                         positive_pixels_in_within_anchor_conflicts=int(positive[conflict].sum())))
        if p:
            grouped.setdefault(case['game'], []).append(Fraction(bound['numerator'], bound['denominator']))
    game_bounds = {game: sum(values, Fraction()) / len(values) for game, values in grouped.items()}
    macro = sum(game_bounds.values(), Fraction()) / len(game_bounds) if game_bounds else None
    terms = []
    for u, v in zip(positive_mass, negative_mass, strict=True):
        if u and v:
            terms.append(-float(u) * math.log(float(u / (u + v)))
                         - float(v) * math.log(float(v / (u + v))))
    conflict = (global_positive > 0) & (global_negative > 0)
    return dict(groups=groups, pixels=int(ids.size), cases=rows,
                global_conflicting_groups=int(conflict.sum()),
                pixels_in_global_conflicting_groups=int((global_positive[conflict] + global_negative[conflict]).sum()),
                positive_pixels_in_global_conflicting_groups=int(global_positive[conflict].sum()),
                balanced_bce_lower_bound=math.fsum(terms),
                positive_loss_mass=math.fsum(positive_mass.tolist()),
                negative_loss_mass=math.fsum(negative_mass.tolist()),
                iou_upper_bound_by_game={game: float(value) for game, value in game_bounds.items()},
                macro_nonempty_iou_upper_bound=float(macro) if macro is not None else None,
                macro_nonempty_iou_upper_bound_fraction=[str(macro.numerator), str(macro.denominator)] if macro is not None else None,
                minimum_iou_ruled_out_by_relaxed_bound=macro < Fraction(1, 4) if macro is not None else None,
                jointly_optimizes_empty_false_positives=False)


def committed(path):
    rel = path.relative_to(ROOT).as_posix()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('hash-object', f'--path={rel}', str(path)) != git('rev-parse', f'HEAD:{rel}'):
        raise ValueError(f'Required file differs from its committed Git blob: {rel}')


def proofs():
    """No model, NPZ decoding or source branch archive access in this preflight."""
    runtime = dict(python=sys.executable, version=sys.version.split()[0], numpy=np.__version__,
                   environment={key: os.environ.get(key) for key in
                                ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if (Path(sys.executable).resolve() != Path('C:/Python314/python.exe').resolve()
            or runtime['version'] != '3.14.3' or runtime['numpy'] != '2.4.4'
            or runtime['environment'] != dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            or any(name.split('.')[0] in ('torch', 'ale_py') for name in sys.modules)):
        raise ValueError('Exact NumPy-only CPU runtime required')
    if json.loads(PROTOCOL.read_text(encoding='utf-8')) != SPEC:
        raise ValueError('Diagnostic specification changed')
    for rel, expected in FIXED.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Fixed evidence changed: {rel}')
        committed(ROOT / rel)
    fit = json.loads((FIT_DIR / 'fit_v0.json').read_text(encoding='utf-8'))
    review = json.loads((FIT_DIR / 'fit_review_v0.json').read_text(encoding='utf-8'))
    if (fit['execution_status'] != 'complete' or fit['passed'] is not False
            or review.get('kind') != 'response_envelope_fit_independent_result_review'
            or review.get('passed') is not True or review.get('source_sha256') != fit['source_sha256']
            or review.get('result_sha256') != FIXED['craft_v6_out/v70_response_envelope_fit/fit_v0.json']):
        raise ValueError('The fixed reviewed complete negative fit is required')
    sources = dict(fit['source_sha256'])
    for rel in SOURCES:
        if rel in sources:
            raise ValueError('A new diagnostic must not replace an old source')
        sources[rel] = sha(ROOT / rel)
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Required source changed: {rel}')
        committed(ROOT / rel)
    return dict(source_sha256=sources, evidence_sha256=FIXED.copy(), runtime=runtime)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Fixed diagnostic budget reached; no retry or extension')


def run():
    checked = proofs()
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (preflight.get('kind') != 'response_envelope_equivalence_independent_preflight'
            or preflight.get('passed') is not True
            or any(preflight.get(key) != checked[key] for key in ('source_sha256', 'evidence_sha256', 'runtime'))):
        raise ValueError('An exact committed independent preflight is required')
    committed(PREFLIGHT)
    names = ('equivalence_v0_attempt.json', 'equivalence_v0_progress.jsonl',
             'equivalence_v0_groups.npz', 'equivalence_v0_payload.json', 'equivalence_v0.json')
    if any((FOLDER / name).exists() for name in names):
        raise FileExistsError('Preserve all previous diagnostic traces; no retry')
    target_manifest = json.loads((TARGET_DIR / 'targets_v0.json').read_text(encoding='utf-8'))
    write_json(FOLDER / names[0], dict(kind='response_envelope_equivalence_attempt',
               created_utc=datetime.now(timezone.utc).isoformat(), protocol=SPEC,
               preflight_sha256=sha(PREFLIGHT), **checked))
    started = time.monotonic()
    deadline = started + SPEC['max_work_seconds'] - SPEC['cleanup_reserve_seconds']
    errors, artifacts, metrics = [], {}, None
    journal = None
    arrays_decoded = 0
    def log(event, **fields):
        journal.write(json.dumps(dict(event=event, seconds=time.monotonic() - started, **fields), allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
    try:
        journal = (FOLDER / names[1]).open('x', encoding='utf-8', newline='\n')
        log('started')
        check_time(deadline)
        with np.load(TARGET_DATA, allow_pickle=False) as data:
            if set(data.files) != {'past_pairs', 'observed_response_envelopes'}:
                raise ValueError('Unexpected array in the fixed derived dataset')
            past = data['past_pairs']
            arrays_decoded += 1
            targets = data['observed_response_envelopes']
            arrays_decoded += 1
        if (list(past.shape) != SPEC['past_shape'] or past.dtype != np.uint8
                or list(targets.shape) != SPEC['target_shape'] or targets.dtype != bool
                or signature(past) != target_manifest['output']['arrays']['past_pairs']
                or signature(targets) != target_manifest['output']['arrays']['observed_response_envelopes']):
            raise ValueError('Fixed derived array dimensions, dtype or signature changed')
        check_time(deadline)
        log('derived_arrays_verified', arrays_decoded=arrays_decoded)
        ids, representatives, collisions = partition_patches(past, checkpoint=lambda: check_time(deadline), radius=SPEC['radius'])
        log('partition_complete', pixels=int(ids.size), groups=len(representatives), hash_collision_groups=collisions)
        check_time(deadline)
        metrics = summarize_partition(ids, targets, target_manifest['cases'])
        metrics['hash_collision_groups'] = collisions
        check_time(deadline)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, group_ids=ids, group_representatives=representatives)
        binary = buffer.getvalue()
        if len(binary) > SPEC['max_binary_output_bytes']:
            raise ValueError('Fixed binary output ceiling exceeded')
        with (FOLDER / names[2]).open('xb') as stream:
            stream.write(binary)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts[names[2]] = dict(sha256=sha_bytes(binary), bytes=len(binary),
                                  arrays=dict(group_ids=signature(ids), group_representatives=signature(representatives)))
        check_time(deadline)
        log('completed', artifacts=artifacts)
    except BaseException as error:
        errors.append(dict(stage='diagnostic', type=type(error).__name__, message=str(error)))
    finally:
        if journal is not None:
            try:
                log('finished', errors=errors)
                journal.close()
            except BaseException as error:
                errors.append(dict(stage='journal_close', type=type(error).__name__, message=str(error)))
    payload = dict(kind='response_envelope_equivalence_scientific_payload', **checked,
                   protocol=SPEC, preflight_sha256=sha(PREFLIGHT), attempt_sha256=sha(FOLDER / names[0]),
                   progress_sha256=sha(FOLDER / names[1]) if (FOLDER / names[1]).exists() else None,
                   errors=errors, metrics=metrics, artifacts=artifacts, arrays_decoded=arrays_decoded,
                   array_count_scope='successful array accessor returns; partial decompression is not counted',
                   source_branch_archives_decoded=0, new_emulated_frames=0, optimizer_updates=0,
                   model_inferences=0, gpu_used=False,
                   inherited_collection_charged_frames=157530,
                   candidate_execution_status='complete' if not errors else 'incomplete', claim=SPEC['claim'])
    payload_hash = None
    try:
        write_json(FOLDER / names[3], payload)
        payload_hash = sha(FOLDER / names[3])
    except BaseException as error:
        errors.append(dict(stage='payload', type=type(error).__name__, message=str(error)))
    elapsed = time.monotonic() - started
    status = 'complete' if not errors and elapsed <= SPEC['max_work_seconds'] else 'incomplete'
    result = dict(payload, kind='response_envelope_equivalence_result', errors=errors,
                  execution_status=status, work_seconds=elapsed, scientific_payload_sha256=payload_hash)
    write_json(FOLDER / names[4], result)
    print(json.dumps(dict(execution_status=status, work_seconds=elapsed, arrays_decoded=arrays_decoded,
                          errors=errors, metrics=metrics), allow_nan=False))
    return 0 if status == 'complete' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
