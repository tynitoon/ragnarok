"""Small standard-library-only archive and budget helpers for the declared trial."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'craft_v6_out/v70_photometric_correspondence'
PROTOCOL = 'docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_PROTOCOL.json'
SOURCES = (
    PROTOCOL, 'docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_DRAFT.md',
    'docs/ARC4_PHOTOMETRIC_CORRESPONDENCE_AMENDMENT_1.md',
    'docs/ARC4_SELECTIVE_CORRESPONDENCE_DRAFT.md',
    'ragnarok/__init__.py', 'ragnarok/learning/__init__.py',
    'ragnarok/evaluation/__init__.py', 'ragnarok/environments/__init__.py',
    'scripts/__init__.py', 'tests/__init__.py',
    'ragnarok/learning/photometric_correspondence_v70.py',
    'ragnarok/evaluation/photometric_scenes_v70.py',
    'ragnarok/evaluation/selective_correspondence_score_v70.py',
    'ragnarok/environments/rgb_branch_trace_v70.py',
    'ragnarok/environments/cold_emu_pool_v70.py',
    'ragnarok/environments/emu_pool.py', 'ragnarok/environments/frame_budget_v70.py',
    'ragnarok/environments/rgb_trace_difference_v70.py',
    'ragnarok/learning/counterfactual_targets_v70.py',
    'ragnarok/learning/visual_instances_v70.py',
    'tests/test_counterfactual_occlusion_limits_v70.py',
    'tests/test_photometric_correspondence_v70.py',
    'scripts/photometric_common_v70.py', 'scripts/photometric_correspondence_v70.py',
    'scripts/photometric_worker_v70.py', 'scripts/judge_photometric_v70.py',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_json_with_sha(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def ensure_space(additional_bytes, *, emergency=False):
    # A fixed reserve admits closing receipts and bounded supervisor stdout.
    used = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    reserve = 0 if emergency else 65536
    if used + additional_bytes + reserve > 64000000:
        raise OSError('The declared artifact budget cannot admit this write')


def write_bytes(path, payload, *, emergency=False):
    ensure_space(len(payload), emergency=emergency)
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value, *, emergency=False):
    payload = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8')
    write_bytes(path, payload, emergency=emergency)


def publish_completion(prefix, intent, receipt_name):
    """Publish a positive commit marker only after durable receipts and validation.

    The pending file is retained. A no-overwrite hard link publishes already
    fsynced, re-read bytes atomically; failed pending writes cannot authorize work.
    """
    receipt_hash = sha(OUT / receipt_name)
    pending = OUT / (prefix+'_completion_pending.json')
    complete = OUT / (prefix+'_completion.json')
    payload = dict(receipt_file=receipt_name, receipt_sha256=receipt_hash,
                   validated_utc=utc(), validated_elapsed_seconds=elapsed(intent))
    write_json(pending, payload, emergency=True)
    if read_json(pending) != payload or sha(OUT / receipt_name) != receipt_hash:
        raise ValueError('Completion payload or durable receipt changed')
    # Count both retained directory entries conservatively for artifact accounting.
    ensure_space(pending.stat().st_size, emergency=True)
    if elapsed(intent) >= intent['wall_limit_seconds']-1:
        raise TimeoutError('No room remains to publish the positive completion marker')
    os.link(pending, complete)  # Fails if the destination already exists.
    return sha(complete)


def source_hashes():
    return {path: sha(ROOT / path) for path in SOURCES}


def committed_sources():
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *SOURCES], cwd=ROOT,
                   check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *SOURCES], cwd=ROOT,
                   check=True, capture_output=True)
    return source_hashes()


def verify_sources(expected):
    if expected != source_hashes():
        raise ValueError('Frozen source bytes changed')


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')


def runtime():
    return dict(executable=sys.executable, python=sys.version,
                packages={name: importlib.metadata.version(name)
                          for name in ('numpy', 'torch', 'scipy', 'pytest')})


def elapsed(intent):
    return (time.perf_counter_ns() - intent['started_perf_counter_ns']) / 1e9


def check_time(intent, *, reserve=True):
    margin = intent.get('cleanup_reserve_seconds', 0) if reserve else 0
    if elapsed(intent) >= intent['wall_limit_seconds'] - margin:
        raise TimeoutError('Declared cumulative wall budget or cleanup reserve reached')


def artifact_inventory():
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(OUT.iterdir()) if p.is_file()}
    return files, sum(item['bytes'] for item in files.values())
