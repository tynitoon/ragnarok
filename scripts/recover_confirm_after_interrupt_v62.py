"""One sealed continuation of s2 after the 10 September process interruption.

Only orchestration changes. Preserve six complete worlds and the complete M4
checkpoint, replay the unfinished Fa arm through frozen --resume, never extend
the first recovery's absolute deadline. No model is imported by preflight.
"""
import argparse
import copy
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts import recover_confirm_v62 as prior_runner

base = prior_runner.base
EVIDENCE = 'v62_confirm_interruption_20260910'
STATE = 'v62_confirm_second_recovery_state.json'
REVIEW = 'v62_confirm_second_recovery_review.json'
LAUNCH = 'v62_confirm_second_recovery_launch.json'
SOURCE_PATHS = ('scripts/recover_confirm_after_interrupt_v62.py',
                'tests/test_recover_confirm_after_interrupt_v62.py',
                'scripts/launch_detached_confirm_v62.py',
                'tests/test_launch_detached_confirm_v62.py')
SNAPSHOTS = ('interruption.json', 'v62_confirm_recovery_state.json',
             'v62_confirm_recovery_stdout.log', 'v62_confirm_recovery_stderr.log',
             'v61_confirm_s2.log', 's2_8320_M4.json', 'v61_confirm.json',
             'benchmark_console_v0.txt', '.gitattributes',
             'process_lifetime_wmi_probe.py', 'process_lifetime_wmi_probe.json',
             'process_lifetime_wmi_probe_completed.json')
M4 = 'v61_confirm_ckpt/s2_8320_M4.json'
CLEANUP_RESERVE_SECONDS = 30.
POLL_SECONDS = .5
HANDSHAKE_SECONDS = 10.


def utc_now():
    return datetime.now(timezone.utc)


def deadline_from(prior):
    start = datetime.fromisoformat(prior['started_utc'])
    base.require(start.utcoffset() is not None, 'Timezone-aware original start required')
    seconds = prior['provenance']['design']['remaining_seconds']
    base.require(math.isfinite(seconds) and seconds > 0, 'Finite positive original allowance required')
    # Round DOWN to an integer second, avoiding a fractional extension through
    # float/timedelta rounding. Down time and investigation consume this window.
    return datetime.fromtimestamp(math.floor(start.timestamp() + seconds), timezone.utc)


def validate_saved_m4(path):
    rows = base.read_json(path)
    base.require(isinstance(rows, list) and len(rows) == 3, 'Complete saved M4 stream required')
    for row, goal, tier in zip(rows, (5, 11, 1), (2, 3, 4)):
        base.require(row.get('goal') == goal and row.get('tier') == tier
                     and isinstance(row.get('curve'), list) and len(row['curve']) == 7
                     and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                             for v in row['curve']), 'Saved M4 curve is incomplete or changed')
        for field in ('rows', 'demos'):
            base.require(isinstance(row.get(field), list) and len(row[field]) == 6
                         and all(type(v) is int and v >= 0 for v in row[field]),
                         'Saved M4 round counters are incomplete')
    return rows


def other_controllers():
    """Read process command lines, selecting only this experiment's Python jobs."""
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | "
        "Where-Object { $_.ProcessId -ne " + str(os.getpid()) +
        " -and $_.CommandLine -match 'scripts[.\\\\/](confirm_v61|confirm_resume_v62|recover_confirm_v62|recover_confirm_after_interrupt_v62)' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    raw = subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                                  text=True, encoding='utf-8').strip()
    value = json.loads(raw) if raw else []
    return value if isinstance(value, list) else [value]


def require_detached():
    """Refuse GPU work inside a launcher-owned Windows Job Object."""
    base.require(os.name == 'nt', 'This recovery is specific to the inspected Windows host')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
    inside = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    base.require(not inside.value, 'Launch reviewed recovery outside the transient parent job before GPU work')


def run_child_until(argv, root, remaining):
    """Supervise the fixed deadline after spawn and throughout execution.

    Leave thirty seconds for Windows process termination and final evidence.
    The relative timeout in the historical helper does not include Popen time.
    """
    if remaining() <= CLEANUP_RESERVE_SECONDS:
        raise TimeoutError('Insufficient original time for dispatch and cleanup')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    child = subprocess.Popen(argv, cwd=root, shell=False, creationflags=flags)
    try:
        print(f'Child PID {child.pid}: {argv}', flush=True)
        while True:
            available = remaining() - CLEANUP_RESERVE_SECONDS
            if available <= 0:
                raise TimeoutError('Stopping before original deadline; cleanup time reserved')
            try:
                code = child.wait(timeout=min(POLL_SECONDS, available))
            except subprocess.TimeoutExpired:
                continue
            if code:
                raise subprocess.CalledProcessError(code, argv)
            return
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        raise


def await_dispatch(root, proof):
    """No GPU or experiment state until the launcher publishes its PID receipt.

    The launcher may fail or exit at any time. A pending receipt times out;
    a dispatched receipt hands sole deadline ownership to this controller.
    """
    started = time.monotonic()
    path = root / 'craft_v6_out' / LAUNCH
    deadline = datetime.fromisoformat(proof['absolute_deadline_utc'])
    while time.monotonic() - started < HANDSHAKE_SECONDS and utc_now() < deadline:
        if path.exists():
            row = base.read_json(path)
            if row.get('status') == 'dispatched':
                base.require(row.get('kind') == 'v62_detached_second_recovery_launch'
                             and row.get('controller_pid') == os.getpid()
                             and row.get('review_sha256') == proof['review_sha256']
                             and row.get('source_sha256') == proof['source_sha256']
                             and row.get('absolute_deadline_utc') == proof['absolute_deadline_utc']
                             and row.get('cwd') == str(root)
                             and row.get('argv') == [proof['command']['argv'][0], '-u', '-m',
                                                    'scripts.recover_confirm_after_interrupt_v62'],
                             'Dispatched receipt must match this exact controller and review')
                return base.digest(path)
            base.require(row.get('status') == 'starting', 'Launcher failed before release')
        time.sleep(.05)
    raise TimeoutError('No completed launch handshake; no experiment was dispatched')


def verify_preserved(root, proof):
    out = root / 'craft_v6_out'
    prior_runner.verify_recovery_inputs(root, proof['prior_provenance'])
    base.verify_completed(out, proof['completed_before'])
    for path, expected in proof['source_sha256'].items():
        base.require(base.digest(root / path) == expected, 'Second recovery source changed')
    for path, expected in proof['evidence_sha256'].items():
        base.require(base.digest(root / path) == expected, 'Second recovery evidence changed')
    if 'review_sha256' in proof:
        base.require(base.digest(out / REVIEW) == proof['review_sha256'], 'Independent recovery review changed')
    if 'launch_receipt_sha256' in proof:
        base.require(base.digest(out / LAUNCH) == proof['launch_receipt_sha256'], 'Dispatched launch receipt changed')
    directory = out / EVIDENCE
    for name in ('v62_confirm_recovery_state.json', 'v62_confirm_recovery_stdout.log',
                 'v62_confirm_recovery_stderr.log'):
        base.require((out / name).read_bytes() == (directory / name).read_bytes(),
                     'Previous controller files must remain unchanged')
    base.require((out / 'v61_confirm_s2.log').read_bytes().startswith((directory / 'v61_confirm_s2.log').read_bytes()),
                 'Preserve every line of the interrupted s2 log')
    base.require((out / M4).read_bytes() == (directory / 's2_8320_M4.json').read_bytes(),
                 'Previously completed M4 checkpoint changed')
    validate_saved_m4(out / M4)


def proofs(root):
    out = root / 'craft_v6_out'
    directory = out / EVIDENCE
    old = base.read_json(directory / 'v62_confirm_recovery_state.json')
    event = base.read_json(directory / 'interruption.json')
    base.require(event.get('kind') == 'v62_confirm_processes_unavailable_interruption'
                 and old['status'] == 'running' and old['stage'] == 'confirm_s2'
                 and [v['stage'] for v in old['completed']] == ['confirm_s0', 'confirm_s1'],
                 'Only the preserved second interruption is supported')
    base.require(len(event.get('snapshots', [])) == 6, 'Complete initial interruption snapshots required')
    base.require({r['snapshot'] for r in event['snapshots']} == {
        'v62_confirm_recovery_state.json', 'v62_confirm_recovery_stdout.log',
        'v62_confirm_recovery_stderr.log', 'v61_confirm_s2.log',
        's2_8320_M4.json', 'benchmark_console_v0.txt'}, 'Exact initial snapshot set required')
    for row in event['snapshots']:
        base.require(row.get('present') is True and row['snapshot'] in SNAPSHOTS,
                     'Every initial interruption file must be preserved')
        saved = directory / row['snapshot']
        base.require(saved.stat().st_size == row['bytes'] and base.digest(saved) == row['sha256'],
                     'Interruption snapshot size or hash differs')
    base.require(old['started_utc'] == '2026-09-09T09:37:40Z'
                 and old['provenance']['design']['remaining_seconds'] == 84742.0902837,
                 'Original recovery clock/allowance changed')
    design = old['provenance']['design']
    base.require(all(design[k] == v for k, v in dict(b_max=6, g_units=2, n_units_per_lineage=3, num_envs=64).items()),
                 'Preserve B6/N9/G2; do not redesign from remaining time')
    for index in range(2):
        base.require(base.validate_lineage(out, index, 6, 2) == old['completed'][index]['artifacts'],
                     'Every completed world and checkpoint must remain exact')
    command = copy.deepcopy(old['provenance']['commands'][2])
    base.require(command == dict(stage='confirm_s2', lineage=2, argv=[
        'C:\\Python314\\python.exe', '-u', '-m', 'scripts.confirm_v61', '--lineage', '2',
        '--n-units', '3', '--b-max', '6', '--num-envs', '64', '--store-persists', '--g-units', '2', '--resume']),
        'Only the original s2 resume command is allowed')
    source = {p: base.digest(root / p) for p in SOURCE_PATHS}
    evidence = {(directory / p).relative_to(root).as_posix(): base.digest(directory / p) for p in SNAPSHOTS}
    probe = base.read_json(directory / 'process_lifetime_wmi_probe.json')
    completed_probe = base.read_json(directory / 'process_lifetime_wmi_probe_completed.json')
    base.require(probe['in_job'] is False and probe['pid'] == completed_probe['pid'] == 17288
                 and completed_probe['completed'] - probe['started'] >= 20,
                 'Completed detached WMI lifetime witness required')
    proof = dict(source_sha256=source, evidence_sha256=evidence,
                 prior_provenance=old['provenance'], completed_before=old['completed'], command=command,
                 original_recovery_started_utc=old['started_utc'],
                 original_recovery_allowance_seconds=design['remaining_seconds'],
                 absolute_deadline_utc=deadline_from(old).isoformat(), design=design,
                 cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS, poll_seconds=POLL_SECONDS,
                 handshake_seconds=HANDSHAKE_SECONDS,
                 preserved_m4_sha256=base.digest(out / M4))
    verify_preserved(root, proof)
    for relative in {*source, *evidence}:
        base.committed(root, root / relative)
    return proof


def preflight(root):
    out = root / 'craft_v6_out'
    base.require(not (out / STATE).exists() and not (out / (STATE + '.tmp')).exists(),
                 'Preserve the existing second recovery attempt')
    proof = proofs(root)
    base.require(not other_controllers(), 'Another experiment controller or child is still present')
    base.require(sorted(p.name for p in (out / 'v61_confirm_ckpt').glob('s2_*')) == ['s2_8320_M4.json']
                 and not (out / 'v61_confirm_s2.json').exists(), 'Unexpected additional s2 progress; review before dispatch')
    expected = [u for s in range(2) for u in base.read_json(out / f'v61_confirm_s{s}.json')['units']]
    base.require(base.read_json(out / 'v61_confirm.json') == dict(b_max=6, store_persists=True, units=expected)
                 and (out / 'v61_confirm.json').read_bytes() == (out / EVIDENCE / 'v61_confirm.json').read_bytes(),
                 'Preserve the complete six-world input aggregate')
    review = base.read_json(out / REVIEW)
    base.require(review.get('kind') == 'v62_second_recovery_independent_review' and review.get('passed') is True
                 and all(review.get(k) == v for k, v in proof.items()), 'Exact independent recovery review required')
    base.committed(root, out / REVIEW)
    return dict(**proof, review_sha256=base.digest(out / REVIEW))


def run_pipeline(root, proof):
    out = root / 'craft_v6_out'
    require_detached()
    proof = dict(proof, launch_receipt_sha256=await_dispatch(root, proof))
    base.require(not other_controllers(), 'Another experiment controller or child appeared before dispatch')
    absolute = datetime.fromisoformat(proof['absolute_deadline_utc'])
    initial_remaining = (absolute - utc_now()).total_seconds()
    base.require(initial_remaining > 0, 'Original absolute deadline has already passed')
    started = time.monotonic()
    state = dict(status='starting', controller_pid=os.getpid(), started_utc=utc_now().isoformat(),
                 provenance=proof, completed=copy.deepcopy(proof['completed_before']),
                 initial_remaining_seconds=initial_remaining, elapsed_seconds=0., detached_from_job=True)
    path = out / STATE
    base.save_state(path, state, create=True)

    def remaining():
        # A backward wall-clock jump cannot increase the launch-time allowance.
        value = min((absolute - utc_now()).total_seconds(), initial_remaining - (time.monotonic() - started))
        if value <= 0:
            raise TimeoutError('Original ARC3 recovery deadline exhausted')
        return value

    code = 1
    try:
        verify_preserved(root, proof)
        remaining()
        state.update(status='running', stage='confirm_s2', elapsed_seconds=time.monotonic() - started)
        base.save_state(path, state)
        print(f"Resuming s2 only, deadline {proof['absolute_deadline_utc']}, remaining {remaining():.3f}s", flush=True)
        run_child_until(proof['command']['argv'], root, remaining)
        verify_preserved(root, proof)
        artifacts = base.validate_lineage(out, 2, 6, 2)
        remaining()
        state['completed'].append(dict(stage='confirm_s2', artifacts=artifacts))
        expected = [u for s in range(3) for u in base.read_json(out / f'v61_confirm_s{s}.json')['units']]
        base.require(base.read_json(out / 'v61_confirm.json') == dict(b_max=6, store_persists=True, units=expected),
                     'Complete nine-world aggregate required')
        base.verify_completed(out, state['completed'])
        remaining()
        state.update(status='confirm_complete_review_required', merged_sha256=base.digest(out / 'v61_confirm.json'))
        code = 0
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        state.update(status='budget_exhausted', error=str(error))
    except KeyboardInterrupt:
        state.update(status='interrupted', error='Interrupted again; no automatic retry')
        code = 130
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        state.update(elapsed_seconds=time.monotonic() - started, stopped_utc=utc_now().isoformat())
        base.save_state(path, state)
        print(f"Stopped: {state['status']}", flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    proof = preflight(root)
    if args.dry_run:
        print(json.dumps(proof, indent=2, allow_nan=False))
    else:
        raise SystemExit(run_pipeline(root, proof))
