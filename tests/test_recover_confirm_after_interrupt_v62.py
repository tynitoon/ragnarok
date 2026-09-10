"""No model/process launch: fixed-deadline continuation and preservation checks."""
from datetime import datetime, timedelta, timezone
import json
import subprocess

import pytest

from scripts import recover_confirm_after_interrupt_v62 as run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def m4():
    return [dict(goal=g, tier=t, curve=[.5] * 7, rows=[1] * 6, demos=[0] * 6)
            for g, t in zip((5, 11, 1), (2, 3, 4))]


def setup(tmp_path, monkeypatch):
    out = tmp_path / 'craft_v6_out'; out.mkdir()
    preserved = out / 'old_controller.json'; preserved.write_bytes(b'old immutable controller')
    begin = datetime(2026, 9, 10, tzinfo=timezone.utc)
    clock = dict(wall=begin, mono=0.)
    monkeypatch.setattr(run, 'utc_now', lambda: clock['wall'])
    monkeypatch.setattr(run.time, 'monotonic', lambda: clock['mono'])
    monkeypatch.setattr(run, 'require_detached', lambda: None)
    monkeypatch.setattr(run, 'await_dispatch', lambda *a: 'launchhash')
    monkeypatch.setattr(run, 'other_controllers', lambda: [])
    monkeypatch.setattr(run, 'verify_preserved', lambda *a: None)
    monkeypatch.setattr(run.base, 'verify_completed', lambda *a: None)
    monkeypatch.setattr(run.base, 'validate_lineage', lambda *a: {'s2-complete': 'hash'})
    for i in range(3):
        write(out / f'v61_confirm_s{i}.json', {'units': [f's{i}u{j}' for j in range(3)]})
    merged = dict(b_max=6, store_persists=True, units=[f's{i}u{j}' for i in range(3) for j in range(3)])
    write(out / 'v61_confirm.json', merged)
    calls = []
    monkeypatch.setattr(run, 'run_child_until', lambda argv, root, remaining: calls.append((argv, remaining())))
    proof = dict(absolute_deadline_utc=(begin + timedelta(seconds=100)).isoformat(),
                 completed_before=[dict(stage='confirm_s0', artifacts={'old0': 'h0'}),
                                   dict(stage='confirm_s1', artifacts={'old1': 'h1'})],
                 command=dict(argv=['original-s2-only']))
    return out, preserved, clock, proof, calls


def test_deadline_keeps_original_start_and_rounds_down():
    old = dict(started_utc='2026-09-09T09:37:40Z',
               provenance=dict(design=dict(remaining_seconds=84742.0902837)))
    assert run.deadline_from(old) == datetime(2026, 9, 10, 9, 10, 2, tzinfo=timezone.utc)
    old['started_utc'] = '2026-09-09T09:37:40'
    with pytest.raises(ValueError, match='Timezone'):
        run.deadline_from(old)


@pytest.mark.parametrize('fault', ['partial', 'nan', 'wrong_goal', 'counter', 'corrupt'])
def test_incomplete_saved_arm_is_rejected(tmp_path, fault):
    path = tmp_path / 'm4.json'; rows = m4()
    if fault == 'partial': rows.pop()
    elif fault == 'nan': rows[1]['curve'][0] = float('nan')
    elif fault == 'wrong_goal': rows[0]['goal'] = 2
    elif fault == 'counter': rows[0]['rows'][0] = -1
    write(path, rows)
    if fault == 'corrupt': path.write_text('[{"goal":', encoding='utf-8')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        run.validate_saved_m4(path)


def test_valid_saved_arm(tmp_path):
    path = tmp_path / 'm4.json'; write(path, m4())
    assert run.validate_saved_m4(path) == m4()


def test_only_s2_runs_and_old_results_stay_preserved(tmp_path, monkeypatch):
    out, old, _, proof, calls = setup(tmp_path, monkeypatch)
    assert run.run_pipeline(tmp_path, proof) == 0
    state = run.base.read_json(out / run.STATE)
    assert state['status'] == 'confirm_complete_review_required'
    assert [r['stage'] for r in state['completed']] == ['confirm_s0', 'confirm_s1', 'confirm_s2']
    assert calls == [(['original-s2-only'], 100.)]
    assert old.read_bytes() == b'old immutable controller'
    before = (out / run.STATE).read_bytes()
    with pytest.raises(ValueError): run.run_pipeline(tmp_path, proof)
    assert (out / run.STATE).read_bytes() == before and len(calls) == 1


@pytest.mark.parametrize('failure,status', [(subprocess.CalledProcessError(1, ['s2']), 'failed'),
                                         (subprocess.TimeoutExpired(['s2'], 100), 'budget_exhausted'),
                                         (KeyboardInterrupt(), 'interrupted')])
def test_failure_stays_final_and_preserves_completed_prefix(tmp_path, monkeypatch, failure, status):
    out, old, _, proof, calls = setup(tmp_path, monkeypatch)
    def fail(*args):
        calls.append(args); raise failure
    monkeypatch.setattr(run, 'run_child_until', fail)
    assert run.run_pipeline(tmp_path, proof) != 0
    state = run.base.read_json(out / run.STATE)
    assert state['status'] == status and len(state['completed']) == 2 and len(calls) == 1
    assert old.read_bytes() == b'old immutable controller'


def test_backward_wall_clock_cannot_extend_budget(tmp_path, monkeypatch):
    out, _, clock, proof, calls = setup(tmp_path, monkeypatch)
    def spent(*args):
        clock['mono'] = 40.
        clock['wall'] -= timedelta(hours=1)
    monkeypatch.setattr(run, 'verify_preserved', spent)
    assert run.run_pipeline(tmp_path, proof) == 0
    assert calls == [(['original-s2-only'], 60.)]


def test_time_in_validation_consumes_the_original_window(tmp_path, monkeypatch):
    out, _, clock, proof, calls = setup(tmp_path, monkeypatch)
    def spent(*args): clock['mono'] = 101.
    monkeypatch.setattr(run, 'verify_preserved', spent)
    assert run.run_pipeline(tmp_path, proof) == 1
    assert not calls and run.base.read_json(out / run.STATE)['status'] == 'budget_exhausted'


def test_postpaid_preservation_failure_cannot_publish_complete(tmp_path, monkeypatch):
    out, _, _, proof, calls = setup(tmp_path, monkeypatch)
    checks = []
    def changing(*args):
        checks.append(1)
        if len(checks) > 1: raise ValueError('saved checkpoint changed')
    monkeypatch.setattr(run, 'verify_preserved', changing)
    assert run.run_pipeline(tmp_path, proof) == 1
    state = run.base.read_json(out / run.STATE)
    assert state['status'] == 'failed' and len(state['completed']) == 2 and len(calls) == 1


@pytest.mark.parametrize('reason', ['attached', 'duplicate', 'expired'])
def test_no_action_or_state_before_launch_guards(tmp_path, monkeypatch, reason):
    out, _, clock, proof, calls = setup(tmp_path, monkeypatch)
    if reason == 'attached':
        def refuse(): raise ValueError('attached to a transient job')
        monkeypatch.setattr(run, 'require_detached', refuse)
    elif reason == 'duplicate': monkeypatch.setattr(run, 'other_controllers', lambda: [{'pid': 42}])
    else: clock['wall'] += timedelta(seconds=101)
    with pytest.raises(ValueError): run.run_pipeline(tmp_path, proof)
    assert not calls and not (out / run.STATE).exists()


@pytest.mark.parametrize('spawn_seconds', [7., 71.])
def test_spawn_time_is_charged_and_cleanup_is_reserved(tmp_path, monkeypatch, spawn_seconds):
    clock = {'remaining': 100.}; waits = []; events = []
    class Child:
        pid = 42
        stopped = False
        def wait(self, timeout):
            waits.append(timeout)
            if self.stopped: return -1
            clock['remaining'] -= timeout
            raise subprocess.TimeoutExpired(['s2'], timeout)
        def poll(self): return -1 if self.stopped else None
        def terminate(self):
            events.append(clock['remaining']); self.stopped = True
        def kill(self): raise AssertionError('first termination succeeded')
    def spawn(*a, **k):
        clock['remaining'] -= spawn_seconds
        return Child()
    monkeypatch.setattr(run.subprocess, 'Popen', spawn)
    with pytest.raises(TimeoutError):
        run.run_child_until(['s2'], tmp_path, lambda: clock['remaining'])
    assert events == [min(30., 100. - spawn_seconds)]
    assert all(v <= run.POLL_SECONDS for v in waits[:-1])
    assert waits[-1] == 5


def test_forward_clock_change_stops_on_next_poll(tmp_path, monkeypatch):
    clock = {'remaining': 100.}; events = []
    class Child:
        pid = 42
        stopped = False
        def wait(self, timeout):
            if self.stopped: return -1
            clock['remaining'] = 29.
            raise subprocess.TimeoutExpired(['s2'], timeout)
        def poll(self): return None
        def terminate(self): self.stopped = True; events.append('terminate')
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: Child())
    with pytest.raises(TimeoutError):
        run.run_child_until(['s2'], tmp_path, lambda: clock['remaining'])
    assert events == ['terminate']


def test_reserve_is_required_before_any_spawn(tmp_path, monkeypatch):
    def forbidden(*a, **k): raise AssertionError('too late to dispatch')
    monkeypatch.setattr(run.subprocess, 'Popen', forbidden)
    with pytest.raises(TimeoutError, match='cleanup'):
        run.run_child_until(['s2'], tmp_path, lambda: 30.)


@pytest.mark.parametrize('fault', ['pending', 'failed', 'wrong_pid', 'wrong_review', None])
def test_handshake_requires_published_matching_receipt(tmp_path, monkeypatch, fault):
    begin = datetime(2026, 9, 10, tzinfo=timezone.utc)
    clock = {'mono': 0.}
    monkeypatch.setattr(run.time, 'monotonic', lambda: clock['mono'])
    monkeypatch.setattr(run.time, 'sleep', lambda dt: clock.update(mono=clock['mono'] + dt))
    monkeypatch.setattr(run, 'utc_now', lambda: begin)
    proof = dict(absolute_deadline_utc=(begin + timedelta(seconds=100)).isoformat(),
                 review_sha256='review', source_sha256={'source': 'hash'},
                 command=dict(argv=['original-python']))
    row = dict(kind='v62_detached_second_recovery_launch', status='dispatched',
               controller_pid=run.os.getpid(), review_sha256='review',
               source_sha256=proof['source_sha256'], absolute_deadline_utc=proof['absolute_deadline_utc'],
               cwd=str(tmp_path), argv=['original-python', '-u', '-m', 'scripts.recover_confirm_after_interrupt_v62'])
    if fault == 'pending': row['status'] = 'starting'
    if fault == 'failed': row['status'] = 'failed'
    if fault == 'wrong_pid': row['controller_pid'] = -1
    if fault == 'wrong_review': row['review_sha256'] = 'different'
    path = tmp_path / 'craft_v6_out' / run.LAUNCH
    write(path, row)
    if fault is None:
        assert run.await_dispatch(tmp_path, proof) == run.base.digest(path)
    else:
        with pytest.raises((TimeoutError, ValueError)):
            run.await_dispatch(tmp_path, proof)
    assert clock['mono'] < 10.1
    assert not (path.parent / run.STATE).exists()
