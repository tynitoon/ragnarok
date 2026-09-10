"""Synthetic launch failure/race checks; no OS process or GPU is started."""
import json
import subprocess
import sys

import pytest

from scripts import launch_detached_confirm_v62 as launch


def setup(tmp_path, monkeypatch):
    out = tmp_path / 'craft_v6_out'; out.mkdir()
    proof = dict(command=dict(argv=[sys.executable]), source_sha256={},
                 review_sha256='review', absolute_deadline_utc='2026-09-10T09:10:02+00:00')
    monkeypatch.setattr(launch.controller, 'require_detached', lambda: None)
    monkeypatch.setattr(launch.controller, 'preflight', lambda root: proof)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    events = []
    class Child:
        pid = 4242
        def poll(self): return None
        def terminate(self): events.append('terminate')
        def wait(self, timeout): events.append(('wait', timeout)); return -1
    def spawn(argv, **kwargs):
        events.append((argv, kwargs))
        assert json.loads((out / launch.LAUNCH).read_text())['status'] == 'starting'
        kwargs['stdout'].write(b'child stdout')
        kwargs['stderr'].write(b'child stderr')
        return Child()
    monkeypatch.setattr(launch.subprocess, 'Popen', spawn)
    return out, events


def test_hidden_exclusive_dispatch_keeps_receipt_and_logs(tmp_path, monkeypatch):
    out, events = setup(tmp_path, monkeypatch)
    assert launch.launch(tmp_path) == 4242
    row = json.loads((out / launch.LAUNCH).read_text())
    assert row['status'] == 'dispatched' and row['controller_pid'] == 4242
    assert events[0][1]['creationflags'] == subprocess.CREATE_NO_WINDOW
    assert (out / launch.STDOUT).read_bytes() == b'child stdout'
    before = (out / launch.LAUNCH).read_bytes()
    with pytest.raises(ValueError): launch.launch(tmp_path)
    assert (out / launch.LAUNCH).read_bytes() == before and len(events) == 1


@pytest.mark.parametrize('guard', ['job', 'preflight', 'cuda', 'old_log'])
def test_guards_precede_receipt_and_dispatch(tmp_path, monkeypatch, guard):
    out, events = setup(tmp_path, monkeypatch)
    def refused(*a): raise ValueError('guard')
    if guard == 'job': monkeypatch.setattr(launch.controller, 'require_detached', refused)
    if guard == 'preflight': monkeypatch.setattr(launch.controller, 'preflight', refused)
    if guard == 'cuda': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    if guard == 'old_log': (out / launch.STDOUT).write_bytes(b'preserved')
    with pytest.raises(ValueError): launch.launch(tmp_path)
    assert not events and not (out / launch.LAUNCH).exists()


def test_dispatch_failure_is_recorded_without_retry(tmp_path, monkeypatch):
    out, events = setup(tmp_path, monkeypatch)
    def failed(*a, **k): raise OSError('simulated spawn failure')
    monkeypatch.setattr(launch.subprocess, 'Popen', failed)
    with pytest.raises(OSError): launch.launch(tmp_path)
    assert json.loads((out / launch.LAUNCH).read_text())['status'] == 'failed'
    assert (out / launch.STDOUT).exists() and (out / launch.STDERR).exists()
    with pytest.raises(ValueError): launch.launch(tmp_path)


@pytest.mark.parametrize('published', [False, True])
def test_late_publication_failure_never_kills_the_deadline_supervisor(tmp_path, monkeypatch, published):
    out, events = setup(tmp_path, monkeypatch)
    saved = launch.controller.base.save_state
    def fail(path, row, create=False):
        if path.name == launch.LAUNCH and row['status'] == 'dispatched':
            if published: saved(path, row, create=create)
            raise OSError('failure around handshake publication')
        return saved(path, row, create=create)
    monkeypatch.setattr(launch.controller.base, 'save_state', fail)
    with pytest.raises(OSError): launch.launch(tmp_path)
    assert len(events) == 1 and isinstance(events[0], tuple)
    row = json.loads((out / launch.LAUNCH).read_text())
    assert row['status'] == ('dispatched' if published else 'starting')
    failure = json.loads((out / launch.LAUNCH_ERROR).read_text())
    assert failure['controller_pid'] == 4242
    # Pending release means controller timeout before GPU; published release
    # preserves the independent supervisor instead of orphaning its child.
