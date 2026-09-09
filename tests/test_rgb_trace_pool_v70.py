"""Synthetic native engine only, including actual spawn/pipe serialization."""
import numpy as np
import pytest
from ragnarok.environments import rgb_trace_pool_v70 as pool
from ragnarok.environments import rgb_branch_trace_v70 as trace
from ragnarok.environments.emu_pool import EmulatorPool, EmulatorConfig
from tests.test_rgb_branch_trace_v70 import engine


class Connection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False
    def recv(self):return next(self.messages)
    def send(self, value):self.sent.append(value)
    def close(self):self.closed = True


def serve(monkeypatch, messages, subject):
    monkeypatch.setattr(pool,'_RGBTraceColdEngine',lambda config:subject)
    # The fake object has the same paid ledger but does not construct a cold ROM.
    subject.accounting = subject._budget.report
    connection = Connection([*messages,('close',())])
    pool._serve_rgb_trace(connection,None)
    assert connection.closed and connection.sent[-1] == (True,None)
    return connection.sent[1:-1]


def test_worker_rejects_native_state_oracle_and_snapshot_commands(monkeypatch):
    subject = engine()
    forbidden = ['getRAM','reward','lives','_capture','clone','restore','release']
    replies = serve(monkeypatch,[(name,()) for name in forbidden]+[
        ('step',(1,)),('branch_pair_trace',(1,0,4)),('report',())],subject)
    assert all(not ok and error[0]=='ValueError' and 'Unknown' in error[1]
               for ok,error in replies[:len(forbidden)])
    assert replies[-3][0] and replies[-3][1].shape == (210,160,3)
    assert replies[-2][0] and all(t.raw.shape==(4,210,160,3) for t in replies[-2][1])
    assert replies[-1] == (True,dict(accounting=subject._budget.report(),failure=None))
    assert subject._budget.charged_frames == 732


def test_paid_trace_failure_keeps_report_rpc_and_forbids_following_act(monkeypatch):
    subject = engine()
    def broken(self,frame):raise RuntimeError('synthetic allocation failure')
    monkeypatch.setattr(trace._RecordingDeque,'append',broken)
    replies = serve(monkeypatch,[('branch_pair_trace',(1,0,4)),('step',(1,)),('report',())],subject)
    assert replies[0][0] is False and replies[0][1][0]=='ValueError'
    assert replies[1][0] is False and 'stopped' in replies[1][1][1]
    assert replies[2][0] is True and replies[2][1]['failure']['error_type']=='RuntimeError'
    assert replies[2][1]['accounting']['charged_frames'] == 661


def test_pool_cannot_resend_after_failed_batch_but_can_request_reports(monkeypatch):
    obj = pool.RGBTracePool.__new__(pool.RGBTracePool)
    obj._stopped = False
    sends = []
    def failed(self, actions):
        sends.append(list(actions))
        raise ValueError('paid worker failure')
    monkeypatch.setattr(EmulatorPool,'step',failed)
    monkeypatch.setattr(EmulatorPool,'_one',lambda self,index,command,*args:(index,command))
    with pytest.raises(ValueError,match='worker'):obj.step([1])
    for call in (lambda:obj.step([0]),lambda:obj.branch_pair_trace(0),lambda:obj.branch_pair(0)):
        with pytest.raises(ValueError,match='stopped'):call()
    assert sends == [[1]] and obj.report() == (0,'report')


def test_ordinary_post_act_rgb_failure_is_also_paid_and_permanently_stopped():
    subject = engine()
    def broken():raise RuntimeError('synthetic screen failure')
    subject._ale.getScreenRGB = broken
    with pytest.raises(RuntimeError,match='screen failure'):subject.step(1)
    assert subject._budget.charged_frames == 601 and subject._ale.frame == 1
    with pytest.raises(ValueError,match='stopped'):subject.step(1)


def test_native_invariant_failure_with_no_ledger_delta_still_stops_and_marks_unknown_cost():
    subject = engine()
    def invalid_act(action):subject._ale.frame += 2
    subject._ale.act = invalid_act
    with pytest.raises(RuntimeError,match='Expected one real'):subject.step(1)
    assert subject._ale.frame == 2 and subject._budget.charged_frames == 600
    assert subject._trace_failure['accounting_complete'] is False
    with pytest.raises(ValueError,match='stopped'):subject.branch_pair_trace(1)


def _synthetic_worker(connection, config):
    # Runs in the child process. The real engine constructor is never called.
    def create(_config):
        subject = engine()
        subject.accounting = subject._budget.report
        return subject
    pool._RGBTraceColdEngine = create
    pool._serve_rgb_trace(connection, config)


class SyntheticTracePool(pool.RGBTracePool):
    _worker_target = staticmethod(_synthetic_worker)


def test_real_spawn_and_pipe_preserve_trace_arrays_reports_and_cleanup():
    configs = [EmulatorConfig(game='pong',seed=i) for i in range(2)]
    with SyntheticTracePool(configs) as subject:
        processes = list(subject._processes)
        images = subject.step([1,0])
        assert len(images)==2 and all(rgb.shape==(210,160,3) for rgb in images)
        a,b = subject.branch_pair_trace(1,control=0,frames=8,index=0)
        assert a.raw.shape==b.raw.shape==(8,210,160,3)
        assert np.array_equal(a.endpoint,a.pooled[-1]) and np.array_equal(b.endpoint,b.pooled[-1])
        assert subject.report(0)['accounting']['charged_frames']==740
        assert subject.report(1)['accounting']['charged_frames']==604
        with pytest.raises(ValueError,match='Unknown'):subject._one(0,'getRAM')
        assert len(subject.step([0,1]))==2
    assert all(not p.is_alive() for p in processes)
