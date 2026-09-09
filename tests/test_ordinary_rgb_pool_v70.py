"""Closed RPC and actual Windows workers with synthetic native objects only."""
import numpy as np
import pytest

from ragnarok.environments import ordinary_rgb_pool_v70 as pool_module
from ragnarok.environments import ordinary_rgb_trace_v70 as trace_module
from ragnarok.environments.emu_pool import EmulatorConfig,EmulatorPool
from ragnarok.environments.rgb_trace_pool_v70 import RGBTracePool
from tests.test_ordinary_rgb_trace_v70 import subject
from tests.test_rgb_trace_pool_v70 import Connection


def serve(monkeypatch,messages,engine):
    monkeypatch.setattr(pool_module,'_OrdinaryRGBTraceEngine',lambda config:engine)
    engine.accounting=engine._budget.report
    connection=Connection([*messages,('close',())])
    pool_module._serve_ordinary_rgb(connection,None)
    assert connection.closed and connection.sent[-1]==(True,None)
    return connection.sent[1:-1]


def test_worker_returns_only_rgb_traces_and_rejects_oracle_and_native_state(monkeypatch):
    engine=subject()
    forbidden=['getRAM','reward','lives','game_over','cold_loads','getEpisodeFrameNumber',
               'clone','restore','release','_capture','archive_save']
    replies=serve(monkeypatch,[(name,()) for name in forbidden]+[
        ('observe_pair',()),('step',(1,)),('branch_pair_trace',(1,0,4)),('report',())],engine)
    assert all(not ok and 'Unknown' in error[1] for ok,error in replies[:len(forbidden)])
    assert replies[-4][1].shape==(2,210,160,3)
    assert trace_module.validate_trace(replies[-3][1])==4
    assert replies[-2][0] and len(replies[-2][1])==2
    assert replies[-1][1]['accounting']['charged_frames']==732


def test_paid_worker_failure_is_reported_without_a_following_action(monkeypatch):
    engine=subject()
    def broken():raise MemoryError('synthetic screen allocation')
    engine._ale.getScreenRGB=broken
    replies=serve(monkeypatch,[('step',(1,)),('step',(1,)),('report',())],engine)
    assert not replies[0][0] and 'stopped' in replies[1][1][1]
    assert replies[-1][0] and replies[-1][1]['accounting']['charged_frames']==601
    assert replies[-1][1]['failure']['accounting_complete'] is False


def test_parent_rejects_malformed_paid_reply_and_latches_actions_but_not_reports(monkeypatch):
    obj=pool_module.OrdinaryRGBPool.__new__(pool_module.OrdinaryRGBPool)
    obj._stopped=False
    sends=[]
    def malformed(self,actions):
        sends.append(list(actions))
        return [np.zeros((210,160,3),np.uint8)]
    monkeypatch.setattr(RGBTracePool,'step',malformed)
    monkeypatch.setattr(EmulatorPool,'_one',lambda self,index,command,*args:(index,command))
    with pytest.raises(ValueError):obj.step([1])
    for call in (lambda:obj.step([0]),lambda:obj.branch_pair_trace(1),lambda:obj.branch_pair(1)):
        with pytest.raises(ValueError,match='stopped'):call()
    assert sends==[[1]] and obj.report()==(0,'report')


def synthetic_worker(connection,config):
    def create(_):
        engine=subject()
        engine.accounting=engine._budget.report
        return engine
    pool_module._OrdinaryRGBTraceEngine=create
    pool_module._serve_ordinary_rgb(connection,config)


class SyntheticOrdinaryPool(pool_module.OrdinaryRGBPool):
    _worker_target=staticmethod(synthetic_worker)


def test_real_spawn_preserves_start_raw_pool_branch_fidelity_and_closes_workers():
    configs=[EmulatorConfig(game='pong',seed=i) for i in range(2)]
    with SyntheticOrdinaryPool(configs) as pool:
        processes=list(pool._processes)
        starts=[pool.observe_pair(i) for i in range(2)]
        traces=pool.step([1,0])
        for i,trace in enumerate(traces):
            assert trace_module.validate_trace(trace)==4
            np.testing.assert_array_equal(trace.start_pair,starts[i])
            np.testing.assert_array_equal(trace_module.endpoint_pair(trace),pool.observe_pair(i))
        before=pool.observe_pair(0)
        branches=pool.branch_pair_trace(1,frames=8)
        assert all(len(t.raw)==8 for t in branches)
        np.testing.assert_array_equal(pool.observe_pair(0),before)
        assert pool.report(0)['accounting']['charged_frames']==740
        assert pool.report(1)['accounting']['charged_frames']==604
        assert all(trace_module.validate_trace(t)==4 for t in pool.step([0,1]))
    assert all(not p.is_alive() for p in processes)
