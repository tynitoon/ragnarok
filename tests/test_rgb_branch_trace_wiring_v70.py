"""Synthetic native-wiring checks and durable failure evidence; no ALE."""
from collections import deque
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import check_rgb_branch_trace_v70 as check
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.environments.rgb_branch_trace_v70 import _RGBTraceColdEngine
from ragnarok.environments.frame_budget_v70 import FrameBudget
from tests.test_rgb_branch_trace_v70 import engine, PixelNative


def synthetic(cls, cfg):
    obj = engine(cls, terminal=cfg.max_episode_frames, credit=0)
    obj.config = cfg
    obj._actions = {i:i for i in range(18)}
    obj._bootstrap_frames = 132
    obj._budget = FrameBudget(max_charged_frames=cfg.max_charged_frames)
    obj._budget.record_play(132,agent_steps=0)
    obj.accounting = obj._budget.report
    def cold_start(*,experimental):
        assert not experimental
        obj._budget.record_play(132,agent_steps=0)
        obj._ale = PixelNative(cfg.max_episode_frames)
        obj._frames = deque([obj._ale.getScreenRGB(),obj._ale.getScreenRGB()],maxlen=2)
        obj._last_action = 0
        obj._sticky_rng = random.Random(19)
    obj._cold_start = cold_start
    return obj


def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'OUTPUT',tmp_path/'result.json')
    monkeypatch.setattr(check,'preflight',lambda:{'proof':'synthetic'})
    monkeypatch.setattr(check,'_RGBTraceColdEngine',lambda cfg:synthetic(_RGBTraceColdEngine,cfg))
    monkeypatch.setattr(check,'_ColdPixelEngine',lambda cfg:synthetic(_ColdPixelEngine,cfg))


@pytest.mark.parametrize('warmup,limit,expected',[(128,108000,None),(127,509,1),(128,509,0)])
def test_independent_frame_witness_preserves_pixels_and_declared_call_counter(monkeypatch,tmp_path,warmup,limit,expected):
    setup(monkeypatch,tmp_path)
    created = []
    def create(name,factory):
        obj = factory();created.append(obj);return obj
    result = check.direct_case(create,'pong',0,warmup,limit,expected)
    assert result['passed'] and len(created)==3
    assert result['trace_frames']==([8,8] if expected is None else [expected,expected])
    assert created[0].accounting()['experiment_agent_steps']==2
    assert created[2].accounting()['experiment_agent_steps']==sum(result['trace_frames'])


def test_preflight_failure_does_not_create_attempt_or_native_engine(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'OUTPUT',tmp_path/'missing/result.json')
    def reject():raise ValueError('review absent')
    monkeypatch.setattr(check,'preflight',reject)
    monkeypatch.setattr(check,'_RGBTraceColdEngine',lambda cfg:pytest.fail('opened engine'))
    with pytest.raises(ValueError,match='review absent'):check.run()
    assert list(tmp_path.iterdir())==[]


def test_second_constructor_failure_keeps_first_cost_and_forbids_retry(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    def broken(cfg):raise RuntimeError('constructor without return')
    monkeypatch.setattr(check,'_ColdPixelEngine',broken)
    with pytest.raises(RuntimeError,match='constructor'):check.run()
    record = json.loads((tmp_path/'result_attempt.json').read_text())
    assert record['known_charged_frames']==132 and record['constructor_count']==2
    assert record['constructor_without_return']=='pong:18700:endpoint_witness'
    assert record['accounting_complete'] is False and len(record['engine_reports'])==1
    with pytest.raises(ValueError,match='no retry'):check.run()


def test_reversed_frame_witness_is_detected_with_all_three_paid_ledgers(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    original = check.frame_witness_pair
    monkeypatch.setattr(check,'frame_witness_pair',lambda engine:tuple(reversed(original(engine))))
    with pytest.raises(ValueError,match='Per-frame'):check.run()
    record = json.loads((tmp_path/'result_attempt.json').read_text())
    assert record['known_charged_frames']==3*(132+512+136)
    assert len(record['engine_reports'])==3 and record['constructor_count']==3
    assert record['completed_cases']==[]


def test_complete_synthetic_wiring_has_26_paid_reports_and_one_declared_counter_difference(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path)
    class Pool:
        def __init__(self,configs):
            self.engine = synthetic(_RGBTraceColdEngine,configs[0])
            self._processes = [SimpleNamespace(is_alive=lambda:False)]
        def _one(self,index,command):raise ValueError('Unknown RGB trace command')
        def step(self,actions):return [self.engine.step(actions[0])]
        def branch_pair_trace(self,*args,**kwargs):return self.engine.branch_pair_trace(*args,**kwargs)
        def report(self):return dict(accounting=self.engine.accounting(),failure=self.engine._trace_failure)
        def close(self):pass
    monkeypatch.setattr(check,'RGBTracePool',Pool)
    result = check.run()
    assert result['passed'] and result['constructor_count']==len(result['engine_reports'])==26
    assert len(result['cases'])==9 and all(c['passed'] for c in result['cases'])
    assert result['charged_frames']==sum(r['accounting']['charged_frames'] for r in result['engine_reports'])
    assert result['charged_frames']<=52000 and result['accounting_complete']
    assert json.loads((tmp_path/'result_attempt.json').read_text())['status']=='complete'
