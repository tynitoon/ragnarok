"""Synthetic native acts only; no ROM or real emulator is constructed."""
from collections import deque
from dataclasses import fields,replace

import numpy as np
import pytest

from ragnarok.environments import ordinary_rgb_trace_v70 as ordinary
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.environments.frame_budget_v70 import BudgetExceeded
from tests.test_rgb_branch_trace_v70 import engine,PixelNative


def subject(terminal=999,credit=600):
    obj=engine(ordinary._OrdinaryRGBTraceEngine,terminal=terminal,credit=credit)
    obj._ordinary_starts=None
    obj._bootstrap_frames=132
    return obj


@pytest.mark.parametrize('terminal',[1,3,999])
def test_trace_matches_unchanged_ordinary_engine_and_counts_only_real_acts(terminal):
    obj=subject(terminal)
    witness=engine(_ColdPixelEngine,terminal=terminal)
    start=np.stack(tuple(obj._frames))
    result=obj.step_trace(1)
    expected=witness.step(1)
    assert ordinary.validate_trace(result)==min(terminal,4)
    np.testing.assert_array_equal(result.start_pair,start)
    np.testing.assert_array_equal(result.endpoint,expected)
    assert obj._budget.report()==witness._budget.report()
    assert obj._sticky_rng.getstate()==witness._sticky_rng.getstate()
    assert obj._ale.cloneState(True)==witness._ale.cloneState(True)
    np.testing.assert_array_equal(ordinary.endpoint_pair(result),np.stack(tuple(obj._frames)))
    result.start_pair[:]=255;result.raw[:]=255;result.endpoint[:]=255
    np.testing.assert_array_equal(obj.observe(),expected)


def test_cold_start_screen_is_pixels_not_a_native_reset_annotation():
    obj=subject(terminal=1)
    chronology=ordinary.RGBChronology(obj.observe(raw=True))
    first=obj.step_trace(1)
    assert chronology.accept(first).end_frame==1
    before=obj._budget.charged_frames
    def cold(*,experimental):
        assert experimental is False
        obj._budget.record_play(132,agent_steps=0)
        obj._ale=PixelNative()
        obj._frames=deque([obj._ale.getScreenRGB(),obj._ale.getScreenRGB()],maxlen=2)
        obj._last_action=0
    obj._cold_start=cold
    result=obj.step_trace(0)
    update=chronology.accept(result)
    assert update.visual_discontinuity and update.epoch==1 and update.start_frame==0 and update.end_frame==4
    assert obj._budget.charged_frames-before==136
    assert set(f.name for f in fields(result))=={'start_pair','raw','pooled','endpoint'}


def test_identical_start_pixels_and_pool_buffer_changes_do_not_prove_a_reset():
    start=np.zeros((2,3,4,3),np.uint8)
    start[0]=255  # Only the earlier pooling buffer changes, not the actual latest screen.
    trace=ordinary.OrdinaryRGBTrace(start,np.empty((0,3,4,3),np.uint8),np.empty((0,3,4,3),np.uint8),np.maximum(*start))
    chronology=ordinary.RGBChronology(start[-1])
    update=chronology.accept(trace)
    assert not update.visual_discontinuity and update.epoch==0 and update.end_frame==0
    np.testing.assert_array_equal(ordinary.endpoint_pair(trace),start)


def test_ordinary_trace_leaves_paired_branch_replay_unchanged():
    obj=subject()
    obj.step_trace(1)
    before=obj.observe()
    pair=obj.branch_pair_trace(1,frames=4)
    assert len(pair)==2 and obj._ordinary_starts is None and obj._recording is None
    np.testing.assert_array_equal(obj.observe(),before)
    assert ordinary.validate_trace(obj.step_trace(0))==4


def test_late_output_failure_preserves_charges_and_stops_further_actions(monkeypatch):
    obj=subject()
    before=obj._budget.charged_frames
    def fail(*args,**kwargs):raise MemoryError('synthetic final trace allocation')
    monkeypatch.setattr(ordinary,'OrdinaryRGBTrace',fail)
    with pytest.raises(MemoryError):obj.step_trace(1)
    assert obj._budget.charged_frames==before+4 and obj._ale.frame==4
    assert obj._recording is obj._ordinary_starts is None
    with pytest.raises(ValueError,match='stopped'):obj.step_trace(1)
    assert obj._trace_failure['accounting_complete'] is False


def test_budget_refusal_precedes_any_recorded_act_and_manual_sessions_are_refused():
    obj=subject(credit=4998)
    before=obj._budget.report()
    with pytest.raises(BudgetExceeded):obj.step_trace(1)
    assert obj._budget.report()==before and obj._ale.frame==0 and obj._trace_failure is None
    obj=subject()
    obj._saves['manual']=object()
    with pytest.raises(ValueError,match='manual snapshot'):obj.step_trace(1)
    assert obj._ale.frame==0


def test_bad_trace_or_allocation_failure_cannot_advance_the_pixel_chronology(monkeypatch):
    obj=subject()
    chronology=ordinary.RGBChronology(obj.observe(raw=True))
    trace=obj.step_trace(1)
    malformed=replace(trace,pooled=np.zeros_like(trace.pooled))
    with pytest.raises(ValueError,match='Pooled trace'):chronology.accept(malformed)
    assert chronology._frame==chronology._epoch==0
    def fail(*args,**kwargs):raise MemoryError('synthetic chronology result')
    monkeypatch.setattr(ordinary,'RGBChronologyStep',fail)
    with pytest.raises(MemoryError):chronology.accept(trace)
    assert chronology._frame==chronology._epoch==0
