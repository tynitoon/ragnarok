"""Synthetic ordinary/branch/terminal histories and monotone charges; no emulator."""
from collections import deque
import random
from types import SimpleNamespace
import numpy as np
import pytest
from ragnarok.evaluation.restorable_ram_history_v70 import _RestorableTripleRAMTap,_ScorerHistoryColdEngine
from ragnarok.environments.frame_budget_v70 import FrameBudget


class Native:
    def __init__(self,terminal=999):
        self.frame=0;self.ram=np.zeros(128,dtype=np.uint8);self.terminal=terminal
    def getRAM(self):return self.ram
    def getEpisodeFrameNumber(self):return self.frame
    def getScreenRGB(self):return np.full((210,160,3),self.ram[0],dtype=np.uint8)
    def game_over(self):return self.frame>=self.terminal
    def act(self,action):
        self.frame+=1;self.ram[:]=(int(self.ram[0])+1+action)%256;return 0.
    def cloneState(self,include_rng):
        assert include_rng is True
        return self.frame,self.ram.copy()
    def restoreState(self,state):self.frame=state[0];self.ram=state[1].copy()


def engine(terminal=999):
    native=Native(terminal);events=[]
    obj=_ScorerHistoryColdEngine.__new__(_ScorerHistoryColdEngine)
    obj.config=SimpleNamespace(mode='lab',frame_skip=4,sticky_probability=.25,max_snapshots=64)
    obj._ale=_RestorableTripleRAMTap(native,lambda tap:events.append(tap.frame_numbers))
    obj._frames=deque([native.getScreenRGB(),native.getScreenRGB()],maxlen=2)
    obj._sticky_rng=random.Random(19);obj._last_action=0;obj._saves={};obj._actions={0:0,1:1}
    obj._budget=FrameBudget(max_charged_frames=5000);obj._budget.record_play(600,agent_steps=0)
    return obj,events


def same_history(a,b):
    assert all(np.array_equal(x,y) for x,y in zip(a[:3],b[:3])) and a[3:]==b[3:]


def test_branch_restores_all_three_ram_and_preserves_ordinary_rgb_rng_trajectory():
    subject,events=engine();control,_=engine()
    assert np.array_equal(subject.step(1),control.step(1))
    history=subject._ale.export_history();charge=subject._budget.charged_frames
    intervention,noop=subject.branch_pair(1,control=0,frames=4)
    assert not np.array_equal(intervention,noop)
    same_history(subject._ale.export_history(),history)
    assert subject._budget.charged_frames==charge+128 and subject._budget.report()['restores']==2
    for action in (0,1,0):
        assert np.array_equal(subject.step(action),control.step(action))
        same_history(subject._ale.export_history(),control._ale.export_history())
    assert len(events)==24  # 16ordinary acts plus8branch acts, restores add no acts.


def test_terminal_truncation_keeps_real_branch_cost_and_anchor_history():
    subject,_=engine(terminal=5);subject.step(1)
    saved=subject._ale.export_history();charge=subject._budget.charged_frames
    subject.branch_pair(1,control=0,frames=4)
    same_history(subject._ale.export_history(),saved)
    assert subject._budget.charged_frames==charge+122
    subject.step(0)
    assert subject._ale.frame_numbers==(3,4,5)


def test_restore_does_not_alias_saved_history_and_cold_first_act_is_explicit():
    subject,_=engine();saved=subject._capture();subject.step(1);subject._restore(saved)
    assert subject._ale.frame_numbers==(0,0,0) and subject._ale.recorded_actions==0
    subject._ale.act(0)
    assert subject._ale.frame_numbers==(0,0,1) and subject._ale.cold_history_fallback
    subject._ale.previous[0]=99
    assert saved[4][1][0]==0


@pytest.mark.parametrize('change',['current','frame','early_duplicate'])
def test_inconsistent_saved_history_is_rejected(change):
    subject,_=engine();saved=list(subject._ale.export_history())
    if change=='current':saved[2][0]=1
    elif change=='frame':saved[3]=(0,0,1)
    else:saved[0][0]=1
    with pytest.raises(ValueError,match='does not match'):subject._ale.import_history(tuple(saved))


def test_raw_native_restore_still_cannot_bypass_scorer_history():
    subject,_=engine();state=subject._ale.native.cloneState(include_rng=True);subject.step(1)
    subject._ale.native.restoreState(state)
    with pytest.raises(RuntimeError,match='outside the capture tap'):subject._ale.act(1)


@pytest.mark.parametrize('branch',[False,True])
def test_decoder_failure_keeps_the_executed_frame_paid_and_does_not_continue(branch):
    subject,_=engine();charge=subject._budget.charged_frames
    def broken(tap):raise RuntimeError('Synthetic decoder failure after native act')
    subject._ale._on_frame=broken
    with pytest.raises(RuntimeError,match='decoder failure'):
        if branch:subject.branch_pair(1,control=0,frames=4)
        else:subject.step(1)
    report=subject._budget.report()
    assert subject._budget.charged_frames==charge+(61 if branch else 1)
    assert report['restores']==(1 if branch else 0)
    assert subject._ale.frame_numbers==((0,0,0) if branch else (0,0,1))
    assert np.array_equal(subject.observe(raw=True),subject._ale.native.getScreenRGB())
