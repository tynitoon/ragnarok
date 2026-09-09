"""Synthetic RGB-only trace fidelity and paid failure; no ROM is constructed."""
from collections import deque
import random
from types import SimpleNamespace
import numpy as np
import pytest
from ragnarok.environments import rgb_branch_trace_v70 as trace
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine
from ragnarok.environments.frame_budget_v70 import FrameBudget, BudgetExceeded


class PixelNative:
    """No RAM/reward/lives accessor, and an internal RNG included in snapshots."""
    def __init__(self, terminal=999):
        self.frame = self.value = 0
        self.terminal = terminal
        self.rng = random.Random(73)
    def game_over(self): return self.frame >= self.terminal
    def getEpisodeFrameNumber(self): return self.frame
    def getScreenRGB(self):
        rgb = np.zeros((210,160,3),dtype=np.uint8)
        rgb[:,self.frame%160] = self.value
        return rgb
    def act(self, action):
        self.frame += 1
        self.value = (self.value+1+action+self.rng.randrange(7))%256
        return 99999  # Must not affect any output.
    def cloneState(self, include_rng):
        assert include_rng is True
        return self.frame,self.value,self.rng.getstate()
    def restoreState(self, saved):
        self.frame,self.value = saved[:2]
        self.rng.setstate(saved[2])


def engine(cls=trace._RGBTraceColdEngine, terminal=999, credit=600):
    obj = cls.__new__(cls)
    obj.config = SimpleNamespace(mode='lab',frame_skip=4,sticky_probability=.25,max_snapshots=64)
    obj._ale = PixelNative(terminal)
    obj._frames = deque([obj._ale.getScreenRGB(),obj._ale.getScreenRGB()],maxlen=2)
    obj._sticky_rng = random.Random(19)
    obj._last_action = 0
    obj._saves = {}
    obj._actions = {0:0,1:1}
    obj._budget = FrameBudget(max_charged_frames=5000)
    obj._budget.record_play(credit,agent_steps=0)
    obj._recording = None
    obj._trace_failure = None
    return obj


@pytest.mark.parametrize('terminal', [999,5,4])
def test_traces_are_each_actual_raw_and_pooled_frame_with_exact_restoration(terminal):
    subject = engine(terminal=terminal)
    witness = engine(_ColdPixelEngine,terminal=terminal)
    ordinary = engine(_ColdPixelEngine,terminal=terminal)
    for obj in (subject,witness,ordinary):obj.step(1)
    original = subject.observe()
    charge = subject._budget.charged_frames
    traces = subject.branch_pair_trace(1,control=0,frames=4)
    expected = witness.branch_pair(1,control=0,frames=4)
    count = min(4,terminal-4)
    assert subject._budget.report() == witness._budget.report()
    assert subject._budget.charged_frames == charge+120+2*count
    assert np.array_equal(subject.observe(),original)
    for action,t,e in zip((1,0),traces,expected,strict=True):
        replay = engine(_ColdPixelEngine,terminal=terminal)
        replay.step(1)
        assert t.raw.shape == t.pooled.shape == (count,210,160,3)
        assert t.raw.dtype == t.pooled.dtype == np.uint8
        assert np.array_equal(t.endpoint,e)
        for i in range(count):
            pooled = replay._advance(action,1,experimental=True)
            assert np.array_equal(t.raw[i],replay.observe(raw=True))
            assert np.array_equal(t.pooled[i],pooled)
    # Returned arrays do not alias the restored engine's image history.
    traces[0].endpoint[:] = 255
    if count:traces[0].raw[:] = 255
    assert np.array_equal(subject.observe(),original)
    if terminal==999:
        for action in (0,1,0):
            assert np.array_equal(subject.step(action),ordinary.step(action))
            assert subject._sticky_rng.getstate() == ordinary._sticky_rng.getstate()
            assert subject._ale.cloneState(True) == ordinary._ale.cloneState(True)


def test_denied_budget_has_no_trace_act_restore_or_failure_latch():
    subject = engine(credit=132)
    before = subject._budget.report()
    with pytest.raises(BudgetExceeded):subject.branch_pair_trace(1,frames=8)
    assert subject._budget.report()==before and subject._ale.frame==0
    assert subject._recording is None and subject._trace_failure is None
    subject.step(1)


def test_recorder_failure_stays_paid_restores_anchor_and_blocks_later_actions(monkeypatch):
    subject = engine()
    original = subject.observe()
    before = subject._budget.charged_frames
    def broken(self,frame):raise RuntimeError('synthetic trace allocation failure')
    monkeypatch.setattr(trace._RecordingDeque,'append',broken)
    with pytest.raises(RuntimeError,match='allocation failure'):subject.branch_pair_trace(1)
    assert subject._budget.charged_frames==before+61
    assert subject._budget.report()['restores']==1
    assert subject._ale.frame==0 and np.array_equal(subject.observe(),original)
    assert subject._recording is None and type(subject._frames) is deque
    for call in (lambda:subject.step(1),lambda:subject.branch_pair(1),lambda:subject.clone(),
                 lambda:subject.restore('unknown'),lambda:subject.branch_pair_trace(1)):
        with pytest.raises(ValueError,match='stopped'):call()
    assert subject._budget.charged_frames==before+61
