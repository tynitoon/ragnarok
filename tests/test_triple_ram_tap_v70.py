"""History alignment and state discontinuity tests without an emulator."""
import numpy as np
import pytest
from ragnarok.evaluation.triple_ram_tap_v70 import _TripleRAMTap


class Native:
    def __init__(self):
        self.frame=0
        self.ram=np.zeros(128,dtype=np.uint8)
    def getRAM(self):return self.ram
    def getEpisodeFrameNumber(self):return self.frame
    def act(self,action):
        self.frame+=1
        self.ram[:]=self.frame
        return 1.25


def test_first_raw_pair_has_explicit_bootstrap_fallback():
    tap=_TripleRAMTap(Native())
    assert tap.act(3)==1.25
    assert tap.frame_numbers==(0,0,1) and tap.cold_history_fallback
    assert (tap.before_previous[0],tap.previous[0],tap.current[0])==(0,0,1)


def test_repeat_four_retains_the_predecessor_of_each_source_frame():
    tap=_TripleRAMTap(Native())
    for _ in range(4):tap.act(3)
    assert tap.frame_numbers==(2,3,4) and not tap.cold_history_fallback
    assert (tap.before_previous[0],tap.previous[0],tap.current[0])==(2,3,4)
    assert not np.shares_memory(tap.previous,tap.native.ram)
    saved=tap.previous.copy();tap.act(4)
    assert np.array_equal(saved,tap.before_previous)


@pytest.mark.parametrize('change',['frame','ram'])
def test_unannounced_restore_is_rejected_before_the_next_action(change):
    native=Native();tap=_TripleRAMTap(native);tap.act(1)
    if change=='frame':native.frame=0
    else:native.ram[10]=99
    with pytest.raises(RuntimeError,match='outside the capture tap'):tap.act(1)
    assert tap.recorded_actions==1


def test_new_cold_native_gets_separate_history():
    old=_TripleRAMTap(Native())
    for _ in range(4):old.act(0)
    new=_TripleRAMTap(Native());new.act(0)
    assert new.frame_numbers==(0,0,1) and old.frame_numbers==(2,3,4)
    assert np.all(new.before_previous==0)


def test_native_multi_frame_skip_is_rejected():
    native=Native();original=native.act
    def two_frames(action):original(action);return original(action)
    native.act=two_frames;tap=_TripleRAMTap(native)
    with pytest.raises(RuntimeError,match='exactly one'):tap.act(0)


def test_hot_installation_cannot_claim_cold_fallback():
    native=Native();native.frame=5
    with pytest.raises(ValueError,match='cold frame zero'):_TripleRAMTap(native)
