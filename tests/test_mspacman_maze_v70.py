from pathlib import Path
import numpy as np
from ragnarok.evaluation.mspacman_maze_visible_v70 import MsPacmanPixelJudge,decode_visible_ram

FIXTURE=Path(__file__).resolve().parents[1]/'craft_v6_out/v70_mspacman_calibration/coverage_development_v1'


def frame(index):
    start=(index-1)//256*256+1
    with np.load(FIXTURE/f'frames_{start:05d}_{start+255:05d}.npz',allow_pickle=False) as z:
        return z['rgb'][(index-1)%256].copy(),z['ram'][(index-1)%256].copy()


def test_actual_ghost_wrap_has_two_visible_pieces():
    _,previous=frame(3797);_,current=frame(3798)
    boxes=[o.box for o in decode_visible_ram(current,previous) if o.kind=='ghost']
    assert len(boxes)==3
    assert (0,53,1,8) in boxes
    assert (153,51,7,10) in boxes


def test_gray_flash_uses_only_walls_seen_in_past_rgb():
    judge=MsPacmanPixelJudge()
    first,_=frame(1);flashing,_=frame(4329)
    judge.step(first)
    labels=judge.step(flashing)
    assert sum(o.kind=='ghost' for o in labels)==2
    assert not any(o.kind in ('pill','power_pill') for o in labels)
    # An unrelated gray ghost outside the learned walls must survive.
    modified=first.copy();modified[80:90,76:84]=(214,214,214)
    assert any(o.kind=='ghost' and o.box==(76,80,8,10) for o in judge.step(modified))


def test_second_palette_learns_its_own_pill_registry():
    judge=MsPacmanPixelJudge()
    first,_=frame(1);blue,_=frame(7891)
    judge.step(first);labels=judge.step(blue)
    assert sum(o.kind=='pill' for o in labels)>=120
    assert sum(o.kind=='power_pill' for o in labels)==2
    assert judge.layout==(101,111,228)


def test_power_pill_keeps_only_its_uncovered_rows():
    _,previous=frame(418);_,current=frame(419)
    boxes=[o.box for o in decode_visible_ram(current,previous) if o.kind=='power_pill']
    assert (8,19,4,3) in boxes
    assert (8,147,4,7) in boxes
