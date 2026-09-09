from pathlib import Path
import numpy as np
import pytest
from ragnarok.evaluation.mspacman_fruit_v70 import fruit_render_boxes,fruit_support


FIXTURE=Path(__file__).resolve().parents[1]/'craft_v6_out/v70_mspacman_calibration/coverage_development_v1'


def frame(index):
    start=(index-1)//256*256+1
    with np.load(FIXTURE/f'frames_{start:05d}_{start+255:05d}.npz',allow_pickle=False) as z:
        return z['ram'][(index-1)%256].copy(),z['rgb'][(index-1)%256].copy()


def test_actual_alternating_visibility_and_prior_snapshot():
    a,_=frame(1313);b,_=frame(1314);c,rgb=frame(1315)
    assert fruit_render_boxes(b,a)==[]
    assert [o.box for o in fruit_render_boxes(c,b)]==[(62,49,8,10)]
    expected=np.all(rgb[:171]==(184,50,50),axis=2)
    assert np.array_equal(fruit_support(c,b),expected)


def test_second_type_and_pointer_carry():
    a,_=frame(5405);b,rgb=frame(5406)
    assert [o.box for o in fruit_render_boxes(b,a)]==[(152,100,6,8)]
    assert np.array_equal(fruit_support(b,a),np.all(rgb[:171]==(184,50,50),axis=2))
    broken=a.copy();broken[51]^=1
    with pytest.raises(ValueError,match='pointer'):
        fruit_render_boxes(b,broken)


def test_absent_fruit_cannot_be_invented_from_logical_coordinates():
    a,_=frame(1313)
    assert a[11]>0 and a[17]>0
    assert fruit_render_boxes(a)==[]
    assert not fruit_support(a).any()
