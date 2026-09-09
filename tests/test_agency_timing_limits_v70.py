"""Possible timing ambiguities, with simulated RGB and no model/ME labels."""
import numpy as np

from ragnarok.environments.rgb_trace_difference_v70 import difference
from tests.test_rgb_trace_difference_v70 import clip


def image(a,b):
    value=np.zeros((16,24,3),np.uint8)
    value[2:4,a:a+2]=255
    value[10:12,b:b+2]=128
    return value


def branches(*,sticky_first):
    # Two persistent objects exist at the shared start. Within each simulated
    # frame the action moves A, then B follows A immediately. B is downstream
    # by the simulator's update rule, not by an annotation supplied to a model.
    anchor=image(4,10)
    def branch(button):
        a,b=4,10;frames=[];last=0
        for t in range(4):
            actual=last if sticky_first and t==0 else button
            before=a;a+=actual
            b+=a-before
            last=actual;frames.append(image(a,b))
        return clip(np.stack(frames),anchor)
    return branch(1),branch(0)


def region_first(result,rows):
    values=result.first_changed_frame[rows]
    changed=values[values>0]
    return int(changed.min()) if len(changed) else None


def test_direct_and_downstream_objects_can_both_diverge_in_first_observed_frame():
    result=difference(*branches(sticky_first=False),image_kind='raw')
    assert region_first(result,slice(2,4))==1
    assert region_first(result,slice(10,12))==1


def test_sticky_action_can_delay_direct_control_to_second_frame():
    result=difference(*branches(sticky_first=True),image_kind='raw')
    assert not result.changed[0].any()
    assert region_first(result,slice(2,4))==2
    assert region_first(result,slice(10,12))==2
