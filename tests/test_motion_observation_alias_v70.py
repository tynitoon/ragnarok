"""Constructive ambiguity witness for the frozen pooled-only motion input.

Synthetic target labels are used only to exhibit two possible futures. They
never enter the model. This does not estimate an Atari ambiguity/error rate.
"""
import torch

from ragnarok.learning.parser_v70 import CounterfactualParser,max_pool_rgb


def sprite(x):
    image=torch.zeros((1,3,16,16),dtype=torch.uint8)
    image[:,:,6:8,x:x+2]=255
    return image


def test_opposite_observed_motion_has_same_pooled_input_but_incompatible_future():
    torch.set_num_threads(1)
    # Right: positions4->5, then6,7,8,9. Left:5->4, then3,2,1,0.
    right_source=max_pool_rgb(sprite(4),sprite(5))
    left_source=max_pool_rgb(sprite(5),sprite(4))
    right_future=max_pool_rgb(sprite(8),sprite(9))
    left_future=max_pool_rgb(sprite(1),sprite(0))
    assert torch.equal(right_source,left_source)
    assert not torch.equal(right_future,left_future)
    torch.manual_seed(764)
    with torch.device('cpu'):model=CounterfactualParser()
    effect=torch.zeros((1,12))
    with torch.no_grad():
        a=model(right_source,effect);b=model(left_source,effect)
    for name in ('embedding','foreground_logits','displacement','controllability_logits','solidity_logits'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    # Triangle inequality: any deterministic image prediction shared by both
    # inputs has mean L1 across the two targets >= half their L1 separation.
    separation=(right_future.float()/255-left_future.float()/255).abs().mean()
    assert float(separation)==0.046875
    assert float(separation/2)==0.0234375


def test_ordered_own_observations_keep_the_directional_information_without_a_label():
    right=torch.cat((sprite(4),sprite(5)),dim=1)
    left=torch.cat((sprite(5),sprite(4)),dim=1)
    assert not torch.equal(right,left)
    # Both are just the available RGB pair; no direction, button, RAM or reward
    # annotation is appended to distinguish them.
    assert right.shape==left.shape==(1,6,16,16) and right.dtype==left.dtype==torch.uint8
