"""Ordered motion representational/gradient mechanics, synthetic RGB only."""
import torch
import pytest

from ragnarok.learning.ordered_motion_parser_v70 import OrderedMotionParser
from ragnarok.learning.parser_v70 import max_pool_rgb
from ragnarok.learning.pixel_prediction_v70 import prediction_terms
from ragnarok.learning.gradient_guard_v70 import admit_adam_gradients
from ragnarok.learning.archived_pixel_diagnostics_v70 import weight_sha256
from tests.test_motion_observation_alias_v70 import sprite


torch.set_num_threads(1)


def parser():
    torch.manual_seed(764)
    with torch.device('cpu'):return OrderedMotionParser()


def test_only_motion_can_distinguish_reversed_rgb_order_with_the_same_pooled_source():
    model=parser()
    with torch.no_grad():
        right=model(sprite(4),sprite(5));left=model(sprite(5),sprite(4))
    for name in ('embedding','foreground_logits','controllability_logits','solidity_logits'):
        torch.testing.assert_close(getattr(right,name),getattr(left,name),rtol=0,atol=0)
    assert not torch.equal(right.displacement,left.displacement)
    # Distinguishable is not a claim that either untrained field is correct.
    assert torch.isfinite(right.displacement).all() and torch.isfinite(left.displacement).all()


def test_effect_hypothesis_still_changes_only_displacement():
    model=parser();unknown=torch.zeros((1,12));hypothesis=unknown.clone();hypothesis[0,3]=1
    with torch.no_grad():
        a=model(sprite(4),sprite(5),unknown);b=model(sprite(4),sprite(5),hypothesis)
    for name in ('embedding','foreground_logits','controllability_logits','solidity_logits'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    assert not torch.equal(a.displacement,b.displacement)


def test_prediction_backward_reaches_ordered_branch_without_updating_weights_or_inputs():
    model=parser();before=weight_sha256(model)
    previous,current=sprite(4),sprite(5);saved_previous,saved_current=previous.clone(),current.clone()
    output=model(previous,current)
    terms=prediction_terms(max_pool_rgb(previous,current),max_pool_rgb(sprite(8),sprite(9)),output)
    (terms['image_l1']+terms['foreground_bce']).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.ordered_motion.parameters())
    admission=admit_adam_gradients(model,max_norm=1.)
    assert admission.global_norm_after<=1.000001 and weight_sha256(model)==before
    assert torch.equal(previous,saved_previous) and torch.equal(current,saved_current)
    model.zero_grad(set_to_none=True)


def test_pooled_only_calls_cannot_silently_reach_the_new_displacement_head():
    model=parser()
    with pytest.raises(ValueError):model(max_pool_rgb(sprite(4),sprite(5)),torch.zeros((1,12)))
    with pytest.raises(TypeError):model(sprite(4))
    with pytest.raises(ValueError):model(sprite(4),sprite(5),torch.tensor([1]))


@pytest.mark.parametrize('shape',[(1,3,8,8),(2,3,18,22)])
def test_native_and_half_pyramid_preserve_spatial_shape_for_an_ordered_pair(shape):
    model=parser();previous=torch.zeros(shape,dtype=torch.uint8);current=previous.clone()
    current[:,:,3:5,3:5]=255
    with torch.no_grad():output=model(previous,current)
    for name,channels in (('embedding',32),('foreground_logits',1),('displacement',2),('controllability_logits',1),('solidity_logits',1)):
        value=getattr(output,name)
        assert value.shape==(shape[0],channels,*shape[-2:]) and torch.isfinite(value).all()
