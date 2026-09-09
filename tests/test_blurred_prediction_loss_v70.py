"""Loss geometry on synthetic RGB only; no network or optimizer updates."""
import pytest
import torch

from ragnarok.learning.blurred_prediction_loss_v70 import blurred_prediction_loss
from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb
from ragnarok.learning.parser_v70 import max_pool_rgb
from tests.test_motion_observation_alias_v70 import sprite

torch.set_num_threads(1)


@pytest.mark.parametrize('direction',[-1,1])
def test_local_blurred_gradient_points_toward_the_future_where_native_translation_l1_is_flat(direction):
    source=max_pool_rgb(sprite(4),sprite(5)).float()/255
    target=(max_pool_rgb(sprite(8),sprite(9)) if direction==1 else max_pool_rgb(sprite(0),sprite(1))).float()/255
    dx=torch.tensor(.25*direction,requires_grad=True)
    field=torch.stack((dx.expand(1,16,16),dx.expand(1,16,16)*0),dim=1)
    warped=forward_warp_rgb(source,field)[0]
    result=blurred_prediction_loss(warped,target)
    native_gradient=torch.autograd.grad(result['scale_l1'][1],dx,retain_graph=True)[0]
    gradient=torch.autograd.grad(result['loss'],dx)[0]
    assert abs(float(native_gradient))<1e-8
    assert float(gradient)*direction<0


def test_exact_reconstruction_has_zero_loss_at_every_scale_without_mutating_inputs():
    rgb=max_pool_rgb(sprite(4),sprite(5)).float()/255
    saved=rgb.clone();result=blurred_prediction_loss(rgb,rgb.clone())
    assert result['loss']==0 and all(value==0 for value in result['scale_l1'].values())
    assert torch.equal(rgb,saved)


def test_flat_background_cannot_hide_a_native_pixel_mismatch():
    rgb=torch.zeros((1,3,16,16));future=rgb.clone();future[:,:,8,8]=1
    result=blurred_prediction_loss(rgb,future)
    assert result['loss']>0 and result['scale_l1'][1]>0


def test_future_cannot_be_an_optimizable_target():
    rgb=torch.zeros((1,3,16,16),requires_grad=True)
    future=torch.ones_like(rgb,requires_grad=True)
    with pytest.raises(ValueError):blurred_prediction_loss(rgb,future)


@pytest.mark.parametrize('bad',[float('nan'),-1.,2.])
def test_nonfinite_or_out_of_domain_rgb_is_refused(bad):
    rgb=torch.zeros((1,3,16,16));other=rgb.clone();other[0,0,0,0]=bad
    with pytest.raises(ValueError,match='finite BCHW RGB'):blurred_prediction_loss(rgb,other)
