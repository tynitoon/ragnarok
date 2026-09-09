"""Constructive limits of pixel L1; no network, optimizer or native image."""
import torch

from ragnarok.learning.parser_v70 import max_pool_rgb
from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb
from tests.test_motion_observation_alias_v70 import sprite


def scene():
    source=max_pool_rgb(sprite(4),sprite(5)).float()/255
    target=max_pool_rgb(sprite(8),sprite(9)).float()/255
    return source,target


def shifted(source,dx):
    field=torch.stack((dx.expand(1,16,16),dx.expand(1,16,16)*0),dim=1)
    return forward_warp_rgb(source,field)[0]


def test_correct_translation_can_have_no_local_l1_direction_until_supports_overlap():
    torch.set_num_threads(1)
    source,target=scene()
    for value in (-.5,-.25,.25,.5):
        dx=torch.tensor(value,requires_grad=True)
        loss=(shifted(source,dx)-target).abs().mean()
        loss.backward()
        assert float(loss.detach())==.046875
        assert abs(float(dx.grad))<1e-8
    exact=(shifted(source,torch.tensor(4.))-target).abs().mean()
    assert float(exact)==0


def test_discarding_all_source_pixels_beats_stationary_error_without_explaining_motion():
    source,target=scene()
    stationary=(shifted(source,torch.tensor(0.))-target).abs().mean()
    discarded=shifted(source,torch.tensor(32.))
    assert torch.count_nonzero(discarded)==0
    discarded_loss=(discarded-target).abs().mean()
    assert float(stationary)==.046875 and float(discarded_loss)==.0234375
    assert discarded_loss<stationary
