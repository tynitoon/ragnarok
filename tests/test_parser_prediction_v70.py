"""Synthetic CPU geometry, autograd and representation checks, no Atari input."""
import torch
import pytest
from ragnarok.learning.parser_v70 import CounterfactualParser, max_pool_rgb
from ragnarok.learning.pixel_prediction_v70 import forward_warp_rgb, motion_foreground_target, prediction_terms


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sprite():
    previous=torch.zeros((1,3,16,24),device='cpu')
    previous[:,:,5:9,7:11]=1
    following=torch.zeros_like(previous)
    following[:,:,5:9,8:12]=1
    return previous,following


def test_forward_coordinates_move_sprite_right_and_keep_uncovered_pixels_in_loss():
    previous,following=sprite()
    flow=torch.zeros((1,2,16,24),device='cpu');flow[:,0]=1
    reconstructed,coverage=forward_warp_rgb(previous,flow)
    torch.testing.assert_close(reconstructed,following,rtol=0,atol=0)
    assert coverage[:,:,:,0].sum()==0 and (coverage[:,:,:,1:]==1).all()
    huge=torch.full_like(flow,1e30)
    empty,coverage=forward_warp_rgb(previous,huge)
    assert empty.sum()==0 and coverage.sum()==0
    assert (empty-following).abs().mean()>0


def test_prediction_gradient_points_toward_correct_forward_motion():
    previous,following=sprite()
    dx=torch.tensor(.25,requires_grad=True,device='cpu')
    flow=torch.stack((dx.expand(16,24),torch.zeros((16,24))),dim=0)[None]
    reconstructed,_=forward_warp_rgb(previous,flow)
    loss=(reconstructed-following).abs().mean()
    loss.backward()
    assert torch.isfinite(dx.grad) and dx.grad<0
    exact,_=forward_warp_rgb(previous,torch.stack((torch.ones((16,24)),torch.zeros((16,24))))[None])
    assert (exact-following).abs().mean()<loss.detach()


def test_motion_target_uses_source_position_and_has_no_target_gradient():
    previous,following=sprite()
    flow=torch.zeros((1,2,16,24),requires_grad=True)
    with torch.no_grad(): flow[:,0,5:9,7:11]=1
    target=motion_foreground_target(previous,following,flow)
    # The vacated left edge belongs to the source sprite, the new right edge does not.
    assert (target[:,:,5:9,7]>.9).all() and (target[:,:,:,11]==0).all()
    assert not target.requires_grad and target.min()>=0 and target.max()<=1
    assert motion_foreground_target(previous,previous,torch.zeros_like(flow)).sum()==0


def test_motion_signal_alone_does_not_certify_an_object_mask():
    previous,following=sprite()
    global_motion=torch.zeros((1,2,16,24));global_motion[:,0]=1
    target=motion_foreground_target(previous,following,global_motion)
    # A wrongly global field also "explains" the newly covered background.
    assert (target[:,:,5:9,11]>.9).all()
    # A flat overlapping interior is ambiguous, not certified foreground here.
    assert (target[:,:,5:9,8:11]==0).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_extreme_finite_displacements_give_no_outside_foreground_evidence(dtype):
    previous=torch.zeros((1,3,2,3),dtype=dtype)
    following=torch.ones_like(previous)
    for direction in (-1,1):
        field=torch.full((1,2,2,3),direction*torch.finfo(dtype).max,dtype=dtype)
        target=motion_foreground_target(previous,following,field)
        assert torch.isfinite(target).all() and target.sum()==0


def test_one_network_handles_resolutions_and_effects_only_condition_displacement():
    torch.manual_seed(700)
    model=CounterfactualParser().cpu().eval()
    parameters=sum(p.numel() for p in model.parameters())
    for height,width in ((16,24),(17,25),(210,160)):
        rgb=torch.randint(256,(1,3,height,width),dtype=torch.uint8,device='cpu')
        zero=model(rgb)
        changed=model(rgb,torch.ones((1,12),device='cpu'))
        assert zero.embedding.shape==(1,32,height,width)
        assert zero.displacement.shape==(1,2,height,width)
        for key in ('embedding','foreground_logits','controllability_logits','solidity_logits'):
            torch.testing.assert_close(getattr(zero,key),getattr(changed,key),rtol=0,atol=0)
        assert not torch.equal(zero.displacement,changed.displacement)
        assert sum(p.numel() for p in model.parameters())==parameters
    assert all(p.device.type=='cpu' for p in model.parameters())


def test_unweighted_prediction_terms_backpropagate_into_network_without_teacher():
    torch.manual_seed(701)
    model=CounterfactualParser().cpu()
    previous,following=sprite()
    previous=(previous*255).to(torch.uint8);following=(following*255).to(torch.uint8)
    output=model(max_pool_rgb(previous,previous))
    terms=prediction_terms(previous,following,output)
    loss=terms['image_l1']+terms['foreground_bce']
    loss.backward()
    gradients=[p.grad for p in model.pyramid.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)
    assert model.displacement[-1].weight.grad.abs().sum()>0
    assert not terms['foreground_target'].requires_grad
    assert model.controllability.weight.grad is None and model.solidity.weight.grad is None


def test_pixel_input_and_conditioning_reject_nonvisual_or_nonfinite_shapes():
    model=CounterfactualParser().cpu()
    with pytest.raises(ValueError,match='pooled RGB'): model(torch.zeros((1,128),dtype=torch.uint8))
    rgb=torch.zeros((1,3,16,24),dtype=torch.uint8)
    with pytest.raises(ValueError,match='uint8'): model(rgb.float())
    with pytest.raises(ValueError,match='effect'): model(rgb,torch.zeros((1,18)))
    with pytest.raises(ValueError,match='finite'): model(rgb,torch.full((1,12),float('nan')))
    with pytest.raises(ValueError,match='equal'): max_pool_rgb(rgb,rgb[:,:,:,:-1])
    torch.testing.assert_close(max_pool_rgb(rgb,torch.full_like(rgb,17)),torch.full_like(rgb,17))
