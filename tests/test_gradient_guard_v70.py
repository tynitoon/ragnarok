"""Gradient numerics and candidate-allocation failure; synthetic CPU only."""
import numpy as np
import pytest
import torch

from ragnarok.learning import gradient_guard_v70 as guard

torch.set_num_threads(1)


def test_extreme_finite_gradients_are_reduced_before_float32_adam_squares_overflow():
    model=torch.nn.Linear(3,2).float().cpu()
    before={k:v.clone() for k,v in model.state_dict().items()}
    for p in model.parameters():p.grad=torch.full_like(p,2.2714049e22)
    assert torch.isinf(model.weight.grad.square()).all()
    receipt=guard.admit_adam_gradients(model)
    assert receipt.global_norm_before>1e22 and receipt.clipped and receipt.scale<1e-22
    assert receipt.global_norm_after<=1+1e-6 and receipt.elements==8
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,foreach=False,fused=False)
    optimizer.step()  # One synthetic mechanics check, not an Atari learner.
    for p in model.parameters():
        assert torch.isfinite(p).all()
        assert all(torch.isfinite(v).all() for v in optimizer.state[p].values() if isinstance(v,torch.Tensor))
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_reviewed_real_parser_identity_outlier_is_admitted_without_overflowing_one_synthetic_adam_update():
    from tests.test_owned_pixel_objectives_v70 import owner
    from ragnarok.learning.parser_v70 import CounterfactualParser
    from ragnarok.learning.owned_perception_session_v70 import OwnedPerceptionSession
    torch.manual_seed(764)
    model=CounterfactualParser().cpu()
    unused,pool,_=owner(parser=model)
    session=OwnedPerceptionSession(unused._collector,model,workers=1)
    session.step([1],effects=[np.ones(12,np.float32)])
    batch=session.step([0],effects=[np.full(12,.25,np.float32)])
    loss=batch.observations[0].identity.loss
    loss.backward()
    largest=max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    assert largest>1e20 and any(torch.isinf(p.grad.square()).any() for p in model.parameters() if p.grad is not None)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    receipt=guard.admit_adam_gradients(model)
    assert receipt.clipped and receipt.global_norm_before>1e20 and receipt.global_norm_after<=1+1e-6
    for name,p in model.named_parameters():torch.testing.assert_close(p.detach(),before[name],rtol=0,atol=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,foreach=False,fused=False)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    assert any(not torch.equal(p,before[name]) for name,p in model.named_parameters())
    assert pool.report(0)['accounting']['charged_frames']==608
    session.close()


def test_max_float32_fields_and_tiny_allowed_bound_use_float64_scaling():
    model=torch.nn.Linear(2,1,bias=False).float().cpu()
    model.weight.grad=torch.full_like(model.weight,torch.finfo(torch.float32).max)
    receipt=guard.admit_adam_gradients(model,max_norm=1e-8)
    assert receipt.scale<torch.finfo(torch.float32).tiny
    assert torch.count_nonzero(model.weight.grad)==2
    assert receipt.global_norm_after==pytest.approx(1e-8,rel=1e-6)


def test_matches_independent_numpy_global_norm_and_preserves_unclipped_values():
    torch.manual_seed(782)
    model=torch.nn.Linear(7,4).float().cpu()
    for p in model.parameters():p.grad=torch.randn_like(p)
    original=[p.grad.numpy().copy() for p in model.parameters()]
    norm=np.linalg.norm(np.concatenate([a.ravel().astype(np.float64) for a in original]))
    receipt=guard.admit_adam_gradients(model)
    assert receipt.global_norm_before==pytest.approx(norm,rel=1e-14)
    for p,a in zip(model.parameters(),original,strict=True):
        np.testing.assert_allclose(p.grad.numpy(),a.astype(np.float64)/norm,rtol=1e-7,atol=1e-8)
    for p in model.parameters():p.grad=torch.full_like(p,.001)
    before=[p.grad.clone() for p in model.parameters()]
    receipt=guard.admit_adam_gradients(model)
    assert not receipt.clipped and receipt.scale==1
    for p,a in zip(model.parameters(),before,strict=True):torch.testing.assert_close(p.grad,a,rtol=0,atol=0)


@pytest.mark.parametrize('which',['gradient','parameter'])
def test_nonfinite_validation_leaves_every_original_gradient_unchanged(which):
    model=torch.nn.Linear(3,2).float().cpu()
    for p in model.parameters():p.grad=torch.ones_like(p)
    with torch.no_grad():
        if which=='gradient':model.bias.grad[0]=float('nan')
        else:model.bias[0]=float('inf')
    references=[p.grad for p in model.parameters()]
    with pytest.raises(ValueError,match='Finite'):guard.admit_adam_gradients(model)
    assert all(p.grad is old for p,old in zip(model.parameters(),references,strict=True))


@pytest.mark.parametrize('phase',['gradient','receipt'])
def test_late_candidate_or_receipt_allocation_failure_does_not_partially_replace_gradients(monkeypatch,phase):
    model=torch.nn.Linear(3,2).float().cpu()
    for p in model.parameters():p.grad=torch.full_like(p,1e20)
    references=[p.grad for p in model.parameters()]
    original=guard._scaled_gradient
    calls=[]
    def fail_second(gradient,scale):
        calls.append(1)
        if len(calls)==2:raise MemoryError('synthetic second gradient candidate')
        return original(gradient,scale)
    def fail_receipt(*args):raise MemoryError('synthetic receipt')
    if phase=='gradient':monkeypatch.setattr(guard,'_scaled_gradient',fail_second)
    else:monkeypatch.setattr(guard,'GradientAdmission',fail_receipt)
    with pytest.raises(MemoryError):guard.admit_adam_gradients(model)
    assert all(p.grad is old for p,old in zip(model.parameters(),references,strict=True))


def test_zero_gradient_is_not_mistaken_for_evidence_of_useful_learning():
    model=torch.nn.Linear(1,1).float().cpu()
    with pytest.raises(ValueError,match='No gradient'):guard.admit_adam_gradients(model)
    model.weight.grad=torch.zeros_like(model.weight)
    receipt=guard.admit_adam_gradients(model)
    assert receipt.global_norm_before==receipt.global_norm_after==0 and not receipt.clipped
    assert model.bias.grad is None
    for bound in (0,-1,2,float('inf'),float('nan'),True):
        with pytest.raises(ValueError,match='bound'):guard.admit_adam_gradients(model,max_norm=bound)
