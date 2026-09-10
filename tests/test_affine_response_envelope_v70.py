"""Independent mathematical fixtures; no stored observations or native model fitting."""
import numpy as np
import pytest

from ragnarok.learning import affine_response_envelope_v70 as affine


def _problem():
    features = np.array([[0., .2, .8, 0., .2, .8, 1.],
                         [1., .7, .1, 1., .7, .1, 1.],
                         [.3, .1, .5, .3, .1, .5, 1.],
                         [.8, .4, .9, .8, .4, .9, 1.]], dtype=np.float64)
    return features, np.array([0., 1., 0., 1.]), np.array([.2, .3, .4, .1])


def test_input_order_is_rgb_then_rgb_without_coordinates():
    past = np.arange(2 * 2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 2, 3, 4, 3)
    features = affine.pixel_features(past)
    assert features.shape == (24, 7) and features.dtype == np.float64
    for index, (case, y, x) in enumerate(np.ndindex(2, 3, 4)):
        np.testing.assert_array_equal(features[index, :6], past[case, :, y, x].reshape(6) / 255.)
    assert np.all(features[:, 6] == 1)


def test_balanced_weights_keep_each_anchor_and_both_empty_cases():
    targets = np.zeros((12, 2, 3), bool)
    targets[:10, 0, 0] = True
    labels, weights = affine.weighted_labels(targets)
    np.testing.assert_allclose(weights.reshape(12, -1).sum(1), 1 / 12, atol=1e-15)
    assert weights.sum() == pytest.approx(1.)
    assert weights[labels == 1].sum() == pytest.approx(5 / 12)
    logits = np.full(labels.shape, np.log(5 / 7), dtype=np.float64)
    bias_gradient = np.sum(weights * (affine.probabilities(logits) - labels))
    assert abs(bias_gradient) < 1e-15


def test_gradient_and_hessian_match_finite_differences_on_the_original_objective():
    features, labels, weight = _problem()
    coefficients = np.array([.2, -.1, .3, .1, -.4, .2, -.05])
    probability = affine.probabilities(features @ coefficients)
    actual_gradient = affine.gradient(features, probability, labels, weight)
    actual_hessian = affine.hessian(features, probability, weight)
    eps = 1e-5
    for index in range(7):
        shift = np.eye(7)[index] * eps
        numeric_gradient = (affine.objective(features @ (coefficients + shift), labels, weight)
                            - affine.objective(features @ (coefficients - shift), labels, weight)) / (2 * eps)
        assert actual_gradient[index] == pytest.approx(numeric_gradient, abs=2e-11)
        plus = affine.gradient(features, affine.probabilities(features @ (coefficients + shift)), labels, weight)
        minus = affine.gradient(features, affine.probabilities(features @ (coefficients - shift)), labels, weight)
        np.testing.assert_allclose(actual_hessian[:, index], (plus - minus) / (2 * eps), atol=2e-11, rtol=0)


def test_fixed_damping_handles_redundant_channels_and_satisfies_descent_bound():
    features, labels, weight = _problem()
    coefficients = np.zeros(7, np.float64)
    probability = affine.probabilities(features @ coefficients)
    first = affine.gradient(features, probability, labels, weight)
    second = affine.hessian(features, probability, weight)
    assert np.linalg.matrix_rank(second, tol=1e-12) < 7
    assert np.linalg.eigvalsh(second).min() >= -1e-12
    assert np.linalg.eigvalsh(second).max() <= 7 / 4
    candidate = affine.newton_candidate(coefficients, first, second)
    delta = candidate - coefficients
    before = affine.objective(features @ coefficients, labels, weight)
    after = affine.objective(features @ candidate, labels, weight)
    assert after <= before - delta @ second @ delta - 7 / 8 * (delta @ delta) + 1e-12
    assert after < before


def test_channel_permutation_permutates_the_generic_newton_update():
    features, labels, weight = _problem()
    permutation = np.array([2, 0, 1, 5, 3, 4, 6])
    zero = np.zeros(7)
    probability = affine.probabilities(features @ zero)
    first = affine.gradient(features, probability, labels, weight)
    second = affine.hessian(features, probability, weight)
    candidate = affine.newton_candidate(zero, first, second)
    swapped = features[:, permutation]
    other = affine.newton_candidate(zero, affine.gradient(swapped, probability, labels, weight),
                                   affine.hessian(swapped, probability, weight))
    np.testing.assert_allclose(other, candidate[permutation], atol=1e-15, rtol=0)


def test_float64_adam_with_clipping_matches_installed_torch_on_two_fixed_steps():
    import torch
    coefficients = np.linspace(-.2, .2, 7)
    first_moment, second_moment = np.zeros(7), np.zeros(7)
    reference = torch.nn.Parameter(torch.from_numpy(coefficients.copy()))
    optimizer = torch.optim.Adam([reference], lr=.0003, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., amsgrad=False, foreach=False, fused=False)
    for step, derivative in enumerate((np.linspace(-3., 3., 7), np.linspace(.02, -.04, 7)), start=1):
        coefficients, first_moment, second_moment = affine.adam_candidate(
            coefficients, derivative, first_moment, second_moment, step)
        reference.grad = torch.from_numpy(derivative.copy())
        torch.nn.utils.clip_grad_norm_([reference], 1., error_if_nonfinite=True)
        optimizer.step()
        np.testing.assert_allclose(coefficients, reference.detach().numpy(), atol=1e-14, rtol=0)


def test_extreme_finite_logits_keep_a_finite_nonnegative_objective():
    logits = np.array([-1000., 1000., -1000., 1000.])
    labels = np.array([0., 1., 1., 0.])
    weight = np.full(4, .25)
    assert np.isfinite(affine.probabilities(logits)).all()
    assert affine.objective(logits, labels, weight) == pytest.approx(500.)


@pytest.mark.parametrize('bad', [np.zeros((7, 1)), np.zeros(7, np.float32), np.full(7, np.nan)])
def test_candidates_reject_broadcast_dtype_and_nonfinite_contract_errors(bad):
    with pytest.raises(ValueError):
        affine.newton_candidate(bad, np.zeros(7), np.zeros((7, 7)))
    with pytest.raises(ValueError):
        affine.adam_candidate(bad, np.zeros(7), np.zeros(7), np.zeros(7), 1)


def test_problem_boundary_rejects_nonbinary_labels_and_broken_rgb_bound():
    features, labels, weights = _problem()
    affine.validate_problem(features, labels, weights)
    labels[0] = .5
    with pytest.raises(ValueError):
        affine.validate_problem(features, labels, weights)
    labels[0] = 0
    features[0, 0] = 1.01
    with pytest.raises(ValueError):
        affine.validate_problem(features, labels, weights)
