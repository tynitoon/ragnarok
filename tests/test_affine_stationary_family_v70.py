"""Algebraic fixtures only: no learned coefficients, images or generation."""
from fractions import Fraction

import numpy as np
import pytest

from ragnarok.evaluation.affine_stationary_family_v70 import exact_family_bounds


def fraction(row):
    return Fraction(int(row['numerator']), int(row['denominator']))


def test_strict_separator_and_green_blue_extrema():
    coefficients = np.array([1., -.25, -.5, 1., -.25, -.5, -.5])
    result = exact_family_bounds(coefficients)
    assert result['certified']
    assert fraction(result['positive_min_logit']) == 1
    assert fraction(result['negative_max_logit']) == Fraction(-1, 2)
    coefficients[1] = 2.
    assert not exact_family_bounds(coefficients)['negative_strict']
    coefficients[1] = -.25
    coefficients[2] = 2.
    assert not exact_family_bounds(coefficients)['negative_strict']


def test_equality_is_not_a_strict_certificate_and_constants_cannot_separate():
    for bias in (-1., 0., 1.):
        result = exact_family_bounds(np.array([0., 0., 0., 0., 0., 0., bias]))
        assert not result['certified']
    result = exact_family_bounds(np.array([.5, 0., 0., .5, 0., 0., -1.]))
    assert fraction(result['positive_min_logit']) == 0 and not result['positive_strict']


def test_stored_float_bits_remain_exact_and_no_float_cancellation_changes_sign():
    values = np.array([2.**53, 0., 0., 1., 0., 0., -(2.**53)])
    result = exact_family_bounds(values)
    assert fraction(result['positive_min_logit']) == 1 and result['certified']
    assert values[0] + values[3] + values[6] == 0.  # This is why the certificate is explicitly algebraic.
    values[0], values[3], values[6] = .1, .2, -.3
    result = exact_family_bounds(values)
    expected = sum(Fraction.from_float(float(values[i])) for i in (0, 3, 6))
    assert fraction(result['positive_min_logit']) == expected


@pytest.mark.parametrize('bad', [np.zeros((7, 1)), np.zeros(7, np.float32), np.full(7, np.inf)])
def test_reject_malformed_coefficients(bad):
    with pytest.raises(ValueError): exact_family_bounds(bad)
