"""Judge-only algebra on frozen coefficients; never a learner input or target."""
from fractions import Fraction

import numpy as np


def exact_family_bounds(coefficients):
    if (not isinstance(coefficients, np.ndarray) or coefficients.shape != (7,)
            or coefficients.dtype != np.float64 or not np.isfinite(coefficients).all()):
        raise ValueError('Exactly seven finite float64 coefficients required')
    values = [Fraction.from_float(float(value)) for value in coefficients]
    red, green, blue = (values[i] + values[i + 3] for i in range(3))
    bias = values[6]
    positive_min = bias + red + min(Fraction(0), green)
    negative_max = bias + max(Fraction(0), green) + max(Fraction(0), blue)
    def representation(value):
        try:
            approximate = float(value)
        except OverflowError:
            approximate = None
        return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximate=approximate)
    return dict(positive_min_logit=representation(positive_min),
                negative_max_logit=representation(negative_max),
                positive_strict=positive_min > 0, negative_strict=negative_max < 0,
                certified=positive_min > 0 and negative_max < 0,
                scope='Exact affine function on the declared stationary family; not all floating-point kernel roundings')
