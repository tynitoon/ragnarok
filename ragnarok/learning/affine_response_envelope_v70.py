"""Generic seven-parameter RGB logistic calibration, with no environment rules."""
import numpy as np

PARAMETERS = 7
DAMPING = 7 / 4


def _finite_array(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != np.float64 or not np.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite float64 array of shape {shape}')


def validate_problem(features, labels, weight):
    if not isinstance(features, np.ndarray) or features.ndim != 2 or not len(features):
        raise ValueError('Nonempty pixel feature matrix required')
    count = len(features)
    _finite_array(features, (count, PARAMETERS), 'features')
    _finite_array(labels, (count,), 'labels')
    _finite_array(weight, (count,), 'weights')
    if (np.any((features[:, :6] < 0) | (features[:, :6] > 1)) or not np.all(features[:, 6] == 1)
            or not np.all((labels == 0) | (labels == 1)) or not np.all(weight > 0)
            or abs(float(weight.sum()) - 1.) > 1e-12):
        raise ValueError('Bounded RGB, unit bias, binary labels and normalized positive weights required')


def pixel_features(past):
    if past.dtype != np.uint8 or past.ndim != 5 or past.shape[1] != 2 or past.shape[-1] != 3:
        raise ValueError('Two ordered uint8 RGB images per case required')
    rgb = past.transpose(0, 2, 3, 1, 4).reshape(-1, 6).astype(np.float64) / 255
    return np.column_stack((rgb, np.ones(len(rgb), dtype=np.float64)))


def weighted_labels(targets):
    if targets.dtype != bool or targets.ndim != 3 or not targets.size:
        raise ValueError('Nonempty boolean case-height-width targets required')
    flat = targets.reshape(len(targets), -1)
    weight = np.empty(flat.shape, dtype=np.float64)
    for index, labels in enumerate(flat):
        positive, negative = int(labels.sum()), int((~labels).sum())
        if positive and negative:
            weight[index] = np.where(labels, .5 / positive, .5 / negative)
        else:
            weight[index].fill(1 / len(labels))
    weight /= len(flat)
    return flat.reshape(-1).astype(np.float64), weight.reshape(-1)


def probabilities(logits):
    values = np.asarray(logits)
    if values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError('Finite float64 logits required')
    return np.exp(-np.logaddexp(0., -values))


def objective(logits, labels, weight):
    # Signed logits avoid subtracting large nearly equal positive numbers.
    losses = np.logaddexp(0., np.where(labels == 1, -logits, logits))
    value = float(weight @ losses)
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite logistic objective')
    return value


def gradient(features, probability, labels, weight):
    value = features.T @ (weight * (probability - labels))
    if value.shape != (PARAMETERS,) or not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite or malformed logistic gradient')
    return value


def hessian(features, probability, weight):
    curvature = weight * probability * (1 - probability)
    value = features.T @ (curvature[:, None] * features)
    if value.shape != (PARAMETERS, PARAMETERS) or not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite or malformed logistic Hessian')
    return value


def newton_candidate(coefficients, first, second):
    _finite_array(coefficients, (PARAMETERS,), 'coefficients')
    _finite_array(first, (PARAMETERS,), 'gradient')
    _finite_array(second, (PARAMETERS, PARAMETERS), 'Hessian')
    delta = np.linalg.solve(second + DAMPING * np.eye(PARAMETERS), -first)
    candidate = coefficients + delta
    if candidate.dtype != np.float64 or not np.isfinite(candidate).all():
        raise FloatingPointError('Nonfinite Newton coefficients')
    return candidate


def adam_candidate(coefficients, first, first_moment, second_moment, step):
    for name, value in (('coefficients', coefficients), ('gradient', first),
                        ('first moment', first_moment), ('second moment', second_moment)):
        _finite_array(value, (PARAMETERS,), name)
    if np.any(second_moment < 0):
        raise ValueError('Adam second moments cannot be negative')
    if step < 1:
        raise ValueError('Adam updates are one-based')
    norm = float(np.linalg.norm(first))
    if not np.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    clipped = first * min(1., 1. / (norm + 1e-6))
    next_first = .9 * first_moment + .1 * clipped
    next_second = .999 * second_moment + .001 * clipped * clipped
    denominator = np.sqrt(next_second / (1 - .999 ** step)) + 1e-8
    candidate = coefficients - (.0003 / (1 - .9 ** step)) * next_first / denominator
    if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in (candidate, next_first, next_second)):
        raise FloatingPointError('Nonfinite Adam state or coefficients')
    return candidate, next_first, next_second
