"""Auxiliary prediction of observed response coordinates, not physical objects."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResponseEnvelope(nn.Module):
    """Image-only FCN; a constant input remains spatially constant at the edges."""
    def __init__(self):
        super().__init__()
        layers = []
        channels = 6
        for dilation in (1, 2, 4, 8, 16):
            layers.extend((nn.Conv2d(channels, 16, 3, padding=dilation, dilation=dilation,
                                     padding_mode='replicate'), nn.SiLU()))
            channels = 16
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, past):
        return self.head(self.body(past))


def anchor_balanced_bce(logits, target):
    """Equal anchor weight, then equal observed-positive/negative mass when both exist."""
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    axes = tuple(range(1, target.ndim))
    positives = target.sum(dim=axes)
    negatives = (1 - target).sum(dim=axes)
    positive_loss = (pixel_loss * target).sum(dim=axes) / positives.clamp_min(1)
    negative_loss = (pixel_loss * (1 - target)).sum(dim=axes) / negatives.clamp_min(1)
    mixed = .5 * (positive_loss + negative_loss)
    per_anchor = torch.where(positives == 0, negative_loss,
                             torch.where(negatives == 0, positive_loss, mixed))
    return per_anchor.mean()


def average_precision(probability, target):
    """Non-interpolated precision-recall area, grouping equal scores as one threshold."""
    scores = np.asarray(probability).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)
    if not labels.any():
        return None
    order = np.argsort(-scores, kind='stable')
    scores, labels = scores[order], labels[order]
    endpoints = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    true_positives = np.cumsum(labels)[endpoints]
    precision = true_positives / (endpoints + 1)
    recall = true_positives / labels.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def measure(probability, targets, cases):
    probabilities = np.asarray(probability)
    targets = np.asarray(targets)
    if (probabilities.shape != targets.shape or len(cases) != len(targets)
            or targets.dtype != bool or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('Finite probabilities and boolean targets with the exact case shape required')
    rows, grouped = [], {}
    for scores, target, case in zip(probabilities, targets, cases, strict=True):
        positive = scores >= .5
        actual = int(target.sum())
        tp = int((positive & target).sum())
        fp = int((positive & ~target).sum())
        union = int((positive | target).sum())
        row = dict(game=case['game'], anchor_index=case['anchor_index'], target_pixels=actual,
                   true_positive_pixels=tp, false_positive_pixels=fp,
                   iou=tp / union if actual else None,
                   average_precision=average_precision(scores, target),
                   false_positive_fraction=fp / int((~target).sum()) if (~target).any() else None)
        rows.append(row)
        if actual:
            grouped.setdefault(case['game'], []).append(row)
    games = {game: dict(iou=float(np.mean([r['iou'] for r in items])),
                        average_precision=float(np.mean([r['average_precision'] for r in items])))
             for game, items in grouped.items()}
    return dict(cases=rows, nonempty_by_game=games,
                mean_nonempty_iou=float(np.mean([r['iou'] for r in games.values()])) if games else None,
                mean_nonempty_average_precision=float(np.mean([r['average_precision'] for r in games.values()])) if games else None,
                empty_anchor_false_positive_fractions=[r['false_positive_fraction'] for r in rows if r['target_pixels'] == 0])


def feasibility_verdict(measures, completed_updates, expected_updates=256):
    complete = completed_updates == {'visual': expected_updates, 'constant': expected_updates}
    if not complete:
        return dict(passed=False, complete=False, reason='Both fixed update budgets must finish')
    visual = measures['visual']
    controls = [measures[key]['mean_nonempty_iou'] for key in ('constant', 'initial', 'zero')]
    if visual['mean_nonempty_iou'] is None or any(value is None for value in controls):
        return dict(passed=False, complete=True, reason='No nonempty development target')
    absolute = visual['mean_nonempty_iou'] >= .25
    advantage = visual['mean_nonempty_iou'] - max(controls)
    empty = visual['empty_anchor_false_positive_fractions']
    specificity = len(empty) == 2 and all(value <= .001 for value in empty)
    return dict(passed=absolute and advantage >= .10 and specificity, complete=True,
                minimum_iou_pass=absolute, advantage_over_best_control=advantage,
                minimum_advantage_pass=advantage >= .10, each_empty_anchor_pass=specificity,
                claim='Development-set fitting feasibility only; not independent generalization')
