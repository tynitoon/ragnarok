"""Prospective RGB-only association after observation, never a future predictor.

Pointwise learned appearance, temporal walks, and a fixed selective decision.
Internal scores are not calibrated physical-identity probabilities. The owner
must snapshot returned tensors before recording; this module owns no archives.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TEMPERATURE = 0.07
MIN_ROUND_TRIP = 0.90
MIN_SIMILARITY = 0.95
NORMALIZE_EPS = 1e-6
LOG_FLOOR = 1e-12
MAX_PIXELS = 256
MAX_BATCH = 16


@dataclass(frozen=True)
class AssociationProposal:
    destination: torch.Tensor  # B,H,W int64; -1 is the serialized None sentinel.
    admitted: torch.Tensor
    best_index: torch.Tensor | None
    round_trip_score: torch.Tensor | None
    endpoint_similarity: torch.Tensor | None
    computed: bool
    reason: str


class PixelAppearanceEncoder(nn.Module):
    """1712 parameters shared over all RGB pixels, with no positional input."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 1), nn.SiLU(), nn.Conv2d(32, 16, 1),
        )

    def forward(self, rgb):
        if (not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8
                or rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape) < 1):
            raise ValueError('Observed uint8 BCHW RGB images required')
        return F.normalize(self.layers(rgb.float() / 255), dim=1, eps=NORMALIZE_EPS)


def _shape(clips):
    if (not isinstance(clips, torch.Tensor) or clips.dtype != torch.uint8
            or clips.ndim != 5 or clips.shape[1] != 3 or clips.shape[-1] != 3
            or not 1 <= clips.shape[0] <= MAX_BATCH
            or min(clips.shape[2:4]) < 1 or not 2 <= clips.shape[2]*clips.shape[3] <= MAX_PIXELS):
        raise ValueError('Observed Bx3xHxWx3 uint8 clips with B<=16 and 2..256 pixels required')
    return clips.shape[0], clips.shape[2], clips.shape[3]


def encode_clips(encoder, clips):
    """Encode already observed frames independently; no teacher or command input."""
    batch, height, width = _shape(clips)
    if not isinstance(encoder, PixelAppearanceEncoder):
        raise ValueError('The declared pointwise appearance encoder is required')
    frames = clips.permute(0, 1, 4, 2, 3).reshape(batch*3, 3, height, width)
    encoded = encoder(frames)
    return encoded.reshape(batch, 3, 16, height*width).transpose(-1, -2)


def rgb_reference(clips):
    """Experimenter-only fixed chromaticity reference; not a learner input."""
    batch, height, width = _shape(clips)
    return F.normalize(clips.reshape(batch, 3, height*width, 3).float() / 255,
                       dim=-1, eps=NORMALIZE_EPS)


def _features(features):
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
            or features.ndim != 4 or features.shape[1] != 3
            or not 1 <= features.shape[0] <= MAX_BATCH
            or not 2 <= features.shape[2] <= MAX_PIXELS or features.shape[-1] < 1
            or not torch.isfinite(features).all()
            or (torch.linalg.vector_norm(features, dim=-1) > 1.00001).any()):
        raise ValueError('Finite normalized float32 Bx3xNxD appearance features required')


def transitions(features):
    _features(features)
    first, middle, last = features.unbind(dim=1)
    scores01 = torch.bmm(first, middle.transpose(1, 2)) / TEMPERATURE
    scores12 = torch.bmm(middle, last.transpose(1, 2)) / TEMPERATURE
    return (scores01.softmax(dim=-1), scores01.transpose(1, 2).softmax(dim=-1),
            scores12.softmax(dim=-1), scores12.transpose(1, 2).softmax(dim=-1))


def cycle_loss(features):
    """Three unmasked cycles; no inferred visibility gates or intermediate labels.

    Five walk-matrix products: two short cycles and the grouped long cycle
    (A01 A12)(A21 A10). No loss threshold asserts physical correspondence.
    """
    a01, a10, a12, a21 = transitions(features)
    short01 = torch.bmm(a01, a10)
    short12 = torch.bmm(a12, a21)
    outward, returning = torch.bmm(a01, a12), torch.bmm(a21, a10)
    long = torch.bmm(outward, returning)
    losses = [-matrix.diagonal(dim1=-2, dim2=-1).clamp_min(LOG_FLOOR).log().mean()
              for matrix in (short01, short12, long)]
    return torch.stack(losses).mean()


@torch.no_grad()
def associations_from_features(features, *, height, width):
    """Fixed proposal from raw-frame features; caller owns their time provenance."""
    if (type(height) is not int or type(width) is not int or min(height, width) < 1
            or not isinstance(features, torch.Tensor) or features.ndim != 4
            or features.shape[2] != height*width):
        raise ValueError('The source geometry must agree with the feature grid')
    a01, a10, a12, a21 = transitions(features)
    outward, returning = torch.bmm(a01, a12), torch.bmm(a21, a10)
    probability, best = outward.max(dim=-1)
    source = torch.arange(height*width, device=features.device)[None]
    batch = torch.arange(features.shape[0], device=features.device)[:, None]
    round_trip = probability * returning[batch, best, source]
    endpoint = (features[:, 0] * features[:, 2][batch, best]).sum(dim=-1)
    admitted = (round_trip >= MIN_ROUND_TRIP) & (endpoint >= MIN_SIMILARITY)
    destination = torch.where(admitted, best, -1)
    shape = (features.shape[0], height, width)
    return AssociationProposal(
        destination.reshape(shape), admitted.reshape(shape), best.reshape(shape),
        round_trip.reshape(shape), endpoint.reshape(shape), True,
        'appearance_consistency_proposal_not_physical_certificate',
    )


@torch.no_grad()
def propose(encoder, clips, *, image_kind):
    """Post-observation interface; a pooled-only interval cannot certify its endpoint.

    An explicit scalar kind must come from the recording owner. The function
    cannot detect forged raw tags from pixel values. encoder=None selects the
    declared external RGB reference, with the same temporal guard and rule.
    """
    batch, height, width = _shape(clips)
    if image_kind not in ('raw', 'pooled') or not isinstance(image_kind, str):
        raise ValueError('An explicit homogeneous raw or pooled recording kind is required')
    if image_kind == 'pooled':
        shape = (batch, height, width)
        return AssociationProposal(
            torch.full(shape, -1, dtype=torch.int64, device=clips.device),
            torch.zeros(shape, dtype=torch.bool, device=clips.device),
            None, None, None, False, 'temporal_support_not_point_observation',
        )
    features = rgb_reference(clips) if encoder is None else encode_clips(encoder, clips)
    return associations_from_features(features, height=height, width=width)
