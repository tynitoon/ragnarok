"""A generic direct RGB path, initially zero, beside the unchanged image CNN."""
import torch
from torch import nn
from torch.nn import functional as F

from ragnarok.learning.response_envelope_v70 import ResponseEnvelope


class DirectResponseEnvelope(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct the old CNN first, preserving its seeded initialization and RNG consumption.
        self.contextual = ResponseEnvelope()
        self.direct_weight = nn.Parameter(torch.zeros(1, 6, 1, 1))
        self.direct_bias = nn.Parameter(torch.zeros(1))

    def forward(self, past):
        return self.contextual(past) + F.conv2d(past, self.direct_weight, self.direct_bias)
