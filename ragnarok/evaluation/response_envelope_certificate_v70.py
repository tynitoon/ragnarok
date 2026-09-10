"""External constructive certificate for the unchanged ResponseEnvelope FCN.

These fixed judge weights are not a learner initialization or training target.
For stationary binary red channels they give logits +8 on red pixels and -8
elsewhere.  The identity SiLU(t) - SiLU(-t) = t carries the contrast through
the five existing layers; only central kernel coefficients are used.

The factory performs no forward, reads no data and creates no optimizer.  It
also preserves the CPU RNG state so construction cannot reseed a later learner.
"""

import torch

from ragnarok.learning.response_envelope_v70 import ResponseEnvelope


def make_reference_model() -> ResponseEnvelope:
    """Return the exact CPU/float32/eval architecture with frozen judge weights.

    Input channel order is past RGB followed by current RGB, each in [0, 1].
    The mathematical output is 8 * (past_red + current_red) - 8.  This proves
    representability of the declared synthetic family, not optimizer success
    or bitwise equivalence across floating-point backends.
    """
    # ResponseEnvelope initializes weights randomly before they are replaced.
    # No CUDA device is forked or used, including when a default device differs.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = ResponseEnvelope().to(dtype=torch.float32)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        first = model.body[0]
        for red_channel in (0, 3):
            first.weight[0, red_channel, 1, 1] = 4
            first.weight[1, red_channel, 1, 1] = -4
        first.bias[0] = -4
        first.bias[1] = 4

        for index in (2, 4, 6, 8):
            layer = model.body[index]
            layer.weight[0, 0, 1, 1] = 1
            layer.weight[0, 1, 1, 1] = -1
            layer.weight[1, 0, 1, 1] = -1
            layer.weight[1, 1, 1, 1] = 1

        model.head.weight[0, 0, 0, 0] = 2
        model.head.weight[0, 1, 0, 0] = -2

    model.requires_grad_(False)
    model.eval()
    return model
