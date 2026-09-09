"""Prospective multiscale RGB loss; no labels, motion estimator or front-end.

Compare the model's reconstruction and the observed future at native scale and
after two fixed local averages. Earlier P sources/results remain unchanged.
"""
import torch
from torch.nn import functional as F


def blurred_prediction_loss(reconstructed,following):
    """Equal mean of RGB L1 at box widths1/3/7, preserving every image pixel.

The same differentiable filter is applied to both images. Padding is zero and
included in each fixed denominator. Native width1 stays in the objective so a
good blurred fit alone cannot make the complete loss vanish.
"""
    if (not isinstance(reconstructed,torch.Tensor) or not isinstance(following,torch.Tensor)
            or reconstructed.ndim!=4 or reconstructed.shape[1]!=3 or not len(reconstructed)
            or min(reconstructed.shape[-2:])<7 or reconstructed.shape!=following.shape
            or reconstructed.device!=following.device or not reconstructed.is_floating_point()
            or not following.is_floating_point()
            or not torch.isfinite(reconstructed).all() or not torch.isfinite(following).all()
            or (reconstructed<0).any() or (reconstructed>1).any()
            or (following<0).any() or (following>1).any()):
        raise ValueError('Equal finite BCHW RGB in[0,1] with dimensions at least7 required')
    left,right=reconstructed.float(),following.float()
    losses={1:(left-right).abs().mean()}
    for width in (3,7):
        def blur(value):return F.avg_pool2d(value,width,stride=1,padding=width//2,count_include_pad=True)
        losses[width]=(blur(left)-blur(right)).abs().mean()
    return dict(loss=sum(losses.values())/3,scale_l1=losses)
