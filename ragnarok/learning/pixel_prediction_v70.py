"""Self-generated P terms: forward RGB splatting and source-aligned motion evidence.

All targets are images or the parser's own detached displacement. No classical
flow, background model, scorer or object identities supply supervision.
"""
import torch
from torch.nn import functional as F


def _validate(image, displacement):
    if (not isinstance(image,torch.Tensor) or image.ndim != 4 or image.shape[1] != 3
            or image.shape[0] == 0 or min(image.shape[-2:]) == 0 or not image.is_floating_point()
            or not isinstance(displacement,torch.Tensor) or not displacement.is_floating_point()
            or displacement.shape != (image.shape[0],2,*image.shape[-2:])
            or image.device != displacement.device or not torch.isfinite(image).all()
            or not torch.isfinite(displacement).all()):
        raise ValueError('Finite BCHW RGB and B2HW forward displacement on one device required')


def _coordinates(image, displacement):
    height,width=image.shape[-2:]
    y,x=torch.meshgrid(torch.arange(height,device=image.device,dtype=torch.float32),
                       torch.arange(width,device=image.device,dtype=torch.float32),indexing='ij')
    return x+displacement[:,0].float(),y+displacement[:,1].float()


def forward_warp_rgb(image, displacement):
    """Splat source(x,y) to destination(x+dx,y+dy), with bilinear weights.

Return RGB plus destination coverage. Holes are zero, collisions are weighted
means. Loss callers must retain holes in their denominator, not mask them away.
"""
    _validate(image,displacement)
    image=image.float()
    batch,channels,height,width=image.shape
    x,y=_coordinates(image,displacement)
    # Values farther outside cannot contribute. Clamping avoids int64 overflow
    # for finite but extreme fields; their masks still admit no destination.
    x=x.clamp(-2,width+1)
    y=y.clamp(-2,height+1)
    x0,y0=x.floor(),y.floor()
    dx,dy=x-x0,y-y0
    sums=image.new_zeros((batch,channels,height*width))
    coverage=image.new_zeros((batch,1,height*width))
    colors=image.reshape(batch,channels,-1)
    for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=x0+ox,y0+oy
        valid=(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
        indices=(yy.clamp(0,height-1).long()*width+xx.clamp(0,width-1).long()).reshape(batch,1,-1)
        weights=(weight*valid).reshape(batch,1,-1)
        sums=sums.scatter_add(2,indices.expand(-1,channels,-1),colors*weights)
        coverage=coverage.scatter_add(2,indices,weights)
    result=sums/coverage.clamp_min(1e-6)
    return result.reshape_as(image),coverage.reshape(batch,1,height,width)


def motion_foreground_target(previous, following, displacement):
    """Source-pixel evidence that own predicted motion explains a change.

Compare a pixel with the actual following image at its predicted forward
position, and with staying still. This is detached self-supervision, not flow
ground truth. A poor motion estimate can supply a poor foreground target.
"""
    _validate(previous,displacement)
    if (following.shape != previous.shape or following.device != previous.device
            or not following.is_floating_point() or not torch.isfinite(following).all()):
        raise ValueError('A finite following image with the same shape/device is required')
    with torch.no_grad():
        previous,following=previous.float(),following.float()
        x,y=_coordinates(previous,displacement.detach())
        height,width=previous.shape[-2:]
        # Pixel-centre coordinates for align_corners=False, also valid at size1.
        grid=torch.stack(((2*x+1)/width-1,(2*y+1)/height-1),dim=-1)
        moved=F.grid_sample(following,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
        stationary_error=(following-previous).abs().mean(dim=1,keepdim=True)
        moving_error=(moved-previous).abs().mean(dim=1,keepdim=True)
        evidence=((stationary_error-moving_error)/(stationary_error+1/255)).clamp(0,1)
        moving=(displacement.detach().float().square().sum(dim=1,keepdim=True).sqrt()>.25)
        inside=((x>=0)&(x<=width-1)&(y>=0)&(y<=height-1))[:,None]
        return evidence*moving*inside


def prediction_terms(previous_rgb, following_rgb, output):
    """Unweighted losses; training schedule/coefficients are not frozen here."""
    if (previous_rgb.dtype != torch.uint8 or following_rgb.dtype != torch.uint8
            or previous_rgb.shape != following_rgb.shape or previous_rgb.device != following_rgb.device):
        raise ValueError('Equal uint8 previous/following RGB tensors required')
    previous,following=previous_rgb.float()/255,following_rgb.float()/255
    reconstructed,coverage=forward_warp_rgb(previous,output.displacement)
    target=motion_foreground_target(previous,following,output.displacement)
    if (output.foreground_logits.shape != target.shape or output.foreground_logits.device != target.device
            or not torch.isfinite(output.foreground_logits).all()):
        raise ValueError('Finite source-aligned foreground logits required')
    return dict(image_l1=(reconstructed-following).abs().mean(),
                foreground_bce=F.binary_cross_entropy_with_logits(output.foreground_logits.float(),target),
                reconstructed=reconstructed,coverage=coverage,foreground_target=target)
