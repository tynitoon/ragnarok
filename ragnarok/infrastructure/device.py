"""Device detection and tensor utilities."""

import torch


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the default device."""
    return tensor.to(device=DEVICE, dtype=DTYPE)


def to_numpy(tensor: torch.Tensor):
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy()
