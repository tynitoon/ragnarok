"""Checkpoint save/load for training resumption."""

from pathlib import Path
import torch


def save_checkpoint(path: str, **components):
    """Save arbitrary components to a checkpoint file.

    Usage:
        save_checkpoint("ckpt.pt",
            world_model=rssm.state_dict(),
            policy=policy.state_dict(),
            optimizer_wm=opt_wm.state_dict(),
            optimizer_policy=opt_policy.state_dict(),
            step=global_step,
            episode=episode_count,
        )
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(components, ckpt_path)


def load_checkpoint(path: str, device: torch.device | None = None) -> dict:
    """Load a checkpoint file and return its components dict."""
    return torch.load(path, map_location=device, weights_only=False)
