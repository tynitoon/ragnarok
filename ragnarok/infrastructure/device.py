"""Device detection and tensor utilities.

Supports three backends, auto-detected at import time with priority
TPU (XLA) > CUDA > CPU:

- **TPU / XLA**: used when `torch_xla` is importable and an XLA device is
  reachable (i.e. running on a Cloud TPU VM). PyTorch/XLA evaluates lazily —
  ops accumulate into a graph that only executes on `mark_step()`. Training
  loops must call `mark_step()` once per `optimizer.step()`.
- **CUDA**: used on a machine with an NVIDIA GPU and no torch_xla.
- **CPU**: fallback.

The same codebase runs unchanged on a local CUDA workstation and on a Cloud
TPU VM — `torch_xla` is an optional import, absent on the CUDA machine.
"""

import torch

# Optional XLA import. Absent on CUDA/CPU machines; present on TPU VMs.
try:  # pragma: no cover - import path depends on host
    import torch_xla
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


def get_device() -> torch.device:
    """Auto-detect the best available device.

    Priority: TPU (XLA) > CUDA > CPU.
    """
    if _XLA_AVAILABLE:
        try:  # pragma: no cover - only runs on a TPU VM
            return torch_xla.device()
        except Exception:
            # torch_xla importable but no XLA device reachable — fall through.
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32

# True when the active device is an XLA (TPU) device. Drives mark_step().
IS_XLA = str(DEVICE).startswith("xla")

# XLA: torch.distributions validates constructor args by default, and every
# check ends in a `.all()` — a tensor->host sync. On XLA each such sync forces
# a graph execution, and the per-distribution-shape `.all()` graphs bloat the
# executable cache (a Normal/Categorical is built per RSSM step, per policy
# eval — thousands of times). Validation only decides whether *invalid* args
# raise; it never changes numerical results, so disabling it on TPU is safe.
# Left ON for CUDA/CPU so local dev still catches bad distribution args.
if IS_XLA:  # pragma: no cover - only runs on a TPU VM
    torch.distributions.Distribution.set_default_validate_args(False)
    # TPU MXU matmuls default to a single bf16 pass for fp32 inputs. Across a
    # long recurrent unroll (the RSSM world model's 128-step GRU) that
    # rounding error compounds: world-model training diverges on the TPU
    # (observed KL 1.3 -> 10 over 30 rollouts) while staying stable on a CUDA
    # GPU running the identical code. 'highest' forces XLA to compute fp32
    # matmuls faithfully (multi-pass bf16), so the TPU matches the GPU the
    # calibration was done on — calibration-neutral by construction. It costs
    # matmul throughput, but a diverging world model is worthless.
    torch.set_float32_matmul_precision("highest")


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to the default device."""
    return tensor.to(device=DEVICE, dtype=DTYPE)


def to_numpy(tensor: torch.Tensor):
    """Convert tensor to numpy array.

    On XLA this forces a synchronous graph execution + device->host transfer.
    Avoid calling it inside hot per-step loops where possible.
    """
    return tensor.detach().cpu().numpy()


def mark_step() -> None:
    """Flush the XLA lazy-evaluation graph (execute accumulated ops).

    No-op on CUDA/CPU. On XLA, call this once per ``optimizer.step()`` in
    training loops: without it the lazy graph grows unbounded and never
    executes; with it the graph is materialized at a controlled cadence.
    """
    if IS_XLA:
        xm.mark_step()
