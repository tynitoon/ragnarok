"""Running observation normalizer with online mean/variance tracking."""

import torch
import numpy as np


class RunningNormalizer:
    """Normalizes observations using running mean and variance.

    Uses Welford's online algorithm for numerically stable updates.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 5.0, warmup_steps: int = 1000):
        self.shape = shape
        self.clip = clip
        self.warmup_steps = warmup_steps

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.frozen = False

        # Welford accumulators
        self._m2 = np.zeros(shape, dtype=np.float64)

    def freeze(self):
        """Freeze statistics — update() becomes a no-op.

        Use this for off-policy methods (SAC) to keep replay buffer
        data consistent: collect warmup stats, then freeze.
        """
        self.frozen = True

    def update(self, x: np.ndarray):
        """Update running statistics with a new observation (or batch)."""
        if self.frozen:
            return

        if x.ndim == len(self.shape):
            # Single observation
            x = x[np.newaxis, ...]

        for obs in x:
            self.count += 1
            delta = obs - self.mean
            self.mean += delta / self.count
            delta2 = obs - self.mean
            self._m2 += delta * delta2

        if self.count > 1:
            self.var = self._m2 / (self.count - 1)
            self.var = np.maximum(self.var, 1e-6)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation. Returns raw x during warmup."""
        if self.count < self.warmup_steps:
            return x.astype(np.float32)
        normed = (x - self.mean) / np.sqrt(self.var)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a torch tensor using current stats."""
        if self.count < self.warmup_steps:
            return x.float()
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(np.sqrt(self.var), dtype=torch.float32, device=x.device)
        normed = (x.float() - mean) / std
        return normed.clamp(-self.clip, self.clip)

    def state_dict(self) -> dict:
        """Serialize normalizer state."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "m2": self._m2.copy(),
            "shape": self.shape,
            "clip": self.clip,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningNormalizer":
        """Deserialize normalizer from state dict."""
        norm = cls(shape=tuple(state["shape"]), clip=state["clip"], warmup_steps=state["warmup_steps"])
        norm.mean = state["mean"]
        norm.var = state["var"]
        norm.count = state["count"]
        norm._m2 = state["m2"]
        return norm
