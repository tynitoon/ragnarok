"""Elastic Weight Consolidation (EWC) - optional regularizer.

Computes the diagonal Fisher Information Matrix after learning a task
and adds a penalty to prevent important weights from changing too much
when learning a new task. This is supplementary to the frozen-skill
approach (the primary anti-forgetting mechanism).
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EWC:
    """Elastic Weight Consolidation regularizer."""

    def __init__(self, model: nn.Module, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self.params: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def register_task(self, data_loader_fn, num_samples: int = 200):
        """Compute and store Fisher Information Matrix for current task.

        Args:
            data_loader_fn: callable that yields (input, target) batches
            num_samples: number of samples to estimate Fisher
        """
        # Store current parameter values
        self.params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Compute Fisher Information (diagonal approximation)
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        count = 0

        for batch in data_loader_fn():
            if count >= num_samples:
                break

            self.model.zero_grad()
            loss = batch  # Expect data_loader_fn to yield loss tensors
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += 1

        # Average
        for name in fisher:
            fisher[name] /= max(count, 1)

        self.fisher = fisher
        self.model.train()

    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty: sum of Fisher-weighted squared parameter differences."""
        if not self.fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(iter(self.params.values())).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]).pow(2)).sum()

        return self.importance * loss
