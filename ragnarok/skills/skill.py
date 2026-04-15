"""Skill: a crystallized competence the agent has mastered.

When the agent achieves proficiency at a task, the current policy
is frozen and saved as a Skill. Skills persist on disk, survive
restarts, and can be reused for new tasks via transfer learning.
"""

from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class Skill:
    """A frozen, reusable competence."""

    name: str
    env_name: str
    policy_state_dict: dict  # Frozen actor-critic weights
    latent_centroid: np.ndarray  # Mean h_t during training (for skill matching)
    performance: float  # Mean reward at crystallization
    normalizer_state: dict  # Frozen normalizer statistics
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H:%M:%S"))
    episodes_trained: int = 0
    metadata: dict = field(default_factory=dict)
    latent_trunk_state_dict: dict = field(default_factory=dict)  # Transferable trunk weights
    # Env-agnostic RSSM subset: core.gru, core.prior, core.posterior.
    # See RSSM.transferable_state_dict() for rationale. Without this,
    # the transferred policy trunk consumes fresh-random RSSM features
    # on the target env and cross-dim transfer is structurally
    # indistinguishable from scratch (the Phase 3 Bug E observation).
    # default_factory=dict so old .pt files missing this field still
    # deserialize (Skill(**data) succeeds with empty dict).
    rssm_core_state_dict: dict = field(default_factory=dict)
