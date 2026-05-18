"""Tests for device-path RSSM world-model training (Phase 2 Stage 3).

Covers WorldModelTrainer.train_world_model_on_rollout: it consumes a
fixed-shape RolloutBatch straight off the device path (no host ReplayBuffer,
no host->device transfer) and runs the RSSM update on it.
"""

import pytest
import torch

from ragnarok.environments.device_env import DeviceVecCartPole
from ragnarok.learning.rollout import collect_rollout
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.infrastructure.device import DEVICE


def _policy_fn(obs):
    """Uniform-random discrete policy — only needed to drive collection."""
    n = obs.shape[0]
    dist = torch.distributions.Categorical(
        logits=torch.zeros(n, 2, device=obs.device))
    action = dist.sample()
    return action, dist.log_prob(action), torch.zeros(n, device=obs.device)


def _rollout(n=16, t=24):
    return collect_rollout(DeviceVecCartPole(n), _policy_fn, t)


def _trainer():
    # The device path is device-resident — the RSSM must live on DEVICE so
    # its params match the on-device RolloutBatch tensors.
    rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=32, stoch_dim=8,
                encoder_hidden=32).to(DEVICE)
    return WorldModelTrainer(rssm, ReplayBuffer())


class TestTrainWorldModelOnRollout:

    def test_returns_wm_metrics(self):
        metrics = _trainer().train_world_model_on_rollout(
            _rollout(), epochs=1, n_minibatches=4)
        for k in ("wm/total_loss", "wm/recon_loss", "wm/reward_loss",
                  "wm/continue_loss", "wm/kl_loss"):
            assert k in metrics
            assert isinstance(metrics[k], float)

    def test_requires_divisible_env_count(self):
        """N (= env count) must divide n_minibatches evenly — fail loud,
        never silently drop env rows."""
        wm = _trainer()
        with pytest.raises(AssertionError):
            wm.train_world_model_on_rollout(_rollout(n=16), n_minibatches=5)

    def test_one_hot_encodes_discrete_actions(self):
        """Rollout actions are (N, T) int indices; the trainer one-hot
        encodes them to (N, T, action_dim) for the RSSM."""
        batch = _rollout(n=16, t=24)
        assert batch.actions.dim() == 2  # discrete index form
        _trainer().train_world_model_on_rollout(batch, epochs=1,
                                                n_minibatches=4)

    def test_loss_decreases_with_training(self):
        """Repeated updates on one rollout drive the total loss down."""
        wm = _trainer()
        batch = _rollout(n=32, t=32)
        first = wm.train_world_model_on_rollout(batch, epochs=1,
                                                n_minibatches=4)
        last = first
        for _ in range(8):
            last = wm.train_world_model_on_rollout(batch, epochs=2,
                                                   n_minibatches=4)
        assert last["wm/total_loss"] < first["wm/total_loss"]
