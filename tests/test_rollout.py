"""Tests for the device-resident fixed-shape rollout collector.

Pins the shapes of a RolloutBatch and the normalizer contract: ``obs`` is
the normalized observation the policy acts on, ``raw_obs`` is the untouched
env state, and the normalizer's stats stay frozen for the whole rollout
(read-only WHILE collecting — ``update`` is called BETWEEN rollouts).
"""

import torch

from ragnarok.environments.device_env import (
    DeviceVecCartPole, DeviceRunningNormalizer)
from ragnarok.learning.rollout import collect_rollout, RolloutBatch
from ragnarok.infrastructure.device import DEVICE


def _discrete_policy_fn(obs):
    """Uniform-random discrete policy_fn — no net needed for collector tests."""
    n = obs.shape[0]
    dist = torch.distributions.Categorical(logits=torch.zeros(n, 2, device=obs.device))
    action = dist.sample()
    return action, dist.log_prob(action), torch.zeros(n, device=obs.device)


class TestCollectRollout:

    def test_batch_shapes(self):
        N, T = 8, 16
        env = DeviceVecCartPole(N)
        batch = collect_rollout(env, _discrete_policy_fn, T)
        assert isinstance(batch, RolloutBatch)
        assert batch.obs.shape == (N, T, 4)
        assert batch.raw_obs.shape == (N, T, 4)
        assert batch.actions.shape == (N, T)
        assert batch.rewards.shape == (N, T)
        assert batch.dones.shape == (N, T)
        assert batch.logp.shape == (N, T)
        assert batch.values.shape == (N, T)
        assert batch.last_obs.shape == (N, 4)
        assert batch.last_value.shape == (N,)
        assert batch.num_envs == N
        assert batch.horizon == T
        assert batch.total_steps == N * T

    def test_no_normalizer_obs_equals_raw(self):
        """Without a normalizer the policy sees the raw env state verbatim."""
        env = DeviceVecCartPole(4)
        batch = collect_rollout(env, _discrete_policy_fn, 8)
        torch.testing.assert_close(batch.obs, batch.raw_obs)

    def test_normalizer_is_applied(self):
        """With a non-identity normalizer, obs == normalizer(raw_obs) exactly."""
        env = DeviceVecCartPole(6)
        norm = DeviceRunningNormalizer(obs_dim=4)
        # Shift the stats off identity so normalize is not a passthrough.
        norm.update(torch.randn(500, 4, device=DEVICE) * 3.0 + 2.0)
        batch = collect_rollout(env, _discrete_policy_fn, 10, normalizer=norm)
        torch.testing.assert_close(batch.obs, norm.normalize(batch.raw_obs))
        assert not torch.allclose(batch.obs, batch.raw_obs)

    def test_normalizer_frozen_during_rollout(self):
        """Stats are read-only WHILE collecting — update() runs between rollouts."""
        env = DeviceVecCartPole(4)
        norm = DeviceRunningNormalizer(obs_dim=4)
        norm.update(torch.randn(100, 4, device=DEVICE))
        mean_before = norm.mean.clone()
        var_before = norm.var.clone()
        count_before = norm.count.clone()
        collect_rollout(env, _discrete_policy_fn, 12, normalizer=norm)
        torch.testing.assert_close(norm.mean, mean_before)
        torch.testing.assert_close(norm.var, var_before)
        torch.testing.assert_close(norm.count, count_before)

    def test_rewards_are_cartpole_plus_one(self):
        """CartPole pays +1 every step — sanity-check the collected rewards."""
        N, T = 4, 20
        env = DeviceVecCartPole(N)
        batch = collect_rollout(env, _discrete_policy_fn, T)
        torch.testing.assert_close(batch.rewards, torch.ones(N, T, device=DEVICE))

    def test_last_obs_continues_the_rollout(self):
        """last_obs is the env state AFTER the final step — the bootstrap point."""
        env = DeviceVecCartPole(4)
        batch = collect_rollout(env, _discrete_policy_fn, 6)
        torch.testing.assert_close(batch.last_obs, env.state)
