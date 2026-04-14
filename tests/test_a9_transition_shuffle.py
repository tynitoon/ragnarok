"""Tests for A9 mechanism-isolation ablation (preregistration §5).

A9 tests whether transfer benefits come from LEARNED DYNAMICS or from
the RSSM architectural prior. It trains RSSMs on sequences where the
next-state target `obs[:, t]` for t >= 1 is shuffled across trajectories
at each timestep — preserving per-timestep marginals but destroying the
(s_{t-1}, a_{t-1}) → s_t joint dynamics.

If A9-trained WMs still produce transfer, the transfer mechanism is
architectural, not dynamical — that invalidates the paper's main claim
and triggers Plan B (hypernetwork policies, prereg §11).

These tests pin the shuffle behavior so the ablation is faithful to the
prereg spec: it really does break dynamics, it's gated behind an opt-in
flag that defaults OFF for every non-A9 run, and the config wiring is
intact end-to-end.
"""

import numpy as np
import pytest
import torch

from ragnarok.infrastructure.config import RagnarokConfig, WorldModelConfig
from ragnarok.environments.registry import get_env_spec
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.core.agent import RagnarokAgent
from ragnarok.learning.world_model_trainer import WorldModelTrainer
from ragnarok.core.rssm import RSSM
from ragnarok.memory.replay_buffer import ReplayBuffer
from ragnarok.infrastructure.device import DEVICE


def _make_wm_trainer(shuffle: bool, seed: int = 0) -> WorldModelTrainer:
    """Helper: build a standalone trainer with a populated buffer."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=16, stoch_dim=4,
                encoder_hidden=16).to(DEVICE)
    buffer = ReplayBuffer(capacity=100)
    for _ in range(5):
        T = 10
        obs = np.random.randn(T, 4).astype(np.float32)
        acts = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, T)]
        rews = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0
        buffer.add_episode(obs, acts, rews, dones)
    return WorldModelTrainer(
        rssm=rssm, replay_buffer=buffer,
        batch_size=4, seq_length=5,
        shuffle_transitions=shuffle,
    )


class TestShuffleBehavior:
    def test_shuffle_default_is_off(self):
        cfg = RagnarokConfig()
        assert cfg.world_model.shuffle_transitions is False

    def test_shuffle_preserves_shape(self):
        trainer = _make_wm_trainer(shuffle=True)
        obs = np.random.randn(8, 10, 4).astype(np.float32)
        out = trainer._shuffle_next_state_targets(obs)
        assert out.shape == obs.shape
        assert out.dtype == obs.dtype

    def test_shuffle_preserves_first_timestep(self):
        """t=0 must stay unshuffled (initial state aligns with first action)."""
        trainer = _make_wm_trainer(shuffle=True)
        obs = np.arange(8 * 10 * 4, dtype=np.float32).reshape(8, 10, 4)
        out = trainer._shuffle_next_state_targets(obs)
        np.testing.assert_array_equal(out[:, 0], obs[:, 0])

    def test_shuffle_breaks_later_timesteps(self):
        """At t >= 1, the output must differ from input for at least one t
        (with high probability — tests use a controlled seed)."""
        np.random.seed(123)
        trainer = _make_wm_trainer(shuffle=True)
        # Make obs unique per (batch, t) so any swap is detectable
        B, T, D = 16, 10, 4
        obs = np.arange(B * T * D, dtype=np.float32).reshape(B, T, D)
        out = trainer._shuffle_next_state_targets(obs)
        # At least some timesteps past t=0 must have changed
        changed_timesteps = 0
        for t in range(1, T):
            if not np.array_equal(out[:, t], obs[:, t]):
                changed_timesteps += 1
        assert changed_timesteps >= 7  # random perms rarely identity

    def test_shuffle_preserves_marginal_distribution(self):
        """For each t >= 1, the multiset of obs vectors is preserved
        (it's a permutation, not a resampling)."""
        np.random.seed(456)
        trainer = _make_wm_trainer(shuffle=True)
        B, T, D = 10, 8, 4
        obs = np.random.randn(B, T, D).astype(np.float32)
        out = trainer._shuffle_next_state_targets(obs)
        for t in range(T):
            # Sort along batch dim; must match
            s_in = np.sort(obs[:, t].sum(axis=-1))
            s_out = np.sort(out[:, t].sum(axis=-1))
            np.testing.assert_array_almost_equal(s_in, s_out)

    def test_shuffle_permutation_is_per_timestep_independent(self):
        """Different timesteps should get different random permutations —
        otherwise trajectories stay self-consistent (relabeling only)."""
        np.random.seed(789)
        trainer = _make_wm_trainer(shuffle=True)
        B, T, D = 8, 6, 2
        obs = np.arange(B * T * D, dtype=np.float32).reshape(B, T, D)
        out = trainer._shuffle_next_state_targets(obs)
        # Extract the implicit permutation at each t (since obs[:, t] is
        # unique per batch index, we can recover it)
        perms = []
        for t in range(1, T):
            perm_t = []
            for i in range(B):
                matches = np.where(np.all(obs[:, t] == out[i, t], axis=-1))[0]
                perm_t.append(int(matches[0]))
            perms.append(tuple(perm_t))
        # At least two timesteps should have DIFFERENT permutations
        unique = len(set(perms))
        assert unique >= 2, f"expected independent perms, got {unique} unique"


class TestShuffleWiring:
    def test_agent_propagates_flag_to_trainer(self):
        spec = get_env_spec("cartpole")
        cfg = RagnarokConfig(seed=0)
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        cfg.world_model.shuffle_transitions = True
        cfg.curiosity.enabled = False
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            assert agent.wm_trainer.shuffle_transitions is True
        finally:
            env.close()

    def test_default_agent_has_shuffle_off(self):
        spec = get_env_spec("cartpole")
        cfg = RagnarokConfig(seed=0)
        cfg.world_model.obs_dim = spec.obs_dim
        cfg.world_model.action_dim = spec.action_dim
        cfg.curiosity.enabled = False
        env = RagnarokEnv(spec.gym_name, seed=0)
        try:
            agent = RagnarokAgent(cfg, env)
            assert agent.wm_trainer.shuffle_transitions is False
        finally:
            env.close()


class TestShuffleActuallyTrains:
    """Sanity: a shuffle-enabled trainer still runs end-to-end without
    errors. The actual ablation finding (does transfer survive?) is a
    Phase 4 experiment; here we just pin that the path doesn't crash.
    """

    def test_shuffled_train_step_runs(self):
        trainer = _make_wm_trainer(shuffle=True, seed=42)
        metrics = trainer.train_step()
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])

    def test_shuffled_vs_unshuffled_produce_different_loss_trajectories(self):
        """Control check: training with shuffle produces different loss
        values than without, demonstrating the flag is actually affecting
        the loss computation (not a dead branch).
        """
        t_noshuf = _make_wm_trainer(shuffle=False, seed=42)
        t_shuf = _make_wm_trainer(shuffle=True, seed=42)
        # Run a few steps to accumulate divergence
        loss_noshuf = [t_noshuf.train_step()["total_loss"] for _ in range(3)]
        loss_shuf = [t_shuf.train_step()["total_loss"] for _ in range(3)]
        # At least one step should differ (seeding makes the buffer-sample
        # reproducible; the shuffle itself uses its own RNG path)
        assert any(abs(a - b) > 1e-6 for a, b in zip(loss_noshuf, loss_shuf))
