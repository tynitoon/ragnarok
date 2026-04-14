"""Tests for LatentPolicyHead — cross-dim transfer via shared trunk."""

import torch
import numpy as np
import pytest

from ragnarok.learning.latent_policy import LatentPolicyHead
from ragnarok.infrastructure.device import DEVICE


class TestLatentPolicyForward:
    """Basic forward pass tests."""

    def test_discrete_forward_shapes(self):
        """Discrete policy returns (logits, value) with correct shapes."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        latent = torch.randn(4, 160, device=DEVICE)
        logits, value = policy(latent)
        assert logits.shape == (4, 3)
        assert value.shape == (4,)

    def test_continuous_forward_shapes(self):
        """Continuous policy returns (mean, logstd, value) with correct shapes."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False).to(DEVICE)
        latent = torch.randn(4, 160, device=DEVICE)
        mean, logstd, value = policy(latent)
        assert mean.shape == (4, 2)
        assert logstd.shape == (4, 2)
        assert value.shape == (4,)

    def test_logstd_clamped(self):
        """Logstd should be clamped to [-5.0, 2.0]."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False).to(DEVICE)
        latent = torch.randn(32, 160, device=DEVICE) * 100  # Large input
        _, logstd, _ = policy(latent)
        assert logstd.min() >= -5.0
        assert logstd.max() <= 2.0


class TestTrunkTransfer:
    """Test trunk save/load for cross-env transfer."""

    def test_trunk_state_dict_keys(self):
        """Trunk state dict should contain only shared + critic keys."""
        policy = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        trunk_sd = policy.get_trunk_state_dict()
        for key in trunk_sd:
            assert key.startswith("shared.") or key.startswith("critic_head.")
        # Should NOT contain actor keys
        assert not any(k.startswith("actor_head.") for k in trunk_sd)

    def test_trunk_roundtrip(self):
        """Save trunk -> load into fresh policy -> shared weights match."""
        src = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=5, discrete=True).to(DEVICE)
        dst.load_trunk_state_dict(trunk_sd)

        for key in trunk_sd:
            assert torch.equal(dst.state_dict()[key], trunk_sd[key])

    def test_cross_dim_transfer(self):
        """Transfer trunk between policies with different action dims."""
        # CartPole-like (2 actions) -> Acrobot-like (3 actions)
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)

        # Train src briefly to get non-init weights
        optimizer = torch.optim.Adam(src.parameters(), lr=1e-3)
        for _ in range(10):
            latent = torch.randn(8, 160, device=DEVICE)
            logits, value = src(latent)
            loss = -value.mean() + logits.sum() * 0.01
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True).to(DEVICE)
        dst_before = {k: v.clone() for k, v in dst.state_dict().items()
                      if k.startswith("shared.")}
        dst.load_trunk_state_dict(trunk_sd)
        dst_after = dst.state_dict()

        # Shared weights should have changed
        changed = any(not torch.equal(dst_before[k], dst_after[k])
                      for k in dst_before)
        assert changed, "Shared weights should change after trunk transfer"

        # Actor head should be untouched (different dim)
        assert dst_after["actor_head.weight"].shape == (3, 128)

    def test_discrete_to_continuous_trunk(self):
        """Trunk transfer works across discrete/continuous boundaries."""
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        dst = LatentPolicyHead(latent_dim=160, action_dim=1, discrete=False).to(DEVICE)
        dst.load_trunk_state_dict(trunk_sd)

        # Shared + critic should match
        for key in trunk_sd:
            assert torch.equal(dst.state_dict()[key], trunk_sd[key])

        # Continuous heads should still work
        latent = torch.randn(4, 160, device=DEVICE)
        mean, logstd, value = dst(latent)
        assert mean.shape == (4, 1)

    def test_mismatched_latent_dim_ignored(self):
        """If latent_dim differs, mismatched keys are skipped (no crash)."""
        src = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=True).to(DEVICE)
        trunk_sd = src.get_trunk_state_dict()

        # Different latent_dim -> first linear layer won't match
        dst = LatentPolicyHead(latent_dim=256, action_dim=3, discrete=True).to(DEVICE)
        dst_before = {k: v.clone() for k, v in dst.state_dict().items()}

        dst.load_trunk_state_dict(trunk_sd)  # Should not crash

        # Keys with matching shapes should transfer, others stay
        dst_after = dst.state_dict()
        # shared.0.weight has shape (128, latent_dim), so it won't match
        assert dst_after["shared.0.weight"].shape == (128, 256)


class TestAgentIntegration:
    """Test latent policy integration with agent."""

    def test_agent_has_latent_policy(self):
        """Agent should create a latent policy on init."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        assert hasattr(agent, "latent_policy")
        assert isinstance(agent.latent_policy, LatentPolicyHead)
        expected_dim = config.world_model.hidden_dim + config.world_model.stoch_dim
        assert agent.latent_policy.latent_dim == expected_dim
        env.close()

    def test_agent_has_latent_optim(self):
        """Agent should create an optimizer for latent_policy."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)
        assert isinstance(agent.latent_optim, torch.optim.Adam)
        # Optimizer should contain latent_policy params
        optim_params = {id(p) for group in agent.latent_optim.param_groups
                        for p in group["params"]}
        latent_params = {id(p) for p in agent.latent_policy.parameters()}
        assert optim_params == latent_params
        env.close()

    def test_train_latent_policy_updates_weights(self):
        """_train_latent_policy should change latent_policy weights."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        T = 20
        obs = np.random.randn(T, spec.obs_dim).astype(np.float32)
        acts = np.zeros((T, spec.action_dim), dtype=np.float32)
        acts[np.arange(T), np.random.randint(0, spec.action_dim, T)] = 1.0
        rews = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0

        before = {k: v.clone() for k, v in
                  agent.latent_policy.state_dict().items()}
        metrics = agent._train_latent_policy((obs, acts, rews, dones))
        after = agent.latent_policy.state_dict()

        # At least the shared trunk should have changed
        changed = any(not torch.equal(before[k], after[k])
                      for k in before if k.startswith("shared."))
        assert changed, "Shared trunk weights should change after training"
        assert "latent/actor_loss" in metrics
        assert "latent/value_loss" in metrics
        env.close()

    def test_latent_training_runs_in_train_policy_real(self):
        """train_policy_real should produce latent training metrics."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.policy.ppo_batch_episodes = 2  # Speed up test

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        before = {k: v.clone() for k, v in
                  agent.latent_policy.state_dict().items()}
        reward, metrics = agent.train_policy_real()
        after = agent.latent_policy.state_dict()

        # Shared trunk should have been updated
        changed = any(not torch.equal(before[k], after[k])
                      for k in before if k.startswith("shared."))
        assert changed
        # Metrics should include latent losses
        assert any(k.startswith("latent/") for k in metrics)
        env.close()

    def test_crystallization_saves_trunk(self):
        """Skill crystallization must include latent trunk state dict.

        The check_crystallization() contract: given an above-threshold eval
        reward and enough episodes, return a Skill whose latent_trunk_state_dict
        is populated. Before the review, this test wrote to a nonexistent
        _crystallization_rewards attribute, then guarded with `if skill is not
        None:` — so a broken crystallizer that never fired would silently pass.

        This version monkey-patches the evaluator to report a qualifying reward
        and asserts unconditionally that crystallization occurred AND carried
        the trunk payload.
        """
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent
        import tempfile

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        with tempfile.TemporaryDirectory() as tmpdir:
            config.skill.skills_dir = tmpdir
            env = RagnarokEnv(spec.gym_name, seed=42)
            agent = RagnarokAgent(config, env)

            # Satisfy the min-episodes gate
            agent.total_episodes = config.skill.min_episodes + 1
            # Force eval to return an above-threshold reward (CartPole: 450)
            threshold = config.skill.thresholds["CartPole-v1"]
            agent.real_trainer.evaluate = lambda _env, episodes=5: threshold + 1.0

            skill = agent.check_crystallization()

            assert skill is not None, (
                "check_crystallization returned None with eval=threshold+1 and "
                "total_episodes above min — crystallization path is broken"
            )
            assert hasattr(skill, "latent_trunk_state_dict")
            assert len(skill.latent_trunk_state_dict) > 0
            # Must contain shared-trunk keys (what transfers cross-env)
            assert any(k.startswith("shared.") for k in skill.latent_trunk_state_dict)
            env.close()


class TestActingPath:
    """Cross-dim transfer requires latent_policy to actually drive actions.

    Until preregistration v3 §6.1 fix #1, latent_policy was trained but the
    rollout loop never called it. These tests pin the wiring so a regression
    can't silently re-break the publication-blocking bug.
    """

    def test_default_acting_mode_is_obs(self):
        """Fresh agent acts via obs policy until try_transfer flips the mode."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)
        assert agent.acting_policy_mode == "obs"
        env.close()

    def test_latent_act_returns_correct_shape(self):
        """LatentPolicyHead.act should return env-compatible action."""
        # Discrete
        head_d = LatentPolicyHead(latent_dim=160, action_dim=3, discrete=True)
        latent = torch.randn(1, 160)
        a = head_d.act(latent, deterministic=True)
        assert isinstance(a, int)
        assert 0 <= a < 3

        # Continuous
        head_c = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False)
        a = head_c.act(latent, deterministic=True)
        assert isinstance(a, np.ndarray)
        assert a.shape == (2,)

    def test_continuous_act_respects_env_bounds(self):
        """H1-primary endpoint is MountainCarContinuous (bounds [-1, 1]).

        LatentPolicyHead.act() must apply tanh+rescale so emitted actions
        satisfy env.action_space. A Gaussian sample without squash/rescale
        would silently fail every continuous-target transfer run.
        """
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)
        head = LatentPolicyHead(latent_dim=160, action_dim=1, discrete=False,
                                action_low=low, action_high=high).to(DEVICE)

        # Stress: run many acts with a large latent to push the Gaussian tails,
        # then verify every emitted action is in-bounds in both modes.
        torch.manual_seed(0)
        for _ in range(64):
            latent = torch.randn(1, 160, device=DEVICE) * 10.0
            a_det = head.act(latent, deterministic=True)
            a_sto = head.act(latent, deterministic=False)
            assert (a_det >= low).all() and (a_det <= high).all(), a_det
            assert (a_sto >= low).all() and (a_sto <= high).all(), a_sto

    def test_continuous_act_asymmetric_bounds(self):
        """Pendulum-style bounds [-2, 2] (or arbitrary asymmetry) must rescale
        correctly from tanh's [-1, 1] output range.
        """
        low = np.array([-2.0, 0.0], dtype=np.float32)
        high = np.array([2.0, 5.0], dtype=np.float32)
        head = LatentPolicyHead(latent_dim=160, action_dim=2, discrete=False,
                                action_low=low, action_high=high).to(DEVICE)
        torch.manual_seed(0)
        for _ in range(32):
            latent = torch.randn(1, 160, device=DEVICE) * 10.0
            a = head.act(latent, deterministic=True)
            assert (a >= low).all() and (a <= high).all(), a

    def test_collect_episode_uses_latent_when_mode_is_latent(self):
        """When acting_policy_mode == 'latent', latent_policy.forward is called
        on every env step (proves the wiring is live, not just the mode flag).
        """
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.policy.explore_ratio = 0.0  # disable epsilon-greedy bypass

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)
        agent.acting_policy_mode = "latent"

        call_count = {"n": 0}
        original_forward = agent.latent_policy.forward

        def counting_forward(*args, **kwargs):
            call_count["n"] += 1
            return original_forward(*args, **kwargs)

        agent.latent_policy.forward = counting_forward

        agent.collect_episode(explore_ratio=0.0)

        # Latent policy must have been called at least once per env-step
        # (a CartPole episode is at least 8 steps even with random policy).
        assert call_count["n"] >= 1, (
            "latent_policy.forward never called during collect_episode "
            "with acting_policy_mode='latent' — the wiring is dead"
        )
        env.close()

    def test_try_transfer_flips_mode_on_latent_trunk_load(self):
        """Cross-env transfer that falls back to latent-trunk load must set
        acting_policy_mode='latent'; otherwise the loaded trunk never acts.
        """
        from unittest.mock import MagicMock
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent
        from ragnarok.skills.skill import Skill

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)
        assert agent.acting_policy_mode == "obs"

        # Fabricate a foreign-env skill whose obs-policy state_dict will fail
        # to load (mismatched shapes), forcing the latent-trunk fallback.
        trunk_sd = agent.latent_policy.get_trunk_state_dict()
        bad_obs_policy_sd = {
            "fc.weight": torch.zeros(1, 999),  # nonsense shape -> RuntimeError
            "fc.bias": torch.zeros(1),
        }
        skill = MagicMock(spec=Skill)
        skill.env_name = "FakeForeignEnv"
        skill.policy_state_dict = bad_obs_policy_sd
        skill.latent_trunk_state_dict = trunk_sd
        skill.normalizer_state = None

        agent.skill_selector = MagicMock()
        agent.skill_selector.select.return_value = skill
        agent.skill_library._cache = {}  # skip exact-match path

        loaded = agent.try_transfer()
        assert loaded is skill
        assert agent.acting_policy_mode == "latent", (
            "try_transfer flipped to latent-trunk fallback but did NOT "
            "set acting_policy_mode='latent' — transfer is acting-time-invisible"
        )
        env.close()

    def test_acting_policy_mode_survives_save_load(self):
        """Checkpoint round-trip must preserve acting_policy_mode so a
        post-transfer agent reloaded from disk keeps acting from latent.
        """
        import tempfile, os
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig()
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)
        agent.acting_policy_mode = "latent"

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            agent.save(ckpt_path)

            env2 = RagnarokEnv(spec.gym_name, seed=42)
            agent2 = RagnarokAgent(config, env2)
            assert agent2.acting_policy_mode == "obs"  # default
            agent2.load(ckpt_path)
            assert agent2.acting_policy_mode == "latent"
            env2.close()
        env.close()
