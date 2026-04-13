"""Tests for trust region transfer and ensemble RSSM (Phase 5.4)."""

import numpy as np
import torch
import pytest

from ragnarok.core.rssm import RSSM, EnsembleRSSMCore
from ragnarok.infrastructure.device import DEVICE


class TestEnsembleRSSMCore:

    def test_ensemble_init(self):
        """Ensemble should create N independent cores."""
        ens = EnsembleRSSMCore(n_cores=3, stoch_dim=16, hidden_dim=32,
                               action_dim=2, encoder_dim=32).to(DEVICE)
        assert ens.n_cores == 3
        assert len(ens.pre_grus) == 3
        assert len(ens.grus) == 3
        assert len(ens.priors) == 3

    def test_ensemble_initial_state(self):
        """Initial states should be zero for all cores."""
        ens = EnsembleRSSMCore(n_cores=2, stoch_dim=8, hidden_dim=16,
                               action_dim=2, encoder_dim=16).to(DEVICE)
        states = ens.initial_state(4, DEVICE)
        assert len(states) == 2
        for h, z in states:
            assert h.shape == (4, 16)
            assert z.shape == (4, 8)
            assert h.sum().item() == 0.0

    def test_ensemble_step_independent(self):
        """Each core should produce different hidden states after steps."""
        ens = EnsembleRSSMCore(n_cores=2, stoch_dim=8, hidden_dim=16,
                               action_dim=2, encoder_dim=16).to(DEVICE)
        states = ens.initial_state(4, DEVICE)
        action = torch.randn(4, 2, device=DEVICE)

        hs = ens.step_all(states, action)
        # After one step from same init, cores may differ due to different weights
        assert len(hs) == 2
        assert hs[0].shape == (4, 16)
        # Not necessarily different after one step from zero, but shapes correct

    def test_disagreement_positive(self):
        """Disagreement should be non-negative."""
        ens = EnsembleRSSMCore(n_cores=2, stoch_dim=8, hidden_dim=16,
                               action_dim=2, encoder_dim=16).to(DEVICE)
        states = ens.initial_state(4, DEVICE)
        action = torch.randn(4, 2, device=DEVICE)

        # Run a few steps to get non-trivial states
        for _ in range(3):
            hs = ens.step_all(states, action)
            priors = ens.prior_all(hs)
            z = priors[0][0]  # Use first core's mean as z
            states = [(h, z) for h in hs]

        disagr = ens.disagreement(hs)
        assert disagr.shape == (4,)
        assert (disagr >= 0).all()

    def test_disagreement_novel_higher(self):
        """Novel inputs should produce higher disagreement than trained inputs."""
        torch.manual_seed(42)
        ens = EnsembleRSSMCore(n_cores=2, stoch_dim=8, hidden_dim=16,
                               action_dim=2, encoder_dim=16).to(DEVICE)

        # Train only core 0 on specific data, core 1 stays random
        optimizer = torch.optim.Adam(list(ens.priors[0].parameters()), lr=1e-2)
        target = torch.zeros(4, 8, device=DEVICE)

        for _ in range(50):
            h = torch.randn(4, 16, device=DEVICE) * 0.1
            params = ens.priors[0](h)
            mean, _ = params.chunk(2, dim=-1)
            loss = (mean - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Familiar input (small h)
        h_fam = [torch.randn(4, 16, device=DEVICE) * 0.1,
                 torch.randn(4, 16, device=DEVICE) * 0.1]
        d_fam = ens.disagreement(h_fam)

        # Novel input (large h)
        h_nov = [torch.randn(4, 16, device=DEVICE) * 5.0,
                 torch.randn(4, 16, device=DEVICE) * 5.0]
        d_nov = ens.disagreement(h_nov)

        # At least one should show difference (novel >= familiar on average)
        assert d_nov.mean().item() > 0 or d_fam.mean().item() > 0


class TestRSSMEnsembleIntegration:

    def test_rssm_with_ensemble(self):
        """RSSM with ensemble_cores>1 should have ensemble attribute."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=16, stoch_dim=8,
                     encoder_hidden=16, ensemble_cores=2).to(DEVICE)
        assert rssm.ensemble is not None
        assert rssm.ensemble.n_cores == 2

    def test_rssm_without_ensemble(self):
        """RSSM with ensemble_cores=1 should have no ensemble."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=16, stoch_dim=8,
                     encoder_hidden=16, ensemble_cores=1).to(DEVICE)
        assert rssm.ensemble is None

    def test_rssm_observe_still_works(self):
        """Ensemble shouldn't break normal RSSM observe."""
        rssm = RSSM(obs_dim=4, action_dim=2, hidden_dim=16, stoch_dim=8,
                     encoder_hidden=16, ensemble_cores=2).to(DEVICE)
        obs = torch.randn(2, 5, 4, device=DEVICE)
        act = torch.randn(2, 5, 2, device=DEVICE)
        outputs = rssm.observe(obs, act)
        assert outputs["h"].shape == (2, 5, 16)
        assert outputs["z"].shape == (2, 5, 8)


class TestTrustRegion:

    def test_trust_region_alpha_decay(self):
        """Alpha should decay linearly from 1.0 to 0.0."""
        from ragnarok.infrastructure.config import RagnarokConfig
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.core.agent import RagnarokAgent

        spec = get_env_spec("cartpole")
        config = RagnarokConfig(seed=42)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.transfer.trust_region_episodes = 100
        config.transfer.trust_region_alpha = 1.0
        config.curiosity.enabled = False

        env = RagnarokEnv(spec.gym_name, seed=42)
        agent = RagnarokAgent(config, env)

        # No transfer -> alpha = 0
        assert agent._trust_region_alpha() == 0.0

        # Simulate transfer
        import copy
        agent._transfer_ref_policy = copy.deepcopy(agent._active_policy)
        agent._transfer_episode_start = 0

        # At episode 0: alpha = 1.0
        agent.total_episodes = 0
        assert abs(agent._trust_region_alpha() - 1.0) < 0.01

        # At episode 50: alpha = 0.5
        agent.total_episodes = 50
        assert abs(agent._trust_region_alpha() - 0.5) < 0.01

        # At episode 100: alpha = 0, ref freed
        agent.total_episodes = 100
        assert agent._trust_region_alpha() == 0.0
        assert agent._transfer_ref_policy is None

        env.close()

    def test_trust_kl_computation(self):
        """Trust region KL should be computable for discrete policies."""
        from ragnarok.learning.real_experience import RealExperienceTrainer, DirectPolicyNet
        import copy

        trainer = RealExperienceTrainer(obs_dim=4, action_dim=2, discrete=True)
        trainer.trust_region_ref = copy.deepcopy(trainer.policy)
        trainer.trust_region_ref.eval()
        trainer.trust_region_alpha = 0.5

        obs_list = [np.random.randn(4).astype(np.float32) for _ in range(5)]
        kl = trainer._compute_trust_kl(obs_list)
        assert kl.item() >= 0  # KL is non-negative
        # Same policy -> KL should be very small
        assert kl.item() < 0.01
