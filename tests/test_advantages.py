"""Pinning tests for canonical advantage / lambda-return computations.

Locks the sign convention, bootstrap behavior, and numeric correctness of
the two canonical functions in `ragnarok/learning/advantages.py`. These are
the primitives every trainer (real, dream, dreamer) now shares — a silent
regression here would confound every cross-method benchmark comparison.
"""

import numpy as np
import torch

from ragnarok.learning.advantages import compute_gae, compute_lambda_returns


class TestComputeGAE:
    """Standard GAE for real-experience on-policy training (PPO/A2C)."""

    def test_zero_rewards_zero_values_gives_zero_advantages(self):
        r = np.zeros(5, dtype=np.float32)
        v = np.zeros(5, dtype=np.float32)
        d = np.zeros(5, dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=0.0)
        assert np.allclose(adv, 0.0)
        assert np.allclose(ret, 0.0)

    def test_output_shapes(self):
        n = 10
        r = np.random.randn(n).astype(np.float32)
        v = np.random.randn(n).astype(np.float32)
        d = np.zeros(n, dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=0.5)
        assert adv.shape == (n,)
        assert ret.shape == (n,)
        assert adv.dtype == np.float32
        assert ret.dtype == np.float32

    def test_returns_equal_advantages_plus_values(self):
        n = 8
        r = np.random.randn(n).astype(np.float32)
        v = np.random.randn(n).astype(np.float32)
        d = np.zeros(n, dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=0.3)
        np.testing.assert_allclose(ret, adv + v, rtol=1e-5)

    def test_done_flag_terminates_bootstrap(self):
        """When dones[t] = 1, no future value contributes to advantage[t]."""
        r = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        v = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        d = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # terminal at t=1
        adv, _ = compute_gae(r, v, d, last_value=5.0, gamma=0.99, lam=0.95)
        # At t=1 with dones=1: delta = r - v = 1 - 10 = -9; gae = -9 (no future)
        assert adv[1] == np.float32(-9.0)

    def test_single_step_episode_delta_exact(self):
        """n=1: advantage = r + gamma*last_value - v."""
        r = np.array([2.0], dtype=np.float32)
        v = np.array([1.0], dtype=np.float32)
        d = np.array([0.0], dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=5.0, gamma=0.9, lam=0.95)
        expected_adv = 2.0 + 0.9 * 5.0 - 1.0  # = 5.5
        expected_ret = expected_adv + 1.0      # = 6.5
        np.testing.assert_allclose(adv, [expected_adv], rtol=1e-5)
        np.testing.assert_allclose(ret, [expected_ret], rtol=1e-5)

    def test_gamma_lam_one_gives_monte_carlo_returns(self):
        """With gamma=lam=1 and zero values, returns should be cumsum from end."""
        r = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        v = np.zeros(4, dtype=np.float32)
        d = np.zeros(4, dtype=np.float32)
        adv, ret = compute_gae(r, v, d, last_value=0.0, gamma=1.0, lam=1.0)
        expected_ret = np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(ret, expected_ret, rtol=1e-5)

    def test_does_not_normalize(self):
        """compute_gae must NOT mean-center or std-scale advantages — that is
        the caller's responsibility."""
        r = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        v = np.zeros(3, dtype=np.float32)
        d = np.zeros(3, dtype=np.float32)
        adv, _ = compute_gae(r, v, d, last_value=0.0, gamma=1.0, lam=1.0)
        assert adv.mean() > 5.0  # raw scale preserved


class TestComputeLambdaReturns:
    """Dreamer v2 lambda-returns for imagination rollouts."""

    def test_output_shape(self):
        B, H = 3, 5
        rewards = torch.randn(B, H)
        values = torch.randn(B, H + 1)
        continues = torch.ones(B, H)
        returns = compute_lambda_returns(rewards, values, continues)
        assert returns.shape == (B, H)

    def test_gamma_lam_one_all_continues_gives_cumulative_from_end(self):
        """Constant r=1, continues=1, gamma=lam=1, values=0: returns = (horizon - t)."""
        B, H = 2, 5
        rewards = torch.ones(B, H)
        values = torch.zeros(B, H + 1)
        continues = torch.ones(B, H)
        returns = compute_lambda_returns(rewards, values, continues, gamma=1.0, lam=1.0)
        expected = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0],
                                 [5.0, 4.0, 3.0, 2.0, 1.0]])
        torch.testing.assert_close(returns, expected, atol=1e-5, rtol=1e-5)

    def test_zero_continues_collapses_to_single_step(self):
        """continues=0 at every step → V_t = r_t (no future propagates)."""
        B, H = 1, 4
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        values = torch.tensor([[100.0, 100.0, 100.0, 100.0, 999.0]])
        continues = torch.zeros(B, H)
        returns = compute_lambda_returns(rewards, values, continues)
        torch.testing.assert_close(returns, rewards, atol=1e-5, rtol=1e-5)

    def test_bootstraps_from_last_value_when_lam_zero(self):
        """lam=0, continues=1: V_t = r_t + gamma * v_{t+1} (pure TD(0))."""
        rewards = torch.tensor([[1.0, 1.0]])
        values = torch.tensor([[0.0, 10.0, 20.0]])
        continues = torch.ones(1, 2)
        returns = compute_lambda_returns(rewards, values, continues,
                                         gamma=1.0, lam=0.0)
        # t=1: V_1 = r_1 + 1*1*((1-0)*v_2 + 0*last) = 1 + 20 = 21
        # t=0: V_0 = r_0 + 1*1*((1-0)*v_1 + 0*last) = 1 + 10 = 11
        expected = torch.tensor([[11.0, 21.0]])
        torch.testing.assert_close(returns, expected, atol=1e-5, rtol=1e-5)

    def test_two_step_recursion_exact(self):
        """Hand-computed 2-step case pinning the exact recursion formula."""
        rewards = torch.tensor([[2.0, 3.0]])
        values = torch.tensor([[1.0, 5.0, 7.0]])
        continues = torch.tensor([[0.9, 0.8]])
        gamma, lam = 0.99, 0.95
        # t=1 (last): V_1 = r_1 + gamma*c_1 * ((1-lam)*v_2 + lam*v_2) = r_1 + gamma*c_1*v_2
        V1 = 3.0 + 0.99 * 0.8 * 7.0
        # t=0: V_0 = r_0 + gamma*c_0 * ((1-lam)*v_1 + lam*V_1)
        V0 = 2.0 + 0.99 * 0.9 * ((1 - 0.95) * 5.0 + 0.95 * V1)
        expected = torch.tensor([[V0, V1]])
        returns = compute_lambda_returns(rewards, values, continues, gamma, lam)
        torch.testing.assert_close(returns, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_through_rewards_and_continues(self):
        """Gradients must propagate through r and continues (Dreamer v2)."""
        rewards = torch.ones(1, 3, requires_grad=True)
        values = torch.zeros(1, 4).detach()
        continues = torch.full((1, 3), 0.9, requires_grad=True)
        returns = compute_lambda_returns(rewards, values, continues)
        returns.sum().backward()
        assert rewards.grad is not None
        assert (rewards.grad != 0).any()
        assert continues.grad is not None
        assert (continues.grad != 0).any()


class TestCallSitesUseCanonical:
    """Smoke test: every former owner now imports from advantages.py."""

    def test_real_experience_imports_compute_gae(self):
        from ragnarok.learning import real_experience
        from ragnarok.learning.advantages import compute_gae
        assert real_experience.compute_gae is compute_gae

    def test_dream_augmenter_imports_compute_lambda_returns(self):
        from ragnarok.learning import dream_augmenter
        from ragnarok.learning.advantages import compute_lambda_returns
        assert dream_augmenter.compute_lambda_returns is compute_lambda_returns

    def test_dreamer_imports_compute_lambda_returns(self):
        from ragnarok.learning import dreamer
        from ragnarok.learning.advantages import compute_lambda_returns
        assert dreamer.compute_lambda_returns is compute_lambda_returns
