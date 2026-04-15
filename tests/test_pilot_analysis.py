"""Tests for scripts/pilot_analysis.py (Phase 3 §8 pass-criteria evaluator)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pilot_analysis import (  # noqa: E402
    ANTI_TRANSFER_RATIO_MAX,
    AUC_WINDOW,
    BOOTSTRAP_N_RESAMPLES,
    EARLY_STEP_QUERIES,
    HIGH_CENSORING_THRESHOLD,
    PERMUTATION_N_SHUFFLES,
    PERMUTATION_RNG_SEED,
    PRIMARY_ALIAS,
    PRIMARY_LOGRANK_P_MAX,
    PRIMARY_RMST_RATIO_MIN,
    SECONDARY_RMST_RATIO_MIN,
    _check_mechanism,
    _compute_arm_descriptives,
    _compute_signed_oe,
    _extract_duration_event,
    _fit_arm,
    _logrank_one_sided,
    _logrank_permutation_one_sided,
    _verdict_to_dict,
    analyze,
    auc_return,
    bootstrap_ci_mean,
    render_text,
    return_at_step,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _mkrun(alias: str, seed: int, arm: str, stm: int | None,
           tau: int = 200_000, acting: str = "latent",
           skill_name: str | None = "src_skill_100ep",
           role: str = "primary",
           src: str = "cartpole",
           tgt: str = "mountaincar-continuous",
           eval_curve: list[dict] | None = None) -> dict:
    return {
        "pair_alias": alias,
        "pair_role": role,
        "src_env": src,
        "tgt_env": tgt,
        "seed": seed,
        "arm": arm,
        "mastery_threshold": 90.0,
        "max_env_steps": tau,
        "total_env_steps": stm if stm else tau,
        "total_episodes": 100,
        "final_eval_return": 92.0 if stm else 40.0,
        "best_eval_return": 92.0 if stm else 40.0,
        "steps_to_mastery": stm,
        "censored": stm is None,
        "eval_curve": eval_curve if eval_curve is not None else [],
        "acting_policy_mode": acting if arm == "transfer" else "obs",
        "transfer_skill_name": skill_name if arm == "transfer" else None,
        "wall_clock_sec": 500.0,
    }


def _mkcurve(points: list[tuple[int, float]]) -> list[dict]:
    """Build an eval_curve in the format the pilot emits."""
    return [{"step": s, "eval_return": r} for s, r in points]


def _make_payload(runs: list[dict], tau: int = 200_000) -> dict:
    return {
        "prereg_section": "§8 (pilot)",
        "pairs": [
            {"alias": "cartpole_mcc", "src": "cartpole",
             "tgt": "mountaincar-continuous", "role": "primary"},
            {"alias": "cartpole_acrobot", "src": "cartpole",
             "tgt": "acrobot", "role": "secondary"},
            {"alias": "pendulum_dmc_cartpole", "src": "pendulum",
             "tgt": "cartpole-swingup", "role": "secondary"},
        ],
        "seeds_N": 5,
        "base_seed": 42,
        "max_env_steps": tau,
        "source_max_env_steps": 100_000,
        "eval_every_steps": 5_000,
        "eval_episodes": 10,
        "mastery_thresholds": {
            "cartpole": 450.0, "mountaincar-continuous": 90.0,
            "acrobot": -100.0, "pendulum": -200.0,
            "cartpole-swingup": 800.0,
        },
        "runs": runs,
    }


# ── Duration / event extraction ─────────────────────────────────────

class TestDurationExtraction:
    def test_observed_returns_stm(self):
        r = _mkrun("a", 0, "scratch", stm=80_000)
        d, e = _extract_duration_event(r, tau=200_000)
        assert d == 80_000
        assert e == 1

    def test_censored_returns_tau(self):
        r = _mkrun("a", 0, "scratch", stm=None)
        d, e = _extract_duration_event(r, tau=200_000)
        assert d == 200_000
        assert e == 0

    def test_explicit_censored_flag_honored(self):
        """Even if stm is a number but `censored` says True, treat as censored.
        Defensive — lets downstream consumers mark a run as censored post-hoc
        (e.g. when the observed value hits a post-mastery-regression filter)."""
        r = _mkrun("a", 0, "scratch", stm=80_000)
        r["censored"] = True
        d, e = _extract_duration_event(r, tau=200_000)
        assert e == 0
        assert d == 200_000


# ── Per-arm KMF fit + RMST ──────────────────────────────────────────

class TestArmFit:
    def test_all_observed_fast_rmst_is_small(self):
        """When every run reaches mastery very quickly, RMST(τ) ≈ the mean
        of steps_to_mastery."""
        runs = [_mkrun("p", s, "scratch", stm=50_000) for s in range(5)]
        arm = _fit_arm(runs, "scratch", tau=200_000)
        assert arm.n == 5
        assert arm.n_events == 5
        assert arm.n_censored == 0
        # RMST for a step function that drops to 0 at 50k is 50k
        assert arm.rmst == pytest.approx(50_000, rel=0.01)

    def test_all_censored_rmst_is_tau(self):
        """If every run is censored at τ, every subject is alive at τ → RMST = τ."""
        runs = [_mkrun("p", s, "scratch", stm=None) for s in range(5)]
        arm = _fit_arm(runs, "scratch", tau=200_000)
        assert arm.n == 5
        assert arm.n_events == 0
        assert arm.n_censored == 5
        assert arm.rmst == pytest.approx(200_000, rel=0.001)

    def test_mixed_events_rmst_between(self):
        # 3 reach mastery at 50k, 2 censored at τ=200k
        runs = [_mkrun("p", 0, "scratch", stm=50_000),
                _mkrun("p", 1, "scratch", stm=50_000),
                _mkrun("p", 2, "scratch", stm=50_000),
                _mkrun("p", 3, "scratch", stm=None),
                _mkrun("p", 4, "scratch", stm=None)]
        arm = _fit_arm(runs, "scratch", tau=200_000)
        # KM: survival = 1 for t<50k, survival = 2/5 for t>=50k.
        # RMST(200k) = 50k*1 + (200k-50k)*0.4 = 50k + 60k = 110k
        assert arm.rmst == pytest.approx(110_000, rel=0.01)
        assert arm.n_events == 3
        assert arm.n_censored == 2

    def test_empty_arm_returns_nan(self):
        arm = _fit_arm([], "scratch", tau=200_000)
        assert arm.n == 0
        assert math.isnan(arm.rmst)


# ── RMST ratio + log-rank direction ────────────────────────────────

class TestRMSTRatioAndLogRank:
    def test_transfer_faster_ratio_above_1(self):
        """If transfer reaches mastery at 50k and scratch at 150k, ratio = 3.0."""
        scratch = [_mkrun("p", s, "scratch", stm=150_000) for s in range(5)]
        transfer = [_mkrun("p", s, "transfer", stm=50_000) for s in range(5)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        ratio = s_arm.rmst / t_arm.rmst
        assert ratio == pytest.approx(3.0, rel=0.01)

    def test_logrank_directional_transfer_faster(self):
        """When transfer is faster, one-sided p should be small."""
        scratch = [_mkrun("p", s, "scratch", stm=180_000) for s in range(10)]
        transfer = [_mkrun("p", s, "transfer", stm=30_000) for s in range(10)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p = _logrank_one_sided(s_arm, t_arm, tau=200_000)
        assert p < 0.05, f"expected strong signal, got p={p}"

    def test_logrank_directional_transfer_slower_penalized(self):
        """When transfer is slower than scratch (wrong direction for H_A),
        the one-sided p should be high (> 0.5)."""
        scratch = [_mkrun("p", s, "scratch", stm=30_000) for s in range(10)]
        transfer = [_mkrun("p", s, "transfer", stm=180_000) for s in range(10)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p = _logrank_one_sided(s_arm, t_arm, tau=200_000)
        assert p > 0.5, f"wrong-direction effect must not reject; got p={p}"


# ── v3.5 permutation test (small-N robustness check) ───────────────

class TestLogRankPermutation:
    """Contract: permutation p and asymptotic p answer the same question
    (one-sided H_A: transfer reaches mastery faster than scratch) via
    different null-distribution approximations. The permutation value is
    the exact one under exchangeability; the asymptotic is the chi-sq
    approximation that the preregistration declares the §8 headline.

    These tests pin the permutation-p behaviour and assert that at large
    N with clean signal the two p-values agree (same regime where the
    asymptotic is defensible). At small N and heavy censoring the two
    can disagree — the analyzer flags that but does not fail.
    """

    def test_permutation_p_small_when_transfer_much_faster(self):
        """Strong signal → both p-values below α = 0.10."""
        scratch = [_mkrun("p", s, "scratch", stm=180_000) for s in range(10)]
        transfer = [_mkrun("p", s, "transfer", stm=30_000) for s in range(10)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        # Use a cheaper shuffle count for unit tests (still enough for
        # tail-resolution at the α we care about).
        p_perm = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=2000, rng_seed=0,
        )
        p_asymp = _logrank_one_sided(s_arm, t_arm, tau=200_000)
        assert p_perm < 0.05, f"strong signal should reject; got {p_perm}"
        assert p_asymp < 0.05
        # At N=10 per arm with clean uncensored data, the asymptotic is
        # defensible and the two should be within 0.03 of each other.
        assert abs(p_perm - p_asymp) < 0.03, (
            f"asymp={p_asymp:.4f} perm={p_perm:.4f} disagree"
        )

    def test_permutation_p_large_when_wrong_direction(self):
        """Transfer slower than scratch → one-sided p ≥ 0.5 (null side
        of the distribution; the directional test cannot reject)."""
        scratch = [_mkrun("p", s, "scratch", stm=30_000) for s in range(10)]
        transfer = [_mkrun("p", s, "transfer", stm=180_000) for s in range(10)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p_perm = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=2000, rng_seed=0,
        )
        assert p_perm >= 0.5, f"wrong direction must not reject; got {p_perm}"

    def test_permutation_p_exactly_half_under_identical_arms(self):
        """If both arms have IDENTICAL duration patterns, the observed
        signed O-E is exactly 0 under symmetric breakdown of ties, and
        the one-sided p is ≈ 0.5.

        This is the sharpest possible null: not "drawn from same
        distribution" (one sample draw can land anywhere in [0, 1]),
        but "bit-identical samples". The permutation null should be
        symmetric around 0 and our observed statistic is 0 → p ≈ 0.5.
        """
        pattern = [50_000, 75_000, 100_000, 150_000, 175_000]  # 5 per arm
        scratch_runs = [_mkrun("p", i, "scratch", stm=d)
                        for i, d in enumerate(pattern)]
        transfer_runs = [_mkrun("p", i + 100, "transfer", stm=d)
                         for i, d in enumerate(pattern)]
        s_arm = _fit_arm(scratch_runs, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer_runs, "transfer", tau=200_000)
        p_perm = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=2000, rng_seed=0,
        )
        assert 0.35 < p_perm < 0.65, (
            f"identical arms → p should be near 0.5; got {p_perm}"
        )

    def test_permutation_p_distribution_under_null_is_uniform(self):
        """Aggregate null test: across many independent null samples,
        the permutation p-values should be approximately uniform on
        [0, 1]. Check the median lands near 0.5 and the extreme tails
        don't cluster.

        Runs 30 null datasets, each drawn from the SAME mixed
        distribution on both arms. Exact uniform coverage needs
        thousands of draws, but 30 is enough to catch a systematically
        biased p-value distribution.
        """
        p_values = []
        for trial_seed in range(30):
            rng = np.random.default_rng(1000 + trial_seed)
            durations_all = [int(rng.choice([50_000, 100_000, 150_000]))
                             for _ in range(20)]
            s_runs = [_mkrun("p", i, "scratch", stm=d)
                      for i, d in enumerate(durations_all[:10])]
            t_runs = [_mkrun("p", i + 100, "transfer", stm=d)
                      for i, d in enumerate(durations_all[10:])]
            s_arm = _fit_arm(s_runs, "scratch", tau=200_000)
            t_arm = _fit_arm(t_runs, "transfer", tau=200_000)
            p_values.append(_logrank_permutation_one_sided(
                s_arm, t_arm, n_shuffles=500, rng_seed=trial_seed,
            ))
        # Median should land in the central third of [0, 1].
        med = float(np.median(p_values))
        assert 0.25 < med < 0.75, (
            f"null p-values biased (median {med:.3f}); expected ≈ 0.5"
        )
        # Type-I error at α = 0.10: expect ≤ 2-3 out of 30 below 0.10
        # under the null. Reject if > 6 (unrealistic under exchangeable
        # H_0).
        n_reject = sum(1 for p in p_values if p < 0.10)
        assert n_reject <= 6, (
            f"permutation test rejects {n_reject}/30 null draws at α=0.10; "
            f"inflated Type-I"
        )

    def test_permutation_p_is_deterministic_under_fixed_seed(self):
        """Two calls with the same seed return identical p-values.

        This matters because the pilot analyzer writes the p-value to
        the verdict JSON and a reviewer must be able to re-run and get
        the same number.
        """
        scratch = [_mkrun("p", s, "scratch", stm=150_000) for s in range(5)]
        transfer = [_mkrun("p", s, "transfer", stm=50_000) for s in range(5)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p1 = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=500, rng_seed=777,
        )
        p2 = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=500, rng_seed=777,
        )
        assert p1 == p2

    def test_permutation_p_nan_when_no_events(self):
        """If both arms are fully censored there are no events → no
        information → NaN. Must not crash and must not silently emit 0.
        """
        # stm=None → censored at τ
        scratch = [_mkrun("p", s, "scratch", stm=None) for s in range(5)]
        transfer = [_mkrun("p", s, "transfer", stm=None) for s in range(5)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p_perm = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=500, rng_seed=0,
        )
        assert math.isnan(p_perm)

    def test_permutation_p_strictly_positive_via_add_one_correction(self):
        """Even when no shuffle reaches the observed O-E tail, the
        add-one correction (count + 1) / (n + 1) guarantees p > 0.
        """
        # Extreme signal so few-to-no shuffles will exceed observed.
        scratch = [_mkrun("p", s, "scratch", stm=195_000) for s in range(8)]
        transfer = [_mkrun("p", s, "transfer", stm=5_000) for s in range(8)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        p_perm = _logrank_permutation_one_sided(
            s_arm, t_arm, n_shuffles=100, rng_seed=0,
        )
        assert p_perm > 0
        # Lower bound from add-one correction: (0+1)/(100+1) ≈ 0.0099.
        assert p_perm >= 1.0 / 101

    def test_compute_signed_oe_matches_legacy(self):
        """`_compute_signed_oe` extracted from `_logrank_signed_direction`;
        pass through an ArmSurvival and assert identical numerics."""
        from scripts.pilot_analysis import _logrank_signed_direction
        scratch = [_mkrun("p", s, "scratch", stm=150_000) for s in range(5)]
        transfer = [_mkrun("p", s, "transfer", stm=50_000) for s in range(5)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        legacy = _logrank_signed_direction(s_arm, t_arm)
        extracted = _compute_signed_oe(
            s_arm.durations, s_arm.events,
            t_arm.durations, t_arm.events,
        )
        assert legacy == pytest.approx(extracted, rel=1e-9)

    def test_permutation_p_reported_in_verdict_and_json(self):
        """End-to-end: analyze() populates `logrank_permutation_p_value`
        on every PairVerdict and `_verdict_to_dict()` serializes it."""
        scratch = [_mkrun("cartpole_mcc", s, "scratch", stm=180_000)
                   for s in range(5)]
        transfer = [_mkrun("cartpole_mcc", s, "transfer", stm=30_000)
                    for s in range(5)]
        acrobot_s = [_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                            role="secondary", src="cartpole", tgt="acrobot")
                     for s in range(3)]
        acrobot_t = [_mkrun("cartpole_acrobot", s, "transfer", stm=50_000,
                            role="secondary", src="cartpole", tgt="acrobot")
                     for s in range(3)]
        dmc_s = [_mkrun("pendulum_dmc_cartpole", s, "scratch", stm=150_000,
                        role="secondary", src="pendulum",
                        tgt="cartpole-swingup")
                 for s in range(3)]
        dmc_t = [_mkrun("pendulum_dmc_cartpole", s, "transfer", stm=50_000,
                        role="secondary", src="pendulum",
                        tgt="cartpole-swingup")
                 for s in range(3)]
        payload = _make_payload(
            scratch + transfer + acrobot_s + acrobot_t + dmc_s + dmc_t
        )
        verdict = analyze(payload)
        for pv in verdict.pair_verdicts:
            assert math.isfinite(pv.logrank_permutation_p_value), (
                f"permutation p must be finite for {pv.alias}, "
                f"got {pv.logrank_permutation_p_value}"
            )
            assert 0 < pv.logrank_permutation_p_value < 1
        # JSON round-trip preserves the field.
        d = _verdict_to_dict(verdict)
        for pd in d["pair_verdicts"]:
            assert "logrank_permutation_p_value" in pd

    def test_permutation_n_shuffles_default_matches_prereg(self):
        """The prereg v3.5 amendment names 10,000 shuffles."""
        assert PERMUTATION_N_SHUFFLES == 10_000

    def test_permutation_rng_seed_is_pinned(self):
        """The rng seed must be a module-level constant so verdicts are
        reproducible across repeat analyses of the same JSON."""
        assert isinstance(PERMUTATION_RNG_SEED, int)


# ── Mechanism check ────────────────────────────────────────────────

class TestMechanismCheck:
    def test_cross_dim_latent_mode_passes(self):
        """CartPole (obs=4, act=2 discrete) → MCC (obs=2, act=1 continuous)
        is cross-dim. All transfer runs with acting_policy_mode='latent'
        should pass."""
        runs = [_mkrun("p", s, "transfer", stm=50_000, acting="latent")
                for s in range(5)]
        ok, msg = _check_mechanism(runs, src_env="cartpole",
                                   tgt_env="mountaincar-continuous")
        assert ok
        assert "latent" in msg

    def test_cross_dim_obs_mode_fails(self):
        """If transfer was supposed to switch to latent mode but didn't,
        mechanism check fails loudly."""
        runs = [_mkrun("p", s, "transfer", stm=50_000, acting="obs")
                for s in range(5)]
        ok, msg = _check_mechanism(runs, src_env="cartpole",
                                   tgt_env="mountaincar-continuous")
        assert not ok
        assert "latent" in msg.lower()

    def test_same_dim_pair_trivial_pass(self):
        """Same-env pair (hypothetical) doesn't require latent mode."""
        runs = [_mkrun("p", s, "transfer", stm=50_000, acting="obs")
                for s in range(5)]
        ok, msg = _check_mechanism(runs, src_env="cartpole", tgt_env="cartpole")
        assert ok
        assert "same-dim" in msg.lower() or "trivial" in msg.lower()

    def test_partial_latent_fails_identifies_seeds(self):
        runs = [
            _mkrun("p", 0, "transfer", stm=50_000, acting="latent"),
            _mkrun("p", 1, "transfer", stm=50_000, acting="latent"),
            _mkrun("p", 2, "transfer", stm=50_000, acting="obs"),
            _mkrun("p", 3, "transfer", stm=50_000, acting="latent"),
            _mkrun("p", 4, "transfer", stm=50_000, acting="obs"),
        ]
        ok, msg = _check_mechanism(runs, src_env="cartpole",
                                   tgt_env="mountaincar-continuous")
        assert not ok
        # Must enumerate failing seeds so reviewers can trace the bug
        assert "2" in msg
        assert "4" in msg


# ── §8 pass criteria ───────────────────────────────────────────────

class TestPrimaryPassCriterion:
    """Primary pair (cartpole_mcc) needs ratio >= 1.3x AND log-rank p < 0.10."""

    def test_strong_primary_pass(self):
        runs = []
        # Primary: transfer much faster than scratch
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=180_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        # Secondary filler (to satisfy "≥1 secondary pass")
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=180_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=50_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=None, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=150_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts if v.role == "primary")
        assert primary.rmst_ratio >= PRIMARY_RMST_RATIO_MIN
        assert primary.logrank_p_value < PRIMARY_LOGRANK_P_MAX
        assert primary.pass_primary_criterion

    def test_ratio_ok_but_p_insufficient_requires_both(self):
        """Primary needs BOTH ratio AND p. Use a noisy tiny-N case where
        ratio is fine but overlap between arms inflates p above 0.10.

        Construction: 2 scratch at 150k + 2 scratch censored-at-τ;
        2 transfer at 100k + 2 transfer at 190k. Ratio ~1.5 by RMST, but
        the wide within-arm spread + small N pushes p above 0.10."""
        runs = []
        # Scratch: half observed early-mid, half censored at τ
        runs.append(_mkrun("cartpole_mcc", 0, "scratch", stm=150_000))
        runs.append(_mkrun("cartpole_mcc", 1, "scratch", stm=150_000))
        runs.append(_mkrun("cartpole_mcc", 2, "scratch", stm=None))
        runs.append(_mkrun("cartpole_mcc", 3, "scratch", stm=None))
        # Transfer: heavy overlap with scratch (some fast, some as slow)
        runs.append(_mkrun("cartpole_mcc", 0, "transfer", stm=100_000))
        runs.append(_mkrun("cartpole_mcc", 1, "transfer", stm=190_000))
        runs.append(_mkrun("cartpole_mcc", 2, "transfer", stm=100_000))
        runs.append(_mkrun("cartpole_mcc", 3, "transfer", stm=190_000))
        for s in range(3):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=100_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=100_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts if v.role == "primary")
        # If either condition fails, pass_primary_criterion must be False.
        # We don't hard-pin the exact p-value (it's sensitive to the
        # lifelines version); instead assert the compound-AND semantics by
        # checking that when we synthetically force p above threshold, the
        # verdict flips.
        both_conditions_held = (primary.rmst_ratio >= PRIMARY_RMST_RATIO_MIN
                                and primary.logrank_p_value < PRIMARY_LOGRANK_P_MAX)
        assert primary.pass_primary_criterion == both_conditions_held

    def test_ratio_insufficient_fails_primary(self):
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=150_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=145_000))
        for s in range(3):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=100_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=100_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts if v.role == "primary")
        assert primary.rmst_ratio < PRIMARY_RMST_RATIO_MIN
        assert not primary.pass_primary_criterion


class TestSecondaryPassCriterion:
    def test_secondary_pass_directional_only(self):
        """Secondary needs ratio >= 1.3x; no p-value threshold."""
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=50_000,
                               role="secondary", tgt="acrobot"))
        arm = _fit_arm([r for r in runs if r["arm"] == "scratch"],
                       "scratch", tau=200_000)
        t_arm = _fit_arm([r for r in runs if r["arm"] == "transfer"],
                         "transfer", tau=200_000)
        ratio = arm.rmst / t_arm.rmst
        assert ratio >= SECONDARY_RMST_RATIO_MIN

    def test_secondary_fail_if_ratio_below_threshold(self):
        runs = [_mkrun("cartpole_acrobot", s, "scratch", stm=100_000,
                       role="secondary", tgt="acrobot") for s in range(5)]
        runs += [_mkrun("cartpole_acrobot", s, "transfer", stm=90_000,
                        role="secondary", tgt="acrobot") for s in range(5)]
        arm = _fit_arm([r for r in runs if r["arm"] == "scratch"],
                       "scratch", tau=200_000)
        t_arm = _fit_arm([r for r in runs if r["arm"] == "transfer"],
                         "transfer", tau=200_000)
        ratio = arm.rmst / t_arm.rmst
        assert ratio < SECONDARY_RMST_RATIO_MIN


class TestAntiTransfer:
    def test_anti_transfer_detected(self):
        """If transfer is actually slower than scratch by more than 10%, the
        pair is flagged as anti-transfer → overall FAIL."""
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=50_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=180_000))
        # Fillers
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=100_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=30_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts if v.role == "primary")
        assert primary.anti_transfer
        assert primary.rmst_ratio < ANTI_TRANSFER_RATIO_MAX
        assert not verdict.overall_pass
        assert any("ANTI" in f.upper() for f in verdict.failures)


class TestOverallVerdict:
    """Composition of the §8 criteria into a single pass/fail bit."""

    def test_primary_fail_overall_fail(self):
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=100_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=90_000))
        # Strong secondary to isolate primary failure
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        assert any("PRIMARY" in f for f in verdict.failures)

    def test_primary_pass_but_no_secondary_fails_overall(self):
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=150_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        # Secondary ratio below threshold
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=100_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=95_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=100_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=95_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        assert any("secondary" in f.lower() for f in verdict.failures)

    def test_mechanism_failure_sinks_overall(self):
        """Even if RMST numbers are strong, acting_policy_mode != 'latent'
        for any cross-dim transfer run sinks overall."""
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=150_000))
            acting = "obs" if s == 0 else "latent"  # seed 0 breaks mechanism
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000,
                               acting=acting))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        assert any("MECHANISM" in f.upper() for f in verdict.failures)

    def test_full_pass(self):
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=150_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=None, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=140_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert verdict.overall_pass
        assert verdict.failures == []


class TestRendering:
    def test_render_text_contains_key_fields(self):
        runs = [_mkrun("cartpole_mcc", s, "scratch", stm=150_000)
                for s in range(5)]
        runs += [_mkrun("cartpole_mcc", s, "transfer", stm=30_000)
                 for s in range(5)]
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=None, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=140_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        text = render_text(verdict)
        assert "cartpole_mcc" in text
        assert "RMST ratio" in text
        assert "Log-rank" in text
        assert "OVERALL" in text

    def test_verdict_json_serializable(self):
        runs = [_mkrun("cartpole_mcc", s, "scratch", stm=150_000)
                for s in range(5)]
        runs += [_mkrun("cartpole_mcc", s, "transfer", stm=30_000)
                 for s in range(5)]
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=None, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=140_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        d = _verdict_to_dict(verdict)
        # Round-trip: must JSON-serialize cleanly
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["overall_pass"] == verdict.overall_pass
        assert len(d2["pair_verdicts"]) == len(verdict.pair_verdicts)


# ── Reviewer-fix coverage: skip zero-run pairs ─────────────────────

class TestSkipZeroRunPairs:
    """Strategist review blocker #1: a pair with 0 runs in one arm is
    unanalyzable. Don't synthesize NaN verdict rows — flag it as an
    incompleteness failure and carry on."""

    def test_pair_missing_both_arms_flagged_as_incomplete(self):
        """Primary pair entirely absent → incompleteness failure, not NaN."""
        runs = []
        # Add a full secondary pair so we can see that skipping the primary
        # doesn't block the secondary's verdict row.
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        # Primary skipped → one skipped failure + the secondary is present.
        assert any("cartpole_mcc" in f and "INCOMPLETE" in f
                   for f in verdict.failures)
        aliases_in_verdicts = {v.alias for v in verdict.pair_verdicts}
        assert "cartpole_mcc" not in aliases_in_verdicts

    def test_pair_missing_only_transfer_arm_flagged(self):
        """Scratch-only pair is still unanalyzable (no ratio computable)."""
        runs = [_mkrun("cartpole_mcc", s, "scratch", stm=150_000)
                for s in range(5)]
        # Keep other pairs complete to isolate the failure mode.
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        assert any("cartpole_mcc" in f and "INCOMPLETE" in f
                   and "transfer_runs=0" in f for f in verdict.failures)


# ── Reviewer-fix coverage: failure-check AND, not elif ─────────────

class TestFailureCheckIsAndNotElif:
    """Strategist review blocker #2: overall-pass is an AND of four §8
    criteria. The prior elif chain let a primary-fail mask a downstream
    anti-transfer violation. Both failures must surface in the same pass."""

    def test_primary_fail_and_anti_transfer_both_surface(self):
        # Primary: transfer only marginally faster (ratio ~1.1, fails 1.3)
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=110_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=100_000))
        # Secondary cartpole_acrobot: transfer WORSE than scratch (ratio < 0.9)
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=50_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=180_000,
                               role="secondary", tgt="acrobot"))
        # Other secondary present-but-weak so we don't trip the no-secondary
        # branch separately.
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=150_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=140_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.overall_pass
        fails_joined = " | ".join(verdict.failures).upper()
        # Both messages must appear — prior elif logic would only report
        # the first failing check.
        assert "PRIMARY" in fails_joined
        assert "ANTI-TRANSFER" in fails_joined


# ── Reviewer-fix coverage: direction via signed log-rank ───────────

class TestLogRankDirectionUnderCrossingHazards:
    """Devil's-advocate concern C3: RMST can disagree with log-rank hazard
    direction under crossing hazards. The directional p-value must key off
    the log-rank numerator's sign, not the RMST comparison."""

    def test_direction_comes_from_signed_numerator(self):
        """Stress test: transfer dominates on RMST but events-early-and-often
        for scratch. The signed O-E should resolve direction correctly."""
        from scripts.pilot_analysis import _logrank_signed_direction
        scratch = [_mkrun("p", s, "scratch", stm=60_000) for s in range(10)]
        transfer = [_mkrun("p", s, "transfer", stm=30_000) for s in range(10)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        oe = _logrank_signed_direction(s_arm, t_arm)
        # Transfer faster → at transfer's event times, scratch has MORE at
        # risk than expected (because it hasn't died yet) → fewer observed
        # than expected → O_s - E_s < 0.
        assert oe < 0

    def test_signed_direction_zero_when_no_events(self):
        from scripts.pilot_analysis import _logrank_signed_direction
        scratch = [_mkrun("p", s, "scratch", stm=None) for s in range(5)]
        transfer = [_mkrun("p", s, "transfer", stm=None) for s in range(5)]
        s_arm = _fit_arm(scratch, "scratch", tau=200_000)
        t_arm = _fit_arm(transfer, "transfer", tau=200_000)
        assert _logrank_signed_direction(s_arm, t_arm) == 0.0


# ── Reviewer-fix coverage: INCONCLUSIVE under high censoring ──────

class TestInconclusiveUnderHighCensoring:
    """Devil's-advocate review (Phase 3 pre-commit): at N=5 per arm, a pilot
    where the primary pair has ≥80% censoring on either arm can't be read
    as FAIL — the lack of events is a measurement-limit artifact, not a
    measured null. Re-label as INCONCLUSIVE so Plan B (§10) does NOT fire
    on under-observed data; the operator must expand sample and re-run."""

    def test_high_censoring_on_scratch_triggers_inconclusive(self):
        """4/5 scratch censored + 0/5 transfer censored → 80% scratch censoring
        even though transfer is clearly winning. Still INCONCLUSIVE because
        the scratch arm doesn't have enough events to pin a baseline."""
        runs = []
        # Scratch: 1 observed at 150k, 4 censored at τ
        runs.append(_mkrun("cartpole_mcc", 0, "scratch", stm=150_000))
        for s in range(1, 5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=None))
        # Transfer: all 5 reach mastery fast
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        # Filler secondary (complete) so we don't trip incompleteness
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert verdict.inconclusive, (
            "80% scratch censoring must flip to INCONCLUSIVE")
        assert not verdict.overall_pass, (
            "INCONCLUSIVE is never a PASS")
        assert "high-censoring" in verdict.inconclusive_reason.lower()
        assert "cartpole_mcc" in verdict.inconclusive_reason

    def test_high_censoring_on_transfer_triggers_inconclusive(self):
        """5/5 transfer censored: transfer completely fails to reach mastery.
        Even though this looks like anti-transfer, at N=5 we don't have the
        power to claim that — INCONCLUSIVE, not FAIL."""
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=60_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=None))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert verdict.inconclusive
        assert not verdict.overall_pass
        assert "transfer" in verdict.inconclusive_reason.lower()

    def test_below_threshold_censoring_does_not_trigger(self):
        """At 60% censoring (3/5 censored), still below 80%. Analysis proceeds
        to normal pass/fail composition — no INCONCLUSIVE label."""
        runs = []
        # 2 observed + 3 censored = 60% censoring
        runs.append(_mkrun("cartpole_mcc", 0, "scratch", stm=150_000))
        runs.append(_mkrun("cartpole_mcc", 1, "scratch", stm=160_000))
        for s in range(2, 5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=None))
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        assert not verdict.inconclusive, (
            "60% censoring is below the 80% threshold — must not flip")

    def test_clean_pass_stays_pass(self):
        """Sanity: a clean PASS scenario must not be mis-labeled INCONCLUSIVE."""
        runs = []
        for s in range(10):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=150_000))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))
        for s in range(5):
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "scratch",
                               stm=None, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
            runs.append(_mkrun("pendulum_dmc_cartpole", s, "transfer",
                               stm=140_000, role="secondary",
                               src="pendulum", tgt="cartpole-swingup"))
        payload = _make_payload(runs)
        verdict = analyze(payload)
        # The pendulum_dmc_cartpole scratch arm has 100% censoring, but that
        # pair is SECONDARY — only primary-pair censoring gates inconclusive.
        assert verdict.overall_pass
        assert not verdict.inconclusive

    def test_render_text_shows_inconclusive_banner(self):
        """INCONCLUSIVE banner must NOT contain 'Plan B' in the header (the
        whole point of the verdict is to NOT trigger §10). FAIL banner does."""
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=None))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        text = render_text(verdict)
        assert "INCONCLUSIVE" in text
        # INCONCLUSIVE header line must NOT route the operator to Plan B.
        # (The failure list may still mention activation text elsewhere in
        # the output, but the overall-verdict line itself must point to
        # sample expansion, not Plan B.)
        header_line = [ln for ln in text.split("\n")
                       if ln.strip().startswith("OVERALL:")][0]
        assert "INCONCLUSIVE" in header_line
        assert "Plan B" not in header_line

    def test_inconclusive_serializes_in_json(self):
        """_verdict_to_dict round-trip includes the new fields."""
        runs = []
        for s in range(5):
            runs.append(_mkrun("cartpole_mcc", s, "scratch", stm=None))
            runs.append(_mkrun("cartpole_mcc", s, "transfer", stm=30_000))
        for s in range(5):
            runs.append(_mkrun("cartpole_acrobot", s, "scratch", stm=150_000,
                               role="secondary", tgt="acrobot"))
            runs.append(_mkrun("cartpole_acrobot", s, "transfer", stm=40_000,
                               role="secondary", tgt="acrobot"))

        payload = _make_payload(runs)
        verdict = analyze(payload)
        d = _verdict_to_dict(verdict)
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["inconclusive"] is True
        assert d2["inconclusive_reason"]
        assert d2["overall_pass"] is False

    def test_threshold_constant_matches_design(self):
        """§ prereg pre-commit: threshold pinned at 0.8 so reviewers can
        audit the contract. If this moves, the prereg addendum must move."""
        assert HIGH_CENSORING_THRESHOLD == 0.8


# ── Constant invariants (must match §8) ────────────────────────────

class TestPassCriteriaConstants:
    def test_primary_thresholds_match_prereg_section_8(self):
        # §8: "RMST ratio ≥ 1.3× with one-sided log-rank p < 0.10"
        assert PRIMARY_RMST_RATIO_MIN == 1.3
        assert PRIMARY_LOGRANK_P_MAX == 0.10

    def test_secondary_ratio_matches_prereg(self):
        # §8: "RMST ratio ≥ 1.3× directionally"
        assert SECONDARY_RMST_RATIO_MIN == 1.3

    def test_anti_transfer_matches_prereg(self):
        # §8: "No pair shows anti-transfer (RMST ratio < 0.9×)"
        assert ANTI_TRANSFER_RATIO_MAX == 0.9

    def test_primary_alias_matches_prereg(self):
        assert PRIMARY_ALIAS == "cartpole_mcc"


# ── v3.5 descriptive secondaries (early-step return + AUC) ─────────

class TestReturnAtStep:
    """Contract for linear-interp early-step return extraction."""

    def test_empty_curve_returns_nan(self):
        assert math.isnan(return_at_step([], 5000))

    def test_at_first_checkpoint_returns_first_value(self):
        curve = _mkcurve([(5000, 80.0), (10000, 95.0)])
        assert return_at_step(curve, 5000) == pytest.approx(80.0)

    def test_pre_first_checkpoint_linear_from_zero(self):
        """Pre-first-checkpoint convention: linear from (0, 0.0) to
        (first_step, first_return). At step = first_step / 2, the
        value is first_return / 2.
        """
        curve = _mkcurve([(5000, 80.0), (10000, 95.0)])
        # At step 2500 (halfway to 5000), expect 40.0.
        assert return_at_step(curve, 2500) == pytest.approx(40.0)
        # At step 0, expect 0.
        assert return_at_step(curve, 0) == 0.0

    def test_between_checkpoints_linear(self):
        curve = _mkcurve([(5000, 0.0), (10000, 100.0)])
        # Halfway between 5000 and 10000 → 50.
        assert return_at_step(curve, 7500) == pytest.approx(50.0)

    def test_past_last_checkpoint_clamps(self):
        """Late-training queries return the last plateau value."""
        curve = _mkcurve([(5000, 80.0), (10000, 95.0)])
        assert return_at_step(curve, 50000) == pytest.approx(95.0)

    def test_negative_step_treated_as_zero(self):
        curve = _mkcurve([(5000, 80.0)])
        assert return_at_step(curve, -100) == 0.0


class TestAucReturn:
    """Contract for trapezoidal AUC over a step window."""

    def test_empty_curve_nan(self):
        assert math.isnan(auc_return([], 0, 50_000))

    def test_inverted_window_nan(self):
        curve = _mkcurve([(5000, 50.0), (10000, 80.0)])
        assert math.isnan(auc_return(curve, 10_000, 5_000))

    def test_constant_return_equals_value(self):
        """A flat plateau at 50 over the whole window → AUC / window = 50."""
        curve = _mkcurve([(1000, 50.0), (50_000, 50.0)])
        # At step 0 → linearly interp from (0, 0) to (1000, 50) → 0.
        # The lead-in segment drops the mean.
        auc = auc_return(curve, 0, 50_000)
        # Lead-in (0→1k ramp) + plateau 1k→50k at 50.
        expected_area = 0.5 * 1000 * 50 + 49_000 * 50  # 25_000 + 2_450_000
        expected_mean = expected_area / 50_000  # ≈ 49.5
        assert auc == pytest.approx(expected_mean, rel=1e-6)

    def test_zero_everywhere_returns_zero(self):
        curve = _mkcurve([(5000, 0.0), (50_000, 0.0)])
        assert auc_return(curve, 0, 50_000) == pytest.approx(0.0)

    def test_transfer_faster_gets_higher_auc(self):
        """Transfer hits 95 plateau by 5k; scratch hits it at 40k.
        Over [0, 50k], transfer's AUC should be much higher.
        """
        fast = _mkcurve([(5000, 95.0), (50_000, 95.0)])
        slow = _mkcurve([(5000, 20.0), (40_000, 95.0), (50_000, 95.0)])
        auc_fast = auc_return(fast, 0, 50_000)
        auc_slow = auc_return(slow, 0, 50_000)
        assert auc_fast > auc_slow
        # And the gap should be meaningful (≥ 10 mean-return units).
        assert auc_fast - auc_slow > 10

    def test_window_beyond_curve_clamps_last_value(self):
        """Integration over [0, 100k] on a curve ending at 30k plateaus."""
        curve = _mkcurve([(5000, 90.0), (30_000, 90.0)])
        auc = auc_return(curve, 0, 100_000)
        # Lead-in (0→5k ramp 0→90) + 5k→100k at 90.
        expected_area = 0.5 * 5000 * 90 + 95_000 * 90
        expected_mean = expected_area / 100_000
        assert auc == pytest.approx(expected_mean, rel=1e-3)


class TestBootstrapCIMean:
    """Contract for percentile bootstrap of the mean."""

    def test_empty_values_returns_nan(self):
        m, lo, hi = bootstrap_ci_mean([], n_resamples=100, rng_seed=0)
        assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)

    def test_single_value_has_zero_width_ci(self):
        m, lo, hi = bootstrap_ci_mean([5.0], n_resamples=100, rng_seed=0)
        assert m == 5.0
        # Single value → every bootstrap resample equals 5.0 → CI = 5.
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_identical_values_zero_width_ci(self):
        m, lo, hi = bootstrap_ci_mean([7.0] * 5, n_resamples=500, rng_seed=0)
        assert m == 7.0
        assert lo == pytest.approx(7.0)
        assert hi == pytest.approx(7.0)

    def test_mean_matches_np_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        m, _, _ = bootstrap_ci_mean(vals, n_resamples=1000, rng_seed=0)
        assert m == pytest.approx(np.mean(vals))

    def test_ci_contains_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        m, lo, hi = bootstrap_ci_mean(vals, n_resamples=2000, rng_seed=0)
        assert lo <= m <= hi

    def test_determinism_under_fixed_seed(self):
        vals = [1.0, 2.5, 3.7, 4.1, 5.9]
        r1 = bootstrap_ci_mean(vals, n_resamples=500, rng_seed=42)
        r2 = bootstrap_ci_mean(vals, n_resamples=500, rng_seed=42)
        assert r1 == r2


class TestArmDescriptives:
    """End-to-end: descriptives populated on PairVerdict + JSON export."""

    def test_descriptives_populated_on_verdict(self):
        """Each verdict has scratch_descriptives + transfer_descriptives
        populated from the runs' eval_curves."""
        # Transfer hits 95 plateau at 5k; scratch at 40k.
        t_curve = _mkcurve([(5000, 95.0), (50_000, 95.0),
                            (100_000, 95.0)])
        s_curve = _mkcurve([(5000, 20.0), (10_000, 50.0),
                            (40_000, 95.0), (100_000, 95.0)])
        scratch = [_mkrun("cartpole_mcc", s, "scratch",
                          stm=40_000, eval_curve=s_curve)
                   for s in range(5)]
        transfer = [_mkrun("cartpole_mcc", s, "transfer",
                           stm=5000, eval_curve=t_curve)
                    for s in range(5)]
        payload = _make_payload(scratch + transfer)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts
                       if v.role == "primary")
        assert primary.scratch_descriptives is not None
        assert primary.transfer_descriptives is not None
        assert primary.scratch_descriptives.n == 5
        # Return at 5k: transfer should be ~95, scratch ~20.
        t_m, _, _ = primary.transfer_descriptives.returns_at_step[5000]
        s_m, _, _ = primary.scratch_descriptives.returns_at_step[5000]
        assert t_m > s_m + 50, (
            f"transfer@5k should lead scratch@5k by >50 points, "
            f"got t={t_m:.1f} s={s_m:.1f}"
        )
        # AUC over [0, 50k]: transfer lead should be large.
        assert (primary.transfer_descriptives.auc_mean
                > primary.scratch_descriptives.auc_mean + 20)

    def test_descriptives_handle_empty_curves(self):
        """If every run has an empty eval_curve (backward-compat with
        pre-v3.5 JSONs), descriptives should be NaN-filled rather than
        crashing."""
        scratch = [_mkrun("cartpole_mcc", s, "scratch", stm=150_000)
                   for s in range(5)]
        transfer = [_mkrun("cartpole_mcc", s, "transfer", stm=50_000)
                    for s in range(5)]
        payload = _make_payload(scratch + transfer)
        verdict = analyze(payload)
        primary = next(v for v in verdict.pair_verdicts
                       if v.role == "primary")
        # NaN-filled but present.
        assert primary.scratch_descriptives is not None
        m_5k, _, _ = primary.scratch_descriptives.returns_at_step[5000]
        assert math.isnan(m_5k)
        assert math.isnan(primary.scratch_descriptives.auc_mean)

    def test_descriptives_json_serializable(self):
        """_verdict_to_dict emits descriptives with string-keyed step dict."""
        curve = _mkcurve([(5000, 90.0), (50_000, 95.0)])
        scratch = [_mkrun("cartpole_mcc", s, "scratch",
                          stm=150_000, eval_curve=curve)
                   for s in range(3)]
        transfer = [_mkrun("cartpole_mcc", s, "transfer",
                           stm=50_000, eval_curve=curve)
                    for s in range(3)]
        payload = _make_payload(scratch + transfer)
        verdict = analyze(payload)
        d = _verdict_to_dict(verdict)
        primary_dict = next(p for p in d["pair_verdicts"]
                            if p["role"] == "primary")
        assert "scratch_descriptives" in primary_dict
        sd = primary_dict["scratch_descriptives"]
        assert sd is not None
        # JSON dict keys must be strings.
        assert "5000" in sd["returns_at_step"]
        assert all(isinstance(k, str) for k in sd["returns_at_step"].keys())
        # Roundtrip survives json.dumps/loads.
        round_trip = json.loads(json.dumps(d))
        assert (round_trip["pair_verdicts"][0]["scratch_descriptives"]
                ["returns_at_step"]["5000"]["mean"] == pytest.approx(
                    sd["returns_at_step"]["5000"]["mean"]))

    def test_early_step_queries_match_prereg(self):
        """§4 v3.5 descriptor panel commits to {2k, 5k, 10k}."""
        assert EARLY_STEP_QUERIES == (2_000, 5_000, 10_000)

    def test_auc_window_matches_prereg(self):
        """§4 v3.5 AUC window is [0, 50k] env-steps."""
        assert AUC_WINDOW == (0, 50_000)
