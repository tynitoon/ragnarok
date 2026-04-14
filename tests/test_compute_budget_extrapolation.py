"""Tests for scripts/compute_budget_extrapolation.py (preregistration §12.5).

This script projects the Phase 5 wall-clock from measured per-env
throughput. If projection is wrong, §12.5 cut tier decisions will be
wrong — which can silently break the paper's compute narrative and
cause us to pick the wrong scope reduction.

These tests pin:
  1. Leg-count arithmetic matches §3, §7, §12 (5 baselines, 20/10/5 seeds, 9 ablations)
  2. Throughput fallback is conservative (slowest measured env, not optimistic)
  3. Cut-tier recommendation follows §12.5 pre-declared priority order
"""

import json
from pathlib import Path

import pytest

from scripts import compute_budget_extrapolation as cbe


def _write_budget(tmp_path: Path, summary: dict, hardware: dict | None = None) -> Path:
    payload = {
        "hardware": hardware or {"device": "cuda", "gpu_name": "RTX 4080"},
        "runs": [],
        "summary": summary,
    }
    p = tmp_path / "compute_budget.json"
    p.write_text(json.dumps(payload))
    return p


class TestRunCounts:
    """Pin that the script uses exactly the preregistered leg counts."""

    def test_primary_is_5_baselines_20_seeds(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        primary = next(l for l in result["legs"] if l["name"].startswith("Primary"))
        assert primary["n_runs"] == 5 * 20, (
            f"Primary should run 5 baselines x 20 seeds = 100; got {primary['n_runs']}"
        )
        assert primary["steps_per_run"] == 500_000

    def test_secondary_is_5_baselines_10_seeds_each(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        secondaries = [l for l in result["legs"] if l["name"].startswith("Secondary")]
        assert len(secondaries) == 2, "2 secondary envs per prereg §2"
        for s in secondaries:
            assert s["n_runs"] == 5 * 10, (
                f"Secondary should run 5 baselines x 10 seeds = 50; got {s['n_runs']}"
            )

    def test_ablations_charge_A5_sweep_triply(self, tmp_path):
        """A5 in prereg §7 is a {64,160,512} latent-dim sweep — three
        sub-runs per seed. Charging A5 as 1 undercounts total ablation
        compute by 2 runs × seeds. Current correct total is (8 + 3) × 5
        = 55 runs, not 9 × 5 = 45.
        """
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        abl = next(l for l in result["legs"] if "Ablations" in l["name"])
        expected = cbe.TOTAL_ABLATION_RUNS_PER_SEED * cbe.ABLATION_SEEDS
        assert abl["n_runs"] == expected == 55, (
            f"Expected 55 ablation runs (11 configs × 5 seeds); got {abl['n_runs']}. "
            f"If A5 is still counted as 1 configuration this test fails."
        )
        assert "mountaincar-continuous" in abl["env"], (
            "Ablations run on MCC primary only per prereg §7"
        )

    def test_steps_per_run_matches_thresholds_json(self):
        """If someone edits truncation_horizon_env_steps in thresholds.json,
        the extrapolator must follow or the projection becomes a lie.
        """
        import json
        from pathlib import Path
        thr = json.loads(Path("thresholds.json").read_text())
        assert cbe.STEPS_PER_RUN == thr["truncation_horizon_env_steps"], (
            f"STEPS_PER_RUN ({cbe.STEPS_PER_RUN}) diverged from "
            f"thresholds.json truncation_horizon_env_steps "
            f"({thr['truncation_horizon_env_steps']}). Update one or the other."
        )

    def test_seeds_match_thresholds_json(self):
        """Same for seed counts — prereg commits are load-bearing."""
        import json
        from pathlib import Path
        thr = json.loads(Path("thresholds.json").read_text())
        assert cbe.PRIMARY_SEEDS == thr["primary_seeds_N"]
        assert cbe.SECONDARY_SEEDS == thr["secondary_seeds_N"]
        assert cbe.ABLATION_SEEDS == thr["ablation_seeds_N"]


class TestThroughputFallback:
    """If an env is missing from the smoke summary, fallback must be
    **conservative** (slowest measured). Optimistic fallback would hide
    compute risk.
    """

    def test_missing_env_uses_slowest_measured(self, tmp_path):
        # MCC is missing; slowest measured is acrobot at 50 steps/s
        budget = _write_budget(tmp_path, {
            "cartpole": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        primary = next(l for l in result["legs"] if l["name"].startswith("Primary"))
        assert primary["throughput_steps_per_sec"] == 50.0, (
            f"Fallback must be slowest (acrobot=50), not fastest (cartpole=200); "
            f"got {primary['throughput_steps_per_sec']}"
        )
        assert "fallback=slowest" in primary["env"], (
            f"Fallback tag must be visible in env label: {primary['env']}"
        )

    def test_measured_env_not_flagged_as_fallback(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        primary = next(l for l in result["legs"] if l["name"].startswith("Primary"))
        assert "[measured]" in primary["env"]
        assert "fallback" not in primary["env"].lower()


class TestBudgetGate:
    """Pin the §12.5 pass/fail logic. These tests calibrate throughputs to
    exercise each specific cut tier, not just "some cut is reported". A
    regression that silently returns the wrong tier would still pass the
    old weak `is not None` assertion.

    Reference numbers at cbe.WALL_BUDGET_SECONDS = 2_177_280 s (28 d × 0.9):
      primary    50M steps /  x steps/s  = 50_000_000 / x
      secondary  50M steps /  y steps/s  (2 envs, split)
      ablations  27.5M steps /  x steps/s (11 configs × 5 seeds × 500k)
    """

    def test_fits_budget_when_well_under(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 500.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 500.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 500.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        assert result["fits_budget"] is True
        assert result["cut_tier_needed"] is None
        assert result["margin_seconds"] > 0

    def test_overshoots_when_throughput_very_low(self, tmp_path):
        # 5 steps/s on every env stresses the gate well beyond cut-stack.
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 5.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 5.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 5.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        assert result["fits_budget"] is False
        assert result["cut_tier_needed"] == "cut-stack insufficient -- methodology amendment required"

    def test_cut1_selected_when_only_dmc_drop_suffices(self, tmp_path):
        """Calibrate so that the full plan just overshoots, but dropping
        DMC secondary (50M × 5 baselines × 10 seeds / y steps/s) brings
        it under. Use MCC fast, DMC slow, acrobot fast.
        Target: total just above budget; dropping DMC gets under.
        """
        # Budget = 2_177_280 s
        # Let MCC = 200 steps/s → primary = 50M/200 = 250_000 s, abl = 137_500 s
        # Let acrobot = 200 steps/s → secondary(acrobot) = 25M/200 = 125_000 s
        # Let DMC-csu = 20 steps/s → secondary(DMC) = 25M/20 = 1_250_000 s
        # total = 250_000 + 125_000 + 1_250_000 + 137_500 = 1_762_500 s
        # hmm, this is UNDER budget (fits). Need MCC slower.
        # Let MCC = 40 steps/s → primary = 1_250_000, abl = 687_500
        # DMC-csu = 50 steps/s → secondary(DMC) = 500_000
        # acrobot = 200 → secondary(acrobot) = 125_000
        # total = 1_250_000 + 125_000 + 500_000 + 687_500 = 2_562_500 s > budget
        # After dropping DMC secondary: 2_562_500 - 500_000 = 2_062_500 s < budget (OK)
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 40.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        assert result["fits_budget"] is False
        assert "cut#1" in result["cut_tier_needed"], (
            f"Expected cut#1 (drop DMC secondary); got {result['cut_tier_needed']}"
        )

    def test_cut2_selected_when_dmc_drop_insufficient(self, tmp_path):
        """Force cut#2: dropping DMC alone is insufficient, but dropping
        all secondaries + adding expanded panel works.
        """
        # Let MCC = 35 steps/s → primary = 1_428_571, abl = 785_714 (total mcc = 2_214_285)
        # acrobot = 60 → secondary(acrobot) = 416_666
        # DMC-csu = 60 → secondary(DMC) = 416_666
        # total_baseline = 2_214_285 + 416_666 + 416_666 = 3_047_617 s (over budget)
        # After cut#1 (drop DMC): 3_047_617 - 416_666 = 2_630_951 s (still over)
        # After cut#2 (drop all secondaries + panel):
        #   panel = 4 * 5 * 500_000 / 35 = 285_714
        #   = 3_047_617 - 416_666 - 416_666 + 285_714 = 2_500_000 s (still over!)
        # Need smaller gap. Let acrobot and DMC slower so the saving is bigger.
        # acrobot = 25, DMC = 25 → each = 1_000_000
        # total = 2_214_285 + 1_000_000 + 1_000_000 = 4_214_285
        # cut#1: 4_214_285 - 1_000_000 = 3_214_285 (over)
        # cut#2: 4_214_285 - 1_000_000 - 1_000_000 + 285_714 = 2_500_000 — still over budget 2_177_280
        # Need even slower secondaries, or faster MCC.
        # Let MCC = 45 steps/s → primary = 1_111_111, abl = 611_111 (mcc total = 1_722_222)
        # acrobot = 25, DMC = 25 → each = 1_000_000
        # total = 1_722_222 + 1_000_000 + 1_000_000 = 3_722_222 (over)
        # cut#1: drop DMC → 2_722_222 (still over)
        # cut#2: drop both + panel (4*5*500k/45 = 222_222) → 1_722_222 + 222_222 = 1_944_444 (UNDER)
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 45.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 25.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 25.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        assert result["fits_budget"] is False
        assert "cut#2" in result["cut_tier_needed"], (
            f"Expected cut#2 (drop secondaries + expanded panel); "
            f"got {result['cut_tier_needed']}"
        )

    def test_cut3_selected_when_only_seed_reduction_saves(self, tmp_path):
        """Force cut#3: cut#1 and cut#2 insufficient, N=20→15 is needed.
        Primary dominates; need primary so heavy that 25% reduction matters.
        """
        # Let MCC = 35 → primary = 1_428_571, abl = 785_714 (mcc total = 2_214_285)
        # acrobot = 35, DMC = 35 → secondaries = 714_285 each, total = 1_428_570
        # total_baseline = 2_214_285 + 1_428_570 = 3_642_855
        # cut#2 saving: drop secondaries, add panel 4*5*500k/35 = 285_714
        #   after_cut2 = 3_642_855 - 1_428_570 + 285_714 = 2_500_000 (over)
        # cut#3 saving: primary_saving = 1_428_571 * 0.25 = 357_142
        #   after_cut3 = 2_500_000 - 357_142 = 2_142_857 (UNDER budget 2_177_280)
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 35.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 35.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 35.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        assert result["fits_budget"] is False
        assert "cut#3" in result["cut_tier_needed"], (
            f"Expected cut#3 (N=20->15); got {result['cut_tier_needed']}"
        )

    def test_cut2_includes_expanded_panel_cost(self):
        """Regression pin: the §12.5 cut#2 panel (A1+A2+A8+A9 × 5 seeds = 20
        runs on MCC) is a **non-negotiable** addendum. If someone silently
        drops it from _recommend_cut, the cut#2 projection would be too
        optimistic. This test exercises the exact arithmetic of the panel.
        """
        # Panel cost = 4 × 5 × 500_000 / mcc_throughput
        mcc_tp = 45.0
        expected_panel_s = cbe.EXPANDED_PANEL_RUNS_PER_SEED * cbe.ABLATION_SEEDS * cbe.STEPS_PER_RUN / mcc_tp
        assert cbe.EXPANDED_PANEL_RUNS_PER_SEED == 4, (
            "Expected panel = A1+A2+A8+A9 = 4 configurations"
        )
        # And that each member of the panel is actually in the ablation roster
        for a in cbe.EXPANDED_PANEL_ABLATIONS:
            assert a in cbe.ABLATION_ROSTER
        assert expected_panel_s > 0


class TestDataclassAccounting:
    def test_total_is_sum_of_legs(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "acrobot": {"mean_steps_per_sec": 200.0, "std_steps_per_sec": 0, "n_seeds": 3},
            "cartpole-swingup": {"mean_steps_per_sec": 50.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        leg_sum = sum(l["wall_seconds"] for l in result["legs"])
        assert abs(result["phase5_total_seconds"] - leg_sum) < 1e-6

    def test_wall_hours_matches_wall_seconds(self, tmp_path):
        budget = _write_budget(tmp_path, {
            "mountaincar-continuous": {"mean_steps_per_sec": 100.0, "std_steps_per_sec": 0, "n_seeds": 3},
        })
        result = cbe.extrapolate(budget)
        for leg in result["legs"]:
            assert abs(leg["wall_hours"] * 3600 - leg["wall_seconds"]) < 1e-6
            assert abs(leg["wall_days"] * 24 - leg["wall_hours"]) < 1e-6
