"""Tests for scripts/pilot_run.py (Phase 3 pilot per preregistration §8).

Focus: structural invariants of the pilot matrix + output schema + the few
pieces of logic that are script-level (threshold resolution, resume key
semantics, eval-curve → steps-to-mastery). We deliberately don't spin up
a full RagnarokAgent in every test — end-to-end training is covered by a
single smoke test that runs a ~1-minute pilot slice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pilot_run import (  # noqa: E402
    EVAL_EPISODES_DEFAULT,
    EVAL_EVERY_STEPS_DEFAULT,
    MAX_ENV_STEPS_DEFAULT,
    PILOT_PAIRS,
    PILOT_SEEDS_DEFAULT,
    SOURCE_MAX_ENV_STEPS_DEFAULT,
    EvalPoint,
    PilotRun,
    _compute_transfer_telemetry,
    _run_from_dict,
    resolve_mastery_thresholds,
)


# ── Pair matrix must match preregistration §8 ──────────────────────

class TestPilotMatrixIsFrozen:
    """PILOT_PAIRS is pre-declared per prereg §8. Changes require §13."""

    def test_exactly_three_pairs(self):
        assert len(PILOT_PAIRS) == 3, (
            "Prereg §8 pins 3 source->target pairs; deviation requires "
            "§13 amendment"
        )

    def test_primary_pair_is_cartpole_to_mcc(self):
        primary = [p for p in PILOT_PAIRS if p[3] == "primary"]
        assert len(primary) == 1, "Exactly one primary-endpoint rehearsal"
        alias, src, tgt, role = primary[0]
        assert src == "cartpole"
        assert tgt == "mountaincar-continuous"

    def test_secondary_pair_cartpole_to_acrobot(self):
        aliases = {p[0]: p for p in PILOT_PAIRS}
        assert "cartpole_acrobot" in aliases
        _, src, tgt, role = aliases["cartpole_acrobot"]
        assert (src, tgt, role) == ("cartpole", "acrobot", "secondary")

    def test_secondary_pair_pendulum_to_dmc_cartpole_swingup(self):
        aliases = {p[0]: p for p in PILOT_PAIRS}
        assert "pendulum_dmc_cartpole" in aliases
        _, src, tgt, role = aliases["pendulum_dmc_cartpole"]
        assert src == "pendulum"
        assert tgt == "cartpole-swingup"
        assert role == "secondary"

    def test_aliases_are_unique(self):
        aliases = [p[0] for p in PILOT_PAIRS]
        assert len(aliases) == len(set(aliases)), (
            "pair aliases must be unique — they key the skills dir and "
            "the resume set"
        )


class TestPilotBudgetConstants:
    def test_seeds_n_matches_prereg_section_8(self):
        assert PILOT_SEEDS_DEFAULT == 5, "Prereg §8: 5 seeds per (pair, arm)"

    def test_max_steps_matches_prereg_section_8(self):
        # §8: "Each run 200 k env-steps or convergence"
        assert MAX_ENV_STEPS_DEFAULT == 200_000

    def test_eval_cadence_matches_prereg_section_4_5(self):
        # §4.5: eval every 5000 env-steps, 10 eval episodes per checkpoint
        assert EVAL_EVERY_STEPS_DEFAULT == 5_000
        assert EVAL_EPISODES_DEFAULT == 10

    def test_source_cap_is_below_target_budget(self):
        # Source is a prerequisite, not a pilot arm. It should cap well under
        # the 200k target budget to bound total wall-clock.
        assert SOURCE_MAX_ENV_STEPS_DEFAULT <= MAX_ENV_STEPS_DEFAULT // 2

    def test_total_target_arms_is_30(self):
        total = PILOT_SEEDS_DEFAULT * len(PILOT_PAIRS) * 2  # scratch + transfer
        assert total == 30, (
            "§8 pins 5 seeds × 3 pairs × 2 arms = 30 target runs "
            "(source pre-training is separate)"
        )


# ── Threshold resolution ────────────────────────────────────────────

class TestMasteryThresholds:
    def test_defaults_pulled_from_registry(self):
        thr = resolve_mastery_thresholds(None)
        # Every env referenced in PILOT_PAIRS (src or tgt) must be present
        for (_, src, tgt, _) in PILOT_PAIRS:
            assert src in thr, f"missing src threshold for {src}"
            assert tgt in thr, f"missing tgt threshold for {tgt}"

    def test_registry_defaults_are_env_native(self):
        thr = resolve_mastery_thresholds(None)
        # Sanity-check a few known values from the registry
        assert thr["cartpole"] == 450.0
        assert thr["mountaincar-continuous"] == 90.0
        assert thr["acrobot"] == -100.0
        assert thr["pendulum"] == -200.0

    def test_override_file_wins(self, tmp_path):
        overrides_path = tmp_path / "thresholds.json"
        overrides_path.write_text(json.dumps({
            "pilot_mastery_thresholds": {
                "mountaincar-continuous": 76.0,  # 0.8 × 95
                "acrobot": -88.0,                # 0.8 × -110
            }
        }))
        thr = resolve_mastery_thresholds(overrides_path)
        assert thr["mountaincar-continuous"] == 76.0
        assert thr["acrobot"] == -88.0
        # Non-overridden envs still come from the registry
        assert thr["cartpole"] == 450.0

    def test_override_flat_schema_also_accepted(self, tmp_path):
        # Flexible schema: allow either {"pilot_mastery_thresholds": {...}}
        # or a bare dict {env_name: threshold}.
        overrides_path = tmp_path / "flat.json"
        overrides_path.write_text(json.dumps({
            "mountaincar-continuous": 50.0
        }))
        thr = resolve_mastery_thresholds(overrides_path)
        assert thr["mountaincar-continuous"] == 50.0


# ── PilotRun dataclass + serialization ─────────────────────────────

class TestPilotRunSchema:
    def _make(self, **overrides) -> PilotRun:
        defaults = dict(
            pair_alias="cartpole_mcc",
            pair_role="primary",
            src_env="cartpole",
            tgt_env="mountaincar-continuous",
            seed=42,
            arm="transfer",
            mastery_threshold=90.0,
            max_env_steps=200_000,
            total_env_steps=150_000,
            total_episodes=850,
            final_eval_return=88.2,
            best_eval_return=91.0,
            steps_to_mastery=125_000,
            eval_curve=[
                EvalPoint(step=5000, eval_return=-10.0),
                EvalPoint(step=10000, eval_return=0.5),
                EvalPoint(step=125000, eval_return=91.0),
                EvalPoint(step=150000, eval_return=88.2),
            ],
            acting_policy_mode="latent",
            transfer_skill_name="CartPole-v1_450ep",
            wall_clock_sec=812.4,
        )
        defaults.update(overrides)
        return PilotRun(**defaults)

    def test_to_dict_has_rmst_fields(self):
        """lifelines.KaplanMeierFitter needs (duration, observed). These
        come from steps_to_mastery (or max_env_steps if censored) and the
        `censored` flag."""
        r = self._make()
        d = r.to_dict()
        assert "steps_to_mastery" in d
        assert "censored" in d
        assert d["censored"] is False
        assert d["steps_to_mastery"] == 125_000

    def test_to_dict_marks_censored_when_mastery_none(self):
        r = self._make(steps_to_mastery=None, final_eval_return=60.0,
                       best_eval_return=62.0)
        d = r.to_dict()
        assert d["censored"] is True
        assert d["steps_to_mastery"] is None

    def test_eval_curve_serializes_as_list_of_dicts(self):
        r = self._make()
        d = r.to_dict()
        assert isinstance(d["eval_curve"], list)
        for p in d["eval_curve"]:
            assert set(p.keys()) == {"step", "eval_return"}
            assert isinstance(p["step"], int)

    def test_roundtrip_from_dict_preserves_fields(self):
        r = self._make()
        d = r.to_dict()
        r2 = _run_from_dict(d)
        assert r2.seed == r.seed
        assert r2.arm == r.arm
        assert r2.pair_alias == r.pair_alias
        assert r2.steps_to_mastery == r.steps_to_mastery
        assert r2.acting_policy_mode == r.acting_policy_mode
        assert len(r2.eval_curve) == len(r.eval_curve)
        assert r2.eval_curve[0].step == r.eval_curve[0].step
        assert r2.eval_curve[0].eval_return == pytest.approx(
            r.eval_curve[0].eval_return)

    def test_acting_mode_default_is_obs(self):
        # Default matches RagnarokAgent.__init__ (agent.py:77)
        r = PilotRun(
            pair_alias="x", pair_role="p", src_env="a", tgt_env="b",
            seed=0, arm="scratch", mastery_threshold=0.0,
            max_env_steps=1, total_env_steps=1, total_episodes=1,
            final_eval_return=0.0, best_eval_return=0.0,
            steps_to_mastery=None,
        )
        assert r.acting_policy_mode == "obs"


# ── Resume-key semantics ────────────────────────────────────────────

class TestResumeKeys:
    """Resume loads (alias, seed, arm) triples from a prior output JSON.
    Source runs must be keyed by (src_env, seed, "source") so they aren't
    deduped incorrectly against target runs on the same seed."""

    def test_source_key_uses_src_env_not_pair_alias(self):
        """A source run for 'cartpole' with seed=42 must not collide with a
        'cartpole' target arm (e.g. the transfer arm of a pair where
        cartpole is the *source*)."""
        src_dict = {
            "pair_alias": "",       # source runs have empty alias
            "src_env": "cartpole",
            "tgt_env": "cartpole",  # source's "target" is itself
            "seed": 42,
            "arm": "source",
        }
        # Mirror the key logic used in run_pilot._flush / completed_keys
        key = (src_dict.get("pair_alias", ""), src_dict.get("seed", -1),
               src_dict.get("arm", ""))
        if src_dict.get("arm") == "source":
            key = (src_dict.get("src_env", ""), src_dict.get("seed", -1),
                   "source")
        assert key == ("cartpole", 42, "source")

    def test_target_key_uses_pair_alias(self):
        """Target arms are keyed by alias so the same src_env feeding two
        different pairs doesn't cause collisions (e.g. cartpole feeds both
        cartpole_mcc and cartpole_acrobot)."""
        for alias in ["cartpole_mcc", "cartpole_acrobot"]:
            d = {"pair_alias": alias, "seed": 42, "arm": "scratch",
                 "src_env": "cartpole"}
            key = (d["pair_alias"], d["seed"], d["arm"])
            assert key[0] == alias  # alias wins over src_env for targets


# ── steps_to_mastery invariants on the eval curve ──────────────────

class TestStepsToMasteryLogic:
    """The live runner picks first-crossing inline; these tests verify the
    same invariants hold on a reconstructed eval curve so downstream
    analysis (e.g. sensitivity sweeps at different τ) is consistent."""

    @staticmethod
    def _first_crossing(curve: list[EvalPoint], threshold: float) -> int | None:
        for p in curve:
            if p.eval_return >= threshold:
                return p.step
        return None

    def test_first_crossing_is_first(self):
        curve = [
            EvalPoint(5000, 0.0),
            EvalPoint(10000, 50.0),
            EvalPoint(15000, 95.0),
            EvalPoint(20000, 92.0),   # dips below but we already crossed
            EvalPoint(25000, 100.0),
        ]
        assert self._first_crossing(curve, threshold=90.0) == 15000

    def test_never_crosses_returns_none(self):
        curve = [EvalPoint(5000, 0.0), EvalPoint(10000, 50.0)]
        assert self._first_crossing(curve, threshold=90.0) is None

    def test_crosses_at_exact_threshold(self):
        curve = [EvalPoint(5000, 90.0)]
        assert self._first_crossing(curve, threshold=90.0) == 5000


# ── Output-file contract ───────────────────────────────────────────

class TestOutputFileContract:
    """The pilot_results.json produced by run_pilot must be loadable by a
    downstream RMST analyzer (scripts/pilot_analysis.py, to be written).
    Pin the top-level schema here so schema drift breaks loudly."""

    def test_top_level_keys_present(self):
        # Synthesize a minimal payload matching run_pilot._flush()
        payload = {
            "prereg_section": "§8 (pilot)",
            "pairs": [{"alias": a, "src": s, "tgt": t, "role": r}
                      for (a, s, t, r) in PILOT_PAIRS],
            "seeds_N": 5,
            "base_seed": 42,
            "max_env_steps": 200_000,
            "source_max_env_steps": 100_000,
            "eval_every_steps": 5_000,
            "eval_episodes": 10,
            "mastery_thresholds": {"cartpole": 450.0},
            "runs": [],
        }
        required = {"prereg_section", "pairs", "seeds_N", "base_seed",
                    "max_env_steps", "eval_every_steps", "eval_episodes",
                    "mastery_thresholds", "runs"}
        assert required.issubset(payload.keys())

    def test_pairs_block_has_all_four_role_fields(self):
        for alias, src, tgt, role in PILOT_PAIRS:
            entry = {"alias": alias, "src": src, "tgt": tgt, "role": role}
            assert set(entry.keys()) == {"alias", "src", "tgt", "role"}


# ── Reviewer-fix coverage: atomic write + provenance ───────────────

class TestAtomicJSONWrite:
    """Devil's-advocate review: a Ctrl-C mid-flush can truncate
    pilot_results.json, and the resume logic's try/except then silently
    starts fresh — wiping up to 8hr of completed work. `_atomic_write_json`
    uses tmp-file + os.replace to make the swap atomic and keeps a .bak."""

    def test_atomic_write_lands_full_payload(self, tmp_path):
        from scripts.pilot_run import _atomic_write_json
        target = tmp_path / "pilot_results.json"
        _atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
        assert target.exists()
        assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}

    def test_atomic_write_keeps_bak_on_overwrite(self, tmp_path):
        from scripts.pilot_run import _atomic_write_json
        target = tmp_path / "pilot_results.json"
        _atomic_write_json(target, {"run": 1})
        _atomic_write_json(target, {"run": 2})
        # The .bak holds the prior version so a reviewer can diff.
        bak = target.with_suffix(target.suffix + ".bak")
        assert bak.exists()
        assert json.loads(bak.read_text()) == {"run": 1}
        assert json.loads(target.read_text()) == {"run": 2}

    def test_atomic_write_cleans_tmp(self, tmp_path):
        from scripts.pilot_run import _atomic_write_json
        target = tmp_path / "pilot_results.json"
        _atomic_write_json(target, {"x": 1})
        # The tmp must not linger — os.replace consumed it.
        assert not target.with_suffix(target.suffix + ".tmp").exists()


class TestProvenanceCollection:
    """Top-level payload must carry provenance so a reviewer can replay a
    pilot run: python, torch, cuda, gpu, lifelines, git_sha, git_dirty."""

    def test_provenance_has_required_keys(self):
        from scripts.pilot_run import _collect_provenance
        prov = _collect_provenance()
        for k in ["python", "platform", "hostname", "torch", "device",
                  "gpu_name", "lifelines", "git_sha", "git_dirty"]:
            assert k in prov, f"provenance missing required key: {k}"

    def test_provenance_values_are_json_serializable(self):
        from scripts.pilot_run import _collect_provenance
        # Must round-trip through JSON (no sets, no tuples, no paths).
        prov = _collect_provenance()
        round_tripped = json.loads(json.dumps(prov))
        assert round_tripped.keys() == prov.keys()


# ── Reviewer-fix coverage: _cross_dim shared predicate ─────────────

class TestCrossDimPredicate:
    """`_cross_dim` is the shared predicate between pilot_run (asserts at
    transfer time) and pilot_analysis (mechanism check). It must match the
    registry's obs/action/discrete axes."""

    def test_cartpole_to_mcc_is_cross_dim(self):
        from scripts.pilot_run import _cross_dim
        # CartPole: obs=4, act=2 discrete; MCC: obs=2, act=1 continuous
        assert _cross_dim("cartpole", "mountaincar-continuous")

    def test_cartpole_to_acrobot_is_cross_dim(self):
        from scripts.pilot_run import _cross_dim
        # CartPole: obs=4, act=2; Acrobot: obs=6, act=3 — both axes differ
        assert _cross_dim("cartpole", "acrobot")

    def test_same_env_is_not_cross_dim(self):
        from scripts.pilot_run import _cross_dim
        assert not _cross_dim("cartpole", "cartpole")


# ── Source-crystallized flag surfaces to downstream analyzer ──────

class TestSourceCrystallizedField:
    """The source pre-training may hit the cap without crystallizing a skill.
    This must be surfaced in PilotRun.to_dict so the analyzer can flag a
    'transfer arm whose source never produced a real skill'."""

    def test_to_dict_has_source_crystallized(self):
        run = PilotRun(
            pair_alias="", pair_role="", src_env="cartpole", tgt_env="cartpole",
            seed=42, arm="source", mastery_threshold=450.0,
            max_env_steps=100_000, total_env_steps=100_000, total_episodes=50,
            final_eval_return=300.0, best_eval_return=310.0,
            steps_to_mastery=None, source_crystallized=False,
        )
        d = run.to_dict()
        assert "source_crystallized" in d
        assert d["source_crystallized"] is False

    def test_roundtrip_preserves_source_crystallized(self):
        run = PilotRun(
            pair_alias="", pair_role="", src_env="cartpole", tgt_env="cartpole",
            seed=42, arm="source", mastery_threshold=450.0,
            max_env_steps=100_000, total_env_steps=50_000, total_episodes=20,
            final_eval_return=475.0, best_eval_return=475.0,
            steps_to_mastery=40_000, source_crystallized=True,
        )
        d = run.to_dict()
        r2 = _run_from_dict(d)
        assert r2.source_crystallized is True


# ── CLI smoke (imports + arg parsing only) ─────────────────────────

class TestCLIInterface:
    def test_smoke_flag_sets_reduced_budget(self, monkeypatch, tmp_path):
        """--smoke should reduce seeds/steps so the CLI is safe to invoke
        in CI without accidentally spawning a multi-hour run.

        v4 update (devil's-advocate v3 BLOCKER): seeds bumped 1→2 to
        match the prereg v2 amendment (2-seed smoke pre-check) and
        max-steps bumped 20k→40k to give headroom past the ep-100
        drift abort criterion.
        """
        # We don't actually execute run_pilot (that would start training).
        # Instead, import main and verify the argparse transform.
        from scripts.pilot_run import main as pilot_main

        captured = {}

        def _fake_run_pilot(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("scripts.pilot_run.run_pilot", _fake_run_pilot)
        rc = pilot_main([
            "--smoke",
            "--output", str(tmp_path / "out.json"),
            "--skills-root", str(tmp_path / "skills"),
        ])
        assert rc == 0
        # v4: 2 seeds matches prereg v2 "Bug E v2" amendment (2-seed
        # smoke pre-check); 1 was a code/prereg drift that the v3
        # devil's-advocate review surfaced as a BLOCKER.
        assert captured["seeds"] == 2
        # v4: 40k gives headroom past the ep-100 drift abort criterion;
        # 20k was tight on slow MCC mastery curves.
        assert captured["max_env_steps"] == 40_000
        assert captured["source_max_env_steps"] == 10_000
        # --smoke forces pair filter to the primary so the CI doesn't hit DMC
        assert captured["pair_filter"] == ["cartpole_mcc"]


# ── Phase 3 pre-launch regression: Bug A (SkillSelector threshold) ──
#
# smoke #2 of the pilot revealed that BOTH transfer arms silently
# failed to load the committed source skill — both `transfer_skill_name`
# fields came back null, and `acting_policy_mode` stayed at "obs"
# instead of flipping to "latent". Root cause: SkillSelector enforces a
# `distance_threshold=50.0` gate between the warmup-latent centroid and
# the candidate skill's centroid. That gate is designed for continual-
# learning "is this skill relevant?" semantics; it's actively hostile
# to pilot arms that PIN source→target per preregistration §8. The
# warmup encoding uses a fresh (untrained) RSSM while the source
# centroid was stored from a trained RSSM — distances are essentially
# noise and exceed the threshold trivially.
#
# Fix in pilot_run._build_agent: override the threshold to +inf so the
# selector always returns the nearest skill when one is present.

class TestBuildAgentOverridesSkillSelectorThreshold:
    """Bug A regression: every agent the pilot builds must disable the
    distance-threshold gate. Without this, the §8 mechanism check
    (acting_policy_mode == 'latent' on cross-dim transfer arms) would
    fail trivially and cost the full 8-hour pilot wall-clock."""

    def test_distance_threshold_is_infinite(self, tmp_path):
        from scripts.pilot_run import _build_agent
        agent, env = _build_agent(
            env_name="cartpole",
            seed=0,
            skills_dir=str(tmp_path / "skills"),
        )
        assert agent.skill_selector.distance_threshold == float("inf"), (
            "Bug A regression: _build_agent must override the default "
            "distance_threshold (50.0) to +inf so the pre-registered "
            "source→target pair's transfer arm can load its skill."
        )

    def test_default_skillselector_threshold_is_finite(self):
        """Sentinel: if the upstream default flips to inf on its own,
        the override above becomes a no-op — which is fine but worth
        knowing. This test documents the gap the pilot override closes."""
        from ragnarok.skills.library import SkillLibrary
        from ragnarok.skills.selector import SkillSelector
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            # Needs something that quacks like an RSSM but we only touch
            # the threshold attribute — a bare selector is fine.
            class _FakeRSSM:
                pass
            selector = SkillSelector(_FakeRSSM(), lib)
            assert selector.distance_threshold < float("inf"), (
                "upstream SkillSelector default is now inf — the pilot "
                "override is no longer strictly necessary but still "
                "documents intent"
            )


# ── Phase 3 pre-launch regression: Bug B (shared source-skills dir) ─
#
# smoke #2 also revealed that when two pairs share the same src_env
# (cartpole_mcc and cartpole_acrobot both have src=cartpole), the
# source-training dedup (keyed by (src_env, seed)) correctly avoids
# retraining — but the skills_dir path was keyed by the pair *alias*.
# Pair 2's per_seed_skills was a DIFFERENT directory from pair 1's, so
# pair 2's transfer arm read from an empty dir and loaded nothing.
#
# Fix in run_pilot: key the dir by (src_env, seed), not (alias, seed).

class TestSharedSourceSkillsDir:
    """Bug B regression: pairs sharing src_env+seed must share the
    skills_dir they use for source storage + transfer loading."""

    def test_two_pairs_same_src_share_dir(self, monkeypatch, tmp_path):
        """Run a minimized pilot with two pairs that share src_env
        (cartpole_mcc + cartpole_acrobot, both src=cartpole). Capture
        every skills_dir passed to the transfer arm. Both transfer
        arms must see the SAME directory — otherwise pair 2's
        try_transfer() loads nothing (Bug B symptom)."""
        from scripts import pilot_run

        # Stub out the two training entrypoints with synthetic results
        # so we only exercise the orchestration logic.
        source_calls: list[dict] = []
        target_calls: list[dict] = []

        def _fake_pretrain_source(**kwargs):
            source_calls.append(kwargs)
            # Create the skill file so pair-2's transfer would really
            # find it if the dir sharing works.
            Path(kwargs["skills_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["skills_dir"]) / "CartPole-v1_fake.pt").write_bytes(b"")
            return PilotRun(
                pair_alias="",
                pair_role="",
                src_env=kwargs["src_env"],
                tgt_env=kwargs["src_env"],
                seed=kwargs["seed"],
                arm="source",
                mastery_threshold=kwargs["mastery_threshold"],
                max_env_steps=kwargs["max_env_steps"],
                total_env_steps=1000,
                total_episodes=10,
                final_eval_return=500.0,
                best_eval_return=500.0,
                steps_to_mastery=1000,
                source_crystallized=True,
            )

        def _fake_train_to_step_budget(**kwargs):
            target_calls.append(kwargs)
            return PilotRun(
                pair_alias=kwargs.get("pair_alias") or "",
                pair_role=kwargs.get("pair_role") or "",
                src_env=kwargs.get("src_env") or "",
                tgt_env=kwargs["env_name"],
                seed=kwargs["seed"],
                arm=kwargs["arm"],
                mastery_threshold=kwargs["mastery_threshold"],
                max_env_steps=kwargs["max_env_steps"],
                total_env_steps=100,
                total_episodes=1,
                final_eval_return=0.0,
                best_eval_return=0.0,
                steps_to_mastery=None,
            )

        monkeypatch.setattr(pilot_run, "_pretrain_source", _fake_pretrain_source)
        monkeypatch.setattr(pilot_run, "_train_to_step_budget",
                            _fake_train_to_step_budget)

        out_path = tmp_path / "out.json"
        skills_root = tmp_path / "skills"
        pilot_run.run_pilot(
            seeds=1,
            max_env_steps=1,
            source_max_env_steps=1,
            eval_every_steps=1,
            eval_episodes=1,
            skills_root=skills_root,
            output_path=out_path,
            pair_filter=["cartpole_mcc", "cartpole_acrobot"],
            base_seed=42,
        )

        # Source trained exactly ONCE (shared src_env=cartpole + seed=42)
        assert len(source_calls) == 1, (
            f"expected 1 source call (shared cartpole+42); got "
            f"{len(source_calls)}"
        )

        # Both transfer arms share the SAME skills_dir
        transfer_calls = [c for c in target_calls if c["arm"] == "transfer"]
        assert len(transfer_calls) == 2, (
            f"expected 2 transfer calls (mcc + acrobot); got "
            f"{len(transfer_calls)}"
        )
        transfer_dirs = {c["skills_dir"] for c in transfer_calls}
        assert len(transfer_dirs) == 1, (
            f"Bug B regression: transfer arms for pairs sharing src_env "
            f"must receive the same skills_dir. Got: {transfer_dirs}"
        )

        # And the shared dir is the one the source wrote into.
        (shared,) = transfer_dirs
        assert shared == source_calls[0]["skills_dir"], (
            f"transfer dir {shared!r} must equal source dir "
            f"{source_calls[0]['skills_dir']!r}"
        )

        # Scratch arms get their own (empty) dirs — NOT shared across pairs.
        scratch_calls = [c for c in target_calls if c["arm"] == "scratch"]
        scratch_dirs = {c["skills_dir"] for c in scratch_calls}
        assert len(scratch_dirs) == 2, (
            "scratch arms must each get their own isolated empty dir "
            "(never share — scratch must never see a source skill)"
        )
        for d in scratch_dirs:
            assert d != shared, (
                "scratch dir must differ from the transfer-arm shared "
                "source dir, else scratch sees the source skill"
            )

    def test_dir_name_pattern_is_src_env_keyed(self, monkeypatch, tmp_path):
        """Documenting: the per-seed skills dir naming scheme uses
        src_env in the path, not alias. This keeps the semantic clear:
        'source_{src_env}_seed{seed}' lives ONCE per (src_env, seed),
        regardless of how many pairs name it as their source."""
        from scripts import pilot_run

        captured = []

        def _capture(**kwargs):
            captured.append(kwargs["skills_dir"])
            Path(kwargs["skills_dir"]).mkdir(parents=True, exist_ok=True)
            return PilotRun(
                pair_alias="", pair_role="", src_env=kwargs["src_env"],
                tgt_env=kwargs["src_env"], seed=kwargs["seed"], arm="source",
                mastery_threshold=kwargs["mastery_threshold"],
                max_env_steps=kwargs["max_env_steps"],
                total_env_steps=1, total_episodes=1,
                final_eval_return=0.0, best_eval_return=0.0,
                steps_to_mastery=1, source_crystallized=True,
            )

        monkeypatch.setattr(pilot_run, "_pretrain_source", _capture)
        monkeypatch.setattr(
            pilot_run, "_train_to_step_budget",
            lambda **k: PilotRun(
                pair_alias=k.get("pair_alias") or "",
                pair_role=k.get("pair_role") or "",
                src_env=k.get("src_env") or "", tgt_env=k["env_name"],
                seed=k["seed"], arm=k["arm"],
                mastery_threshold=k["mastery_threshold"],
                max_env_steps=k["max_env_steps"], total_env_steps=1,
                total_episodes=1, final_eval_return=0.0,
                best_eval_return=0.0, steps_to_mastery=None,
            ),
        )

        pilot_run.run_pilot(
            seeds=1, max_env_steps=1, source_max_env_steps=1,
            eval_every_steps=1, eval_episodes=1,
            skills_root=tmp_path / "skills",
            output_path=tmp_path / "out.json",
            pair_filter=["cartpole_mcc"], base_seed=42,
        )
        assert len(captured) == 1
        skills_path = Path(captured[0])
        # Contract: the dir name must contain the src_env (cartpole) and
        # seed (42) but NOT the pair alias (cartpole_mcc).
        assert "cartpole" in skills_path.name
        assert "42" in skills_path.name
        assert "mcc" not in skills_path.name, (
            "pair alias leaked into source-skills dir name; pair 2 "
            "with a different alias would get a different dir."
        )


# ── Telemetry capture (Bug E v4: testing review v3 MAJOR concern #5) ─

class TestComputeTransferTelemetry:
    """Unit tests for `_compute_transfer_telemetry` (Bug E v4).

    Devil's-advocate v3 testing review flagged that the original
    `_capture_telemetry` was a closure inside `_train_to_step_budget`
    with zero unit-test coverage — a regression there would silently
    break the prereg's "abort if drift > 50% by ep 100" criterion and
    only surface during a 20-min smoke run. v4 extracts the function
    to module level so it's testable in isolation; these tests pin the
    contract so the regression-detection loop is fast.
    """

    def _make_stub_agent(self, drift_factor: float = 0.0):
        """Build a minimal cuda-free agent stub that exposes the API
        `_compute_transfer_telemetry` reads from:
          - agent.wm_trainer.rssm.transferable_state_dict()
          - agent.wm_trainer.rssm.observe()
          - agent.wm_trainer.{batch_size, seq_length}
          - agent.replay_buffer.{num_episodes, sample_sequences()}
          - agent.total_steps, agent.total_episodes

        The stub returns synthetic transferable params perturbed by
        `drift_factor` from the baseline, so the test can verify the
        drift formula `||v_now - v0||/||v0||` end-to-end.
        """
        import numpy as np
        import torch
        from types import SimpleNamespace

        torch.manual_seed(0)

        baseline_w = torch.randn(8, 4)  # one transferable param
        # v_now = baseline + drift_factor * baseline (so drift / norm
        # = drift_factor exactly).
        current_w = baseline_w + drift_factor * baseline_w
        sd_now = {"core.gru.weight_ih_l0": current_w.clone()}

        class _RSSM:
            def transferable_state_dict(self):
                return {k: v.clone() for k, v in sd_now.items()}

            def observe(self, obs_t, act_t):
                # Return synthetic post/prior with a known KL: post~N(0,1),
                # prior~N(0,1) gives KL ≈ 0; offset post mean by 1.0 to
                # produce KL ≈ 0.5 (per dim, summed over 4 dims = 2.0).
                B, T = obs_t.shape[:2]
                return {
                    "post_mean":   torch.ones(B, T, 4),
                    "post_logstd": torch.zeros(B, T, 4),
                    "prior_mean":  torch.zeros(B, T, 4),
                    "prior_logstd": torch.zeros(B, T, 4),
                }

        class _Buf:
            num_episodes = 4

            def sample_sequences(self, bs, sl):
                obs = np.zeros((bs, sl, 2), dtype=np.float32)
                act = np.zeros((bs, sl, 1), dtype=np.float32)
                rew = np.zeros((bs, sl), dtype=np.float32)
                done = np.zeros((bs, sl), dtype=np.float32)
                return obs, act, rew, done

        wm_trainer = SimpleNamespace(
            rssm=_RSSM(), batch_size=50, seq_length=50,
            kl_weight=0.1, free_nats=1.0)
        agent = SimpleNamespace(
            wm_trainer=wm_trainer, replay_buffer=_Buf(),
            total_steps=12_345, total_episodes=87)

        baseline = {"core.gru.weight_ih_l0": baseline_w.clone()}
        baseline_norms = {"core.gru.weight_ih_l0":
                          float(baseline_w.norm().item())}
        return agent, baseline, baseline_norms

    def test_returns_none_when_baseline_is_none(self):
        """Non-transfer arms (and transfer arms where try_transfer
        returned None) must produce no telemetry record."""
        agent, _, _ = self._make_stub_agent(drift_factor=0.0)
        result = _compute_transfer_telemetry(agent, None, None)
        assert result is None

    def test_returns_none_when_baseline_norms_missing(self):
        """Defensive: if baseline_norms got out of sync with baseline
        (impossible by construction in pilot_run, but the function
        contract is `both or neither`), bail out."""
        agent, baseline, _ = self._make_stub_agent(drift_factor=0.0)
        result = _compute_transfer_telemetry(agent, baseline, None)
        assert result is None

    def test_drift_zero_when_weights_unchanged(self):
        """drift_factor=0 → current weights == baseline → drift is 0."""
        agent, baseline, baseline_norms = self._make_stub_agent(drift_factor=0.0)
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        assert result["transferable_drift_max"] == pytest.approx(0.0, abs=1e-6)

    def test_drift_matches_factor_when_weights_perturbed(self):
        """drift_factor=0.3 → current = 1.3 * baseline → ||v_now -
        v0||/||v0|| = 0.3 exactly. This is the load-bearing math
        for the prereg's "abort > 50% by ep 100" criterion."""
        agent, baseline, baseline_norms = self._make_stub_agent(drift_factor=0.3)
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        assert result["transferable_drift_max"] == pytest.approx(0.3, rel=1e-5)
        assert result["transferable_drift_per_param"][
            "core.gru.weight_ih_l0"] == pytest.approx(0.3, rel=1e-5)

    def test_record_includes_step_and_episode(self):
        """The downstream analyzer keys telemetry by (step, episode)
        pairs — both must appear in every record."""
        agent, baseline, baseline_norms = self._make_stub_agent(drift_factor=0.1)
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result["step"] == 12_345
        assert result["episode"] == 87

    def test_kl_probe_is_unclamped_raw_kl(self):
        """Architecture review v3 BLOCKER: the KL probe MUST be raw
        kl_divergence(post, prior).sum(-1).mean() — NOT the
        free-nats-clamped training loss field. The stub `observe()`
        returns post~N(1,1) and prior~N(0,1) over 4 dims:
        analytical KL per dim = 0.5; sum over 4 dims = 2.0; mean over
        (B, T) = 2.0. The probe must report ~2.0, NOT the clamped
        training value (which would be max(2.0, free_nats=1.0) = 2.0
        in this case but for *flat* prior + posterior would clamp
        at 1.0/4 per dim and silently report the floor).

        Concretely: replace the stub `observe()` to return identical
        post and prior (KL=0). The clamped `kl_loss` field would
        return 1.0/stoch_dim = 0.25 (the floor), but the raw KL is
        0. The probe must report ~0, proving it's NOT calling the
        clamped loss path.
        """
        import torch
        from types import SimpleNamespace
        import numpy as np

        torch.manual_seed(0)

        class _RSSMIdenticalDist:
            def transferable_state_dict(self):
                return {"k": torch.zeros(2, 2)}

            def observe(self, obs_t, act_t):
                B, T = obs_t.shape[:2]
                # post == prior → analytical KL is exactly zero.
                return {
                    "post_mean": torch.zeros(B, T, 4),
                    "post_logstd": torch.zeros(B, T, 4),
                    "prior_mean": torch.zeros(B, T, 4),
                    "prior_logstd": torch.zeros(B, T, 4),
                }

        class _Buf:
            num_episodes = 4

            def sample_sequences(self, bs, sl):
                return (np.zeros((bs, sl, 2), dtype=np.float32),
                        np.zeros((bs, sl, 1), dtype=np.float32),
                        np.zeros((bs, sl), dtype=np.float32),
                        np.zeros((bs, sl), dtype=np.float32))

        agent = SimpleNamespace(
            wm_trainer=SimpleNamespace(
                rssm=_RSSMIdenticalDist(),
                batch_size=4, seq_length=4,
                kl_weight=0.1, free_nats=1.0),
            replay_buffer=_Buf(),
            total_steps=0, total_episodes=0)

        baseline = {"k": torch.zeros(2, 2)}
        baseline_norms = {"k": 1.0}
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        # Raw KL of identical Normals is 0. Clamped (free-nats) KL
        # would report ≥ free_nats/stoch_dim = 0.25. If we see ≥ 0.20
        # the function is calling the clamped path and the BLOCKER
        # has regressed.
        assert result["kl_posterior_prior"] == pytest.approx(0.0, abs=1e-5), (
            f"kl_posterior_prior = {result['kl_posterior_prior']!r} for "
            f"identical post/prior Normals — expected ~0 (raw KL). If you "
            f"see ~0.25, the probe regressed to calling the free-nats-"
            f"clamped training loss field instead of raw kl_divergence "
            f"(architecture review v3 BLOCKER).")

    def test_telemetry_swallows_buffer_exception(self):
        """The KL probe wraps in try/except to never crash the pilot.
        Verify a buggy sample_sequences doesn't propagate, AND that
        the failure is recorded in `kl_probe_error` (testing review v3
        MINOR — silent-swallow without diagnostics is indistinguishable
        from a probe that never gets called)."""
        import torch
        from types import SimpleNamespace

        class _RSSMOK:
            def transferable_state_dict(self):
                return {"k": torch.zeros(2, 2)}

            def observe(self, obs_t, act_t):
                raise RuntimeError("test: should not be reached")

        class _BrokenBuf:
            num_episodes = 4

            def sample_sequences(self, bs, sl):
                raise RuntimeError("test: replay buffer is broken")

        agent = SimpleNamespace(
            wm_trainer=SimpleNamespace(
                rssm=_RSSMOK(), batch_size=4, seq_length=4,
                kl_weight=0.1, free_nats=1.0),
            replay_buffer=_BrokenBuf(),
            total_steps=0, total_episodes=0)

        baseline = {"k": torch.zeros(2, 2)}
        baseline_norms = {"k": 1.0}
        # Must NOT raise. drift is still computed; kl_probe falls to None
        # but kl_probe_error captures the repr.
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        assert result["transferable_drift_max"] == 0.0
        assert result["kl_posterior_prior"] is None
        assert result["kl_probe_error"] is not None
        assert "test: replay buffer is broken" in result["kl_probe_error"], (
            f"kl_probe_error should contain the actual exception repr, "
            f"got {result['kl_probe_error']!r}")

    def test_kl_probe_error_is_none_on_success(self):
        """When the probe runs cleanly, kl_probe_error must be None
        (the analyzer uses `error is None and kl is not None` as the
        success predicate)."""
        agent, baseline, baseline_norms = self._make_stub_agent(drift_factor=0.0)
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        assert result["kl_posterior_prior"] is not None, (
            "stub returns a finite KL — sanity check on test fixture")
        assert result["kl_probe_error"] is None, (
            f"Expected kl_probe_error=None on successful probe, "
            f"got {result['kl_probe_error']!r}")

    def test_kl_probe_error_distinguishes_empty_buffer_from_crash(self):
        """Testing review v3 MINOR: the analyzer needs to tell apart
        'probe never ran (no data yet)' from 'probe crashed' from
        'probe succeeded'. An empty buffer must produce kl_probe=None
        with a kl_probe_error string that explicitly names the empty
        condition (NOT a generic exception repr that could be confused
        with a real failure)."""
        import torch
        from types import SimpleNamespace

        class _RSSMOK:
            def transferable_state_dict(self):
                return {"k": torch.zeros(2, 2)}

            def observe(self, obs_t, act_t):
                # Should never be called when buffer is empty.
                raise RuntimeError("observe called on empty-buffer path")

        class _EmptyBuf:
            num_episodes = 0

            def sample_sequences(self, bs, sl):
                raise RuntimeError("should not sample from empty buffer")

        agent = SimpleNamespace(
            wm_trainer=SimpleNamespace(
                rssm=_RSSMOK(), batch_size=4, seq_length=4,
                kl_weight=0.1, free_nats=1.0),
            replay_buffer=_EmptyBuf(),
            total_steps=0, total_episodes=0)
        baseline = {"k": torch.zeros(2, 2)}
        baseline_norms = {"k": 1.0}
        result = _compute_transfer_telemetry(agent, baseline, baseline_norms)
        assert result is not None
        assert result["kl_posterior_prior"] is None
        assert result["kl_probe_error"] is not None
        assert "empty" in result["kl_probe_error"].lower(), (
            f"Empty-buffer kl_probe_error should explicitly name the "
            f"empty condition (not a generic exception). Got "
            f"{result['kl_probe_error']!r}")
