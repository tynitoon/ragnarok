"""Tests for skill system."""

import numpy as np
import torch
import pytest
import tempfile
import os
from pathlib import Path
from ragnarok.skills.skill import Skill
from ragnarok.skills.library import SkillLibrary


class TestSkill:
    def test_create_skill(self):
        skill = Skill(
            name="test_skill",
            env_name="CartPole-v1",
            policy_state_dict={"weight": torch.randn(3, 3)},
            latent_centroid=np.zeros(128),
            performance=450.0,
            normalizer_state={"mean": np.zeros(4)},
        )
        assert skill.name == "test_skill"
        assert skill.performance == 450.0


class TestSkillLibrary:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="test",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0, 2.0])},
                latent_centroid=np.array([1.0, 2.0, 3.0]),
                performance=500.0,
                normalizer_state={},
            )
            lib.save_skill(skill)
            assert "test" in lib.list_skills()

            # Load in new library
            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("test")
            assert loaded is not None
            assert loaded.performance == 500.0

    def test_find_nearest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            for i, name in enumerate(["skill_a", "skill_b", "skill_c"]):
                skill = Skill(
                    name=name,
                    env_name="test",
                    policy_state_dict={},
                    latent_centroid=np.array([float(i * 10)] * 3),
                    performance=100.0,
                    normalizer_state={},
                )
                lib.save_skill(skill)

            # Query close to skill_b (centroid [10, 10, 10])
            nearest, dist = lib.find_nearest(np.array([11.0, 11.0, 11.0]))
            assert nearest.name == "skill_b"


# ── Phase 3 pre-launch regression: committed source skills ─────────
#
# skills_data/*.pt are the CRITICAL artifacts for Phase 3's transfer arm.
# If the Skill dataclass schema silently changes (or Phase 2.3's SAC
# rewrite touched anything that a skill checkpoint indirectly depends on),
# these files become unloadable and the transfer arm silently falls back
# to scratch — which would be a measured "null result" on a broken
# pipeline. These tests load the committed .pt files directly and verify
# the `try_transfer` contract still reads them.

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills_data"
CARTPOLE_SKILL = SKILLS_DIR / "CartPole-v1_280ep.pt"
MOUNTAINCAR_SKILL = SKILLS_DIR / "MountainCar-v0_560ep.pt"


class TestCommittedSkillsLoadable:
    """Phase 3 pre-launch: every skill checkpoint committed to git must be
    loadable via the current Skill dataclass + SkillLibrary API. Guards
    against silent schema drift."""

    @pytest.mark.skipif(not CARTPOLE_SKILL.exists(),
                        reason="CartPole skill checkpoint not present on this worktree")
    def test_cartpole_skill_torch_load(self):
        """Raw torch.load path — the same call SkillLibrary makes at
        construction time (library.py:28). If this raises, the Phase 3
        pilot's cartpole_mcc + cartpole_acrobot pairs are both broken."""
        data = torch.load(CARTPOLE_SKILL, weights_only=False)
        assert isinstance(data, dict), "skill file must pickle to a dict"
        # Required keys per Skill.__init__ signature
        for key in ("name", "env_name", "policy_state_dict",
                    "latent_centroid", "performance", "normalizer_state"):
            assert key in data, f"missing required key: {key}"

    @pytest.mark.skipif(not CARTPOLE_SKILL.exists(),
                        reason="CartPole skill checkpoint not present on this worktree")
    def test_cartpole_skill_construct_skill_dataclass(self):
        """Round-trip: raw dict must construct a valid Skill. This catches
        the case where the dataclass gained a non-defaulted field that
        older checkpoints don't carry."""
        data = torch.load(CARTPOLE_SKILL, weights_only=False)
        skill = Skill(**data)
        assert skill.env_name == "CartPole-v1"
        assert isinstance(skill.policy_state_dict, dict)
        assert len(skill.policy_state_dict) > 0, (
            "empty policy_state_dict → try_transfer would load nothing")
        # Every value must be a torch.Tensor (or tensor-like). try_transfer
        # does `v.to(DEVICE)` in the dict comprehension, which requires
        # tensor-like semantics.
        for k, v in skill.policy_state_dict.items():
            assert hasattr(v, "to"), (
                f"policy_state_dict[{k!r}] not tensor-like (type={type(v)})")
        assert isinstance(skill.latent_centroid, np.ndarray)
        assert skill.latent_centroid.ndim == 1
        assert isinstance(skill.normalizer_state, dict)

    @pytest.mark.skipif(not MOUNTAINCAR_SKILL.exists(),
                        reason="MountainCar skill checkpoint not present on this worktree")
    def test_mountaincar_skill_construct_skill_dataclass(self):
        """Same contract for MountainCar-v0 source skill (discrete, used as
        a sanity-check for same-dim transfer paths)."""
        data = torch.load(MOUNTAINCAR_SKILL, weights_only=False)
        skill = Skill(**data)
        assert skill.env_name == "MountainCar-v0"
        assert isinstance(skill.policy_state_dict, dict)
        assert len(skill.policy_state_dict) > 0

    @pytest.mark.skipif(not CARTPOLE_SKILL.exists() or not MOUNTAINCAR_SKILL.exists(),
                        reason="skill checkpoints not present on this worktree")
    def test_skill_library_discovers_committed_skills(self):
        """SkillLibrary constructed on the actual skills_data/ directory
        must find both committed .pt files with no warnings. This is the
        exact call pilot_run.py makes before launching a transfer arm."""
        import io
        import contextlib
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), \
                contextlib.redirect_stdout(stderr):
            lib = SkillLibrary(skills_dir=str(SKILLS_DIR))
        names = lib.list_skills()
        assert len(names) >= 2, (
            f"SkillLibrary found {len(names)} skills; expected ≥2. "
            f"stderr: {stderr.getvalue()!r}")
        assert "CartPole-v1_280ep" in names
        assert "MountainCar-v0_560ep" in names
        # And no "Warning: failed to load skill ..." output was emitted.
        assert "failed to load skill" not in stderr.getvalue(), (
            f"SkillLibrary silently skipped a skill: {stderr.getvalue()}")

    @pytest.mark.skipif(not CARTPOLE_SKILL.exists(),
                        reason="CartPole skill checkpoint not present on this worktree")
    def test_cartpole_skill_moves_to_device(self):
        """try_transfer does `{k: v.to(DEVICE) for k, v ...}` (agent.py:742).
        If any tensor lives on a device the pilot host can't reach (e.g. an
        MPS checkpoint on a CUDA host), that call raises. Exercise it."""
        from ragnarok.infrastructure.device import DEVICE
        data = torch.load(CARTPOLE_SKILL, weights_only=False)
        skill = Skill(**data)
        moved = {k: v.to(DEVICE) for k, v in skill.policy_state_dict.items()}
        # Every tensor must now be on DEVICE (smoke-check one).
        first_tensor = next(iter(moved.values()))
        assert first_tensor.device.type == DEVICE.type, (
            f"expected {DEVICE.type}, got {first_tensor.device.type}")


# ── Phase 3 pre-launch regression: Bug C (save_skill dropped trunk) ─
#
# Smoke #3 exposed that SkillLibrary.save_skill serialized only a
# hard-coded subset of Skill fields, OMITTING latent_trunk_state_dict.
# Effect: the in-memory Skill produced by check_crystallization had a
# populated trunk, but the on-disk .pt stripped it. The very next
# process (the pilot's transfer arm, which loads via SkillLibrary)
# saw an empty trunk → the RuntimeError branch in try_transfer
# (agent.py:764) returned None → acting_policy_mode stayed "obs" →
# the §8 mechanism check trivially fails on every cross-dim run.
#
# These tests lock the save/load contract: EVERY Skill field must
# survive a disk round-trip.

class TestSaveSkillPreservesTrunk:
    """Bug C regression: save_skill must serialize
    latent_trunk_state_dict so that cross-dim transfer works."""

    def test_trunk_survives_save_load_roundtrip(self):
        """Direct round-trip: Skill with trunk → save → load from
        fresh library → trunk must still be populated."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            trunk = {
                "shared.0.weight": torch.randn(64, 4),
                "shared.0.bias": torch.randn(64),
                "shared.2.weight": torch.randn(64, 64),
                "shared.2.bias": torch.randn(64),
            }
            skill = Skill(
                name="trunk_test",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(128),
                performance=500.0,
                normalizer_state={},
                latent_trunk_state_dict=trunk,
            )
            lib.save_skill(skill)

            # Fresh library reads from disk — this is the pilot's pattern
            # (source process saves; transfer process loads).
            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("trunk_test")
            assert loaded is not None
            assert len(loaded.latent_trunk_state_dict) == len(trunk), (
                "Bug C regression: latent_trunk_state_dict was dropped "
                "during save. save_skill must serialize every Skill "
                "field, not a hand-curated subset."
            )
            for k, v in trunk.items():
                assert k in loaded.latent_trunk_state_dict
                assert torch.allclose(loaded.latent_trunk_state_dict[k], v)

    def test_trunk_survives_when_empty(self):
        """Default (empty-dict) trunk must also round-trip cleanly —
        otherwise older skills without a trunk would fail to load."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="no_trunk",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(128),
                performance=500.0,
                normalizer_state={},
                # latent_trunk_state_dict defaults to {}
            )
            lib.save_skill(skill)

            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("no_trunk")
            assert loaded is not None
            assert loaded.latent_trunk_state_dict == {}

    def test_serialized_data_has_trunk_key(self):
        """Byte-level contract: the .pt file on disk must contain the
        `latent_trunk_state_dict` key. Guards against a future
        save_skill rewrite that rearranges fields without preserving
        this one."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="byte_check",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(128),
                performance=500.0,
                normalizer_state={},
                latent_trunk_state_dict={"shared.0.weight": torch.zeros(3, 3)},
            )
            lib.save_skill(skill)

            # Load raw bytes (bypass SkillLibrary) and inspect keys.
            raw = torch.load(Path(tmpdir) / "byte_check.pt",
                             weights_only=False)
            assert "latent_trunk_state_dict" in raw, (
                "Bug C regression: serialized .pt is missing the "
                "latent_trunk_state_dict key. Cross-dim transfer will "
                "silently fail."
            )
            assert len(raw["latent_trunk_state_dict"]) == 1

    def test_every_skill_dataclass_field_is_serialized(self):
        """Meta-test: save_skill must serialize every field declared
        on the Skill dataclass. This catches future fields being added
        to the dataclass without also being added to save_skill's
        hand-curated dict — which is exactly how Bug C went unnoticed."""
        import dataclasses
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="all_fields",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(128),
                performance=500.0,
                normalizer_state={},
                episodes_trained=42,
                metadata={"note": "hi"},
                latent_trunk_state_dict={"k": torch.zeros(2, 2)},
            )
            lib.save_skill(skill)

            raw = torch.load(Path(tmpdir) / "all_fields.pt",
                             weights_only=False)
            expected_fields = {f.name for f in
                               dataclasses.fields(Skill)}
            # Every dataclass field must land in the serialized blob.
            missing = expected_fields - set(raw.keys())
            assert not missing, (
                f"save_skill is silently dropping fields: {missing}. "
                f"Add them to the serialized dict so cross-process "
                f"transfer doesn't lose state."
            )
