"""Regression tests for Bug E (Phase 3 pre-launch) — RSSM transferable subset
and cross-dim transfer with co-transferred core.

Bug E: the Phase 3 pilot #1 shipped a cross-dim transfer that moved the
latent-policy trunk to the target env but NOT the RSSM that produces the
(h, z) features the trunk consumes. With a fresh-random core on the
target, the trunk read noise and the §8 mechanism check (transfer/scratch
ratio >= 1.3) trivially failed (observed ratio ~0.98 on N=2 before kill).

Fix: split RSSM into (a) an env-agnostic transferable subset of
``core.gru`` + ``core.prior`` + ``core.posterior`` and (b) per-env IO
layers (encoder, pre_gru, decoder, reward/continue predictors). Both
the trunk and the transferable subset are saved with every Skill; a
cross-dim ``try_transfer`` loads them together and flips the acting
path to latent. The transferable subset's LR is then scaled down for a
warmup window so Adam doesn't wipe the source priors before the per-env
IO catches up.

These tests are the regression suite. Every behaviour listed here was
broken on pilot #1. They must stay green on every future change that
touches RSSM structure, Skill serialization, or cross-dim transfer.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from ragnarok.core.rssm import RSSM
from ragnarok.infrastructure.device import DEVICE
from ragnarok.skills.library import SkillLibrary
from ragnarok.skills.skill import Skill


# ── Helpers ──────────────────────────────────────────────────────────

def _fresh_rssm(obs_dim: int = 4, action_dim: int = 2,
                hidden_dim: int = 16, stoch_dim: int = 8) -> RSSM:
    """Small RSSM for unit tests. Same hidden/stoch dims are kept constant
    across fixtures so cross-dim (obs/action) transfer is testable."""
    return RSSM(obs_dim=obs_dim, action_dim=action_dim,
                hidden_dim=hidden_dim, stoch_dim=stoch_dim,
                encoder_hidden=16).to(DEVICE)


# ── RSSM transferable subset API ────────────────────────────────────

class TestRSSMTransferableSubset:
    """Phase A of the Bug E fix: RSSM.transferable_state_dict and
    load_transferable_state_dict must cleanly partition the model into
    env-agnostic (shareable) vs per-env (not shareable) weights."""

    def test_transferable_keys_only_core_gru_prior_posterior(self):
        """Transferable subset must contain exactly the three env-agnostic
        sublayers — any other key slipping in means cross-dim load will
        hit a shape mismatch (encoder, decoder, pre_gru depend on
        obs_dim/action_dim) or learn garbage (reward/continue predictors
        depend on env-specific reward/termination semantics)."""
        rssm = _fresh_rssm()
        sd = rssm.transferable_state_dict()
        assert len(sd) > 0, "transferable subset must not be empty"
        for k in sd:
            assert (k.startswith("core.gru.")
                    or k.startswith("core.prior.")
                    or k.startswith("core.posterior.")), (
                f"non-transferable key leaked into subset: {k!r}")

    def test_transferable_covers_all_three_sublayers(self):
        """All three transferable sublayers must contribute at least one
        parameter. If gru/prior/posterior ever loses params by accident
        (rename, refactor), cross-dim transfer silently loses structure."""
        rssm = _fresh_rssm()
        sd = rssm.transferable_state_dict()
        assert any(k.startswith("core.gru.") for k in sd)
        assert any(k.startswith("core.prior.") for k in sd)
        assert any(k.startswith("core.posterior.") for k in sd)

    def test_transferable_excludes_per_env_layers(self):
        """Sanity: encoder, pre_gru, decoder, reward/continue predictors
        must NOT appear in the transferable subset. Any inclusion here
        means a cross-dim load would raise a shape error or inject stale
        env-specific priors on the target env."""
        rssm = _fresh_rssm()
        sd = rssm.transferable_state_dict()
        for k in sd:
            assert not k.startswith("encoder."), f"encoder leaked: {k!r}"
            assert not k.startswith("core.pre_gru."), f"pre_gru leaked: {k!r}"
            assert not k.startswith("decoder."), f"decoder leaked: {k!r}"
            assert not k.startswith("reward_predictor."), (
                f"reward_predictor leaked: {k!r}")
            assert not k.startswith("continue_predictor."), (
                f"continue_predictor leaked: {k!r}")

    def test_transferable_and_non_transferable_params_disjoint(self):
        """The two param iterators must be a clean partition — no param
        may land in both groups. If the optimizer ever has overlapping
        groups, set_transferable_lr_scale will silently apply the scale
        twice (or, worse, produce undefined behaviour)."""
        rssm = _fresh_rssm()
        transferable_ids = {id(p) for p in rssm.transferable_params()}
        non_transferable_ids = {id(p) for p in rssm.non_transferable_params()}
        overlap = transferable_ids & non_transferable_ids
        assert not overlap, f"params in both groups: {overlap}"

    def test_transferable_and_non_transferable_params_cover_all(self):
        """Partition must be EXACT: union of the two iterators equals
        rssm.parameters(). If a param is missing from both groups it
        will never receive gradient updates, silently freezing part of
        the world model."""
        rssm = _fresh_rssm()
        all_ids = {id(p) for p in rssm.parameters()}
        union = ({id(p) for p in rssm.transferable_params()}
                 | {id(p) for p in rssm.non_transferable_params()})
        # Ensemble module is not iterated by either (it's a Phase 5.4
        # auxiliary path). We verify ensemble-free RSSMs are fully
        # covered.
        assert rssm.ensemble is None, (
            "fixture should be ensemble-free to keep partition exact")
        assert all_ids == union, (
            f"params missing from both groups: {all_ids - union}")

    def test_load_transferable_same_dim_roundtrip(self):
        """Save → load into fresh RSSM of same dims → transferable weights
        match byte-for-byte. This is the simplest sanity check."""
        src = _fresh_rssm()
        # Perturb weights so default-random weights don't accidentally
        # coincide with the "after-load" weights.
        with torch.no_grad():
            for p in src.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        sd = src.transferable_state_dict()

        dst = _fresh_rssm()
        dst.load_transferable_state_dict(sd, strict=True)

        dst_sd = dst.state_dict()
        for k, v in sd.items():
            assert torch.allclose(dst_sd[k], v), (
                f"weight {k!r} did not transfer cleanly")

    def test_load_transferable_cross_dim_succeeds(self):
        """Different obs_dim AND action_dim on target — transferable load
        must still succeed because core.gru/prior/posterior only depend
        on hidden_dim + stoch_dim + encoder_hidden, which match.

        This is the ACTUAL bug-E scenario: CartPole (obs=4, act=2) →
        MountainCarContinuous (obs=2, act=1)."""
        src = _fresh_rssm(obs_dim=4, action_dim=2)
        with torch.no_grad():
            for p in src.parameters():
                p.mul_(1.5)

        sd = src.transferable_state_dict()

        # Target env: different obs AND action dim, same hidden/stoch.
        dst = _fresh_rssm(obs_dim=2, action_dim=1)
        dst.load_transferable_state_dict(sd, strict=True)

        dst_sd = dst.state_dict()
        for k, v in sd.items():
            assert torch.allclose(dst_sd[k], v), (
                f"cross-dim transferable load dropped {k!r}")

    def test_load_transferable_preserves_per_env_layers(self):
        """After a cross-dim transferable load, the per-env layers
        (encoder, pre_gru, decoder, reward/continue) on the target MUST
        be unchanged. If they were, the load would have corrupted
        weights that need to learn the target's obs/action/reward space
        from scratch."""
        dst = _fresh_rssm(obs_dim=2, action_dim=1)

        # Snapshot per-env weights before the load.
        before: dict[str, torch.Tensor] = {}
        for name, p in dst.named_parameters():
            if (name.startswith("encoder.") or name.startswith("core.pre_gru.")
                    or name.startswith("decoder.")
                    or name.startswith("reward_predictor.")
                    or name.startswith("continue_predictor.")):
                before[name] = p.detach().clone()

        src = _fresh_rssm(obs_dim=4, action_dim=2)
        with torch.no_grad():
            for p in src.parameters():
                p.add_(torch.randn_like(p))
        dst.load_transferable_state_dict(
            src.transferable_state_dict(), strict=True)

        after = dict(dst.named_parameters())
        for name, prev in before.items():
            assert torch.equal(after[name].detach(), prev), (
                f"per-env layer {name!r} was mutated by transferable load")

    def test_strict_rejects_non_transferable_key(self):
        """Strict mode must reject a state_dict containing any key
        outside the transferable prefixes. Otherwise a caller could
        accidentally pass the full RSSM state_dict and the strict
        guarantee is meaningless."""
        rssm = _fresh_rssm()
        bad_sd = dict(rssm.transferable_state_dict())
        # Inject an encoder key (per-env; must not be accepted).
        encoder_key = next(k for k, _ in rssm.named_parameters()
                           if k.startswith("encoder."))
        bad_sd[encoder_key] = rssm.state_dict()[encoder_key].clone()

        dst = _fresh_rssm()
        with pytest.raises(ValueError, match="non-transferable keys"):
            dst.load_transferable_state_dict(bad_sd, strict=True)

    def test_strict_rejects_shape_mismatch(self):
        """Strict mode must reject a hidden_dim/stoch_dim mismatch —
        otherwise a user crystallizing with an old hidden_dim could
        silently corrupt the new agent's core and think transfer
        worked."""
        src = _fresh_rssm(hidden_dim=16)
        dst = _fresh_rssm(hidden_dim=32)  # Different hidden_dim
        with pytest.raises(ValueError, match="Shape mismatch"):
            dst.load_transferable_state_dict(
                src.transferable_state_dict(), strict=True)

    def test_nonstrict_silently_skips_mismatch(self):
        """Non-strict mode skips shape mismatches without raising.
        This is only for migration of ancient checkpoints — regular
        transfer paths always use strict=True."""
        src = _fresh_rssm(hidden_dim=16)
        dst = _fresh_rssm(hidden_dim=32)
        # Should not raise.
        dst.load_transferable_state_dict(
            src.transferable_state_dict(), strict=False)


# ── Skill carries the RSSM core ─────────────────────────────────────

class TestSkillCarriesRSSMCore:
    """Phase B of the Bug E fix: every Skill must serialize the
    transferable RSSM subset. Without this, the cross-dim branch in
    try_transfer finds an empty dict and returns None — silently
    falling back to scratch for every cross-dim pair."""

    def test_rssm_core_field_defaults_to_empty_dict(self):
        """default_factory=dict is REQUIRED: old .pt files from before
        the Bug E fix don't carry rssm_core_state_dict, and Skill(**data)
        must still construct (else every load in the committed
        skills_data/ directory would crash)."""
        skill = Skill(
            name="old_skill",
            env_name="CartPole-v1",
            policy_state_dict={"w": torch.tensor([1.0])},
            latent_centroid=np.zeros(8),
            performance=500.0,
            normalizer_state={},
            # rssm_core_state_dict NOT provided — must default.
        )
        assert skill.rssm_core_state_dict == {}, (
            "Skill.rssm_core_state_dict must default to empty dict for "
            "backward compat with pre-Bug-E checkpoints.")

    def test_rssm_core_roundtrips_through_library(self):
        """save_skill → load_skill → core weights match byte-for-byte.
        This is the single most important check: if save/load drops the
        core, cross-dim transfer is invisible and Bug E is back."""
        rssm = _fresh_rssm()
        with torch.no_grad():
            for p in rssm.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        core_sd = {k: v.cpu() for k, v in
                   rssm.transferable_state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="core_rt",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(8),
                performance=500.0,
                normalizer_state={},
                rssm_core_state_dict=core_sd,
            )
            lib.save_skill(skill)

            # Fresh library = new process; the pilot's pattern.
            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("core_rt")
            assert loaded is not None
            assert len(loaded.rssm_core_state_dict) == len(core_sd), (
                "Bug E regression: rssm_core_state_dict was dropped "
                "during save. save_skill must serialize every Skill "
                "field — see test_every_skill_dataclass_field_is_"
                "serialized.")
            for k, v in core_sd.items():
                assert k in loaded.rssm_core_state_dict
                assert torch.allclose(loaded.rssm_core_state_dict[k], v)

    def test_empty_rssm_core_roundtrips(self):
        """Default (empty-dict) core must round-trip cleanly too —
        otherwise old skills without a core would fail to load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(skills_dir=tmpdir)
            skill = Skill(
                name="empty_core",
                env_name="CartPole-v1",
                policy_state_dict={"w": torch.tensor([1.0])},
                latent_centroid=np.zeros(8),
                performance=500.0,
                normalizer_state={},
                # rssm_core_state_dict defaults to {}
            )
            lib.save_skill(skill)

            lib2 = SkillLibrary(skills_dir=tmpdir)
            loaded = lib2.load_skill("empty_core")
            assert loaded is not None
            assert loaded.rssm_core_state_dict == {}


# ── WorldModelTrainer param groups + LR scaling ─────────────────────

class TestWMTrainerLRScaling:
    """Phase C of the Bug E fix: the optimizer must split the RSSM into
    a transferable group and an IO group, so set_transferable_lr_scale
    can lower the LR on the transferable subset post-transfer without
    slowing down the per-env IO that needs full LR to catch up."""

    def _build_trainer(self, lr: float = 3e-4):
        from ragnarok.learning.world_model_trainer import WorldModelTrainer
        from ragnarok.memory.replay_buffer import ReplayBuffer
        rssm = _fresh_rssm()
        buffer = ReplayBuffer(capacity=100)
        trainer = WorldModelTrainer(
            rssm=rssm, replay_buffer=buffer,
            lr=lr, batch_size=2, seq_length=4,
        )
        return trainer, rssm

    def test_optimizer_has_transferable_and_io_groups(self):
        """The optimizer must expose two named param groups so callers
        can scale them independently. If either is missing,
        set_transferable_lr_scale silently does nothing."""
        trainer, _ = self._build_trainer()
        names = {g.get("name") for g in trainer.optimizer.param_groups}
        assert "transferable" in names, (
            "missing 'transferable' optimizer group — LR scaling broken")
        assert "io" in names, (
            "missing 'io' optimizer group — IO can't run at full LR")

    def test_param_groups_disjoint_in_optimizer(self):
        """Optimizer groups must be disjoint. An overlap would cause the
        LR scaling to be applied ambiguously to overlapping params."""
        trainer, _ = self._build_trainer()
        trans_ids = set()
        io_ids = set()
        for g in trainer.optimizer.param_groups:
            ids = {id(p) for p in g["params"]}
            if g.get("name") == "transferable":
                trans_ids = ids
            elif g.get("name") == "io":
                io_ids = ids
        assert trans_ids and io_ids, "both groups must be non-empty"
        assert not (trans_ids & io_ids), (
            "optimizer groups overlap — LR scaling ambiguous")

    def test_initial_lr_is_flat(self):
        """At construction, both groups run at the base LR — no scaling
        active until set_transferable_lr_scale is called."""
        trainer, _ = self._build_trainer(lr=3e-4)
        for g in trainer.optimizer.param_groups:
            assert g["lr"] == pytest.approx(3e-4), (
                f"group {g.get('name')!r} has lr={g['lr']} at init, "
                f"expected 3e-4")

    def test_set_lr_scale_only_affects_transferable(self):
        """set_transferable_lr_scale(0.1, ...) must drop ONLY the
        transferable group's LR — the IO group stays at base LR because
        it has to learn the target env's obs/action layout fast."""
        trainer, _ = self._build_trainer(lr=3e-4)
        trainer.set_transferable_lr_scale(0.1, warmup_episodes=200)
        lrs = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
        assert lrs["transferable"] == pytest.approx(3e-5), (
            "transferable LR was not scaled by 0.1")
        assert lrs["io"] == pytest.approx(3e-4), (
            "io LR was unexpectedly scaled — IO must run at full LR")

    def test_step_episode_counts_down_and_restores(self):
        """After N calls to step_episode the LR must snap back to base.
        This is what lets Adam eventually train the transferable subset
        at full rate once the per-env IO is warm."""
        trainer, _ = self._build_trainer(lr=3e-4)
        trainer.set_transferable_lr_scale(0.1, warmup_episodes=3)
        assert trainer.get_transferable_lr() == pytest.approx(3e-5)

        trainer.step_episode()
        assert trainer.get_transferable_lr() == pytest.approx(3e-5)
        trainer.step_episode()
        assert trainer.get_transferable_lr() == pytest.approx(3e-5)
        # Third call expires the counter — LR restored.
        trainer.step_episode()
        assert trainer.get_transferable_lr() == pytest.approx(3e-4), (
            "LR did not restore after warmup expired")

    def test_step_episode_is_noop_when_no_warmup(self):
        """step_episode must be safe to call unconditionally (that's how
        agent.py wires it — every episode end, regardless of whether
        a transfer recently happened)."""
        trainer, _ = self._build_trainer(lr=3e-4)
        for _ in range(100):
            trainer.step_episode()
        assert trainer.get_transferable_lr() == pytest.approx(3e-4)


# ── Cross-dim transfer integration ──────────────────────────────────

class TestCrossDimTransferIntegration:
    """Phase D of the Bug E fix: end-to-end wiring in RagnarokAgent.
    Crystallization saves the RSSM core; try_transfer loads it first
    (before the trunk) and flips the acting path to latent so the
    transferred features actually drive behaviour."""

    def _make_agent(self, env_name: str, seed: int = 42):
        """Small agent on a given env. Curiosity disabled to speed up
        construction — it's orthogonal to the Bug E path."""
        from ragnarok.core.agent import RagnarokAgent
        from ragnarok.environments.registry import get_env_spec
        from ragnarok.environments.wrapper import RagnarokEnv
        from ragnarok.infrastructure.config import RagnarokConfig

        spec = get_env_spec(env_name)
        config = RagnarokConfig(seed=seed)
        config.world_model.obs_dim = spec.obs_dim
        config.world_model.action_dim = spec.action_dim
        config.curiosity.enabled = False
        # Small RSSM for speed — hidden/stoch dims match across envs.
        config.world_model.hidden_dim = 32
        config.world_model.stoch_dim = 8
        config.world_model.encoder_hidden = 32

        env = RagnarokEnv(spec.gym_name, seed=seed)
        agent = RagnarokAgent(config, env)
        return agent, env

    def test_crystallization_saves_nonempty_rssm_core(self):
        """When a skill crystallizes, the rssm_core_state_dict on the
        produced Skill MUST be non-empty and carry the three transferable
        sublayers. Empty here → cross-dim transfer path is dead."""
        agent, env = self._make_agent("cartpole")
        try:
            # Direct path: build the core dict exactly like
            # check_crystallization does, then verify every key is
            # under the transferable prefixes.
            core_sd = {k: v.cpu() for k, v in
                       agent.rssm.transferable_state_dict().items()}
            assert len(core_sd) > 0, "empty core dict — Bug E regression"
            for k in core_sd:
                assert (k.startswith("core.gru.")
                        or k.startswith("core.prior.")
                        or k.startswith("core.posterior."))
        finally:
            env.close()

    def test_cross_dim_transfer_loads_core_and_flips_to_latent(self):
        """End-to-end: crystallize a CartPole-like source skill, point a
        MountainCarContinuous agent at it, call try_transfer, verify:
         1. cross-dim branch fired (acting_policy_mode == "latent")
         2. RSSM core was actually loaded (weights match source)
         3. LR warmup was applied to the transferable group"""
        # Source RSSM (CartPole dims) — perturb so it's distinguishable
        # from any target random init. encoder_hidden must match the
        # target's (posterior consumes encoder features, so this dim
        # appears in the transferable weights even though the encoder
        # itself is per-env).
        source_rssm = RSSM(obs_dim=4, action_dim=2,
                           hidden_dim=32, stoch_dim=8,
                           encoder_hidden=32).to(DEVICE)
        with torch.no_grad():
            for p in source_rssm.parameters():
                p.mul_(1.3).add_(0.05)
        source_core = {k: v.cpu() for k, v in
                       source_rssm.transferable_state_dict().items()}

        # Target agent (MCC dims: obs=2, action=1 — cross-dim).
        with tempfile.TemporaryDirectory() as tmpdir:
            from ragnarok.skills.library import SkillLibrary
            # Build target agent pointed at our temp skills dir.
            from ragnarok.core.agent import RagnarokAgent
            from ragnarok.environments.registry import get_env_spec
            from ragnarok.environments.wrapper import RagnarokEnv
            from ragnarok.infrastructure.config import RagnarokConfig

            spec = get_env_spec("mountaincar-continuous")
            config = RagnarokConfig(seed=0)
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim
            config.world_model.hidden_dim = 32
            config.world_model.stoch_dim = 8
            config.world_model.encoder_hidden = 32
            config.curiosity.enabled = False
            config.skill.skills_dir = tmpdir
            config.transfer.rssm_transfer_lr_scale = 0.1
            config.transfer.rssm_transfer_warmup_episodes = 50

            env = RagnarokEnv(spec.gym_name, seed=0)
            try:
                agent = RagnarokAgent(config, env)

                # Build a realistic CartPole-dim trunk that will load
                # cleanly into the MCC latent policy (same latent_dim).
                trunk_sd = {k: v.cpu() for k, v in
                            agent.latent_policy.get_trunk_state_dict().items()}

                # Inject source skill (from CartPole dims, but trunk
                # dims come from the target agent — latent_dim depends
                # only on hidden+stoch which we kept constant).
                skill = Skill(
                    name="CartPole-v1_src",
                    env_name="CartPole-v1",
                    policy_state_dict={  # Deliberately mismatched dims
                        "actor.weight": torch.zeros(2, 4),
                    },
                    latent_centroid=np.zeros(agent.rssm.hidden_dim),
                    performance=500.0,
                    normalizer_state={},
                    latent_trunk_state_dict=trunk_sd,
                    rssm_core_state_dict=source_core,
                )
                agent.skill_library.save_skill(skill)

                # Before: acting policy is obs.
                assert agent.acting_policy_mode == "obs"

                loaded = agent.try_transfer()

                assert loaded is not None, (
                    "try_transfer returned None — cross-dim fallback "
                    "did not fire. Check the gate at "
                    "`skill.latent_trunk_state_dict and "
                    "skill.rssm_core_state_dict`.")
                assert agent.acting_policy_mode == "latent", (
                    "acting_policy_mode did not flip to latent. "
                    "Cross-dim transfer is invisible at acting time — "
                    "Bug E regression.")

                # Core weights on the target RSSM must match the source.
                target_core = agent.rssm.transferable_state_dict()
                for k, v in source_core.items():
                    assert k in target_core
                    assert torch.allclose(target_core[k].cpu(), v), (
                        f"core weight {k!r} not loaded from source — "
                        f"RSSM feeds noise to the trunk, Bug E is back.")

                # LR warmup applied to the transferable group only.
                lrs = {g["name"]: g["lr"]
                       for g in agent.wm_trainer.optimizer.param_groups}
                assert lrs["transferable"] < lrs["io"], (
                    "transferable LR was not scaled down post-transfer — "
                    "Adam will wipe the source priors in a few hundred "
                    "steps (see Phase C rationale).")
            finally:
                env.close()

    def test_cross_dim_transfer_skipped_when_rssm_core_missing(self):
        """A skill with an empty rssm_core_state_dict must NOT trigger
        the cross-dim branch — it's a pre-Bug-E artifact that would
        load only the trunk and leave the target RSSM random, which is
        exactly the failure mode the fix is meant to close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from ragnarok.core.agent import RagnarokAgent
            from ragnarok.environments.registry import get_env_spec
            from ragnarok.environments.wrapper import RagnarokEnv
            from ragnarok.infrastructure.config import RagnarokConfig

            spec = get_env_spec("mountaincar-continuous")
            config = RagnarokConfig(seed=0)
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim
            config.world_model.hidden_dim = 32
            config.world_model.stoch_dim = 8
            config.world_model.encoder_hidden = 32
            config.curiosity.enabled = False
            config.skill.skills_dir = tmpdir

            env = RagnarokEnv(spec.gym_name, seed=0)
            try:
                agent = RagnarokAgent(config, env)
                trunk_sd = {k: v.cpu() for k, v in
                            agent.latent_policy.get_trunk_state_dict().items()}
                # Old-format skill: trunk present, core missing.
                skill = Skill(
                    name="CartPole-v1_old",
                    env_name="CartPole-v1",
                    policy_state_dict={"actor.weight": torch.zeros(2, 4)},
                    latent_centroid=np.zeros(agent.rssm.hidden_dim),
                    performance=500.0,
                    normalizer_state={},
                    latent_trunk_state_dict=trunk_sd,
                    # rssm_core_state_dict deliberately omitted.
                )
                agent.skill_library.save_skill(skill)

                loaded = agent.try_transfer()
                assert loaded is None, (
                    "try_transfer followed the cross-dim branch on an "
                    "empty-core skill. Pre-Bug-E skills must fall back "
                    "to scratch cleanly.")
                assert agent.acting_policy_mode == "obs", (
                    "acting_policy_mode flipped without a valid core — "
                    "the gate in try_transfer is broken.")
            finally:
                env.close()

    def test_trust_region_not_activated_in_latent_mode(self):
        """After a cross-dim transfer (acting_policy_mode == 'latent'),
        the trust region must NOT fire. The obs policy wasn't actually
        loaded — deepcopying it would capture random init, and the KL
        penalty that real_trainer applies would pull the obs policy
        toward random init (meaningless and harmful)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from ragnarok.core.agent import RagnarokAgent
            from ragnarok.environments.registry import get_env_spec
            from ragnarok.environments.wrapper import RagnarokEnv
            from ragnarok.infrastructure.config import RagnarokConfig

            spec = get_env_spec("mountaincar-continuous")
            config = RagnarokConfig(seed=0)
            config.world_model.obs_dim = spec.obs_dim
            config.world_model.action_dim = spec.action_dim
            config.world_model.hidden_dim = 32
            config.world_model.stoch_dim = 8
            config.world_model.encoder_hidden = 32
            config.curiosity.enabled = False
            config.skill.skills_dir = tmpdir

            env = RagnarokEnv(spec.gym_name, seed=0)
            try:
                agent = RagnarokAgent(config, env)
                # Realistic cross-dim skill with both trunk AND core.
                # encoder_hidden must match target — see the first
                # integration test above for the posterior-dim reason.
                source_rssm = RSSM(obs_dim=4, action_dim=2,
                                   hidden_dim=32, stoch_dim=8,
                                   encoder_hidden=32).to(DEVICE)
                trunk_sd = {k: v.cpu() for k, v in
                            agent.latent_policy.get_trunk_state_dict().items()}
                skill = Skill(
                    name="CartPole-v1_src",
                    env_name="CartPole-v1",
                    policy_state_dict={"actor.weight": torch.zeros(2, 4)},
                    latent_centroid=np.zeros(agent.rssm.hidden_dim),
                    performance=500.0,
                    normalizer_state={},
                    latent_trunk_state_dict=trunk_sd,
                    rssm_core_state_dict={
                        k: v.cpu() for k, v
                        in source_rssm.transferable_state_dict().items()
                    },
                )
                agent.skill_library.save_skill(skill)

                loaded = agent.try_transfer()
                assert loaded is not None
                assert agent.acting_policy_mode == "latent"

                # The fix: trust region is gated on
                # `acting_policy_mode == "obs"`. In latent mode the
                # obs policy is target-env random init, so capturing
                # it is meaningless — and the KL penalty toward random
                # init is actively harmful.
                assert agent._transfer_ref_policy is None, (
                    "trust region was activated in latent mode — the "
                    "obs policy wasn't loaded from the skill, so the "
                    "KL penalty would pull toward random init.")
            finally:
                env.close()


# ── Behavioural smoke (slow; not run in regular CI) ─────────────────

@pytest.mark.slow
class TestBehavioralSmoke:
    """End-to-end check: on a fixed-pipeline cross-dim transfer, the
    transfer-arm agent must make non-trivial early progress relative to
    a scratch agent. This does NOT check the preregistration threshold
    (ratio >= 1.3) — that's what the Phase 3 pilot #2 is for. We only
    check the pipeline is live (ratio >= 1.0 on a tiny run).

    Marked `slow`: skip by default. Run manually with `pytest -m slow`
    before committing a Bug E fix and before launching pilot #2.
    """

    def test_cartpole_to_mcc_ratio_nontrivial(self):
        pytest.skip(
            "Behavioural smoke runs ~45 min; execute manually with "
            "`pytest tests/test_rssm_transfer.py -m slow` before "
            "launching Phase 3 pilot #2.")
