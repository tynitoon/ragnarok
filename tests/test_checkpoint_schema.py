"""Pinning tests for checkpoint schema v2 (preregistration Phase 1.4).

Phase 1.4 froze the checkpoint schema:
  - Required keys: rssm, policy, latent_policy, acting_policy_mode,
    normalizer, episodic_memory, total_episodes, total_steps,
    episode_rewards, schema_version.
  - No backward-compat shims — pre-Phase-5.3 ckpts with `actor_critic`
    or missing `acting_policy_mode` must fail loudly.
  - Architecture mismatch in load_state_dict must propagate (not be
    silently swallowed).

The migration script `scripts.migrate_checkpoints` upgrades old ckpts
once; after migration the strict loader must accept them (modulo any
still-absent keys that cannot be synthesized, like `latent_policy`).
"""

import os
import tempfile

import pytest
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.environments.registry import get_env_spec
from ragnarok.environments.wrapper import RagnarokEnv
from ragnarok.core.agent import RagnarokAgent


def _make_agent():
    spec = get_env_spec("cartpole")
    cfg = RagnarokConfig(seed=0)
    cfg.world_model.obs_dim = spec.obs_dim
    cfg.world_model.action_dim = spec.action_dim
    cfg.curiosity.enabled = False
    env = RagnarokEnv(spec.gym_name, seed=0)
    return RagnarokAgent(cfg, env), env


class TestSchemaVersion:
    def test_save_writes_schema_version(self):
        agent, env = _make_agent()
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            try:
                agent.save(path)
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                assert ckpt["schema_version"] == agent.CHECKPOINT_SCHEMA_VERSION
                assert agent.CHECKPOINT_SCHEMA_VERSION == 2
            finally:
                os.unlink(path)
        finally:
            env.close()

    def test_load_round_trip_on_fresh_save(self):
        """A freshly-saved ckpt loads cleanly into a new agent."""
        a1, env = _make_agent()
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            try:
                a1.save(path)
                a2, _ = _make_agent()
                a2.load(path)  # Must not raise
            finally:
                os.unlink(path)
        finally:
            env.close()


class TestStrictLoadRejectsOldSchema:
    def _write_ckpt(self, path: str, **overrides):
        """Write a synthetic checkpoint with custom fields."""
        agent, env = _make_agent()
        try:
            agent.save(path)
        finally:
            env.close()
        # Patch the saved dict
        if overrides:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            for k, v in overrides.items():
                if v is _DELETE:
                    ckpt.pop(k, None)
                else:
                    ckpt[k] = v
            torch.save(ckpt, path)

    def test_missing_schema_version_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, schema_version=_DELETE)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"schema_version"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_old_schema_version_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, schema_version=1)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"schema_version=1"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_error_message_points_to_migration_script(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, schema_version=None)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"migrate_checkpoints"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_missing_policy_key_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, policy=_DELETE)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"missing required keys"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_missing_acting_policy_mode_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, acting_policy_mode=_DELETE)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"acting_policy_mode"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_missing_latent_policy_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self._write_ckpt(path, latent_policy=_DELETE)
            agent, env = _make_agent()
            try:
                with pytest.raises(ValueError, match=r"latent_policy"):
                    agent.load(path)
            finally:
                env.close()
        finally:
            os.unlink(path)

    def test_architecture_mismatch_raises(self):
        """A policy state_dict with wrong shapes must raise, not silently
        load a random-init policy."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            # Corrupt the policy state_dict with bad shapes
            agent, env = _make_agent()
            try:
                agent.save(path)
            finally:
                env.close()
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            # Inject a bogus tensor for a random policy param
            policy_keys = list(ckpt["policy"].keys())
            assert len(policy_keys) > 0
            ckpt["policy"][policy_keys[0]] = torch.zeros(1)  # Wrong shape
            torch.save(ckpt, path)

            a2, env2 = _make_agent()
            try:
                with pytest.raises(RuntimeError):
                    a2.load(path)
            finally:
                env2.close()
        finally:
            os.unlink(path)


class TestMigrationScript:
    def test_migration_upgrades_old_ckpt(self):
        """Old-format ckpt (actor_critic, no acting_policy_mode) → v2."""
        from scripts.migrate_checkpoints import migrate_one
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            # Synthesize an old-format ckpt
            old_ckpt = {
                "rssm": {},  # Empty, contents don't matter for this test
                "actor_critic": {"dummy": torch.zeros(1)},
                "normalizer": {"mean": [0.0], "std": [1.0]},
                "episodic_memory": {},
                "total_episodes": 42,
                "total_steps": 1000,
                "episode_rewards": [1.0, 2.0, 3.0],
            }
            torch.save(old_ckpt, path)

            status = migrate_one(Path(path))
            assert "MIG" in status
            assert "actor_critic" in status

            new_ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert new_ckpt["schema_version"] == 2
            assert "policy" in new_ckpt
            assert "actor_critic" not in new_ckpt
            assert new_ckpt["acting_policy_mode"] == "obs"
        finally:
            os.unlink(path)

    def test_migration_idempotent(self):
        """Running migration twice is a no-op."""
        from scripts.migrate_checkpoints import migrate_one
        from pathlib import Path

        agent, env = _make_agent()
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            try:
                agent.save(path)
                status = migrate_one(Path(path))
                assert "OK" in status or "already" in status
            finally:
                os.unlink(path)
        finally:
            env.close()

    def test_migration_dry_run_does_not_write(self):
        from scripts.migrate_checkpoints import migrate_one
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            old_ckpt = {"actor_critic": {"dummy": torch.zeros(1)}}
            torch.save(old_ckpt, path)

            status = migrate_one(Path(path), dry_run=True)
            assert "DRY" in status

            # File should still have actor_critic, no schema_version
            check = torch.load(path, map_location="cpu", weights_only=False)
            assert "actor_critic" in check
            assert "schema_version" not in check
        finally:
            os.unlink(path)


# Sentinel object used by _write_ckpt helper to signal key deletion.
# (None is a valid value we might want to inject.)
class _Delete:
    pass


_DELETE = _Delete()
