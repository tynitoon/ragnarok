"""Tests for scripts/smoke_benchmark.py (preregistration §12.5).

The smoke benchmark feeds directly into compute_budget.json, which
determines whether the H1 primary endpoint (20 seeds × 500k steps × 2
arms = 20M env-steps) fits the 28-day wall-budget on RTX 4080.

An earlier version of the smoke called `collect_episode() +
train_world_model(5) + train_policy_dream(2)` which bypassed
`train_policy_real` — the actual training entry point. That produced
throughput figures ~3-5x optimistic for discrete envs (missed the PPO
batching and 4-epoch replay) and simply wrong for continuous (never
touched the SAC Q-network updates). G1 architecture review flagged this
as a publication risk; these tests pin the fix.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts import smoke_benchmark


class TestSmokeCallsCanonicalTrainingLoop:
    """Pin that `_run_one` exercises train_policy_real — the true
    bottleneck of the H1 run — not a shortcut bypass."""

    def test_run_one_calls_train_policy_real(self):
        """On a single iteration, train_policy_real must be called."""
        call_log: list[str] = []

        def make_fake_agent(config, env):
            agent = MagicMock()
            agent.total_steps = 0
            agent.total_episodes = 0
            agent.replay_buffer = MagicMock()
            agent.replay_buffer.num_episodes = 0
            agent.sac_trainer = None

            def step_train():
                call_log.append("train_policy_real")
                agent.total_steps += 100
                agent.total_episodes += 1
            agent.train_policy_real = step_train
            agent.train_world_model = lambda steps: call_log.append(
                f"wm{steps}") or None
            agent.train_policy_dream = lambda steps: call_log.append(
                f"dream{steps}") or None
            return agent

        with patch.object(smoke_benchmark, "RagnarokAgent", make_fake_agent), \
                patch.object(smoke_benchmark, "make_env", lambda *a, **k: MagicMock(close=lambda: None)):
            smoke_benchmark._run_one("cartpole", seed=0, target_steps=500)

        # Must have at least one train_policy_real call
        assert "train_policy_real" in call_log, (
            "smoke bypassed train_policy_real — throughput would be "
            "optimistic vs the canonical train.py loop"
        )

    def test_run_one_schedules_wm_training(self):
        """wm training must fire on the same cadence as train.py (every 10
        iters, gated on ≥10 episodes in buffer)."""
        call_log: list[str] = []

        def make_fake_agent(config, env):
            agent = MagicMock()
            agent.total_steps = 0
            agent.total_episodes = 0
            agent.replay_buffer = MagicMock()
            agent.replay_buffer.num_episodes = 20  # above gate
            agent.sac_trainer = None

            def step_train():
                agent.total_steps += 100
                agent.total_episodes += 1
                call_log.append("real")
            agent.train_policy_real = step_train
            agent.train_world_model = lambda steps: call_log.append(f"wm{steps}")
            agent.train_policy_dream = lambda steps: call_log.append(f"dream{steps}")
            return agent

        with patch.object(smoke_benchmark, "RagnarokAgent", make_fake_agent), \
                patch.object(smoke_benchmark, "make_env", lambda *a, **k: MagicMock(close=lambda: None)):
            smoke_benchmark._run_one(
                "cartpole", seed=0, target_steps=2000,
                wm_train_every=10, wm_train_steps=50,
                dream_train_every=20, dream_train_steps=20,
            )

        wm_calls = [c for c in call_log if c.startswith("wm")]
        assert len(wm_calls) >= 1, (
            f"wm training never fired; call_log={call_log[:20]}"
        )
        assert all(c == "wm50" for c in wm_calls)

    def test_run_one_skips_dream_for_sac_envs(self):
        """Continuous envs use SAC and must NOT run dream augmentation
        (train.py mirrors this skip)."""
        call_log: list[str] = []

        def make_fake_agent(config, env):
            agent = MagicMock()
            agent.total_steps = 0
            agent.total_episodes = 0
            agent.replay_buffer = MagicMock()
            agent.replay_buffer.num_episodes = 30
            agent.sac_trainer = MagicMock()  # continuous path

            def step_train():
                agent.total_steps += 100
                agent.total_episodes += 1
                call_log.append("real")
            agent.train_policy_real = step_train
            agent.train_world_model = lambda steps: call_log.append(f"wm{steps}")
            agent.train_policy_dream = lambda steps: call_log.append(f"dream{steps}")
            return agent

        with patch.object(smoke_benchmark, "RagnarokAgent", make_fake_agent), \
                patch.object(smoke_benchmark, "make_env", lambda *a, **k: MagicMock(close=lambda: None)):
            smoke_benchmark._run_one(
                "pendulum", seed=0, target_steps=10_000,
            )

        dream_calls = [c for c in call_log if c.startswith("dream")]
        assert dream_calls == [], (
            f"dream fired on SAC env — expected skip; got {dream_calls[:5]}"
        )

    def test_default_flags_are_benchmark_clean(self):
        """The smoke must disable reward_shaping and env_overrides so the
        measured throughput reflects the default (H1-reportable) code path
        — not a tuned variant that will overstate efficiency."""
        captured = {}

        def capture_agent(config, env):
            captured["reward_shaping"] = config.reward_shaping.enabled
            captured["env_overrides"] = config.env_overrides.enabled
            agent = MagicMock()
            agent.total_steps = 10_000  # exit loop immediately
            agent.total_episodes = 0
            agent.replay_buffer = MagicMock()
            agent.replay_buffer.num_episodes = 0
            agent.sac_trainer = None
            agent.train_policy_real = lambda: None
            return agent

        with patch.object(smoke_benchmark, "RagnarokAgent", capture_agent), \
                patch.object(smoke_benchmark, "make_env", lambda *a, **k: MagicMock(close=lambda: None)):
            smoke_benchmark._run_one("cartpole", seed=0, target_steps=1)

        assert captured["reward_shaping"] is False
        assert captured["env_overrides"] is False


class TestSmokeOutputSchema:
    """Pin the compute_budget.json schema so downstream extrapolation
    scripts don't silently break."""

    def test_summarize_groups_by_env(self):
        runs = [
            smoke_benchmark.SmokeRun("cartpole", 42, 1000, 10.0, 100.0, 5),
            smoke_benchmark.SmokeRun("cartpole", 43, 1000, 20.0, 50.0, 5),
            smoke_benchmark.SmokeRun("pendulum", 42, 1000, 40.0, 25.0, 5),
        ]
        summary = smoke_benchmark._summarize(runs)
        assert set(summary.keys()) == {"cartpole", "pendulum"}
        assert summary["cartpole"]["n_seeds"] == 2
        assert summary["pendulum"]["n_seeds"] == 1
        assert abs(summary["cartpole"]["mean_steps_per_sec"] - 75.0) < 1e-6
