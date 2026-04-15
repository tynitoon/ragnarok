"""Unit tests for `scripts.smoke_verdict`.

This is the CLI tool that enforces the smoke pre-check abort rule
the prereg commits to (v3.4 amendments "Bug E v2" through "Bug E v4"):
EITHER seed showing `transferable_drift_max > 0.50` at any telemetry
checkpoint with `episode <= 100` aborts the pilot #2 launch.

Devil's-advocate v4 review explicitly identified this as the
operational gate for pilot #2 launch ("Launch pilot #2 iff (a)
telemetry arrays non-empty + kl_probe_error: null for both transfer
seeds, (b) max drift < 0.50 at every checkpoint <= ep 100 on both
seeds"). Without an automated tool, this gate is unenforced.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from scripts.smoke_verdict import (
    DRIFT_ABORT_THRESHOLD,
    DRIFT_ABORT_EPISODE_GATE,
    EXPECTED_BUFFER_EMPTY_MSG,
    SeedVerdict,
    SmokeVerdict,
    evaluate,
    main,
)


def _make_telemetry(*, drifts_at_ep: dict[int, float],
                    probe_errors: dict[int, str | None] | None = None
                    ) -> list[dict]:
    """Build a synthetic telemetry list. Each dict mirrors the v4
    schema produced by `_compute_transfer_telemetry` in pilot_run.py.
    """
    if probe_errors is None:
        probe_errors = {}
    out = []
    for ep, drift in sorted(drifts_at_ep.items()):
        out.append({
            "step": ep * 100,  # arbitrary; analyzer keys on episode
            "episode": ep,
            "transferable_drift_max": drift,
            "transferable_drift_per_param": {"core.gru.weight_ih_l0": drift},
            "kl_posterior_prior": 0.5,  # any non-None float
            "kl_probe_error": probe_errors.get(ep),
        })
    return out


def _make_run(*, seed: int = 42, arm: str = "transfer",
              source_crystallized: bool = True,
              telemetry: list[dict] | None = None,
              pair_alias: str = "cartpole_mcc") -> dict:
    """Build a single PilotRun-equivalent dict for the analyzer."""
    return {
        "pair_alias": pair_alias,
        "pair_role": "primary",
        "src_env": "cartpole",
        "tgt_env": "mountaincar-continuous",
        "seed": seed,
        "arm": arm,
        "source_crystallized": source_crystallized,
        "telemetry": telemetry or [],
    }


def _make_payload(runs: list[dict]) -> dict:
    """Wrap runs in the top-level payload structure smoke_verdict
    expects."""
    return {"runs": runs}


class TestDriftAbortRule:
    """The core prereg rule: EITHER seed > 50% drift in ep <= 100 → ABORT."""

    def test_clean_run_below_threshold_proceeds(self):
        tele = _make_telemetry(drifts_at_ep={50: 0.10, 100: 0.20, 150: 0.30})
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is True
        assert len(v.seed_verdicts) == 1
        assert v.seed_verdicts[0].pass_
        assert v.seed_verdicts[0].drift_abort_triggered is False

    def test_drift_above_threshold_in_window_aborts(self):
        # 0.55 > 0.50 at ep 80 (within window) → ABORT
        tele = _make_telemetry(drifts_at_ep={50: 0.20, 80: 0.55, 100: 0.30})
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is False
        assert v.seed_verdicts[0].drift_abort_triggered is True
        assert v.seed_verdicts[0].max_drift_in_window == pytest.approx(0.55)

    def test_drift_at_threshold_does_not_abort(self):
        # 0.50 == threshold → strict-greater check → still PROCEED
        tele = _make_telemetry(drifts_at_ep={100: 0.50})
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is True
        assert v.seed_verdicts[0].drift_abort_triggered is False

    def test_drift_above_threshold_outside_window_does_not_abort(self):
        # 0.60 at ep 150 (>gate of 100) → does NOT trigger abort
        tele = _make_telemetry(drifts_at_ep={50: 0.10, 100: 0.20, 150: 0.60})
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is True
        assert v.seed_verdicts[0].drift_abort_triggered is False
        # Window max should reflect only ep <= 100.
        assert v.seed_verdicts[0].max_drift_in_window == pytest.approx(0.20)

    def test_either_seed_aborts(self):
        """The prereg rule is OR across seeds — single seed > 50% halts
        pilot launch even if the other seed is clean."""
        clean_tele = _make_telemetry(drifts_at_ep={50: 0.05, 100: 0.10})
        bad_tele = _make_telemetry(drifts_at_ep={50: 0.05, 100: 0.65})
        payload = _make_payload([
            _make_run(seed=42, telemetry=clean_tele),
            _make_run(seed=43, telemetry=bad_tele),
        ])
        v = evaluate(payload)
        assert v.proceed is False  # ABORT due to seed 43
        # Per-seed verdicts: 42 passes, 43 fails
        by_seed = {sv.seed: sv for sv in v.seed_verdicts}
        assert by_seed[42].pass_
        assert not by_seed[43].pass_

    def test_no_telemetry_with_crystallized_source_fails(self):
        """If source crystallized but telemetry is empty, the snapshot
        block in _train_to_step_budget regressed silently. Fail."""
        payload = _make_payload([_make_run(
            telemetry=[], source_crystallized=True)])
        v = evaluate(payload)
        assert v.proceed is False
        assert "telemetry pipeline regression" in v.summary_lines[-1] or \
               any("regression" in n for n in v.seed_verdicts[0].notes)

    def test_no_telemetry_uncrystallized_source_also_fails(self):
        """If the source didn't crystallize, transfer fell back to scratch
        and the smoke can't validate anything. Still fails the gate."""
        payload = _make_payload([_make_run(
            telemetry=[], source_crystallized=False)])
        v = evaluate(payload)
        assert v.proceed is False
        # Note string distinguishes this from the regression case.
        assert any("scratch" in n for n in v.seed_verdicts[0].notes)


class TestProbeErrorRule:
    """A non-None, non-buffer-empty kl_probe_error indicates a real probe
    crash. The analyzer must catch this."""

    def test_buffer_empty_error_does_not_fail(self):
        """The empty-buffer sentinel is expected on the first
        checkpoint(s) before the replay buffer fills. NOT an error."""
        tele = _make_telemetry(
            drifts_at_ep={50: 0.10, 100: 0.20},
            probe_errors={50: EXPECTED_BUFFER_EMPTY_MSG, 100: None},
        )
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is True
        assert v.seed_verdicts[0].probe_error_count == 0

    def test_real_probe_exception_fails(self):
        tele = _make_telemetry(
            drifts_at_ep={50: 0.10, 100: 0.20},
            probe_errors={100: "RuntimeError('CUDA out of memory')"},
        )
        payload = _make_payload([_make_run(telemetry=tele)])
        v = evaluate(payload)
        assert v.proceed is False
        assert v.seed_verdicts[0].probe_error_count == 1


class TestSchemaCompatibility:
    """The analyzer must accept either the flat 'runs' top-level list
    or the older 'pairs' grouped schema."""

    def test_flat_runs_schema(self):
        payload = {"runs": [_make_run(
            telemetry=_make_telemetry(drifts_at_ep={100: 0.10}))]}
        v = evaluate(payload)
        assert len(v.seed_verdicts) == 1

    def test_grouped_pairs_schema(self):
        payload = {
            "pairs": [{
                "pair_alias": "cartpole_mcc",
                "transfer_runs": [_make_run(
                    telemetry=_make_telemetry(drifts_at_ep={100: 0.10}))],
            }]
        }
        v = evaluate(payload)
        assert len(v.seed_verdicts) == 1

    def test_non_transfer_arms_ignored(self):
        """Source and scratch arms shouldn't appear in the verdict."""
        payload = _make_payload([
            _make_run(arm="source", telemetry=[]),
            _make_run(arm="scratch", telemetry=[]),
            _make_run(arm="transfer",
                      telemetry=_make_telemetry(drifts_at_ep={100: 0.10})),
        ])
        v = evaluate(payload)
        assert len(v.seed_verdicts) == 1
        assert v.seed_verdicts[0].arm == "transfer"

    def test_no_transfer_runs_aborts(self):
        """If there are no transfer arms at all, can't validate launch."""
        payload = _make_payload([
            _make_run(arm="source", telemetry=[]),
            _make_run(arm="scratch", telemetry=[]),
        ])
        v = evaluate(payload)
        assert v.proceed is False


class TestCLI:
    """End-to-end check on the main() CLI: exit code 0 on PROCEED,
    1 on ABORT, 2 on file-not-found."""

    def test_main_returns_zero_on_proceed(self, tmp_path, capsys):
        path = tmp_path / "smoke.json"
        payload = _make_payload([_make_run(
            telemetry=_make_telemetry(drifts_at_ep={100: 0.10}))])
        path.write_text(json.dumps(payload))
        rc = main([str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PROCEED" in out

    def test_main_returns_one_on_abort(self, tmp_path, capsys):
        path = tmp_path / "smoke.json"
        payload = _make_payload([_make_run(
            telemetry=_make_telemetry(drifts_at_ep={100: 0.65}))])
        path.write_text(json.dumps(payload))
        rc = main([str(path)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "ABORT" in out

    def test_main_returns_two_on_file_not_found(self, tmp_path, capsys):
        rc = main([str(tmp_path / "nope.json")])
        assert rc == 2
