"""Smoke-pre-check verdict tool.

Consumes a `pilot_results.json` produced by `scripts.pilot_run --smoke`
and renders a PROCEED / ABORT verdict against the prereg's smoke
abort criteria (preregistration.md v3.4 amendments "Bug E v2" through
"Bug E v4").

Decision rule (v4 prereg):
- For each transfer-arm run, find the maximum `transferable_drift_max`
  across all telemetry checkpoints with `episode <= 100`.
- If EITHER seed shows `max_drift > 0.50` at any such checkpoint, ABORT
  the pilot launch (asymmetric loss: a single broken seed is sufficient
  evidence the LR warmup is not doing its job; the cost of a false
  abort is ~3 GPU-h of rework, the cost of a false launch is ~20
  GPU-h burned + weeks of debugging).
- Otherwise: PROCEED to pilot #2.

Additional sanity checks (v5):
- Telemetry must be non-empty for every transfer-arm run with a
  crystallized source. An empty telemetry list signals the snapshot
  block at `_train_to_step_budget` failed silently (e.g. try_transfer
  returned None and we should have produced an empty list, but ALSO
  the source must not have crystallized — distinguish those cases).
- `kl_probe_error` must be `None` (the success state) on every
  checkpoint, OR equal to the explicit `"buffer empty (num_episodes=0)"`
  string on the very first checkpoints before the buffer fills. Any
  other repr (i.e. a real exception) is a probe regression and ABORT.

Exit code: 0 on PROCEED, 1 on ABORT.

Usage:
    python -m scripts.smoke_verdict smoke_bug_e_v4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Import the buffer-empty sentinel from the producer module rather than
# duplicating the literal — v5 architecture review caught the brittle
# silent-drift coupling: if the producer changed the string without
# updating the consumer, the analyzer would treat real probe errors as
# the expected buffer-empty case (false PROCEED).
from scripts.pilot_run import TELEMETRY_BUFFER_EMPTY_SENTINEL

# Prereg-committed thresholds (v3.4 amendment "Bug E v2" / v4):
DRIFT_ABORT_THRESHOLD = 0.50  # ||Δθ|| > 50% of initial norm
DRIFT_ABORT_EPISODE_GATE = 100  # checked at episode <= this

# Re-exported for tests that originally imported from this module
# (kept to avoid churn; canonical source is scripts.pilot_run).
EXPECTED_BUFFER_EMPTY_MSG = TELEMETRY_BUFFER_EMPTY_SENTINEL


@dataclass
class SeedVerdict:
    """Per-seed evaluation of one transfer-arm run."""
    pair_alias: str
    seed: int
    arm: str
    source_crystallized: bool | None
    n_telemetry: int
    n_checkpoints_at_or_below_ep_gate: int
    max_drift_in_window: float | None
    drift_abort_triggered: bool
    probe_error_count: int  # non-None, non-buffer-empty error count
    notes: list[str]

    @property
    def pass_(self) -> bool:
        # A seed PASSES the smoke gate iff:
        #   - its transfer arm has crystallized source AND telemetry
        #     was actually emitted (an empty telemetry list with a
        #     crystallized source means the snapshot block regressed)
        #   - max drift in window <= threshold (or was never measured
        #     because no checkpoints landed in window — caller decides)
        #   - no real probe exceptions
        if self.drift_abort_triggered:
            return False
        if self.probe_error_count > 0:
            return False
        if self.source_crystallized is False:
            # Source didn't crystallize → transfer fell back to scratch
            # → telemetry never produced → can't validate the gate.
            # This is NOT a smoke pass.
            return False
        if self.n_telemetry == 0:
            # Source crystallized (or status unknown) but telemetry is
            # empty — the _train_to_step_budget snapshot block silently
            # failed. Cannot validate the gate; fail closed.
            return False
        return True


@dataclass
class SmokeVerdict:
    """Aggregated smoke pre-check verdict."""
    proceed: bool
    seed_verdicts: list[SeedVerdict]
    summary_lines: list[str]


def _evaluate_run(run: dict) -> SeedVerdict:
    """Apply the prereg abort rule to one transfer-arm run."""
    pair_alias = run.get("pair_alias", "<unknown>")
    seed = run.get("seed", -1)
    arm = run.get("arm", "<unknown>")
    source_crystallized = run.get("source_crystallized")
    telemetry = run.get("telemetry", []) or []
    notes: list[str] = []

    # Filter to checkpoints within the abort window.
    in_window = [
        rec for rec in telemetry
        if isinstance(rec.get("episode"), int)
        and rec["episode"] <= DRIFT_ABORT_EPISODE_GATE
    ]

    if not telemetry:
        if source_crystallized is False:
            notes.append("no telemetry: source didn't crystallize "
                         "(transfer fell back to scratch)")
        else:
            notes.append("no telemetry emitted despite crystallized "
                         "source — telemetry pipeline regression "
                         "(_train_to_step_budget snapshot block?)")

    max_drift_in_window: float | None = None
    drift_abort_triggered = False
    if in_window:
        drifts = [
            float(rec["transferable_drift_max"])
            for rec in in_window
            if rec.get("transferable_drift_max") is not None
        ]
        if drifts:
            max_drift_in_window = max(drifts)
            if max_drift_in_window > DRIFT_ABORT_THRESHOLD:
                drift_abort_triggered = True
                notes.append(
                    f"DRIFT ABORT: max ||Δθ||/||θ_0|| = "
                    f"{max_drift_in_window:.3f} > "
                    f"{DRIFT_ABORT_THRESHOLD} threshold within "
                    f"episode <= {DRIFT_ABORT_EPISODE_GATE}")

    # Count "real" probe errors — anything other than None or the
    # explicit buffer-empty sentinel. The empty-buffer string is
    # expected on the first 1-2 checkpoints before the replay buffer
    # has its first episode; any other repr is a probe regression.
    probe_error_count = 0
    for rec in telemetry:
        err = rec.get("kl_probe_error")
        if err is None:
            continue
        if err == EXPECTED_BUFFER_EMPTY_MSG:
            continue
        probe_error_count += 1
        notes.append(f"probe error at ep={rec.get('episode')}: {err!r}")

    return SeedVerdict(
        pair_alias=pair_alias,
        seed=seed,
        arm=arm,
        source_crystallized=source_crystallized,
        n_telemetry=len(telemetry),
        n_checkpoints_at_or_below_ep_gate=len(in_window),
        max_drift_in_window=max_drift_in_window,
        drift_abort_triggered=drift_abort_triggered,
        probe_error_count=probe_error_count,
        notes=notes,
    )


def evaluate(payload: dict) -> SmokeVerdict:
    """Apply the smoke gate to a pilot_results.json payload.

    Schema notes (v5 architecture review): the current `pilot_run.py`
    always writes a flat top-level ``"runs"`` list (see
    ``pilot_run._flush``). The grouped ``"pairs"`` fallback below is
    NOT a path that any current writer produces — it's kept as a
    backward-compat shim for hand-edited test fixtures and for the
    case where a future analyzer might emit per-pair grouping. The
    ``test_grouped_pairs_schema`` test pins this shim's contract so
    a future refactor that "cleans up" the fallback breaks a test
    instead of silently dropping support.
    """
    runs = payload.get("runs")
    if not runs:
        # Backward-compat / forward-compat shim — see docstring.
        runs = []
        for pair_block in payload.get("pairs", []):
            for arm_key in ("source_runs", "scratch_runs",
                            "transfer_runs"):
                runs.extend(pair_block.get(arm_key, []))

    transfer_runs = [r for r in runs if r.get("arm") == "transfer"]

    seed_verdicts = [_evaluate_run(r) for r in transfer_runs]

    summary_lines: list[str] = []
    summary_lines.append(
        f"Smoke verdict on {len(transfer_runs)} transfer-arm run(s):")
    for v in seed_verdicts:
        flag = "PASS" if v.pass_ else "FAIL"
        drift_str = (f"{v.max_drift_in_window:.3f}"
                     if v.max_drift_in_window is not None
                     else "n/a")
        summary_lines.append(
            f"  [{flag}] {v.pair_alias} seed={v.seed} "
            f"crystallized={v.source_crystallized} "
            f"n_tele={v.n_telemetry} "
            f"max_drift_in_window={drift_str} "
            f"probe_errors={v.probe_error_count}")
        for n in v.notes:
            summary_lines.append(f"         - {n}")

    if not seed_verdicts:
        proceed = False
        summary_lines.append("  ABORT: no transfer-arm runs found.")
    else:
        proceed = all(v.pass_ for v in seed_verdicts)
        summary_lines.append(
            f"  ==> {'PROCEED to pilot #2' if proceed else 'ABORT pilot launch'}")
        if not proceed:
            summary_lines.append(
                "      Per prereg v3.4 amendment 'Bug E v4', EITHER "
                "seed triggering ||Δθ|| > 50% within ep<=100 holds "
                "the pilot launch.")

    return SmokeVerdict(
        proceed=proceed,
        seed_verdicts=seed_verdicts,
        summary_lines=summary_lines,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply smoke pre-check abort rule to a pilot_run JSON")
    parser.add_argument("results_json", type=Path,
                        help="Path to smoke pilot_results.json")
    args = parser.parse_args(argv)

    if not args.results_json.exists():
        print(f"[smoke_verdict] file not found: {args.results_json}",
              file=sys.stderr)
        return 2

    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    verdict = evaluate(payload)
    for line in verdict.summary_lines:
        print(line)
    return 0 if verdict.proceed else 1


if __name__ == "__main__":
    sys.exit(main())
