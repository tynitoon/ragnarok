"""Compute-budget extrapolation (preregistration §12.5).

Takes `compute_budget.json` (measured per-env throughput on actual
hardware) and projects total wall-clock for the full H1 benchmark:

    Primary:   5 baselines x 20 seeds x 500k steps on MCC
    Secondary: 5 baselines x 10 seeds x 500k steps on Acrobot + DMC-csu
    Ablations: 9 ablations x 5 seeds x 500k steps on MCC only

If projection exceeds 28 days at 90% GPU duty cycle, prints which
§12.5 cut order tier is needed.

Usage:
    python -m scripts.compute_budget_extrapolation
    python -m scripts.compute_budget_extrapolation --budget custom_budget.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


# Canonical spec per preregistration §2, §3, §7, §12
PRIMARY_ENV = "mountaincar-continuous"    # MCC per thresholds.json
SECONDARY_ENVS = ["acrobot", "cartpole-swingup"]  # DMC mapped
N_BASELINES = 5                    # §3: SB3 PPO/SAC, DreamerV3, ContDreamer, Ragnarok-scratch, Ragnarok-transfer
PRIMARY_SEEDS = 20
SECONDARY_SEEDS = 10
ABLATION_SEEDS = 5
STEPS_PER_RUN = 500_000            # truncation_horizon_env_steps

# Ablation roster per thresholds.json. A5 is a 3-point dim sweep — each sub-run
# is independent and all are charged against compute. A5b is deferred to a
# separate preregistration (H2-confirmatory), excluded here.
# Total ablation runs on MCC: (single-row ablations × 1 + A5 × 3) × ABLATION_SEEDS
ABLATION_ROSTER = {
    "A1": 1,  # Frozen trunk (single configuration)
    "A2": 1,  # Randomly-initialized trunk
    "A3": 1,  # Source-task reward shuffle
    "A4": 1,  # ObsEncoder-only transfer
    "A5": 3,  # Latent-dim sweep {64, 160, 512} — three sub-runs per seed
    "A6": 1,  # No-retrieval (fixed skill)
    "A7": 1,  # Random-retrieval control
    "A8": 1,  # Equal-FLOP-source Continual-Dreamer
    "A9": 1,  # Shuffled-dynamics RSSM
}
TOTAL_ABLATION_RUNS_PER_SEED = sum(ABLATION_ROSTER.values())  # = 11

# The §12.5 cut#2 expanded-panel roster: "A1+A2+A8+A9 mechanism panel" per
# preregistration §12.5. 4 single-config ablations × 5 seeds = 20 runs.
EXPANDED_PANEL_ABLATIONS = ["A1", "A2", "A8", "A9"]
EXPANDED_PANEL_RUNS_PER_SEED = sum(ABLATION_ROSTER[a] for a in EXPANDED_PANEL_ABLATIONS)  # = 4

# Compute envelope
WALL_BUDGET_DAYS = 28
DUTY_CYCLE = 0.90
# WALL_BUDGET_SECONDS already bakes in the 0.9 duty-cycle factor; comparing
# raw compute-seconds against it is the correct gate (no double multiplication).
WALL_BUDGET_SECONDS = WALL_BUDGET_DAYS * 24 * 3600 * DUTY_CYCLE  # ≈ 2.18M s = 605 GPU-hr

# Additional fixed-cost items from §12.5
CONTINUAL_DREAMER_SOURCE_HR = 75   # Phase 4, not Phase 5 -- separate line item


@dataclass
class LegBudget:
    """One benchmark leg (e.g. 'Primary (MCC)', 'Ablation A9 on MCC')."""
    name: str
    env: str                              # human-readable, may include tags
    env_key: str                          # canonical env name, used for matching
    n_runs: int
    steps_per_run: int
    throughput_steps_per_sec: float

    @property
    def total_steps(self) -> int:
        return self.n_runs * self.steps_per_run

    @property
    def wall_seconds(self) -> float:
        return self.total_steps / max(self.throughput_steps_per_sec, 1e-6)

    @property
    def wall_hours(self) -> float:
        return self.wall_seconds / 3600

    @property
    def wall_days(self) -> float:
        return self.wall_hours / 24


def _lookup_throughput(summary: dict, env: str, fallback: float) -> tuple[float, str]:
    """Look up mean steps/sec for an env; return (value, source_tag).

    Uses the measured mean if present; otherwise the slowest-measured env as
    the best available proxy. NOTE: slowest-of-measured is "conservative" only
    in the sense of being worse than the non-slowest measured envs — the
    *truly missing* env (e.g. MountainCarContinuous before the full smoke has
    run it) may in principle be slower than anything measured. When relying
    on the fallback, the caller should re-run extrapolation after the smoke
    measures the missing env directly. The returned tag makes the substitution
    visible in every printed line.
    """
    if env in summary:
        return summary[env]["mean_steps_per_sec"], "measured"
    if summary:
        slowest = min(summary.values(), key=lambda s: s["mean_steps_per_sec"])
        slowest_env = next(k for k, v in summary.items()
                           if v["mean_steps_per_sec"] == slowest["mean_steps_per_sec"])
        return slowest["mean_steps_per_sec"], f"fallback=slowest-measured({slowest_env})"
    return fallback, "fallback=hardcoded"


def extrapolate(budget_path: Path) -> dict:
    data = json.loads(budget_path.read_text())
    summary = data.get("summary", {})

    # Build the leg list
    legs: list[LegBudget] = []

    # Primary endpoint: MCC, 5 baselines x 20 seeds
    prim_tp, prim_src = _lookup_throughput(
        summary, PRIMARY_ENV, fallback=100.0)
    legs.append(LegBudget(
        name="Primary (MCC) -- 5 baselines x 20 seeds",
        env=f"{PRIMARY_ENV} [{prim_src}]",
        env_key=PRIMARY_ENV,
        n_runs=N_BASELINES * PRIMARY_SEEDS,
        steps_per_run=STEPS_PER_RUN,
        throughput_steps_per_sec=prim_tp,
    ))

    # Secondary endpoints: 2 envs x 5 baselines x 10 seeds
    for env in SECONDARY_ENVS:
        tp, src = _lookup_throughput(summary, env, fallback=100.0)
        legs.append(LegBudget(
            name=f"Secondary ({env}) -- 5 baselines x 10 seeds",
            env=f"{env} [{src}]",
            env_key=env,
            n_runs=N_BASELINES * SECONDARY_SEEDS,
            steps_per_run=STEPS_PER_RUN,
            throughput_steps_per_sec=tp,
        ))

    # Ablations on primary endpoint only. The §7 A5 row is a 3-point
    # latent-dim sweep: the full cost is TOTAL_ABLATION_RUNS_PER_SEED × seeds,
    # not N_ABLATIONS × seeds. Charging 9 × 5 = 45 underbudgeted by 10 runs;
    # correct total is 11 × 5 = 55.
    legs.append(LegBudget(
        name=(f"Ablations A1-A9 on MCC -- "
              f"{TOTAL_ABLATION_RUNS_PER_SEED} x {ABLATION_SEEDS} seeds "
              f"(A5 sweep counted 3x)"),
        env=f"{PRIMARY_ENV} [{prim_src}]",
        env_key=PRIMARY_ENV,
        n_runs=TOTAL_ABLATION_RUNS_PER_SEED * ABLATION_SEEDS,
        steps_per_run=STEPS_PER_RUN,
        throughput_steps_per_sec=prim_tp,
    ))

    total_seconds = sum(l.wall_seconds for l in legs)
    # Continual-Dreamer source-training is Phase 4, not Phase 5 -- but we
    # still report it for transparency.
    phase4_source_seconds = CONTINUAL_DREAMER_SOURCE_HR * 3600

    return {
        "hardware": data.get("hardware", {}),
        "legs": [
            {
                "name": l.name,
                "env": l.env,
                "env_key": l.env_key,
                "n_runs": l.n_runs,
                "steps_per_run": l.steps_per_run,
                "total_steps": l.total_steps,
                "throughput_steps_per_sec": l.throughput_steps_per_sec,
                "wall_seconds": l.wall_seconds,
                "wall_hours": l.wall_hours,
                "wall_days": l.wall_days,
            } for l in legs
        ],
        "phase5_total_seconds": total_seconds,
        "phase5_total_hours": total_seconds / 3600,
        "phase5_total_days_100pct_duty": total_seconds / 86400,
        # clock-days when using 90% duty: at 90% duty the machine produces
        # 86400 * 0.9 compute-seconds per calendar day, so projecting
        # total_seconds compute onto a 90%-duty calendar takes
        # total_seconds / (86400 * 0.9) calendar days.
        "phase5_calendar_days_at_90pct_duty": total_seconds / (86400 * DUTY_CYCLE),
        "phase4_continual_dreamer_source_hours": CONTINUAL_DREAMER_SOURCE_HR,
        "wall_budget_days_90pct": WALL_BUDGET_DAYS,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "fits_budget": total_seconds <= WALL_BUDGET_SECONDS,
        "margin_seconds": WALL_BUDGET_SECONDS - total_seconds,
        "margin_hours": (WALL_BUDGET_SECONDS - total_seconds) / 3600,
        "cut_tier_needed": _recommend_cut(legs, total_seconds),
    }


def _recommend_cut(legs: list[LegBudget], total_s: float) -> str | None:
    """§12.5 pre-declared cut order if budget exceeded.

    Tiers tried in order, first-fit:
      cut#1: drop DMC secondary endpoint (cartpole-swingup leg).
             Prereg v3 §12.5 cut#1 text reads "drop DMC ablations", but
             prereg v3 already consolidates all ablations onto MCC, so the
             operational cut#1 is "drop DMC secondary".
      cut#2: drop ALL secondaries + add the expanded A1+A2+A8+A9 mechanism
             panel on MCC (non-negotiable companion per prereg §12.5).
      cut#3: cut#2 + N=20→15 on primary (power drops to ~0.62).
    """
    if total_s <= WALL_BUDGET_SECONDS:
        return None

    # Resolve key legs up front via env_key / name prefix — invariant of
    # human-readable env label formatting.
    primary_leg = next((l for l in legs if l.name.startswith("Primary")), None)
    assert primary_leg is not None, (
        "extrapolate() must always produce a Primary leg; legs list is corrupt"
    )

    # DMC secondary leg must exist structurally (extrapolate() constructs it
    # unconditionally from SECONDARY_ENVS). If it does not, the caller has
    # mutated the leg list — fail loud rather than silently skipping cut#1.
    dmc_secondary = next(
        (l for l in legs if l.env_key == "cartpole-swingup"), None)
    assert dmc_secondary is not None, (
        "DMC cartpole-swingup leg missing from legs list; cannot evaluate "
        "§12.5 cut#1. Check that SECONDARY_ENVS still includes it."
    )

    # Cut 1: drop DMC secondary.
    after_cut1 = total_s - dmc_secondary.wall_seconds
    if after_cut1 <= WALL_BUDGET_SECONDS:
        return "cut#1 -- drop DMC secondary endpoint"

    # Cut 2: drop ALL secondaries + add expanded panel.
    secondary_seconds = sum(l.wall_seconds for l in legs
                            if l.name.startswith("Secondary"))
    # expanded panel: A1+A2+A8+A9 × 5 seeds on MCC (4 × 5 = 20 runs)
    expanded_panel_runs = EXPANDED_PANEL_RUNS_PER_SEED * ABLATION_SEEDS
    tp_mcc = primary_leg.throughput_steps_per_sec
    expanded_panel_seconds = expanded_panel_runs * STEPS_PER_RUN / tp_mcc
    after_cut2 = total_s - secondary_seconds + expanded_panel_seconds
    if after_cut2 <= WALL_BUDGET_SECONDS:
        return "cut#2 -- MCC-only paper + expanded A1+A2+A8+A9 mechanism panel"

    # Cut 3: cut#2 + N=20->15 on primary.
    primary_saving = primary_leg.wall_seconds * (1 - 15.0/20.0)
    after_cut3 = after_cut2 - primary_saving
    if after_cut3 <= WALL_BUDGET_SECONDS:
        return "cut#3 -- MCC-only + N=20->15 (power 0.62)"

    return "cut-stack insufficient -- methodology amendment required"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path,
                        default=Path("compute_budget.json"),
                        help="compute_budget.json path")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Optional JSON output path (writes extrapolation)")
    args = parser.parse_args(argv)

    if not args.budget.exists():
        print(f"ERROR: {args.budget} does not exist. Run "
              f"`python -m scripts.smoke_benchmark` first.")
        return 1

    result = extrapolate(args.budget)

    # Pretty print
    print(f"Hardware: {result['hardware']}")
    print(f"\nBenchmark legs:")
    for leg in result["legs"]:
        print(f"  {leg['name']:60s}")
        print(f"    env={leg['env']}")
        print(f"    runs={leg['n_runs']:4d} x {leg['steps_per_run']:,} steps = "
              f"{leg['total_steps']:,} total steps")
        print(f"    @ {leg['throughput_steps_per_sec']:6.1f} steps/s  "
              f"-> {leg['wall_hours']:6.1f} GPU-hr  "
              f"= {leg['wall_days']:5.2f} days")

    print(f"\n--- Phase 5 total ---")
    print(f"  {result['phase5_total_hours']:6.1f} GPU-hr  "
          f"= {result['phase5_total_days_100pct_duty']:5.2f} clock-days "
          f"(100% duty) = {result['phase5_calendar_days_at_90pct_duty']:5.2f} "
          f"clock-days (90% duty)")
    print(f"\n  Wall budget (preregistered): {WALL_BUDGET_DAYS} days x "
          f"90% duty = {WALL_BUDGET_SECONDS/3600:.1f} GPU-hr")
    print(f"  Phase 4 Continual-Dreamer source training (separate): "
          f"{result['phase4_continual_dreamer_source_hours']} GPU-hr")

    if result["fits_budget"]:
        margin = result["margin_hours"]
        print(f"\n  PASS FITS -- margin {margin:.1f} GPU-hr "
              f"({margin/24:.2f} clock-days)")
    else:
        overshoot = -result["margin_hours"]
        print(f"\n  FAIL OVERSHOOTS by {overshoot:.1f} GPU-hr "
              f"({overshoot/24:.2f} clock-days)")
        print(f"  Recommended: {result['cut_tier_needed']}")

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
