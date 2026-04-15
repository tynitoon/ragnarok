"""Phase 3 pilot analyzer (preregistration §8 pass-criteria evaluator).

Consumes `pilot_results.json` produced by `scripts/pilot_run.py` and
computes, per pair:

  - Kaplan-Meier survival curves for {scratch, transfer} arms (duration =
    steps_to_mastery; event = not censored)
  - RMST (restricted mean survival time) for each arm, with τ =
    max_env_steps from the pilot config
  - RMST ratio: scratch_RMST / transfer_RMST  (ratio > 1 means transfer is
    faster to mastery — which is what we want)
  - One-sided log-rank test (H_A: transfer survives shorter than scratch;
    equivalently, transfer reaches mastery faster)
  - Mechanism check: acting_policy_mode == "latent" for all transfer runs
    whose source and target have different (obs_dim or action_dim)

Then renders the §8 pass verdict:
  - Primary pair (cartpole_mcc): RMST ratio >= 1.3x AND log-rank p < 0.10
  - ≥1 secondary pair: RMST ratio >= 1.3x directionally (no p threshold)
  - No pair has RMST ratio < 0.9x (anti-transfer)
  - Mechanism check passes for every pair whose pilot required latent transfer

Usage:
    python -m scripts.pilot_analysis pilot_results.json
    python -m scripts.pilot_analysis pilot_results.json --json-output verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Force UTF-8 stdio (see pilot_run.py for rationale). The verdict renderer
# uses §, τ, ≥, — glyphs that match the preregistration language; without
# this, render_text() would crash on Windows cp1252 the moment a verdict
# line contains one of them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── §8 pass-criteria thresholds (pinned) ───────────────────────────

PRIMARY_RMST_RATIO_MIN = 1.3
PRIMARY_LOGRANK_P_MAX = 0.10
SECONDARY_RMST_RATIO_MIN = 1.3
ANTI_TRANSFER_RATIO_MAX = 0.9  # any pair with ratio < this fails

PRIMARY_ALIAS = "cartpole_mcc"

# INCONCLUSIVE regime (devil's-advocate review, Phase 3 pre-commit).
# When the primary pair's scratch OR transfer arm has ≥80% runs censored
# at τ, the sample is too thin on observed events to pick a direction or
# a ratio with any confidence. In that regime we report INCONCLUSIVE
# (distinct from FAIL) so the pilot does NOT trigger Plan B — Plan B is
# for a measured null/anti-transfer, not for "we didn't observe enough
# events to tell." Caller should expand seeds before re-running analysis.
HIGH_CENSORING_THRESHOLD = 0.8


# ── Data model ──────────────────────────────────────────────────────

@dataclass
class ArmSurvival:
    arm: str                    # "scratch" or "transfer"
    n: int                      # total runs in this arm
    n_events: int               # runs that reached mastery (observed)
    n_censored: int             # runs censored at τ
    rmst: float                 # restricted mean survival time
    rmst_variance: float        # variance from lifelines (0 if not returned)
    median_survival: float      # median from KMF
    durations: list[int]        # steps_to_mastery or τ, per run
    events: list[int]           # 1 if observed, 0 if censored


@dataclass
class PairVerdict:
    alias: str
    role: str                   # "primary" | "secondary"
    scratch: ArmSurvival
    transfer: ArmSurvival
    rmst_ratio: float           # scratch / transfer  (ratio > 1 = transfer faster)
    logrank_p_value: float
    # v3.5 — permutation-based one-sided p (exact under exchangeability).
    # Reported alongside the asymptotic log-rank as a small-sample
    # robustness check. At N=5 per arm, the lifelines chi-sq asymptotic
    # is known to drift; the permutation p from 10k label shuffles is
    # exact. §8 headline p is still the asymptotic value — this field
    # is a robustness-of-conclusion probe, not a replacement.
    logrank_permutation_p_value: float
    anti_transfer: bool         # ratio < 0.9
    mechanism_check_passed: bool
    mechanism_details: str
    pass_primary_criterion: bool
    pass_secondary_criterion: bool
    tau: int                    # truncation horizon (env-steps)
    # v3.5 descriptive secondaries (non-gating; see §4 + §10 B0).
    # Populated only from runs that include an `eval_curve` field;
    # paper reports these alongside RMST with bootstrap 95 % CIs.
    scratch_descriptives: "ArmDescriptives | None" = None
    transfer_descriptives: "ArmDescriptives | None" = None


@dataclass
class PilotVerdict:
    overall_pass: bool          # True only for a clean PASS
    pair_verdicts: list[PairVerdict]
    failures: list[str]         # human-readable list of failed criteria
    inconclusive: bool = False  # True → primary has high-censoring regime;
                                # do NOT invoke Plan B, expand sample first
    inconclusive_reason: str = ""


# ── Helpers ─────────────────────────────────────────────────────────

def _runs_for(runs: list[dict], alias: str, arm: str) -> list[dict]:
    return [r for r in runs
            if r.get("pair_alias") == alias and r.get("arm") == arm]


def _extract_duration_event(run: dict, tau: int) -> tuple[int, int]:
    """Return (duration, event_observed) for lifelines.

    event_observed = 1 iff run reached mastery; 0 iff censored at τ.
    """
    stm = run.get("steps_to_mastery")
    if stm is None or run.get("censored", stm is None):
        return (tau, 0)
    return (int(stm), 1)


def _fit_arm(runs: list[dict], arm: str, tau: int) -> ArmSurvival:
    """Fit a KMF to one arm and compute RMST(τ)."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time

    durations = []
    events = []
    for r in runs:
        d, e = _extract_duration_event(r, tau)
        durations.append(d)
        events.append(e)

    if not durations:
        return ArmSurvival(
            arm=arm, n=0, n_events=0, n_censored=0,
            rmst=float("nan"), rmst_variance=float("nan"),
            median_survival=float("nan"),
            durations=[], events=[],
        )

    kmf = KaplanMeierFitter(label=arm)
    kmf.fit(durations, event_observed=events)
    rmst, var = restricted_mean_survival_time(kmf, t=tau, return_variance=True)

    return ArmSurvival(
        arm=arm,
        n=len(durations),
        n_events=int(sum(events)),
        n_censored=int(len(events) - sum(events)),
        rmst=float(rmst),
        rmst_variance=float(var),
        median_survival=float(kmf.median_survival_time_)
        if np.isfinite(kmf.median_survival_time_) else float("nan"),
        durations=durations,
        events=events,
    )


def _logrank_signed_direction(scratch: ArmSurvival,
                              transfer: ArmSurvival) -> float:
    """Return sum_t (O_scratch - E_scratch) at each distinct event time.

    This is the signed form of the log-rank statistic's numerator. Its sign
    carries the hazard-direction information that lifelines' chi-sq
    `test_statistic` squares away. Positive → scratch has MORE observed
    events than expected under H_0 of equal hazards → scratch reaches
    mastery FASTER than transfer → wrong direction for H_A.
    Negative → transfer faster → H_A direction.
    """
    # Collapse to per-time-point tallies across the pooled sample.
    all_durs = list(scratch.durations) + list(transfer.durations)
    # For at-risk counts we need EVERY subject, censored or not, sorted.
    # Event times are only those where an observed event happens.
    event_times = sorted({
        d for d, e in zip(all_durs,
                          list(scratch.events) + list(transfer.events))
        if e == 1
    })
    if not event_times:
        return 0.0

    oe = 0.0
    for t in event_times:
        n_s = sum(1 for d in scratch.durations if d >= t)
        n_t = sum(1 for d in transfer.durations if d >= t)
        n = n_s + n_t
        if n == 0:
            continue
        d_s = sum(1 for d, e in zip(scratch.durations, scratch.events)
                  if d == t and e == 1)
        d_t = sum(1 for d, e in zip(transfer.durations, transfer.events)
                  if d == t and e == 1)
        d = d_s + d_t
        e_s = d * (n_s / n)
        oe += (d_s - e_s)
    return oe


def _logrank_one_sided(scratch: ArmSurvival, transfer: ArmSurvival,
                       tau: int) -> float:
    """One-sided log-rank p-value for H_A: transfer < scratch in survival time.

    lifelines' `logrank_test` returns a two-sided p (from a chi-sq statistic,
    which is sign-stripped). We halve it when the observed direction matches
    H_A (transfer reaches mastery faster than scratch); otherwise we report
    1 - p/2 — the directional test cannot reject when the effect runs the
    wrong way.

    **Direction from the signed log-rank numerator, not RMST.**
    Devil's-advocate review (Phase 3 pre-commit) flagged that using RMST
    comparison for direction breaks under crossing hazards: curves can cross
    so that transfer wins on RMST over [0, τ] yet loses on hazard direction
    (or vice versa). The log-rank test itself is a hazard-based test, so the
    directionally-matched sign comes from its own numerator O - E, not from
    RMST. We compute the signed sum manually since lifelines exposes only
    the chi-sq statistic.
    """
    from lifelines.statistics import logrank_test

    if scratch.n == 0 or transfer.n == 0:
        return float("nan")

    res = logrank_test(
        durations_A=scratch.durations,
        durations_B=transfer.durations,
        event_observed_A=scratch.events,
        event_observed_B=transfer.events,
        t_0=tau,
    )
    p_two_sided = float(res.p_value)

    # Low-event-regime guard (devil's-advocate review, Phase 3 pre-commit).
    # When both arms mostly censor, O-E collapses to near-zero and the
    # direction calculation is dominated by tied events at the truncation
    # horizon. At N=5 per arm on MCC, regimes with <2 observed events per
    # arm are routine under the prereg's own 40% censoring model — and
    # picking a direction from 0 or 1 observed event is essentially noise.
    # In that regime, report the non-directional two-sided p (i.e. a "can't
    # decide direction" outcome rather than a false confident flip). The
    # primary-pair gate at §8 (p < 0.10 one-sided) therefore can't trigger
    # off tied-event coincidence.
    oe = _logrank_signed_direction(scratch, transfer)
    if abs(oe) < 1e-9 or scratch.n_events < 2 or transfer.n_events < 2:
        return p_two_sided  # non-directional fallback; caller treats as no H_A support

    # H_A direction: transfer faster ↔ scratch has FEWER observed events
    # than expected ↔ signed (O_scratch - E_scratch) < 0.
    if oe < 0:
        return p_two_sided / 2.0
    return 1.0 - (p_two_sided / 2.0)


# ── v3.5 permutation test (small-N robustness check) ───────────────
#
# The lifelines log-rank test returns a p-value from a chi-sq asymptotic.
# That asymptotic is known to drift at small N with heavy censoring: the
# chi-sq reference distribution assumes the O-E numerator is approximately
# normal under H_0, which requires enough event times to average out. At
# N=5 per arm on MCC with ~30-40% censoring, we routinely see 3-4 observed
# events per arm — well below the regime where the asymptotic is
# defensible.
#
# The permutation test is *exact* under exchangeability (the null
# hypothesis): shuffle the arm labels across the pooled sample while
# preserving the per-arm sample sizes, recompute the signed O-E on each
# shuffle, and read p off the empirical tail. No distributional assumption
# beyond "arms are exchangeable under H_0", which is what the log-rank
# asserts anyway.
#
# We use the same signed O-E numerator as the asymptotic path, so the two
# p-values answer literally the same question — the difference is only
# how the null distribution is obtained.
#
# Cost: O(n_shuffles × n_event_times × n) per pair. At pilot scale (n=10,
# event times ≤10), 10k shuffles costs ~100 ms. No vectorization needed.

PERMUTATION_N_SHUFFLES = 10_000
PERMUTATION_RNG_SEED = 20260415  # pinned for reproducibility; see §5


def _compute_signed_oe(durations_s: list[int], events_s: list[int],
                       durations_t: list[int],
                       events_t: list[int]) -> float:
    """Signed O-E numerator. Pulled out of `_logrank_signed_direction`
    so the permutation loop can call it on arbitrary arm assignments
    without round-tripping through `ArmSurvival`.
    """
    event_times = sorted({
        d for d, e in zip(durations_s + durations_t,
                          events_s + events_t)
        if e == 1
    })
    if not event_times:
        return 0.0

    oe = 0.0
    for t in event_times:
        n_s = sum(1 for d in durations_s if d >= t)
        n_t = sum(1 for d in durations_t if d >= t)
        n = n_s + n_t
        if n == 0:
            continue
        d_s = sum(1 for d, e in zip(durations_s, events_s)
                  if d == t and e == 1)
        d_t = sum(1 for d, e in zip(durations_t, events_t)
                  if d == t and e == 1)
        d = d_s + d_t
        e_s = d * (n_s / n)
        oe += (d_s - e_s)
    return oe


def _logrank_permutation_one_sided(
    scratch: ArmSurvival,
    transfer: ArmSurvival,
    n_shuffles: int = PERMUTATION_N_SHUFFLES,
    rng_seed: int = PERMUTATION_RNG_SEED,
) -> float:
    """One-sided permutation p-value for H_A: transfer reaches mastery
    faster than scratch (signed O-E observed < 0 direction).

    Returns a Monte Carlo estimate of
        P_{H_0}(signed_OE_shuffled ≤ signed_OE_observed)
    using `n_shuffles` random label permutations that preserve the
    per-arm sample sizes. The "add 1" numerator/denominator correction
    guarantees p > 0 even if no shuffle reaches the tail (standard for
    permutation p).

    Direction convention matches `_logrank_one_sided`: lower one-sided
    p means *more* evidence for transfer-faster. If the observed O-E is
    in the H_A-wrong direction (> 0), the one-sided p is ≥ 0.5 (the
    permutation distribution cannot support a directional alternative
    when the point estimate itself is on the wrong side).
    """
    if scratch.n == 0 or transfer.n == 0:
        return float("nan")
    n_s = scratch.n
    n_t = transfer.n
    pooled_durs = list(scratch.durations) + list(transfer.durations)
    pooled_events = list(scratch.events) + list(transfer.events)
    n = n_s + n_t

    # Observed statistic.
    observed_oe = _compute_signed_oe(
        scratch.durations, scratch.events,
        transfer.durations, transfer.events,
    )
    # If there are no events at all on either arm, the statistic is
    # identically zero under all permutations — punt to NaN (no
    # information). This is consistent with the asymptotic fallback
    # that already reports the two-sided p in this regime.
    if _no_events_either_arm(scratch, transfer):
        return float("nan")

    rng = np.random.default_rng(rng_seed)
    # Count shuffles at least as extreme as observed in the H_A
    # direction (observed < 0 → shuffled ≤ observed).
    # We always count using the same ≤ direction; if observed > 0
    # (wrong direction), the tail count trends toward the upper half
    # of the null and p ≥ 0.5 naturally.
    count_le = 0
    indices = np.arange(n)
    for _ in range(n_shuffles):
        rng.shuffle(indices)
        s_idx = indices[:n_s]
        t_idx = indices[n_s:]
        durs_s = [pooled_durs[i] for i in s_idx]
        evts_s = [pooled_events[i] for i in s_idx]
        durs_t = [pooled_durs[i] for i in t_idx]
        evts_t = [pooled_events[i] for i in t_idx]
        shuffled_oe = _compute_signed_oe(durs_s, evts_s, durs_t, evts_t)
        if shuffled_oe <= observed_oe:
            count_le += 1

    # Standard "add 1" permutation-p correction: ensures p > 0 and
    # matches the one-sided testing convention when no shuffle is as
    # extreme as the observed.
    return float((count_le + 1) / (n_shuffles + 1))


def _no_events_either_arm(scratch: ArmSurvival,
                          transfer: ArmSurvival) -> bool:
    return (scratch.n_events == 0 and transfer.n_events == 0)


# ── v3.5 descriptive secondaries (early-step return + AUC) ─────────
#
# The prereg §4 headline metric remains samples-to-mastery RMST. But the
# mastery threshold on MCC (~90/100) is plateau territory — the real
# transfer signal is expected in the first 2-10k env-steps when the
# transferred prior is still load-bearing. §8 does NOT gate on these
# values (they are descriptive, not inferential); but the paper WILL
# report them with bootstrap 95 % CIs in the same panel as the RMST
# headline, and §10 Plan B0 uses them as companion analyses.
#
# Contract:
#   - return_at_step(eval_curve, step)  → float
#     Linear interp; pre-first-eval queries anchored at (0, 0.0)
#     (conservative — assumes random-policy-bad baseline).
#   - auc_return(eval_curve, lo, hi)    → float
#     Trapezoidal AUC over [lo, hi] env-steps; same interp convention.
#     Normalized by (hi - lo) so the value has units of "mean return".
#   - bootstrap_ci_mean(values, n, seed) → (mean, lo95, hi95)
#     Percentile bootstrap over seed-level values.
#
# The 0-anchor convention is the load-bearing choice: all 3 envs in
# scope (MCC, Acrobot, DMC-cartpole-swingup) start at near-0 or
# negative random-policy return, so a linear ramp from (0, 0) to the
# first eval is a defensible lower bound on early-phase performance.
# A pair where the agent starts above 0 is not in scope for the v3.5
# secondaries; it would need a per-env baseline column in the payload.

EARLY_STEP_QUERIES = (2_000, 5_000, 10_000)    # env-steps
AUC_WINDOW = (0, 50_000)                        # env-steps
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_RNG_SEED = 20260415                   # reproducibility


def return_at_step(eval_curve: list[dict], step: int) -> float:
    """Linear-interpolate the eval return at `step` env-steps.

    Convention:
      - Before the first eval checkpoint: linear from (0, 0.0) to
        (first_step, first_return). "0.0" assumes random-policy ≈ 0,
        which is true for MCC / Acrobot / DMC-cartpole-swingup.
      - After the last eval checkpoint: clamp to last_return (the
        agent's late-training plateau value).
      - Between checkpoints: standard linear interp.

    Returns NaN if the curve is empty.
    """
    if not eval_curve:
        return float("nan")
    steps = [float(e["step"]) for e in eval_curve]
    rets = [float(e["eval_return"]) for e in eval_curve]
    if step <= 0:
        return 0.0
    if step < steps[0]:
        # Pre-first-checkpoint ramp from (0, 0).
        frac = step / steps[0]
        return frac * rets[0]
    if step >= steps[-1]:
        return rets[-1]
    # Binary-ish linear scan (curves are ≤ 100 points; no need for
    # bisect here).
    for i in range(len(steps) - 1):
        if steps[i] <= step <= steps[i + 1]:
            if steps[i + 1] == steps[i]:
                return rets[i]
            frac = (step - steps[i]) / (steps[i + 1] - steps[i])
            return rets[i] + frac * (rets[i + 1] - rets[i])
    # Should be unreachable given the clamps above.
    return rets[-1]


def auc_return(eval_curve: list[dict], lo: int, hi: int) -> float:
    """Trapezoidal AUC of eval_return over [lo, hi] env-steps, divided
    by (hi - lo) so the output has units of "mean return over window".

    The same 0-anchor / last-clamp convention from `return_at_step`
    applies so a run that mastered at 5k env-steps and held plateau
    return gets a much higher AUC than a run that took 40k to climb.
    """
    if not eval_curve or hi <= lo:
        return float("nan")
    # Build the integration grid: endpoints + all eval points strictly
    # inside [lo, hi]. Ensures the integral is exact on the piecewise
    # linear curve we're interpolating.
    inner = [float(e["step"]) for e in eval_curve
             if lo < float(e["step"]) < hi]
    grid = [float(lo)] + inner + [float(hi)]
    # Remove duplicates while preserving order.
    seen = set()
    grid_unique: list[float] = []
    for s in grid:
        if s not in seen:
            seen.add(s)
            grid_unique.append(s)
    grid_unique.sort()

    values = [return_at_step(eval_curve, int(s)) for s in grid_unique]
    area = 0.0
    for i in range(len(grid_unique) - 1):
        dx = grid_unique[i + 1] - grid_unique[i]
        area += 0.5 * (values[i] + values[i + 1]) * dx
    return area / (hi - lo)


def bootstrap_ci_mean(values: list[float], n_resamples: int,
                      rng_seed: int, alpha: float = 0.05
                      ) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean. Returns (mean, lo95, hi95)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    # Single vectorized resample step for speed.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(resample_means, alpha / 2))
    hi = float(np.quantile(resample_means, 1 - alpha / 2))
    return (mean, lo, hi)


@dataclass
class ArmDescriptives:
    """v3.5 descriptive secondaries per arm (not §8-gating).

    Populated from `runs`' `eval_curve`s. Each entry has the mean +
    bootstrap 95 % CI across seeds.
    """
    arm: str
    n: int
    returns_at_step: dict[int, tuple[float, float, float]]
    # key: step (e.g. 2000, 5000, 10000); value: (mean, lo95, hi95)
    auc_window_lo: int
    auc_window_hi: int
    auc_mean: float
    auc_lo95: float
    auc_hi95: float


def _compute_arm_descriptives(runs: list[dict], arm: str) -> ArmDescriptives:
    """Extract early-step returns + AUC from each run's eval_curve."""
    arm_runs = [r for r in runs if r.get("arm") == arm]
    n = len(arm_runs)

    returns_at_step: dict[int, tuple[float, float, float]] = {}
    for q in EARLY_STEP_QUERIES:
        per_seed = [return_at_step(r.get("eval_curve", []), q)
                    for r in arm_runs]
        m, lo, hi = bootstrap_ci_mean(
            per_seed,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            rng_seed=BOOTSTRAP_RNG_SEED + q,  # different seed per query
        )
        returns_at_step[q] = (m, lo, hi)

    auc_per_seed = [
        auc_return(r.get("eval_curve", []), AUC_WINDOW[0], AUC_WINDOW[1])
        for r in arm_runs
    ]
    m_auc, lo_auc, hi_auc = bootstrap_ci_mean(
        auc_per_seed,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        rng_seed=BOOTSTRAP_RNG_SEED - 1,
    )
    return ArmDescriptives(
        arm=arm,
        n=n,
        returns_at_step=returns_at_step,
        auc_window_lo=AUC_WINDOW[0],
        auc_window_hi=AUC_WINDOW[1],
        auc_mean=m_auc,
        auc_lo95=lo_auc,
        auc_hi95=hi_auc,
    )


def _check_mechanism(transfer_runs: list[dict], src_env: str,
                     tgt_env: str) -> tuple[bool, str]:
    """Per §8: acting_policy_mode == 'latent' for every transfer run in
    pairs that require cross-dim transfer.

    Cross-dim is determined by env registry obs_dim/action_dim. If the
    pair is same-dim (e.g. hypothetical cartpole→cartpole rehearsal), an
    'obs' mode is legitimate and the check passes trivially.
    """
    from ragnarok.environments.registry import get_env_spec
    src_spec = get_env_spec(src_env)
    tgt_spec = get_env_spec(tgt_env)
    cross_dim = (src_spec.obs_dim != tgt_spec.obs_dim
                 or src_spec.action_dim != tgt_spec.action_dim
                 or src_spec.is_discrete != tgt_spec.is_discrete)

    if not cross_dim:
        return (True, "same-dim pair — mechanism check trivially passes")

    modes = [r.get("acting_policy_mode", "obs") for r in transfer_runs]
    loaded = [r.get("transfer_skill_name") for r in transfer_runs]

    latent_count = sum(1 for m in modes if m == "latent")
    loaded_count = sum(1 for s in loaded if s)
    total = len(modes)

    if total == 0:
        return (False, "no transfer runs to check")

    if latent_count < total:
        missing = [r.get("seed") for r in transfer_runs
                   if r.get("acting_policy_mode") != "latent"]
        return (
            False,
            f"{latent_count}/{total} transfer runs have acting_policy_mode=='latent'; "
            f"seeds failing mechanism check: {missing}"
        )

    return (True, f"{latent_count}/{total} transfer runs on 'latent' mode; "
                  f"{loaded_count}/{total} loaded a skill")


# ── Main analysis ───────────────────────────────────────────────────

def analyze(payload: dict) -> PilotVerdict:
    runs = payload["runs"]
    pairs = payload.get("pairs", [])
    tau = int(payload["max_env_steps"])

    pair_verdicts: list[PairVerdict] = []
    failures: list[str] = []
    skipped_pairs: list[str] = []

    any_secondary_pass = False
    any_anti_transfer = False
    primary_verdict: PairVerdict | None = None
    primary_absent = False

    for p in pairs:
        alias = p["alias"]
        role = p["role"]
        src_env = p["src"]
        tgt_env = p["tgt"]

        scratch_runs = _runs_for(runs, alias, "scratch")
        transfer_runs = _runs_for(runs, alias, "transfer")

        # Strategist review blocker #1: a pair with 0 runs in either arm is
        # unanalyzable — don't synthesize a NaN verdict row that clutters the
        # output. Flag it as a failure (the pilot was incomplete) and move on.
        if not scratch_runs or not transfer_runs:
            skipped_pairs.append(alias)
            failures.append(
                f"{alias}: INCOMPLETE — scratch_runs={len(scratch_runs)}, "
                f"transfer_runs={len(transfer_runs)} (both arms required)"
            )
            if role == "primary":
                primary_absent = True
            continue

        scratch = _fit_arm(scratch_runs, "scratch", tau)
        transfer = _fit_arm(transfer_runs, "transfer", tau)

        if scratch.rmst > 0 and transfer.rmst > 0:
            ratio = scratch.rmst / transfer.rmst
        else:
            ratio = float("nan")

        p_logrank = _logrank_one_sided(scratch, transfer, tau)
        # v3.5 robustness: exact permutation p alongside the asymptotic.
        # §8 headline uses the asymptotic (backward-compat with prereg);
        # the permutation value is reported for robustness and gates
        # §10 Plan B0 (which requires BOTH p<0.10 per the prereg v3.5
        # amendment).
        p_perm = _logrank_permutation_one_sided(scratch, transfer)
        mech_ok, mech_msg = _check_mechanism(transfer_runs, src_env, tgt_env)

        anti = (np.isfinite(ratio) and ratio < ANTI_TRANSFER_RATIO_MAX)
        if anti:
            any_anti_transfer = True
            failures.append(
                f"{alias}: ANTI-TRANSFER (RMST ratio = {ratio:.2f} < "
                f"{ANTI_TRANSFER_RATIO_MAX})"
            )

        pass_primary = False
        pass_secondary = False
        if role == "primary":
            pass_primary = (
                np.isfinite(ratio)
                and ratio >= PRIMARY_RMST_RATIO_MIN
                and np.isfinite(p_logrank)
                and p_logrank < PRIMARY_LOGRANK_P_MAX
            )
        if role == "secondary":
            pass_secondary = (
                np.isfinite(ratio)
                and ratio >= SECONDARY_RMST_RATIO_MIN
            )
            if pass_secondary:
                any_secondary_pass = True

        if not mech_ok:
            failures.append(f"{alias}: MECHANISM CHECK FAILED — {mech_msg}")

        # v3.5 descriptive secondaries (non-gating, paper panel + B0
        # companion). Best-effort: if eval_curve is empty/missing on
        # every run in an arm we still construct an ArmDescriptives
        # with NaN values rather than failing the analysis.
        scratch_desc = _compute_arm_descriptives(scratch_runs, "scratch")
        transfer_desc = _compute_arm_descriptives(transfer_runs, "transfer")

        v = PairVerdict(
            alias=alias, role=role,
            scratch=scratch, transfer=transfer,
            rmst_ratio=float(ratio),
            logrank_p_value=float(p_logrank),
            logrank_permutation_p_value=float(p_perm),
            anti_transfer=bool(anti),
            mechanism_check_passed=bool(mech_ok),
            mechanism_details=mech_msg,
            pass_primary_criterion=bool(pass_primary),
            pass_secondary_criterion=bool(pass_secondary),
            tau=tau,
            scratch_descriptives=scratch_desc,
            transfer_descriptives=transfer_desc,
        )
        if role == "primary":
            primary_verdict = v
        pair_verdicts.append(v)

    # Compose overall pass per §8.
    #
    # Strategist review blocker #2: the prior version used `elif` which made
    # the outcome order-dependent (a FAILED primary pair would also mask a
    # downstream anti-transfer failure). §8 is an AND of four independent
    # criteria; evaluate all four, concatenate the failures, then take AND.
    overall = True
    if primary_absent or primary_verdict is None:
        overall = False
        if not primary_absent:  # avoid double-logging
            failures.insert(0, "PRIMARY pair (cartpole_mcc) absent from pilot data")
    else:
        if not primary_verdict.pass_primary_criterion:
            overall = False
            failures.insert(
                0,
                f"PRIMARY {primary_verdict.alias}: ratio="
                f"{primary_verdict.rmst_ratio:.2f} (need ≥ {PRIMARY_RMST_RATIO_MIN}), "
                f"p={primary_verdict.logrank_p_value:.3f} "
                f"(need < {PRIMARY_LOGRANK_P_MAX})"
            )
    if pair_verdicts and not any_secondary_pass:
        # Only reportable if at least one secondary was scored. If every
        # secondary was skipped above, the skip-failure already captured it.
        has_scored_secondary = any(v.role == "secondary" for v in pair_verdicts)
        if has_scored_secondary:
            overall = False
            failures.append(
                f"No secondary pair met RMST ratio ≥ {SECONDARY_RMST_RATIO_MIN}"
            )
    if any_anti_transfer:
        overall = False  # anti-transfer messages already appended
    if pair_verdicts and not all(v.mechanism_check_passed for v in pair_verdicts):
        overall = False  # mechanism-fail messages already appended

    # High-censoring-regime check (devil's-advocate review, Phase 3 pre-commit).
    # Must run BEFORE returning: if the primary pair has ≥80% censoring on
    # either arm, re-label the verdict as INCONCLUSIVE so the caller does
    # NOT activate Plan B on what is effectively a measurement-limit
    # artifact. Overall pass is already False in this regime (ratio/p
    # criteria would have failed anyway), but the failure-vs-inconclusive
    # distinction is load-bearing for the §10 decision tree.
    inconclusive = False
    inconclusive_reason = ""
    if primary_verdict is not None and primary_verdict.scratch.n > 0 \
            and primary_verdict.transfer.n > 0:
        scratch_cens = primary_verdict.scratch.n_censored / primary_verdict.scratch.n
        transfer_cens = primary_verdict.transfer.n_censored / primary_verdict.transfer.n
        if (scratch_cens >= HIGH_CENSORING_THRESHOLD
                or transfer_cens >= HIGH_CENSORING_THRESHOLD):
            inconclusive = True
            inconclusive_reason = (
                f"PRIMARY {primary_verdict.alias}: high-censoring regime — "
                f"scratch censored {primary_verdict.scratch.n_censored}/"
                f"{primary_verdict.scratch.n} ({scratch_cens:.0%}), "
                f"transfer censored {primary_verdict.transfer.n_censored}/"
                f"{primary_verdict.transfer.n} ({transfer_cens:.0%}); "
                f"threshold {HIGH_CENSORING_THRESHOLD:.0%}. Expand sample "
                f"size before invoking Plan B (§10)."
            )
            overall = False  # INCONCLUSIVE is never a PASS

    return PilotVerdict(
        overall_pass=overall,
        pair_verdicts=pair_verdicts,
        failures=failures,
        inconclusive=inconclusive,
        inconclusive_reason=inconclusive_reason,
    )


# ── Rendering ───────────────────────────────────────────────────────

def render_text(verdict: PilotVerdict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  PILOT VERDICT (preregistration §8)")
    lines.append("=" * 70)

    for v in verdict.pair_verdicts:
        lines.append(f"\n{v.alias}  (role={v.role}, τ={v.tau:,} env-steps)")
        lines.append(
            f"  scratch : n={v.scratch.n:>2d}  "
            f"events={v.scratch.n_events:>2d}/{v.scratch.n:<2d}  "
            f"RMST={v.scratch.rmst:>9,.0f}"
        )
        lines.append(
            f"  transfer: n={v.transfer.n:>2d}  "
            f"events={v.transfer.n_events:>2d}/{v.transfer.n:<2d}  "
            f"RMST={v.transfer.rmst:>9,.0f}"
        )
        lines.append(f"  RMST ratio (scratch/transfer): {v.rmst_ratio:.3f}")
        lines.append(f"  Log-rank p (one-sided): {v.logrank_p_value:.4f}")
        # v3.5 robustness probe: permutation p alongside asymptotic.
        # Large disagreement (> 0.05) is itself a reportable finding.
        perm_p = v.logrank_permutation_p_value
        if np.isfinite(perm_p):
            disagree = abs(perm_p - v.logrank_p_value)
            flag = "  [⚠ asymp/perm disagree > 0.05]" if disagree > 0.05 else ""
            lines.append(
                f"  Log-rank p (permutation, "
                f"N={PERMUTATION_N_SHUFFLES:,}): {perm_p:.4f}{flag}"
            )
        else:
            lines.append(
                "  Log-rank p (permutation): n/a (no events observed)"
            )
        status = []
        if v.role == "primary":
            status.append("PRIMARY PASS" if v.pass_primary_criterion
                          else "PRIMARY FAIL")
        if v.role == "secondary":
            status.append("SECONDARY PASS" if v.pass_secondary_criterion
                          else "SECONDARY fail (directional)")
        if v.anti_transfer:
            status.append("ANTI-TRANSFER")
        status.append("mechanism OK" if v.mechanism_check_passed
                      else "MECHANISM FAIL")
        lines.append(f"  Status: {'  |  '.join(status)}")
        lines.append(f"  Mechanism: {v.mechanism_details}")

        # v3.5 descriptive secondaries (non-gating, paper panel + §10 B0).
        if v.scratch_descriptives and v.transfer_descriptives:
            lines.append(
                "  Descriptive secondaries (non-gating; bootstrap "
                "95 % CI over seeds):"
            )
            for step in EARLY_STEP_QUERIES:
                s_m, s_lo, s_hi = v.scratch_descriptives.returns_at_step[step]
                t_m, t_lo, t_hi = v.transfer_descriptives.returns_at_step[step]
                lines.append(
                    f"    return@{step:>5,}  steps:  "
                    f"scratch={s_m:>7.2f} [{s_lo:>6.2f}, {s_hi:>6.2f}]  "
                    f"transfer={t_m:>7.2f} [{t_lo:>6.2f}, {t_hi:>6.2f}]"
                )
            sd = v.scratch_descriptives
            td = v.transfer_descriptives
            lines.append(
                f"    AUC [{sd.auc_window_lo:,}, "
                f"{sd.auc_window_hi:,}]:   "
                f"scratch={sd.auc_mean:>7.2f} [{sd.auc_lo95:>6.2f}, "
                f"{sd.auc_hi95:>6.2f}]  "
                f"transfer={td.auc_mean:>7.2f} [{td.auc_lo95:>6.2f}, "
                f"{td.auc_hi95:>6.2f}]"
            )

    lines.append("\n" + "-" * 70)
    if verdict.overall_pass:
        lines.append("  OVERALL: PASS — proceed to Phase 4 / G2 review gate")
    elif verdict.inconclusive:
        # INCONCLUSIVE is NOT a FAIL — do NOT invoke Plan B on a measurement
        # artifact. Print the underlying failures for transparency but
        # re-label the final verdict so the caller knows to expand sample.
        lines.append(
            "  OVERALL: INCONCLUSIVE — expand sample size before §10 decision")
        lines.append(f"  Reason: {verdict.inconclusive_reason}")
        if verdict.failures:
            lines.append("  Underlying criterion evaluation (FYI — do NOT act on these):")
            for f in verdict.failures:
                lines.append(f"    - {f}")
    else:
        lines.append("  OVERALL: FAIL — activate Plan B per §10")
        lines.append("  Failures:")
        for f in verdict.failures:
            lines.append(f"    - {f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def _verdict_to_dict(v: PilotVerdict) -> dict:
    def arm_dict(a: ArmSurvival) -> dict:
        return {
            "arm": a.arm, "n": a.n, "n_events": a.n_events,
            "n_censored": a.n_censored,
            "rmst": a.rmst, "rmst_variance": a.rmst_variance,
            "median_survival": a.median_survival,
            "durations": a.durations, "events": a.events,
        }

    def desc_dict(a: "ArmDescriptives | None") -> dict | None:
        if a is None:
            return None
        return {
            "arm": a.arm,
            "n": a.n,
            # JSON keys must be strings; cast step -> str here.
            "returns_at_step": {
                str(step): {"mean": m, "lo95": lo, "hi95": hi}
                for step, (m, lo, hi) in a.returns_at_step.items()
            },
            "auc_window_lo": a.auc_window_lo,
            "auc_window_hi": a.auc_window_hi,
            "auc_mean": a.auc_mean,
            "auc_lo95": a.auc_lo95,
            "auc_hi95": a.auc_hi95,
        }

    pair_dicts = []
    for p in v.pair_verdicts:
        d = asdict(p)
        d["scratch"] = arm_dict(p.scratch)
        d["transfer"] = arm_dict(p.transfer)
        # Override with the keyed-str desc layout so downstream
        # consumers get JSON-safe int keys.
        d["scratch_descriptives"] = desc_dict(p.scratch_descriptives)
        d["transfer_descriptives"] = desc_dict(p.transfer_descriptives)
        pair_dicts.append(d)

    return {
        "overall_pass": v.overall_pass,
        "inconclusive": v.inconclusive,
        "inconclusive_reason": v.inconclusive_reason,
        "failures": v.failures,
        "pair_verdicts": pair_dicts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="pilot_results.json path")
    parser.add_argument("--json-output", type=Path, default=None,
                        help="Write verdict JSON to this path (in addition to stdout)")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist")
        return 1

    payload = json.loads(args.input.read_text())
    verdict = analyze(payload)
    print(render_text(verdict))

    if args.json_output:
        args.json_output.write_text(
            json.dumps(_verdict_to_dict(verdict), indent=2))
        print(f"\nWrote verdict JSON to {args.json_output}")

    # Exit code triage so shell callers can branch without parsing text:
    #   0  PASS            → proceed to Phase 4 / G2
    #   2  FAIL            → activate Plan B (§10)
    #   3  INCONCLUSIVE    → expand sample, re-run before §10 decision
    if verdict.overall_pass:
        return 0
    if verdict.inconclusive:
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
