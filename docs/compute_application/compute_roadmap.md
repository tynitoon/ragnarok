# Compute Roadmap — TPU Hours Budget & Experiment Plan

**Purpose:** precise, numerically accountable plan for how TPU compute would be used, month by month, with pre-registered go/no-go gates.

---

## GPU-to-TPU conversion rationale

All cost estimates below are anchored to observed wall-clock times on the project's current hardware (single RTX 4080, recorded in `pilot_results.json` provenance fields). TPU equivalents use conservative 2× throughput assumption (standard TPU v3-8 vs RTX 4080 for RSSM-class workloads; JAX/XLA tuning should reduce further but is not assumed here).

| Experiment unit | RTX 4080 wall-clock | Conservative TPU v3-8 equivalent |
|---|---|---|
| 1 cartpole source crystallization | ~5 min | ~3 min |
| 1 scratch MCC run to mastery | ~60 min | ~30 min |
| 1 transfer MCC run to mastery | ~60 min | ~30 min |
| 1 pair × N=5 (source + scratch + transfer) | ~7 hours | ~3.5 hours |
| Full pilot (3 pairs × N=5) | 35 hours | ~17 hours |

Month-1 calibration sweep will produce empirical TPU-hour values and replace these conservative estimates in the Month-1 report.

---

## Month 1 — validate pipeline + execute preregistered ablations

**Budget request: 1 TPU v3-8, 30 days, preemptible overflow. Expected actual usage: ~40 TPU-hours.**

### 1.1 Calibration sweep (~10 TPU-hours)

**Goal:** validate that the PyTorch→JAX port of RSSM produces identical results on the Band B rescue dataset (ratio 1.605, LOO min 1.435, p=0.259).

**Exit criterion:** TPU-replicated ratio within ±0.05 of the RTX 4080 Band B result at matched seeds. If larger drift, investigate before proceeding.

### 1.2 A10 adversarial pair ablation (~15 TPU-hours)

**Goal:** test cross-action-type transfer on a *non-pendular* target to falsify the devil's-advocate attack "the primary pair is trivially transferable by construction."

**Setup:** Pendulum (source, continuous Box) → DMC-finger-spin (target, continuous Box, non-pendular dynamics). N=5. Same §8 criteria (ratio ≥ 1.30, p < 0.10) but pre-registered in §12.5 as adversarial.

**Pre-registered outcomes:**
- Ratio ≥ 1.30 at p < 0.10 → generality of cross-action-type claim supported; paper claim is strengthened.
- Ratio ∈ [1.00, 1.30) → claim narrowed to "cross-action-type works when source and target share physics class."
- Ratio < 1.00 (transfer actively harms) → cross-action-type claim rejected; paper pivots to Plan B0 or abandoned.

### 1.3 A11 GRU-shuffled mechanism ablation (~10 TPU-hours)

**Goal:** verify the mechanism claim. Preregistered in §10 B0 clause 3.

**Setup:** on the primary pair (CartPole → MCC), N=5, replace the transferred GRU weights with a random permutation of the same tensor (preserves spectrum, destroys structure). Compare RMST ratio to the real-transfer arm.

**Pre-registered threshold:** real-transfer ratio − shuffled-transfer ratio ≥ 0.10 → mechanism confirmed. Gap < 0.10 → mechanism claim cannot be defended; Plan B0 is invalidated as-specified.

### 1.4 Buffer / debug (~5 TPU-hours)

Unexpected reruns, JAX-port bugs, re-verification.

### Month 1 deliverables (hard commitments)

- Public monthly report (published within 48 h of month-end): per-experiment results, TPU-hour consumption, go/no-go verdicts.
- Calibration benchmark GPU-hr↔TPU-hr, used to adjust Month 2–3 budget.
- Updated `preregistration.md` §13 amendment recording A10 + A11 outcomes and their implications.
- Updated `pilot_results.json` / new `pilot_a10_results.json` / `pilot_a11_results.json` committed to the public repo.

---

## Months 2–3 — Post-1 horizontal scale (stretch, conditional)

**Budget request: renewable based on Month-1 outcomes. Expected: ~60 TPU-hours if Month-1 gates pass.**

### 2.1 Post-1 horizontal-scale pilot (~40 TPU-hours)

**Goal:** extend the skill library from 3 to ~10 skills via 7 new source-target pairs, spanning DMControl (cheetah-run, walker-walk, hopper-hop, quadruped-stand) and MetaWorld (pick-place, reach, button-press).

**Pre-registered design:** N=5 seeds per pair, same analysis pipeline as pilot #2, same §8 criteria adjusted for per-task mastery thresholds.

**Go/no-go for this phase:** executed **only** if Month 1 A10 lands at ratio ≥ 1.20 and A11 lands with mechanism gap ≥ 0.10. Otherwise paused and re-planned via multi-agent review.

### 2.2 Buffer + analysis (~20 TPU-hours)

Re-runs, analysis iteration, data-quality fixes.

### Month 2–3 deliverables

- `pilot_post1_results.json` with seed-level data for all 7 new pairs.
- Updated preregistration amendment.
- Second and third TRC monthly reports.
- Draft methodology blog post (by end of Month 2, if not earlier).

---

## Month 4 — dissemination (compute-negligible)

### 4.1 If Month 1 + Months 2–3 pass thresholds

- Workshop paper draft (RLC 2026 workshop track or NeurIPS 2026 RL workshop).
- Paper supplementary materials: seed-level data, reproduction scripts, multi-agent review artifacts, chronology audit.
- Public blog post on preregistration methodology.
- Final TRC report (includes per-experiment TPU-hour consumption table, reproducibility score, peer-review status).

### 4.2 If Month-1 gates fail

- Public blog post on methodology (unconditional commitment).
- Negative-result technical report posted to the public repo under `docs/`.
- Final TRC report documenting the abandoned paper path and justifying compute usage with the ablations that *were* completed.
- Main-track paper deferred to a future application with stronger empirical foundation.

---

## Total compute ask summary

| Phase | TPU-hours | Conditional? |
|---|---|---|
| Month 1 — validate + A10 + A11 | ~40 | Initial grant |
| Months 2–3 — Post-1 scale | ~60 | Conditional on Month-1 gates |
| Month 4 — dissemination | ~0 | Unconditional if prior phases ran |
| **Total upper bound** | **~100** | |

Requested initial allocation: **~40 TPU-hours over 30 days** (1 TPU v3-8 on-demand + preemptible overflow). Renewable after Month 1 report.

This is intentionally below the typical TRC allocation. The goal is to earn renewal with production rather than over-ask upfront and under-deliver.

---

## Transparency on compute cost estimates

The estimates above could be wrong by a factor of 2× in either direction. RSSM workloads on TPU are sensitive to batch size, sequence length, and JAX/XLA compilation choices that cannot be fully characterized until the Month-1 calibration sweep completes. Every monthly report will include a "cost reality check" section comparing estimated vs actual, with lessons for the next month's budget.
