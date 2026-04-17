# Methodology Showcase — Ragnarok

**One-page summary of the project's methodological artifacts, for compute-grant reviewers and workshop reviewers who want to verify research rigor in under 5 minutes.**

---

## Why this matters for a compute grant

A compute-grant reviewer's main concern is: *"will this person produce verifiable science with the TPU-hours, or burn them on unreproducible experiments?"* Methodology is the best available signal for that question, especially in the absence of a publication track record. Ragnarok's methodology is deliberately **over-engineered for N=10** — explicitly because the author has no institutional peer review to lean on. The artifacts below exist to prove claims don't drift between commit time and publication time.

---

## 1. Preregistration as a file, not a gesture

`preregistration.md` (1364 lines, git-tracked from the project's first week) commits every hypothesis, threshold, analysis, and kill criterion **before** the data exists. Amendments go in §13 with commit SHA, timestamp, and rationale. Currently 11 amendments across v3.0 → v3.7.

**Reviewer verification (~30 seconds):**
```bash
git log preregistration.md --oneline | head -20
# Shows every amendment as its own timestamped commit.

git show <commit-SHA>:preregistration.md | grep -A 3 "§8"
# Any reviewer can recover the prereg state as of any commit and verify
# that the primary threshold (ratio ≥ 1.30, p < 0.10) did not change
# after pilot data was collected.
```

## 2. Chronology audit — finding and correcting integrity defects self-initiated

`reviews/chronology_audit.md` is a solo-initiated audit (triggered by a devil's-advocate review, not by external demand) that found the v3.5 preregistration text claimed "B0 committed pre-data" when the commit timestamp showed B0 was committed during pilot execution. The audit:

1. Reconstructed the run-by-run pilot timeline from `pilot_run.log` wall-clock fields.
2. Established that at B0 commit time, 2/5 primary seeds were complete showing partial ratio 1.10 (below the B0 band's floor of 1.15, so the band was *not* fit to observed data).
3. Concluded: *pre-outcome* (before N=5 RMST could be computed), but *not pre-pilot*. This is weaker than the §8 claim (fully pre-pilot) but still defensible.
4. The v3.6 amendment replaces the misleading text with an explicit chronology statement and the full audit is part of the paper's supplementary materials.

This is a full, public, self-initiated integrity correction. It is rare in solo-dev research.

## 3. Multi-agent adversarial reviews at every decision gate

Before each phase-transition (pre-pilot-launch, mid-pilot, post-pilot-verdict, pre-compute-grant submission), 3–6 specialized LLM agents review the plan independently and in parallel. Dissent is logged. See:

- `reviews/post_pilot_backlog.md` — 4-agent mid-pilot review 2026-04-15
- `reviews/research_directions.md` — 4-agent research-planning review 2026-04-16
- `reviews/pre_trc_4agent_review_2026-04-17.md` — 4-agent pre-submission review (this one)

Each review specifies the critical corrections identified and which were integrated. These reviews are LLM-based and therefore **not a substitute for external human peer review** — but they catch issues that single-author workflows miss (amendment fatigue, novelty-claim gerrymandering, framing liabilities) before they reach an external reviewer.

## 4. Kill criteria that actually kill

`preregistration.md` §11 lists conditions under which the project is explicitly abandoned — not redefined, not rescued. §13 amendments add new kill criteria each time a new decision gate appears (e.g., v3.7 Band C: "ratio < 1.20 OR p ≥ 0.20 OR LOO min < 1.00 → project abandonment of primary-pair workshop path").

## 5. Seed-level data in git

`pilot_results.json` (40 runs), `pilot_bandb_results.json` (15 runs, Band B rescue), `pilot_bandc_results.json` (Band C N=10 extension, in flight at time of writing). Each entry contains:

- `env_name`, `arm` (source / scratch / transfer), `seed`, `pair_alias`, `pair_role`
- `total_env_steps`, `steps_to_mastery`, `final_eval_return`, `best_eval_return`, `wall_clock_sec`
- `eval_curve` — the full learning curve as `[(step, eval_return), ...]`
- `transfer_skill_name`, `acting_policy_mode` (mechanism check)
- `provenance`: git SHA at run time, Python version, PyTorch version, CUDA version, GPU model

**Reviewer verification (~2 minutes):**
```bash
git clone https://gitlab.com/mortier.jeremie/ragnarok
cd ragnarok
python -m scripts.pilot_analysis pilot_bandb_results.json
# Reproduces the RMST ratio, log-rank p-value, permutation p-value,
# mechanism check, and §8 verdict shown in the proposal.
```

## 6. Test suite as a drift-prevention artifact

444 passing tests (`pytest tests/`), covering:
- RSSM transferable-subset shape compatibility across obs/action dims
- Skill crystallization / library serialization
- SAC policy on continuous envs
- Curiosity module (latent KL)
- World-model trainer invariants
- Pilot pipeline smoke mode

The test suite is the execution-time safety net for the claim *"the transferable subset actually transfers."* Bugs caught in this test suite during Phase 3 (documented in preregistration §13 Bug E v1 through v5.3) prevented silent data corruption that would have inflated transfer arm results.

---

## What the methodology does *not* substitute for

- **External peer review.** Workshop submission (conditional on Band C + A10 + A11) remains the real external validation. Multi-agent reviews are adversarial scaffolding, not a proof of correctness.
- **Power at N=10.** Prereg §5 concedes that N=10 is underpowered at HR=1.5 with ~40% censoring. The Band C extension is the project's best effort at power; the paper will report the observed power and frame the claim accordingly.
- **Generality of the claim.** The primary pair is the most favorable cross-action-type pair imaginable. A10 (non-pendular adversarial pair) and Post-1 horizontal scale are required for any generality statement.

---

*All verifications in this document are public, commit-SHA-anchored, and do not require privileged access. A reviewer with 5 minutes and git can verify every claim made here.*
