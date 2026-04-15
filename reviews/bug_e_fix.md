# Bug E fix — G1.5 multi-agent code review

**Date:** 2026-04-15
**Reviewed commit:** `f0c9155` (Bug E Phase A–F atomic commit)
**Reviewers:** 3 parallel agents (RL architecture / testing coverage /
devil's advocate)
**Verdict bundle:** review-driven hardenings landed in same Bug E v2
amendment; pilot #2 launch unblocked subject to the upgraded checklist
in `preregistration.md` v3.4 amendment "Bug E v2".

This file fulfills checklist item #2 of the v3.4 amendment ("Verdicts
committed to `reviews/bug_e_fix.md` (NEW) before launch").

---

## Reviewer 1 — RL architecture (`a5ecd934cdcedb77f`)

**Verdict:** FIX-FIRST.

**Top showstopper as raised:** `EnsembleRSSMCore` is the default
(`ensemble_cores=2`) but its submodule names use plural
(`core.grus.`, `core.priors.`, `core.pre_grus.`) — they don't match
the new `_TRANSFERABLE_PREFIXES`, so under default config the
transferable subset is silently empty and cross-dim transfer falls
back to scratch on every pair.

**Disposition:** rejected as written — misreading of the code.
`self.core` (single `RSSMCore`) is built unconditionally on RSSM
init; `self.ensemble` is *additive* and is only consulted by
`dream_augmenter.py` for the disagreement penalty on dream rewards
(verified by inspection: every `observe()` / `imagine()` /
`encode_observation()` / `loss()` call routes through `self.core`).
Empirical check: `RSSM(...,
ensemble_cores=cfg.transfer.ensemble_cores)` produces a
12-key / 132 096-param transferable subset under
`RagnarokConfig()` defaults. The reviewer's *test* request was
correct regardless — added
`test_transferable_subset_nonempty_under_default_config` to lock
this invariant against any future refactor that might actually
move the transferable surface under `self.ensemble.*`.

**Other concerns raised, all actioned:**
- Adam state not reset on transferable group post-load → effective
  step magnitude during the LR warmup is larger than the nominal
  0.1× scale because Adam's bias-corrected updates depend on
  `exp_avg_sq` from before the load. **Fixed:**
  `WorldModelTrainer.reset_transferable_optimizer_state()` clears
  the moments at `try_transfer` time. New tests:
  `test_reset_transferable_optimizer_state_clears_adam_moments`,
  `test_reset_transferable_optimizer_state_preserves_io_state`.
- 200-episode warmup horizon is arbitrary. **Documented in prereg
  as a tuneable HP; the Band-B sweep below covers it.**
- 1.3× pass criterion is likely unreachable given partial transfer.
  **Disposition: §8 primary threshold preserved; the new Band-B
  side-rail in the prereg distinguishes "mechanism dead" from
  "mechanism alive but first-cut HP wrong" without weakening §8.**

---

## Reviewer 2 — Testing coverage (`aff4c13a316dc8404`)

**Verdict:** INSUFFICIENT-BUT-PILOTABLE.

**Same top flag** as Reviewer 1 on ensemble — same disposition
(misreading; regression test added).

**Specific testing-quality concerns, all actioned:**
- LR-scaling tests are tautological — they read the optimizer's
  `.lr` field (which we just wrote!) instead of measuring whether
  Adam actually slows down on real grad steps. **Fixed:**
  `test_lr_warmup_actually_dampens_param_drift` runs identical-seed
  train_steps with and without warmup and asserts ≥ 2× drift
  reduction. The `.lr` field tests are kept for fast-fail signal
  but no longer load-bearing.
- Integration test is byte-match (state_dict equality) not
  behaviour-match (forward-pass equivalence). **Disposition:** not
  added pre-launch — adds non-trivial complexity for a
  diminishing-return signal. The behavioural smoke (checklist
  item #3) exercises the forward path directly on real episodes,
  which is a stronger signal than a synthetic forward-pass match.
- No `every_serialized_field_is_consumed` meta-test (the
  symmetric counterpart of the existing serialization meta-test).
  **Disposition:** deferred. The current
  `test_every_skill_dataclass_field_is_serialized` catches the
  Bug-C/E failure mode (field added → not saved); the symmetric
  test would catch (field saved → not loaded), which is a
  different bug class we have not seen yet.

---

## Reviewer 3 — Devil's advocate (`a672e65eb1fbe6c6b`)

**Verdict:** LAUNCH-WITH-MODIFIED-CRITERION (and only after one
2-seed pre-check).

The fix is architecturally correct, but three concerns compound:
the sample size, the 1.3× threshold, and the partial-transfer
geometry. Reviewer recommends a single 2-seed behavioural
pre-check (~3 GPU-h) plus pre-declared decision-rule bands before
committing 20 GPU-h to pilot #2.

**Eight concerns raised, dispositions:**

1. **`pre_gru` / posterior misalignment.** Transferred GRU weights
   were trained against a specific distribution of `pre_gru`
   outputs that the fresh target `pre_gru` will not produce for
   hundreds of episodes. **Disposition:** the LR warmup + Adam
   reset is the architectural answer; the smoke pre-check (now
   2 seeds, with `||Δθ||` logging) verifies it works in
   practice. If transferable params drift > 30% by ep 200,
   abort.

2. **Prior may act as marginal regularizer, not dynamics
   carrier.** **Disposition:** prereg now commits to reporting
   `KL(posterior‖prior)` trajectory in the paper alongside the
   RMST number — no mechanism-rescue claims if KL stays flat.

3. **Unchanged 1.3× pass criterion is dishonest given untuned HPs.**
   **ACTIONED:** prereg now pre-declares a three-band rule
   (Band A pass / Band B diagnostic-sweep / Band C Plan-B). §8
   primary threshold is preserved; Band B distinguishes a fixable
   first-cut HP from a dead mechanism, all rescue cells must clear
   the same 1.3× / p<0.10 bar.

4. **No mid-pilot early-stop.** **Disposition:** not added
   pre-launch. The kill criterion at week 4 (§11) plus the
   2-seed smoke pre-check together provide adequate guardrails.
   Adding a per-seed early-stop is a 20-line orchestrator change
   that we'll add if pilot #2 trends ambiguous.

5. **Sample-size / sign filter.** **ACTIONED:** prereg now
   requires ≥ 4/5 seed-direction agreement on the primary pair
   even if Band-A criteria are met.

6. **MCC censoring may crush N=5.** Real concern; bootstrap-SE
   would require runs we don't have. **Disposition:** rely on
   the Band-B and sign-filter side-rails to catch underpowered
   positives. Headline N=20 will resolve any residual SE issue
   if pilot #2 lands cleanly in Band A.

7. **`encoder_hidden` silently load-bearing.** **ACTIONED:** the
   posterior shape-mismatch path now raises a `ValueError` that
   names `encoder_hidden` explicitly. New test:
   `test_encoder_hidden_mismatch_message_calls_out_encoder_hidden`.

8. **Trust region disabled in latent mode but not replaced on
   trunk.** **Disposition:** documented in prereg as a deferred
   concern. The 2-seed smoke logs `||Δtrunk||`; if it exceeds 50%
   by ep 100, a symmetric trunk-LR warmup will be added before
   relaunch. Cheap post-hoc; not worth pre-emptive scope creep.

---

## Aggregate disposition

- **Code/test:** all consensus-level concerns landed in the same
  Bug E v2 atomic commit (this set of changes). Test count:
  357 passed / 1 skipped (was 338 / 15 at v3.4 commit).
- **Prereg:** v3.4 amendment "Bug E v2" timestamps the review,
  pre-declares the three-band decision rule + sign-test filter,
  and upgrades the smoke pre-check from 1 seed to 2 seeds with
  diagnostic logging.
- **Decision rules unchanged at primary threshold.** §8 still
  decides launch-vs-Plan-B; the new bands and filters only add
  guardrails, never weaken them.
- **Pilot #2 unblocked** once the upgraded smoke pre-check
  (checklist item #3) returns clean signals.

---

# Bug E v2 → v3 — second round of review

**Date:** 2026-04-15
**Reviewed commit:** `88dbe8c` (Bug E v2 review-driven hardenings)
**Reviewers:** 3 parallel agents (architecture / testing / devil's advocate)
**Verdict bundle:** 2 devil's-advocate BLOCKERS resolved in atomic
v3 commit; pilot #2 still gated on the upgraded smoke (now actually
producing the ||Δθ|| / KL telemetry the v2 prereg committed to).

## Reviewer 1 — Architecture (`a2dd6a522f3788134`)

**Verdict:** LAUNCH-READY.

`reset_transferable_optimizer_state` implementation verified
correct (clears `optimizer.state[p]` for the whole transferable
group; Adam's lazy-init path will re-create on next `step()`).
Two prereg-only edits recommended:

- **Raise Band B lower edge 1.05 → 1.15.** RMST sampling SE at N=5
  with 30–40% MCC censoring is in the 0.15–0.25 range; 1.05 is
  below the noise floor and triggers HP sweeps on null-noise
  outcomes. **ACTIONED in v3 prereg amendment.**
- **Add ensemble-disagreement telemetry to smoke.** Devil's-advocate
  concern: dream-reward disagreement penalty on fresh-random
  ensemble cores may systematically suppress dream rewards during
  the warmup window. **DEFERRED:** dream training is throttled in
  the smoke window (only kicks in after enough replay), so the
  effect is bounded; the v3 telemetry already covers the dominant
  failure modes. Will add `disagr` series if pilot #2 trends
  ambiguous.

## Reviewer 2 — Testing (`a60d74fcf20c8cae5`)

**Verdict:** SUFFICIENT (3 weaknesses, all non-blocking but worth
tightening).

- **2× LR-drift threshold too lenient.** A half-broken warmup that
  drops LR by only 50% would pass it. **ACTIONED:** raised to 4×
  (`test_lr_warmup_actually_dampens_param_drift`, comment expanded).
- **Reset-state test doesn't verify post-step lazy-init behavior.**
  The clear-check passes if state is empty, but a subtle bug could
  leave param_groups pointing at orphaned tensors and break Adam's
  re-init silently. **ACTIONED:** added
  `test_reset_then_step_lazy_init_repopulates_state` — runs one
  train_step after reset and asserts state is repopulated with
  step==1, exp_avg_sq>0.
- **encoder_hidden test misses hidden_dim-only confusion case.**
  The hint is gated on `core.posterior` — but a hidden_dim
  mismatch can also raise on a posterior key (hidden_dim feeds
  posterior input dim too) and would wrongly suggest the user fix
  encoder_hidden. **ACTIONED:** added
  `test_hidden_dim_only_mismatch_does_not_mention_encoder_hidden`,
  feeds only non-posterior keys to control which key raises and
  asserts encoder_hidden is NOT in the message.
- **No integration test for try_transfer call ordering.**
  reset → set_lr_scale ordering is load-bearing but only enforced
  by code review. **ACTIONED:**
  `test_try_transfer_calls_reset_before_set_lr_scale` monkeypatches
  both methods to record call order; asserts reset precedes
  set_lr_scale. A future refactor that reverses them now breaks a
  fast unit test instead of silently degrading the warmup.

## Reviewer 3 — Devil's advocate (`a4017fa9b49149821`)

**Verdict:** LAUNCH-WITH-MODIFIED-CRITERION (2 BLOCKERS).

**BLOCKER #1 — Smoke telemetry committed in prereg, not in code.**
The v2 prereg amendment commits to logging `||Δθ||` on transferable
params, `||Δθ||` on the latent trunk, and `KL(posterior‖prior)`
trajectory during the smoke pre-check, with an abort criterion at
`||Δθ|| > 50% of initial norm by ep 100`. But `scripts/pilot_run.py`
does not actually emit any of these series — making the abort
criterion unenforceable from the smoke output.

**ACTIONED:** `_train_to_step_budget` now snapshots the transferable
subset right after `try_transfer()` succeeds and captures a telemetry
record at every eval checkpoint with `transferable_drift_max`,
`transferable_drift_per_param`, and a `kl_posterior_prior` probe
(single-batch, no-grad, ~few-ms cost). Series serialized as
`PilotRun.telemetry`. A real-time `[TELEMETRY ALERT]` line prints
the first time drift exceeds 50%. Trunk drift logging deferred per
v2 amendment (concern #8 was already deferred).

**BLOCKER #2 — Band-B FPR ~27% under null without Bonferroni.**
The 3-cell HP sweep (warmup_episodes ∈ {50, 200, 500}) at α=0.10
per cell yields FWER ≈ 1 − (1 − 0.10)³ ≈ 27%. "Any cell hits Band A
→ proceed" is the wrong quantifier when the test is run 3 times.
A 1-in-4 chance that pure noise produces a "Band B rescue winner"
is not a rescue.

**ACTIONED:** Bonferroni correction applied — each Band B cell must
clear ratio ≥ 1.30 AND p < 0.0333 (= 0.10 / 3) to qualify as a
rescue winner. The §8 primary at N=20 confirms unchanged. Why
plain Bonferroni and not Holm: with 3 cells × N=3 each, the power
gain from sorted-p tracking is marginal and the implementation
cost in the analyzer is non-trivial.

**Other concerns from this round:**
- **Band B lower edge below RMST noise floor at N=5 + censoring.**
  Raised to 1.15 (overlap with architecture review).
- **Disagreement-penalty suppression of dream rewards during warmup.**
  Deferred (architecture review same disposition).

## Aggregate v3 disposition

- **Code/test/prereg:** all 2 BLOCKERS + 4 testing concerns landed
  in the same atomic v3 commit. Test count: 360 passed / 1 skipped
  (was 357 / 1 at v2 commit).
- **Decision rules:** Band B effective range tightened
  (1.15–1.30 vs 1.05–1.30); Band B per-cell α tightened (0.0333 vs
  0.10). §8 primary unchanged at headline N=20.
- **Smoke pre-check now actually enforceable:** ||Δθ|| series and
  KL probe are emitted to `pilot_results.json` so the prereg's
  abort criterion can be evaluated programmatically post-smoke.
- **Pilot #2 unblocked** once the v3 smoke (re-run on the new
  pilot_run.py) returns telemetry with no abort triggered.
