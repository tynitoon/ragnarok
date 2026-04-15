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
