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

---

# Bug E v3 → v4 — third round of review

**Date:** 2026-04-15
**Reviewed commit:** `e24832c` (Bug E v3 hardenings)
**Reviewers:** 3 parallel agents (architecture / testing / devil's
advocate)
**Verdict bundle:** 1 architecture MAJOR + 2 testing MAJORs + 1
devil's-advocate BLOCKER + 3 devil's-advocate MAJORs all resolved
in atomic v4 commit; pilot #2 launch unblocked subject to v4 smoke
re-run on the now-correct (2-seed, 40k-step, telemetry-emitting)
pilot_run.py.

## Reviewer 1 — Architecture (3rd round)

**Verdict:** FIX-ONE-MAJOR.

The v3 implementation of `_capture_telemetry` calls
`rssm.loss(obs, actions)["kl_loss"]` to populate the
`kl_posterior_prior` probe. That field is the **free-nats-clamped
training objective**, computed as
`torch.clamp(kl, min=free_nats/stoch_dim).sum(-1).mean()` (default
`free_nats=1.0`, `stoch_dim ∈ {8, 16, 32}` per env). The clamp
floor IS the expected raw KL value during the first hundreds of
episodes — so the probe is structurally **incapable** of detecting
the very failure mode the v3 amendment claimed it would detect
("prior crushed by posterior"). A flat-prior, flat-posterior
configuration would yield raw KL ≈ 0 but the probe would report
≥ 0.25 — the floor.

**ACTIONED in v4:** the telemetry function calls `rssm.observe(obs,
actions)` directly and computes
`kl_divergence(Normal(post_m, post_s.exp()),
Normal(prior_m, prior_s.exp())).sum(-1).mean()` — no clamping, no
weight, no per-dim averaging that would obscure low values. New
test `test_kl_probe_is_unclamped_raw_kl` constructs identical
post/prior Normals and asserts the probe reports ~0 (would fail at
~0.25 if the clamped path regressed). All other architecture
review findings: LAUNCH-READY.

## Reviewer 2 — Testing (3rd round)

**Verdict:** INSUFFICIENT-WITHOUT-FIX (2 MAJORs).

- **MAJOR #1: Telemetry function has zero unit-test coverage.**
  The v3 implementation lives as a closure inside
  `_train_to_step_budget`, so it can't be imported and tested in
  isolation. A regression in the closure (e.g. swapping the KL
  probe for a clamped one — see architecture MAJOR above) would
  only surface during a smoke run, not during PR-time tests.
  **ACTIONED in v4:** `_compute_transfer_telemetry` extracted to
  module level. Seven new unit tests in `TestComputeTransferTelemetry`:
  baseline=None handling (×2), drift math (×2), step/episode pass-
  through, raw-KL guarantee, exception swallowing, kl_probe_error
  semantics on success / empty-buffer / crash (×3 — last 2 added
  alongside the v4 MINOR fix below).

- **MAJOR #2: `test_hidden_dim_only_mismatch_does_not_mention_
  encoder_hidden` sidesteps reality.** The v3 test filters the
  source state_dict to non-posterior keys only, controlling which
  key raises first. In production usage the user passes the FULL
  `transferable_state_dict()`, and iteration order over the dict
  determines which key raises — if posterior happens to iterate
  first under hidden_dim mismatch, the encoder_hidden hint fires
  and misdirects the user. **ACTIONED in v4:** new test
  `test_hidden_dim_mismatch_unfiltered_dict_no_encoder_hint`
  exercises the realistic full-dict call path and asserts the
  message does NOT mention encoder_hidden. The test's docstring
  documents the iteration-order dependence (`gru → prior →
  posterior`) and warns that any future refactor changing prefix
  order would correctly fail this test.

## Reviewer 3 — Devil's advocate (3rd round)

**Verdict:** LAUNCH-WITH-MODIFIED-CRITERION (1 BLOCKER + 3 MAJORs).

**BLOCKER #1 — `--smoke` flag still hardcodes seeds=1.** The v2
prereg amendment commits to a 2-seed smoke pre-check, and the v3
amendment doubles down on it (with the abort criterion enforceable
via the new telemetry). But `scripts/pilot_run.py:--smoke` was
never updated and still sets `args.seeds = 1`. Any operator
running the CLI smoke per the prereg-documented invocation produces
a single-seed smoke that **violates the prereg**. The whole
2-seed-with-||Δθ||-aggregation discipline is dead code if the flag
that's supposed to enable it doesn't.

**ACTIONED in v4:** `--smoke` now sets `args.seeds = 2` and
`args.max_steps = 40_000`. (40k bumped from 20k because the
ep-100 abort criterion would land in the no-margin zone for slow
mastery curves at the v3 default.) Help text and usage docstring
updated. Test
`test_smoke_flag_sets_reduced_budget` updated to assert the new
values, and the assertions explain WHY they changed (citing the
prereg amendment numbers) so a future reverter has to look at the
prereg before flipping the test back.

**MAJOR #1 — Band B sweep statistically dead.** Power analysis on
the v3 Bonferroni-corrected design at α = 0.0333, df = 2,
ratio = 1.5, σ = 0.25 yields **power ≈ 7.4%**. A 3-cell sweep
where each cell has 7% chance of correctly identifying a real 1.5×
effect is not a rescue mechanism, it's a coin toss. The original
intent of Band B was "if the §8 primary is null but the mechanism
is alive at a different HP, find that HP" — but the cell that
contains the right HP only fires 7% of the time even when right.

**ACTIONED in v4:** Band B collapsed from 3 cells to 1 cell at
warmup_episodes=200 (the only cell with prior architectural
justification). With 1 cell, no multiplicity correction is needed,
α reverts to 0.10 (matches §8 primary), and at N=5 the same
ratio/σ assumption gives power ≈ 50%. If pilot #2 lands in Band B,
a follow-up sweep with proper N can refine; if it lands in Band C,
Plan B is the answer, not a wider exploratory net.

**MAJOR #2 — Band B lower edge 1.15 still in noise floor.** At
σ = 0.25 (upper of the v3-estimated 0.15–0.25 RMST noise range),
the null p-value for a 1.15 ratio is ≈ 0.17 — above the 10% bar
that §8 primary uses. So a Band B "rescue winner" at ratio = 1.15
has worse statistical evidence than the §8 primary requires.

**ACTIONED in v4:** Band B lower edge raised 1.15 → 1.20 (null
p ≈ 0.10 at σ = 0.25, matches §8 α exactly). Effective Band B
window after v4: ratio ∈ [1.20, 1.30) at p < 0.10 OR ratio ≥ 1.30
at p ∈ [0.10, 0.20).

**MAJOR #3 — 2-seed smoke aggregation rule undefined.** The v2
amendment commits to 2-seed smoke with telemetry abort criterion
"||Δθ|| > 50% by ep 100", but doesn't specify how to combine the
two seeds: abort if EITHER seed triggers, or only if BOTH? The
default reading would be "mean across seeds" which masks
catastrophic single-seed failures.

**ACTIONED in v4:** prereg pre-declares "EITHER seed triggers
abort" (pessimistic — a single broken seed is sufficient evidence
to hold the 20-GPU-h pilot launch). Coded into the smoke output
via per-seed telemetry series; analyzer aggregation rule pre-
declared so post-hoc cherry-picking is impossible.

**Other concerns from this round:**
- **MINOR — silent kl_probe failures lose diagnostic info.** v3
  has `except Exception: kl_probe = None`. The bare swallow means
  a permanently-broken probe is indistinguishable from a working
  probe that never gets called (e.g. if the buffer happens to be
  empty at every checkpoint). **ACTIONED:** added
  `kl_probe_error: str | None` field to the telemetry record,
  populated with `repr(e)[:200]` on exception or with an explicit
  "buffer empty" string when num_episodes < 1. Two new tests
  pin the contract:
  `test_kl_probe_error_is_none_on_success` and
  `test_kl_probe_error_distinguishes_empty_buffer_from_crash`.

## Aggregate v4 disposition

- **Code/test/prereg:** 1 BLOCKER + 5 MAJORs + 1 MINOR all landed
  in the same atomic v4 commit. Test count: 356 passed / 15
  skipped on Python 3.14 main env (the 14-test DMC delta vs the
  v3 baseline of 360/1 is environment, not regression — DMC tests
  run only on venv310). v4 added 4 new tests on top of the v3
  baseline:
  - `test_hidden_dim_mismatch_unfiltered_dict_no_encoder_hint`
  - `test_kl_probe_error_is_none_on_success`
  - `test_kl_probe_error_distinguishes_empty_buffer_from_crash`
  - 2 new assertions extending `test_telemetry_swallows_buffer_exception`
- **Decision rules:** Band B further tightened (1-cell vs 3-cell;
  1.20 lower edge vs 1.15; per-cell α back to 0.10 since
  multiplicity correction no longer needed); 2-seed aggregation
  pre-declared (EITHER seed > 50% triggers abort). §8 primary
  unchanged at headline N=20.
- **Smoke flag now actually executes the prereg-committed
  protocol:** seeds=2, max_steps=40k, telemetry emitted to
  pilot_results.json, kl_probe_error field present.
- **Pilot #2 unblocked** once v4 smoke (re-run on this commit's
  pilot_run.py) returns clean telemetry with no abort triggered
  on either seed.

---

# Bug E v4 → v5 — fourth round of review

**Date:** 2026-04-15
**Reviewed commit:** `b73bc0e` (Bug E v4 hardenings)
**Reviewers:** 3 parallel agents (architecture / testing / devil's
advocate)
**Verdict bundle:** Architecture LAUNCH-READY; Testing
SUFFICIENT-BUT-PILOTABLE (1 MINOR — docstring math); Devil's
advocate LAUNCH-WITH-MODIFIED-CRITERION (no code blocker, but
explicit operational gate: v4 smoke must complete with non-empty
telemetry + drift < 50% on BOTH seeds at ep 100). PLUS one v5
BLOCKER discovered live during the v4 smoke run that no review
caught (because no review actually ran the smoke end-to-end on
a fresh skills directory).

## Reviewer 1 — Architecture (4th round)

**Verdict:** LAUNCH-READY.

All 4 v4 changes architecturally correct:
1. `_compute_transfer_telemetry` extraction + raw-KL rewrite —
   verified clamp-floor math (free_nats=1.0, stoch_dim ∈ {8,16,32}
   → floor 0.03–0.125; clamped path returns floor × stoch_dim
   = free_nats = 1.0). Closure capture pattern verified safe at
   pilot_run.py:560-568 (baseline tensors `.detach().clone().cpu()`,
   no grad leak, agent ref by reference is correct since the probe
   reads current state).
2. `--smoke` flag fix verified with corresponding test update.
3. `kl_probe_error` field — JSON-roundtrip verified (str | None
   are native JSON types); no downstream consumer breakage (analyzer
   doesn't read telemetry yet, so trivially compatible).
4. Band B 1-cell collapse — pushed back on the "warmup=200 is the
   only justified cell" claim with a 3-failure-mode analysis
   (effect at warmup=50, at warmup=500, U-shaped non-monotone).
   Concluded the only topology the 1-cell rescue actually misses
   is non-monotone, which has no architectural justification
   (LR-warmup mechanism is monotone in horizon length). Verdict:
   1-cell collapse is not hiding signal.

Two non-blocking observations: probe batch/seq caps (`min(8,bs)`,
`min(50,sl)`) not asserted in tests; and `kl_probe_error` is
write-only (no analyzer reads it). The latter is addressed in v5
by `scripts/smoke_verdict.py`.

## Reviewer 2 — Testing (4th round)

**Verdict:** SUFFICIENT-BUT-PILOTABLE.

All 5 v4 testing changes pin the contracts they claim. Two real
gaps named (both non-blocking):
- No integration test for telemetry end-to-end (snapshot baseline
  → train → emit → JSON → roundtrip). All 9 unit tests use stubs.
  A mutant that breaks JSON-serializability of the record dict
  passes every unit test and fails at first write during a pilot.
  Worth adding post-pilot; for launch, the smoke run serves as
  de-facto integration test.
- No test of the `transferable_baseline_norms` snapshot path at
  pilot_run.py:498-500. A bug populating an empty dict would only
  surface at first drift computation where the `or 1.0` fallback
  silently returns meaningless drift.

One MINOR (fixed in v5): docstring math error in
`test_kl_probe_is_unclamped_raw_kl` — claimed clamped path returns
0.25 (forgetting the post-clamp `.sum(-1)`); actually returns
free_nats × stoch_dim/stoch_dim = 1.0. The test assertion
(`approx(0.0, abs=1e-5)`) is correct and load-bearing — it would
fail at the clamped value of 1.0 by 100,000× the tolerance — only
the comment math was wrong.

## Reviewer 3 — Devil's advocate (4th round)

**Verdict:** LAUNCH-WITH-MODIFIED-CRITERION (no code blocker; one
operational gate).

**Refuted concerns** (the prompt deliberately included some that
the reviewer should sanity-check):
- **KL probe variance at bs=8, sl=50: REFUTED.** Empirical
  CoV ~0.6% across batches (averaging over B×T=400 samples per
  dim); the prompt's "30%+ sample variance" intuition was wrong
  (`.sum(-1).mean()` averages over the grid, not over a single
  triple). Probe SE is 3 orders of magnitude tighter than any
  plausible signal range.
- **σ=0.25 conservatism: REFUTED.** v2 smoke shows 0/4 censoring
  at 20k env steps — at the 200k pilot budget MCC censoring will
  be < 20%, not 30-50%. So σ ≈ 0.15-0.20 is more likely than the
  v3-cited 0.25, making 1.20 conservative, not aggressive.
- **Mastery extrapolation from v2 smoke: PARTIALLY ACCEPTED.**
  v2 smoke ratios 1.025× / 0.99× are statistically zero — but
  this is expected at 20k where both arms hit mastery in ~5k
  steps and RMST saturates. The N=5 distribution at the pilot
  budget is the actual statistical signal, not per-seed
  extrapolation.

**Mild dissent (not a blocker):** Band B should arguably have
been at warmup=50 rather than warmup=200 given the v2 smoke shows
transfer mastering in 127/135 episodes (warmup=200 exceeds the
mastery horizon). v4's "200 is the only architecturally-justified
cell" argument is a ceiling, not a calibrated estimate. Disposition:
keep warmup=200 since Band B is secondary; if pilot #2 lands in
Band C, a follow-up sweep can refine.

**Operational gate (the actual launch criterion):** "Launch pilot
#2 iff (a) `smoke_bug_e_v4.json` contains non-empty `telemetry`
arrays with `kl_posterior_prior` values and `kl_probe_error: null`
for both transfer seeds, and (b) max drift < 0.50 at every
checkpoint <= ep 100 on both seeds." Currently no script enforces
this — addressed in v5 by `scripts/smoke_verdict.py`.

## v5 BLOCKER discovered live (not by any reviewer)

While the 4th-round reviews ran, I launched the v4 smoke. The
source-cartpole arm reached the v4 default `--smoke` source cap
of 10k steps without crystallizing (eval=19, threshold=450). With
no crystallized source skill, the transfer arm would fall back
to scratch and the smoke's whole point — validating the
telemetry-emitting transfer code path — would be defeated.

Root cause: the v4 commit bumped seeds (1→2) and max_steps
(20k→40k) but left source_max_steps at the v3 value of 10k. The
v2 smoke that "worked" (cartpole crystallized at 354s + 256s)
had source_max=100k, but that was set via manual CLI override,
not the `--smoke` default. So the `--smoke` flag has been
nominally broken on the source-crystallization side since v3;
the v2 smoke result was an accident of operator override.

Why no reviewer caught it: all 4 review rounds were code-level;
none of them ran the smoke against a fresh (no-skills-on-disk)
checkout. The v2 smoke output that reviewers cited
(`mode=latent` on both seeds) was from a manually-invoked run
with custom flags. Operational testing exposed a class of bug
that code review structurally cannot.

**ACTIONED in v5:** `--smoke` source_max_steps bumped 10k→100k
to match the known-good v2 smoke. Help text and usage docstring
expanded to call out the previous failure mode explicitly so
future operators don't re-introduce the bug. Test
`test_smoke_flag_sets_reduced_budget` updated to assert the new
100k value with a citation to the live-discovery in this review
section.

## v5 additions

Beyond the BLOCKER fix and the testing MINOR (docstring), v5 also
lands the operational-gate enforcement the devil's-advocate review
called for:
- **`scripts/smoke_verdict.py`** — CLI tool that consumes a
  smoke `pilot_results.json`, applies the prereg's EITHER-seed
  drift abort rule, validates `kl_probe_error` is None or the
  expected buffer-empty sentinel, and exits 0 (PROCEED) or 1
  (ABORT). Replaces "operator reads JSON manually" with an
  enforceable shell-pipeable verdict.
- **`tests/test_smoke_verdict.py`** — 16 unit tests pinning the
  contract: drift at threshold doesn't abort (strict-greater);
  drift > threshold within ep 100 aborts; drift > threshold
  outside window does NOT abort; EITHER-seed rule (clean +
  bad → ABORT); empty telemetry on crystallized source → FAIL
  (pipeline regression); empty telemetry on uncrystallized
  source → FAIL (transfer fell back to scratch); buffer-empty
  probe error doesn't fail; real probe exception fails;
  schema-compatibility for both flat-runs and grouped-pairs JSON;
  CLI exit codes (0/1/2).

## Aggregate v5 disposition

- **Code/test/prereg:** 1 live-discovered BLOCKER + 1 testing
  MINOR + the operational-gate enforcement tool all landed in
  the same atomic v5 commit. New test count: ~373 passed (356
  v4 baseline + 16 smoke_verdict + 1 docstring update doesn't
  add tests).
- **Decision rules unchanged at primary §8 threshold.** v5 only
  fixes one bug (smoke source cap) and adds enforcement
  infrastructure.
- **Pilot #2 unblocked** once the v5 smoke (re-run on this commit's
  pilot_run.py) passes `python -m scripts.smoke_verdict
  smoke_bug_e_v5.json` with exit code 0.

## Bug E v5 → v5.2 — fifth round of review (devil's advocate, post-v5)

Devil's advocate v5 review (LAUNCH-WITH-MODIFIED-CRITERION):
the v5 changeset has no code BLOCKER and the EITHER-seed
semantics + sentinel-sharing concerns were already discharged
(v5 introduced `TELEMETRY_BUFFER_EMPTY_SENTINEL` as a shared
constant; the "duplicated literal" worry in the prompt was
stale — described the v4 state, not v5). v5.1 separately
landed the architecture-review polish (sentinel hoist + grouped-
pairs fallback documentation).

The reviewer surfaced **one MAJOR with empirical teeth**:

**MAJOR — `SOURCE_MAX_ENV_STEPS_DEFAULT = 100k` has thin margin
for unseen pilot-#2 seeds.** Observed cartpole crystallization
across n=3 prior runs (seeds 42/43/44 in
`pilot_results.json.broken_trunk` and `smoke_bug_e_v2.json`):
32k / 40k / 60k env steps. Sample mean ≈ 44k, std ≈ 14k.
Mean+4σ lands right at 100k — the previous default left near-
zero headroom for the right tail of cartpole's training-time
distribution. Pilot #2 introduces seeds 44, 45, 46; if any of
them sits in the heavy tail (RL training-time distributions
are NOT gaussian, they're heavy-tailed), the source run hits
the cap without crystallizing and the matching transfer arm
degrades to scratch — one wasted data point out of 30. The
harness fail-closes on this case (`SeedVerdict.pass_` returns
False when `source_crystallized is False`), so the analyzer
correctly flags it. But each lost arm is 3.3% of statistical
power.

**ACTIONED in v5.2:** raised `SOURCE_MAX_ENV_STEPS_DEFAULT`
100k → 150k. Math: this gives mean+5σ headroom on the existing
cartpole sample (still tail-bounded for the heavy-tail risk),
adds ~5 GPU-min per source-arm in pilot #2 (~50 min total
across 9 source arms — negligible vs the 200k pilot-arm
budget). **`--smoke` is unchanged** — it explicitly overrides
to 100k because we've already observed seeds 42/43 crystallize
at <60k, the smoke cap doesn't need the extra margin, and the
saved ~2-min × 2-seed iteration time matters for smoke
turnaround. Test `test_source_cap_is_below_target_budget`
relaxed from `<= MAX_ENV_STEPS_DEFAULT // 2` (an ad-hoc
invariant) to `< MAX_ENV_STEPS_DEFAULT` (the meaningful
invariant: source cap stays below the pilot-arm budget so
source can never out-spend the arm itself).

The reviewer's two MINORs:
- *Strict-greater check at threshold (drift > 0.50)*: matches
  prereg language at line 919 ("drift > 0.50"). No off-by-one.
  Accept.
- *2-seed smoke can't validate seeds 44/45/46*: the LR-warmup
  mechanism is not seed-dependent (same `reset_transferable_
  optimizer_state()` call path, same warmup scheduler), so the
  mechanism probe transfers across seeds. The empirical source-
  cap risk is the only seed-specific concern, and v5.2
  addresses it. Accept with operational note: monitor source
  crystallization in real time during pilot #2.

**Operator-facing addition (no code change yet):** during pilot
#2 launch, the operator should grep the live log for
`source_crystallized=False` after each source arm completes;
if observed, pause and bump `SOURCE_MAX_ENV_STEPS_DEFAULT`
again (200k as the second relief valve) before resuming the
affected seeds. Pre-registered as "Bug E v6" remedy if needed.

## Aggregate v5.2 disposition

- **Code/test/prereg:** v5.1 (constant hoist + doc) + v5.2
  (source-cap raise + test relaxation) committed atomically
  on top of v5. Test count unchanged at 372 passed / 15
  skipped.
- **Decision rules unchanged at primary §8 threshold.** v5.2
  is purely an operational margin bump on a non-prereg
  constant (no preregistration amendment needed).
- **Pilot #2 unblocked** once the v5 smoke (re-run on this
  commit's `pilot_run.py`) passes
  `python -m scripts.smoke_verdict smoke_bug_e_v5.json`
  with exit code 0.
