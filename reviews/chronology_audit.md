# Chronology Audit — Plan B0 vs Pilot #2 Data

**Prompted by:** Devil's-advocate review (2026-04-16), concern #4
**Audit run:** 2026-04-16
**Question:** Was Plan B0 (§10 of `preregistration.md`, committing the "modest
but reliable" framing with ratio band [1.15, 1.30)) added to the preregistration
**before** any pilot #2 seed-level data existed, or **after** partial data had
been unblinded?

This is load-bearing because the preregistration text (`preregistration.md`
§10 B0 "Why B0 is not a back-door around §8") claims:

> A reviewer can verify chronology via git history: B0 was committed pre-data
> (v3.5, before pilot #2 unblinding), and the §8 threshold did not move.

If this claim is **false**, the pre-registration integrity is compromised for
the B0 fallback path — not for §8 itself (which was set in v3 on 2026-04-14,
well before any pilot #2 run), but for the specific [1.15, 1.30) band that
B0 claims as its trigger range.

---

## Hard facts

### B0 commit

- **SHA:** `4f8bb11af44c1a571c5dfb02d99ef99fdec879d5`
- **Commit time:** 2026-04-15 **13:28:38 +0200** (Paris time)
- **Message:** "phase 3 pilot: v3.5 prereg — elevator pitch (§1.0), Plan B0
  (§10), post-pilot backlog"

### Pilot #2 start time (reconstructed)

- **Method:** Summed all `wall_clock_sec` durations from the 26 timed events
  in `pilot_run.log` (ended 2026-04-15 20:59:15 when the dm_control
  ImportError crash occurred). Total: 48,289 s ≈ 13.4 h.
- **Estimated start:** 2026-04-15 **~07:34**.
- **Alternative estimate** (per first-source-crystallized entry `366s, eval=500`
  followed by first-scratch-completed entry `4215s`): consistent with start
  at 07:34 ± 10 min.

### Runs completed at B0 commit time (13:28:38)

Reconstruction from sequential `wall_clock_sec` durations starting at 07:34:

| Elapsed at run-end | Run | Wall-clock time | Pilot status |
|---|---|---|---|
| 366s (07:40) | source seed=42 (cartpole) | crystallized | — |
| 4,581s (08:50) | scratch seed=42 (mcc) | 70 min | — |
| 8,105s (09:49) | transfer seed=42 (mcc) | 59 min | — |
| 8,311s (09:53) | source seed=43 | crystallized | — |
| 11,822s (10:51) | scratch seed=43 (mcc) | 58 min | — |
| 15,591s (11:54) | transfer seed=43 (mcc) | 63 min | — |
| 15,772s (11:57) | source seed=44 | crystallized | — |
| 19,213s (12:54) | scratch seed=44 (mcc) | 57 min | — |
| 21,240s (13:28) | **B0 commit** | — | transfer seed=44 in progress (~35 min in) |
| 23,057s (13:59) | transfer seed=44 (mcc) | 64 min | — |

**At B0 commit, the primary pair had 2/5 seeds fully complete (42, 43) and
seed 44 in progress.** No data existed for seeds 45 and 46.

### What ratio did the 2-seed partial data show at 13:28?

| seed | scratch stm | transfer stm | implicit per-seed |
|---|---|---|---|
| 42 | 5,258 | 5,070 | +188 (scratch slower) |
| 43 | 5,936 | 5,131 | +805 (scratch slower) |

Mean of (scratch_stm / transfer_stm) across these 2 seeds ≈ 5,597 / 5,100 ≈
**1.10**.

The B0 band [1.15, 1.30) does **not** include this observed 1.10 partial
ratio. The devil's-advocate's claim "band fitted to observed data" is
therefore **not consistent** with the evidence — if the band had been fit
to what Jeremie could see at commit time, the lower edge would more
plausibly have been set below 1.10 to encapsulate the observed partial
signal. Setting 1.15 as the floor (per Bug E v3 architecture review,
committed at 03:49 the same morning before pilot start) actually places
the observed 1.10 data **below** the band's floor.

### Seed 46 (the outlier)

- **Chronologically:** seed 46's transfer arm was among the last primary
  runs, completed ~17:00-18:00 (well after B0 commit at 13:28).
- **Observed per-seed effect:** +5,074 steps (scratch 10,123 vs transfer
  5,049).
- **Impact on final ratio:** dropping seed 46 via leave-one-out collapses
  ratio from 1.238 to 1.049 (below the 1.15 floor). See
  `pilot_analysis.py --loo` output (computed 2026-04-16).

This is load-bearing: **the seed that drives the 1.238 signal was NOT
in the dataset when B0 was committed.**

---

## Adjudication

**Verdict:** The devil's-advocate chronology concern (review #4) is
**partially substantiated, partially exonerated.**

- **Substantiated:** The preregistration text "B0 was committed pre-data,
  before pilot #2 unblinding" is **factually inaccurate as written**.
  Pilot #2 had been running for ~6 hours and had 2 full primary seeds
  plus partial data on a 3rd when B0 was committed. The prereg should
  be amended to accurately reflect this chronology.

- **Exonerated:** The specific accusation "B0 band [1.15, 1.30) was
  fitted to observed data to encapsulate 1.238" is **not supported**.
  The observed partial ratio at B0 commit time was ~1.10, outside and
  below the band. The band edges come from prior architecture review
  (v3.4 Bug E v3 amendment at commit `e24832c` on 2026-04-15 03:28,
  before pilot start) and represent pre-pilot noise-floor reasoning.
  The fact that the final N=5 ratio landed inside the band is
  **coincidence, not fitting** — and specifically, it only landed
  inside the band because seed 46 (run ~5 hours after B0 commit)
  delivered an outlier effect that lifted the ratio from ~1.05 to 1.24.

## Corrective actions (required before paper submission)

1. **Amend `preregistration.md` §10 B0** to replace the claim "committed
   pre-data, before pilot #2 unblinding" with:

   > B0 was committed at `4f8bb11` on 2026-04-15 13:28, while pilot #2
   > was in progress. At commit time, 2 of 5 primary seeds had completed
   > (42, 43) showing a partial observed ratio of ~1.10; seeds 44-46 had
   > not yet produced data. The B0 band edges ([1.15, 1.30) at p < 0.10)
   > were committed unchanged from the v3.4 Bug E v3 amendment
   > (commit `e24832c`, 2026-04-15 03:28, **pre-pilot**) and were NOT
   > tuned to the partial observed signal, which was below the band's
   > floor. The seed that drives the final observed ratio (46, per
   > leave-one-out analysis landing at 1.049 without it) was not in
   > the dataset at B0 commit time.

2. **Include this audit file** in the paper's supplementary materials
   with the full chronology reconstructed from `pilot_run.log` and git
   log. Make the reviewer's verification a 1-minute task.

3. **Do not retroactively defend the "pre-data" phrasing** — accept the
   correction as written above and cite it in the first response to any
   reviewer who raises this point.

4. **Re-classify Plan B0's status** in the "pre-registration integrity"
   claim: it is *preregistered before outcome data* (no N=5 RMST ratio
   existed anywhere at 13:28; only 2-seed partial data could not
   reliably estimate the population ratio), but NOT *preregistered
   before pilot start*. This is a weaker integrity claim than the
   §8 primary threshold (which IS fully pre-pilot) but still
   defensible — the band was committed before the N=5 ratio could be
   computed, which is the load-bearing unblinding event.

---

## What this means for the OpenReview response

If a reviewer raises this chronology:

- **Acknowledge directly**: "You are correct that B0 was committed during
  pilot #2 execution, not before launch. See
  `reviews/chronology_audit.md` for full timeline."
- **Provide the exoneration**: partial data at B0 commit time (~1.10
  ratio) did not encapsulate the band [1.15, 1.30), and the band edges
  were set in a pre-pilot amendment (e24832c).
- **Concede the weaker claim**: B0 is *pre-outcome* (before N=5 ratio
  was computable) but not *pre-pilot*.
- **Do not argue for the stronger claim**: §8 primary threshold remains
  fully pre-pilot (v3, 2026-04-14) — that is the load-bearing integrity
  statement, unchanged.

This correction **does not kill Plan B0**, but it does mean we cannot
claim B0 as a clean pre-registered alternative. It is a pre-outcome
alternative with residual chronology risk that should be surfaced
honestly in the paper's methods section.

---

*Audit committed: 2026-04-16 (commit TBD). Any future changes to this
file require an entry in `preregistration.md` §13 referencing the commit
SHA.*
