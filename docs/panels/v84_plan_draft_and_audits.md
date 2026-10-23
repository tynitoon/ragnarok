# v84 plan draft (synthesizer) and the three audits (workflow wf_a2b93ac1-81e, unedited)

# v84 plan: carry the questions, not the answers

**Pipeline.** A zero-frame mechanism gate (G0, on the A1 logs) runs alongside the build of a repaired fresh climber (V2). One carried object then goes to live climbs against tuned-default (T), single-best-process (S), placebo (P), dose and same-game ceiling (K\*) controls. The sealed pool is opened at most once.

*Synthesizer note.* I read and grepped files only. I ran no Python, emulator or GPU job and edited no file. I did not open anything under `F:\dev\ragnarok_v83\craft_v6_out\v83\explore\` and did not list its contents. Section 3 (G0) is written to be frozen and committed today, 23 October 2026, before anyone reads the v83 exploratory output.

---

## 0. Summary

- **What v84 decides first.**
  - G0, zero frames, about 5 hours of CPU, verdict on day 2. On the 82 existing A1 streams, does a zero-mean prior covariance Σ, built from 40 other games, make a fresh consequence learner (the GOOD/BAD heads) reach its own end-of-stream predictive skill on a new game from fewer frames of the same experience?
  - It must beat the best of three controls on each unit: a tuned isotropic learner F, a spectrum-matched permuted-label placebo P, and a spectrum-matched input-PCA prior U.
  - It must also hold on the late part of the curve.
  - The same-game ceiling K\* must show that there is room.
- **Built in parallel: V2.** This is a probe-free Go-Explore climber with frame-code cells. It keeps playing after a life loss, plays the bar's four button processes, and is scored on cumulative true rises. It goes through a cheap exploratory "does the fix work" gate (G1a), then a claim-bearing room gate (G1b) on all 41 opened games at A1 strength.
- **One carried object goes live.**
  - **Route Σ:** if G0 passes, carry Σ, plugged into V2's GOOD/BAD steering heads. Because the prior has mean zero, the steered arms are byte-identical to the fresh arm until the first refit that sees both classes.
  - **Route S:** if G0 fails, the fallback is a "search sense" (the P1 controller merged with the P3 expansion critic), also zero-mean. It must pass the same knowledge-clock gate on V2's own room logs.
  - Live gates in order: same-game WORTH (K\*), then a development rehearsal against T, S, P and dose, then the sealed test once.
- **Measures.**
  - Primary claim (play): the matched-rung climb, averaged over rungs, counted from each arm's own first rung, with a late version from the second rung.
  - Separate mechanism verdict on the sealed games: the same knowledge clock, run offline on the sealed eligibility streams.
- **Odds (mine, honest).**
  - G0 pass: about 0.25.
  - V2 has room on at least 13 of 35 development games: about 0.2.
  - Sealed FASTER-PARTIAL or better: about 1% (range 0.5–2%); FASTER-2×: about 0.3%.
  - A clean, decisive answer to G0 by day 2: about 0.95.
- **Budget.** 14 calendar days or 2.4G billed frames, whichever comes first. Development is capped at 1.45G. CPU only, 3 workers at 1.2 GB or less each. GPU: 0 h.

---

## 1. Ground facts used (checked against files)

| Fact | Value | Source |
|---|---|---|
| A1 verdict | KC-room, 6/35 dev room (alien, amidar, bank_heist, ice_hockey, jamesbond, qbert); calibration breakout, ms_pacman; σ̂² 1.165, se_proj 0.480 | `room_verdict.json` |
| A1 streams | 82 npz (41 games × seeds 9131, 9132), code 2b22e94, unsteered (`steer` null) | `room_*.json`, `room_job_v83.py:78-82` |
| Rows | 5,434,791 in total (lost rows included); per run 3,600 (krull 9132) to 99,103 (pong 9132) | `n_rows` in the run JSONs |
| Row content | row_view (packed 4×64×64), row_screen, row_lost, row_mx/my, row_lives, row_slot, row_it, row_t, row_rise_lag, row_drop_lag, row_avail; iter_billed; rec_actions, base_state | `logs.py`, `room_job_v83.py` |
| Features | 427 (`views.features`), labels from `logs.labels` (GOOD: rise ≤ 24 steps; BAD: lives decrease ≤ 40 steps, censoring rule of note 6) | `views.py`, `logs.py`, `promise.py` |
| Throughput | median wall about 666 s per 2M billed (about 3.0k fps per worker); peak commit 857–1,246 MB; entry 82k–194k frames | run JSONs |
| Probe share | 26.7–26.8% of 164M; ≥ 64% on 7 games | A1 result |
| Root-state sharing | Identical entry frames on both seeds for several games (assault, fishing_derby, freeway, breakout, video_pinball). Streams can share early states, so G0 deduplicates test rows. | run JSONs |
| Exploratory analysis | `scripts/explore_promise_v83.py` tests the full-stream AUC of cross-game mean heads (W₋g) on the 8 room games. Its store and outputs live in `explore/` (sealed). Two judges saw one log line ("good alien False"). | judges' notes, script |
| Seeds in use | 9130–9256 (v83), 9101–9111 and 9201–9211 (v82), 8101–8106 | `pools_v83.py`, protocol §9 |

**Contamination record.**
- The single exploratory line above was seen by two judges before G0 was frozen; it is recorded here.
- G0's thresholds come from P5 (phase 1). They are hardened here and frozen before `explore_promise.json` is read.
- G0 builds its own feature store and never reads `explore/promise_store`.
- The exploratory result may be read only after `g0_verdict.json` is committed. It is then reported with an "exploratory" label.

---

## 2. Shape of v84

```
Day 0-2   G0 knowledge clock (A1 logs, 0 frames) -----> route Σ (PASS) or route S (any KC)
Day 1-3   V2 build ---> G-build tests ---> bars ---> G1a fix-works (exploratory, 48M)
Day 4-5   G1b room A1' (41 games x 2 seeds x 4M, claim-bearing feasibility)
Day 5     offline: sources for Σ / search sense, T schedule, S choice; G0' (report) / G0-S (route S gate)
Day 6-7   G2 WORTH live (V2, F, K*)
Day 7-9   G3 dev rehearsal (X, P, T, S, dose, U) [+ freeze 2]
Day 9-10  adversarial review + sealed freeze (hashes)
Day 10-13 B0 sealed bars + eligibility -> sealed knowledge clock -> B1 sealed live arms
Day 13-14 verdicts, report, handoff
```

Rules that hold throughout:
- Only one live route per v84, and only one object reaches the sealed pool.
- Every carried object has mean zero, so a head start is impossible by construction. A prefix-identity test certifies this.

---

## 3. G0: the knowledge clock (freeze today)

### G0.1 Question
On a held-out development game, the logistic GOOD/BAD learner learns from one A1 stream in order and is scored on the other stream. Does a zero-mean prior N(0, κΣ_X) reach fractions of the fresh learner's end-of-stream AUC in fewer billed frames? Σ_X encodes which ego-centric, action-rotated directions mattered on 40 other games. It must do better than the per-unit best of F (isotropic, tuned), P (permuted-label Σ, spectrum-matched) and U (input-PCA Σ, spectrum-matched). This must hold over the whole curve and over its late part, and the same-game ceiling must show room.

### G0.2 Data and integrity
- **Inputs.** `F:\dev\ragnarok_v83\craft_v6_out\v83\room\room_{game}_{seed}.npz` and the matching `.json`, 41 games (35 DEV + 6 CALIBRATION as in `pools_v83.py`) × seeds 9131, 9132.
  - Every file is verified against `a1_manifest.json` SHA-256 before use. Any mismatch is an Interruption.
- **Never opened.** Any path containing `craft_v6_out\v83\explore`. The loader asserts this, and a test greps the G0 code for that path.
- **Not inputs.** Bars, `room_verdict.json`, the true score, RAM, game names inside the learner. Scripts hold names for bookkeeping only; learner functions receive arrays.
- **Outputs.** `F:\dev\ragnarok_v84\craft_v6_out\v84\g0\` (`store/`, `sources.npz`, `g0_kappa.json`, `g0_units.json`, `curves.npz`, `g0_verdict.json`), each with provenance (code SHA, clean tree, input hashes).

### G0.3 Rows, features, labels, clock
- **Kept rows (per stream).** Rows with `row_lost == False` (v83 build note 16).
  - If more than 24,000, draw 24,000 uniformly without replacement with `default_rng([9701, game_index, seed])`. `game_index` is the position in DEV + CALIBRATION.
  - Keep the original row order.
- **Features.** v83 `promise.feats(view, screen, False, (mx, my), lives, slot, (210, 160))`, giving 427 values, stored as float16.
  - The v84 copy must be byte-identical to v83 on 1,000 random rows (test).
- **Labels.** v83 `logs.labels`:
  - GOOD: lag = rise_lag, H = 24.
  - BAD: lag = drop_lag, H = 40, stops_on_event = True.
  - steps = 100.
  - A value of −1 (censored) is ignored by that head.
- **Clock.** c_i = `iter_billed[row_it[i]] − entry.entry_frames`: billed exploration frames at the end of the rollout that produced the row, which is when its label becomes known.
  - It must be non-decreasing; otherwise Interruption.
- **Blocks.** block_i = `row_it[i] // 50`.

### G0.4 Splits
For each game g, two directions d ∈ {(A=9131, B=9132), (A=9132, B=9131)}.
- **Learning stream.** All kept rows of A, in order.
- **Test set (per head).** Kept rows of B in odd blocks, uncensored for that head.
  - Rows whose exact tuple (view bytes, screen bytes, slot, lives) occurs in A are removed (blake2b hash); the removed fraction is reported.
  - If more than 5,000 remain, draw 5,000 uniformly with `default_rng([9702, game_index, seed_B, head])`.
  - The test set is fixed and identical for every arm (common random numbers).
- **K\* source.** Kept rows of B in even blocks, uncensored, deduplicated against that test set.
- **Σ sources for g.** All kept rows of both streams of the 40 other games.
- **κ tuning.** The 6 calibration games only. Units are scored on the 35 development games only.

### G0.5 Learner (shared by every arm, and later by the live heads: `ragnarok/agent_v84/consequence.py`)
- **Standardisation S₋g.** Mean and SD over all kept rows of the 80 streams of the other 40 games, with SD floored at 0.05 (v83 `standardise` semantics). Identical for every arm on g.
- **Fit at checkpoint n.** Minimise over (w, b):

  Σ_{i ≤ n, uncensored} [softplus(z_i) − y_i·z_i] + (1/(2κ_a))·wᵀ Σ_a⁻¹ w, with z_i = b + x̃_iᵀw.

  - The loss is a sum, so the prior's weight fades as 1/n.
  - The intercept is free.
  - Implemented as x' = Σ_a^{1/2} x̃ with an isotropic penalty on v = Σ_a^{-1/2} w.
  - L-BFGS-B, at most 200 iterations, float64 accumulation, warm-started from the previous checkpoint (from 0 at the first fit).
- **Both-classes rule.** While the uncensored rows seen so far lack either class, the head is intercept-only. Its test scores are then constant and its AUC is exactly 0.5 for every zero-mean arm (the certificate).
- **Checkpoints.** n_k = ⌈128·2^{k/4}⌉ for k = 0, 1, … while n_k < n_A, plus n_A itself.
- **Score.** Test AUC (Mann–Whitney, ties counted half) at every checkpoint.

### G0.6 Arms and priors
Every Σ is normalised to trace 427.

| Arm | Σ | Role |
|---|---|---|
| F | I | Fresh learner, tuned κ |
| X | norm[(1/G)·Σ_{h≠g} û_h û_hᵀ + τ²I], û_h = w_h/‖w_h‖ in S₋g, τ² = 0.1/427 | Carried hypothesis space |
| P | Eigenvectors of the same construction from **permuted-label** source heads, with X's eigenvalues (rank-matched) | Placebo (decision) |
| U | Eigenvectors of the equal-game-weighted mean covariance of the source games' S₋g-standardised rows, with X's eigenvalues | Input statistics only (decision) |
| K\* | norm[û_g,d û_g,dᵀ + τ²I], û from a head fitted on the K\* source rows of the same game | Same-game ceiling (WORTH) |
| X-mean | Prior N(w̄, κI), w̄ = mean of the source w_h in S₋g | Report only: what a head start looks like |
| X_diag | diag(Σ_X) | Report only: feature relevance versus directions |
| X_8 | Σ from 8 source games, 3 draws (`default_rng([9704, g])`), mean λ over draws | Report only: dose |
| P_shift | v83 circular-shift placebo (shift of at least 10 rollouts, seed 9707), spectrum-matched | Report only (weaker placebo) |

- **Source heads w_h.** For each of the 41 games and each head:
  - Fitted on all kept, uncensored rows of both streams of h, with an intercept per stream, ridge 1e-4 on the mean loss, at most 300 L-BFGS iterations, in h's own pooled standardisation S_h.
  - Mapped to S₋g by w ↦ w ⊙ (sd₋g / sd_h). This is exact up to the ridge; it is declared.
  - A source head needs at least 30 positives and 30 negatives, otherwise it is left out (listed).
  - Permuted heads use the same recipe with labels permuted uniformly among the uncensored rows of each stream (`default_rng([9703, game_index, seed, head])`).
- **Dropped from P5.** X-rep (no choice of carried object at G0) and the savestate exam (see §13).

### G0.7 κ tuning (before any development unit is scored)
- **Grid.** κ ∈ {4⁻⁵, 4⁻⁴, 4⁻³, 4⁻², 4⁻¹, 1}, tuned per arm (F, X, P, U, K\*, and also X-mean, X_diag, P_shift).
- **Tuning curves.** The 6 calibration games × 2 heads × 2 directions, wherever the learning stream and test set each have at least 30 rows of each class. Σ for calibration game c is built from the other 40 games, and K\* from c's own even blocks.
  - Declared: this uses development games' source heads, but never their curves or test rows.
- **Criterion.** Mean over the tuning curves of the mean AUC over checkpoints (an area under the curve on the log-row axis). Highest wins; exact ties go to the smaller κ.
- **Commit.** `g0_kappa.json` is committed with its hash before any development curve is computed.

### G0.8 Units (decided on F alone; `g0_units.json` committed before any X/P/U/K\* development curve)
A unit is a (development game, head). It is eligible only if, **in both directions**:
- the learning stream has at least 30 positive and 30 negative uncensored rows;
- the test set has at least 30 of each class;
- AUC_F,end (the mean of F's AUC at the last 3 checkpoints) is at least 0.60 for GOOD or 0.65 for BAD;
- AUC_F,end − AUC_slot ≥ 0.03, where AUC_slot is the test AUC of the per-slot positive rate fitted on the whole learning stream (unseen slots get the stream rate).

### G0.9 Metric: time to a fraction of the novice's end skill
- **Levels.** L_q = 0.5 + q·(AUC_F,end − 0.5) for q ∈ Q = {0.25, 0.375, 0.5, 0.625, 0.75}, per (unit, direction).
- **Crossing time N_a(q).**
  - Find the first k with AUC_a(k) ≥ L_q **and** AUC_a(k+1) ≥ L_q (held for two checkpoints).
  - If k ≥ 1, interpolate linearly in log-clock between checkpoints k−1 and k.
  - If k = 0, N = c(n_0).
  - Floor: 5,000 frames.
- **Censoring (always against X).** Carried arms (X, K\*, X-mean, X_diag, X_8) that never cross get N = 2·c_end. Controls (F, P, U, P_shift) get N = c_end.
- **Primary.** λ_a = mean over q ∈ Q of log N_a(q).
- **Late.** For q ∈ {0.5, 0.625, 0.75}, D_a(q) = N_a(q) − N_a(0.25).
  - Censoring: a carried arm missing N(q) gets 2·c_end. A control missing N(q) but with N(0.25) gets c_end − N(0.25). A control missing N(0.25) gets 5,000.
  - λ_late = mean of log max(5,000, D).
- **Aggregation.** Per unit, average over the 2 directions. Per game, ℓ_g = mean over its eligible units.
  - The comparator C\* is chosen per unit as argmin over {F, P, U} of λ (separately for λ_late).
  - R = exp(mean_g ℓ_g), with each game weighted equally.

### G0.10 Statistics
- One-level **game-cluster** percentile bootstrap of ℓ_g: 2,000 resamples, `default_rng(9705)`, 2.5% and 97.5%. Two heads of one game never count as independent.
- Reported: one-sided sign-flip permutation p (exact up to 2²⁰, otherwise 10⁵ Monte-Carlo draws), the leave-one-game-out minimum, the per-level profile R(q), and Hodges–Lehmann.
- PASS is an intersection of conditions, so multiplicity does not inflate it.

### G0.11 Pass, kill, decision

| # | Condition |
|---|---|
| (i) | At least 12 eligible units spanning at least 8 development games |
| (ii) | R(K\*:F) ≥ 2.0 with lower bound ≥ 1.25 |
| (iii) | R(X : per-unit best of F, P, U) ≥ 1.5 with lower bound > 1.1 |
| (iv) | Lower bound of R_late(X : best of F, P, U) > 1 |
| (v) | Ceiling guard: mean over units and directions of (AUC_X,end − AUC_F,end) ≥ −0.01 |
| (vi) | Integrity: hashes match, prefix certificate (AUC exactly 0.5 for every zero-mean arm while intercept-only), Σ_P and Σ_U eigenvalues equal Σ_X's (relative error ≤ 1e-9), traces = 427, sealed-path guard, clean code |

- **PASS (all of i–vi).** Route Σ: Σ is the v84 carried object for live arms.
- **KC-knowledge-room (i fails).** The A1 streams do not hold enough learnable consequence structure to measure learning speed.
- **KC-knowledge-worth (ii fails).** Even the game's own structure does not make this learner 2× faster.
- **KC-knowledge-speed (iii, iv or v fails).** Carried consequence structure does not make the learner faster than tuned controls.
  - Caption GENERIC-SHRINKAGE: lower bound of R(X:F) > 1, and lower bound of R(P:F) or R(U:F) > 1.
  - Caption FIRST-RULES-ONLY: (iii) passes and (iv) fails.
- **Interruption (vi fails).** No verdict; fix the code and rerun.
- **Any KC.** Route Σ is closed for v84 live arms and route S becomes the candidate (§5.2). The negative is committed and kept.
- **Status of any G0 outcome.** Development evidence on burned logs. It is never a competence or sealed claim.

### G0.12 Written now (no effect on the verdict)
- If X helps, R(q) falls with q (P5 predicted about 3× at q = 0.25 and about 1.5× at q = 0.75), and BAD carries more than GOOD.
- X-mean shows AUC above 0.5 at the first checkpoint (a head start) and a smaller late ratio than X.
- R(X_8) < R(X).

### G0.13 Tests before scoring (all must pass; no real-data curve is looked at before this)
1. Features and labels byte-identical to v83 on 1,000 random rows.
2. Prefix certificate on real streams.
3. Spectrum and trace identities.
4. Sealed-path guard.
5. Deduplication applied.
6. **Synthetic controls.** 41 Gaussian "games", 24k-row streams.
   - Labels drawn from heads lying in a shared 5-dimensional subspace: the pipeline must give R(X : best control) ≥ 2.
   - Independent random heads: R must fall in [0.8, 1.25].
   - Failure means a code bug, fixed before any real-data curve.
7. A one-agent code review of the G0 scripts before step 4 of §G0.14.

### G0.14 Run order and cost
1. Store: about 2M rows, 1.7 GB float16 memmap, 30–60 min on 2 workers.
2. Sources and all Σ: about 330 head fits, 41 leave-one-game-out PCAs, about 30 min.
3. κ tuning, then commit `g0_kappa.json`: 864 curves, about 1.2 h.
4. F curves on development games, then commit `g0_units.json`.
5. All other arms: about 1,260 curves, about 1.5–2 h on 2 workers.
6. Verdict, then commit.

Total: about 5 h wall, 0 emulator frames, 0 GPU, at most 0.8 GB RAM per worker (`OMP_NUM_THREADS=2`), about 2 GB disk.

---

## 4. The fresh climber V2 (every live arm shares it)

**Reused from v83:** SenseEnv (billed ledger, each restore billed 60 frames, evaluator Trace) and the entry (`Body.enter`: background, SignatureAdapter, seeding of "me", passive tracker). The entry is identical for all arms.

| Lesson | Change |
|---|---|
| 1. Climb first | G1a and G1b come before any carried arm; cap 4M (33 of 58 A1 runs were still improving after 1.5M) |
| 2. Bar processes | Four processes, each a bar mode: **Q1** a new uniform slot every step; **Q2** held, repeat 0.95 (v83's V); **Q3** held, repeat 0.75; **Q4** long hold, repeat 0.995, starting from the inherited button with probability 0.5, else a uniform slot. Buttons are drawn over distinct canonical slots. Thompson bandit on ADV, Beta(1 + ADV, 1 + not-ADV), discount 0.995 per expansion, 20% uniform floor, propensities logged |
| 3. Probes | **None after entry.** "Me" comes from `Body.track` only (tracker update or colour reacquire). Probe share and me-lost fraction are reported |
| 4. Lives rule, reachable rungs | A life loss archives the post-loss state (lives are in the key) and play continues; a rollout stops at its second loss or at game end. The ladder uses **cumulative true rises** (§6); only rungs reached by both fresh seeds are measured |
| 5. Progress axis | Cell = (lives, s_bin(S), code_r). code_r is the luminance frame (0.299R + 0.587G + 0.114B), block-averaged with `np.add.reduceat` edges at floor(i·H/h), quantised to d levels. R = {(8,6,4), (11,8,8), (16,12,8), (22,16,8)} as (w, h, d); all four codes are stored. Every 250 expansions, the r whose distinct-key count is closest in log to clip(30√n, 64, 8000) is chosen and the archive is re-keyed (the better cell wins: S, then lives, then shorter depth) |
| 6. Noise | λ averaged over rungs; 6 seeds for X, F and T in the sealed test; winsorised σ̂²; an F-versus-F instrument line |

- **Selection.** w_i = (1 + chosen)^(−1/2) · 2^((sbin − sbin_max)/2) · 2^(−(L_max − L_i)) · (2 if the cell was added or improved in the last 100 expansions).
  - With probability 0.25, a frontier draw replaces it: uniform among the cells at sbin_max that have the most lives.
- **Rollouts.** 100 steps, extended by 50 at each new archive-best S, up to 400. Cap 20k cells, v83 eviction rule.
- **ADV (label for the bandit and the search sense).** The rollout created a cell above its start cell's sbin, or a new cell in the top sbin.
- **Logs.**
  - The v83 RowLog every 4th step. `labels` uses steps = the rollout's final planned length.
  - One expansion row per rollout with the search-sense features (§5.2), process, ADV, billed frames and propensity.
- **G-build tests.**
  - The agent package imports nothing from evaluation and contains no game names (AST test).
  - The ledger never decreases; each restore is billed 60.
  - An **equivalence harness:** with menu {Q2}, the v83 cell key, raw-index draws, no post-loss archiving and no extension, V2 reproduces V's rows, iterations and cells exactly on one smoke game (seed 9700, 50k frames).
  - **Prefix identity:** a steered arm with zero-mean Σ equals V2 byte for byte until the first refit that sees both classes.
  - Commit at most 1.0 GB unsteered and 1.2 GB steered over a 4M-frame smoke; throughput logged.
  - Two failed fix rounds give KC-build.

---

## 5. Carried object and arms

### 5.1 Route Σ (if G0 passes)
- **Heads.** The v83 GOOD/BAD heads over 427 features, refit every 250 expansions on a reservoir of at most 40k rows (float16 storage).
  - MAP fit by `consequence.py` (the same code as G0), standardisation S_src from the source games, κ_a from `g0_kappa.json`.
  - Both-classes rule; BAD stays inactive until a life loss is sensed.
  - Rows where "me" is lost are dropped; while me is lost, steering falls back to V2's own law. The dilution counts against X.
- **Steering.**
  - **Cells:** draw 16 candidates with V2's weights; pick one with probability ∝ exp(β·clip(z, −2, 2)), β = 1, where z is the robust z-score of u(c) = max over slots of [P(GOOD) − λ·P(BAD)], λ = 1, computed on the cell's archived view. Frontier draws are never steered.
  - **Buttons:** at rollout start and at each button change in Q2–Q4, use (1 − α)·uniform over slots + α·softmax(logit GOOD − λ·logit BAD), α = 0.5. Q1 is never steered.
  - With inactive heads, this is exactly V2.
- **Σ sources (declared now).**
  - Built with the G0 recipe from the A1 (V) and A1′ (V2) streams of the source games, 4 streams per game, equal game weight.
  - Leave-one-game-out within development; all 41 games for the sealed test.
  - For K\*: the game's own A1′ rows (seeds 9712–9713, which no arm uses).

| Arm | Definition | Where |
|---|---|---|
| V2 | No heads, flat bandit | WORTH, rehearsal |
| F | V2 + isotropic heads (κ_F) | Everywhere |
| T (tuned defaults) | F + a clock-only process schedule learned on development: per process × clock bin b = min(5, ⌊log2(1 + billed/125k)⌋) × any-rise flag, prior Beta(n0·p̂, n0·(1 − p̂)) with n0 = 50 and p̂ = equal-game mean ADV rate in the A1′ logs (leave-one-game-out in development) | Everywhere |
| S (single best process) | V2 with only the process q\* that maximises equal-game ADV per billed frame in the A1′ development logs (leave-one-game-out); no bandit, no heads | Everywhere |
| X | F + Σ_X (κ_X) | Everywhere |
| P | F + Σ_P (permuted labels within each run, spectrum-matched, κ_P) | Everywhere |
| U | F + Σ_U | Development, 2 seeds, report only |
| X_8 | F + Σ from 8 source games (fixed draw, seed 9704) | Development 2 seeds; sealed 2 seeds, caption only |
| K\* | F + same-game Σ | WORTH only |

X does **not** include T's schedule. FASTER requires X to beat the per-game better of T and S, so the carried content has to be worth more than designer-equivalent tuned defaults.

### 5.2 Route S, the fallback (only if G0 gives any KC)
- **Object.** A zero-mean Σ_s over the slopes of a logistic expansion critic on ADV, with a free intercept per game. Features are frozen at V2's code freeze, before A1′ starts:
  - S-rank, lives gap, log depth rank, log(1 + chosen), smoothed yield and parent yield, log expansions since the cell was added, log steps since the last rise, after-loss flag, coarse-code rarity, share of pixels changed, log occupancy of its sbin;
  - process one-hot (4) and the first button's slot class;
  - budget share, ADV rate over the last 200 expansions, life-loss rate, stagnation, any-rise flag, log frames since the last record, new-cell rate;
  - adapter signature (number of movement slots, whether a spawn slot exists);
  - interactions process × {lives gap, stagnation, loss rate, S-rank}.
- **G0-S (offline, on A1′ expansion logs, day 5).** The G0 machinery with:
  - the learning stream = one seed's expansions; test = the other seed's odd 50-expansion blocks; K\* = even blocks;
  - controls {F, P (ADV fully permuted within each run), U, T_clock (Σ from source heads fitted on the clock and run-context features only)};
  - units = development games (a single head), so condition (i) becomes at least 12 games;
  - all other thresholds identical to §G0.11, frozen now.
- **Live (if G0-S passes).** P3 steering: draw 32 cells with V2's weights, score all (cell, process) pairs, pick ∝ exp(1.5·clip(z, −2, 2)), mixed at α = 0.5 with V2's own draw; refit every 100 expansions.
  - Arms as in §5.1, with T = the clock-only critic prior and S unchanged.
- **Rules.** G0-S is computed and reported on day 5 whatever happens. Route S goes live only if route Σ was closed at G0, and there is no switch of route after day 5.

---

## 6. Live yardstick and statistics (evaluator side)

- **Bars B′** (seed 9706 for development + calibration; 9730 sealed; 9740 second round).
  - Four modes, 10 episodes each, all lives, sticky 0, 0–30 NOOPs, 7,000 steps, minimal action set, unbilled (v83 `bars_v83` plus the held-0.75 mode).
  - Modes: each constant button, per-step uniform, held 0.95, held 0.75.
  - Recorded in **cumulative true rises** per episode, and in true score.
  - B′ = max over modes of mean episode rises; u′ = max(0.5·B′, median positive rise); rungs M_k = B′ + 2^k·u′, k = 0..10.
- **T(M).** Billed frames from frame 0 (entry included) at the first clone whose cumulative true rises reach M.
  - Replay-confirmed for M1, M2 and M_k\* using the note-21 replay (one NOOP before restore). A mismatch is an Interruption.
- **Top rung k\*.** The highest rung reached by **both** room/eligibility seeds, fixed before any arm runs. A unit needs k\* ≥ 3. Every measured rung was therefore reached by the fresh learner on both seeds.
- **Per run.** λ = mean over k = 2..k\* of log max(5,000, T(M_k) − T(M_1)); λ_late is the same over k = 3..k\*, counted from T(M_2).
  - Censoring (v83 `climb`): a carried arm that misses a rung counts the cap (4M); a comparator counts cap − T(M_start); a comparator without the start rung gets the 5,000 floor.
- **Per game.** ℓ_g = mean λ(comparator) − mean λ(X). "The better of T and S" is chosen per game, and again in each bootstrap resample (v83 build note 17). r_g is clipped to [1/8, 8]; R = exp(mean_g ℓ_g).
- **Inference.**
  - Two-level bootstrap (games, then seeds within arm), 2,000 resamples: seed 9714 in development, 9739 sealed.
  - One-sided permutation p (X/F labels within each game, 10⁴ Monte-Carlo draws).
  - Leave-one-game-out minimum; head start HS = geometric mean of T_F(M1)/T_X(M1) (reported, never a claim); a per-rung ratio profile for each arm.
  - Also reported: true net score for games with penalties (pong, fishing_derby, ice_hockey, tennis and others), me-lost fraction, steering-active fraction, censoring.
- **Written now.** X's per-rung ratio falls with k: Σ's advantage fades once F's heads catch up, so R_late < R is expected. This is why the late clause is demanding.

---

## 7. Gates in order

| Gate | Data, arms, seeds | Pass | Kill and meaning |
|---|---|---|---|
| **G0** knowledge clock | §3 | §G0.11 | KC-knowledge-room / -worth / -speed: route Σ closed |
| **G-build** | Tests in §4, smoke 9700 | All pass | KC-build after 2 fix rounds |
| **G1a** fix works (exploratory label) | V2, seed 9710, cap 4M; atlantis, krull, centipede, video_pinball, gopher, name_this_game, zaxxon, bowling, seaquest, phoenix + controls ms_pacman, amidar | At least 6/10 reach M3; at least 8/10 reach M1; controls' T(M3) ≤ 2× V's A1 median re-scored on the v84 ladder; commit ≤ 1.1 GB; ≥ 1,500 billed fps | 4–5/10: one revision of named V2 constants (frontier probability, extension rule, T(n) constants, discount, floor), G1a′ on seed 9711, same rule. ≤ 3/10, or G1a′ fails: **STOP-fresh** ("budgeted savestate exploration with the senses does not climb on most Atari games at 4M"); no live arms; sealed pool stays closed |
| **G1b** room (claim-bearing feasibility) | V2, 41 games, seeds 9712–9713, cap 4M | Per game, both seeds: (a) ≥ 10 sensed rises in the first 300k; (b) H = min over seeds of best rises ≥ M3; (c) median T(M1) ≤ 1.5M; (d) median T(M_k\*) − T(M1) ≥ 300k. **Proceed** if ≥ 13/35 dev AND se_proj = √((σ̂²/3 + 0.09)/(17·n/35)) ≤ 0.23, with σ̂² the pooled (λ₁ − λ₂)²/2 over room dev games, winsorised at its 90th percentile. Label "climbs on most dev games" if ≥ 18/35 | **KC-room-84:** no live transfer arms; the cost map is kept. Option M (§15) |
| **G0′ / G0-S** (day 5, 0 frames) | G0 recipe on A1′ streams (report only) / §5.2 | G0-S: thresholds of §G0.11 | G0-S fails while route Σ is closed: v84 stops at feasibility |
| **G2** WORTH (live, dev) | Room dev games, at most 10 (drawn with 9714 if more); V2, F, K\* on seeds 9715–9718; cap 4M; evaluator stop at M_k\* | R(K\* : better of V2 and F) ≥ 1.75 with lower bound ≥ 1.2, and r_g ≥ 1.5 on ≥ 1/3 of units | **KC-worth-live:** "same-game structure is not this climber's bottleneck"; sealed stays closed. Flag KC-harm if R(F:V2) < 0.8 |
| **G3** development rehearsal | Same units; X, P, T, S on 9715–9718 (paired with G2); U and X_8 on 9715–9716 | **GO:** R(X:F) ≥ 1.4 with lower bound > 1; R(X : better of T,S) ≥ 1.2 with lower bound > 0.9; lower bound of R(X:P) > 1; lower bound of R_late(X:F) > 1 | One **freeze 2**, Σ knobs only: κ_X and κ_P × {¼, 4}, τ² ∈ {0.03, 0.3}, rank cut r ∈ {10, 20}. X and P rerun on 9719–9722; F, T, S and V2 reused. Second failure: **KC-dev**, sealed stays closed |
| **Sealed freeze** | — | 3-agent adversarial review; commit the SHA-256 of Σ_dev, Σ_P, Σ_8, the T schedule, the S choice, κ and code | — |
| **B0** sealed eligibility | Bars 9730; V2 on 9731–9732, cap 4M, the G1b room rule; `sealed_eligibility.json` + hash committed before any arm | At least 6 units | Fewer than 6: second round once (bars 9740, eligibility 9741–9742, arms 9743–9748); still fewer than 6: **NOT MEASURABLE** |
| **B-clock** sealed knowledge clock (mechanism, separate label) | G0 recipe on the sealed eligibility streams, Σ_dev, G0 κ (frozen); K\* reported | **UNDERSTANDS-FASTER:** ≥ 8 units on ≥ 6 games; R(X : best of F, P, U) ≥ 1.5 with lower bound > 1.1; lower bound of R_late > 1; ceiling guard. Committed before the B1 verdict is computed | NOT-REPLICATED (kept) |
| **B1** sealed live (once) | X, F, T on 9733–9738 (6 seeds); P, S on 9733–9736; X_8 on 9733–9734; cap 4M; stop at M_k\* | Verdict ladder below | — |

**B1 verdicts (the first that applies).**
1. **NOT MEASURABLE.**
2. **INCONCLUSIVE:** the instrument line F(9733–35) / F(9736–38) falls outside [0.67, 1.5], or at least 40% of X or F runs are censored, or the prefix certificate is violated.
3. **FASTER-2×**, all of:
   - R(X:F) ≥ 2.0, lower bound ≥ 1.3, p ≤ 0.025;
   - R(X : better of T,S) ≥ 1.4 with lower bound > 1;
   - lower bound of R(X:P) > 1;
   - R_late(X:F) ≥ 1.5 with lower bound > 1;
   - lower bound of R_late(X : better of T,S) > 1;
   - r_g(X:F) > 1 on ≥ 2/3 of units;
   - leave-one-game-out minimum of R(X:F) ≥ 1.5.
4. **FASTER-PARTIAL:** lower bounds of R(X:F), R(X : better of T,S), R(X:P) and R_late(X:F) all > 1, and p ≤ 0.025.
5. **TUNED-DEFAULTS:** lower bounds of R(X:F) and R(X:P) > 1, but not against T,S ("carried knowledge indistinguishable from designer tuning").
6. **GENERIC-STEERING:** lower bounds of R(X:F) and R(P:F) > 1, and lower bound of R(X:P) ≤ 1.
7. **HEAD-START ONLY:** HS ≥ 1.5 with lower bound > 1.
8. **NEGATIVE TRANSFER:** upper bound of R(X:F) < 1, or upper bound of R(X : better of T,S) < 1.
9. **NULL.**

Captions:
- "accumulates" if the lower bound of ℓ(X) − ℓ(X_8) > 0. It is underpowered at 2 seeds, which is stated.
- Capture ratio φ = log R(X:F) / log R(K\*:F), from development.

Power, written now: with about 9 units, 6 seeds and σ̂² ≈ 0.8, se ≈ 0.20. A true 2× passes FASTER-2× about half the time; a true 1.5× passes PARTIAL about half the time.

---

## 8. Pools and seeds

- **Pools.**
  - 35 development + 6 calibration games (all burned by A1): G0 (development = units, calibration = κ tuning), G1a, G1b, G2, G3, and the sources for Σ, T and S.
  - 24 sealed: opened only at B0, after the sealed freeze. The 7 hard-exploration games (montezuma_revenge, pitfall, venture, private_eye, solaris, gravitar, skiing) are assumed ineligible in G_proj = 17.
  - 9 second-round games: only if B0 finds fewer than 6 units, or under Option M.
  - Never opened: boxing, frostbite, riverraid, kangaroo. Excluded: enduro.
- **Seed block 9700–9749 (new).**
  - Grep today across all 9 worktrees: no 96xx or 97xx in any `.py`.
  - In docs, 97xx appears only as counts (9753, 9792, outside the block).
  - No output file name contains 96xx or 97xx in `ragnarok`, `ragnarok_v83run` or `ragnarok_v83` (the sealed `explore` folder was not listed).
  - Re-grep at the protocol commit; the `explore` folder is re-checked only after the G0 verdict.

| Seed | Use |
|---|---|
| 9700 | Smoke runs (never a measurement) |
| 9701 / 9702 / 9703 / 9704 / 9705 | G0 row thinning / test draw / placebo permutation / dose draws / bootstrap and p |
| 9706 | Development + calibration bars (4 modes) |
| 9707 | G0 P_shift (report only) |
| 9708–9709 | Offline reserve |
| 9710 / 9711 | G1a / G1a′ |
| 9712–9713 | G1b room runs, source logs, K\* sources |
| 9714 | Development unit draw, live placebo permutation, development bootstrap |
| 9715–9718 | G2 and G3 arms (paired across arms) |
| 9719–9722 | Freeze 2 |
| 9723–9729 | Reserve |
| 9730 / 9731–9732 / 9733–9738 / 9739 | Sealed bars / eligibility / arms / bootstrap and p |
| 9740 / 9741–9742 / 9743–9748 | Second round: bars / eligibility / arms |
| 9749 | Reserve |

---

## 9. Compute, RAM, GPU

Throughput basis: about 3.0k billed fps per unsteered worker (A1 median). For steered arms (feature scoring and refits), assume about 1.5k fps; this is measured at G-build. If it is below 1.5k, development units drop from 10 to 8 (a rule based on throughput, not on results).

| Stage | Billed frames | Wall (3 workers) |
|---|---|---|
| G0, G0′, G0-S, B-clock | 0 | about 5 h + 1 h + 1 h + 1 h CPU |
| Bars (development 41 + sealed 24) | about 54M | about 1 h |
| G1a (+ G1a′) | 48M (+48M) | 1.5 h (+1.5 h) |
| G1b room | 328M | about 10–12 h |
| G2 WORTH (10 units × 12 runs, early stop) | about 300M | about 15 h |
| G3 rehearsal (10 × 18 runs) | about 450M | about 26 h |
| Freeze 2 (optional, 80 runs) | about 200M | about 12 h |
| B0 eligibility (48 runs) | 192M | about 7 h |
| B1 (about 9 units × 28 runs) | about 630M | about 36 h |
| **Total** | **about 2.0G (2.3G with options)**; caps: development 1.45G, total 2.4G | **about 95–110 h** |

- **RAM.** Unsteered worker ≤ 1.0 GB, steered ≤ 1.2 GB (reservoir 40k), G0 worker ≤ 0.8 GB. Three workers ≤ 3.6 GB; two when Tavern needs RAM. Supervisor guard at 14 GB free (≥ 12 GB rule). Supervisor stops are logged with the outputs.
- **GPU.** 0 h. Nothing in v84 uses torch.
- **Disk.** About 2 GB for the G0 store, about 25 GB for run npz files.

---

## 10. Schedule (day 0 = Fri 23 Oct)

| Day | Work |
|---|---|
| 0 (23 Oct) | Worktree `F:\dev\ragnarok_v84` (branch `arc4-v84`); commit `docs/ARC4_V84_PROTOCOL.md` (French) with §3 transcribed and the seed grep. **G0 frozen.** |
| 1 (24) | G0 code, synthetic tests, 1-agent review; store; κ tuning, commit `g0_kappa.json`; V2 build starts |
| 2 (25) | F curves, commit `g0_units.json`; all arms; commit `g0_verdict.json`; route decision; v83 exploratory result read and labelled exploratory |
| 3 (26) | V2 tests and smoke (9700); bars (9706); G1a (9710) and its verdict |
| 4 (27) | G1a′ if needed; 2-agent adversarial review of V2, steering and route-S features; launch G1b overnight |
| 5 (28) | G1b verdict committed; Σ, T and S sources (leave-one-game-out); G0′ (report); G0-S if route S |
| 6–7 (29–30) | G2 WORTH and its verdict |
| 7–9 (30 Oct–1 Nov) | G3 rehearsal; GO, or freeze 2 |
| 9–10 (1–2 Nov) | Freeze 2 runs if needed; 3-agent review; sealed freeze (hashes) |
| 10–11 (2–3 Nov) | B0 bars and eligibility, committed; B-clock verdict committed |
| 11–13 (3–5 Nov) | B1 |
| 13–14 (5–6 Nov) | Verdicts, short report in French, handoff |

Slack is thin. If freeze 2 is used, the sealed test slips about one day. A sealed test that has started is always finished.

---

## 11. Odds (my estimates; correlated gates, so about 1.5–2× the independent product)

| Event | P |
|---|---|
| G0 pass (room 0.75 × worth 0.85 × speed 0.4) | about 0.25 |
| G-build | 0.85 |
| G1a (with G1a′) | 0.55 |
| G1b ≥ 13/35 \| G1a | 0.45 → overall about 0.21 |
| "Climbs on most dev games" (≥ 18/35) | about 0.10 overall |
| Route Σ: WORTH \| G0, room | 0.45; GO 0.4; ≥ 6 sealed units 0.75; PARTIAL+ \| measurable 0.45 (2× 0.2) |
| Route S: G0-S 0.3; WORTH 0.35; GO 0.35 | — |
| **Sealed FASTER-PARTIAL or better** | **about 1%** (Σ about 0.6%, S about 0.4%) |
| **Sealed FASTER-2×** | **about 0.3%** |
| Sealed TUNED-DEFAULTS or GENERIC label | about 1% |
| B-clock UNDERSTANDS-FASTER (needs the live route to reach sealed) | about 0.5% (about 0.12 with Option M) |
| Clean decisive answer at G0 by day 2 | about 0.95 |

Most likely ending: KC-knowledge-speed on day 2; then V2 either fails G1a or G1b (KC-room-84), or route S dies at G0-S or WORTH around day 7. In hand at that point: a frozen 41-game mechanism negative, an A1-strength cost map for a probe-free climber, and the full control harness.

---

## 12. What a positive would and would not establish

- **B1 FASTER (PARTIAL or 2×)** would establish this, on never-opened Atari games, with billed rewinds, the score-rise and lives senses, and sticky 0 practice: an explorer carrying a zero-mean covariance over ego-centric consequence features, learned from its own play on 41 earlier games, climbs between rungs matched on the fresh learner's own reach, including the late rungs, faster than:
  - the same explorer without it;
  - a placebo prior with an identical spectrum;
  - the better of tuned search defaults and the single best button process.
  
  The head start is zero by construction and certified.
- **B-clock UNDERSTANDS-FASTER:** from the same experience, the experienced learner reaches the novice's end-of-stream skill at predicting points and deaths in fewer frames. This is a mechanism claim, labelled separately.
- **G0 pass alone:** development evidence on burned logs. It is not a sealed result and not about play.
- **Not established by any of these:**
  - live play without savestates (the distillation wall stands);
  - perception of score or lives (they are senses);
  - "any game" (only games with room, likely 30–45% of Atari);
  - NES, 16-bit or PC;
  - accumulation (the dose caption is underpowered);
  - state-of-the-art sample efficiency;
  - human-like learning in general.
  
  A head start is never reported as faster learning.

---

## 13. Disagreements between judges, resolved

| Issue | Positions | Resolution |
|---|---|---|
| Frame | J1, J2: P5 first. J3: P1 as frame | P5's clock is G0. Live machinery = P3 (cells, post-loss archiving, probe-free, λ over rungs, WORTH order, dose) + P1 (bar modes as processes, held 0.75, bandit floor, T/S, TUNED-DEFAULTS label, instrument line, more seeds) |
| Carried object | J1: Σ. J2: whichever passes. J3: P1 controller | Σ if G0 passes; otherwise the search sense (P1 + P3), zero-mean, gated by G0-S. One route, one object |
| Eligibility strength | J2: A1-strength (P1's H ≥ B′ + 2u′ is met by V on 20/35). J3: rungs anchored on F | Room needs H ≥ M3 (A1 strength) **and** only rungs reached by both fresh seeds are measured. Proceed at 13/35; "most" at 18/35 |
| Ladder score | J1, J2: P3's true rises. J3: penalties ignored | Rises (what the agent senses; fixes the root-clone M1 on pong and fishing_derby). Net score is reported alongside; the limitation is stated in every report |
| Placebo | J1, J2: P5's shift. J3: shifts are weak | Full permutation within each run decides; the v83 shift is report-only |
| K\* leak at G0 | J3 | K\* from even 50-iteration blocks, test from odd blocks, deduplicated |
| Clustering | J3 | Game-cluster bootstrap; each game weighted equally |
| Coarse checkpoints | J2 | Quarter-octave checkpoints, log-interpolated, held for 2, averaged over 5 levels |
| Exam as sealed instrument | J2 for it; J1, J3 favour live play | Exam dropped (lineage replay risk, me-lost states, a constant best slot, correlated lineage states). Sealed = live climb (claim) + sealed knowledge clock (mechanism), labelled separately |
| P2 macros | J1: optional. J2: drop. J3: random-library placebo | Not in v84 (one object; depends on me); v85 candidate if G0 passes |
| P4 GPU learner | All defer | v85 platform; no live-play readout in v84 |
| X-rep | J3: multiplicity | Dropped |
| T and S controls | J2, J3: missing in P3 and P5 | Mandatory in development and sealed; X must beat the better of T,S |
| Over-regularised F | J3 (v83 n0 = 50,000) | κ tuned per arm on calibration games; sum-loss MAP |
| Sealed seeds | P1 8, P3 4 | 6 for X, F, T; 4 for P, S; se_proj uses σ̂²/3 |
| Head-start-sensitive filter | J3 on P3's AUC at n = 500 | G0-S uses the knowledge clock with its late clause |
| G1a on known failures | J1, J3 | Exploratory label; the claim-bearing G1b uses all 41 games and new seeds |
| Me dependence | J2 | V2's search is me-free; the heads read me passively (no probes); dilution counts against X |
| Odds | J2: P5's chain is 0.065, not 0.10 | Recomputed in §11 |

Factual corrections kept on record:
- Only pong and fishing_derby pass M1 at the root clone, not ice_hockey or tennis.
- Bar-mode tallies differ only by ties: 17/16/6 + zaxxon + jamesbond tie, or 18/17/6 counting ties.
- A1 rows per run range from 3.6k to 99k, not 60–100k.

---

## 14. Cousins in the literature (from memory; not re-checked; a literature check is due before any novelty claim)

- **Carried covariance prior:**
  - Raina, Ng & Koller 2006 (informative Gaussian priors for logistic regression built from related tasks): the closest cousin.
  - Evgeniou & Pontil 2004; Argyriou, Evgeniou & Pontil 2008; Bakker & Heskes 2003; Yu, Tresp & Schwaighofer 2005; Lawrence & Platt 2004.
  - Baxter 2000; Thrun & Pratt 1998; Caruana 1997.
  - Theory: Tripuraneni, Jin & Jordan 2021; Du et al. 2021.
  - Learned bandit priors, with non-zero means: Kveton et al. 2021; Basu et al. 2021; Cella, Lazaric & Pontil 2020; ALPaCA (Harrison et al. 2018).
  - ARD (MacKay; Neal).
- **Measuring learning speed:** Taylor & Stone 2009; Dawid 1984 (prequential); Blier & Ollivier 2018; Voita & Titov 2020; Whitney et al. 2020 (samples to reach a loss level: closest to the clock).
- **Controls:** Rice 1976; SATzilla (Xu et al. 2008) for single best and virtual best; DAC (Biedenkapp et al. 2020; Adriaensen et al. 2022); Feurer et al. 2015.
- **Explorer and search sense:**
  - Go-Explore (Ecoffet et al. 2019, 2021): V2 is essentially its exploration phase, with adaptive downscaled cells.
  - Agent57 (Badia et al. 2020); temporally-extended ε-greedy (Dabney et al. 2021).
  - Learned node selection (He, Daumé & Eisner 2014; Khalil et al. 2016); Intelligent Go-Explore (Lu, Hu & Clune 2024, which uses a foundation model, not allowed here); cell-free latent Go-Explore (Gallouédec & Dellandréa 2023).
  - Meta-learned exploration (Alet et al. 2020; First-Explore, Norman & Clune 2023; MAESN; E-RL²; Oh et al. 2020).
- **Cross-task transfer and Atari:**
  - Portable options in agent-space (Konidaris & Barto 2007); action priors (Rosman & Ramamoorthy 2012).
  - Actor-Mimic (Parisotto et al. 2016), Progressive Nets (Rusu et al. 2016), MGDT (Lee et al. 2022), Scaled QL (Kumar et al. 2023): mixed or negative transfer to unseen games.
  - Plasticity loss (Ash & Adams 2020; Nikishin et al. 2022; Abbas et al. 2023).
  - Human priors (Tsividis et al. 2017, 2021; Dubey et al. 2018).
- **Honest novelty.**
  - The statistics are classical: Raina et al. did this in 2006.
  - New is the combination:
    - an object learned only from the agent's own savestate play, from pixels and senses, with no pretrained weights;
    - mean zero, so the head start is zero and certified live by prefix identity;
    - a same-experience clock with spectrum-matched placebo and input-PCA controls and per-arm tuned priors;
    - a matched-rung late-climb test on never-opened games against tuned-default and single-best controls.
  - The contribution is an honest test, not a new learning algorithm.

---

## 15. Decisions for Jérémie, and immediate next steps

- **Decision needed (only if it arises).** **Option M.** If G0 passes but the live route dies (STOP-fresh, KC-room-84, KC-worth-live or KC-dev), we could run the knowledge-clock replication alone on the **9 second-round games**. That costs about 72M frames and about 3 h, and gives about a 0.5 chance of a sealed mechanism positive, conditional on G0 passing. It would use up the second-round pool, which is currently the top-up for future sealed tests. **Your yes is needed before the pool is used.**
- **No other decision is needed.** The headline odds (about 1%) and the most likely outcome (a clean negative plus a cost map) are stated in §11.
- **Today:**
  1. Create the worktree `ragnarok_v84`.
  2. Write `docs/ARC4_V84_PROTOCOL.md` in French, transcribing §3 exactly in substance, plus the seed block and grep results.
  3. Commit it with `git commit --only` and the footer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Files to write:**
  - Agent package: `ragnarok/agent_v84/{consequence, framecode, archive, processes, bandit, explorer, steer, critic}.py`, reusing `agent_v83/{adapter, entry, views, logs}`.
  - Evaluation: `ragnarok/evaluation/{senses_v84 (Trace with true rises), milestones_v84, pools_v84}.py`, reusing `climb_stats_v83`.
  - Scripts: `scripts/{g0_store, g0_sources, g0_tune, g0_clock, g0_verdict, bars, v2_job, room_verdict, sources_live, arm_job, worth, rehearsal, sealed, clock_sealed}_v84.py`.
  - Tests: `tests/{test_isolation, test_g0, test_v2, test_prefix_identity}_v84.py`.

---

# Audits

```json
[
 {
  "verdict": "The plan is mostly honest. G0 is plainly marked as development evidence on logs already used. The zero-mean prior and the prefix certificate rule out a head start, censoring always counts against X, T and S stand in for designer tuning, negatives are kept, nothing uses RAM or downloaded weights, and the literature limits are admitted. G0 labels come only from the senses: rise = max(0, ALE reward) and ale.lives() (senses_v83.py:147-150, logs.py:70-93). The features use no colour and no oracle (views.py). G0 can be frozen today once the contamination record and two G0 edge cases are fixed (issues 8 and 9). The rest must be fixed before the matching freezes. The verdict ladder has gaps and one wrong label. Several labels overclaim: \"play\" for a savestate search, UNDERSTANDS-FASTER, the STOP-fresh sentence. Route S has no claim written for it. The V2 label rule selects rows on their outcome. Nothing guards absolute time when the climb is counted from each arm's own first rung. The live placebo may be too weak to separate content from generic shrinkage. I read and grepped only. I ran no Python and edited nothing. I did not open or list anything under explore/. The only file names I listed were from a names-only grep, which matched the supervisor log; I checked from its code that this log records only process command lines.",
  "issues": [
   {
    "title": "B1 (and G0) verdict ladder has gaps and one wrong label",
    "severity": "major",
    "where": "Plan §7 'B1 verdicts' items 3-9; §G0.11 KC-knowledge-speed",
    "problem": "Several realistic outcomes get no accurate label. (a) X beats F, T/S and P with p <= 0.025, but the lower bound of R_late is <= 1. PARTIAL fails, TUNED-DEFAULTS needs failure against T/S, GENERIC-STEERING needs LB R(X:P) <= 1, so the result falls to NULL, although X was faster on the early rungs. (b) LB R(X:F) > 1, X vs P undetermined, P not better than F: NULL, although X did gain. (c) X roughly equals F while T or S beats both: the upper bound of R(X : better of T,S) < 1 triggers NEGATIVE TRANSFER, but nothing was transferred negatively. X is only below tuned defaults. (d) HEAD-START ONLY is tested before NEGATIVE TRANSFER, so a head start followed by a slower climb is labelled HEAD-START ONLY and the harm is hidden. (e) All lower bounds > 1 but p > 0.025: NULL. In G0, LB R(X : best of F,P,U) > 1 with R < 1.5 gives KC-knowledge-speed. Its sentence 'does not make the learner faster than tuned controls' is then false.",
    "fix": "Add pre-written labels: EARLY-RUNGS-ONLY (the B1 analogue of FIRST-RULES-ONLY), UNATTRIBUTED-GAIN (X > F, X vs P undetermined), BELOW-TUNED-DEFAULTS (kept separate from NEGATIVE TRANSFER, which keeps only 'UB R(X:F) < 1'), and NOT-SIGNIFICANT. Test NEGATIVE TRANSFER before HEAD-START, or allow both labels. In G0, add a SUB-THRESHOLD caption and reword KC-knowledge-speed as 'not >= 1.5x faster than the best tuned control'."
   },
   {
    "title": "B-clock is named UNDERSTANDS-FASTER, has only two outcomes and is undefined under route S",
    "severity": "major",
    "where": "Plan §0 'Measures', §7 B-clock row, §11, §12",
    "problem": "B-clock measures how many logged frames an offline logistic learner needs to predict a 24/40-step rise or loss on a stream it does not influence. 'UNDERSTANDS' overstates that and invites the result to be presented as learning a game faster. It has only UNDERSTANDS-FASTER or NOT-REPLICATED. Fewer than 8 units, X worse than controls, generic shrinkage and first-rules-only would all read 'NOT-REPLICATED'. Under route S (G0 failed, Σ not carried), it is unclear whether B-clock tests Σ, the critic, or nothing. No rule stops the combination 'B1 not FASTER + B-clock pass' from being headlined as 'learns new games faster'. That is exactly the predictive-result-as-play confusion the lens forbids.",
    "fix": "Rename it PREDICTS-CONSEQUENCES-SOONER (or similar). Give it the full G0 label set: NOT-MEASURABLE (< 8 units), NEGATIVE, GENERIC-SHRINKAGE and FIRST-RULES-ONLY captions, K* reported. State what B-clock does under route S: skip it, or run the G0-S recipe on the sealed expansion logs. Pre-write a headline table B1 x B-clock: for example 'B1 NULL + clock pass = predicts outcomes sooner on new games; no faster climb; not learning faster'."
   },
   {
    "title": "Route S has no pre-written claim sentence",
    "severity": "major",
    "where": "Plan §5.2, §12, §14",
    "problem": "§12 states what a positive would establish only for route Σ (a covariance over ego-centric consequence features). Under route S, the carried object is a learned search heuristic over cells and processes (search-statistic features, adapter signature). A positive there is cross-game algorithm configuration, close to learned node selection and DAC (the judges' point). It is not knowledge about games. Without a written sentence, a route-S FASTER would carry the same headline as route Σ.",
    "fix": "Add §12-S: 'a cross-game learned search controller makes the savestate explorer climb matched rungs faster than ...; it carries search configuration, not knowledge of game consequences'. Use route-specific label names (for example FASTER-SEARCH-SENSE). List its own cousins in §14 (learned node selection, DAC, Intelligent and latent Go-Explore)."
   },
   {
    "title": "The primary measure is called 'play' but is a savestate search climb",
    "severity": "major",
    "where": "Plan §0 'Measures' ('Primary claim (play)'), §6, B1 label names",
    "problem": "T(M) is the billed frame of the first clone whose cumulative rises reach M in a Go-Explore tree with rewinds. It measures how fast a search with savestates finds high-scoring trajectories. It is not a policy that plays, and §12 itself excludes live play. The v83 protocol opened with 'Toute affirmation de v83 porte sur l'ENTRAÎNEMENT AVEC RETOURS EN ARRIÈRE ... et avec les SENS'. v84 drops that header, and the bare labels FASTER-2x and FASTER-PARTIAL will be read as 'learns to play faster'.",
    "fix": "Replace '(play)' with 'search climb with billed rewinds'. Name the labels SEARCH-CLIMB FASTER-2x and SEARCH-CLIMB FASTER-PARTIAL. Copy the v83 header sentence into ARC4_V84_PROTOCOL.md and into every verdict file."
   },
   {
    "title": "Counting from each arm's own M1 has no absolute-time guard",
    "severity": "major",
    "where": "Plan §6 'Per run' λ and λ_late; §7 FASTER-2x and FASTER-PARTIAL",
    "problem": "Both λ (from T(M1)) and λ_late (from T(M2)) are counted from each arm's own rung. No FASTER clause looks at time from frame 0. After its first two-class refit, X can steer worse, reach M1 later, then climb faster. It would be labelled FASTER while reaching M_k* later than F in absolute billed frames, the opposite of 'learns the new game faster'. HS is only reported, and only on pairs where both arms reach M1 (climb_stats_v83.head_start).",
    "fix": "Add a non-inferiority clause to every FASTER label on absolute T(M_k*) from frame 0, for example lower bound of R_abs(X:F) >= 0.9, or at least a point estimate >= 1. Add a LATE-START caption when the upper bound of HS is < 1. Report HS with its interval."
   },
   {
    "title": "V2 label rule selects rows on their outcome, and the RowLog fields overflow",
    "severity": "major",
    "where": "Plan §4 'Logs' ('labels uses steps = the rollout's final planned length') and 'Rollouts' (extended by 50 at each new archive-best S, up to 400)",
    "problem": "A rollout is extended only when it sets a new archive-best S, which needs a rise. Rows near the 100-step end are then kept only when a rise followed them, which inflates GOOD positives. This breaks the v83 rule 'Keeping a row never depends on its outcome' (logs.py:85-93, review A-1) for every live head, and for G0′ and B-clock if they adopt the rule. Separately, RowLog stores t and avail as uint8 (logs.py:53-57) while rollouts reach 400 steps. Values above 255 either raise an error or wrap, which corrupts the censoring.",
    "fix": "Censor each row against the length planned at the row's own step, known before its outcome, not the final length. Widen t and avail to uint16. Add a test that the kept-row mask is identical under label permutation for a fixed extension schedule."
   },
   {
    "title": "Live placebo may be weak, and no generic-shrinkage control reaches sealed",
    "severity": "major",
    "where": "Plan §G0.6 P construction; §5.1 arms table (U 'Development, 2 seeds, report only'); §7 G3, B1",
    "problem": "P's heads are fitted on permuted labels and normalised to unit norm. Noise fits lie mostly along low-variance input directions, so Σ_P can be an anti-informative prior, and with κ capped at 1 it may be over-shrunk. If P is worse than F, 'LB R(X:P) > 1' follows from 'LB R(X:F) > 1' and adds nothing, and GENERIC-STEERING (which needs R(P:F) > 1) can never fire. U, the control for generic input-covariance shrinkage, is in G0 but not in G3 decisions or B1. A sealed FASTER could therefore come from generic anisotropic shrinkage and still be attributed to consequence content.",
    "fix": "Make U a decision arm in G3 and B1 (>= 4 seeds) and require LB R(X:U) > 1 in FASTER. Alternatively, pre-register a placebo-validity rule from G0: if the upper bound of R(P:F) < 1 there, P is declared weak and U (or a cross-game label permutation) replaces it live."
   },
   {
    "title": "Contamination record is incomplete and the unseal rules disagree",
    "severity": "major",
    "where": "Plan §1 'Contamination record'; F:\\dev\\ragnarok_v83\\docs\\ARC4_V83_EXPLORATORY.md seal paragraph",
    "problem": "The record says two judges saw 'good alien False'. That line is reproduced in docs/panels/v84_judges.md, so the synthesizer who hardened the G0 thresholds, every auditor and anyone reading the panel files had seen it before the freeze. The v83 seal note allows reading the result once the v84 thresholds are 'figés et committés', which is the day-0 protocol commit. The plan says only after g0_verdict.json. These are two different unseal points. The seal is defined by path only, yet explore_promise_v83.py ends with print(json.dumps(summary)), so the job's console output or completion notification in the launching session also carries the result. Judge 2's disclosed stray `python -c \"1\"` is not on the record.",
    "fix": "Before the day-0 commit: list every reader of the line (judges 1-2, synthesizer, auditors). Amend the v83 seal note to 'after g0_verdict.json is committed'. Extend the seal to the job's console output and notifications. Record the stray command."
   },
   {
    "title": "G0 frozen text leaves edge cases that would force decisions after the freeze",
    "severity": "minor",
    "where": "Plan §G0.3 'Clock', §G0.7 κ grid",
    "problem": "(a) When Spent is raised mid-rollout, explorer.py:206-213 labels that rollout's rows in `finally`, but `it += 1` and `iters.add` are skipped. Those rows have row_it = len(iter_billed), so c_i = iter_billed[row_it] is undefined in most runs. (b) κ is capped at 1 (grid 4^-5..1) and no rule covers an optimum at the edge. If F's best κ is at the cap, the 'tuned F' is handicapped and the comparison favours X.",
    "fix": "Before the commit: define c for the last partial rollout (cap − entry_frames, or drop its rows, the same for all arms). Pre-declare a symmetric rule: if any arm's κ is at a grid edge, extend the grid one step for all arms."
   },
   {
    "title": "Kill sentences overgeneralise or are missing",
    "severity": "minor",
    "where": "Plan §7 G1a (STOP-fresh), G2 (KC-worth-live), G1b (KC-room-84), G3 (KC-dev), G-build (KC-build)",
    "problem": "STOP-fresh says 'budgeted savestate exploration with the senses does not climb on most Atari games at 4M'. That is drawn from 10 A1 failures picked by hand, on one seed, under an exploratory label. KC-worth-live ('same-game structure is not this climber's bottleneck') generalises from one rank-1 Σ steering setting. KC-room-84, KC-dev and KC-build have no pre-written sentence. Negatives must be accurate too.",
    "fix": "Scope each sentence, for example 'V2 at these constants does not fix the 10 selected A1 failure games within 4M' and 'same-game Σ steering at G0 κ does not speed V2'. Write the missing sentences now."
   },
   {
    "title": "Senses decision is not restated, and the ladder quantity is not defined against the sense",
    "severity": "minor",
    "where": "Plan has no §0 on inputs; §6 'cumulative true rises'; verdict JSON contents",
    "problem": "The owner's rule (22 Oct) requires every result obtained with the senses to say so. v83 room_verdict.json carries a 'senses' field, but the plan specifies none for its verdict files. The v83 §0 rule that game over is only an episode boundary ('jamais d'étiquette, de variable ni de veto') is not restated, although V2 stops rollouts on game end. 'Cumulative true rises' is undefined. The sense is the per-step max(0, 4-frame reward) (senses_v83.py:147). A per-frame evaluator count would differ when a step mixes a gain and a loss, so the claim that the ladder is 'what the agent senses' would not hold exactly.",
    "fix": "Add an inputs section: pixels, own buttons, per-step positive part of the reward (magnitude), ale.lives(); game over as a boundary only; RAM and true score evaluator-only. Every verdict JSON carries the senses line. Define the ladder as the sum over agent steps of max(0, step reward)."
   },
   {
    "title": "Oracle guards for the new code are under-specified",
    "severity": "minor",
    "where": "Plan §4 G-build tests; §5.2 route S features",
    "problem": "G-build lists only 'imports nothing from evaluation' and 'no game names'. The v83 isolation test (tests/test_isolation_v83.py) also bans access to private attributes of handed objects and BAD_ATTR (ram, getRAM, true_score, ale, cloneState, ...), plus eval and getattr. Without it, agent code could reach env._e.ale or _true_score. In route S, 'log frames since the last record' is ambiguous: in v83, 'records' are the evaluator Trace's true-score records.",
    "fix": "Port the full v83 isolation test to agent_v84, consequence.py and critic.py, and keep it running on the reused agent_v83 modules. Define 'record' as the archive-best sensed S."
   },
   {
    "title": "Freeze 2 tunes X but not F, and has no selection rule",
    "severity": "minor",
    "where": "Plan §7 G3 'freeze 2'",
    "problem": "κ_X, κ_P, τ² and the rank cut can be re-tuned on live development data, while κ_F (offline calibration) and T stay fixed. A later sealed X:F ratio then partly measures the extra tuning of X. How one setting is chosen from the grid is not written, which invites choosing after looking.",
    "fix": "Pre-write the selection rule (criterion, tie-break, a single pick). Either give κ_F and T's n0 the same re-tuning budget, or add a caption that X received an extra dev tuning round."
   },
   {
    "title": "'Fresh' and 'new game' wording overstates",
    "severity": "minor",
    "where": "Plan §0 G0 question, §5.1, §6 'Top rung k*'",
    "problem": "F is not fresh: it uses cross-game standardisation (S₋g / S_src) and a κ tuned on calibration games. G0's development games are 'held out' for the learner but not new: they have been burned and studied. 'Every measured rung was therefore reached by the fresh learner on both seeds' describes V2 on the room seeds, not F on the arm seeds.",
    "fix": "Say 'held-out development game', 'baseline learner F (cross-game feature scaling, tuned κ)', and 'rungs reached by V2 on both room seeds'."
   },
   {
    "title": "Near-sibling games in the held-out pools are not addressed",
    "severity": "minor",
    "where": "Plan §8 pools; pools_v83.py SEALED and SECOND",
    "problem": "Sealed pacman is a sibling of ms_pacman, a calibration game and Σ source. The second-round atlantis2 is a variant of the development game atlantis, and could be used by the B0 top-up or Option M. frogger is close to freeway. 'Never-opened' is true, but 'a new game' is weak for a ROM variant. FASTER-PARTIAL has no leave-one-game-out guard.",
    "fix": "Pre-declare per-game reporting plus a 'without near-siblings' line (pacman, atlantis2). Alternatively, remove atlantis and ms_pacman from the sources when scoring their siblings. Add the leave-one-game-out minimum to FASTER-PARTIAL."
   },
   {
    "title": "Novelty statements are not precise enough about the cousins",
    "severity": "minor",
    "where": "Plan §14 'Honest novelty', §0 title",
    "problem": "Raina, Ng & Koller (2006) already used a zero-mean Gaussian prior whose covariance is learned from related tasks. Only the head-start framing and the live prefix certificate are new, not 'mean zero'. Go-Explore (Ecoffet et al. 2021) already adapts the downscaling parameters to a target cell count, so V2's re-keying rule is theirs. 'Carry the questions' needs a caveat: the questions (427 features, ego rotation, the me tracker) are designed by hand, and only their covariance is learned.",
    "fix": "Reword §14 accordingly. Keep the existing note that the literature was cited from memory and must be checked before any novelty claim."
   },
   {
    "title": "The tested object differs from the carried object, and Option M has no labels",
    "severity": "minor",
    "where": "Plan §5.1 'Σ sources', §7 G0′ and B-clock, §15 Option M",
    "problem": "G0 tests Σ built from 2 A1 (V) streams per game, with κ tuned for that Σ. The live arms and B-clock use Σ from 4 streams (A1 + A1′) and V2 streams with the same κ, and G0′ on V2 streams is report-only. A G0 PASS certifies a recipe, not the object that goes to sealed. Option M (second-round knowledge clock) has no explorer, thresholds or labels written.",
    "fix": "State that G0 certifies the recipe. Add a caption if G0′ disagrees with G0, and say explicitly that G0′ cannot reopen route Σ. Pre-write Option M's explorer, unit rule, thresholds and labels before asking Jérémie."
   },
   {
    "title": "The power statement is optimistic",
    "severity": "minor",
    "where": "Plan §7 'Power, written now'",
    "problem": "'A true 2x passes FASTER-2x about half the time': at a true 2x, the point-estimate condition R >= 2.0 alone holds about 50% of the time. Six more conditions (T/S, P, late, 2/3 of units, leave-one-game-out, p) lower it further. The calculation also assumes σ̂² ≈ 0.8, below A1's 1.165, which gives se ≈ 0.23 instead of 0.20.",
    "fix": "Restate the figure as an upper bound, or simulate it once with A1's σ̂², and keep the result in the protocol."
   }
  ]
 },
 {
  "verdict": "G0 can run on this machine as specified, but one gap in its text must be fixed before it is frozen today. The live half of the plan is not realistic as costed.\n\nG0 data and compute check out. After dropping lost rows and capping each stream at 24k, 1,705,576 rows remain, a float16 store of about 1.46 GB (the plan says about 2M rows and 1.7 GB). At roughly 2.5–3k curves of 5–15 s each, G0 needs about 3–6 h on 2 workers at 0.8 GB or less, so a verdict on day 2 is plausible.\n\nThe gap: the clock rule `iter_billed[row_it]` has no value for the last rollout, the one cut off by the billed cap, in 64 of the 82 A1 runs.\n\nThe live chain rests on nominal numbers that A1 itself contradicts:\n- **Throughput.** A1 delivered about 10.3M billed frames per hour on 2 shards, not the 32M per hour assumed for 3 workers.\n- **Killed runs.** The commit guard terminated 24 runs (21 distinct jobs), and each one restarted from zero.\n- **Memory.** Committed memory grows by about 4 KB per logged row on top of about 0.82 GB. A probe-free V2 run at 4M frames using the v83 RowLog would therefore commit about 1.7 GB unsteered and about 2.0 GB steered. That is above the plan's own G-build and G1a limits (1.0, 1.1 and 1.2 GB), so these gates could kill the fresh climber for an engineering reason, not a scientific one.\n\n**Compute and schedule.** Realistic compute from G1b to B1 is about 200–270 h, against 95–110 h planned. B1 alone needs about 75–85 h. The sealed test therefore does not fit in days 11–13 without cuts declared in advance.\n\n**V2 test design.** The byte-exact V2-to-V equivalence test is impossible without V's probe path, because probes also repair the background model. A v83-style steering hook would also break the prefix-identity test.\n\n**Labels.** Rollout extension reintroduces a selection on outcomes into the labels.\n\nFix the clock rule today. Fix logging and RAM limits, the test definitions, the extension label rule and the compute table before V2 and G1a are frozen.",
  "issues": [
   {
    "title": "G0 clock undefined for the last rollout, the one cut off by the billed cap",
    "severity": "blocking",
    "where": "§3 G0.3 'Clock' (frozen today); A1 npz row_it / iter_billed",
    "problem": "Explorer.run adds an iteration's billed frames only after the rollout ends. When Spent is raised mid-rollout, that rollout's rows are labelled anyway (finally block), but no iter_billed entry is written. Checked on all 82 A1 npz files: in 64 runs max(row_it) == len(iter_billed); for example alien 9131 has 1,294 iterations and its last row_it is 1,294. So c_i = iter_billed[row_it[i]] - entry_frames is out of range for up to 24 rows per run. The frozen text only says 'non-decreasing, otherwise Interruption', so the implementer must either declare an Interruption or make an undeclared choice after the freeze.",
    "fix": "Write into the frozen §G0.3: 'rows with row_it >= len(iter_billed), from the rollout cut by the cap, are dropped (their windows are mostly censored anyway); the count is reported'. Add this case to test 1 in §G0.13."
   },
   {
    "title": "Compute table assumes 3 × 3k fps; A1 realised about a third of that on this machine",
    "severity": "major",
    "where": "§9 table, §10 schedule, §15 Option M, odds ('clean answer by day 2' aside)",
    "problem": "A1 ran 164M billed frames on 2 shards (room_v83.py) from 22 Oct 09:48 to about 01:46. That is 15.96 h of elapsed time for 16.73 h of summed run walls: an average of only about 1.05 runs completing at once, about 10.3M billed frames per hour. Launch gates, 117 supervisor actions and 24 terminations (note 22) account for the gap. The plan's 32M per hour gives G1b 10–12 h (realistic 22–32 h), G2 15 h (about 30 h), G3 26 h (about 55–65 h; see the run count issue), freeze 2 12 h (about 27 h), B0 7 h (13–19 h) and B1 36 h (about 75–85 h). The total is about 200–270 h against the 95–110 h planned. With gates in sequence and reviews between them, B1 cannot run in days 11–13. Terminated runs restart from zero, and a killed 4M steered run loses about 45–90 min.",
    "fix": "Base every wall-time figure on A1's realised rate (about 5M per hour per shard unsteered, about half that steered), plus about 25% for reruns. Measure the actual rate during G1a and G1b. Declare now an ordered list of cuts to apply if the calendar binds: drop U and X_8 in development, then 8 development units, then 4 sealed seeds for T, or a third week. Have V2 jobs write partial outputs, or a resumable checkpoint, so that a termination does not erase a 4M run."
   },
   {
    "title": "RAM limits in G-build and G1a cannot be met with v83 row logging; engineering failures can trigger scientific kills",
    "severity": "major",
    "where": "§4 G-build ('≤1.0 GB unsteered, 1.2 GB steered over 4M'), §7 G1a pass ('commit ≤1.1 GB; ≥1,500 billed fps'), §9 RAM",
    "problem": "Measured on A1: peak commit is about 0.82 GB plus about 4 KB per logged row (857 MB at 9.8k rows, 1,224 MB at 99.1k), while working set is only 119–524 MB. RowLog keeps every row in RAM and copies them all into one array at the end. A probe-free V2 at 4M frames logs about 210–240k rows, so expect about 1.7 GB commit unsteered. The v83 F smoke (amidar 9130, steered) already committed 1,140–1,196 MB at 400k frames with only 12.8k rows. A 40k float16 reservoir plus its refit copies adds about 0.3 GB, giving about 2.0 GB steered. Three steered workers would then commit about 6 GB. The supervisor terminates jobs on the commit floor, not only on physical RAM. G1a also mixes engineering criteria (commit, frames per second measured over a wall time that includes supervisor suspensions) with the climbing counts, and does not say whether an engineering miss leads to STOP-fresh.",
    "fix": "Write rows to disk in chunks (np.save per 8k-row chunk, or a memmap) and drop the final copy. Set OPENBLAS_NUM_THREADS / OMP_NUM_THREADS = 1 in explorer workers; the large untouched commit at startup is probably BLAS thread buffers, so verify at G-build. Set the per-worker limit from a measured 4M smoke, then set the worker count from it. Move commit and frames-per-second out of G1a's pass rule into a fix-and-rerun engineering check that never leads to STOP-fresh or KC-build. Measure frames per second on CPU time or excluding suspended intervals."
   },
   {
    "title": "Byte-exact V2-to-V equivalence test conflicts with 'no probes'; the v83 steering hook would fail the prefix test",
    "severity": "major",
    "where": "§4 G-build tests ('equivalence harness', 'prefix identity'); KC-build after 2 fix rounds",
    "problem": "V probes at the root (body.locate while the tracker is lost) and every 8 steps while lost. Each probe bills frames and also repairs the background model bg (entry.py:86-88), which changes every later view and row. V2 has no probes after entry, so it cannot reproduce V's rows, iterations and cells byte for byte. The listed switches also omit V's selection weights (×4 at the best S), and they must disable frontier draws, the Thompson bandit and re-keying, all of which consume RNG. Separately, v83 Promise.pick_cell with inactive heads draws 16 candidates and then one integer (promise.py:179-181). That is equal in distribution, not byte-identical, so reusing that pattern fails the prefix-identity test. Since 2 failed fix rounds mean KC-build, a mis-specified test could kill the build.",
    "fix": "Either give the test a v83 mode that includes the probe and background-repair path and every RNG consumer listed above, or replace it with unit-level checks (shared ledger, Trace, RowLog, labels) and a check that the distributions match. Specify that inactive steering calls V2's own draw exactly. Count only failures of V2's own tests toward KC-build."
   },
   {
    "title": "Rollout extension makes the label filter depend on outcomes, for the live heads and the B-clock streams",
    "severity": "major",
    "where": "§4 'Rollouts' (extend by 50 at each new archive-best S) and 'Logs' ('labels uses steps = the rollout's final planned length')",
    "problem": "A row is kept only if t + H ≤ steps. Extensions happen after a rise sets a new best S. With steps set to the final length, GOOD rows at t = 77–100 (BAD rows at t = 61–100) are kept only when a rise extended the rollout, and that rise often falls inside their own window. This brings back the outcome-dependent selection that v83 review A-1 (build note 6) removed. It inflates GOOD positives in V2 streams, so it affects the live heads, G0′, the A1′ sources for Σ, K* and the sealed knowledge clock (B-clock).",
    "fix": "Keep a row only if t + H ≤ L_t, where L_t is the planned length known at step t (extensions decided at or before t). Log L_t per row. Add a test that a row's kept/censored status never depends on events after t."
   },
   {
    "title": "Synthetic-control test is not costed and its pass rule is not tied to declared parameters",
    "severity": "minor",
    "where": "§G0.13 item 6, §G0.14, day-1 schedule",
    "problem": "Running the full pipeline on 41 Gaussian games × 2 streams × 24k rows × 427 features, under two scenarios, costs about as much as G0 itself: 3–6 h of CPU per scenario, and about 0.7 GB per worker in float64. None of this is in the 'about 5 h'. Whether R ≥ 2 holds for the shared-subspace case depends on the signal strength, noise and n chosen. A failure might reflect the synthetic design rather than a bug, which would stall real-data curves on day 1. The day-1 workload is G0 code, synthetic tests, review, store, κ tuning and the start of V2, so 0.95 for 'decisive by day 2' looks optimistic (about 0.8).",
    "fix": "Freeze a reduced synthetic configuration now: about 12 games, 6k-row streams, arms F, X, P and U, with stated dimensions and signal-to-noise. Base its expected R on a pilot of the synthetic generator only (allowed, since it contains no real data). Budget it separately."
   },
   {
    "title": "κ-tuning count and X_8 κ not specified consistently",
    "severity": "minor",
    "where": "§G0.7 grid, §G0.14 step 3 ('864 curves'), §G0.6 X_8",
    "problem": "G0.7 tunes 8 arms (F, X, P, U, K*, X-mean, X_diag, P_shift) × 6 κ values × up to 24 calibration curves, which is up to 1,152 curves, not 864 (864 is 6 arms). X_8 has no κ rule, so its tuning is left to the implementer after the freeze. Calibration also has no BAD units for freeway and pong (no lives), so there are about 20 tuning curves per κ, not 24. The rule for a K* head that lacks 30/30 classes in its even-block source is also missing.",
    "fix": "State the arm list and count. Give X_8 a rule (for example, X's κ, or tuned the same way). State what happens to K* for a unit whose K* source is short of either class (unit dropped from (ii) only, and listed)."
   },
   {
    "title": "Development frame cap can bind in the middle of a gate; G3 run count is off",
    "severity": "minor",
    "where": "§9 table, §7 G2/G3 rows, budget line ('development capped at 1.45G')",
    "problem": "G3 lists X, P, T, S × 4 seeds plus U and X_8 × 2 seeds: 20 runs per unit, so 200 runs, about 500M frames, not 18 runs and 450M. G2's 300M and G3's figure assume an average early stop at about 2.5M frames. If runs are censored at the 4M cap (likely, given A1's seed spread), G2 costs up to 480M and G3 up to 800M. Development spending could then pass 1.45G during G3 or freeze 2, and no rule says what happens then.",
    "fix": "Correct the count. Declare now what happens if the development cap is reached mid-gate (which arms stop first, and whether the gate is then read as incomplete or as a kill)."
   },
   {
    "title": "The exploratory v83 job is still running and shares G0's resources",
    "severity": "minor",
    "where": "§1 contamination record, §10 day 0–1",
    "problem": "scripts/explore_promise_v83.py has been running since 03:07 (PID 21596, working set 0.5–1.3 GB). It is on the supervisor's worker list. On day 1 it competes with the G0 store build, κ tuning and V2 smoke tests for the 2–4 GB budget and for CPU. The plan does not say whether it is left to finish unread or stopped.",
    "fix": "State the policy. For example: let it finish with its output unread, count it in the day-1 RAM budget, and keep G0 at 2 workers until it exits. Stopping it needs Jérémie's consent."
   },
   {
    "title": "κ tuned on thinned A1 streams is carried into live heads and the sealed clock at a different rows-per-frame rate",
    "severity": "minor",
    "where": "§5.1 ('κ_a from g0_kappa.json'), §7 B-clock ('G0 κ (frozen)')",
    "problem": "G0 thins streams of up to 99k rows to 24k (about 1 row in 4) and uses a sum-loss prior that fades as 1/n. Live heads use a 40k reservoir of every row; once full, the prior's relative weight stops fading. Sealed eligibility streams come from V2 at 4M frames, with rows after life losses and extended rollouts. The same κ therefore means a different prior strength per billed frame. This is symmetric across arms, but it is no longer the tuned operating point the plan relies on.",
    "fix": "Declare it as a known mismatch, or tune live κ on the A1′ (V2) calibration streams on day 5 with the same per-arm rule, before any live arm runs."
   },
   {
    "title": "Bars cut episodes at 7,000 steps while V2 lineages have no depth limit",
    "severity": "minor",
    "where": "§6 Bars B′ and rungs; lesson 4 claim ('same lives rule')",
    "problem": "Bars score episodes of at most 7,000 steps (28k frames). V2 keeps playing after a life loss and extends rollouts, so a lineage can run longer than 7,000 steps on long games (atlantis, video_pinball, centipede). That makes B′ and the rungs easier to reach than the 'same rule' wording suggests. Cumulative true rises must also be summed per 4-frame step, exactly as the sense and the bar compute them, including in the replay.",
    "fix": "Either cap the depth at which a lineage counts for rungs at 7,000 steps, or state the mismatch. Define cumulative true rises per step in senses_v84, and test that the replay (with one NOOP before restore, note 21) matches the live Trace across a life loss."
   },
   {
    "title": "Small factual corrections for the protocol text",
    "severity": "minor",
    "where": "§1 table, §G0.14, §8 seed grep, §9 bars",
    "problem": "(a) The G0 store holds 1,705,576 rows, about 1.46 GB in float16, not about 2M and 1.7 GB. Very short streams: atlantis 9132 has 122 rows left, krull 864 and 2,197, up_n_down 3,888. (b) Seed grep: tracked .py files in all 9 worktrees are clean, but a raw grep of F:\\dev\\ragnarok finds 156 matches inside venv310 and .venv-arc4 site-packages; the re-grep must exclude virtual environments or use git ls-files. (c) Bars are unbilled (bars_v83) but are counted in the billed total. The v83 bars took 6,504 s for 41 games, so 65 games plus the held-0.75 mode take about 3 core-hours, about 1 h on 3 workers. (d) The labels use lag < H, not ≤.",
    "fix": "Correct these figures in ARC4_V84_PROTOCOL.md. Write the grep as `git ls-files '*.py' | xargs grep -E '\\b9[67][0-9]{2}\\b'`."
   }
  ]
 },
 {
  "verdict": "The core design is sound, but it is not ready to freeze as written. Two blocking fixes are needed before today's commit, and several major attribution and power fixes before any live arm.\n\nWhat holds. G0 can fail, and almost every choice in it works against X:\n- the comparator is the per-unit best of F, P and U;\n- censoring always counts against X;\n- PASS is an intersection of conditions;\n- units are chosen on F alone;\n- `g0_kappa.json` and `g0_units.json` are committed before any development curve.\n\nNo gate is already passed by the A1 data. The live yardstick (λ counted from each arm's own M1, HS never a claim) does not confuse a head start with faster learning.\n\nFacts checked against the files:\n- 82 runs, 5,434,791 rows, krull 9132 = 3,600, pong 9132 = 99,103, `steer` null in all 82.\n- 18 runs have fewer than 24k rows.\n- The manifest covers 82 npz, 82 json and 41 bar files.\n- 427 features = 336 + 16 + 4 + 4 + 2 + 64 + 1.\n- The clock (`iter_billed`, which includes the entry, minus `entry_frames`) is correct.\n- Censoring matches `milestones_v83.climb`.\n- The seed block 9700–9749 is clean in project code across all 9 worktrees. The only hits are in `venv310` and `.venv-arc4`.\n\nBlocking:\n1. Only §3 is frozen today. Every later threshold, the route-S features and the B1 ladder would be set after the G0 verdict and after the sealed exploratory result is read on day 2.\n2. The G0 Σ recipe (ridge 1e-4, exactly collinear features, a remap that inflates floored features, normalisation in parameter space) may be dominated by noise, and nothing checks it on the real feature geometry. A G0 negative could then be an artifact of the recipe, yet it would be reported as a 41-game mechanism negative.\n\nMajor:\n- The sealed test lacks the no-heads explorer V2 and the input-PCA control U, and P is probably a weak placebo.\n- About a third of fresh runs on new seeds are expected to be censored at the top rung, and the F-vs-F instrument line trips by noise alone about 10–17% of the time. Neither is in the power estimate.\n- K\\* leaks at G0: even and odd blocks of the same run share lineages.\n- The G0 late clause does not separate a jump right after the first fit from a faster rate.\n- Freeze 2 retunes only X and P, with no written selection rule and broken pairing.\n- G1a, a 10-game pilot, can stop the whole live route with an over-general caption.\n\nWhat I did not open: nothing under `explore/`, and not the post-seal edit of `ARC4_V83_EXPLORATORY.md`. I read only its pre-computation version (`git show b9617d7`). I checked the supervisor logs by match counts and command lines only. They hold no results, but they show the exploratory job (PID 21596) still running and being suspended at 04:07 and 04:10 today. No Python, emulator or GPU job was run, and no file was edited.",
  "issues": [
   {
    "title": "Only §3 is frozen today; the rest is frozen after G0 and the sealed exploratory result are seen",
    "severity": "blocking",
    "where": "§10 day 0; §15 'Today' step 2; §5.2 'features are frozen at V2's code freeze'; §1 contamination record; §7 G1b row",
    "problem": "Today's commit holds only §3 (G0) plus the seed grep. No freeze date before day 2 is given for:\n- the G1b room rule and se_proj;\n- the G2 and G3 thresholds;\n- the freeze-2 knobs;\n- the T and S definitions;\n- the route-S feature list and G0-S;\n- the B-clock specification;\n- the B1 verdict ladder.\n\nOn day 2 the G0 verdict is committed and the v83 exploratory result is read. The G1b rule is set after G1a results (day 3). Every later criterion could therefore be chosen knowing how cross-game consequence prediction behaved on these 41 games. This is exactly the garden-of-forking-paths risk the seal is meant to prevent.",
    "fix": "Commit the whole plan today, in the same commit as §3:\n- all gate thresholds and both verdict ladders (G1a, G1b, G2, G3, B0, B-clock, B1);\n- the T and S recipes;\n- the route-S features and G0-S;\n- the freeze-2 grid with its selection rule;\n- the seed table.\n\nList the only allowed later changes (the named V2 constants at G1a, the named freeze-2 knobs). Each must be a dated amendment that records what triggered it."
   },
   {
    "title": "G0 Σ recipe may be noise-dominated; no positive control on the real feature geometry",
    "severity": "blocking",
    "where": "§G0.6 source heads (ridge 1e-4 on the mean loss, û = w/‖w‖, w ⊙ sd₋g/sd_h); §G0.13 test 6; §G0.11 KC captions; §11 'a frozen 41-game mechanism negative'",
    "problem": "The 427 features are exactly collinear in at least about 83 directions:\n- `views.pool` (views.py:54-57) is mean pooling, so each channel's 4×4 and 2×2 pools are linear in its 8×8 pool (4 × 20 = 80 features);\n- the cone-bin one-hots each sum to 1;\n- the rotation and trait flags are collinear with each other.\n\nMany features are also nearly constant, scaled by the 0.05 floor (promise.py:29). A logistic head with ridge 1e-4 on the mean loss amplifies sampling noise along low-variance directions by roughly 1/(p(1−p)·c_i + λ). The w ⊙ sd₋g/sd_h remap inflates further the features that were floored in game h. Normalising û in parameter space then lets these noise directions dominate each rank-1 term, especially for heads with only 30 positives.\n\nThe same applies to:\n- K\\*, a rank-1 term from one head;\n- P, whose permuted-label heads have covariance proportional to the inverse input covariance, so P concentrates on the lowest-variance directions.\n\nThe synthetic test uses Gaussian games and cannot reveal any of this. A failure of (ii) or (iii) could then come from the recipe, yet §G0.11 words it as 'even the game's own structure does not make this learner 2× faster' or 'carried consequence structure does not make the learner faster', and route Σ closes for v84.",
    "fix": "Add a planted positive control to §G0.13 before the freeze.\n- Use the real kept A1 rows, standardised as for G0.\n- Draw synthetic labels from heads that share a 5-dimensional subspace (no real labels are used, so there is no contamination).\n- Run the identical pipeline and require R(K\\*:F) ≥ 2 and R(X : best control) ≥ 2.\n- Otherwise: Interruption, and the recipe is fixed before any real-label curve.\n\nAlso declare a better-conditioned source-head recipe now, for example:\n- a ridge chosen by cross-validation within each source game; or\n- normalisation in the data metric, û = w/√(wᵀĈw);\n- plus a fixed rule that drops exactly collinear features.\n\nNarrow the KC captions to 'this Σ recipe'."
   },
   {
    "title": "Sealed B1 omits V2 and U; live P is likely a weak placebo; T and S target search defaults, not the head prior",
    "severity": "major",
    "where": "§5.1 arms table; §7 G2 (KC-harm is only a flag), G3 GO, B1 arms and verdicts; §12 'faster than … a placebo prior with an identical spectrum'",
    "problem": "FASTER can pass in two ways that do not show carried knowledge.\n\n- **Steering harm repaired.** If isotropic steering harms (F slower than V2) and X only repairs that harm, X beats F, and T (which also has F's heads) and S without beating plain V2. V2 is not in B1, and R(F:V2) < 0.8 is only a flag at G2.\n- **Generic anisotropy.** U (input PCA), the strongest generic-shrinkage control at G0, runs only in development, on 2 seeds, as a report. A live advantage of X that is generic anisotropic shrinkage is therefore not excluded on the sealed games.\n\nIn addition, P's directions sit on near-null input directions (see the Σ-recipe issue), so after κ tuning P behaves like a crippled F, and lower bound of R(X:P) > 1 is nearly implied by R(X:F). T and S are P1's controls for a search controller (route S). They do not stand for 'designer-tuned head priors' on route Σ.",
    "fix": "- In B1, run V2 and U on 4 seeds each.\n- Require the lower bounds of R(X:V2) and R(X:U) to exceed 1 for FASTER-PARTIAL and FASTER-2×.\n- Put V2 into the 'better of' comparator set.\n- Make KC-harm a kill at G2, or require R(X:V2) in GO.\n- Add a placebo P_rot that applies a Haar-random rotation to Σ_X's eigenvectors (same spectrum, random directions).\n- At G0, report R(P:F) and R(P_rot:F), so a crippled placebo is visible."
   },
   {
    "title": "Top-rung censoring is built in; power and INCONCLUSIVE rules ignore it; the instrument line trips by noise",
    "severity": "major",
    "where": "§6 'Top rung k\\*' and 'Every measured rung was therefore reached by the fresh learner on both seeds'; §7 B1 INCONCLUSIVE; §7 power paragraph",
    "problem": "k\\* is the highest rung reached by both eligibility seeds, i.e. the minimum of two. With exchangeable seeds, a third seed falls below the minimum of the other two about 1/3 of the time (about 1/4 with discrete ties). So 25–33% of F runs in G2, G3 and B1 are expected to miss M_k\\* at the same 4M cap. That is close to the 40% INCONCLUSIVE rule, and those runs get capped λ terms that are generous to F.\n\nThe F-vs-F instrument line compares 3 seeds with 3 seeds over about 9 units. The sd of its log ratio is about √(2σ²/3/9): about 0.24 at σ̂² = 0.8 and 0.29 at 1.165. A perfect instrument therefore falls outside [0.67, 1.5] about 10–17% of the time, which voids the single-use sealed pool. Neither risk is in §7 or §11.\n\nThe stated power ('a true 2× passes FASTER-2× about half the time') covers only the R ≥ 2 point clause. It ignores the R_late, T,S, leave-one-game-out and r_g clauses.",
    "fix": "- Measure rungs up to k\\* − 1, or give arm runs a longer cap (for example 6M, still stopping at M_k\\*).\n- From the G1b runs, estimate the expected censoring and the probability of INCONCLUSIVE, and write them next to the power and odds.\n- Make the instrument line a caption, or set its band from its own sampling sd (for example |log ratio| > 3·se).\n- Restate power for the full FASTER-2× conjunction.\n- Reword 'reached by the fresh learner on both seeds' to 'on both eligibility seeds'."
   },
   {
    "title": "G0 K\\* leaks: even and odd blocks of one run share lineages",
    "severity": "major",
    "where": "§G0.4 'K\\* source' (B even blocks; test = B odd blocks; exact-tuple deduplication); §G0.11 (ii); capture ratio φ",
    "problem": "Go-Explore restores the same archived cells again and again across 50-iteration blocks. Rows from rollouts that start at the same cell in an even block and in an odd block are near-duplicates (same lineage, a few pixels apart), and the exact blake2b deduplication on (view, screen, slot, lives) misses them.\n\nK\\*'s direction is therefore fitted on rows almost identical to its test rows. This inflates R(K\\*:F) and φ, so condition (ii), 'the same-game ceiling shows room', can pass without real room. Judge 3's leak is only partly fixed. The house rule asks for folds grouped by shared start state.",
    "fix": "Build K\\* from a full-stream head on the learning stream A: the learner's own 'oracle future', on a different seed from the test set. Alternatively, exclude whole cell lineages, not exact tuples. Keep the B-block version as report only."
   },
   {
    "title": "G0 late clause does not separate a jump right after the first fit from a faster rate",
    "severity": "major",
    "where": "§G0.9 'Late' (D_a(q) = N_a(q) − N_a(0.25)); §G0.11 (iv) and the FIRST-RULES-ONLY caption",
    "problem": "Under the both-classes rule, every zero-mean arm stays intercept-only until the same row. L_0.25 = 0.5 + 0.25·(AUC_F,end − 0.5) is low (about 0.55 for an end AUC of 0.70), so on many units every arm will cross it at the first two-class checkpoint.\n\nN(0.25) is then the same constant for all arms, and D(q) is just N(q) minus that constant. R_late repeats the primary ratio, including any jump right after the first fit, which is the G0 analogue of a head start. Clause (iv) then cannot catch FIRST-RULES-ONLY.",
    "fix": "Count the late clause from N(0.5), over q ∈ {0.625, 0.75}, or from max(N(0.25), 2 × the first two-class checkpoint). Report the share of (unit, direction) pairs where N(0.25) ties across arms. Apply (iv) only to pairs where F's N(0.25) is strictly later than the first two-class checkpoint."
   },
   {
    "title": "Freeze 2 retunes only X and P on live development data, with no selection rule and unpaired seeds",
    "severity": "major",
    "where": "§7 G3 kill column (freeze 2); §8 seeds 9719–9722",
    "problem": "κ_X, κ_P × {¼, 4}, τ² ∈ {0.03, 0.3} and the rank cut r ∈ {10, 20} are chosen after G3 results are seen. κ_F and T are never retuned on live climbs, and no rule says how the new configuration is chosen.\n\nX and P rerun on 9719–9722 while F, T, S and V2 are reused from 9715–9718, so the pairing and the common random numbers are lost. An X-only regulariser retuned on live climbs would carry a tuned-defaults advantage into the sealed test that neither P (also retuned) nor T and S control. Two tries at GO also double the chance of a false GO in development.",
    "fix": "Retune κ_F on the same grid with the same rule. Write the selection rule now (for example, each arm's own mean λ on the G3 units). Rerun F, P and X together on 9719–9722. Report every configuration tried."
   },
   {
    "title": "G1a, a 10-game pilot, can stop the whole live route with an over-general caption",
    "severity": "major",
    "where": "§7 G1a row (STOP-fresh); house rule 'Never decide from a pilot of a few games'",
    "problem": "G1a runs 10 hand-picked A1 failures plus 2 controls, on one seed, with one revision allowed. A result of 3/10 or less gives STOP-fresh: no live arms, sealed pool closed, captioned 'budgeted savestate exploration with the senses does not climb on most Atari games at 4M'.\n\nA panel chosen for being the hardest cannot support 'most Atari games'. Letting this exploratory pilot kill the claim-bearing G1b (all 41 games, new seeds) goes against the project's pilot rule.",
    "fix": "Allow STOP-fresh at G1a only for build-level failures (controls fail their T(M3) test, throughput or commit limits fail). Otherwise always run G1b. If the kill is kept, word it 'does not climb on the A1-failure panel'."
   },
   {
    "title": "Contamination record is incomplete; the exploratory job is still running",
    "severity": "minor",
    "where": "§1 contamination record; §10 days 1–2; §9 RAM",
    "problem": "`F:\\dev\\ragnarok\\craft_v6_out\\ram_supervisor.log` lines 353 and 355 show SUSPEND of PID 21596 running `scripts\\explore_promise_v83.py` at 04:07:51 and 04:10:54 today. I checked by match counts and those two command lines only; the log holds no results. The job is therefore still alive, competing for RAM, and writing `explore/`. The script also prints its summary to stdout.\n\nThe proposals (03:24) and judges (03:46) files were written before the seal commit d19a3f9 (03:47); the declaration commit b9617d7 was at 03:06. The record says only that two judges saw one line.",
    "fix": "- Record that timeline.\n- State that no one tails the job's output.\n- When the job ends, write the SHA-256 of `explore_promise.json` and `run.log` into the G0 freeze and verdict commits without reading them.\n- Extend the sealed-path guard to any captured copy of its stdout.\n- Count its RAM in the day-1 worker plan."
   },
   {
    "title": "G0 decided by a percentile bootstrap over few game clusters",
    "severity": "minor",
    "where": "§G0.10, §G0.11 (iii) and (iv)",
    "problem": "With 8–12 game clusters, percentile-bootstrap lower bounds tend to undercover. The sign-flip permutation p is computed but is not part of PASS.",
    "fix": "Add a one-sided sign-flip p ≤ 0.05 to (iii) and (iv), or use a studentised or BCa interval. Declare it now."
   },
   {
    "title": "Synthetic null band conflicts with the conservative design; no rule if the design itself fails",
    "severity": "minor",
    "where": "§G0.13 test 6",
    "problem": "Under the null, the per-unit minimum over three controls and the asymmetric censoring (2·c_end for X versus c_end for controls) push R(X : best control) below 1, possibly below 0.8. The rule calls any failure 'a code bug', which invites a post-freeze change to frozen rules after a synthetic result.",
    "fix": "Test the null on R(X:F) and R(X:P) separately, or set the band from a simulated null distribution. Declare that any change after a synthetic failure is a dated amendment, committed before any real-data curve."
   },
   {
    "title": "Rows of the final, interrupted iteration have no iter_billed entry",
    "severity": "minor",
    "where": "§G0.3 clock; F:\\dev\\ragnarok_v83\\ragnarok\\agent_v83\\explorer.py (run loop)",
    "problem": "When `Spent` is raised mid-rollout, the `finally: self.rows.label(...)` block keeps that rollout's rows with `row_it = it`. But `it += 1` and `self.iters.add(...)` are skipped, so `iter_billed[row_it[i]]` is out of range for up to about 24 rows per stream. The frozen text covers only non-monotone clocks.",
    "fix": "Declare in §3 now: drop rows with `row_it ≥ len(iter_billed)`, and report how many."
   },
   {
    "title": "V2 row-keeping depends on outcomes",
    "severity": "minor",
    "where": "§4 Logs ('labels uses steps = the rollout's final planned length'); F:\\dev\\ragnarok_v83\\ragnarok\\agent_v83\\logs.py:85-95",
    "problem": "Rollouts are extended by 50 steps at each new archive-best S. With steps set to the final planned length, late rows survive censoring only when a later rise extended the rollout. That is the selection on outcome that v83 review A-1 forbids. It affects the live heads, G0′ and B-clock.",
    "fix": "Keep a row only if t + H ≤ the planned length in force at step t."
   },
   {
    "title": "B-clock is not a replication of G0, and its specification is ambiguous",
    "severity": "minor",
    "where": "§7 B-clock row; §12; house rule 'offline results are development evidence only'",
    "problem": "B-clock runs on V2 streams: probe-free 'me', play continues after a loss, rollout length varies. It uses 'Σ_dev' without saying which one: the G0 recipe from the A1 V streams, or the live Σ from V and V2 streams. Its κ was tuned for a different Σ, and the label rules differ. A pass is a new measurement, not a replication of G0. A sealed offline 'UNDERSTANDS-FASTER' also strains the house rule that offline results are development evidence only.",
    "fix": "At the full freeze, specify the Σ source, κ (retuned on calibration V2 streams by the G0.7 rule) and the label rules. Label a pass 'offline mechanism evidence, not competence'. Rename the failure label from NOT-REPLICATED to NOT-SHOWN-SEALED."
   },
   {
    "title": "Verdict labels do not fit route Σ",
    "severity": "minor",
    "where": "§7 B1 verdicts 5 and 8",
    "problem": "NEGATIVE TRANSFER fires when the upper bound of R(X : better of T,S) is below 1, even if X is at least as fast as F. TUNED-DEFAULTS reads 'carried knowledge indistinguishable from designer tuning', but T and S tune the search, not the heads.",
    "fix": "Trigger NEGATIVE TRANSFER on X:F (and X:V2) only. Add a separate BELOW-TUNED-SEARCH label. Reword TUNED-DEFAULTS for route Σ."
   },
   {
    "title": "Rungs can be reached by lineage depth beyond the bar's 7,000-step cap",
    "severity": "minor",
    "where": "§6 bars B′; G1b 'climbs on most dev games' label",
    "problem": "Bar episodes stop at 7,000 steps (F:\\dev\\ragnarok_v83\\scripts\\bars_v83.py:35). `agent_v71/env.py` sets no per-episode frame limit, and V2 now keeps playing after a life loss, so its savestate lineages have no depth limit. In games that do not end, M3 ≥ 5·B′ can be reached by survival depth alone. This is fair between arms but inflates the feasibility label.",
    "fix": "Report, per game, the lineage depth at M_k against 7,000 steps. Alternatively, compute the room label on rises counted within the bar's step horizon."
   },
   {
    "title": "Specification gaps to close in the frozen text",
    "severity": "minor",
    "where": "§G0.5, §G0.7, §G0.9, §G0.12, §5.2, §6 bars, §4 G-build",
    "problem": "- X-mean against the both-classes rule: G0.12 expects X-mean above 0.5 at the first checkpoint, but G0.5 forces every arm to stay intercept-only (v83 kept carried weights).\n- Which κ X_8 uses is not stated.\n- The κ-tuning curve count is 8 arms × 6 κ × 24 = 1,152, not 864, and the synthetic tests add time not in the 5 h.\n- B′ says 'four modes' and also 'each constant button': how many episodes per button?\n- D for a carried arm that misses N(q): 2·c_end, or 2·c_end − N(0.25)?\n- G0-S has a single head: is its AUC threshold 0.60 or 0.65?\n- No rule if an arm's best κ sits at a grid edge (4⁻⁵ or 1).\n- The equivalence harness cannot reproduce V exactly once probes are removed, because V probes while 'me' is lost.",
    "fix": "Settle each point in today's text. For the harness, use a smoke game where V made no probes, or keep a probe flag inside the harness."
   },
   {
    "title": "Translation 'in substance' and the rerun policy after an Interruption",
    "severity": "minor",
    "where": "§15 Today step 2; §G0.11 Interruption",
    "problem": "A French transcription 'in substance' leaves room for the frozen rules to drift. 'Fix the code and rerun' after an Interruption, possibly after curves have been seen, risks silent re-scoring (v83 lesson 7).",
    "fix": "Commit §3 verbatim in English with its hash, and make the French text a translation that defers to it. On any Interruption, keep and commit the first-pass outputs and label the rerun."
   },
   {
    "title": "Factual corrections to §1 and §8",
    "severity": "minor",
    "where": "§8 seed grep; §1 probe share",
    "problem": "- 'No output file name contains 96xx or 97xx in ragnarok' is literally false. For example, `F:\\dev\\ragnarok\\craft_v6_out\\v70_mspacman_calibration\\coverage_development_v1\\frames_09729_09984.npz` exists; these are frame ranges, not seeds, so it is harmless.\n- The claim about `.py` files holds: the only hits are in `venv310` (120 files) and `.venv-arc4` (36).\n- 'Probe share ≥ 64% on 7 games' means 7 development games (9 counting seaquest and asteroids).",
    "fix": "Reword the seed-grep line to 'no seed use'. Add the venv exclusion to the grep record. Say '7 development games'."
   }
  ]
 }
]
```
