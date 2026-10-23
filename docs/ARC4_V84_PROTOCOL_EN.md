# v84 plan (final): carry the questions, not the answers

**Pipeline.** v84 starts with G0, a gate that uses no emulator frames. It runs on the A1 logs and is preceded by a planted positive control on the real feature geometry. In parallel, a repaired fresh climber, V2, is built and goes through a claim-bearing room gate (G1b). At most one carried object then goes to live search climbs. It is measured against these controls:
- the no-carry explorer (V2);
- the tuned search defaults (T);
- the single best process (S);
- two structure placebos (P, U);
- a dose arm and a same-game ceiling (K\*).

The sealed pool is opened at most once.

**Editor's note.**
- I read and grepped only. I ran no Python, emulator or GPU job and edited no file.
- I read the brief, the judges' file, the P5 section of the proposals (§2 to §6), and these parts of the v83 code: `explorer.py:180-216`, `logs.py:40-100`, `views.py:40-70`, the `def` lines of `promise.py`, the `pools_v83.py` constants, `bars_v83.py:1-80`, and the `def` lines of `milestones_v83.py` and `climb_stats_v83.py`.
- I listed the top-level names in `F:\dev\ragnarok_v83\craft_v6_out\v83\`, so I know an `explore` folder exists. I neither listed nor opened anything inside it. I did not read the post-seal version of `ARC4_V83_EXPLORATORY.md`.
- Like every earlier reader of `v84_judges.md`, I have seen the line "good alien False" (§3.4).
- For the other proposals I relied on the draft and the judges.

---

## 0. Summary

- **What v84 decides first: G0, the knowledge clock (Part A).**
  - Cost: 0 frames and about 8–10 h of CPU. Verdict on day 2 (p ≈ 0.6) or day 3 (p ≈ 0.9).
  - Before any real label is used, a planted positive control on the real A1 rows must show that the recipe can detect shared structure.
  - Question: on the 82 A1 streams, does a zero-mean prior covariance Σ, learned from 40 other games, make a consequence learner reach its own end-of-stream predictive skill on a new game from fewer frames of the same experience?
  - Conditions:
    - X beats the per-unit best of F (tuned isotropic), P (permuted-label placebo, same spectrum) and U (input-PCA prior, same spectrum);
    - it holds on the late part of the curve, counted from each arm's own half-skill time;
    - the same-game ceiling K\*_A, built only from the learning stream, shows that there is room.
- **Built in parallel: V2.** A probe-free, Go-Explore-style climber with frame-code cells.
  - It keeps playing after a life loss and plays the bar's four button processes.
  - It logs rows to disk with outcome-free censoring.
  - G1a is exploratory and kills nothing except a broken build. G1b is the claim-bearing room gate on all 41 opened games at A1 strength.
- **At most one carried object goes live.**
  - Route Σ if G0 passes.
  - Route S (the search sense) otherwise, and only if G0-S passes.
  - Both objects are zero-mean, so the steered arms are byte-identical to F until the first two-class refit (certified).
- **Measures.**
  - Primary: the matched-rung **search climb with billed rewinds**, averaged over rungs and counted from each arm's own first rung.
  - Also required: a late version from the second rung, and an absolute-time guard from frame 0.
  - Separately labelled mechanism readout on the sealed games: the knowledge clock (PREDICTS-SOONER).
- **Odds (mine).**
  - G0 PASS ≈ 0.2.
  - V2 room on ≥ 13/35 development games ≈ 0.21.
  - Sealed SEARCH-CLIMB FASTER-PARTIAL or better ≈ 0.5% (0.2–1.5%); FASTER-2× ≈ 0.15%.
  - Most likely ending: a clean frozen G0 negative, a cost map for a probe-free climber, and the full control harness.
- **Budget.**
  - Billed frames: development ≤ 1.4G, sealed ≤ 1.1G, total ≤ 2.5G.
  - 3 workers, ≤ 1.2 GB each. GPU: 0 h.
  - Calendar: 14 days nominal. At the throughput A1 actually achieved, a chain that is still alive needs about 18–24 days, so an ordered cut ladder is declared now (§10).

---

## 1. Ground facts (checked against files; auditor checks marked †)

| Fact | Value | Source |
|---|---|---|
| A1 verdict | KC-room, 6/35 development games with room (alien, amidar, bank_heist, ice_hockey, jamesbond, qbert); calibration: breakout, ms_pacman; σ̂² 1.165, se_proj 0.480 | `room_verdict.json` |
| A1 streams | 82 npz (41 games × seeds 9131, 9132), code 2b22e94, unsteered (`steer` null) | `room_*.json`, `room_job_v83.py:78-82` |
| Rows | 5,434,791 in total; 3,600 per run (krull 9132) up to 99,103 (pong 9132); 18 runs < 24k rows† | npz headers |
| G0 store (lost rows dropped, capped at 24k) | 1,705,576 rows, about 1.46 GB in float16†. Shortest: atlantis 9132 (122), krull (864 and 2,197), up_n_down (3,888) | feasibility audit |
| Cut-rollout tail | When `Spent` is raised mid-rollout, the rows are labelled in `finally`, but `it += 1` and `iters.add` are skipped. 64/82 runs have max(row_it) == len(iter_billed)†, so up to 24 rows per run have no clock | `explorer.py:206-213` |
| RowLog dtypes | t, avail, rise_lag, drop_lag are uint8 (they would overflow for 400-step rollouts) | `logs.py:53-57` |
| Labels | hit = lag **<** H; kept iff t + H ≤ steps and (avail ≥ H or (hit and stops_on_event)) | `logs.py:85-93` |
| Features | 427 = 336 + 16 + 4 + 4 + 2 + 64 + 1†. `views.pool` is a mean pooling, so the 4×4 and 2×2 pools are linear in the 8×8 pool; together with the one-hot sums, about 83 directions are exactly collinear† | `views.py:54-57` |
| SD floor | 0.05 (v83 `SD_FLOOR`); v83 F used L2-SP with n0 = 50,000 | `promise.py:7,29` |
| Throughput | Per worker, median about 666 s per 2M billed frames (≈ 3.0k fps). Realised overall in A1: ≈ 10.3M billed frames/h (≈ 1.05 runs at once on 2 shards, 117 supervisor actions, 24 terminations that restarted from zero)† | run JSONs, supervisor log |
| Memory | Peak commit ≈ 0.82 GB + ≈ 4 KB per logged row (857 MB at 9.8k rows, 1,224 MB at 99.1k)†; working set 119–524 MB | run JSONs |
| Probe share | 26.7–26.8% of 164M; ≥ 64% on 7 development games (9 counting seaquest and asteroids) | A1 result |
| v83 bars | 10 episodes per constant button, plus uniform and repeat 0.95; 7,000 steps; sticky 0; 0–30 NOOPs; unbilled | `bars_v83.py:1-4,57-80` |
| Root sharing | Identical entry frames on both seeds for assault, fishing_derby, freeway, breakout, video_pinball | run JSONs |
| Exploratory job | `scripts/explore_promise_v83.py` (PID 21596) still running: SUSPEND events at 04:07:51 and 04:10:54 today (supervisor log lines 353 and 355; only command lines were checked)† | `ram_supervisor.log` |
| Seeds in use | 9130–9256 (v83), 9101–9111 and 9201–9211 (v82), 8101–8106 | `pools_v83.py`, v83 protocol §9 |

**Seed-block check (9700–9749).**
- `git ls-files '*.py' | xargs grep -nE '\b9[67][0-9]{2}\b'` is clean in all 9 worktrees†.
- A raw grep hits only third-party files in `venv310` (120) and `.venv-arc4` (36)†.
- Some output names contain 97xx as frame ranges, not seeds (for example `coverage_development_v1\frames_09729_09984.npz`). So the record reads **"no seed use"**, not "no occurrence".
- The grep is repeated at the protocol commit. The `explore` folder is checked only after the G0 verdict.

---

## 2. Audit resolutions

### 2.1 Blocking and major issues

| # | Issue (audit, severity) | Resolution |
|---|---|---|
| 1 | Only §3 was to be frozen today (can-it-fail, blocking) | **The whole plan is frozen today, in one commit.** That covers every gate threshold, both verdict ladders, T and S, the route-S features and G0-S, B-clock, Option M, the cut ladder and the seeds. Later changes are limited to the dated amendments listed in §3.3. Freeze 2 is removed (#8). |
| 2 | Σ recipe may be dominated by noise; no positive control on the real geometry (can-it-fail, blocking) | Part A.5–A.7 and A.15: (a) a shared, label-free reduced basis drops the numerical null space (eigenvalue ≤ 1e-5·λ_max of the source input covariance); (b) source heads exclude features floored in their own game; (c) source ridge chosen by 2-fold cross-stream validation; (d) **planted controls on the real A1 rows** (synthetic labels: a shared 5-dimensional subspace, and a null), run before any real label is touched; (e) one dated recipe amendment is allowed, a second failure gives **RECIPE-INSENSITIVE**; (f) KC captions read "with this Σ recipe". Normalisation in the data metric is rejected (#2.2). |
| 3 | G0 clock undefined for the rollout cut by the billed cap (feasibility, blocking) | A.3: rows with `row_it ≥ len(iter_billed)` are dropped, counted and tested (T2). |
| 4 | Sealed test lacks V2 and U; P may be a weak placebo; T and S are search defaults, not head priors (can-it-fail, honesty; major) | V2 (4 seeds) and U (4 seeds) are decision arms in G3 and B1. FASTER requires X to beat **the no-carry field N = best of {V2, T, S}** and **the structure placebos Π = better of {P, U}**, both chosen per game and per resample. P_rot (Haar-rotated Σ_X) and a WEAK-PLACEBO caption are reported at G0. On route Σ, the tuned head-prior defaults are F with per-arm tuned κ plus U; TUNED-DEFAULTS is renamed BELOW-NO-CARRY-FIELD. |
| 5 | Censoring at the top rung, an instrument line that trips by noise, optimistic power (can-it-fail, major) | Arm runs get a **6M cap** and still stop at M_top (eligibility stays at 4M). The F-vs-F instrument gives INCONCLUSIVE only if its **99%** interval excludes 1. Censoring forecast and power simulation (full conjunction, measured σ̂²) are written at the sealed freeze. Power is restated as an upper bound (§7.6). Wording fixed to "rungs reached by V2 on both eligibility seeds". |
| 6 | K\* leaks at G0 (can-it-fail, major) | **K\*_A** = a head fitted on the whole learning stream A (the learner's own "oracle future"), scored on the other seed's test set. The B-even-block version is report-only. |
| 7 | The G0 late clause does not separate a jump at the first fit (can-it-fail, major) | The late clause is counted **from each arm's own N(0.5)**, over q ∈ {0.625, 0.75}, with p ≤ 0.05. The share of (unit, direction) pairs whose N(0.25) ties across arms is reported. |
| 8 | Freeze 2 retunes only X and P, with no rule and broken pairing (can-it-fail, major; honesty, minor) | **Freeze 2 is removed.** Live κ is retuned **for every steered arm** (F, X, P, U, K\*) on day 5, offline, on the V2 calibration streams by the A.9 rule, before any live arm. G3 is a single shot: GO or KC-dev. Seeds 9719–9722 are reserved and unused. |
| 9 | G1a (a 10-game pilot) could stop the whole live route (can-it-fail, honesty; major) | G1a is exploratory and **cannot kill**, except through build failures (control regression after a code fix round counts toward KC-build). G1b always runs. The G1a caption is scoped to "the A1-failure panel". |
| 10 | Compute is about 2–3× underestimated (feasibility, major) | §10 gives nominal and A1-realised walls, an average arm cost measured at G2, an ordered cut ladder C1–C4 triggered only by projected calendar, and **deterministic restarts** (a killed run is rerun from zero with the same seed and gives identical output, tested at G-build). |
| 11 | RAM limits unreachable with v83 logging; engineering misses could kill science (feasibility, major) | Rows go to disk in 8k-row chunks (no final copy); explorer workers use OMP/OPENBLAS/MKL = 1 thread. Limits and worker count are set from a measured 4M smoke. Commit size and fps are **engineering checks (fix and rerun), never STOP or KC**. fps is measured on CPU time, excluding suspensions. |
| 12 | Byte-exact V2↔V equivalence is impossible without probes; a v83-style hook breaks prefix identity (feasibility, major) | The equivalence harness is replaced by a scripted-trajectory test (ledger, Trace, rows, labels) and a determinism test. Prefix identity requires inactive steering to call **V2's own draw function, with no extra RNG draws**. Only failures of V2's own tests count toward KC-build. |
| 13 | Rollout extension makes censoring depend on outcomes; uint8 overflow (feasibility, honesty; major) | A row is kept iff t + H ≤ **L_t** (the planned length in force at step t, logged per row). t and avail become uint16. Test: the kept mask is unchanged when events after t are altered. |
| 14 | The B1 and G0 ladders have gaps and a wrong label (honesty, major) | New labels: EARLY-RUNGS-ONLY, NOT-SIGNIFICANT, BELOW-NO-CARRY-FIELD, UNATTRIBUTED-GAIN. NEGATIVE TRANSFER fires only on X:F or X:V2, and is tested before any positive. HEAD-START is a caption that can accompany any label. G0 gets SUB-THRESHOLD and a reworded KC-knowledge-speed (Part C). |
| 15 | B-clock named UNDERSTANDS-FASTER, has two outcomes, undefined under route S (honesty, major) | Renamed **PREDICTS-SOONER-SEALED** ("offline mechanism evidence, not competence"), with the full label set (NOT-SHOWN-SEALED, CLOCK-NOT-MEASURABLE, G0 captions). Under route S it runs the G0-S recipe on the sealed expansion logs. A B1 × B-clock headline table is written now (§13.4). |
| 16 | Route S has no claim sentence (honesty, major) | §13.2 gives route S its own sentence and object tag ("search sense: carries search configuration, not knowledge of game consequences"). |
| 17 | The primary measure was called "play" (honesty, major) | Renamed **search climb with billed rewinds**; labels are SEARCH-CLIMB FASTER-…; the v83-style claim header goes into the protocol and every verdict file (§3.1). |
| 18 | No absolute-time guard (honesty, major) | FASTER-PARTIAL requires the point estimate R_abs(X:F) ≥ 1, and FASTER-2× requires its lower bound ≥ 0.9, where R_abs is the time to M_top from frame 0. LATE-START caption if the upper bound of HS < 1. HS reported with its interval. |
| 19 | Contamination record incomplete; unseal rules disagree; job still running (honesty, major; others, minor) | §3.4: full timeline and list of readers; the seal is extended to stdout and notifications; the job is left to finish unread and its outputs hashed unread; the unseal point is **after `g0_verdict.json` is committed** (dated amendment to the v83 seal note); its RAM is counted. |

### 2.2 Minor issues

**Adopted:**
- sign-flip p ≤ 0.05 added to G0 (iii) and (iv);
- one-sided synthetic null bands, and a rule after a planted failure;
- X-mean exempt from the both-classes rule (it is report-only);
- X_8 uses κ_X; κ grid-edge rule; corrected curve counts;
- B′ episodes defined; G0-S threshold 0.60; censoring defined for D;
- English text authoritative, French a translation;
- Interruptions keep and commit their first-pass outputs;
- live κ mismatch fixed (day-5 retune);
- room label "climbs on most" computed with a 7,000-step lineage-depth cap;
- senses and inputs section; full v83 isolation test ported; "record" = archive-best sensed S;
- "fresh" reworded; near-sibling line plus the leave-one-game-out minimum in PARTIAL; novelty reworded;
- "G0 certifies the recipe" stated; Option M written in full; power restated;
- G3 run count corrected; development-cap rule mid-gate; factual corrections (§1).

**Rejected, with reasons:**
- **Normalising source heads in the data metric (û = w/√(wᵀĈw)).** It gives more weight to heads whose weight lies along low-variance, noisy directions. Cross-validated ridge, exclusion of floored features and null-space projection address the cause, and the planted control checks them.
- **BCa or studentised intervals.** The sign-flip p is simpler and is added instead.
- **P_rot as a decision comparator.** Adding a fourth control to the per-unit minimum makes the null bias against X stronger. It is reported instead, and U covers generic anisotropy.
- **Measuring only up to k\*−1.** That leaves the late clause undefined on k\* = 3 units; the 6M arm cap is used instead.
- **A lineage-depth cap on arm T(M).** It is fair between arms; the cap applies only to the "climbs on most" label.

---

## 3. Claim header, inputs, freeze, contamination

### 3.1 Claim header (in the protocol and in every verdict JSON, field `claim_header`)
- **FR:** « Toute affirmation de v84 porte sur une RECHERCHE AVEC RETOURS EN ARRIÈRE FACTURÉS (savestates, 60 trames par chargement, chaque trame émulée facturée) et utilise les SENS accordés le 22 octobre 2026 (hausses du score, nombre de vies). Ce n'est pas du jeu en direct sans savestates. »
- **EN:** "Every v84 claim concerns a SEARCH WITH BILLED REWINDS and uses the SENSES granted on 22 October 2026 (score rises, lives count). It is not live play without savestates."

### 3.2 Agent inputs (field `senses` in every verdict JSON)
- **Allowed:** screen pixels; the agent's own buttons; the per-step positive part of the 4-frame reward, max(0, r_step), with its magnitude; `ale.lives()`.
- **Game over** is an episode boundary only: never a label, a variable or a veto.
- **Evaluator only:** RAM, true score, game names, `clone()` internals.
- **No** downloaded weights. Reserved games (boxing, frostbite, riverraid, kangaroo) are never opened; enduro is excluded.
- **Ladder quantity:** the sum over the lineage's agent steps of max(0, r_step). This is exactly what the agent senses, computed on the evaluator side from the Trace. Net score is reported alongside; penalties are ignored by the ladder, and this limitation is stated in every report.

### 3.3 What is frozen today and what may change later
**Frozen today (23 Oct 2026):** this entire document, committed as `docs/ARC4_V84_PROTOCOL_EN.md` (authoritative, with its hash). `docs/ARC4_V84_PROTOCOL.md` is the French translation and says that it defers to the English text.

**Allowed later changes.** Each one is a dated amendment that records its trigger, committed before the data it affects:
1. G1a′: one revision of the named V2 constants (§8, G1a row).
2. Recipe amendment after a planted-control failure (A.15 T8), once, before any real-label computation.
3. Cut ladder C1–C4 (§10), triggered only by projected calendar or throughput.
4. Engineering changes that do not alter agent behaviour or criteria: resource gates, lazy imports, disk logging, worker count.
5. Interruptions: first-pass outputs kept and committed, the rerun labelled as such.

Nothing else changes: no threshold, arm, seed, feature list or label.

### 3.4 Contamination record and seal policy
**Timeline (today):**
- 03:06: commit b9617d7 declares the v83 exploratory analysis.
- 03:07: `explore_promise_v83.py` starts (PID 21596).
- 03:24: proposals written. 03:46: judges written.
- 03:47: seal commit d19a3f9.
- 04:07:51 and 04:10:54: the supervisor suspends PID 21596, so the job is still alive.

**What leaked:** one log line, "good alien False", seen by judges 1 and 2 in `explore/run.log` and reproduced in `v84_judges.md`. Every reader of that file therefore saw it before this freeze: the synthesizer, the three auditors and this editor. The G0 thresholds come from P5 (written before the line was seen). They were hardened afterwards, and that hardening is recorded here.

**Other events on record:**
- Judge 2's stray `python -c "1"` (a no-op).
- This editor listed the top-level names of `craft_v6_out\v83\` (not the contents of `explore`).

**Policy:**
- The job is left to finish. Stopping it would be Jérémie's call, and it is not proposed.
- Nobody tails its output or reads its stdout, console capture or completion notification. If a notification surfaces automatically, the time and the reader are recorded as a contamination event.
- When it ends, the SHA-256 of `explore_promise.json` and `run.log` go into the next commit, computed without displaying content.
- The sealed-path guard covers `craft_v6_out\v83\explore\` and any captured copy of the job's stdout.
- The G0 store never reads `explore/promise_store`.
- **Unseal point:** only after `g0_verdict.json` is committed. The v83 seal note gets a dated amendment saying so. The result is then reported with the label "exploratory".
- Its RAM (0.5–1.3 GB working set) is counted: G0 runs on 2 workers until the job exits.

---

## 4. Shape of v84

```
Day 0      Freeze the whole plan (EN authoritative + FR translation), seed grep, contamination record
Day 1-3    G0: store -> tests -> PLANTED CONTROLS -> review -> sources -> kappa -> units -> arms -> verdict
Day 1-4    V2 build -> G-build tests -> bars (9706) -> G1a (exploratory, no kill) -> 2-agent review
Day 4-6    G1b room A1' (41 games x 2 seeds x 4M, claim-bearing)
Day 6      offline: live Sigma / search-sense sources, live kappa retune (all arms), T, S; G0' (report); G0-S
Day 7-9    G2 WORTH (V2, F, K*) on <= 8 room dev units
Day 9-12   G3 rehearsal (X, P, U, T, S, X_8; V2 and F reused) -> GO or KC-dev
Day 12-13  3-agent review, sealed freeze (hashes, power and censoring forecast, cuts)
Day 13-15  B0 sealed bars + eligibility -> B-clock committed -> B1 sealed arms (once)
Day 16+    verdicts, report (FR), handoff   [nominal; at A1-realised throughput, see section 10]
```

Rules that hold throughout:
- One live route and one sealed object per v84.
- Every carried object has mean zero, so live arms are prefix-identical to F until the first two-class refit (a certified test).

---

## 5. The fresh climber V2 (shared by every live arm)

**Reused from v83:** SenseEnv (billed ledger, 60 frames per restore, evaluator Trace) and the entry (`Body.enter`: background, SignatureAdapter, "me" seeding, passive tracker). The entry is identical for all arms.

| A1 lesson | V2 change |
|---|---|
| 1. Climb first | G1b (claim-bearing, 41 games) comes before any carried arm. Eligibility cap 4M (33/58 A1 runs were still improving after 1.5M). |
| 2. Bar processes | Four processes, each one a bar mode: **Q1** a new uniform slot every step; **Q2** held, repeat 0.95; **Q3** held, repeat 0.75; **Q4** long hold, repeat 0.995, inheriting the button with probability 0.5, otherwise a uniform slot. Buttons are drawn over distinct canonical slots. A Thompson bandit on ADV, Beta(1 + ADV, 1 + not-ADV), discounted 0.995 per expansion with a 20% uniform floor; propensities logged. |
| 3. Probes | **None after entry.** "Me" comes from `Body.track` only. Probe share (≈ 0) and me-lost fraction are reported. |
| 4. Lives and reachable rungs | A life loss archives the post-loss state (lives are in the key) and play continues. A rollout stops at its second loss or at game end (a boundary only). The ladder uses cumulative sensed rises; only rungs reached by V2 on both eligibility seeds are measured. |
| 5. Progress axis | Cell = (lives, s_bin(S), code_r). code_r is the block-averaged luminance frame (0.299R + 0.587G + 0.114B, `np.add.reduceat` edges at floor(i·H/h)), quantised to d levels, with R = {(8,6,4), (11,8,8), (16,12,8), (22,16,8)} as (w, h, d). All four codes are stored. Every 250 expansions, the r whose distinct-key count is closest in log to clip(30√n, 64, 8000) is chosen and the archive is re-keyed (better cell wins: S, then lives, then shorter depth). This adaptive downscaling is Go-Explore's own (Ecoffet et al. 2021). |
| 6. Noise | λ averaged over rungs; 6 sealed seeds for X and F; σ̂² winsorised; F-vs-F instrument line. |

- **Selection.** w_i = (1 + chosen)^(−1/2) · 2^((sbin − sbin_max)/2) · 2^(−(L_max − L_i)) · (2 if the cell was added or improved in the last 100 expansions).
  - With probability 0.25, a frontier draw replaces it: uniform among the cells at sbin_max that have the most lives.
- **Rollouts.** 100 steps, extended by 50 at each new archive-best S, up to 400. Cap 20k cells, v83 eviction rule.
- **ADV.** The rollout created a cell above its start cell's sbin, or a new cell in the top sbin.
- **Logs.**
  - Row logging (written to disk in 8k-row chunks):
    - one row every 4th step, in v83 format with t and avail as uint16;
    - each row also carries **L_t**, the planned rollout length at step t;
    - **labels rule:** kept iff t + H ≤ L_t and (avail ≥ H or (hit and stops_on_event)).
  - One expansion row per rollout: search-sense features (§6.2), process, ADV, billed frames, propensity.
  - The A1 streams (G0) keep v83's `logs.labels` with steps = 100, byte for byte.
- **Determinism.** Given (game, seed), a run is fully determined. A run killed by the supervisor is rerun from zero with the same seed and must give identical outputs.

**G-build tests.** Only failures of tests 1–6 count toward KC-build: two failed fix rounds give KC-build.
1. **Isolation.** The full v83 isolation test, ported to `agent_v84`, `consequence.py` and `critic.py`, and still run on the reused `agent_v83` modules:
   - no `evaluation` imports and no game names;
   - no private-attribute access on handed objects;
   - the v83 BAD_ATTR list (ram, getRAM, true_score, ale, cloneState and the rest);
   - no `eval` and no dynamic `getattr`.
2. Ledger never decreases; each restore is billed 60.
3. **Scripted trajectory.** A fixed sequence of buttons and restores through SenseEnv yields an identical ledger, Trace, rows and labels (with L_t ≡ 100) in V2's logging path and in v83's.
4. **Label causality.** The kept/censored mask is unchanged when events after t are altered (synthetic sequences, extensions included).
5. **Prefix identity.** A steered arm with zero-mean Σ equals V2 byte for byte until the first refit that sees both classes. Inactive steering calls V2's own draw, with no extra RNG draws.
6. **Determinism.** Two smoke runs (seed 9700, 50k frames) are identical.
7. **Engineering, fix and rerun, never a kill.** 4M smoke runs, unsteered and steered: peak commit, working set, fps on CPU time. Targets: ≤ 1.0 GB unsteered, ≤ 1.2 GB steered. If missed, the worker count becomes floor(3.6 GB / measured).

---

## 6. Carried object and arms

### 6.1 Route Σ (only if G0 PASS and G1b PROCEED)
- **Heads.**
  - The v83 GOOD/BAD heads over 427 features, in the reduced basis Q_src computed from the source inputs (A.5).
  - Refit every 250 expansions on a reservoir of at most 40k rows (float16), by MAP with `consequence.py`, the same code as G0.
  - Standardisation S_src; live κ_a from the day-6 retune.
  - Both-classes rule; BAD stays inactive until a life loss is sensed.
  - Rows where "me" is lost are dropped. While me is lost, V2's own law steers; this dilution counts against X.
- **Steering.**
  - **Cells:** draw 16 candidates with V2's weights and pick one with probability ∝ exp(clip(z, −2, 2)). Here z is the robust z-score of u(c) = max over slots of [P(GOOD) − P(BAD)], computed on the cell's archived view. Frontier draws are never steered.
  - **Buttons:** at rollout start and at each button change in Q2–Q4, use 0.5·uniform over slots + 0.5·softmax(logit GOOD − logit BAD). Q1 is never steered.
  - With inactive heads this is exactly V2 (test 5).
- **Σ sources.**
  - Part A recipe (A.5–A.8) on the A1 (V) and A1′ (V2, seeds 9712–9713) streams: 4 streams per game, equal game weight.
  - Leave-one-game-out in development; all 41 games for the sealed test.
  - K\* live: a head on the game's own A1′ rows.
- **Live κ retune (day 6, before any live arm).** The A.9 rule for F, X, P, U and K\* on the A1′ calibration streams (V2 rows, L_t rule). X_8 uses κ_X; T uses κ_F.
- **G0 certifies the recipe, not the live object.**
  - G0′ (the Part A recipe on the A1′ development streams) is report-only. It gets an AGREE or DISAGREE caption and can neither reopen nor close route Σ.
  - DISAGREE means one of two cases:
    - G0 passed, but R(X:C\*) on A1′ is below 1;
    - G0 failed, but R(X:C\*) on A1′ is at least 1.5 with a lower bound above 1.1.

| Arm | Definition | Runs in |
|---|---|---|
| V2 | No heads, flat bandit | G2, G3 (reused from G2), B1 (4 seeds) |
| F | V2 + isotropic heads (κ_F); the baseline learner, with cross-game feature scaling and tuned κ | G2, G3, B1 (6 seeds) |
| T (tuned search defaults) | F + a clock-only process schedule: per process × clock bin b = min(5, ⌊log2(1 + billed/125k)⌋) × any-rise flag, prior Beta(n0·p̂, n0·(1 − p̂)) with n0 = 50 and p̂ = equal-game mean ADV rate in the A1′ logs (leave-one-game-out in development, all 41 for sealed) | G3, B1 (4) |
| S (single best process) | V2 with only q\* = argmax of equal-game ADV per billed frame in the A1′ logs (leave-one-game-out in development, all 41 for sealed); no bandit, no heads | G3, B1 (4) |
| X | F + Σ_X (κ_X) | G3, B1 (6) |
| P | F + Σ_P (labels permuted within each run, spectrum-matched, κ_P) | G3, B1 (4) |
| U | F + Σ_U (input-PCA eigenvectors, X's spectrum, κ_U) | G3, B1 (4) |
| X_8 | F + Σ from 8 source games (fixed draw, seed 9704), κ_X | G3 (2 seeds), dose caption |
| K\* | F + same-game Σ | G2 only |

**Comparator sets** (chosen per game, and again in every bootstrap resample, v83 build note 17):
- **N** (no-carry field) = the fastest of {V2, T, S};
- **Π** (structure placebos) = the faster of {P, U}.

### 6.2 Route S, the fallback (only if G0 gives any KC or RECIPE-INSENSITIVE, G1b PROCEED, and G0-S PASS)
- **Object.** A zero-mean Σ_s over the slopes of a logistic expansion critic on ADV, with a free intercept per game. The features, frozen now, are:
  - S-rank, lives gap, log depth rank, log(1 + chosen), smoothed yield and parent yield, log expansions since the cell was added, log steps since the last rise, after-loss flag, coarse-code rarity, share of pixels changed, log occupancy of its sbin;
  - process one-hot (4) and the first button's slot class;
  - budget share, ADV rate over the last 200 expansions, life-loss rate, stagnation, any-rise flag, **log billed frames since the last increase of the archive-best sensed S**, new-cell rate;
  - adapter signature (number of movement slots, whether a spawn slot exists);
  - interactions of process with {lives gap, stagnation, loss rate, S-rank}.
- **G0-S (day 6, offline, on the A1′ expansion logs).** The Part A machinery with:
  - learning stream = one seed's expansions; test = the other seed's odd 50-expansion blocks; K\*_A = a head on the whole learning stream;
  - controls C\* = per-unit best of {F, P (ADV fully permuted within each run), U, T_clock (Σ from source heads fitted on the clock and run-context features only)};
  - units = development games (one head); AUC_F,end ≥ 0.60; condition (i) becomes ≥ 12 games;
  - all other thresholds as in A.13.
  - G0-S is computed and reported on day 6 whatever happens.
- **Live.** P3 steering: draw 32 cells with V2's weights, score every (cell, process) pair, pick ∝ exp(1.5·clip(z, −2, 2)), mixed at 0.5 with V2's own draw; refit every 100 expansions.
  - Arms F_S, X_S, P_S, U_S, with T = the T_clock prior, plus S and V2 as in §6.1.
- **No route switch after day 6.**

---

## 7. Live yardstick and statistics (evaluator side)

### 7.1 Bars B′
- Seeds: 9706 (development + calibration), 9730 (sealed), 9740 (second round).
- Modes: each constant button (10 episodes per button; the best one is the mode value), uniform per step, held 0.95, held 0.75. 10 episodes each.
- Conditions: all lives, sticky 0, 0–30 NOOPs, 7,000 steps, minimal action set, fresh emulator per episode, unbilled.
- Recorded per episode in cumulative sensed rises and in true score.
- B′ = max over modes of the mean episode rises. u′ = max(0.5·B′, median positive per-step rise). Rungs M_k = B′ + 2^k·u′, for k = 0..10.

### 7.2 Times and rungs
- **T(M).** Billed frames from frame 0 (entry included) at the first clone whose cumulative sensed rises reach M.
  - Replay-confirmed for M1, M2 and M_top with the note-21 replay (one NOOP before restore). A mismatch is an Interruption.
- **k\*.** The highest rung reached by V2 on **both** room or eligibility seeds within 4M, fixed before any arm runs. A unit needs k\* ≥ 3. M_top = M_k\*.
- **Arm runs.** Cap 6M billed frames; evaluator stop at M_top.

### 7.3 Per run
- λ = mean over k = 2..k\* of log max(5,000, T(M_k) − T(M_1)).
- λ_late = the same over k = 3..k\*, counted from T(M_2).
- λ_abs = log T(M_top), from frame 0.
- **Censoring (v83 `climb`).**
  - A carried arm (X, P, U, X_8, K\*) that misses a rung counts the cap (6M).
  - A comparator (V2, F, T, S) counts cap − T(M_start); without its start rung it gets the 5,000 floor.
  - For λ_abs, a miss counts the cap for every arm (neutral, guard only).
  - P and U count as carried for censoring, which is conservative for X:Π. Declared.

### 7.4 Per game and inference
- ℓ_g = mean λ(comparator) − mean λ(X); r_g is clipped to [1/8, 8]; R = exp(mean over games of ℓ_g).
- Two-level bootstrap (games, then seeds within arm), 2,000 resamples: seed 9714 in development, 9739 sealed.
- One-sided permutation p: X and F labels permuted within each game, 10⁴ Monte-Carlo draws.
- Also computed:
  - the leave-one-game-out minimum;
  - HS = geometric mean of T_F(M1)/T_X(M1), with its interval (reported, never a claim; LATE-START caption if its upper bound < 1);
  - a per-rung ratio profile for each arm.
- **Instrument line.** F on 9733–35 against F on 9736–38. INCONCLUSIVE if its 99% bootstrap interval excludes 1.
- **Reported:**
  - true net score (pong, fishing_derby, ice_hockey, tennis, …);
  - me-lost fraction, steering-active fraction, censoring;
  - lineage depth at each M_k against the 7,000-step bar horizon;
  - per-game table;
  - a "without near-siblings" line (pacman ~ ms_pacman; frogger ~ freeway; atlantis2 ~ atlantis);
  - capture ratio φ = log R(X:F) / log R(K\*:F), from development.

### 7.5 Written now
- X's per-rung ratio falls with k, so R_late < R is expected.
- U ≈ F live.
- X_8 < X (dose).

### 7.6 Power (upper bounds; a simulation with the measured σ̂² is committed at the sealed freeze and changes no criterion)
- With 8 units, 6 seeds for X and F, and σ̂² = 0.8–1.2, se(log R(X:F)) ≈ 0.18–0.22, plus heterogeneity between games.
- At a true 2×, the R ≥ 2.0 point clause alone holds about 50% of the time; the **full FASTER-2× conjunction passes ≤ 35%**.
- At a true 1.5×, FASTER-PARTIAL passes ≤ 40%.

---

## 8. Gates in order

| Gate | Data, arms, seeds | Pass | Kill and sentence |
|---|---|---|---|
| **G0** knowledge clock | Part A (A1 logs, 0 frames) | A.13 | KC-knowledge-room / -worth / -speed or RECIPE-INSENSITIVE: route Σ closed (Part C) |
| **G-build** | Tests 1–7 in §5, smoke 9700 | Tests 1–6 pass; test 7 sets the worker count | **KC-build** after 2 fix rounds: "V2 as specified could not be built to pass its own tests; nothing is claimed about climbing" |
| **G1a** (exploratory, **no kill**) | V2, seed 9710, cap 4M. Panel: atlantis, krull, centipede, video_pinball, gopher, name_this_game, zaxxon, bowling, seaquest, phoenix. Controls: ms_pacman, amidar | Reported: M1 and M3 counts out of 10; controls' T(M3) against 2× V's A1 median re-scored on the v84 ladder | If < 6/10 reach M3: one revision of named constants (frontier probability, extension rule, re-key target constants, bandit discount, floor), G1a′ on 9711. The configuration with more M3 reaches is kept (ties: the original). If the controls fail in both configurations, it counts as a build fix round. Caption: "V2 at constants c reaches M3 on k/10 games of the A1-failure panel". G1b always runs. |
| **G1b** room (claim-bearing) | V2, 41 games, seeds 9712–9713, cap 4M | Per game, both seeds: (a) ≥ 10 sensed rises in the first 300k; (b) H = min over seeds of best rises ≥ M3; (c) median T(M1) ≤ 1.5M; (d) median T(M_k\*) − T(M1) ≥ 300k. **PROCEED** if ≥ 13/35 development games AND se_proj = √((σ̂²/3 + 0.09)/(17·n/35)) ≤ 0.23, with σ̂² the pooled (λ₁ − λ₂)²/2 over room development games, winsorised at the 90th percentile. Label "climbs on most dev games" if ≥ 18/35 with M3 reached at lineage depth ≤ 7,000 steps | **KC-room-84:** "V2 at its frozen constants does not reach A1-strength room (M3 on both seeds within 4M) on ≥ 13 of 35 development games." No transfer arms; the cost map is kept; Option M may apply (§16) |
| **Day-6 offline** | Live Σ and search-sense sources, live κ retune, T, S; G0′ (report); G0-S if route S | G0-S: A.13 thresholds as adapted in §6.2 | **KC-search-sense** (route S only): "the cross-game search sense does not speed the offline learning of expansion yield beyond tuned controls." v84 ends at feasibility |
| **G2** WORTH (live, development) | Room development games, ≤ 8 (drawn with 9714 if more; C1: 6). V2, F, K\* on 9715–9718 | R(K\* : better of V2, F) ≥ 1.75 with lower bound ≥ 1.2, and r_g ≥ 1.5 on ≥ 1/3 of units | **KC-worth-live:** "on these units, same-game prior steering at the frozen κ does not speed the climb by 1.75×; the head prior is not this climber's bottleneck at this setting"; sealed stays closed. Flag **KC-harm** if R(F:V2) < 0.8 |
| **G3** rehearsal (single shot) | Same units. X, P, U, T, S on 9715–9718 (V2 and F reused, paired); X_8 on 9715–9716 | **GO** (all): R(X:F) ≥ 1.4 with lower bound > 1; R(X:N) ≥ 1.2 with lower bound > 0.9; lower bound of R(X:Π) > 1; lower bound of R_late(X:F) > 1; point R_abs(X:F) ≥ 1 | **KC-dev:** "the carried object does not pass the development rehearsal"; sealed stays closed |
| **Sealed freeze** | — | 3-agent adversarial review. Committed: SHA-256 of Σ_dev(41), Σ_P, Σ_U, the T schedule, the S choice, live κ and code; power simulation; censoring forecast (from G2/G3); cut decisions | — |
| **B0** sealed eligibility | Bars 9730; V2 on 9731–9732, cap 4M; G1b rule (a)–(d) per game; `sealed_eligibility.json` and its hash committed before any arm | ≥ 6 units. If more than 8 (C3: 6), draw with 9749 | Fewer than 6: second round once (bars 9740, eligibility 9741–9742), units pooled. Still fewer than 6: **NOT MEASURABLE** |
| **B-clock** (mechanism, separate label) | Route Σ: Part A recipe on the B0 V2 streams (L_t rule), Σ_dev(41), live κ; K\*_A reported. Route S: the G0-S recipe on the B0 expansion logs | **PREDICTS-SOONER-SEALED:** ≥ 8 units on ≥ 6 games; R(X:C\*) ≥ 1.5 with lower bound > 1.1 and p ≤ 0.05; lower bound of R_late > 1 with p ≤ 0.05; ceiling guard. Committed before the B1 verdict is computed | NOT-SHOWN-SEALED or CLOCK-NOT-MEASURABLE (kept) |
| **B1** sealed live (once) | X, F on 9733–9738; T, S, V2, P, U on 9733–9736 (C4: 3 seeds); cap 6M; stop at M_top. Second-round units use 9743–9748 | Verdict ladder (Part C) | — |

---

## 9. Pools and seeds

- **35 development + 6 calibration games** (all burned by A1). Used for:
  - G0 (development = units, calibration = κ);
  - G1a, G1b, G2, G3;
  - the sources for Σ, T and S;
  - the planted controls (inputs only).
- **24 sealed games:** opened only at B0, after the sealed freeze. The 7 hard-exploration games (montezuma_revenge, pitfall, venture, private_eye, solaris, gravitar, skiing) are assumed ineligible, so G_proj = 17.
- **9 second-round games:** only for the B0 top-up or Option M.
- **Never opened:** boxing, frostbite, riverraid, kangaroo. **Excluded:** enduro.

| Seed | Use |
|---|---|
| 9700 | Smoke runs and test row draws (never a measurement) |
| 9701 / 9702 / 9703 / 9704 / 9705 | G0 row thinning / test draw / label permutation / dose draws / bootstrap and p |
| 9706 | Development + calibration bars |
| 9707 | G0 P_shift (report) |
| 9708 | Planted controls (game draw, subspace, labels) |
| 9709 | P_rot rotations (report) |
| 9710 / 9711 | G1a / G1a′ |
| 9712–9713 | G1b room runs, A1′ source logs, K\* sources |
| 9714 | Development unit draw, live placebo permutation, development bootstrap |
| 9715–9718 | G2 and G3 arms (paired) |
| 9719–9729 | Reserve (9719–9722 formerly freeze 2; unused) |
| 9730 / 9731–9732 / 9733–9738 / 9739 | Sealed bars / eligibility / arms / bootstrap and p |
| 9740 / 9741–9742 / 9743–9748 | Second round: bars / eligibility / arms |
| 9749 | Sealed unit draw (if more than 8) |

---

## 10. Compute, RAM, GPU, cut ladder

**Planning basis.**
- Per worker: 3.0k fps unsteered (A1 median) and 2.0k fps steered (measured at G-build).
- "Nominal" assumes 3 concurrent workers. "A1-realised" applies A1's achieved overall rate (about 0.36 of nominal, so ×2.8 on wall time).
- Average arm run cost: 3.2M (6M cap, stop at M_top), measured at G2.

| Stage | Billed frames | Wall, nominal | Wall, A1-realised |
|---|---|---|---|
| G0 (with planted controls), G0′, G0-S, B-clock | 0 | 8–10 h + 3 × 1–2 h CPU | same |
| Bars (41 + 24, +9) | ≈ 54M evaluator, unbilled | 1–2 h | 2–4 h |
| G1a (+ G1a′) | 48M (+48M) | 1.5 h (+1.5 h) | 4.5 h (+4.5 h) |
| G1b | 328M | 10 h | 32 h |
| G2: 8 units × 12 runs | ≈ 307M | 13 h | 36 h |
| G3: 8 units × 22 runs | ≈ 563M | 25 h | 70 h |
| B0: 48 runs | 192M | 6 h | 19 h |
| B1: 8 units × 32 runs | ≈ 819M | 35 h | 97 h |
| **Total** | **≈ 2.3G** (development 1.29G, sealed 1.01G; +72M second round) | **≈ 105 h** | **≈ 265 h** |

**Caps.** Development 1.4G, sealed 1.1G, total 2.5G.
- At each gate's start, its cost is projected from measured run costs. If it would cross the cap by more than 0.1G, the cut ladder applies first.
- A gate is never read on incomplete data: if it cannot complete, it is reported **INCOMPLETE** and the sealed pool stays closed.

**Cut ladder.** Applied in order at the start of G2 and again at the sealed freeze, only when the projected end date of B1 is later than day 21, and never because of results:
- C1: development units 8 → 6.
- C2: X_8 dropped from G3.
- C3: sealed units capped at 6 instead of 8.
- C4: B1 comparator arms (T, S, V2, P, U) go from 4 to 3 seeds.

B1 starts only if its projected end after the cuts is at most day 24; otherwise Jérémie is asked. A started sealed test is always finished.

**RAM.**
- Unsteered worker ≤ 1.0 GB and steered ≤ 1.2 GB (targets after disk logging; measured).
- G0 worker ≤ 0.8 GB (OMP_NUM_THREADS = 2); explorer workers use 1 BLAS thread.
- Three workers ≤ 3.6 GB; two while the exploratory job runs or when Tavern needs RAM.
- Supervisor guard at 14 GB free (the 12 GB rule). Supervisor stops are logged with the outputs.

**GPU:** 0 h; nothing in v84 uses torch. **Disk:** about 2 GB for the G0 store, about 30 GB for run outputs.

---

## 11. Schedule (day 0 = Friday 23 October)

| Day | Work |
|---|---|
| 0 (23 Oct) | Worktree `F:\dev\ragnarok_v84` (branch `arc4-v84`). Commit `ARC4_V84_PROTOCOL_EN.md` (this plan) and `ARC4_V84_PROTOCOL.md` (French), seed grep and contamination record. **Whole plan frozen.** |
| 1 (24) | G0 code and tests T1–T7, store, planted controls T8, 1-agent review T9. V2 build starts. |
| 2 (25) | Sources; κ tuning → `g0_kappa.json`; F curves → `g0_units.json`; other arms; `g0_verdict.json` (or day 3). Route decision. Unseal the v83 exploratory result after the verdict commit. |
| 3 (26) | V2 tests and smoke (9700); bars (9706); G1a (9710). |
| 4 (27) | G1a′ if needed; 2-agent adversarial review of V2, steering and the critic; launch G1b. |
| 5–6 (28–29) | G1b verdict committed; day-6 offline work (sources, live κ, T, S, G0′, G0-S). |
| 7–9 (30 Oct–1 Nov) | G2 WORTH, verdict. |
| 9–12 (1–4 Nov) | G3, GO or KC-dev. |
| 12–13 (4–5 Nov) | 3-agent review; sealed freeze. |
| 13–15 (5–7 Nov) | B0 → B-clock committed → B1 (nominal). |
| 16+ | Verdicts, short report in French, handoff. At A1-realised throughput with all cuts, a chain still alive ends around day 18–22. |

---

## 12. Odds (mine; gates are correlated, so about 1.5–2× the independent product)

| Event | P |
|---|---|
| Planted controls pass (first try or after one amendment) | ≈ 0.85 |
| G0 PASS | ≈ 0.2 |
| RECIPE-INSENSITIVE | ≈ 0.1 |
| G-build | 0.85 |
| G1b PROCEED (≥ 13/35) | ≈ 0.21 overall |
| "Climbs on most dev games" (≥ 18/35, depth-capped) | ≈ 0.08 |
| Route Σ, given G0 PASS and room: WORTH 0.45; GO 0.35; ≥ 6 sealed units 0.75; PARTIAL or better given measurable 0.35 | — |
| Route S: G0-S 0.3; WORTH 0.35; GO 0.3 | — |
| **Sealed SEARCH-CLIMB FASTER-PARTIAL or better** | **≈ 0.5%** (0.2–1.5%) |
| **Sealed SEARCH-CLIMB FASTER-2×** | **≈ 0.15%** |
| Another sealed gain label (EARLY-RUNGS-ONLY, BELOW-NO-CARRY-FIELD, GENERIC-STEERING, UNATTRIBUTED-GAIN) | ≈ 1% |
| PREDICTS-SOONER-SEALED (live route reaches sealed) | ≈ 0.4%; with Option M ≈ 7% overall |
| Clean G0 verdict by day 3 | ≈ 0.85 (day 2: ≈ 0.6) |

**Most likely ending:** KC-knowledge-speed (or RECIPE-INSENSITIVE) by day 3, then KC-room-84 at G1b or the death of route S at G0-S or WORTH. In hand at that point:
- a frozen 41-game mechanism negative for this recipe;
- a cost map for a probe-free climber at A1 strength;
- the full control harness.

---

## 13. What a positive would and would not establish

### 13.1 Route Σ: SEARCH-CLIMB FASTER (consequence prior)
On never-opened Atari games, a savestate explorer with billed rewinds and the score-rise and lives senses carries a zero-mean covariance over hand-designed, ego-centric consequence features, learned only from its own play on 41 earlier games. With it, the explorer climbs matched rungs (reached by V2 on both eligibility seeds) faster, including the late rungs and without being later in absolute time, than:
- the same explorer without it;
- placebo and input-statistics priors with an identical spectrum;
- the fastest of the no-carry explorer, tuned search defaults and the single best process.

The head start is zero by construction and certified.

### 13.2 Route S: SEARCH-CLIMB FASTER (search sense)
A cross-game learned search controller makes the savestate explorer climb matched rungs faster than the same controls. **It carries search configuration, not knowledge of game consequences.** Its closest cousins are learned node selection, algorithm configuration (DAC) and Intelligent and latent Go-Explore.

### 13.3 Mechanism readouts
- **PREDICTS-SOONER-SEALED:** from the same logged experience, the carried prior makes an offline predictor of rises and losses reach the novice's end-of-stream skill in fewer frames. This is offline mechanism evidence, not competence.
- **G0 PASS:** development evidence on burned logs. It certifies the recipe, not the sealed object.

### 13.4 Headline table (B1 × B-clock)

| | Clock PREDICTS-SOONER-SEALED | Clock not shown |
|---|---|---|
| B1 SEARCH-CLIMB FASTER-… | "Climbs faster with savestates and predicts consequences sooner on new games" | "Climbs faster with savestates; the mechanism is not shown on the sealed games" |
| B1 anything else | "Predicts outcomes sooner on new games; no faster climb. This is not learning to play faster" | Negative; the B1 label as written |

### 13.5 Not established by any outcome
- live play without savestates (the distillation wall stands);
- perception of score or lives (they are senses);
- "any game" (only games with room, likely 30–45% of Atari);
- NES, 16-bit or PC;
- accumulation (the dose caption is from development only and underpowered);
- state-of-the-art sample efficiency;
- human-like learning in general.

A head start is never reported as faster learning.

---

## 14. Disagreements between judges, resolved

| Issue | Resolution |
|---|---|
| Frame (P5 vs P1) | P5's clock is G0. Live machinery: P3 (cells, post-loss archiving, probe-free, λ over rungs, WORTH order, dose) + P1 (bar processes, held 0.75, bandit floor, T and S, instrument line, more seeds) |
| Carried object | Σ if G0 passes, otherwise the search sense gated by G0-S. One route, one object |
| Eligibility strength | H ≥ M3 (A1 strength), and only rungs reached by V2 on both eligibility seeds are measured. P1's H ≥ B′ + 2u′ is rejected: V already meets it on 20/35 |
| Ladder score | Cumulative sensed rises; net score reported; limitation stated |
| Placebo | Full permutation within each run decides; the shift is report-only; U is a decision arm; P_rot is reported |
| K\* leak | K\*_A from the learning stream |
| Clustering | Game-cluster bootstrap, equal game weight |
| Checkpoints | Quarter-octave, log-interpolated, held for 2, averaged over 5 levels |
| Exam as sealed instrument | Dropped (lineage-replay risk, me-lost states, a constant best slot, correlated states) |
| P2 macros, P4 GPU learner, X-rep | Not in v84 (v85 candidates); X-rep dropped |
| T and S | Mandatory on both routes; V2 added to the no-carry field |
| Over-regularised F | κ tuned per arm; sum-loss MAP |
| G1a on known failures | Exploratory, no kill |
| Me dependence | V2's search is me-free; the heads read me passively; dilution counts against X |

Factual corrections on record:
- Only pong and fishing_derby pass M1 at the root clone.
- Bar-mode tallies differ only by ties.
- A1 rows per run range from 3.6k to 99k.

---

## 15. Cousins and honest novelty (from memory; a literature check is due before any novelty claim)

- **Covariance prior from related tasks:**
  - Raina, Ng & Koller 2006 already used a **zero-mean** Gaussian prior with a covariance learned from related tasks.
  - Also: Evgeniou & Pontil 2004; Argyriou, Evgeniou & Pontil 2008; Bakker & Heskes 2003; Yu, Tresp & Schwaighofer 2005; Lawrence & Platt 2004; Baxter 2000; Thrun & Pratt 1998; Caruana 1997.
  - Theory: Tripuraneni, Jin & Jordan 2021; Du et al. 2021.
  - Learned bandit priors with non-zero means: Kveton et al. 2021; Basu et al. 2021; Cella, Lazaric & Pontil 2020; ALPaCA.
- **Measuring learning speed:** Taylor & Stone 2009; Dawid 1984; Blier & Ollivier 2018; Voita & Titov 2020; Whitney et al. 2020.
- **Controls:** Rice 1976; SATzilla; DAC (Biedenkapp et al. 2020; Adriaensen et al. 2022); Feurer et al. 2015.
- **Explorer and search sense:**
  - Go-Explore (Ecoffet et al. 2019, 2021), including the adaptive downscaling that V2 reuses.
  - Agent57; temporally-extended ε-greedy; learned node selection (He, Daumé & Eisner 2014; Khalil et al. 2016).
  - Intelligent Go-Explore (uses a foundation model, not allowed here); latent Go-Explore.
  - Meta-learned exploration (Alet et al. 2020; First-Explore; MAESN; E-RL²).
- **Cross-task transfer on Atari:**
  - Konidaris & Barto 2007; Rosman & Ramamoorthy 2012.
  - Actor-Mimic, Progressive Nets, MGDT, Scaled QL: mixed or negative transfer to unseen games.
  - Plasticity loss (Ash & Adams 2020; Nikishin et al. 2022; Abbas et al. 2023).
  - Human priors (Tsividis et al. 2017, 2021; Dubey et al. 2018).
- **Honest novelty.**
  - Not new: the statistics and the zero-mean prior.
  - The caveat on "carry the questions": the questions themselves (427 features, ego rotation, the me tracker) are hand-designed; only their covariance is learned.
  - New is the test:
    - an object learned only from the agent's own savestate play, from pixels and senses, with no pretrained weights;
    - a live prefix-identity certificate of zero head start;
    - a same-experience clock with spectrum-matched placebo and input-PCA controls, a planted positive control on the real feature geometry, and per-arm tuned priors;
    - a matched-rung late-climb test with an absolute-time guard on never-opened games, against the no-carry field.
  - **The contribution is an honest test, not a new algorithm.**

---

## 16. Decisions for Jérémie, and next steps

- **Option M (your yes is needed before it is used).**
  - When: G0 PASS, but the live route dies (KC-build, KC-room-84, KC-worth-live or KC-dev).
  - What runs:
    - bars 9740 and V2 eligibility 9741–9742 on the **9 second-round games** (about 72M frames; about 3 h nominal, about 8 h realised);
    - then the knowledge clock with the live Σ recipe (all 41 games) and live κ.
  - Unit rule A.10. Thresholds: ≥ 6 units on ≥ 4 games; R(X:C\*) ≥ 1.5 with lower bound > 1.1 and exact sign-flip p ≤ 0.05; lower bound of R_late > 1; ceiling guard.
  - Labels: PREDICTS-SOONER-SECOND-ROUND, NOT-SHOWN-SECOND-ROUND, CLOCK-NOT-MEASURABLE.
  - Chance of a positive given G0 PASS: about 0.4. It **uses up the second-round pool**, which is otherwise the top-up for future sealed tests.
- **For information only.** A live chain still alive at day 12 (p ≈ 2–4%) probably ends around day 18–22 at the throughput A1 achieved. The cut ladder is declared, and B1 starts only if its projected end is at most day 24; beyond that you are asked.
- **The exploratory v83 job** is left to finish, unread. Stopping it would be your call; it is not proposed.
- **No other decision is needed.**
- **Today:**
  1. Create the worktree.
  2. Write `ARC4_V84_PROTOCOL_EN.md` (this plan, verbatim) and `ARC4_V84_PROTOCOL.md` (French translation that defers to it), with the seed grep (`git ls-files` form, venvs excluded) and the contamination record.
  3. Commit them with `git commit --only` and the footer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  4. Amend the v83 seal note (unseal only after `g0_verdict.json`).

---

# Part A. First-gate specification (G0), ready to freeze

**A.0 Status.**
- This is development evidence on burned logs. It certifies a recipe and never yields a competence claim or a sealed claim.
- This English text is authoritative.

**A.1 Question.**
- Setting: on a held-out development game, the logistic GOOD/BAD learner learns from one A1 stream, in order, and is scored on the other seed's stream.
- Question: does a zero-mean prior N(0, κΣ_X) reach fixed fractions of F's end-of-stream AUC in fewer billed frames? Σ_X encodes which ego-centric, action-rotated directions mattered on 40 other games.
- Conditions:
  - X must beat the per-unit best of F, P and U;
  - over the whole curve, and late (counted from each arm's own N(0.5));
  - with room shown by the same-game ceiling K\*_A;
  - after a planted positive control on the real feature geometry has shown that the recipe can detect shared structure.

**A.2 Data and integrity.**
- **Inputs:** `F:\dev\ragnarok_v83\craft_v6_out\v83\room\room_{game}_{seed}.npz` and `.json`, for the 41 games (DEV + CALIBRATION, in `pools_v83.py` order) × seeds 9131, 9132.
  - Each file is verified against `a1_manifest.json` SHA-256 before use. A mismatch is an Interruption.
- **Never opened:** any path containing `craft_v6_out\v83\explore`, or any captured stdout of `explore_promise_v83.py`. The loader asserts this and test T5 greps the G0 code.
- **Not inputs:** bars, `room_verdict.json`, true score, RAM, game names inside learner functions (scripts hold names for bookkeeping only; learner functions receive arrays).
- **Outputs:** `F:\dev\ragnarok_v84\craft_v6_out\v84\g0\` (`store/`, `planted.json`, `sources.npz`, `g0_kappa.json`, `g0_units.json`, `curves.npz`, `g0_verdict.json`), each with code SHA, clean-tree flag, input hashes, `claim_header` and `senses`.

**A.3 Rows, features, labels, clock.**
- **Kept rows (per stream):**
  - `row_lost == False`;
  - and `row_it < len(iter_billed)`. Rows of the rollout cut by the billed cap have no clock and are dropped; the count is reported per run.
  - If more than 24,000 remain, draw 24,000 uniformly without replacement with `default_rng([9701, game_index, seed])`, keeping the original order.
- **Features:** v83 `promise.feats(view, screen, False, (mx, my), lives, slot, (210, 160))`, giving 427 values computed in float64 and stored as float16.
- **Labels:** v83 `logs.labels` with steps = 100.
  - GOOD: lag = rise_lag, H = 24.
  - BAD: lag = drop_lag, H = 40, stops_on_event = True.
  - hit = lag < H; −1 (censored) is ignored by that head.
- **Clock:** c_i = `iter_billed[row_it[i]] − entry_frames`. It must be non-decreasing in stream order, otherwise Interruption.
- **Checkpoint index:** n counts kept rows of the learning stream in order. The fit at n uses the uncensored rows among them, and c(n) is the clock of the n-th row.
- **Blocks:** block_i = `row_it[i] // 50`.

**A.4 Splits.**
- For each game g, two directions d ∈ {(A = 9131, B = 9132), (A = 9132, B = 9131)}.
- **Learning stream:** all kept rows of A, in order.
- **Test set (per head):** kept rows of B in odd blocks, uncensored for that head.
  - Rows whose exact tuple (view bytes, screen bytes, slot, lives) occurs in A are removed (blake2b hash), and the removed fraction is reported.
  - If more than 5,000 remain, draw 5,000 with `default_rng([9702, game_index, seed_B, head_index])` (GOOD = 0, BAD = 1).
  - The test set is identical for every arm.
- **Sources for g:** all kept rows of both streams of the other 40 games.
- κ is tuned on the 6 calibration games only; units are scored on the 35 development games only.

**A.5 Shared, label-free geometry (per held-out game g; identical for every arm).**
- S₋g: mean and SD over all kept rows of the 80 source streams; x̃ = (x − m)/max(sd, 0.05).
- C₋g: equal-game-weighted covariance of x̃ over the 40 source games.
- Q₋g: the eigenvectors of C₋g with eigenvalue > 1e-5·λ_max. This removes the exact collinearities of mean pooling, one-hot sums and flag identities, including float16 rounding.
  - d₋g is their number. The learner uses x̂ = Q₋gᵀ x̃.
  - Reported: d₋g, and the share of g's x̃ variance outside Q₋g.

**A.6 Learner (`ragnarok/agent_v84/consequence.py`, shared with the live heads).**
- **Fit at checkpoint n:** minimise over (v, b)

  Σ_{i ≤ n, uncensored} [softplus(z_i) − y_i·z_i] + (1/(2κ_a))·vᵀ Σ_a⁻¹ v, with z_i = b + x̂_iᵀv.

  - The loss is a sum, so the prior's weight fades as 1/n. The intercept is free.
  - Implemented as x′ = Σ_a^{1/2} x̂ with an isotropic penalty.
  - L-BFGS-B, at most 200 iterations, float64 accumulation, warm-started from the previous checkpoint (from 0 at the first fit).
- **Both-classes rule (every arm except X-mean):** while the uncensored rows seen so far lack either class, the head is intercept-only. Test scores are then constant and the AUC is exactly 0.5 (the certificate).
- **Checkpoints:** n_k = ⌈128·2^{k/4}⌉ for k = 0, 1, … while n_k < n_A, plus n_A itself.
- **Score:** test AUC (Mann–Whitney, ties counted half) at every checkpoint.

**A.7 Source heads (once per source game h and head type, in h's own geometry).**
- **Standardisation:** S_h, pooled over both streams of h. Features with raw SD_h < 0.05 are excluded (weight fixed at 0).
- **Fit:** logistic with one intercept per stream and ridge (ρ/2)‖w‖² on the mean log-loss.
- **Choice of ρ:** from {1e-3, 1e-2, 1e-1, 1}, by 2-fold cross-stream validation:
  - fit on one stream;
  - on the other stream, refit only its intercept, then compute the log-loss;
  - average both directions; ties go to the larger ρ.
  - If either stream has fewer than 10 rows of a class, ρ = 0.1.
- **Final fit:** on both streams with the chosen ρ, at most 300 L-BFGS iterations.
- A source head needs at least 30 positives and 30 negatives over both streams; otherwise it is left out and listed.
- **Mapping to g:** w′ = w ⊙ (sd₋g/sd_h), an exact reparameterisation of the same function of raw features (declared). Then v_h = Q₋gᵀ w′ and û_h = v_h/‖v_h‖.
- **Permuted heads:** the same recipe after permuting labels uniformly among the uncensored rows of each stream, `default_rng([9703, game_index, seed, head_index])`.
- **Shifted heads (P_shift, report only):** v83 circular shift of at least 10 rollouts, seed 9707.

**A.8 Arms.**
- Each head type has its own Σ. Every Σ is d₋g × d₋g and normalised to trace d₋g; τ² = 0.1/d₋g.

| Arm | Σ (or prior) | Role |
|---|---|---|
| F | I | Baseline learner (cross-game scaling, tuned κ) |
| X | norm[(1/G)·Σ_h û_h û_hᵀ + τ²I] over the included source heads | Carried hypothesis space |
| P | Eigenvectors of the same construction from permuted heads, with X's eigenvalues (rank-matched) | Placebo (decision) |
| U | Q₋g axes ordered by C₋g eigenvalue, with X's eigenvalues rank-matched (largest to largest) | Input statistics only (decision) |
| K\*_A | norm[û_A û_Aᵀ + τ²I]; û_A from a head fitted in x̂ on the whole learning stream A of g (ρ by 2-fold over A's even and odd blocks) | Same-game ceiling (decision for (ii)) |
| X-mean | N(v̄, κI), v̄ = mean of the mapped, projected, unnormalised v_h; slopes start at v̄; exempt from the both-classes rule | Report only: what a head start looks like |
| X_8 | Σ_X built from 8 source games, 3 draws `default_rng([9704, game_index])`, mean λ over draws, κ = κ_X | Report only: dose |
| P_shift | Spectrum-matched shifted-label Σ | Report only |
| P_rot | R Σ_X Rᵀ, R Haar-random from `default_rng([9709, game_index, head_index])` | Report only: random directions, same spectrum |
| K\*_Beven | As K\*_A, but from B's even blocks, deduplicated against the test set (leak-prone) | Report only |

**A.9 κ tuning (before any development curve).**
- **Grid:** κ ∈ {4⁻⁵, 4⁻⁴, 4⁻³, 4⁻², 4⁻¹, 1}, tuned separately for F, X, P, U, K\*_A, X-mean, P_shift and P_rot. X_8 uses κ_X; K\*_Beven uses κ_K\*A.
- **Tuning curves:** the 6 calibration games × heads × 2 directions, wherever the learning stream and the test set each have at least 30 rows of each class (freeway and pong have no BAD).
  - For calibration game c, Σ is built from the other 40 games and K\*_A from c's own learning stream.
  - Declared: this uses development games as sources, but never their curves or test rows.
- **Criterion:** mean over the tuning curves of the mean AUC over checkpoints. Highest wins; exact ties go to the smaller κ.
- **Edge rule:** if the chosen κ of a decision arm (F, X, P, U, K\*_A) is 4⁻⁵ or 1, extend the grid once by one factor of 4 on that side for every arm, and re-select.
- **Commit:** `g0_kappa.json` with its hash, before any development curve.

**A.10 Units (decided on F alone).**
- A unit is a (development game, head). It is eligible only if, in **both** directions:
  - the learning stream has at least 30 positive and 30 negative uncensored rows;
  - the test set has at least 30 of each class;
  - AUC_F,end (mean of F's AUC at the last 3 checkpoints) is at least 0.60 for GOOD or 0.65 for BAD;
  - AUC_F,end − AUC_slot ≥ 0.03, where AUC_slot is the test AUC of the per-slot positive rate fitted on the whole learning stream (unseen slots get the stream rate).
- **Commit:** `g0_units.json`, before any non-F development curve.

**A.11 Metric.**
- **Levels:** L_q = 0.5 + q·(AUC_F,end − 0.5), for q ∈ Q = {0.25, 0.375, 0.5, 0.625, 0.75}, per (unit, direction).
- **Crossing time N_a(q):**
  - the first k with AUC_a(k) ≥ L_q and AUC_a(k+1) ≥ L_q (held for two checkpoints); the last checkpoint cannot start a crossing;
  - if k ≥ 1, log c is interpolated linearly in AUC between checkpoints k−1 and k;
  - if k = 0, N = c(n_0);
  - floor: 5,000 frames.
- **Never crossing:** carried arms (X, K\*_A, K\*_Beven, X-mean, X_8) get 2·c_end; controls (F, P, U, P_shift, P_rot) get c_end.
- **Primary:** λ_a = mean over q ∈ Q of log N_a(q).
- **Late:** for q ∈ {0.625, 0.75}, D_a(q) = N_a(q) − N_a(0.5).
  - Carried arm: a missing N(q) is set to 2·c_end; a missing N(0.5) gives D = 2·c_end.
  - Control: missing N(q) but with N(0.5) gives D = c_end − N(0.5); a missing N(0.5) gives D = 5,000.
  - λ_late = mean of log max(5,000, D).
- **Aggregation:**
  - per unit, average each arm's λ over the 2 directions;
  - C\* = per-unit argmin over {F, P, U} of λ (separately for λ_late);
  - ℓ_unit = λ_C\* − λ_X; ℓ_g = mean over the game's eligible units; R = exp(mean over games of ℓ_g), each game weighted equally;
  - a pairwise R(A:B) is computed the same way with B fixed.
- **Reported:** the share of (unit, direction) pairs where every zero-mean decision arm has the same N(0.25).

**A.12 Statistics.**
- Game-cluster percentile bootstrap of ℓ_g: 2,000 resamples, `default_rng(9705)`, 2.5% and 97.5%. Two heads of one game never count as independent.
- One-sided sign-flip p over games: exact if at most 20 games, otherwise 10⁵ Monte-Carlo draws (seed 9705).
- Also reported: leave-one-game-out minimum, per-level profile R(q), Hodges–Lehmann, R(P:F), R(U:F), R(P_rot:F), R(X-mean:F), R(X_8:F).

**A.13 Pass, kill, decision.** Evaluated in the order (vi), (i), (ii), then (iii)–(v).

| # | Condition |
|---|---|
| (i) | At least 12 eligible units spanning at least 8 development games |
| (ii) | R(K\*_A:F) ≥ 2.0 with lower bound ≥ 1.25 |
| (iii) | R(X:C\*) ≥ 1.5, lower bound > 1.1, sign-flip p ≤ 0.05 |
| (iv) | R_late(X:C\*_late) lower bound > 1 and sign-flip p ≤ 0.05 |
| (v) | Ceiling guard: mean over units and directions of (AUC_X,end − AUC_F,end) ≥ −0.01 |
| (vi) | Integrity: tests T1–T9 passed (T8 included); hashes match; prefix certificate on real streams; eigenvalues of P, U and P_rot equal X's (relative error ≤ 1e-9); traces = d₋g; sealed-path guard; clean tree |

- **PASS** (all of i–vi): route Σ; the recipe is certified for live arms.
- **KC-knowledge-room** (i fails): "the A1 streams do not hold enough learnable consequence structure to measure learning speed."
- **KC-knowledge-worth** (ii fails): "with this recipe, even the game's own learned direction does not make this learner 2× faster."
- **KC-knowledge-speed** (iii, iv or v fails): "with this Σ recipe, carried consequence structure does not make the learner at least 1.5× faster than the best tuned control, on the whole curve and late." Captions:
  - GENERIC-SHRINKAGE: lower bound of R(X:F) > 1, and lower bound of R(P:F) or R(U:F) > 1;
  - FIRST-RULES-ONLY: (iii) passes and (iv) fails;
  - SUB-THRESHOLD: lower bound of R(X:C\*) > 1, but (iii) fails;
  - WEAK-PLACEBO: upper bound of R(P:F) < 1.
- **RECIPE-INSENSITIVE** (T8 fails twice): "the recipe cannot detect planted shared structure on the real feature geometry." No real-label curve is computed. Route Σ is closed; this is not a mechanism negative.
- **Interruption** (vi fails): no verdict. First-pass outputs are kept and committed; the rerun is labelled.
- **Any KC or RECIPE-INSENSITIVE:** route Σ is closed for v84 live arms, route S becomes the candidate, and the negative is committed and kept.

**A.14 Written now (no effect on the verdict).**
- If X helps, R(q) falls with q (about 3× at q = 0.25, about 1.5× at q = 0.75), and BAD carries more than GOOD.
- X-mean shows AUC above 0.5 at the first checkpoint and a smaller late ratio than X.
- R(X_8) < R(X).
- R(P_rot:F) ≤ 1.

**A.15 Tests before scoring.** No real-label computation (source heads included) runs before T1–T9 pass.
- **T1.** Features and labels byte-identical to v83 on 1,000 random rows (draw with seed 9700; features compared in float64 before the float16 cast).
- **T2.** Clock non-decreasing; cut-rollout rows dropped and counted.
- **T3.** Prefix certificate on synthetic streams (on real streams it is checked as part of (vi)).
- **T4.** Spectrum and trace identities.
- **T5.** Sealed-path guard (loader assertion, plus a grep of the G0 code for `explore`).
- **T6.** Deduplication applied and counted.
- **T7.** Gaussian unit test for code correctness: 6 games, 2k-row streams, 20 dimensions. A shared subspace must give R(X:F) > 1.5; independent heads must give R(X:F) ≤ 1.25. A failure is a bug: it is fixed and recorded.
- **T8. Planted controls on the real geometry.**
  - **Games and rows:** 12 games drawn from DEV + CALIBRATION with `default_rng(9708)`; their real kept rows, with streams capped at 6,000 by the A.3 rule; the A.5 geometry computed within the 12-game pool, leaving one game out.
  - **Labels:** synthetic, one planted head per game, y ~ Bernoulli(σ(b + x̂ᵀw)).
    - w is scaled so that the logit SD over that game's rows is 1.5; b gives a base rate of 0.15.
    - Labels are drawn with `default_rng([9708, game_index, seed])`.
    - The GOOD keep mask is used (label-free: t + 24 ≤ 100 and avail ≥ 24).
  - **Scenarios:**
    - *Positive:* w_g = V·a_g, with V a Haar-random 5-dimensional orthonormal basis and a_g ~ N(0, I₅).
    - *Null:* each w_g Haar-random, independent across games.
  - **Pipeline:** the same code as G0. κ is tuned per arm by A.9 on 4 games (drawn with seed 9708); units, metric and statistics are computed on the other 8.
  - **Pass:**
    - positive: R(K\*_A:F) ≥ 2.0 and R(X:C\*) ≥ 2.0;
    - null: R(X:F) ≤ 1.25 and R(X:C\*) ≤ 1.1.
  - **Failure:**
    - A null failure is treated as leakage or a bug: fix, record, rerun.
    - A positive failure allows one dated recipe amendment, then T8 is rerun. Allowed knobs only: the ρ grid; τ²; truncating Σ_X to its top r ∈ {10, 20, 40} eigen-directions; the null-space threshold.
    - A second positive failure gives RECIPE-INSENSITIVE.
  - The generator settings (dimension 5, logit SD 1.5, base rate 0.15) are frozen now.
- **T9.** One-agent code review of the G0 scripts before step 3 of A.16.

**A.16 Run order and cost.**
1. Store: 1,705,576 rows, about 1.46 GB float16 memmap, 30–60 min on 2 workers.
2. T1–T7; T8 (about 1–1.5 h); T9.
3. Sources: 41 games × {GOOD, BAD} × {real, permuted} with cross-validation (about 1,500 fits), shifted heads, 41 geometries; about 1.5–2 h.
4. κ tuning (about 960 curves, about 1.5 h); commit `g0_kappa.json`.
5. F curves on development games (at most 140); commit `g0_units.json`.
6. All other arms (about 600–900 curves, about 2–3 h).
7. Verdict; commit `g0_verdict.json`, including the hashes of the exploratory outputs if the job has finished.

Total: about 8–10 h wall on 2 workers (at most 0.8 GB each, OMP_NUM_THREADS = 2), 0 emulator frames, 0 GPU, about 2 GB disk.

**A.17 Amendments.** Only the T8 recipe amendment and Interruption reruns are allowed. Each is dated, records its trigger and is committed before the data it affects.

---

# Part B. Code to write, in order

1. `ragnarok/agent_v84/__init__.py`, `consequence.py`: MAP logistic regression with a Σ prior, reduced basis, both-classes rule, warm start.
2. `scripts/g0_store_v84.py`: manifest check, sealed-path guard, kept rows, cut-tail drop, features, labels, clock, deduplication hashes, memmap.
3. `tests/test_g0_v84.py`: T1–T7, plus crossing-time, censoring and aggregation unit tests.
4. `scripts/g0_planted_v84.py`: T8. Then the review (T9).
5. `scripts/g0_sources_v84.py`: S₋g, C₋g, Q₋g; source heads (cross-validated ρ, floored-feature exclusion, mapping); permuted and shifted heads; every Σ.
6. `scripts/g0_tune_v84.py` → `g0_kappa.json`.
7. `scripts/g0_clock_v84.py`: F-only mode → `g0_units.json`; then all arms → `curves.npz`.
8. `scripts/g0_verdict_v84.py`: statistics, labels, captions, provenance → `g0_verdict.json`.
9. `ragnarok/evaluation/senses_v84.py` (Trace with cumulative sensed rises and lineage depth), `milestones_v84.py` (B′, rungs, T(M), replay confirmation), `pools_v84.py`; reuse `climb_stats_v83`.
10. `scripts/bars_v84.py` (four modes).
11. `ragnarok/agent_v84/rowlog.py`: disk-chunked RowLog with uint16 t/avail, L_t, and the L_t labels rule.
12. `ragnarok/agent_v84/{framecode, archive, processes, bandit, explorer}.py` (V2), reusing `agent_v83/{adapter, entry, views}`.
13. `tests/test_isolation_v84.py` (full v83 isolation test, ported), `tests/test_v2_v84.py` (ledger, scripted trajectory, label causality, determinism).
14. `scripts/v2_job_v84.py` (G1a, G1b and B0 runs; registered with the supervisor; deterministic restart), `scripts/room_verdict_v84.py`.
15. `ragnarok/agent_v84/steer.py` (route Σ), `tests/test_prefix_identity_v84.py`.
16. `ragnarok/agent_v84/critic.py` (route-S features and critic), `scripts/g0s_v84.py`.
17. `scripts/sources_live_v84.py`: live Σ (X, P, U, X_8, K\*), live κ retune, T schedule, S choice, G0′.
18. `scripts/arm_job_v84.py`, `scripts/worth_v84.py` (G2), `scripts/rehearsal_v84.py` (G3).
19. `scripts/power_v84.py`: power simulation and censoring forecast at the sealed freeze.
20. `scripts/sealed_v84.py` (B0, B1, verdict ladder), `scripts/clock_sealed_v84.py` (B-clock; Option M mode).

---

# Part C. Outcome labels

| Gate | Label | Condition | Sentence | Consequence |
|---|---|---|---|---|
| G0 | PASS | A.13 (i)–(vi) | Carried consequence covariance makes the learner ≥ 1.5× faster than the best tuned control, overall and late (burned logs) | Route Σ |
| G0 | KC-knowledge-room | (i) fails | The A1 streams do not hold enough learnable consequence structure to measure learning speed | Route S candidate |
| G0 | KC-knowledge-worth | (ii) fails | With this recipe, even the game's own direction does not make the learner 2× faster | Route S candidate |
| G0 | KC-knowledge-speed (+ GENERIC-SHRINKAGE, FIRST-RULES-ONLY, SUB-THRESHOLD, WEAK-PLACEBO) | (iii), (iv) or (v) fails | With this Σ recipe, carried structure does not make the learner ≥ 1.5× faster than the best tuned control, overall and late | Route S candidate |
| G0 | RECIPE-INSENSITIVE | T8 fails twice | The recipe cannot detect planted structure on the real geometry | Route S candidate; not a mechanism negative |
| G0 | Interruption | (vi) fails | No verdict | Fix, label the rerun, keep the first pass |
| G-build | KC-build | Tests 1–6 fail after 2 fix rounds | V2 could not be built to pass its own tests | No live arms |
| G1a | (caption only) | — | V2 at constants c reaches M3 on k/10 games of the A1-failure panel | G1b always runs |
| G1b | PROCEED (+ "climbs on most dev games") | ≥ 13/35 and se_proj ≤ 0.23 (≥ 18/35 depth-capped for the label) | V2 has A1-strength room on n/35 development games | Live route possible |
| G1b | KC-room-84 | Otherwise | V2 does not reach A1-strength room on ≥ 13/35 development games | Cost map kept; Option M |
| G0′ | AGREE / DISAGREE | Report only | Recipe on V2 streams agrees or disagrees with G0 | None |
| G0-S | PASS-S / KC-search-sense | A.13 adapted (§6.2) | The cross-game search sense does (not) speed the offline learning of expansion yield beyond tuned controls | Route S live / v84 ends at feasibility |
| G2 | WORTH / KC-worth-live (+ KC-harm flag) | §8 | Same-game prior steering does (not) speed the climb by 1.75× | G3 / sealed closed |
| G3 | GO / KC-dev | §8 | Carried object passes (fails) the development rehearsal | Sealed freeze / sealed closed |
| B0 | MEASURABLE / NOT MEASURABLE | ≥ 6 units (after the second round) | — | B1 / no sealed claim |
| B-clock | PREDICTS-SOONER-SEALED | §8 | From the same experience, the carried prior predicts rises and losses sooner on new games (offline mechanism evidence, not competence) | Reported with B1 (§13.4) |
| B-clock | NOT-SHOWN-SEALED / CLOCK-NOT-MEASURABLE | Otherwise / < 8 units or < 6 games | — | Kept |
| B1 | 1. NOT MEASURABLE | < 6 units | — | — |
| B1 | 2. INCONCLUSIVE | Instrument 99% interval excludes 1, or ≥ 40% of X or F runs censored at M_top, or prefix violated, or unresolved replay mismatch | — | — |
| B1 | 3. NEGATIVE TRANSFER | Upper bound of R(X:F) < 1 or upper bound of R(X:V2) < 1 | The carried object slows the climb | — |
| B1 | 4. SEARCH-CLIMB FASTER-2× [object] | R(X:F) ≥ 2.0, lower bound ≥ 1.3, p ≤ 0.025; R(X:N) ≥ 1.4, lower bound > 1; lower bound of R(X:Π) > 1; R_late(X:F) ≥ 1.5, lower bound > 1; lower bound of R_late(X:N) > 1; r_g(X:F) > 1 on ≥ 2/3 of units; leave-one-game-out minimum of R(X:F) ≥ 1.5; lower bound of R_abs(X:F) ≥ 0.9 | §13.1 or §13.2 at 2× | Headline |
| B1 | 5. SEARCH-CLIMB FASTER-PARTIAL [object] | Lower bounds of R(X:F), R(X:N), R(X:Π) and R_late(X:F) all > 1; p ≤ 0.025; leave-one-game-out minimum of R(X:F) > 1; point R_abs(X:F) ≥ 1 | §13.1 or §13.2 | Headline |
| B1 | 6. NOT-SIGNIFICANT | All PARTIAL conditions except p ≤ 0.025 | Intervals favour X, but the permutation test does not | — |
| B1 | 7. EARLY-RUNGS-ONLY | All PARTIAL conditions except the late clause | Faster on early rungs only | — |
| B1 | 8. BELOW-NO-CARRY-FIELD | Lower bounds of R(X:F) and R(X:Π) > 1; lower bound of R(X:N) ≤ 1 | The prior helps the steered learner, but not beyond the best no-carry search | — |
| B1 | 9. GENERIC-STEERING | Lower bounds of R(X:F) and R(Π:F) > 1; lower bound of R(X:Π) ≤ 1 | Any structured prior helps; content not shown | — |
| B1 | 10. UNATTRIBUTED-GAIN | Lower bound of R(X:F) > 1, none of the above (failed clauses listed) | X faster than F; attribution or a guard fails | — |
| B1 | 11. HEAD-START ONLY | HS ≥ 1.5 with lower bound > 1 | Earlier first rung, no faster climb | Never called faster learning |
| B1 | 12. NULL | Otherwise | — | — |
| B1 | Captions | HEAD-START (with any label), LATE-START (upper bound of HS < 1), KC-harm, φ, dose (development only), without near-siblings, net score | — | — |
| Option M | PREDICTS-SOONER-SECOND-ROUND / NOT-SHOWN-SECOND-ROUND / CLOCK-NOT-MEASURABLE | §16 | — | Needs Jérémie's yes |

---

# Part D. Résumé pour Jérémie

- On fige aujourd'hui tout le plan v84, pas seulement le premier test. Les seules modifications permises plus tard sont listées et datées.
- Premier test (G0), sans émulateur, sur les 82 journaux A1 : un a priori appris sur 40 autres jeux aide-t-il l'agent à prédire plus vite points et pertes de vie sur un jeu nouveau ? Il faut battre trois témoins bien réglés.
- Avant toute vraie étiquette, un contrôle « planté » vérifie que la méthode sait détecter une structure partagée. S'il échoue deux fois, on le dit, sans le présenter comme un résultat négatif sur le mécanisme.
- Verdict de G0 au jour 2 ou 3. Le résultat exploratoire v83 reste scellé jusqu'à ce verdict.
- En parallèle, on construit V2, un explorateur sans sondes, qui continue après une vie perdue. Il doit trouver de la marge sur au moins 13 des 35 jeux de développement.
- Si les deux tiennent, un seul objet transporté passe en test réel, contre l'explorateur seul, les réglages par défaut, le meilleur processus fixe et deux placebos. Les jeux scellés ne sont ouverts qu'une fois.
- Les mots sont honnêtes : on mesure une recherche avec retours en arrière facturés, pas du jeu en direct. Une avance au départ n'est jamais appelée « apprendre plus vite ».
- Mes chances : environ 0,5 % pour un positif scellé. Le plus probable est un résultat négatif propre en 3 à 6 jours, avec la carte des coûts et tous les témoins prêts.
- Ta décision n'est nécessaire que pour l'option M (utiliser les 9 jeux de seconde ronde si seul G0 réussit). Pour information : une chaîne encore vivante au jour 12 finirait vers les jours 18 à 22.