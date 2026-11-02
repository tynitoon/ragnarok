# v85 plan (final): measure online whether knowledge has room in V2's climb, license a carried object only through information it could hold, and test it once on sealed games

*Final editor's note. I read and grepped files. I used PowerShell and awk only for arithmetic on the v84 JSON: rung gaps, entry frames, wall times and supervisor kills. I ran no Python, emulator or GPU job and edited no file. This plan is to be committed as `docs/ARC4_V85_PROTOCOL_EN.md`, which is authoritative, with a French twin `docs/ARC4_V85_PROTOCOL.md` that defers to it. Both go in a new worktree `F:\dev\ragnarok_v85` on branch `arc4-v85`, created from the v84 tree.*

---

## 0. Summary

**Question.** v85's first decisive gate, G1, asks a cheap question with a verified instrument. The question: does same-game oracle information make V2 climb about 2× faster from a matched rung? The test is **online**: played in the emulator with billed rewinds, unlike v84's offline clock. It is not live play.

**No head start in state, archive or entry.** Every arm of a game starts from the same savestate **s1**. s1 is a state at rung M1 on an earlier V2 run's lineage that reached the top rung. Every arm loads the same cached entry and starts with an empty archive.

**Three channels, three evaluator instruments, one plumbing control:**
- **O_4, "which proposal to bill."** Each expansion, V2's own 4 proposals (cell, process, first slot) are ranked by one unbilled look-ahead each. The winner is then executed on a **fresh random stream** that no look-ahead used. The instrument therefore knows which proposal is productive, not which roll of the dice wins. The old "dice" version (winner replayed on its own lucky stream) survives only as a caption row.
- **K\*map, "where to restart."** Half of the cell draws go to the cell matching the furthest screen of the start's own future successful path.
- **K\*mac, "how to move."** Motor patterns from the **other** start's successful path of the same game, held 4 steps. This is the representation family M can carry.
- **D.** Replays the demonstration. It is a plumbing control only.
- The representation channel is not tested, and every negative says so.

**Before any threshold is fixed:**
- **S0** (0 billed frames): rebuilds the starts and caches the entries. It fixes **counted rungs** from reference data and computes a physics floor.
- **I-stage** (V2s and D only): measures the noise of the shared-start regime. It cross-fits the top rung and plants exact time dilations (×1, ×1.25, ×1.5, ×2, late-only, first-rung jump) in that regime.
- **Calibration rule** (written today): turns those null data into n, τ_pt and τ_lb.
- **NO-ROOM** needs evidence of absence: a 95% upper bound below 2, which must also hold with V2s misses counted at twice the cap.

**Routing by information match.** A channel licenses a carried family only if the family could hold that information:
- ROOM on O_4 licenses **family W**, a cross-game slope prior for an expansion critic (v84 route S, run online for the first time).
- ROOM on K\*mac licenses **family M**, a cross-game motor vocabulary.
- ROOM on K\*map alone licenses neither, since W's critic sees no screen content. v85 then ends with ROOM-UNCARRIED.

All family code and every carried object, control and placebo is written, tested and hashed **before G1 runs**.

**After G1:**
- **G2a, object room.** The family's same-game object against its own learner from scratch.
- **G2b, development rehearsal.** The carried object against that learner, tuned defaults, placebos and (family M) a frozen-vocabulary arm that separates a head start from faster adaptation.
- **One sealed test** on games not opened before v85's eligibility runs.

**Cost.**
- About **0.55G emulated frames up to the G1 verdict**: about 0.32G billed and 0.23G unbilled oracle look-ahead.
- About 1.6G on the development path and 0.75G sealed at most.
- 0 GPU-hours, at most 0.7 GB per worker. First verdict about **day 5–6**.

**Odds (mine; overall unless marked).**

| Event | P |
|---|---|
| G1 ROOM / INCONCLUSIVE / NO-ROOM / does not run | 0.35 / 0.23 / 0.12 / 0.30 |
| G1 ROOM, given that it runs | ≈ 0.5 |
| Sealed SEARCH-CLIMB FASTER-PARTIAL or better | ≈ 1% |
| Sealed SEARCH-CLIMB FASTER-2× | ≈ 0.2–0.3% |

---

## 1. Audit resolutions

### 1.1 Blocking and major issues

| # | Issue (lens, severity) | Fix | Where |
|---|---|---|---|
| B1 | O_4 replayed the winner's own random stream. It was dice foresight, its ROOM was near-certain (G_4 ≈ 3.6–4), it routed G2 to W, and it broke "every frame billed" (can-it-fail, feasibility, honesty: blocking) | The selection row is now **O_4 with independent execution**. Look-ahead j uses stream X_{e,j}; the winner's (cell, process, first slot) runs on X_e, which no look-ahead used. O_1 through the sandbox is byte-identical to V2s (test 6). The dice version is a caption row only (1 run per unit): never in a label, never routes. Every O sentence names "4 unbilled look-ahead rollouts per expansion"; R is also reported on total emulated frames; the claim header names oracle frames. Odds revised | §4.3, A.7, A.13, §14 |
| B2 | Reusing the I-stage comparator in later gates triggers v84's mixed-code Interruption (`room_verdict_v84.py` l.182–186 hashes whole trees) (feasibility: blocking) | **Per-arm import-closure hash** in every run JSON. All G1-path code is committed before the I-stage launches. Family modules are injected objects and never enter V2s's closure. New arms are new modules, so `v85_job.py` is never edited after the I-stage launch. **Reuse certificate:** if V2s's closure changed, 2 V2s runs are re-executed and must match records and exp_meta byte for byte; otherwise all V2s runs are redone (≈ 105M frames, budgeted). A dirty tree, or mixed closures within a gate, is an Interruption | §13.4, A.14 |
| M1 | Channel-to-family mapping unsound; no same-game test of the object before G2 (can-it-fail: major) | **Licensing table:** O_4 licenses W, K\*mac licenses M, K\*map licenses nothing in v85. **G2a object-room gate** (K against F and V2s) before any carried arm | §7.3, §8.1 |
| M2 | ROOM, NO-ROOM and "head start zero" sentences overclaim; labels overlap (can-it-fail and honesty: major) | Sentences state point and bound. The calibration enforces **P(ROOM \| true 1.5×) ≤ 0.25**. NO-ROOM reads "2× excluded at 95%; room up to UB95 not excluded". ROOM-K takes precedence, with a BELOW-2X caption. Every sealed label is defined; "beats" means LB90 > 1. Names: SEARCH-CLIMB FASTER-2× / -PARTIAL. "No head start in state, archive or entry" | A.11, Part C |
| M3 | Family M cannot tell a carried vocabulary (head start) from faster learning (honesty and can-it-fail: major) | Family M gets an **X_frozen** arm (library fixed, no online mining) in G2b and B1, plus an interaction contrast. HEAD-START-VOCABULARY caption when LB90 R(X_on:X_frozen) ≤ 1. "Accelerates in-run learning" wording requires ADAPTS and an interaction lower bound > 0. Family W's prior is zero-mean and acts only through in-run refits (stated) | §4.4, §8.2, §9, §15 |
| M4 | Calibration optimistic about censoring; degenerate first rung; winner's curse; assumed variance (can-it-fail and honesty: major) | (a) **Counted rungs** fixed at S0 from reference data: above S(s1), median reference gap ≥ 50k frames, rungs reached by the same record merged. By hand, all 16 units keep 2–5 rungs. (b) Evaluator stop at the reference top rung. (c) The **top rung is cross-fitted** from the other start's V2s runs. (d) A censored base run stays censored where that is pessimistic. (e) NO-ROOM must also hold with V2s misses at 2× cap. (f) The comparator is resampled in every simulation; residuals are centred per (unit, start) × √(3/2). (g) A **post-G1 check** with realised variance and heterogeneity can only downgrade a label | A.4, A.10, A.12 |
| M5 | Forking paths: family code, κ and the family's oracle settled after G1 (can-it-fail: major) | Before G1, as part of F1: both families' code, κ for every W arm, and all development carried objects, controls and placebos are written, tested and hashed. The family's anchor is named now: the same-game K in G2a. The ceiling-to-label rule uses G2a's K:F lower bound. GENERIC is an equivalence test that removes a row from routing | §4.4, §7.2, §8 |
| M6 | The sandbox snapshot was incomplete, so the stated identities could fail by construction (can-it-fail: major) | Full state list (A.7). `propose()` has no side effects. Separate streams for proposals, look-ahead, execution, steering coins and propensities. Test 2: env, trace, explorer, archive, bandit and tracker hashes equal before and after a sandbox pass. Test 6: O_1 through the sandbox ≡ V2s. The dice caption asserts only actions, frames and max S; a failure aborts that run (counted; one dated fix) | A.6, A.7, Part B |
| M7 | O runs do not survive routine kills; compute undercounted (can-it-fail and feasibility: major) | **Unconditional checkpoint and resume** for every v85 job at expansion boundaries, every ≤ 5 min wall. Per-expansion RNG streams, so no sequential generator state. Resume-identity test counts toward KC-build. Queue: 14.5 GB launch gate, staggered starts, 25-min back-off after any TERMINATE, longest-first dynamic claiming, unlimited passes, v85 jobs in the supervisor's WORKER list. **Entry cached per (unit, start)**, which saves about 133k frames per run. Every stage recounted; caps of 1.6G (development) and 0.75G (sealed) | §13 |
| M8 | The carried macro library (rows log every 4th step) cannot express the oracle's exact patterns (feasibility: major) | One representation for all family-M objects and for K\*mac: **slots held 4 steps**, priv → NOOP. Miners read only `slot, it, t, rise_lag` and the exp_meta ADV flag, never `view`. K\*mac-exact is a caption row; round-trip fidelity is reported | §4.3, §4.4 |
| M9 | No cheap pre-gate before G2's expensive run set (feasibility: major) | G2a: K and F only, about 10 runs per unit, before G2b | §8.1 |
| M10 | K\*mac replayed the demonstration's opening and did not match what M can carry (honesty: major) | Cross-start library (start A uses start B's lineage and vice versa), quantised as above. The exact same-lineage library is a caption only | §4.3 |
| M11 | The NO-ROOM reading went beyond the instruments (honesty: major) | Sentence restricted to these instruments, strengths, units and rungs. MAP-WEAK turns NO-ROOM-K(map) into INCONCLUSIVE-K. "Rewinding substitutes for knowing" and "no carried object can" are dropped. One sentence on how NO-ROOM bounds v84 routes S and Σ | §11, A.11 |

### 1.2 Minor issues

| Issue | Fix |
|---|---|
| Greedy frontier (GF) missing as a W tuned default | GF is a tuned default for W in G2b and B1, and a G1-P placebo |
| Seeds and names | 9873–9876 reserved for the 8-seed rule. Every derived draw uses `default_rng([base, purpose])`. Family M's learner from scratch is **F_M**. Odds marked conditional or overall. Sealed games are "not opened before v85's B0 eligibility runs" |
| Schedule optimistic | Dates given as ranges: G1 verdict about day 5–6, NO-ROOM path ends about day 6, full path about day 13–14 |
| Calibration CPU cost; unresampled comparator | Vectorised bootstrap with index matrices. Statistics stored once per simulation, then evaluated on the grid. Comparator resampled in every simulation. Permutation p left out. G-build test: under 30 min |
| Sandbox pitfalls | Flags instead of deep copies. best_S starts at S(s1). Ledger cap = clock start + 4M. No tracking writes kept |
| W steering cost | Features vectorised per expansion; measured in test 9 |
| A KC-build confined to O | Drops only the selection row (and the dice caption); G1 runs D, K\*map and K\*mac |
| D's share is an artefact | D is called a plumbing control. log R_K / log R_D(unclipped) is reported as "a ratio, not a decomposition" |
| s1 chosen with hindsight; S(s1) handed over | Stated in the claim header and the `senses` field |
| "Live" wording | "Online" means played in the emulator with billed rewinds; "live play" is savestate-free. LP is labelled "savestate-free, sticky 0, not the vision's live mode" and gains an empty-library arm |
| Odds, cost and wall-time figures | Fixed: G1b walls 19–150 min (mean 41; 23 of 82 above 45 min); B1 recounted at 38–42 runs per unit |
| Novelty wording; missing cousins | "A per-channel room table with a plumbing control". Backplay, reverse curriculum, FiGAR and STRAW added |
| Smaller gaps | A D fix that touches V2s's closure triggers the certificate. INSTRUMENT-FAIL stops before G1. Sealed entry seeds allocated. φ named. Route S re-test stated. "A miner reacquires its knowledge quickly" is marked as an extrapolation |

---

## 2. Claim header, words, senses, oracle rule

**Claim header** (protocol and every verdict JSON):

> "Every v85 claim concerns a SEARCH WITH BILLED REWINDS (savestates; 60 frames per restore; every emulated frame of the agent billed). It starts from a savestate shared by all arms: a state, chosen by the evaluator, at the game's first matched rung M1 on an earlier V2 run's lineage that reached the top rung. The root's sensed rise is set to that lineage's own sensed rise and is identical for all arms. It uses the senses granted on 22 October 2026: pixels, its own buttons, max(0, r_step) per step, lives. Evaluator instruments (D, O, K\*) additionally use unbilled look-ahead or future-path data, reported as oracle frames; they are not agents. It is not live play without savestates."

**Words.**
- *Online*: played in the emulator with billed rewinds. *Live play*: savestate-free.
- *Room*: an evaluator instrument's effect.
- *Object room*: a same-game object's effect against its learner from scratch.
- *Transfer*: a carried object's effect against the same learner from scratch on unseen units.
- *Faster learning*: reserved for §15's conditions.

**Agent inputs.**
- Pixels, own buttons, sensed rises, lives; root S = S(s1).
- No RAM, true score, game names or downloaded weights.
- **No v85 component reads RAM, instruments included.**

**Instruments.**
- D, O_4, O_4-dice, K\*map, K\*mac, K\*mac-exact and the G1-P placebos live in `ragnarok/evaluation/`.
- They use only evaluator-held data: future lineages and unbilled look-ahead. They never count as agents and run on development games only.

**Claim-bearing arms.** X, F, the controls and the placebos are built only from agent-side data: logs of the agent's own play on earlier games.

**Pools.** Reserved games (boxing, frostbite, riverraid, kangaroo) are never opened; enduro is excluded.

---

## 3. Ground facts (checked against files)

| Fact | Source |
|---|---|
| 22/35 room development games. The 16 with k\* ≥ 4: alien, amidar, asterix, bank_heist, chopper_command, gopher, jamesbond, kung_fu_master, phoenix, pooyan, qbert, up_n_down, video_pinball, wizard_of_wor, yars_revenge, zaxxon | `room_verdict_v84.json` |
| Natural gap T(M2) − T(M1) is under 20k frames on 19 of 32 runs (as low as 0 on qbert 9712, 320 frames on yars 9713). qbert 9712's first M1 record has cum 775 ≥ M2 = 562.5. Rungs M5 and M6 of jamesbond are reached by the same record on both seeds | G1b records (PowerShell) |
| With the counted-rung rule of A.4 (≥ 50k median gap), by hand: alien {3,4}; amidar {4,5,6}; asterix {3–7}; bank_heist {4,5}; chopper {3,4,5}; gopher {4,5,6}; jamesbond {3,4,5,7,8}; kung_fu {3,4,5}; phoenix {4,5}; pooyan {3,4,5}; qbert {4–7}; up_n_down {3,4,5}; video_pinball {3,4,5}; wizard {3–7}; yars {3,4,5}; zaxxon {3,4} | same; S0 recomputes |
| Slower seed's gap to k\*: up to 3.29M (zaxxon 9712), 2.90M (up_n_down), 2.62M (phoenix, asterix), 2.45M (pooyan). M_top sits near V2's 4M frontier on several units | same |
| Per-run variance of λ: 0.200 raw on the 16 units; 0.376 official (winsorised, 22) | v84 verdict, audits |
| Entry (`Body.enter`) averages 132,654 billed frames per run (82 G1b runs). It is billed to the same ledger, and its adapter keeps only plain tables after construction | G1b JSON; `entry.py`, `adapter.py` |
| G1b wall times: 19–150 min, mean 41; 23 of 82 above 45 min. 31 TERMINATEs in bursts seconds apart that hit every worker; 186 SUSPENDs | G1b JSON; `g1b_supervisor_actions.txt` |
| V2 draws everything from one sequential RNG (`default_rng(seed)`, entry included); the propensity generator is `default_rng([seed, 1])`. `pick_cell` increments `chosen`. The rollout mutates best_S, best_S_billed, best_S_exp, any_rise, hist, coarse_count, tracker pos and miss, and archive slots (eviction at 20k) | `explorer2.py`, `v2_job_v84.py` |
| Rows log the slot only at every 4th step (t ≥ 4). Rows carry `view` (2,048 B) | `explorer2.py` l.300; `rowlog.py` |
| Trace84 records a rung **at clone time**. `SenseEnv.step` calls `trace.on_rise`; `_check` raises `Spent` at the cap | `senses_v84.py`, `senses_v83.py` |
| Replays restore after one NOOP (after-step) or post-reset; a rung counts as confirmed in either context (note 16) | build notes 15–16 |
| A v84 verdict is an Interruption if runs carry more than one `code_tree` (git tree of ragnarok/, scripts/, tests/) or a dirty flag | `room_verdict_v84.py` l.182–186 |
| G1b peak commit 0.21–0.61 GB per 4M run; about 5M billed frames per hour per worker after kills | G1b JSON; feasibility audit |
| Seed block 9800–9899: no project `.py` file in the 11 worktrees contains it | three audits' greps; repeated at S0, F1, F2 |
| Sealed pool: 24 games, 7 hard exploration; second round: 9 games. Near-siblings: pacman~ms_pacman, frogger~freeway, atlantis2~atlantis | `pools_v83.py` |

---

## 4. Mechanism

### 4.1 V2s, the comparator and the base of every arm
V2s keeps V2's laws and constants (v84 §5), with these changes:
- The root is s1, with best_S = S(s1).
- **Per-expansion random streams.** Each stream is seeded by the expansion index alone, so a run can resume from a checkpoint and arms can be compared draw for draw.
- Row logging is off; exp_meta and exp_feat are kept.
- A side-effect-free `propose()`.
- An injected fifth process Q5 and injected steering hooks. Family code is never imported by V2s.

V2s equals V2 in law, not in bytes. Every v85 contrast is V2s against V2s-based arms. The exact specification is in A.6.

### 4.2 Shared starts and counted rungs
- **Starts.** For each unit, start A comes from G1b seed 9712's top lineage and start B from 9713's. s1 is that lineage's first step with cum ≥ M1.
- **Entry.** Run once per (unit, start) and cached.
- **Start identity.** A fresh emulator must re-reach every rung of the lineage above s1 exactly.
- **Counted rungs.** Fixed at S0 from reference data only (A.4). The top rung is cross-fitted at F1 from the other start's V2s runs (A.10.4).
- **Not measured.** Knowledge that speeds the climb to M1 is invisible here. This is declared; it is what "matched rungs" requires.

### 4.3 Instruments for G1 (exact text in A.7)

| Row | Channel | Information used | Role |
|---|---|---|---|
| D | — | The start's own lineage from s1 to M_top, replayed open-loop | Plumbing control |
| **O_4** | Which proposal to bill | One unbilled look-ahead for each of V2s's own 4 proposals; the winner executed on a fresh stream | Gate row; licenses W |
| **K\*map** | Where to restart | Screens (s_bin and coarse code) of the start's own future lineage | Gate row; licenses nothing in v85 |
| **K\*mac** | How to move | 32-step windows before rises on the **other start's** lineage, held 4 steps | Gate row; licenses M |
| O_4-dice | — | As O_4, but the winner is replayed on its own look-ahead stream | Caption only (1 run per unit) |
| K\*mac-exact | — | The same start's own lineage, exact per-step slots | Caption only (1 run per start) |

### 4.4 Carried families (code and development objects frozen at F1, before G1)

**Family W, "which proposal to bill."** This is v84 route S, run online. v84 closed route S only through an instrument ceiling (notes 14, 18), so v85 re-tests it.
- **Object X_S:** a zero-mean prior Σ_s over the slopes of a logistic ADV critic on the 44 search-sense features (v84 §6.2), with a free intercept. Recipe: v84 A.5–A.8, with the null-space threshold at 1e-2 (note 8).
- **Sources:** the G1b expansion logs of the other 40 burned games, leave-one-game-out (LOGO), in development. All 41 minus near-siblings for the sealed test.
- **Steering:** P3 (v84 §6.2). 32 candidate cells are drawn with V2's weights, every (cell, process) pair is scored, and one is picked ∝ exp(1.5·clip(z, −2, 2)), mixed 0.5 with V2s's own draw. All of this uses the steering stream C_e. Refit every 100 expansions; active only after the first refit that sees both classes, so W is identical to V2s before that. Features are vectorised per expansion.
- **κ:** the A.9 rule run offline on the G1b logs of the 4 room calibration games (pong, breakout, seaquest, asteroids), separately for F_S, X_S, K_S, P_S, U_S and T, with the edge rule. Committed at F1.
- **Arms:**
  - F_S: isotropic κ_F, the learner from scratch.
  - K_S: Σ from this game's own G1b expansions, the same-game anchor.
  - P_S: ADV labels permuted within each source run (`[9803, 2]`).
  - U_S: input-PCA, spectrum-matched (`[9803, 3]`).
  - T: the T_clock prior (v84 §6.1).
  - S: the best single process, leave-one-game-out, with no bandit.
  - GF: greedy frontier. With probability 0.5 (C_e), pick the cell with the largest S; ties go to the deepest, then uniform.
- **Head start.** A zero-mean prior with no data predicts a constant, and steering is off until the first two-class refit. The carried object can act only through in-run learning, so no X_frozen arm is defined for W.

**Family M, "how to move."**
- **Object:** a library of 32 macros. Each macro is a run-length sequence of canonical slots, each held for 4 steps; 8–64 steps long, in multiples of 4. Priv slots map to NOOP; an unavailable slot falls back to its movement class without extra, then to NOOP.
- **Miner (frozen now):**
  - Candidates: the 8 rows (32 steps) before each sensed rise in ADV rollouts (rise_lag < 4).
  - Lift: the equal-game mean of [(c_adv + 1)/(N_adv + 1)] ÷ [(c_base + 1)/(N_base + 1)], with c_base taken from one random 32-step window per non-ADV rollout (`[9803, 4]`).
  - Keep: the top 32 by lift, with c_adv ≥ 3 in at least 2 games, pairwise RLE edit distance ≥ 3, and at most 4 macros dominated by one game.
  - Fields read: `slot, it, t, rise_lag` and exp_meta ADV only.
- **Q5:** draws macros ∝ count with a 10% uniform mixture and chains them to the end of the rollout. The bandit covers Q1–Q5.
- **Online:** every 250 expansions the run re-mines its own rollouts (every 4th executed slot); prior pseudo-counts n₀ = 10 merge with the run's own counts.
- **Arms:**
  - X_on: the carried library plus online mining.
  - **X_frozen:** the carried library with no mining.
  - **F_M:** an empty library plus mining. It is identical to V2s until 250 expansions.
  - K_mine: the miner run on this game's own G1b rows, the same-game anchor.
  - DM_on: the designer library (below) plus mining.
  - P_on: windows from non-advancing rollouts, matched on length and slot frequency (`[9803, 1]`), plus mining.
  - S: as in family W.
- **Designer library DM (frozen now), 32 macros:**
  - holds of 8 and 32 steps in 8 directions (16);
  - 16-step holds with extra in 8 directions (8);
  - an extra burst of 4 on, 4 off for 32 steps;
  - left-right and up-down alternations of 8;
  - two diagonal zigzags;
  - NOOP for 16 steps;
  - extra held for 32 steps;
  - right-run-then-jump and left-run-then-jump (8 + 8).

---

## 5. Yardstick: the shared-start climb (SSC), summary
- **Times.** T(M_k) is the number of billed frames from clock start to the first clone with cum ≥ M_k. The cap is 4M billed frames from s1, with an evaluator stop at the reference top rung.
- **Per-run measures:**
  - λ: the mean over the counted rungs of log max(5,000, T(M_k)).
  - λ_late: the same from each arm's own first counted rung onward. It stops an arm that only jumps the first rung.
- **Aggregation:** per-unit ratios r_g, clipped to [1/8, 8]; R is their geometric mean.
- **Bounds:** a two-level bootstrap gives LB90 and UB95.
- **Contrasts:** pairwise intersection-union; every named contrast must pass on its own.

Details are in A.8–A.9.

---

## 6. Before thresholds: S0, I-stage, calibration, F1 (summary; exact text in A.10)
- **S0** (0 billed frames):
  - start rebuild and identity, entry caches;
  - counted rungs, physics floor;
  - natural-run spike-ins to pick U;
  - the dice caption's predicted gain;
  - a stall inventory for v86;
  - seed grep.
- **I-stage** (V2s 6 per unit, D 1 per start):
  - D control and instrument line;
  - cross-fitted top rungs;
  - spike-in calibration in the real regime, with the frozen selection rule picking n, τ_pt and τ_lb.
- **F1 commit:** thresholds, rungs, calibration, every hash (s1 blobs, entry caches, maps, libraries, family code and development objects, closure hashes per arm type) and the budget projection. G1 launches only after F1.

---

## 7. G1, G1-P and routing

### 7.1 G1
- **Rows:** O_4, K\*map, K\*mac; n runs per unit each (half per start). V2s is reused from the I-stage. Caption rows are queued last and never delay the verdict.
- **Labels per row:** ROOM-K, NO-ROOM-K, INCONCLUSIVE-K.
- **Verdict:** ROOM, NO-ROOM or INCONCLUSIVE.
- **Post-G1 check:** may only downgrade a label.

Exact text in A.11–A.13.

### 7.2 G1-P (runs only for ROOM rows; same units and seeds)
- **Placebos.** For O_4: **GF4**, which picks among the same 4 proposals the one whose cell has the highest S (then more lives, then shallower, then lowest j), with no look-ahead. For K\*map: **GF**. For K\*mac: **K̃mac**, another unit's cross-start library (derangement `[9801, 2]`).
- **Labels:**
  - **GENERIC-K** if the UB90 of R(K:placebo) < 1.25. The row is then removed from routing.
  - Otherwise **SEPARATED-K** if LB90 > 1.
  - Otherwise **NOT-SEPARATED-K** (routing proceeds; caption).
- G1-P never changes the G1 verdict.

### 7.3 Routing (frozen)

| G1 or G1-P result | Consequence |
|---|---|
| ROOM-O_4, not GENERIC | Family W licensed, so G2a(W) runs |
| ROOM-K\*mac, not GENERIC | Family M licensed, so G2a(M) runs |
| Both licensed | G2a for both. G2b for the family that passes G2a with the larger point log R(K:F); a tie goes to M |
| ROOM, but only K\*map, or every ROOM row GENERIC | **ROOM-UNCARRIED.** v85 ends: "room exists in channel(s) …; no v85 family holds that information." v86 default: a screen-seeing "where" object (the deferred representation channel) |
| NO-ROOM or INCONCLUSIVE | v85 ends (§11) |

---

## 8. G2: object room, then rehearsal (development; single shot each)

**Common setup.** Same U units, starts, rungs, cap and yardstick. Seeds 9826–9831; V2s reused under the certificate rule.

### 8.1 G2a, object room (each licensed family)

| Family | K (4 runs per unit) | F (6 runs per unit) |
|---|---|---|
| W | K_S | F_S |
| M | K_mine | F_M |

- **OBJECT-ROOM** if:
  - point R(K:F) ≥ 1.4;
  - LB90 of R(K:F) > 1;
  - LB90 of R(K:V2s) > 1.
- Otherwise **KC-object:** "this family's same-game object, built from the agent's own play on that game, does not beat its own learner from scratch online; its carried version is not run." The sealed pool stays closed.
- **Ceiling-to-label rule (frozen):** the sealed test may award FASTER-2× only if G2a gave point R(K:F) ≥ 2.0 and LB90 ≥ 1.5. Otherwise the top sealed label is PARTIAL.

### 8.2 G2b, carried rehearsal (one family)

| Family M arms (new runs per unit) | Family W arms (new runs per unit) | Role |
|---|---|---|
| X_on (6) | X_S (6) | carried |
| X_frozen (4) | — (W cannot hold a static head start) | head start against adaptation |
| DM_on (4), S (4) | T (4), S (4), GF (4) | tuned defaults |
| P_on (4) | P_S (4), U_S (4) | placebos |
| X4_on: 2 draws × 2 runs (`[9804, 1]`) | X4_S: 2 × 2 | dose |
| F_M, K_mine, V2s reused | F_S, K_S, V2s reused | — |

**GO** requires all of:
- R(X:F) ≥ 1.4 with LB90 > 1;
- LB90 > 1 against V2s, each tuned default and each placebo;
- LB90 of R_late(X:F) > 1.

Otherwise **KC-dev:** "the carried object does not beat its own learner from scratch, the tuned defaults and the placebos on development starts." The sealed pool stays closed.

**Reported.**
- φ = log R(X:F) / log R(K:F); the X4 dose.
- For family M:
  - R(X_on:X_frozen);
  - the interaction I = [λ(X_frozen) − λ(X_on)] − [λ(V2s) − λ(F_M)], with a bootstrap interval;
  - HEAD-START-VOCABULARY if LB90 of R(X_on:X_frozen) ≤ 1;
  - ADAPTS if it is > 1.
- CENSORING-SENSITIVE: misses at 2× cap; if a label changes, it drops one step.

---

## 9. Sealed test (once)

**F2 sealed freeze.**
- A 3-agent adversarial review.
- The frozen code builds X_41 and its near-sibling variants (pacman excludes ms_pacman, frogger excludes freeway, atlantis2 excludes atlantis), plus the dose draws X4 and X12 (`[9804, 2]`, `[9804, 3]`).
- SHA-256 of every object, the code and the thresholds is committed.
- A power simulation with the G1 and G2 variances is committed. **Rule:** if P(PARTIAL | true 1.5×) < 0.5 at 6 seeds, X and F go to 8 seeds (9873–9876).

**B0, eligibility.**
- **Pool:** the 17 non-hard sealed games. Bars on seed 9850; natural V2 runs on 9851–9852 from frame 0, cap 4M.
- **A unit needs:** G1b rule (a)–(d) on both seeds, k\* ≥ 4, a valid s1 at ≥ 1 start, and |K_g| ≥ 2.
- **Count:**
  - more than 10 eligible: draw 10 with seed 9860;
  - fewer than 6: one second round (bars 9870, eligibility 9871–9872, units pooled);
  - still fewer than 6: **NOT MEASURABLE**.
- `sealed_eligibility.json` and its hash are committed before any arm.
- The 7 hard-exploration games stay unopened for v86.

**B1, arms.**
- Starts come from the 9851 and 9852 lineages, with entry seeds 9862 and 9863 and the same S0 rules.
- **Family M** (38 runs per unit): X_on, F_M and V2s 6 each; X_frozen, DM_on, S and P_on 4 each; X4 and X12 2 each.
- **Family W** (42 runs per unit): X_S, F_S and V2s 6 each; T, S, GF, P_S and U_S 4 each; X4 and X12 2 each.
- Top rungs are cross-fitted by the frozen verdict code from the V2s runs before any other arm's records are read.

**Labels.** Headers say SEARCH WITH BILLED REWINDS FROM A SHARED SAVESTATE and list the senses. The first match in the Part C order wins: NEGATIVE-TRANSFER, SEARCH-CLIMB FASTER-2×, SEARCH-CLIMB FASTER-PARTIAL, EARLY-RUNGS-ONLY, TUNED-DEFAULTS, GENERIC, NOT-SIGNIFICANT, NULL.

**Captions.**
- ACCUMULATES: Page's L over doses 0/4/12/41 with p ≤ 0.05, and LB90 of R(X41:X4) > 1. Headline figure: the learning-set curve, with F, the tuned defaults and the placebos drawn as flat lines.
- HEAD-START-VOCABULARY or ADAPTS (family M).
- CENSORING-SENSITIVE.
- INSTRUMENT-FAIL makes the result INCONCLUSIVE.
- A "without near-siblings" line.

**LP, live probe** (family M only; report only; after the B1 verdict is committed).
- Savestate-free play at sticky 0. This is **not** the vision's live mode (p = 0.25).
- On each sealed unit: 20 episodes × 7,000 steps from reset, 0–30 NOOPs (seed 9861).
- Four policies: the online bandit over X's library with mining; the same over an empty library (F_M); the same over DM; the unit's best B′ bar mode.
- Reports mean sensed rises in episodes 1–5 against 16–20. Never a claim.

---

## 10. Gates

| Gate | Pass | Stop or kill |
|---|---|---|
| G-build | tests 1–8 (Part B) | **KC-build** after 2 fix rounds. If confined to the O path (tests 2 and 6 on O), only the selection row and the dice caption are dropped |
| S0 | ≥ 10 units with a valid start and ≥ 2 counted rungs; floor ratio ≥ 8; U chosen | **KC-start**, **KC-floor** |
| I-stage | D passes; instrument line holds; selection rule yields (n, τ_pt, τ_lb) | D: one dated fix, then Interruption. **INSTRUMENT-FAIL** stops before G1. **UNDERPOWERED** after one top-up |
| F1 | commit | — |
| **G1** | ROOM on ≥ 1 row | NO-ROOM or INCONCLUSIVE: v85 ends (§11) |
| G1-P | adds GENERIC, SEPARATED or NOT-SEPARATED; routes | ROOM-UNCARRIED ends v85 |
| G2a | OBJECT-ROOM | **KC-object** |
| G2b | GO | **KC-dev** |
| F2 | 3-agent review | — |
| B0 | ≥ 6 units | **NOT MEASURABLE** |
| B1 | label ladder | — |
| LP | report only | — |

**Allowed later changes.** Each is a dated amendment, committed before the data it affects:
1. the cut ladder;
2. engineering that leaves behaviour unchanged (resource gates, worker count, checkpoint cadence), verified by the resume-identity test;
3. one D fix;
4. one calibration top-up;
5. Interruptions, with first-pass outputs kept.

Nothing else changes.

---

## 11. If the answer is no

**NO-ROOM: what it means.** "On these U dense-reward development units, from a savestate shared by all arms at M1, three evaluator instruments did not make this billed-rewind climber 2× faster at the next counted rungs (95% upper bounds below 2, also with V2s misses counted at 2× cap; geometric mean over units). The instruments: best-of-4 of its own proposals by one unbilled look-ahead each, executed on a fresh stream; restarting half the time from the cell matching the furthest screen of a future successful path; and motor patterns of another successful path of the same game, held 4 steps. Effects smaller than 2×, up to the reported upper bounds, are not excluded."

**What it bounds.**
- v84 route S **indirectly**: a critic that predicts, from 44 summary features, the choice look-ahead selection makes is unlikely to beat that selection.
- Route Σ (consequence heads steering cells and buttons) only indirectly, through K\*map and K\*mac.

**Not established:** anything about representation knowledge, other learners or substrates, live play, or effects below 2×.

**INCONCLUSIVE.** The bounds are reported, and the same pivot follows.

**Default v86 after NO-ROOM or INCONCLUSIVE** (Jérémie can redirect): **rescue from stall savestates**, the hard-games proposal repaired.
- **Starts:** stall states from the S0 inventory (plateaus of at least 1M frames), plus the development failure games after screening out ladder artefacts.
- **Score:** P(one rung within 2M) against fresh V2 seeds.
- **Controls:** a demonstration control, the H teleport as a negative control, and calibration by resampling real runs.
- **Sealed:** the 7 hard-exploration games.

**Second axis:** a rewind-price dial (restores at 600 frames, or a restore budget), moving toward live play.

**After ROOM-UNCARRIED:** a screen-seeing "where" object.

**After KC-object or KC-dev:** "there is room, and this object does not capture it." v86 tries another object for the same channel.

---

## 12. Pools and seeds

**Pools.**
- **Development units:** the 16 room development games with k\* ≥ 4, in the draw order of `default_rng([9801, 1])`.
- **Smoke runs:** breakout and pong.
- **Sources:** the G1b logs of all 41 burned games, leave-one-game-out in development.
- **Sealed:** the 17 non-hard sealed games. The 7 hard ones stay sealed for v86. The 9 second-round games are a top-up only.

**Seeds.**

| Seed | Use |
|---|---|
| 9800 | Smoke runs and tests (never a measurement) |
| 9801 | `[9801, 1]` unit order; `[9801, 2]` K̃mac derangement |
| 9802 | S0 and I-stage spike-in simulations (`[9802, b]`) |
| 9803 | `[9803, 1]` P_on; `[9803, 2]` P_S; `[9803, 3]` U_S; `[9803, 4]` miner baseline windows |
| 9804 | `[9804, 1]` development X4 draws; `[9804, 2]` sealed X4; `[9804, 3]` sealed X12 |
| 9805 | Development bootstrap |
| 9806–9808 / 9809–9811 | V2s at start A / B (instrument line: {9806, 9809} against {9807, 9810}) |
| 9812–9814 / 9815–9817 | G1 rows at A / B (n = 4 uses 9812–9813 and 9815–9816); G1-P reuses them |
| 9818 | Caption rows (O_4-dice: start A for units at even draw positions, B for odd; K\*mac-exact at both starts) |
| 9819 / 9820 | Development entry seed and emulator seed for every arm at A / B; D's agent seed |
| 9821–9825 | Reserve |
| 9826–9828 / 9829–9831 | G2a and G2b at A / B. 4-run arms: 9826, 9827, 9829, 9830. Dose draws: draw 1 on 9826 (A) and 9829 (B), draw 2 on 9827 (A) and 9830 (B) |
| 9832–9849 | Development reserve |
| 9850 / 9851–9852 | Sealed bars / eligibility |
| 9853–9855 / 9856–9858 | B1 at A / B (4-run arms: 9853, 9854, 9856, 9857; 2-run arms: 9853, 9856) |
| 9859 / 9860 / 9861 | Sealed bootstrap and permutation / unit draw / LP |
| 9862 / 9863 | Sealed entry and emulator seed at A / B |
| 9870 / 9871–9872 | Second round: bars / eligibility (arms reuse 9853–9858) |
| 9873–9874 / 9875–9876 | Extra B1 seeds at A / B (the 8-seed rule) |
| 9877–9899 | Reserve |

**Reference data.** G1b runs 9712/9713 are read as data only. No new run uses a 97xx seed. Avoid 8101–8106, 9101–9256 and 9700–9749.

---

## 13. Compute, RAM, resumability, queue, schedule

### 13.1 Compute
**Basis.** About 5M emulated frames per hour per worker (G1b after kills), so 15M per hour on 3 workers. V2s from s1 averages about 1.45M billed per run. Steered rows are assumed at 1.1–1.4M. O rows emulate 5× their billed frames (4 look-aheads plus the execution). Entry is cached.

| Stage | Runs | Emulated frames | Wall, 3 workers |
|---|---|---|---|
| S0 | replays and 32 entry caches | ≈ 7M (evaluator) | ≈ 3 h CPU |
| G-build | ≈ 12 smoke runs | ≈ 25M | ≈ 2 h |
| I-stage (U = 12) | 72 V2s + 24 D | ≈ 105M billed | ≈ 8–10 h |
| **G1 core** (U = 12, n = 4) | 48 O_4 + 48 K\*map + 48 K\*mac | ≈ 410M (≈ 180M billed + 230M oracle) | ≈ 27–35 h |
| G1 captions | 12 O_4-dice + 24 K\*mac-exact | ≈ 90M | ≈ 6 h (after the verdict) |
| **Up to the G1 verdict** | | **≈ 0.55G (≈ 0.32G billed)** | |
| G1-P (if ROOM) | ≤ 3 × 48 | ≤ 180M | ≤ 12 h |
| G2a | 120 per family | ≈ 155M per family | ≈ 10 h |
| G2b | 312 (M) / 360 (W) | ≈ 420–470M | ≈ 28–31 h |
| B0 | bars 17 + 34 natural | ≈ 12M + 136M | ≈ 10 h |
| B1 (U ≤ 10) | 380 (M) / 420 (W) | ≈ 500–550M | ≈ 34–37 h |
| LP | ≤ 10 × 4 × 20 episodes | ≈ 22M | ≈ 2 h |

**Caps.** Development 1.6G; sealed 0.75G.

**Budget rule (G1).** At F1, project the cost from the I-stage mean V2s billed per run, counting censored runs at the cap.
- If the projection exceeds 0.6G, O_4 runs at n = 4 whatever n is chosen, and its power is reported from the stored grid.
- Worst case (U = 16, n = 6, O_4 held at 4): about 0.68G, about 45 h.

**Cut ladder** (budget or calendar only, never results):
- C1: drop the G2b dose arm;
- C2: G2 units 12 → 10, the first 10 in draw order;
- C3: sealed units capped at 8;
- C4: drop the B1 dose arms.

### 13.2 RAM, GPU, disk
- Target ≤ 0.7 GB peak commit per worker. Measured 0.21–0.61 GB for 4M runs; v85 runs are shorter, the sandbox makes no clones, and checkpoints are written streaming, never as an in-memory copy.
- 3 workers by default (≤ 2.1 GB). A 4th only while the supervisor shows ample margin; worker count = min(4, ⌊3.6 GB / measured peak⌋).
- GPU: 0 h. Disk: about 12 GB (no row logs; one rolling checkpoint of ≤ 0.3 GB per job).

### 13.3 Resumability and queue (built before the I-stage)
- **Checkpoint/resume** for every v85 job at expansion boundaries, every ≤ 5 min wall, written atomically (tmp + replace). Contents:
  - explorer: archive arrays, codes and fine lists, key maps, free-slot order, clock, evictions, resolution; bandit; hist; coarse_count; best_S triple; any_rise; exp_meta and exp_feat; injected steer or Q5 state;
  - Body (tracker);
  - SenseEnv85: store blobs and meta, next handle, anchor and since, cum and depth, frame and load counters, clock start, oracle frames;
  - Trace85: lineage, bases, records, rise lists.
- **No RNG state is saved**: every stream is indexed by expansion. On resume: fresh emulator, reset, NOOP, then the next expansion restores a handle (after-step context, as in a live run).
- **Queue `queue_v85.py`:**
  - launch gate ≥ 14.5 GB free on min(RAM, commit);
  - starts staggered by 2 min;
  - 25-min back-off after any TERMINATE;
  - longest-first dynamic job claiming;
  - unlimited passes.
- **Supervisor:** `v85_job`, `v85_stage0`, `v85_calib` and `v85_objects` go in `scripts/ram_supervisor.py`'s WORKER tuple. Guard at 14 GB free. Launches follow the ragnarok-run skill.

### 13.4 Code identity
- **Each run records:** git commit, dirty flag, and the **closure hash**: SHA-256 over the sorted (path, file SHA-256) pairs of every project module in `sys.modules` at run end, plus the job script.
- **Within a gate:** each arm type has exactly one closure hash, equal to the one committed for that gate. Anything else is an Interruption.
- **Reusing V2s in a later gate:** valid if its closure hash is unchanged. Otherwise the certificate runs: V2s seeds 9806 and 9809 on the first unit in draw order are re-executed, and their records and exp_meta must be byte-identical. If the certificate fails, all V2s runs are redone.

### 13.5 Schedule (day 0 = Saturday 24 October, plan commit)

| Day | Work |
|---|---|
| 0–2 | Worktree; protocol EN/FR; Part B items 1–11; G-build; S0 |
| 2–2.5 | S0 commit; 2-agent review of the sandbox, V2s, checkpointing and verdict code |
| 2.5–3.5 | I-stage (≈ 8–10 h); **meanwhile Part B items 12–13** (families, objects, κ); cross-fit; calibration; **F1** |
| 3.5–5.5 | **G1; verdict about day 5–6.** NO-ROOM, INCONCLUSIVE or UNDERPOWERED paths end about day 6 |
| 6–7 | G1-P (if ROOM); G2a |
| 7–9 | G2b; verdict about day 9 |
| 9–10 | 3-agent review; F2 |
| 10–13 | B0, then B1; verdict about day 13; LP |
| 13–14 | French report and handoff |

---

## 14. Odds (mine; gates are correlated)

| Event | P |
|---|---|
| KC-start / KC-floor | 0.04 / 0.01 |
| KC-build, whole / confined to O | 0.03 / 0.06 |
| D passes (≤ 1 fix) and instrument line holds | 0.93 |
| Calibration qualifies (U ≤ 16, n ≤ 6) | 0.8 |
| **G1 runs** | ≈ 0.70 |
| O_4: ROOM / INCONCLUSIVE / NO (given G1 runs) | 0.30 / 0.35 / 0.35 |
| K\*map: ROOM / INCONCLUSIVE / NO (given G1 runs) | 0.35 / 0.30 / 0.35 |
| K\*mac: ROOM / INCONCLUSIVE / NO (given G1 runs) | 0.07 / 0.23 / 0.70 |
| G1, given it runs: ROOM / INCONCLUSIVE / NO-ROOM | 0.50 / 0.33 / 0.17 |
| **G1 overall: ROOM / INCONCLUSIVE / NO-ROOM / does not run** | **0.35 / 0.23 / 0.12 / 0.30** |
| Given G1 runs: W licensed / M licensed / ROOM-UNCARRIED | 0.25 / 0.06 / 0.20 |
| G2a OBJECT-ROOM: W / M | 0.35 / 0.40 |
| G2b GO given OBJECT-ROOM: W / M | 0.30 / 0.35 |
| Sealed measurable | 0.8 |
| PARTIAL or better / 2×, given GO and measurable | 0.4 / 0.1 |
| **Sealed SEARCH-CLIMB FASTER-PARTIAL or better, overall** | **≈ 1%** |
| **Sealed SEARCH-CLIMB FASTER-2×, overall** | **≈ 0.2–0.3%** |
| ACCUMULATES | < 0.3% |

**Most likely endings.**
- About 0.30: G1 does not run, mostly UNDERPOWERED. That is a decision for Jérémie.
- About 0.33: ROOM, then a stop at ROOM-UNCARRIED, GENERIC, KC-object or KC-dev.
- About 0.23: INCONCLUSIVE.
- About 0.12: NO-ROOM.
- About 0.02–0.03: the sealed test is reached.

**Why so low.** Carried knowledge must beat its own learner from scratch, not just V2s. For the critic, v84 notes 14 and 20 show a fresh learner catching up quickly. For the miner, that is an extrapolation.

---

## 15. What each outcome would and would not establish

**G1 ROOM-K.**
- **Establishes:** "From a savestate shared by all arms at M1 on U development games, same-game information of kind K, supplied by an evaluator instrument (O_4 with 4 unbilled look-ahead rollouts per expansion), made this billed-rewind climber reach the next counted rungs faster: point R̂ (≥ τ_pt), 90% lower bound ≥ τ_lb. The calibration in the same regime gives P(ROOM | true 1.5×) ≤ 0.25 and P(ROOM | true 2×) ≥ 0.8."
- **Does not establish:** that such information can be learned or carried; faster learning; live play; that O_4 bounds every critic.

**G2a OBJECT-ROOM.**
- **Establishes:** a same-game object of the family, built from the agent's own earlier play on that game, speeds this climb against its own learner from scratch on development starts.
- **Does not establish:** transfer.

**Sealed SEARCH-CLIMB FASTER-2× or -PARTIAL, family M.**
- **Establishes:** "On N Atari games not opened before v85's eligibility runs, a search with billed rewinds whose motor-pattern process starts from a vocabulary mined from its own play on 41 earlier games, and keeps mining online, climbed the next counted rungs ≥ 2× (or significantly) faster than the same miner starting empty. It was also faster than the plain climber, designer macros, its best single process and a non-advancing placebo vocabulary. Time is counted in billed frames; the senses are pixels, buttons, score rises and lives."
- **Head start against adaptation:**
  - With HEAD-START-VOCABULARY: "a carried vocabulary speeds the search climb" (a head start, not faster learning).
  - "The carried knowledge accelerates in-run learning" only with ADAPTS and an interaction lower bound > 0.

**Sealed, family W.** "…whose expansion critic learns with a slope prior learned on 41 earlier games…". The prior acts only through in-run refits, so the effect is on learning speed.

**Never established by any outcome:**
- live play;
- faster arrival at M1;
- true score;
- faster seeing;
- transfer to NES;
- the hard-exploration games;
- human-like learning;
- accumulation without ACCUMULATES.

**NO-ROOM.**
- **Establishes:** the bounded negative of §11.
- **Does not establish:** anything about representation, other substrates, live play, or effects below 2×.

---

## 16. Cousins and honest novelty
*From memory; a literature check is due before any novelty claim.*

- **Starting from shared or demonstration states:** Go-Explore (Ecoffet et al. 2019; Nature 2021); Salimans & Chen 2018; Backplay (Resnick et al. 2018); reverse curriculum generation (Florensa et al. 2017); restart distributions (Kakade & Langford 2002).
- **Learned cells or selection:** policy-based and Latent Go-Explore (Gallouédec & Dellandréa 2023); Intelligent Go-Explore (Lu, Hu & Clune 2024; foundation models, excluded here); learned node selection in branch and bound (He, Daumé & Eisner 2014); ExIt and AlphaZero; Levin tree search and policy-guided heuristic search (Orseau et al. 2018; Orseau & Lelis 2021).
- **Look-ahead as a verifier:** best-of-N (Brown et al. 2024); hindsight optimisation (Yoon et al. 2008).
- **Exploration controllers:** Agent57 and NGU (Badia et al. 2020); MEME (Kapturowski et al. 2022); temporally extended ε-greedy (Dabney et al. 2021).
- **Macros, skills and options:** McGovern & Sutton 1998; Durugkar et al. 2016; Chang et al. 2019 (macros reused across Atari, the closest prior work for M); FiGAR (Sharma et al. 2017); STRAW (Vezhnevets et al. 2016); action priors (Rosman & Ramamoorthy 2012); action grammars (Lange & Faisal 2019); options, option-critic, DIAYN, portable options, eigenoptions.
- **Meta-learned exploration:** MAESN, E-RL², Alet et al. 2020, First-Explore, AdA.
- **Measurement and priors:** jumpstart against time-to-threshold (Taylor & Stone 2009); learning sets (Harlow 1949); human priors (Dubey et al. 2018; Tsividis et al. 2017); random against learned features (Burda et al. 2019).

**Not new:** any mechanism.

**New is the test.** A pre-registered **per-channel room table with a plumbing control**, measured online from shared savestates, with:
- exact time-dilation spike-ins planted in the same regime before thresholds are instantiated;
- a dice-free selection instrument;
- routing by information match;
- an object-room gate;
- a transfer contrast against the same learner from scratch, with a frozen-vocabulary arm separating head start from adaptation;
- pairwise tuned-default and placebo controls;
- a learning-set dose curve;
- no RAM anywhere, no downloaded weights, and only the score-rise and lives senses.

The per-channel shares are ratios, not a decomposition.

---

# Part A. First-gate specification (G1 with S0, I-stage and F1), ready to freeze

**A.0 Status.**
- Frozen by the day-0 commit of this document.
- F1 fills only these blanks, by the rules below: U and the unit list; K_{g,s}; n, τ_pt, τ_lb; hashes; the budget projection.
- Allowed changes: §10 only.

**A.1 Question.** For each channel c ∈ {selection (O_4), where (K\*map), how (K\*mac)}: does same-game information supplied by the evaluator make V2s reach the next counted rungs about 2× faster, in billed frames, from a savestate shared by all arms at M1? D is the plumbing control. The representation channel is not tested.

**A.2 Units.**
- **Eligible:** the 16 units of §12, kept or dropped by A.3 and A.4.
- **Order:** `default_rng([9801, 1]).permutation(eligible)`; the first U are used.
- **U ∈ {10, 12, 16}:** by A.10.1(c), capped at the number of valid units. One top-up before F1 takes the next units in order.

**A.3 Starts (S0, 0 billed frames).** For unit g and reference run r (A = 9712, B = 9713):
1. **Top lineage.** Take the first record with cum ≥ M_{k\*}. Replay its `rec_actions` from `base_state` in the after-step context; else post-reset (note 16). Neither matches: the start is invalid.
2. **s1.** The state at the first lineage step with cum ≥ M1. Store the ALE blob, S(s1), lives, the step index, the ALE actions s1 → M_top and the per-step cum.
3. **Entry cache.**
   - `SenseEnv85(g, seed=E)` with E = 9819 (A) or 9820 (B).
   - `Body.enter(env, default_rng([E, 0]), E)`.
   - Pickle Body (background, adapter tables, tracker colours and size) and record its SHA-256 and the entry frames.
   - Every arm at that start loads this Body. Entry is billed to the start and off every clock.
4. **Start identity.** Fresh `SenseEnv85(g, seed=E)`, reset, one NOOP, restore s1, replay the lineage above s1. Every rung M2..M_{k\*} must be reached at its recorded step with its recorded cum. A failure makes the start invalid.
5. **Drops.** An invalid start is dropped. A unit with no valid start is dropped, with no replacement. **KC-start** if fewer than 10 units have a valid start and ≥ 2 counted rungs.

**A.4 Counted rungs (S0, reference data only).**
- **Rule.** K_g = {k : 2 ≤ k ≤ k\*, M_k > S(s1) at every valid start of g, and the median over the two reference runs of T_r(M_k) − T_r(M1) ≥ 50,000 billed frames}.
- Rungs reached by the same record on both reference runs are merged; the lowest is kept.
- A unit is dropped if |K_g| < 2.
- k_first = min K_g; k_ref = max K_g, which sets the evaluator stop.
- §3 lists the expected result. The 50k threshold was set with the G1b gaps in view and uses no v85 data.

**A.5 Run protocol (every development arm).**
- **Environment:** `SenseEnv85(g, seed=E_s, trace=Trace85)`; Body from the cache.
- **Start:** `env.start_from(blob, S(s1))` does reset, one NOOP, restore s1 (billed 60); then cum = S(s1), depth = 0, trace base = the s1 blob, clock start = billed now.
- **Cap and stop:** cap at clock start + 4,000,000 billed frames; evaluator stop at the first clone with cum ≥ M_{k_ref}.
- **Times:** T(M_k) = billed frames since clock start at the first clone with cum ≥ M_k. A miss is censored at 4M.
- **Records:** [billed since clock, cum, true, depth, oracle frames since clock].
- **Replay confirmation:** the records reaching M_{k_first} and the highest reached counted rung are replayed on a fresh emulator (E_s; reset, NOOP, restore s1, the lineage's actions since s1). cum must match exactly; a mismatch is an Interruption.
- **Checkpointing:** per §13.3.
- **JSON fields:** the records; T for each counted rung; confirmations; billed and oracle frames; coverage and ties (A.7); s1, Body and object hashes; closure hash; dirty flag; peak commit; wall time; `claim_header`; `senses`.

**A.6 V2s (`ragnarok/agent_v85/explorer2s.py`).** v84 V2 (§5 constants; 20k-cell cap; re-key every 250 expansions), with:
- **Root:** the root is s1, cloned after `start_from`. best_S = S(s1), best_S_billed = clock start, best_S_exp = 0, any_rise = False.
- **Streams.** For expansion e:

  | Stream | Seed | Draws |
  |---|---|---|
  | P_{e,j} | `[seed, 85, e, j, 0]` | cell, process, first slot of proposal j |
  | X_{e,j} | `[seed, 85, e, j, 1]` | look-ahead j |
  | X_e | `[seed, 85, e, 1000]` | the executed rollout's in-rollout draws |
  | C_e | `[seed, 85, e, 2000]` | every steering coin and steered pick |
  | propensity | `[seed, 1, e]` | — |

- **`propose(e, j, cell=None)`** returns (cell, process, first slot) from P_{e,j}. It increments no `chosen` count and touches neither the bandit nor the archive.
- **V2s** executes `propose(e, 0)` on X_e. `chosen` is incremented only for the executed cell.
- **Logging:** row logging off; exp_meta and exp_feat kept.
- **Hooks:** steering hooks (`pick_cell`, `pick_process`, `on_put`, `on_iter`) and a Q5 process object are **injected**; explorer2s imports no family or evaluation module. With no library, the bandit has k = 4 and the run is byte-identical to V2s; with a library, k = 5.

**A.7 Instruments (`ragnarok/evaluation/oracle_v85.py`; development only; never agents).**
- **Sandbox `env.sandbox()`.**
  - Saves and restores: ALE state, frame and load counters (sandbox frames go to `oracle_frames`, never billed), anchor and since, cum and depth.
  - Disables `trace.on_clone`, `trace.on_rise`, the cap check, and `clone()` (which raises).
  - Explorer side: local copies of best_S, best_S_billed and best_S_exp (extension rule) and of any_rise. Tracker pos, miss and any mutable tracker field are saved and restored. The archive is read-only plus a per-candidate overlay of the keys the candidate would create (eviction not simulated; a documented approximation). No bandit, hist, coarse_count or exp-row writes.
  - Test 2: state hashes of env, trace, explorer, archive, bandit and tracker are equal before and after a sandbox pass.
- **D.**
  - The first rollout plays the start's own lineage actions (ALE indices) from s1 open-loop, with ordinary puts, up to M_{k_ref}; the evaluator stop then ends the run.
  - Agent seed E_s; 1 run per start.
- **O_4 (gate row).** Each expansion:
  1. Candidates j = 0..3 from `propose(e, j)`.
  2. Each is played in the sandbox on X_{e,j}.
  3. Score s_j = (max(0, max S in the rollout − best_S), ADV against archive ∪ overlay, number of keys new to archive ∪ overlay), compared lexicographically. Ties go to the lowest j.
  4. The winner w is executed for real on **X_e**: billed, traced, archived, `chosen` incremented, the bandit updated on q_w, the propensity of q_w logged.

  O_1 (1 candidate through the sandbox) ≡ V2s byte for byte (test 6). Reported: `oracle_frames`; **O-TIES**, the share of expansions where all 4 scores tie, as a caption if above 70%; the rank correlation between the winner's look-ahead score and the executed outcome.
- **K\*map (gate row).**
  - **Map:** (s_bin, coarse code) along the start's own lineage above s1, every 4th step, consecutive duplicates removed.
  - **Map index**, set on each put: the largest i with equal s_bin and a coarse-code Hamming distance ≤ 4 of 48 blocks; otherwise −1.
  - **Steering**, each expansion: if a C_e coin is below 0.5 and some cell has a map index ≥ 0, pick uniformly (C_e) among the cells with the largest index. Process and first slot then come from `propose(e, 0, cell)`. Otherwise V2s's own draw.
  - **Coverage** = the share of coins below 0.5 that found a matching cell. **MAP-WEAK** if coverage < 30%.
- **K\*mac (gate row).**
  - **Library**, built from the **other** start's lineage above its s1 (if the other start is invalid, the unit has no K\*mac row):
    - the 32-step windows ending at each sensed rise;
    - the actions at steps 0, 4, …, 28 of each window, mapped to canonical slots through this start's cached adapter (inverse map), each held 4 steps;
    - unmapped actions and priv slots become NOOP;
    - run-length encoded; unique sequences weighted by count.
  - **Q5** as in §4.4, with in-rollout draws from X_e.
  - **Reported:** round-trip fidelity, the share of priv or NOOP-mapped steps, the library size.
- **Caption rows** (never in labels or routing; queued after the core rows; the verdict never waits for them):
  - **O_4-dice:** as O_4, but the winner is re-executed on X_{e,w}. Actions, frames and max S are asserted equal to its look-ahead; a failure aborts the run (counted; one dated fix). The realised R is reported beside S0's prediction G_4.
  - **K\*mac-exact:** the start's own lineage, every step's slot.

**A.8 Yardstick per run.** For (unit g, start s, arm a), over counted rungs K_{g,s} (A.10.4):
- λ = mean over k ∈ K_{g,s} of log max(5,000, T(M_k)).
- λ_late = mean over k ∈ K_{g,s} \ {k_first} of log max(5,000, T(M_k) − T(M_{k_first})). If M_{k_first} is missed, each term is log(4M).
- **Instrument rows additionally** report T on total emulated frames (billed + oracle).

**A.9 Statistics.**
- ℓ_g(a:c) = mean over valid starts s of [mean over runs λ(c; g, s) − mean over runs λ(a; g, s)]. r_g = exp(ℓ_g), clipped to [1/8, 8]. R = exp(mean_g log r_g).
- **Bootstrap:** two levels (units, then runs within each (unit, start, arm)), 2,000 resamples, `[9805, ·]`.
  - LB90 = 10th percentile; UB90 = 90th; UB95 = 95th.
  - R_late is computed the same way from λ_late.
- **Breadth:** the share of units with r_g ≥ 1.5.
- **Censoring rate:** the share of an arm's runs that miss M_{k_top(g,s)}.
- Every named contrast is tested on its own (intersection-union).

**A.10 Stages before G1.**

1. **S0 (committed before any v85 live frame).**
   - (a) A.3 and A.4 on 16 units × 2 starts.
   - (b) **Physics floor** per start: F = 4·Δdepth(s1 → M_{k_ref}) + 60·⌈Δdepth/400⌉, compared with the natural gap. **KC-floor** if the median ratio is below 8.
   - (c) **Natural spike-ins** on the 32 G1b runs: counted rungs, timed from their own M1. Pick the cheapest U ∈ {10, 12, 16} with P(ROOM | ×2) ≥ 0.8 at n = 4, τ = 0.2, thresholds 1.75 / 1.25.
   - (d) **G_4 prediction** for the dice caption: p = the share of expansions between M1 and M_{k_ref} that raised archive-best S; G_4 = mean over segments of (1 − (1 − p)^4)/p.
   - (e) **Stall inventory**, report only.
   - (f) Seed grep.
2. **I-stage runs.** On the first U units: V2s 3 per start (9806–9808 at A, 9809–9811 at B); D 1 per start.
   - **D control:** r_g(D:V2s) ≥ 4 on ≥ 80% of units, with every D confirmation exact. First failure: one dated fix (if V2s's closure changes, the certificate of §13.4 runs), then D reruns. Second failure: Interruption; G1 does not run.
   - **Instrument line:** V2s {9806, 9809} against {9807, 9810}. **INSTRUMENT-FAIL**, which stops before G1, if its 99% interval excludes 1.
3. **Regime ratio** (report): natural V2 from its own M1 against V2s from s1, on the counted rungs.
4. **Cross-fitted top rung (F1).**
   - For (g, s): k_top(g, s) = the highest k ∈ K_g reached by **all 3** V2s runs at the other start s′. If s′ is invalid, k_top = k_ref.
   - K_{g,s} = {k ∈ K_g : k ≤ k_top(g, s)}. Drop (g, s) if |K_{g,s}| < 2.
   - Fixed before any steered row runs.
5. **Calibration (`v85_calib.py`; vectorised; < 30 min).** On the I-stage V2s runs at the F1 rungs, per simulation b (`[9802, b]`), 1,000 simulations per scenario:
   - **Comparator:** for each (g, s), 3 run vectors resampled from that (g, s)'s V2s runs as mean + centred residual × √(3/2).
   - **Three steered rows:** n/2 base vectors per (g, s), resampled the same way.
   - **Planting:** T_k ← T_k / D_eff for every counted k, with D_eff = D·exp(τ·z), z ~ N(0, 1) per (unit, row).
   - **Extra run noise:** every T of a planted run is multiplied by exp(ε), ε ~ N(0, 0.5·σ̂²_{g,s}), so the variance is ×1.5.
   - **Scenarios:** D ∈ {2, 1.5, 1.25, 1}; **D_late** (gaps after the run's own M_{k_first} halved); **D_jump** (T(M_{k_first}) set to 5,000, later gaps unchanged); and all three rows at D = 1 together.
   - **Censoring:** a censored base run stays at the cap when computing P(ROOM | D2), P(NO-ROOM | D2), mean R̂(D2) and mean R̂_late(D_late). It gets cap / D_eff when computing every false-ROOM rate.
   - **Labels:** computed by the frozen verdict functions, with 1,000 bootstrap resamples.
   - **τ:** 0.2 is primary; 0 and 0.4 are reported.
   - **Grid:** n ∈ {4, 6}, τ_pt ∈ {1.6, 1.7, 1.8, 2.0}, τ_lb ∈ {1.15, 1.25, 1.35}. Take the smallest n, then the largest τ_pt, then the largest τ_lb that satisfy all of:
     - (i) P(ROOM | D2) ≥ 0.80;
     - (ii) **P(ROOM | D1.5) ≤ 0.25**;
     - (iii) P(ROOM | D1.25) ≤ 0.10;
     - (iv) P(any ROOM among 3 | all D1) ≤ 0.05;
     - (v) P(ROOM | D_jump) ≤ 0.05;
     - (vi) P(NO-ROOM | D2) ≤ 0.05;
     - (vii) P(NO-ROOM | D1) ≥ 0.80;
     - (viii) mean R̂(D2) ∈ [1.8, 2.2] and mean R̂_late(D_late) ∈ [1.8, 2.2].
   - **If nothing qualifies:** one top-up (6 V2s and 2 D runs on each further valid unit, in order), then recompute. If still nothing: **UNDERPOWERED**; G1 does not run, and Jérémie decides.
   - **Reported:** every operating characteristic, including P(NO-ROOM | D1.5).
6. **F1 commit.** `f1_thresholds.json`:
   - U and the units, K_{g,s}, n, τ_pt, τ_lb and the operating characteristics;
   - the calibration JSON;
   - the budget projection and the O_4 rule (§13.1);
   - SHA-256 of the s1 blobs, entry caches, maps and libraries, family code, development objects and κ;
   - closure hashes per arm type;
   - the seed grep.

**A.11 Labels per row K ∈ {O_4, K\*map, K\*mac}, against V2s.**
- **ROOM-K** requires all of:
  - (a) point R ≥ τ_pt;
  - (b) LB90 ≥ τ_lb;
  - (c) LB90 of R_late > 1;
  - (d) r_g ≥ 1.5 on ≥ 40% of units;
  - (e) fewer than 40% of K's runs and fewer than 40% of V2s's runs censored.

  Caption **BELOW-2X** if, in addition, UB95 < 2.
- **NO-ROOM-K** (only if not ROOM-K) requires all of:
  - (a) UB95 < 2;
  - (b) UB95 < 2 with V2s misses counted at 8M;
  - (c) at most 25% of K's runs and at most 25% of V2s's runs censored;
  - (d) for K\*map: not MAP-WEAK.
- **INCONCLUSIVE-K:** anything else.

**A.12 Post-G1 operating-characteristic check (downgrade only).** For each row:
- v_K = pooled within-(g, s) variance of λ(K) / that of V2s; τ̂_K = method-of-moments heterogeneity of ℓ_g.
- If v_K > 1.5 or τ̂_K > 0.4, rerun the frozen calibration at (v_K, τ̂_K) for that row.
- ROOM-K becomes INCONCLUSIVE-K if (ii)–(v) fail. NO-ROOM-K becomes INCONCLUSIVE-K if (vi) fails.

**A.13 Verdict, sentences, routing.**
- **Interruption** (no verdict): a replay mismatch, mixed or changed closure hashes, a dirty tree, or a missing comparator certificate.
- **ROOM:** at least one ROOM-K. Sentence (per row): "From a savestate shared by all arms at M1 on U development games, same-game information of kind K, supplied by an evaluator instrument [O_4: with 4 unbilled look-ahead rollouts per expansion; on total emulated frames R = …], made this billed-rewind climber reach the next counted rungs faster: point R̂ (≥ τ_pt), 90% lower bound ≥ τ_lb (calibrated: P(ROOM | 1.5×) ≤ 0.25, P(ROOM | 2×) ≥ 0.8). This is room: not transfer, not a carried object, not faster learning."
- **NO-ROOM:** all three rows NO-ROOM-K (or, after an O-confined KC-build, both remaining rows, with "selection channel not tested"). Sentence: §11, with each row's UB95.
- **INCONCLUSIVE:** otherwise. "Room for 2× was neither shown nor excluded in any tested channel." Bounds reported.
- **Routing:** §7.3, after G1-P (§7.2).

**Captions and reports:**
- MAP-WEAK, O-TIES, BELOW-2X;
- per-unit r_g and per-rung profiles;
- censoring rates;
- the regime ratio;
- oracle frames and R on total emulated frames;
- K\*map coverage; K\*mac fidelity;
- log R_K / log R_D(unclipped), "a ratio, not a decomposition";
- the caption rows.

**A.14 Seeds and budget.** Seeds per §12. Budget per §13.1. No code commit touches a closure in use while that gate's runs are in flight.

**A.15 Outputs.** `F:\dev\ragnarok_v85\craft_v6_out\v85\`:
- `s0/` (`starts.json`, entry caches, `rungs.json`, `floor.json`, `natural_spikeins.json`, `stalls.json`);
- `istage/`;
- `f1_thresholds.json`, `calib.json`;
- `g1/` (run JSONs, `g1_verdict.json`);
- `g1p/`.

Each JSON carries the commit, closure hash, dirty flag, input hashes, `claim_header` and `senses`.

---

# Part B. Code to write, in order

**Before the I-stage** (all committed; their closures hashed):

1. **Worktree and protocol.** `F:\dev\ragnarok_v85` (branch `arc4-v85`); protocol EN/FR committed.
2. **`ragnarok/evaluation/senses_v85.py`.**
   - SenseEnv85: `start_from(blob, S)`; `sandbox()` (flags, counter save and restore, clone forbidden); cap relative to clock start; `oracle_frames`.
   - Trace85: records relative to clock start, oracle-frame column, evaluator stop, `replay85` from the s1 blob.
   - State export and import for checkpoints.
3. **`ragnarok/evaluation/start_v85.py`.** Top-lineage replay (both contexts), s1 extraction, start identity, Body entry cache, the counted-rung rule.
4. **`ragnarok/agent_v85/explorer2s.py`.** V2s per A.6: streams, side-effect-free `propose`, injected steering and Q5 hooks, `to_state` and `from_state`, vectorised `_features`.
5. **`ragnarok/evaluation/oracle_v85.py`.** D, O_m (sandbox, overlay, independent execution), O_4-dice, K\*map, K\*mac and K\*mac-exact, the placebos GF4, GF and K̃mac.
6. **`ragnarok/evaluation/ckpt_v85.py`.** Streaming atomic checkpoints every ≤ 5 min at expansion boundaries; resume.
7. **`scripts/v85_job.py`.** One run (unit, start, arm, seed). Arms come from a registry of modules (`ragnarok/evaluation/arms_v85/<arm>.py`), so new arms are new files and this script is never edited after the I-stage launch. Closure hash and JSON per A.5.
8. **`scripts/v85_stage0.py`.** A.10.1.
9. **`scripts/v85_calib.py`.** A.10.5, vectorised; writes `f1_thresholds.json`.
10. **`scripts/v85_verdict.py`.** SSC, bootstrap, labels and captions for the I-stage, G1, G1-P, G2a, G2b and B1; closure and certificate checks; censoring sensitivity.
11. **`scripts/queue_v85.py`** and the supervisor's WORKER entries (§13.3).

**During the I-stage, committed before G1 at F1** (outside V2s's and the instruments' closures):

12. **`ragnarok/agent_v85/macros.py`** (Q5 process, miner, DM, P_on generator, X_frozen switch) and **`ragnarok/agent_v85/critic_s.py`** (route S online steering, refits, prior application). Arm modules for F_M, X_on, X_frozen, K_mine, DM_on, P_on, S, F_S, X_S, K_S, P_S, U_S, T and GF.
13. **`scripts/v85_objects.py`.** Builds and hashes the development objects: X (LOGO per unit), K_mine, K_S, P_on, P_S, U_S, T, S, DM, the X4 draws, and κ by the A.9 rule. The sealed objects are built at F2 by the same frozen code.

**G-build tests** (failures of tests 1–8 after 2 fix rounds give KC-build; if confined to the O path, only the selection row and the dice caption are dropped):
1. **Isolation.** `agent_v85` has no `evaluation` imports, game names, RAM or private access. Instruments live only in `evaluation`. explorer2s imports no family module.
2. **Ledger and sandbox.** The ledger never decreases; each restore bills 60. Sandbox frames are never billed and never traced: a sandbox rollout that crosses a rung leaves Trace unchanged. Every state hash is equal before and after a sandbox pass.
3. **Start identity** on 2 units, both starts, from cached Bodies.
4. **Determinism.** Two smoke runs on seed 9800 are identical.
5. **Resume identity.** Kill at a random expansion, including mid-sandbox, then resume: the final JSON is byte-identical apart from wall and provenance fields.
6. **Prefix identity.** Each of these is byte-identical to V2s:
   - O_1 through the sandbox;
   - K\*map with an empty map;
   - Q5 with an empty library;
   - F_M, for 250 expansions;
   - every W arm, until its first two-class refit.
7. **Verdict and calibration code.**
   - Exact R recovered on synthetic data when neither floor nor cap binds.
   - Censoring treatments.
   - Intersection-union logic, label precedence, the cross-fit rule.
   - The comparator certificate.
   - Calibration runtime under 30 min.
8. **O_4-dice re-execution** matches its look-ahead on smoke runs (actions, frames, max S).
9. **Engineering (never a kill).** Peak commit, frames per second per arm type, checkpoint size and time, W steering cost, worker count.

---

# Part C. Outcome labels

| Gate | Label | Condition | Meaning (sentence) | Consequence |
|---|---|---|---|---|
| G-build | KC-build | Tests 1–8 fail after 2 fix rounds | The v85 instruments or yardstick could not be built to pass their identity tests | v85 stops. If confined to O: selection row dropped, G1 proceeds |
| S0 | KC-start | < 10 units with a valid start and ≥ 2 counted rungs | Shared savestates cannot be rebuilt reliably; the yardstick does not exist | v85 stops |
| S0 | KC-floor | Median physics-floor ratio < 8 | Lineage depth leaves no room for 2× | v85 stops |
| I-stage | Interruption (D) | D fails twice | The shared-start pipeline does not reproduce a demonstration | G1 does not run |
| I-stage | INSTRUMENT-FAIL | V2s seed-split 99% interval excludes 1 | The yardstick disagrees with itself | G1 does not run |
| I-stage | UNDERPOWERED | No (n, τ_pt, τ_lb) meets (i)–(viii) after one top-up | The yardstick cannot separate 2× from 1.5× and 1× at affordable seeds | G1 does not run; Jérémie decides |
| G1 row | ROOM-K | A.11 (a)–(e), and the A.12 check holds | Same-game information of kind K has room: point ≥ τ_pt, LB90 ≥ τ_lb | Routing (§7.3) |
| G1 row | NO-ROOM-K | A.11 NO-ROOM (a)–(d), and the A.12 check holds | This instrument, at this strength, did not give 2× (UB95 < 2); smaller room not excluded | — |
| G1 row | INCONCLUSIVE-K | Otherwise | Neither shown nor excluded | — |
| G1 | ROOM | ≥ 1 ROOM-K | Room exists in a named channel (not transfer, not learning) | G1-P, then G2a or ROOM-UNCARRIED |
| G1 | NO-ROOM | All rows NO-ROOM-K | The §11 sentence | v85 ends; v86 stall rescue by default |
| G1 | INCONCLUSIVE | Otherwise | Room for 2× neither shown nor excluded | v85 ends; same pivot |
| G1 | Interruption | Replay mismatch, mixed closure, dirty tree, missing certificate | No verdict | One dated fix; rerun labelled |
| G1-P | GENERIC-K | UB90 of R(K:placebo) < 1.25 | The room is explained by a generic heuristic | Row removed from routing |
| G1-P | SEPARATED-K / NOT-SEPARATED-K | LB90 > 1 / otherwise | The oracle beats / does not clearly beat its placebo | Caption; routing proceeds |
| Routing | ROOM-UNCARRIED | ROOM, but no row licenses a family | Room exists; no v85 family holds that information | v85 ends; v86: a screen-seeing "where" object |
| G2a | OBJECT-ROOM | Point R(K:F) ≥ 1.4; LB90 R(K:F) > 1; LB90 R(K:V2s) > 1 | The family's same-game object beats its own learner from scratch online | G2b |
| G2a | KC-object | Otherwise | The same-game object does not beat its learner from scratch; the carried version is not run | Sealed stays closed |
| G2b | GO | §8.2 | The carried object beats F, V2s, the tuned defaults and the placebos on development starts | F2 |
| G2b | KC-dev | Otherwise | The carried object does not beat its own learner from scratch, the tuned defaults and the placebos | Sealed stays closed |
| B0 | NOT MEASURABLE | < 6 eligible units after the second round | The sealed pool cannot host the yardstick | v85 ends |
| B1 (1) | NEGATIVE-TRANSFER | UB95 of R(X:F) < 1 | The carried object slows the climb | Kept |
| B1 (2) | SEARCH-CLIMB FASTER-2× | Allowed by the ceiling rule. R(X:F) ≥ 2.0, LB90 ≥ 1.3, p ≤ 0.025. R(X:c) ≥ 1.5 with LB90 > 1 for c = V2s and each tuned default. LB90 > 1 against each placebo. LB90 of R_late(X:F) > 1. Leave-one-game-out minimum of R(X:F) > 1 | §15 sentence | Report |
| B1 (3) | SEARCH-CLIMB FASTER-PARTIAL | R(X:F) ≥ 1.25, LB90 > 1, p ≤ 0.025. LB90 > 1 against V2s, each tuned default and each placebo. LB90 of R_late(X:F) > 1 | §15 sentence, "significantly faster" | Report |
| B1 (4) | EARLY-RUNGS-ONLY | The PARTIAL λ clauses pass; LB90 of R_late(X:F) ≤ 1 | Faster only at the first counted rung | Report |
| B1 (5) | TUNED-DEFAULTS | A tuned default beats V2s (LB90 > 1) and X fails its clause against it | Earlier games tuned the search; this is not faster learning | Report |
| B1 (6) | GENERIC | X beats F, V2s and every tuned default (LB90 > 1), but not a placebo | Vocabulary or steering effect without game content | Report |
| B1 (7) | NOT-SIGNIFICANT | Point R(X:F) ≥ 1.25, but LB90 ≤ 1 or p > 0.025 | Not shown | Report |
| B1 (8) | NULL | Otherwise | No effect shown | Report |
| Caption | ACCUMULATES | Page's L over doses 0/4/12/41 with p ≤ 0.05, and LB90 of R(X41:X4) > 1 | More earlier games, faster climb | Headline curve |
| Caption | HEAD-START-VOCABULARY / ADAPTS (M) | LB90 of R(X_on:X_frozen) ≤ 1 / > 1 | Fixed vocabulary advantage / in-run mining adds speed. "Accelerates in-run learning" also needs an interaction lower bound > 0 | Sentence wording |
| Caption | CENSORING-SENSITIVE | Misses at 2× cap change a claim label | — | The label drops one step |
| Caption | BELOW-2X, MAP-WEAK, O-TIES | A.11, A.7 | Instrument and strength notes | MAP-WEAK turns NO-ROOM-K(map) into INCONCLUSIVE-K |

---

# Part D. Résumé pour Jérémie

v84 a donné V2, un grimpeur qui monte vraiment. Mais aucun savoir n'a encore été essayé en jeu (avec retours en arrière payés).

Premier test de v85 : tous les bras repartent de la même sauvegarde, au premier palier de chaque jeu, donc sans avance de départ.

On mesure si un savoir « tricheur » du même jeu fait grimper V2 environ deux fois plus vite, par trois voies :
- choisir parmi ses propres essais : un coup d'œil gratuit en avant, puis l'essai choisi est rejoué avec un hasard neuf, pour ne pas acheter de la chance ;
- savoir où repartir : les écrans du bon chemin ;
- savoir comment bouger : les gestes d'un autre bon chemin.

Avant de fixer les seuils, on plante sur de vrais runs des accélérations connues (×1, ×1,25, ×1,5, ×2) et on vérifie que la règle les lit juste. Dire « pas de place » exige une borne haute sous 2.

Une voie n'ouvre un objet transportable que si cet objet peut porter cette information. On teste d'abord sa version « même jeu » contre le même apprenant parti de zéro.

Ensuite seulement vient la version apprise sur 40 autres jeux. On la compare aux réglages, à des placebos et à une version figée, pour séparer avance de départ et apprentissage plus rapide. Puis on la teste une seule fois sur des jeux neufs.

Tout le code et tous les objets sont figés avant le premier résultat. Les runs reprennent après une coupure.

Coût : environ 0,55 milliard de trames jusqu'au premier verdict (vers le jour 5-6), sans GPU, moins de 0,7 Go par tâche.

Chances :
- de la place quelque part : environ 35 % (50 % si le test tourne) ;
- un gain transporté et confirmé : environ 1 % ;
- un 2× confirmé : environ 0,3 %.

Si c'est « non », le résultat est net et borné. Suite par défaut : les jeux où la recherche se bloque, ou moins de retours en arrière.

Rien à décider maintenant. Seul cas qui t'attend : une mesure trop bruitée pour trancher (« sous-puissance »).