# v85 plan draft and audits (workflow wf_8db018d6-25c, unedited)

# v85 plan: measure live whether knowledge has room in V2's climb, then carry one object and test it once on sealed games

*Synthesis of 5 proposals and 3 judges. Editor's note: I only read and grepped files. I used awk for arithmetic on the v84 JSON. I ran no Python, emulator or GPU job, and edited no file. The plan is to be committed as `docs/ARC4_V85_PROTOCOL_EN.md`, with a French twin, in a new worktree `F:\dev\ragnarok_v85` on branch `arc4-v85`.*

---

## 0. Summary

**Question.** v85's first decisive gate asks, live and cheaply: does any knowledge short of a demonstration make V2 climb about 2× faster from a matched rung?

**Shared savestate.** Every arm of a game starts from the same savestate **s1**: the state at rung M1 on a reference V2 lineage of that game. Each arm starts with an empty archive and an identical entry. Head start in state, archive and perception is therefore zero by construction.

**Where knowledge can act.** Knowledge can change V2's time through four channels: which proposal gets billed (selection), where it restarts (cells), how it moves (button proposals), and how it keys cells (representation). G1 runs one same-game oracle on each of the first three:
- **O_4**: best-of-4 foresight over V2's own proposals. It is an evaluator instrument.
- **K\*map**: the future lineage's screens steer where V2 restarts ("where").
- **K\*mac**: the demonstration's own motor patterns, with no index on place or time ("how").
- **D**: replays the demonstration. It is the trivial positive control.
- The representation channel is **not tested** in v85. This is declared, and every negative names the channels it covers.

**Before any threshold is frozen:**
- **S0** (0 billed frames): exact time-dilation spike-ins on the G1b runs, a physics floor, and the rebuild of s1.
- **I-stage** (live, V2s and D only): measures the noise of the shared-start regime and re-runs the spike-ins inside that regime.
- **F1**: a rule committed today turns those null data into thresholds and seed counts.

**Labels per channel.** ROOM, NO-ROOM or INCONCLUSIVE. NO-ROOM requires evidence of absence: a 95% upper bound below 2.

**If ROOM.** One carried, game-agnostic family is built:
- **W**: v84's cross-game critic over cells and processes (route S), run live for the first time;
- **M**: a cross-game motor vocabulary, added as a fifth button process.

It is rehearsed on development games against the same learner starting from scratch, tuned defaults and placebos, with pairwise contrasts. Then it is tested once on never-opened sealed games, with a learning-set dose curve.

**If NO-ROOM.** v85 ends with a live, calibrated negative. The default for v86 is rescue from stall savestates and the 7 unopened hard-exploration games.

**Cost and timing.**
- First decisive verdict about **day 4–5**.
- About **0.47G emulated frames**: 0.25G billed, plus 0.22G of unbilled oracle look-ahead.
- 0 GPU-hours. At most 0.7 GB per worker.

**My odds.**

| Event | P |
|---|---|
| G1 ROOM (mostly through foresight) | ≈ 0.6 |
| G1 NO-ROOM | ≈ 0.13 |
| G1 INCONCLUSIVE | ≈ 0.25 |
| Sealed PARTIAL or better | ≈ 2–3% |
| Sealed FASTER-2× | ≈ 0.5–1% |

---

## 1. Claim header, senses, oracle rule

**Claim header.** It appears in the protocol and in every verdict JSON:

> "Every v85 claim concerns a SEARCH WITH BILLED REWINDS (savestates; 60 frames per restore; every emulated frame of the agent billed), started from a savestate shared by all arms at the game's first matched rung (M1). It uses the senses granted on 22 October 2026: pixels, its own buttons, max(0, r_step) per step, lives. It is not live play without savestates."

**Agent inputs.**
- Pixels, own buttons, sensed rises and lives.
- The agent-side S at the root is S(s1): the agent's own cumulative sensed rise along the reference lineage, which is its own earlier play.
- No RAM, no true score, no game names, no downloaded weights.
- **No v85 component reads RAM, instruments included.** The representation oracle that would need RAM is deferred (§11).

**Instruments.** D, O_m, K\*map, K\*mac and the G1-P placebos are evaluator instruments:
- they live in `ragnarok/evaluation/` and inject only evaluator-held data (future lineages, unbilled look-ahead);
- they never count as agents and carry no claim;
- they run on development games only, never on sealed games.

**Claim-bearing arms.** X, F, the controls and the placebos are built only from agent-side data: logs of the agent's own play on earlier games.

**Pools.** Reserved games (boxing, frostbite, riverraid, kangaroo) are never opened. Enduro is excluded.

---

## 2. Ground facts (checked against files)

| Fact | Source |
|---|---|
| 22/35 room development games; the 16 with k\* ≥ 4 are alien, amidar, asterix, bank_heist, chopper_command, gopher, jamesbond, kung_fu_master, phoenix, pooyan, qbert, up_n_down, video_pinball, wizard_of_wor, yars_revenge, zaxxon (14 have k\* ≥ 5) | `room_verdict_v84.json` |
| T(M1) on room development games: 97,572 (qbert 9713) to 510,492 (hero 9713). Entry is about 167k frames on alien, whose T1 is 183k, so M1 comes soon after entry | same file; `g1b/alien_9712.json` |
| C = T(M_k\*) − T(M1) on the 16 units: 0.17M (bank_heist) to 3.29M (zaxxon) | same |
| Lineage depth at M3 on the 16 units: 455 (qbert) to 5,355 steps (video_pinball) | same |
| Per-run variance of λ: 0.376 official (winsorised, 22 games); **0.497 raw (22); 0.200 raw on the 16 units with k\* ≥ 4**. Heavy tails: carnival's seed gap is 2.55 log units | computed with awk from the λ pairs |
| Each G1b run stores records [billed, cum, true, depth]; an NPZ with exp_feat (44), exp_meta (8 columns), rec_actions/rec_offsets and base_state; rows (slot, it, t, rise_lag, every 4th step). 742 MB in total | `v2_job_v84.py`, `rowlog.py`, `g1b/` |
| V2 draws everything from one RNG. Steering hooks: `pick_cell`, `pick_process`, `on_put`, `on_iter`. The bandit has k = 4 arms | `explorer2.py` |
| Trace84 records a rung **at clone time** (`on_clone`), so look-ahead that clones would leak rungs | `senses_v84.py` |
| Replays restore after one NOOP (after-step) or post-reset; a rung counts as confirmed in either context (note 16) | build notes 15–16; `room_verdict_v84.py` |
| Peak commit: 0.23–0.56 GB in G1a and up to 0.61 GB in G1b (up_n_down). A 4M run takes 19–45 min wall. G1b achieved about 5–5.5M billed frames per hour per worker | `g1a_report.json`, `g1b/*.json` |
| Seed block 9800–9899: no project `.py` file (virtualenvs excluded) in any of the 11 worktrees contains it | grep |
| Sealed pool: 24 games, 7 of them hard exploration. Second round: 9 games. Near-siblings: pacman~ms_pacman, frogger~freeway, atlantis2~atlantis | `pools_v83.py` |

---

## 3. Judge disagreements, resolved

| Issue | Positions | Resolution |
|---|---|---|
| Backbone | Feasibility and vision: wildcard. Validity: worth-live | **wildcard's shared-savestate room table is the gate.** worth-live supplies the calibration (exact spike-ins) and the foresight oracle O_m as the selection-channel ceiling and graded live control. representation supplies the joint rule that picks design and thresholds. procedural supplies pairwise intersection-union contrasts, certified common prefixes, and σ² inflation for steered arms |
| Primary yardstick | Validity: λ from each arm's own M1 in natural runs, s1 secondary. Feasibility and vision: s1 primary | **s1 primary.** It removes head starts and archive leakage by construction and shortens runs, so no checkpointing is needed. Validity's objection (a new regime with unvalidated noise) is answered by the I-stage, which measures V2s noise in that regime and re-runs the spike-ins there before thresholds are fixed. A regime ratio (natural V2 against V2s) is reported. Cost: knowledge that speeds the climb to M1 is invisible, which is declared and is what "matched rungs" requires |
| O_m engineering risk | Feasibility: highest risk. Validity and vision: keep it, with fixes | **Kept, re-engineered.** Candidates run read-only in an unbilled, untraced sandbox with no clones; the winner is then **re-executed** for real from its own child RNG stream. There are no buffered writes and no trace leak. O_1 ≡ V2s by construction. Fidelity becomes a caption, not a gate condition (this removes the PASS/NARROW ambiguity). O_4 is called a greedy foresight reference, **not** a proven upper bound |
| What "any knowledge" means | representation: replaying the route trivially passes. worth-live overclaimed "most favourable knowledge" | **Room is asked per channel.** D is the trivial positive control. O_4, K\*map and K\*mac are separate rows. Negatives name the channels tested and state that representation was not tested |
| NO-ROOM rule | worth-live: point R < 1.25. wildcard: upper bound < 2 with an INCONCLUSIVE band | **wildcard's evidence of absence, at 95%**, so that P(NO-ROOM \| true 2×) ≤ 0.05, plus a censoring guard |
| Comparator | v84 and representation: the per-game "fastest of". procedural: pairwise | **Pairwise intersection-union.** A minimum over noisy arms biases log R by about 0.25–0.3 |
| Placebos | worth-live's wrong-game K̃ is not sharpness-matched (validity) | **Per channel, only after ROOM** (G1-P): a greedy-frontier steering for "where", and another game's motor patterns for "how". Each has a frozen interpretation |
| hard-games | Vision ranks it 2nd. Feasibility and validity rank it last (thin data, model-based simulation, regression to the mean, a confounded H control) | **Not v85's gate.** It is the default v86 substrate after NO-ROOM or INCONCLUSIVE, repaired: fresh comparator seeds, calibration by resampling real runs, a demonstration control, H as a negative control. Its stall inventory is prepared at 0 frames in S0 |
| representation's eye | Vision: keep it for NES. Validity: K\*see is not an oracle | **Deferred to the NES rung** (vision §1.3). Its X_on:F_on template (carried start against the same learner from scratch) becomes the transfer contrast |
| procedural's postures | Vision: hyperparameter tuning (AutoML), last. Validity: K\*_proc is not a ceiling | **Not carried.** Postures appear only as tuned-default controls (S, T) |
| Seeds | 9800–9849 or 9850–9899 | **9800–9899**: development below 9850, sealed from 9850 up (grep clean) |
| Units | 6, 10, 22, or chosen by rule | **The 16 room development games with k\* ≥ 4 are eligible.** U ∈ {10, 12, 16} is chosen by the S0 rule, with one top-up allowed before F1 |

---

## 4. Mechanism

### 4.1 V2s, the comparator and the base of every arm
V2s keeps V2's laws and constants (v84 §5): cells, selection weights, frontier 0.25, Q1–Q4, Thompson bandit (discount 0.995, floor 0.2), rollouts of 100 + 50 up to 400 steps, 20k-cell cap, re-key every 250 expansions. It has four changes:
1. **The root is s1** (§4.2). Agent-side S = S(s1); depth is 0 at the root.
2. **Randomness.** Expansion e, candidate j draws only from `default_rng([seed, 85, e, j])`, and V2s uses j = 0. Bandit propensities come from `default_rng([seed, 1])`. V2s therefore equals V2 in law, not in bytes. Every v85 contrast is V2s against V2s-based arms.
3. **Row logging off.** exp_meta and exp_feat are kept. Slots stay in memory for family M's online miner.
4. **New hooks.** `propose(e, j)` returns (cell, process, first slot, child stream) with no side effects. A fifth process **Q5** is active only when a macro library is non-empty; with an empty library the bandit has k = 4 and the run is byte-identical to V2s.

### 4.2 Shared start s1
For each unit and each reference run (start A = G1b seed 9712, start B = 9713):
1. Take the first record with cum ≥ M_top. Replay its `rec_actions` from `base_state` in the after-step context, or else the post-reset context (note 16).
2. s1 is the state on that lineage at the first step with cum ≥ M1. Store the ALE state blob, S(s1), lives, the step index and the lineage actions from s1 to M_top. Because s1 lies on the M_top lineage, D is an exact replay.
3. **Start-identity check** (0 billed frames). Fresh emulator, entry, reset, one NOOP, restore s1, replay the lineage above s1: every rung M2..M_k\* must be reached at its recorded step with its recorded cum.
   - A failing start is dropped.
   - A unit with no valid start is dropped, with no replacement.
4. **Arm run at a start.**
   - `SenseEnv85(game, seed=entry seed)`, with entry seed 9806 for A and 9809 for B.
   - `Body.enter` with `default_rng([entry seed, 0])`, identical for every arm at that start, billed but off the clock.
   - Then reset, one NOOP, restore s1, and the clock starts.
   - T(M) = billed frames since the clock started, including 60 per restore.
   - The trace base is the s1 blob, so replays use the after-step context.

### 4.3 Instruments (G1)
- **D (demonstration, positive control).**
  - The first rollout replays the lineage from s1 to M_top open-loop, with ordinary puts; V2s continues afterwards.
  - Deterministic: 1 run per start.
- **O_m (foresight, m ∈ {2, 4}).** Each expansion:
  1. Draws candidates j = 0..m−1 through `propose(e, j)`; j = 0 is V2s's own draw.
  2. Plays each candidate inside `env.unbilled()`. This saves and restores the ALE state, ledger, trace, anchor and since-buffer, cum and depth. It allows no clone and no archive write. Archive-dependent quantities (ADV, new cells, the extension rule) are computed against a read-only view plus a per-candidate overlay of the keys the candidate would create.
  3. Picks a winner by (rise of archive-best S, then ADV, then number of new cells), compared lexicographically; ties go to the lowest j.
  4. **Re-executes the winner** for real from its own child stream: billed, traced, archived, and the bandit updated.
  - Assertion: the executed trajectory equals the sandbox one (actions, max S, frames, ADV, new cells).
  - Sandbox frames are reported as unbilled `oracle_frames`. O_1 ≡ V2s.
- **K\*map ("where").**
  - The map is (s_bin, coarse code (8,6,4)) along the start's lineage above s1, every 4th step, with consecutive duplicates removed.
  - On each put, a cell gets a map index: the largest i with equal s_bin and a coarse-code Hamming distance of at most 4 of 48 blocks; otherwise −1.
  - Each expansion, with probability 0.5 (drawn from the child stream), it selects uniformly among the cells with the largest map index ≥ 0. Otherwise it takes V2s's own draw, frontier included.
  - Coverage is the share of steered draws that found a match. Caption MAP-WEAK below 30%.
- **K\*mac ("how").**
  - The library is the 32-step windows ending at each sensed rise on the lineage above s1. Exact actions are mapped to canonical slots through this start's entry adapter; an action with no slot becomes NOOP. Windows are run-length encoded, and unique sequences are weighted by count.
  - Q5 draws a macro in proportion to its count, with a 10% uniform mixture, and chains macros to the end of the rollout. There is no index on cell or time.
  - The bandit covers Q1–Q5 under V2's rules.

### 4.4 Carried families (code written only for the family G1 selects)
- **Family W, "where to try"** (v84 route S, live).
  - **Object:** a zero-mean prior Σ_s over the slopes of a logistic critic of ADV on the 44 search-sense features, with a free intercept per game. Recipe: v84 A.5–A.8 with the null-space threshold amended to 1e-2 (note 8). Sources: the G1b expansion logs of the other 40 burned games (leave-one-game-out) in development; all 41 for the sealed test.
  - **Steering (v84 P3):** 32 candidate cells × processes are scored; one is picked with probability ∝ exp(1.5·clip(z, −2, 2)) and mixed 0.5 with V2s's draw. Refit every 100 expansions. Steering is active only after the first refit that sees both classes, so the run is prefix-identical to V2s before that.
  - **Learner from scratch F_S:** the same critic with the isotropic κ_F. κ follows the v84 A.9 rule on the G1b logs of the 4 room calibration games and is frozen before G2.
- **Family M, "how to move".**
  - **Object:** a library of 32 macros. Each is a run-length sequence of canonical slots (sx, sy, extra), 8–64 steps long in multiples of 4. An unavailable slot falls back to its movement class without extra, then to NOOP.
  - **Miner (frozen now):**
    - Candidates are the 8 rows (32 steps) before each sensed rise inside ADV rollouts.
    - Lift is the equal-game mean of [(c_adv+1)/(N_adv+1)] ÷ [(c_base+1)/(N_base+1)]. c_base comes from one random 32-step window per non-ADV rollout (seed 9803).
    - Keep the top 32 by lift with c_adv ≥ 3 in at least 2 games, pairwise RLE edit distance ≥ 3, and at most 4 macros dominated by one game.
  - **Online:** every 250 expansions the miner re-mines the run's own rollouts. Prior pseudo-counts (n₀ = 10 per macro) merge with the run's own counts.
  - **Learner from scratch W:** the same Q5 and miner with an empty initial library. It is byte-identical to V2s until its first mine at 250 expansions (certified prefix).

---

## 5. Yardstick: the shared-start climb (SSC)

**Per run** (unit g, start s, arm a):
- **T(M_k):** billed frames from clock start to the first clone with cum ≥ M_k. Cap 4M from s1; a miss counts the cap for every arm.
- **λ** = mean over k = 2..k\* of log max(5,000, T(M_k)).
- **λ_late** = mean over k = 3..k\* of log max(5,000, T(M_k) − T(M2)), counted from each arm's own M2. It is log(cap) if M2 is missed.
- **Replay confirmation:** M2 and M_top are replay-confirmed in every run, from the s1 blob in the after-step context. A mismatch is an Interruption.

**Aggregation.**
- ℓ_g(a:c) = mean over starts of [mean λ(c) − mean λ(a)]; r_g = exp(ℓ_g), clipped to [1/8, 8]; R = exp(mean over g of log r_g).
- Two-level bootstrap (units, then runs within arm and start), 2,000 resamples. Used: the one-sided 90% lower bound (**LB90**) and 95% upper bound (**UB95**).
- One-sided permutation p, with labels permuted within (unit, start), 10⁴ draws.
- **Pairwise intersection-union:** every named contrast must pass on its own.

**Guards.**
- **NO-ROOM-K** requires at most 25% of K's runs and at most 25% of V2s runs censored at M_top.
- If 40% or more of a gate's runs are censored, the gate is **INCONCLUSIVE**.
- Claim arms get a sensitivity line with misses counted at 2× cap. If it changes a label: caption CENSORING-SENSITIVE, and the label drops one step.

**Instrument line.** V2s on {9806, 9809} against {9807, 9810}. INSTRUMENT-FAIL (the gate is INCONCLUSIVE) if its 99% interval excludes 1.

**Reported:**
- per-rung profiles and the per-unit r_g table;
- the regime ratio: V2 natural, from its own M1 in G1b, against V2s from s1;
- oracle frames and coverage;
- each oracle's share of the demonstration, log R_K / log R_D.

**Inherited from v84:** bars (seed 9706), M_k = B′ + 2^k·u′, and k\* from G1b.

**Head start and early jumps.** Head start is zero by construction: same state, empty archive, same entry. The late clause stops an arm that only jumps the first rung.

---

## 6. Ceiling checks before any threshold is frozen

**Day-0 commit** (this plan): arms, yardstick, labels, kill sentences and the **selection rule** below. F1 only fills in numbers by that rule.

### S0: 0 billed frames, committed before any v85 live frame
1. **s1 rebuild and start identity** (§4.2) on 16 units × 2 starts. **KC-start** if fewer than 10 units have a valid start: "shared savestates cannot be rebuilt reliably; the yardstick does not exist."
2. **Physics floor** per start.
   - F = 4·Δdepth(s1 → M_top) + 60·⌈Δdepth/400⌉, compared with natural C.
   - Pass if the median ratio is ≥ 8 (expected ≥ 20).
   - Otherwise **KC-floor**: "lineage depth leaves no room for 2×."
3. **Preliminary exact spike-ins** on the 32 natural runs, counted from their own M1.
   - Planted effects: D2, D1.5, D1.25, D1, D_late (gaps after own M2 halved), D_jump (T(M2) set to the 5k floor, later gaps unchanged).
   - Arms are built by resampling each unit's centred run vectors; σ² ×1.5 for steered arms; heterogeneity τ ∈ {0, 0.2, 0.4}.
   - Picks the cheapest U ∈ {10, 12, 16} with P(ROOM | D2) ≥ 0.8 at n = 4, τ = 0.2, using thresholds 1.75/1.25.
4. **Foresight prediction.**
   - From exp_meta and the records, p is the share of expansions between M1 and M_top that raised archive-best S.
   - Predicted gain G_m = mean over segments of (1 − (1 − p)^m)/p.
   - Expected, not a gate: 1 ≈ R(O_1) ≤ R(O_2) ≤ R(O_4) ≲ G_4.
5. **Stall inventory** (report only, for v86). Per G1b run: S, rung and billed frames at the last record, and the stall length up to 4M.
6. **Seed grep** over all 11 worktrees.

### I-stage: live, V2s and D only, before any steered arm
- **Runs.** U units drawn with seed 9801 from the valid units. V2s: 6 runs per unit (3 per start). D: 1 run per start.
- **D control.** Requires r_g(D:V2s) ≥ 4 on at least 80% of units, with every D rung replay-confirmed.
  - First failure: one dated fix, then D reruns.
  - Second failure: Interruption; G1 does not run.
- **Instrument line**, as in §5.
- **Recalibration in the real regime.** The spike-ins of S0.3 are planted on V2s runs resampled from the same unit (centred run vectors × √(6/5)).
  - 1,000 simulations per scenario, each through the frozen verdict code with 1,000 bootstrap resamples.
  - Three steered arms share one comparator. σ² ×1.5. τ = 0.2 is primary; 0 and 0.4 are reported.
- **Frozen selection rule.** Search n ∈ {4, 6}, τ_pt ∈ {1.6, 1.75, 2.0}, τ_lb ∈ {1.15, 1.25, 1.35}. Take the smallest n, then the largest τ_pt, then the largest τ_lb that satisfy all of:
  - P(ROOM \| D2) ≥ 0.80;
  - P(ROOM \| D1.25) ≤ 0.10;
  - P(any ROOM among 3 \| all D1) ≤ 0.05;
  - P(ROOM \| D_jump) ≤ 0.05;
  - P(NO-ROOM \| D2) ≤ 0.05;
  - P(NO-ROOM \| D1) ≥ 0.80;
  - mean R̂(D2) ∈ [1.8, 2.2] and mean R̂_late(D_late) ∈ [1.8, 2.2].

  If nothing qualifies: add V2s runs on the remaining valid units once (before F1) and recompute. If still nothing qualifies: **UNDERPOWERED**; G1 does not run and Jérémie decides.
- **Budget rule.** If the G1 projection exceeds 0.7G emulated frames, O_4 runs at n = 4 whatever n is chosen, and its power is reported.
- **F1 commit:** U, n, τ_pt, τ_lb; SHA-256 of the s1 blobs, maps, libraries and code; the calibration JSON.

---

## 7. G1, the first decisive gate (live, development)

**Arms**, each run from both starts:

| Arm | Runs per unit | Role |
|---|---|---|
| V2s | 6 (reused from the I-stage) | comparator |
| O_4 | n | selection channel (greedy foresight) |
| K\*map | n | where to restart |
| K\*mac | n | how to move |
| O_2 | 1 | dose caption |

**Labels for each K ∈ {O_4, K\*map, K\*mac}, against V2s:**
- **ROOM-K:** point R ≥ τ_pt; LB90 ≥ τ_lb; LB90 of R_late > 1; r_g ≥ 1.5 on at least 40% of units.
- **NO-ROOM-K:** UB95 < 2, with the censoring guard.
- **INCONCLUSIVE-K:** anything else.

**G1 verdict.**
- **ROOM** if any ROOM-K. The family is chosen by a frozen rule: the ROOM arm with the largest point log R picks it (O_4 or K\*map → **W**; K\*mac → **M**). A tie goes to M.
  - Sentence: "Same-game oracle knowledge of kind K, used by these rules, makes this billed-rewind climber at least τ_pt× faster at the next rungs from a shared M1 savestate on U development games. This is room, not transfer and not faster learning."
- **NO-ROOM** if all three are NO-ROOM-K. Sentence: "From a savestate shared by all arms at M1 on U development games, the following did not make this billed-rewind climber 2× faster at the next rungs (95% upper bounds below 2):
  - greedy foresight over four of V2's own proposals;
  - the future path's screens steering where it restarts;
  - the demonstration's own motor patterns.

  Only the full demonstration did. The cell-representation channel was not tested."
- **INCONCLUSIVE** otherwise. Sentence: "Room for 2× was neither shown nor excluded in any tested channel." G2 does not run; see §11.

**Captions.**
- O-DOSE if R(O_2) lies outside [1, R(O_4)].
- O-FIDELITY if log R(O_4) < 0.5·log G_4 (path dependence).
- MAP-WEAK if coverage < 30%.
- Per-arm shares log R_K / log R_D.
- The regime ratio.

**G1-P follow-up** (same units and seeds; runs only for an arm that got ROOM; can only add a caption):
- **K\*map ROOM → GF**, a greedy-frontier placebo: with probability 0.5, pick the cell with the largest S (ties: deepest, then uniform). **ROOM-MAP-GENERIC** if LB90 of R(K\*map:GF) ≤ 1.
- **K\*mac ROOM → K̃mac**, another unit's lineage library (derangement drawn with 9801). **ROOM-MAC-GENERIC** if LB90 of R(K\*mac:K̃mac) ≤ 1. The motor patterns then help without game content, and the designer-macro control DM becomes decisive.

---

## 8. G2, the development rehearsal (single shot, only after ROOM)

**Common setup:** the same U units, starts, cap and yardstick; seeds 9826–9831; V2s reused from the I-stage.

| Family M arm | Seeds | Family W arm | Seeds | Role |
|---|---|---|---|---|
| X_on (cross-game library + online mining) | 6 | X_S (cross-game Σ_s) | 6 | carried |
| W (empty library + online mining) | 6 | F_S (isotropic) | 6 | same learner from scratch (F) |
| DM_on (designer library + online mining) | 4 | T (T_clock prior, v84 §6.1) | 4 | tuned default |
| S (best single process, leave-one-game-out, no bandit) | 4 | S | 4 | tuned default |
| P_on (library mined from non-advancing windows, matched on length and slot frequency, seed 9803) | 4 | P_S (labels permuted), U_S (input-PCA, spectrum-matched) | 4 each | placebos |
| K_mine (X's recipe on this game's own G1b rows) | 4 | K_S (Σ from this game's own G1b expansions) | 4 | same-game anchor |
| X4_on (4 source games, 2 draws, seed 9804) | 2 × 2 | X4_S | 2 × 2 | dose |

**DM, the designer library (frozen now), 32 macros:**
- holds of 8 and 32 steps in 8 directions (16);
- 16-step holds with extra in 8 directions (8);
- an extra burst of 4 on, 4 off for 32 steps;
- left-right and up-down alternations of 8;
- two diagonal zigzags;
- NOOP for 16 steps;
- extra held for 32 steps;
- a right-run-then-jump and a left-run-then-jump pattern (8 + 8).

**GO requires all of:**
- R(X:F) ≥ 1.4 with LB90 > 1;
- LB90 > 1 against V2s, against each tuned default and against each placebo;
- LB90 of R_late(X:F) > 1.

Otherwise **KC-dev**: "the carried object does not beat its own learner from scratch, the tuned defaults and the placebos on development starts." The sealed pool stays closed.

**Reported.**
- φ = log R(X:V2s) / log R(oracle of the family:V2s).
- WORTH-LIVE = R(K:F).
- The X4 dose.
- ANOMALY caption if X passes while the LB90 of R(K:F) is ≤ 1. The F2 review must then explain it.

**Ceiling-to-label rule.** The sealed test pursues FASTER-2× only if the family's G1 oracle has a point R ≥ 2.0. Otherwise the ladder tops out at PARTIAL. This is declared at F2.

---

## 9. Sealed test (once)

**F2 sealed freeze.**
- 3-agent adversarial review.
- Commit SHA-256 of X (41 sources), F, controls, placebos, dose draws, code and thresholds.
- Commit a power simulation using the variances measured in G1 and G2.
- Pre-declared rule: if P(PARTIAL | true 1.5×) < 0.5, X and F go from 6 to 8 seeds.

**B0, eligibility.**
- **Pool:** the 17 non-hard sealed games (SEALED minus HARD_EXPLORATION).
- **Runs:** bars on seed 9850 (v84 §7.1); natural V2 runs on 9851–9852 from frame 0, cap 4M.
- **A unit needs:** the G1b rule (a)–(d) on both seeds, k\* ≥ 4, and a valid s1 (S0.1).
- **Count:**
  - more than 10 eligible: draw 10 with 9860;
  - fewer than 6: one second round (bars 9870, eligibility 9871–9872, units pooled);
  - still fewer than 6: **NOT MEASURABLE**.
- **Before any arm:** `sealed_eligibility.json` and its hash are committed.
- **Near-siblings:** the libraries and priors for pacman, frogger and atlantis2 exclude ms_pacman, freeway and atlantis; a "without near-siblings" line is reported.
- **The 7 hard-exploration games stay unopened** (kept for v86).

**B1, arms.** Starts come from the lineages of 9851 and 9852. Seeds 9853–9858; fresh comparators throughout.
- X, F and V2s: 6 runs each.
- Each tuned default and placebo: 4 runs.
- Doses X4 and X12: 2 runs each.

**Labels.** Headers say SEARCH WITH BILLED REWINDS FROM A SHARED SAVESTATE and list the senses. Checked in this order:
1. **NEGATIVE-TRANSFER:** UB95 of R(X:F) < 1.
2. **FASTER-2×** (only if the ceiling rule allows), requires all of:
   - R(X:F) ≥ 2.0 with LB90 ≥ 1.3 and p ≤ 0.025;
   - R(X:c) ≥ 1.5 with LB90 > 1 for c = V2s and each tuned default;
   - LB90 > 1 against each placebo;
   - LB90 of R_late(X:F) > 1;
   - leave-one-game-out minimum of R(X:F) > 1.
3. **FASTER-PARTIAL**, requires all of:
   - R(X:F) ≥ 1.25 with LB90 > 1 and p ≤ 0.025;
   - LB90 > 1 against V2s, each tuned default and each placebo;
   - LB90 of R_late(X:F) > 1.
4. **GENERIC (vocabulary or steering):** X beats F and V2s, but not a placebo.
5. **TUNED-DEFAULTS:** a tuned default beats V2s (LB90 > 1) and X fails its clause against it. Sentence: "earlier games tuned the search; this is not faster learning."
6. **EARLY-RUNGS-ONLY**, **NOT-SIGNIFICANT**, **NULL**.

**Captions.**
- **ACCUMULATES:** Page's L over doses 0/4/12/41 with p ≤ 0.05, and LB90 of R(X41:X4) > 1. Headline figure: a learning-set curve, with F, the tuned defaults and the placebos drawn as flat lines.
- CENSORING-SENSITIVE.
- INSTRUMENT-FAIL makes the result INCONCLUSIVE.

**LP, live probe** (family M only; report only; runs after the B1 verdict is committed).
- On each sealed unit, without savestates, from reset: 20 episodes × 7,000 steps (all lives, sticky 0, 0–30 NOOPs, seed 9861).
- Three policies: an online bandit over X's library with online mining; the same over DM; the unit's best B′ bar mode.
- Reports mean sensed rises in episodes 1–5 against 16–20. It is the project's first live-play number for a carried object, and never a claim.

---

## 10. Gates

| Gate | Pass | Stop or kill |
|---|---|---|
| G-build | tests 1–8 (§17) | **KC-build** after 2 fix rounds |
| S0 | ≥ 10 valid units; floor ratio ≥ 8; U chosen | **KC-start**, **KC-floor** |
| I-stage | D passes; instrument line holds; the rule yields (n, τ) | Interruption (D, after 1 fix); **UNDERPOWERED** |
| F1 | commit | — |
| **G1** | ROOM on at least one channel | **NO-ROOM** (v85 ends; §11); **INCONCLUSIVE** (v85 ends; §11) |
| G1-P | only adds captions | — |
| G2 | GO | **KC-dev**; sealed stays closed |
| F2 | 3-agent review | — |
| B0 | ≥ 6 units | **NOT MEASURABLE** |
| B1 | label ladder (§9) | — |
| LP | report only | — |

**Code freeze.** No code commit while a gate's runs are in flight; mixed code is an Interruption.

---

## 11. If the answer is no

**NO-ROOM.**
- **What it means.** On dense-reward room games, from a shared M1 savestate, this billed-rewind climber's speed is not limited, at these strengths, by three things: which of its proposals it bills, where it restarts, or how it moves. Rewinding and retrying substitute for knowing.
- **Consequences.** v84's two closed routes stay closed, now by a live, calibrated ceiling rather than an offline clock. No carried object of these kinds can show 2× on this substrate.
- **Not established.** Anything about representation knowledge. Anything about live play.

**INCONCLUSIVE.** The bounds are reported, and it leads to the same pivot.

**Default v86** (Jérémie can redirect; otherwise this holds): **rescue from stall savestates**. This takes the hard-games proposal, repaired:
- **Starts:** shared stall states from the S0 inventory, i.e. plateaus where V2 stops climbing for at least 1M frames, plus the development failure games after screening out ladder artefacts (k\* = 0; compressed sports ladders).
- **Score:** P(one rung within 2M) from the shared stall state, compared with **fresh** V2 seeds.
- **Controls:**
  - a demonstration control from a further-reaching run;
  - the H teleport as a negative control;
  - calibration by resampling real runs.
- **Sealed test:** the 7 hard-exploration games.

**Second axis:** a **rewind-price dial** (restores priced at 600 frames, or a restore budget), moving toward live play.

**Third:** a representation-channel row. Either a learned eye, kept for the NES rung where seeing becomes expensive (vision §1.3), or an evaluator-only RAM-cell oracle. The latter needs an explicit decision because of the 22 October rule.

**ROOM, then KC-dev:** "there is room, and this carried object does not capture it." v86 tries another object for the same channel. The sealed pool stays closed.

---

## 12. Pools and seeds

**Pools.**
- **Development units:** the 16 room development games with k\* ≥ 4.
- **Smoke runs:** breakout and pong.
- **Sources for X, P and S:** the G1b logs of all 41 burned games (leave-one-game-out in development).
- **Sealed:** the 17 non-hard sealed games. The 7 hard ones stay sealed for v86. The 9 second-round games are a top-up only.

**Seeds.**

| Seed | Use |
|---|---|
| 9800 | Smoke runs and tests (never a measurement) |
| 9801 | Development unit draw; K̃mac derangement |
| 9802 | S0 and I-stage spike-in simulations |
| 9803 | Placebo libraries and priors; miner baseline windows |
| 9804 | Dose draws (development and sealed) |
| 9805 | Development bootstrap and permutation |
| 9806–9808 / 9809–9811 | V2s at start A / start B (9806 and 9809 are also the entry seeds; D runs nominally on 9806) |
| 9812–9814 / 9815–9817 | G1 steered arms at A / B (n = 4 uses 9812–9813 and 9815–9816); G1-P reuses them |
| 9818 | O_2 (start A on odd draw positions, B on even) |
| 9819–9825 | Reserve |
| 9826–9828 / 9829–9831 | G2 at A / B (4-seed arms use 9826, 9827, 9829, 9830) |
| 9832–9849 | Development reserve |
| 9850 / 9851–9852 | Sealed bars / eligibility |
| 9853–9855 / 9856–9858 | B1 at A / B |
| 9859 / 9860 / 9861 | Sealed bootstrap and p / unit draw / live probe |
| 9870 / 9871–9872 | Second round: bars / eligibility (arms reuse 9853–9858) |
| 9873–9899 | Reserve |

**Reference data.** The G1b runs 9712/9713 are read as data only: s1, lineages, logs. No new run uses a 97xx seed. Avoid 8101–8106, 9101–9256 and 9700–9749. The grep is repeated at F1 and F2.

---

## 13. Compute, RAM, schedule

**Basis.** About 5M emulated frames per hour per worker, as realised in G1b. That gives 15M/h on 3 workers and 20M/h on 4. Assumed average runs from s1: V2s 1.3M; steered arms 1.0–1.2M. O_4 emulates about 5× its billed frames, O_2 about 3×.

| Stage | Runs | Emulated frames | Wall, 3 workers | Wall, 4 workers |
|---|---|---|---|---|
| S0 | replays | ≈ 2M (evaluator) | 2–3 h CPU | — |
| G-build smoke runs | ≈ 10 | ≈ 20M | 1.5 h | — |
| I-stage (U = 12) | 72 V2s + 24 D | ≈ 95M billed | ≈ 6.5 h | 5 h |
| **G1 core** (n = 4) | 48 O_4 + 12 O_2 + 48 + 48 | ≈ 372M (156M billed + 216M oracle) | ≈ 25 h | 19 h |
| G1-P (if ROOM) | ≤ 96 | ≤ 115M | ≤ 8 h | 6 h |
| G2 | 12 × 32 | ≈ 420M | ≈ 28 h | 21 h |
| B0 | bars 17 + 34 | ≈ 12M unbilled + 136M | ≈ 10 h | 7.5 h |
| B1 | ≤ 10 × 30 | ≈ 330M | ≈ 22 h | 17 h |
| LP | ≤ 30 × 20 episodes | ≈ 17M unbilled | 1 h | — |
| **Total** | | **≈ 1.5G (≈ 1.2G billed)** | **≈ 100 h** | **≈ 76 h** |

**Worst-case G1** (U = 16, n = 6, O_4 held at n = 4 by the budget rule): ≈ 0.70G, about 47 h on 3 workers.

**Caps.** Development 1.3G and sealed 0.6G emulated frames.

**Cut ladder.** Applied by budget or calendar, never because of results:
- C1: drop the dose arm from G2;
- C2: G2 units 12 → 10 (first 10 in draw order);
- C3: sealed units capped at 8;
- C4: drop the B1 dose arms.

**RAM.**
- Target ≤ 0.7 GB peak commit per worker (measured 0.23–0.61 GB for 4M runs; v85 runs are shorter, and oracle data stay under 20 MB).
- 3 workers by default (≤ 2.1 GB). A 4th only while the supervisor shows ample margin.
- Supervisor guard at 14 GB free (the 12 GB rule). Launches follow the ragnarok-run skill.
- GPU: 0 h. Disk: about 10 GB (no row logs).

**Resumability.** Runs are deterministic. A killed run reruns from zero, and its JSON is written at the end. If the median O_4 wall time exceeds 90 min at G-build, O runs checkpoint every 500 expansions (archive arrays and clone blobs, bandit, ledger, trace; the RNG needs only the expansion index), with a resume-identity test. This is engineering, never a kill.

**Schedule** (day 0 = 24 October, plan commit).

| Day | Work |
|---|---|
| 0–2 | Worktree, code, G-build, S0 |
| 2 | S0 commit; 2-agent review of the instruments and V2s |
| 2–3 | I-stage (≈ 6 h); calibration; F1 commit |
| 3–5 | **G1; verdict ≈ day 4–5** |
| 5 | G1-P (if ROOM) |
| 5–8 | Family code, sources, G2; verdict ≈ day 8 |
| 8–9 | 3-agent review; F2 |
| 9–11 | B0, then B1; verdict ≈ day 11; LP |
| 12 | French report and handoff |

The NO-ROOM or INCONCLUSIVE path ends around day 5.

---

## 14. Odds (mine; gates are correlated)

| Event | P |
|---|---|
| KC-start / KC-floor | 0.05 / 0.02 |
| D passes (with at most 1 fix) | 0.95 |
| Calibration qualifies (U ≤ 16, n ≤ 6) | 0.85 |
| G1 runs | ≈ 0.75 |
| O_4: ROOM / INCONCLUSIVE / NO | 0.55 / 0.25 / 0.20 |
| K\*map: ROOM / INCONCLUSIVE / NO | 0.35 / 0.30 / 0.35 |
| K\*mac: ROOM / INCONCLUSIVE / NO | 0.15 / 0.25 / 0.60 |
| G1, given it runs: ROOM / INCONCLUSIVE / NO-ROOM | 0.62 / 0.25 / 0.13 |
| Family W / family M, given G1 runs | 0.52 / 0.10 |
| G2 GO: W / M | 0.12 / 0.25 (≈ 0.09 given G1 runs) |
| Sealed measurable | 0.8 |
| PARTIAL or better / 2×, given GO and measurable | 0.4 / 0.12 |
| **Sealed FASTER-PARTIAL or better, overall** | **≈ 2–3%** |
| **Sealed FASTER-2×, overall** | **≈ 0.5–1%** |
| ACCUMULATES | < 0.5% |

**Most likely endings.**
- About 0.40: G1 ROOM, carried mainly by foresight (perhaps also "where"), then KC-dev. "There is room in the climb; a carried critic does not capture it."
- About 0.25: the gate does not run (UNDERPOWERED or KC).
- About 0.18: INCONCLUSIVE.
- About 0.10: NO-ROOM.

Why so low overall: within a run, a fresh critic or miner reacquires its knowledge within hundreds of expansions (v84 notes 14 and 20). So carried knowledge must beat its own learner starting from scratch, not just V2s.

---

## 15. What a positive would and would not establish

**G1 ROOM-K.**
- **Establishes:** same-game knowledge of kind K, given by an oracle, has room on this substrate: it makes the climb about τ_pt× faster from a matched rung, measured on a yardstick whose ceiling was verified in the same regime.
- **Does not establish:** that such knowledge can be learned or carried; that the agent learns faster; live play; that foresight is an upper bound for every critic.

**Sealed FASTER-2× or PARTIAL** (family M; family W reads the same with "critic from a slope prior learned on 41 earlier games").
- **Establishes:** "On N never-opened Atari games, from a savestate shared by all arms at the game's first matched rung, a search with billed rewinds whose motor-pattern process starts from a vocabulary mined from its own play on 41 earlier games, and keeps mining online, climbed the next rungs ≥ 2× (or significantly) faster than the same miner starting empty. It was also faster than the plain climber, designer macros, its best single process and a non-advancing placebo vocabulary. Time is counted in billed frames; the senses are pixels, buttons, score rises and lives."
- **Does not establish:**
  - live play;
  - faster arrival at M1 (not measured);
  - true score;
  - faster seeing;
  - transfer to NES;
  - the hard-exploration games;
  - human-like learning;
  - accumulation, unless ACCUMULATES holds.

**NO-ROOM.**
- **Establishes:** a live, calibrated negative for three named channels at named strengths.
- **Does not establish:** anything about representation knowledge, other substrates, or live play.

---

## 16. Cousins and honest novelty

From memory; a literature check is due before any novelty claim.

**Starting from shared or demonstration states.**
- Go-Explore (Ecoffet et al. 2019; Nature 2021): adaptive downscaled cells, which V2 inherits; domain-knowledge cells; backward robustification.
- Salimans & Chen 2018: learning from one demonstration by starting from its states.
- Restart distributions (Kakade & Langford 2002).

**Learned cells or selection.**
- Policy-based Go-Explore.
- Latent Go-Explore (Gallouédec & Dellandréa 2023).
- Intelligent Go-Explore (Lu, Hu & Clune 2024), which relies on foundation models and is excluded here.
- Learned node selection in branch and bound (He, Daumé & Eisner 2014).
- ExIt (Anthony et al. 2017) and AlphaZero.
- Levin tree search and policy-guided heuristic search (Orseau et al. 2018; Orseau & Lelis 2021).

**Foresight as an oracle verifier.** Best-of-N with an oracle verifier (Brown et al. 2024, "Large Language Monkeys"); hindsight optimisation (Yoon et al. 2008).

**Exploration controllers.**
- Agent57 and NGU (Badia et al. 2020); MEME (Kapturowski et al. 2022). V2's process bandit is a small cousin.
- Temporally extended ε-greedy (Dabney et al. 2021), the analogue of DM and of Q2–Q4.

**Macros, skills and options.**
- McGovern & Sutton 1998; Durugkar et al. 2016.
- Chang et al. 2019, macros reused across Atari games: the closest prior work for family M.
- Action priors (Rosman & Ramamoorthy 2012); action grammars (Lange & Faisal 2019).
- Options (Sutton, Precup & Singh 1999); option-critic; DIAYN; portable options (Konidaris & Barto 2007); eigenoptions.

**Meta-learned exploration.** MAESN, E-RL², Alet et al. 2020, First-Explore, AdA.

**Measurement and human priors.**
- Jumpstart against time-to-threshold (Taylor & Stone 2009).
- Learning sets (Harlow 1949).
- Human priors (Dubey et al. 2018; Tsividis et al. 2017).
- Random against learned features (Burda et al. 2019).

**Not new:** any of the mechanisms.

**New is the test.** A live decomposition of a demonstration's value into channels (greedy foresight, where, how), run from shared savestates as a pre-registered ceiling, with:
- exact time-dilation spike-ins planted in the same regime before thresholds are instantiated;
- per-channel placebos;
- a transfer contrast against the same learner starting from scratch, with pairwise tuned-default and placebo controls;
- a learning-set dose curve on never-opened games;
- no RAM anywhere, no downloaded weights, and only the score-rise and lives senses.

---

## 17. Code to write (in order) and G-build tests

1. **`ragnarok/evaluation/start_v85.py`.** s1 rebuild in both contexts; s1 blob, S, lives, lineage above s1; start-identity check.
2. **`SenseEnv85`.** `start_from(blob, cum)` (entry, then reset, NOOP, restore, clock start, trace base = blob); `unbilled()` context, which saves and restores the ALE state, ledger, trace, anchor/since, cum and depth, and forbids clones.
3. **`ragnarok/agent_v85/explorer2s.py`.** V2s with child streams, start root, `propose(e, j)`, the Q5 hook and logging off.
4. **`ragnarok/evaluation/oracle_v85.py`** (instruments): D, O_m (sandbox plus re-execution), K\*map (map index set on put), K\*mac (lineage library), GF, K̃mac.
5. **`scripts/v85_stage0.py`.** Rebuild, physics floor, natural spike-ins, foresight prediction, stall inventory, seed grep.
6. **`scripts/v85_job.py`.** One run (game, start, arm, seed, cap); JSON with T(M_k) from s1, replay confirmations, oracle frames, coverage, peak commit.
7. **`scripts/v85_calib.py`.** I-stage spike-ins and the frozen selection rule, producing `f1_thresholds.json`.
8. **`scripts/v85_verdict.py`.** SSC, two-level bootstrap, permutation, intersection-union contrasts, labels and captions (G1, G1-P, G2, B1).
9. **After G1 only:** `agent_v85/macros.py` (Q5, miner, DM, P) **or** `agent_v85/critic_s.py` (route S live, κ rule).
10. **Queue and supervisor:** `queue_v85.py`, with jobs registered as workers.

**G-build tests.** Failures of tests 1–7 count toward KC-build.
1. **Isolation.** `agent_v85` has no `evaluation` imports, game names, RAM or private access. Instruments are only in `evaluation`.
2. **Ledger.** It never decreases; each restore is billed 60. Unbilled sandbox frames are never billed and never traced: a sandbox rollout that crosses a rung leaves Trace unchanged.
3. **Start identity** on 2 units.
4. **Determinism.** Two smoke runs on seed 9800 are identical, and a killed-then-rerun run is identical too.
5. **Prefix identity.** O_1, K\*map with an empty map, and Q5 with an empty library are each ≡ V2s. W ≡ V2s for 250 expansions. Family-W arms ≡ V2s until the first two-class refit.
6. **O re-execution identity.**
7. **Verdict and spike-in code.** Exact R recovered when neither floor nor cap binds; censoring; intersection-union logic on synthetic data.
8. **Engineering (never a kill).** Peak commit and fps per arm type; worker count = min(4, ⌊3.6 GB / measured peak⌋); whether O runs need checkpoints.

---

## 18. Résumé pour Jérémie

**Où on en est.** v84 a donné un grimpeur, V2, qui monte vraiment sur 22 jeux sur 35. En revanche, l'« horloge du savoir » hors ligne ne pouvait pas montrer 2×, même avec un savoir presque parfait. Aucun savoir n'a donc encore été essayé en vrai.

**Le premier test de v85 est simple et peu coûteux.** Tous les bras partent de la même sauvegarde, au premier palier de chaque jeu. Ils n'ont donc aucune avance au départ. On mesure si un savoir, même tricheur, fait grimper V2 deux fois plus vite. On teste trois sortes de savoir :
- « prévoir » : savoir lequel de 4 essais va marcher ;
- « où » : voir les écrans du bon chemin ;
- « comment » : connaître les gestes de la démonstration.

Le rejeu complet de la démonstration sert seulement à vérifier l'instrument.

**La règle est vérifiée avant de fixer les seuils.** On plante d'abord, sur de vrais runs de V2, des accélérations connues exactement (×1,25, ×1,5, ×2). On vérifie que la règle les lit juste. Les seuils sortent ensuite d'une règle écrite aujourd'hui. Pour dire « pas de place », il faut une borne haute inférieure à 2, pas seulement un échec.

**Si un savoir a de la place**, on construit un seul objet transportable d'un jeu à l'autre :
- soit un critique « où essayer », appris sur 40 autres jeux ;
- soit un vocabulaire de gestes, appris sur 40 autres jeux.

On le compare au même apprenant parti de zéro, aux réglages par défaut et à des placebos. Puis on le teste une seule fois sur des jeux jamais ouverts.

**S'il n'y a de place nulle part**, c'est un résultat net : avec des retours en arrière payés, réessayer remplace savoir. On passera alors aux jeux où la recherche reste bloquée (Montezuma, Pitfall…) ou à moins de retours en arrière, vers le jeu en direct.

**Coût et chances.**
- Premier verdict vers le jour 4-5, sans GPU, avec moins de 0,7 Go par tâche.
- Chances : de la place quelque part, environ 60 %. Un gain partiel transporté et confirmé, environ 2-3 %. Un 2× confirmé, environ 1 %.

**Rien n'est à décider maintenant.** Après un « non », la suite par défaut est le terrain des jeux bloqués, sauf si tu préfères la piste « moins de retours en arrière ».

---

# Audits

```json
[
 {
  "verdict": "NOT READY TO FREEZE: 1 blocking, 5 major, 6 minor issues. What holds up: the senses rule is respected (no v85 component reads RAM, and claim-bearing arms are built only from the agent's own logs: exp_feat, exp_meta and rows hold no true score). The claim header names the search with billed rewinds. The rule that turns null data into thresholds (F1) is committed before any data. Negatives are kept. Head start is zero by construction between arms. The ground facts I checked are correct: the 16 units with k* ≥ 4 (14 with k* ≥ 5); T1 from 97,572 to 510,492; C from 0.17M to 3.29M; depth from 455 to 5,355; raw σ² 0.497 on 22 games and 0.200 on 16 (recomputed with awk); seed block 9800-9899 clean in all 11 worktrees; pools and near-siblings. One fact is off: 23 of 82 G1b runs took more than 45 min. The main defect: the most likely ROOM source, O_4, picks the luckiest of 4 random rollouts using look-ahead that costs nothing, then replays the same random stream. That is unbilled compute plus luck, not knowledge that a (cell, process) critic could carry. Yet O_4 sends the plan to family W, is described as \"oracle knowledge\", and breaks the header's \"every emulated frame billed\". Several label sentences state point thresholds as if they were guaranteed lower bounds. The NO-ROOM reading goes further than its instruments. Nothing separates a head start in vocabulary from faster learning in family M. Files read: F:\\dev\\ragnarok\\docs\\panels\\v85_design_brief.md, F:\\dev\\ragnarok_v84\\ragnarok\\agent_v84\\explorer2.py, F:\\dev\\ragnarok_v84\\ragnarok\\agent_v84\\rowlog.py, F:\\dev\\ragnarok_v84\\ragnarok\\evaluation\\senses_v84.py, F:\\dev\\ragnarok_v84\\ragnarok\\evaluation\\senses_v83.py, F:\\dev\\ragnarok_v84\\craft_v6_out\\v84\\room_verdict_v84.json, F:\\dev\\ragnarok_v84\\docs\\ARC4_V84_BUILD_NOTES.md, F:\\dev\\ragnarok\\ARC4_VISION.md (it is at the repo root, not in docs\\).",
  "issues": [
   {
    "title": "O_4/O_2 measure free look-ahead plus stream luck, yet they decide ROOM and send the plan to family W",
    "severity": "blocking",
    "problem": "§4.3 O_m plays m candidates without billing, then re-executes the winner from the same child stream (assertion: trajectory equals the sandbox one). In explorer2.py the outcome of a rollout is dominated by random button draws: Q1 draws a uniform slot every step, Q2-Q4 switch at random (line 297). So O_4 mostly picks the luckiest of 4 rollouts and gets them for free. It is not knowledge of where to try, and a (cell, process) critic like W cannot hold it. Yet: (a) the plan's own odds expect ROOM mostly through O_4 (0.55); (b) the family rule 'largest point log R picks the family; O_4 → W' almost guarantees W; (c) the ROOM sentence calls it 'same-game oracle knowledge'; (d) the claim header says 'every emulated frame of the agent billed', which O arms break by about 4× oracle_frames; (e) the French résumé says 'savoir lequel de 4 essais va marcher' and never says the trials are free. The gate's likeliest positive therefore says almost nothing about room for the carried object, and G2 (≈0.42G frames) is then spent on it.",
    "fix": "Make the selection-channel oracle O_4-indep: rank the m proposals by their unbilled sandbox outcome, then execute the winner's (cell, process, first slot) with a fresh, independent child stream, e.g. default_rng([seed,85,e,m]). Drop the trajectory-identity assertion for this arm. Keep the current O_4 only as a report-only caption ('look-ahead plus luck'), never as a router. Route to W only on ROOM of O_4-indep or K*map. Optionally add a W-rule ceiling arm: W's own steering (exp(1.5·clip z), mixed 0.5), fed the O_4-indep sandbox score over 4 candidates, so that the rule linking ceiling to label reads the family's own mechanism (the v84 lesson in note 14). Every O sentence and the header should say: 'unbilled look-ahead of m sandbox rollouts per expansion (a simulator oracle; oracle_frames reported), not knowledge the agent could hold'. Revise the odds and the French résumé to match."
   },
   {
    "title": "K*mac replays part of the demonstration and does not match what family M can carry",
    "severity": "major",
    "problem": "K*mac uses the exact per-step actions of the same lineage, starting from the same s1. Under a deterministic emulator (sticky 0), drawing the first window at the root reproduces the demonstration's opening exactly, and lucky chaining replays more. So 'where' leaks into 'how'. Separately, family M's cross-game miner works on G1b rows, which log the slot only every 4th step (explorer2.py line 300; rowlog.py). Q1/Q3 change slots between rows, so the mined macros are 4-step-held approximations. K*mac's room is therefore not family M's ceiling. Yet the rule 'K*mac ROOM → M' and the §8 ceiling-to-label rule (family oracle point R ≥ 2.0) both read it as if it were.",
    "fix": "At start A, build K*mac from start B's lineage, and the other way round (same game, different path). Use the representation the miner can carry: each logged slot held 4 steps, from the rows. Alternatively, mine family M from the exact rec_actions in the G1b NPZ and say so. Report the exact same-lineage library only as a caption. The ceiling-to-label rule must use the matched-representation oracle."
   },
   {
    "title": "Label sentences overstate their thresholds; labels overlap or are undefined",
    "severity": "major",
    "problem": "(1) The ROOM sentence says 'at least τ_pt× faster', but ROOM-K only requires point R ≥ τ_pt (possibly 1.6) and LB90 ≥ τ_lb (1.15-1.35). (2) The sealed FASTER-2× sentence says '≥ 2×' with LB90 ≥ 1.3. (3) ROOM-K and NO-ROOM-K can both hold, e.g. point 1.65 with LB90 1.3 and UB95 1.95; no precedence is given per channel. (4) The sealed labels EARLY-RUNGS-ONLY, NOT-SIGNIFICANT and NULL have no conditions, and 'beats' in GENERIC is undefined. (5) The brief asks about '~2×', but P(ROOM | true 1.5×) is neither required nor reported, so ROOM may mean about 1.5×. (6) 'FASTER-2×' does not say what is faster.",
    "fix": "Rewrite each sentence with its point and its bound: 'point R ≥ τ_pt, 90% lower bound ≥ τ_lb'. Make the labels mutually exclusive (ROOM-K checked first, or NO-ROOM-K additionally requires point R < τ_pt). Define EARLY-RUNGS-ONLY (X:F passes on λ, fails on λ_late), NOT-SIGNIFICANT (point ≥ 1.25, LB90 ≤ 1 or p > 0.025) and NULL (everything else). Define 'beats' as LB90 > 1. Add P(ROOM|D1.5) to the calibration output and the G1 report. Rename to SEARCH-CLIMB FASTER-2× / -PARTIAL, as v84 did."
   },
   {
    "title": "The NO-ROOM reading goes beyond what its instruments tested",
    "severity": "major",
    "problem": "§11, §15 and the résumé say that the climber's speed is 'not limited by which proposals it bills, where it restarts, or how it moves', that 'rewinding and retrying substitute for knowing', that v84's routes are now closed 'by a live, calibrated ceiling', and that 'no carried object of these kinds can show 2× on this substrate'. The instruments are specific usage rules: greedy best-of-4; p = 0.5 toward the deepest match within Hamming ≤ 4 at equal s_bin; count-weighted macro chaining. They run on 10-16 dense-reward units from M1 to M_top. NO-ROOM-K still fires when MAP-WEAK (coverage < 30%) or O-FIDELITY shows the instrument itself was weak. v84 route Σ (consequence heads steering cells and buttons) is not the instrument tested here.",
    "fix": "Restrict the sentence to 'these three instruments, at these strengths and usage rules, on these units and rungs, geometric mean over units'. MAP-WEAK should turn NO-ROOM-K(map) into INCONCLUSIVE-K. Drop 'rewinding substitutes for knowing' (a hypothesis) and the 'no carried object can' sentence. Say v84 route S is bounded by O_4-indep and K*map, and route Σ only indirectly. Fix the French 'réessayer remplace savoir' the same way."
   },
   {
    "title": "Winner's curse and censoring: the spike-in guarantee is undefined and the cost is optimistic",
    "severity": "major",
    "problem": "k* and M_top come from the same two reference runs that supply s1: both reached M_top by construction, as the minimum over two seeds. Fresh V2s seeds starting from s1 will therefore miss M_top more often than the natural runs suggest. The spike-ins do not say how a censored base run is dilated: setting it to cap/2 overstates power; leaving it at the cap understates it. So the frozen constraint P(NO-ROOM | D2) ≤ 0.05 is not well defined. The cost table assumes a mean of 1.3M from s1 with no censoring (the mean natural C on the 16 units is 1.34M), and §5 never states 'evaluator stop at M_top'. A censored O_4 run emulates about 20M frames (≈4 h).",
    "fix": "In the spike-ins, a censored base run stays censored. For each constraint, use the pessimistic treatment: censored at the cap for P(NO-ROOM|D2) and P(ROOM|D2); cap/2 for the false-ROOM rates. State 'evaluator stop at M_top' in §5. Add the censoring rate measured at the I-stage to the G1 budget projection and to the budget rule. Consider defining k* on one reference seed and taking s1 from the other, or report the regime ratio together with the censoring of V2s from s1."
   },
   {
    "title": "Family M cannot tell a carried vocabulary (a head start) from faster learning",
    "severity": "major",
    "problem": "Family M carries answers (32 pre-mined macros with pseudo-counts n0 = 10). A fixed, better vocabulary helps at every rung. It would pass X:F, the λ_late clause (which only stops early jumps) and the dose curve without the in-run miner learning any faster. This is the vision §4.3 counterexample ('X frozen ... passes without learning anything'). The brief's rule is explicit: a head start is not faster learning. The 'learning-set curve' headline invites the faster-learning reading. Family W differs: its zero-mean slope prior acts only through in-run refits.",
    "fix": "Add an X_frozen arm in G2 and B1: the cross-game library with online mining off. Add a pre-written caption/label HEAD-START-VOCABULARY when the LB90 of R(X_on:X_frozen) ≤ 1. In that case §15 must read 'a carried vocabulary speeds the search climb', not anything about learning. State in §4.4 that family W's prior acts only through in-run refits."
   },
   {
    "title": "The sandbox's list of saved state is incomplete against the V2 code",
    "severity": "minor",
    "problem": "`unbilled()` lists the ALE state, ledger, trace, anchor/since, cum and depth. But a V2 rollout also changes self.best_S, best_S_billed and best_S_exp, which drive the extension rule (explorer2.py lines 340-342). It also sets any_rise (line 314), coarse_count, hist and the row buffers; pick_cell increments A.chosen (line 249); tracker pos/miss change. Test 6 checks only the winner's trajectory, so a losing candidate could still leave traces (unbilled experience) in agent state.",
    "fix": "Snapshot and restore the full explorer, archive, bandit and tracker state, or run candidates on a read-only copy. Add a G-build test: after a sandbox pass with no execution, the state hashes of env, trace, explorer, archive, bandit and tracker equal their values before."
   },
   {
    "title": "s1 is chosen with hindsight, S(s1) is handed over, and the sealed games are called 'never-opened'",
    "severity": "minor",
    "problem": "s1 lies on a lineage the evaluator knows reached M_top. On sealed games it comes from the B0 V2 runs of the same game. The agent's root S = S(s1) is supplied by the evaluator, not sensed in this run. §15 still claims results 'on N never-opened Atari games'. Also, if one rise carries s1 past M2, then T(M2) = 0 for every arm, which dilutes λ.",
    "fix": "Header and senses field: 'a savestate on a prior V2 run's lineage that reached M_top, chosen by the evaluator; root S set to that lineage's sensed rise, identical for all arms'. Replace 'never-opened' with 'not opened before v85's B0 eligibility runs'. Require S(s1) < M2, otherwise drop the start."
   },
   {
    "title": "'Live' is used for the search with rewinds; the live probe is not the vision's live mode",
    "severity": "minor",
    "problem": "The plan writes 'measure live', 'run live for the first time' and 'a live, calibrated negative' (listed next to 'does not establish live play'); the résumé says 'en vrai'. All of these describe a search with billed rewinds. The LP is called 'the project's first live-play number', but it uses sticky 0, while vision §2.5/§4.2 live mode uses p = 0.25. LP also has no empty-library arm, so it says nothing about transfer.",
    "fix": "Define 'online (in the emulator, with billed rewinds)' once and keep 'live' for savestate-free play. Label LP 'savestate-free play at sticky 0, not the vision's live mode', or run it at 0.25. Add the empty-library online-mining arm to LP, or state that LP cannot speak to transfer."
   },
   {
    "title": "Odds, cost and timing figures are inconsistent",
    "severity": "minor",
    "problem": "The §0 table and the résumé ('de la place quelque part, environ 60 %') give ROOM ≈ 0.6, but that is conditional on G1 running (0.75): ≈ 0.47 overall. '0.47G emulated frames' is the cost up to G1, not v85's total (≈1.5G). B1 needs 34-38 runs per unit, not '≤ 10 × 30', so the sealed cap of 0.6G is tight. The ground fact 'a 4M run takes 19-45 min' is wrong: 23 of 82 G1b runs took more than 45 min (maximum ≈150 min; mean 41 min).",
    "fix": "Label the §0 odds 'given G1 runs' and add the overall figure. Say '≈0.47G up to the G1 verdict'. Recount B1 runs and recheck the sealed cap. Correct the wall-time range."
   },
   {
    "title": "The novelty wording overclaims a 'decomposition'",
    "severity": "minor",
    "problem": "'A live decomposition of a demonstration's value into channels (greedy foresight, where, how)': O_4 is not derived from the demonstration. The shares log R_K/log R_D are computed with r_g clipped at 8 and do not add up. Missing cousins: Backplay (Resnick et al. 2018), reverse curriculum generation (Florensa et al. 2017), FiGAR action repetition (Sharma et al. 2017), STRAW macro-actions (Vezhnevets et al. 2016).",
    "fix": "Say 'a per-channel room table with a demonstration positive control'. Call the shares 'ratios, not a decomposition'. Add the cousins and keep 'literature check due'."
   },
   {
    "title": "Smaller protocol gaps",
    "severity": "minor",
    "problem": "(a) 'W' names both family W and family M's empty-library learner. (b) A D 'dated fix' may touch shared start/trace code, which would require rerunning the V2s I-stage runs under the code-freeze rule. (c) An INSTRUMENT-FAIL at the I-stage should stop before G1, not after. (d) Seeds are unallocated for the sealed entry runs and for the 6→8-seed rule. (d') priv slots have no meaning across games in cross-game macros. (e) φ's 'oracle of the family' is ambiguous for W (O_4 or K*map?). (f) ROOM-MAP-GENERIC only adds a caption; GF should then join G2/B1 as a tuned default. (g) W re-opens v84 route S, which was closed by an instrument ceiling only (note 18); this should be stated. (h) The §14 claim that a miner 'reacquires within hundreds of expansions' extrapolates notes 14/20, which concern the critic only.",
    "fix": "Rename the learner (e.g. M0). Rerun V2s if a D fix touches shared code. Stop at an I-stage INSTRUMENT-FAIL. Allocate seeds from the 9873+ reserve. Map priv slots to NOOP. Name the oracle used for φ. Make GF a control when ROOM-MAP-GENERIC holds. Add one sentence on route S's re-test under v85. Mark the miner claim as an extrapolation."
   }
  ]
 },
 {
  "verdict": "The plan is feasible on RAM and GPU, and its G1 arithmetic checks out, but it is not ready to freeze: one blocking contradiction and five major issues, mostly in the run harness and the cost accounting. Checked by reading files and doing arithmetic with awk and PowerShell on the v84 JSON; no Python, emulator or GPU job was run.\n\nWhat holds:\n- **RAM.** G1b peak commit was 212–610 MB per 4M run. v85 runs start from s1, are shorter and keep no row logs, and the sandbox makes no clones. The 0.7 GB per-worker target is credible, and 3–4 workers fit the 2–4 GB budget.\n- **Throughput.** G1b's 82 runs took 55.7 worker-hours (median 36 min per 4M run), about 5.9M frames per hour per worker on completed runs. Kill waste brings that to about 5M/h, so the plan's basis is right.\n- **G1 cost.** G1 core ≈ 372M frames ≈ 25 h on 3 workers is consistent with that basis.\n- **Facts.** The 16 eligible units, the variance of 0.200 on λ, the C range and the 98xx seed grep (no hits in the 11 worktrees) all check out.\n\nWhat breaks:\n1. **Blocking.** Reusing the I-stage V2s runs as the comparator in G1 and G2 conflicts with the inherited rule that any change to the code tree makes a gate an Interruption. G2 always runs after the family code is committed, so it would be an Interruption by construction.\n2. **Supervisor kills.** They come in bursts that take out the whole fleet, and v84's 13 GB launch gate makes them thrash. O_4 runs last 1.4–4 h of wall time, and checkpointing is conditional on a smoke-game test. In practice the gate cannot finish reliably without unconditional checkpointing, and that must be built before the code freeze.\n3. **O_4.** It is dice foresight: the winner is replayed with the same random stream, so it knows which random button sequence succeeds. Its ROOM is close to mechanical (gain ≈ 4 when successes are rare) and says nothing about what a critic over cells and processes (family W) can capture. Yet it uses about 60% of G1's frames and decides the routing to W.\n4. **Accounting.** The table leaves out entry (about 133k frames per run, about 130M in total) and undercounts the B1 and G2 arms. The sealed stage comes to about 0.65–0.7G, above its 0.6G cap.\n5. **Macros.** The G1b row logs record a button only every 4th step. The carried macro library and the oracle (exact actions) would therefore not be the same kind of object.\n6. **G2 economy.** G2 has no cheap same-game pre-gate, although its most likely outcome is KC-dev.\n\n**Realistic timing.** Total ≈ 1.65–1.75G frames, ≈ 115 h on 3 workers. First decisive G1 verdict around day 5–6 rather than 4–5. The NO-ROOM path ends around day 6; the full path around day 13–14.",
  "issues": [
   {
    "title": "Reusing the comparator breaks the mixed-code Interruption rule",
    "severity": "blocking",
    "problem": "The v84 verdict code (scripts/room_verdict_v84.py lines 182-186) declares an Interruption if the runs carry more than one code_tree or a dirty flag. code_tree is the git tree hash of ragnarok/, scripts/ and tests/ (provenance_v81.stamp). v85 reuses the 72 I-stage V2s runs as the comparator in G1 and again in G2. G2 runs after macros.py or critic_s.py has been committed, so it carries a different code_tree and is an Interruption by construction. The same happens to G1 if any fix lands after the I-stage. The plan's sentence 'mixed code is an Interruption' never defines what may be reused.",
    "fix": "Freeze a code-identity rule now. (a) All G1-path code (instruments, checkpointing, v85_calib, v85_verdict) is committed before the I-stage launches. F1 commits only data, outside the hashed trees. (b) A comparator may be reused across a later commit only if a certificate passes: 2 fixed V2s runs re-executed on the new tree reproduce their records and exp_meta byte for byte. The verdict hashes each arm's import closure instead of whole trees. If the certificate fails, V2s is rerun. Budget for that: about 95M frames, about 7 h on 3 workers."
   },
   {
    "title": "Supervisor kill bursts need unconditional checkpointing and a launch gate with back-off",
    "severity": "major",
    "problem": "g1b_supervisor_actions.txt shows 31 TERMINATEs in 12 bursts, each killing all 3 workers. That is about one burst every 37 min between 16:06 and 23:21, and every 6-11 min between 21:41 and 22:15. The latter is a thrash cycle: queue_v84.gate relaunches at 13 GB min(RAM, commit), the young jobs (0.07-0.16 GB) are suspended, then killed within about 8 min. The supervisor terminates the largest worker every 3 s while commit is below 12 GB, so a smaller worker does not protect a run. An O_4 run emulates 5x its billed frames: about 1.4 h for a 1.3M billed run and about 4 h for a 4M censored run. queue_v84 also gives up after 3 passes. The plan only adds checkpoints if the median O_4 wall time exceeds 90 min on the breakout/pong smoke runs, which is the wrong test. Adding checkpoints mid-gate would be a code commit, which the freeze forbids. Separately, the claim that 'the RNG needs only the expansion index' is false: rng_prop = default_rng([seed, 1]) is a sequential stream, and free_slots order affects cell indices.",
    "fix": "Build and test checkpoint/resume at expansion boundaries before the I-stage, for every v85 job or at least O_m and any job projected above 30 min. A checkpoint about every 5-10 min of wall time pickles the explorer, Body, the SenseEnv store, the Trace lineage, the bandit, rng_prop and free_slots. Make propensity draws per-expansion, default_rng([seed, 1, e]). Add a G-build resume-identity test (kill at a random expansion, resume, byte-identical JSON) that counts toward KC-build. In the queue: launch gate at 14.5 GB or more for 3 workers, staggered starts, 20-30 min back-off after any TERMINATE, longest-first dynamic job claiming instead of static k/n sharding, unlimited passes. Add the v85 script names to the WORKER list in ram_supervisor.py."
   },
   {
    "title": "O_4 is dice foresight, so routing its ROOM to family W is unjustified",
    "severity": "major",
    "problem": "Each candidate j carries its own child RNG stream, and the winner is re-executed from that same stream, with an assertion that the trajectory is identical. O_4 therefore knows which random button sequence will succeed. When the per-expansion success rate p is small, G_4 = (1-(1-p)^4)/p is close to 4, so ROOM-O_4 is near-mechanical. No critic over (cell, process) features, which is what family W is, can hold that knowledge. The plan's own most likely ending (0.40: ROOM via foresight, then KC-dev) is this failure. The 48 O_4 runs use about 240M of G1's 372M frames.",
    "fix": "Make the selection-channel arm dice-free, at the same cost. Rank the m candidates in the sandbox, then execute the winner's (cell, process, first slot) with a fresh, unevaluated stream (j = m). Route to W only from dice-free O_4 or K*map. Keep dice O_4 as a 1-run-per-unit caption in place of O_2. Side benefit: the re-execution identity assertion (test 6) goes away, so an overlay mismatch no longer crashes a run."
   },
   {
    "title": "The compute table undercounts entry, the B1 and G2 arms, and V2s run length",
    "severity": "major",
    "problem": "(a) Entry is not counted. Body.enter averages 132,654 frames per run in G1b (82k-192k). Over about 1,000 v85 runs that is about 130M frames, about 9 h on 3 workers. D runs are almost all entry. (b) B1 needs 34 runs per unit for family M (X, F, V2s at 6; DM, S, P_on at 4; X4 and X12 at 2) and 38 for family W, not 30. That gives 340-380 runs, about 410-460M frames. Adding B0 (136M), the bars, entry and LP, the sealed stage reaches about 0.65-0.7G, above its 0.6G cap, so cut C3 (8 units) will likely trigger. (c) G2 for family W is 36 runs per unit, about 475M frames, not 420M. (d) The mean natural C of 1.34M is conditioned on success. With a within-unit sd of log C of 0.74 on the 16 units, an unconditioned V2s run from s1 averages about 1.45M. Realistic total: about 1.65-1.75G frames, about 115 h on 3 workers.",
    "fix": "Recount every stage. Cache Body per (unit, start) after entry: the adapter and tracker hold plain data, so pickle them and hash the cache at F1. This saves the frames and guarantees an identical entry. Either pre-declare sealed U = 8 and run the F2 power simulation with it, or raise the sealed cap to 0.7G now."
   },
   {
    "title": "The carried macro library cannot represent the oracle's patterns",
    "severity": "major",
    "problem": "G1b rows record the slot only every 4th rollout step (explorer2.py lines 300-306). X, mined from those rows, can therefore only express 4-step holds; for Q1 rollouts (new slot every step) and Q3, a row's slot is not what was executed. K*mac uses exact lineage actions, and the online miner of X_on and W uses exact in-memory slots. ROOM-K*mac then routes to an object that cannot express the oracle's patterns, and X:W compares objects of different representational power. There is also a RAM risk: a row's view field is 2,048 B, about 420 MB decompressed per run and about 34 GB across the 82 G1b runs.",
    "fix": "Freeze a single representation for every family-M object: macros quantised to 4-step holds. That means K*mac windows built from every 4th lineage step, the online miner fed every 4th slot, and DM and P_on as already specified. Report K*mac's exact-versus-quantised result as a caption. The miner, P_on and K_mine must load only the slot, it, t and rise_lag fields from each chunk, never view."
   },
   {
    "title": "No sequential pre-gate before G2's expensive run set",
    "severity": "major",
    "problem": "The plan's most likely path is ROOM, then KC-dev (P about 0.40). G2 is 384-432 runs, about 0.42-0.48G frames, about 28-32 h, plus about 3 days of family code and source preparation. The same-game anchor (K_mine or K_S) against its learner from scratch F caps what a carried object of that family can show live, yet WORTH-LIVE is report-only.",
    "fix": "Declare now a G2a stage: K and F only, 4 + 6 seeds x U, about 120 runs, about 0.13G, about 9 h. If the LB90 of R(K:F) is at most 1, label KC-worth-live: X is not built or run, and the sealed pool stays closed. Otherwise run G2b with the remaining arms. This roughly halves the expected G2 cost and ends the modal path about 2 days earlier."
   },
   {
    "title": "The schedule is optimistic by about 1-2 days",
    "severity": "minor",
    "problem": "Day 0 is 24 October and already partly spent. Days 0-2 must now hold the O_m sandbox overlay, checkpoint/resume and vectorised statistics. The I-stage is about 8-10 h rather than 6.5 h once entry, the censoring tail (about 1.45M mean) and suspends are included; G1b walls had tails of 95-150 min. G1 is 25 h nominal but about 30-40 h with O_4's wall time and evening kill windows.",
    "fix": "Quote ranges instead of single dates: first G1 verdict about day 5-6, NO-ROOM path ends about day 6, full path about day 13-14. Queue O_4 jobs first (longest first) and preferably in Tavern-quiet hours; G1b had no kills after 23:21."
   },
   {
    "title": "Calibration CPU cost and the simulated comparator",
    "severity": "minor",
    "problem": "The calibration is 1,000 simulations x 1,000 bootstrap resamples x about 30 configurations (5 scenarios x n in {4, 6} x τ in {0, 0.2, 0.4}) x 3 arms, about 9e7 resample statistics. A per-resample Python loop at about 0.2 ms each takes about 5 h on one core, doubled after the top-up. Separately, if the simulated steered arms are drawn from the same 6 V2s runs that form the comparator, without resampling the comparator, the contrast loses its σ²/6 term. The choice of n and the thresholds is then anti-conservative.",
    "fix": "Vectorise the bootstrap with index matrices. Store the per-simulation statistics once and evaluate the 2x3x3 grid of (n, τ_pt, τ_lb) on them. Leave the permutation p out of the calibration, since G1 labels do not use it. Resample the comparator in every simulation. Add a G-build runtime test: under 30 min."
   },
   {
    "title": "Sandbox implementation pitfalls",
    "severity": "minor",
    "problem": "The sandbox must neutralise several side effects in the existing code. SenseEnv.step calls trace.on_rise, and _check() raises Spent at the ledger cap even inside the sandbox. pick_cell mutates A.chosen, body.track mutates tracker state, and _put can call _evict at 20k cells, which the overlay would not mirror. Deep-copying Trace.lineage (one entry per clone ever made, about 1e5) for each candidate would cost seconds per expansion. A naive V2s root also starts with best_S = 0, so the first step at S(s1) counts as a new best.",
    "fix": "Use flags rather than copies: trace and cap checks off in the sandbox, frames and loads counters saved and restored. Skip tracking in the sandbox; keys do not depend on it. The overlay mirrors eviction, or, with dice-free O_4, mismatches are logged instead of asserted. Initialise best_S = S(s1), best_S_billed and any_rise at clock start. Set the ledger cap to entry plus 4M."
   },
   {
    "title": "Family W steering cost is missing from the table",
    "severity": "minor",
    "problem": "P3 steering scores 32 cells x 4 processes = 128 _features calls per expansion. Each call runs np.nonzero over 20,001 slots and a double argsort over depth (explorer2.py lines 210-235). That is about 0.1 s per expansion, or +30-60% wall time on long or censored runs, none of it in the compute table.",
    "fix": "Vectorise: compute ranks, occupancy and context once per expansion and broadcast the process one-hot. Measure the result in G-build test 8."
   },
   {
    "title": "KC-build kills all of v85 if the riskiest instrument fails",
    "severity": "minor",
    "problem": "Tests 5-6 (O_m prefix identity and re-execution identity) exercise the hardest code, the overlay. Under §10, two failed fix rounds give KC-build for the whole of v85, even though D, K*map and K*mac do not need the sandbox.",
    "fix": "Pre-declare that a KC-build confined to O_m drops only the selection-channel row. G1 then runs D, K*map and K*mac, and any negative names the selection channel as untested."
   },
   {
    "title": "The inherited M_top may censor too many V2s runs for NO-ROOM to be reachable",
    "severity": "minor",
    "problem": "k* comes from G1b: the highest rung both natural seeds reached, which selects on success. With a regime ratio near 1, a lognormal model (within-unit sd 0.74) puts V2s censoring at 4M from s1 near 7% on average, but about 23% on asterix and 17% on pooyan. If V2s from s1 is about 2x slower than natural, censoring reaches 20-30%, and NO-ROOM-K, which requires at most 25% of V2s runs censored, becomes unreachable. G1's 25-40 h would be spent for ROOM-or-INCONCLUSIVE only.",
    "fix": "Fix an I-stage rule now. For a unit where at least 2 of 6 V2s runs miss M_top, lower M_top to the highest rung reached by at least 5 of 6 V2s runs, and drop the unit if that is below M3. M_top is set at F1, before any steered arm runs. The lineage above s1 is truncated to match."
   },
   {
    "title": "K*mac adapter round-trip and the D share are artefacts",
    "severity": "minor",
    "problem": "Lineage ALE actions are mapped to slots through the v85 entry adapter (seeds 9806 and 9809), which differs from the G1b adapter (seeds 9712 and 9713). Actions that map many-to-one or to no slot become NOOP, so K*mac is silently weakened. Separately, D's r_g is always clipped at 8 (its T(M_k) sit at the 5k floor), so D is a plumbing check only, and the share log R_K / log R_D is just log R_K / log 8.",
    "fix": "Let the instrument play exact ALE action indices on the evaluator side, or report round-trip fidelity per unit. Call D a plumbing control, and report the shares against D's unclipped R, or drop them."
   }
  ]
 },
 {
  "verdict": "NOT FREEZABLE AS WRITTEN: 1 blocking, 6 major and 2 minor issues. The yardstick is sound. The shared savestate s1 removes any head start in state, archive and entry. Its ceiling is checked live before the thresholds freeze: the physics floor, the D control (a 4x ratio is possible), and spike-ins planted in the same regime and run through the frozen verdict code. The selection rule is frozen on day 0, NO-ROOM needs evidence of absence, and contrasts are pairwise. The ground facts check out against the files: 16 units with k* >= 4 (14 with k* >= 5), raw sigma^2 = 0.200 recomputed from the lambda pairs, C from 0.17M to 3.29M, M3 depth from 455 to 5,355 steps, and no project .py file in the 11 worktrees contains a seed from 9800-9899. The core problem is the first decisive gate. It can barely fail, and passing it does not mean what the plan says. From the G1b records, about 0.2-3% of expansions between M1 and M_top raise the archive-best S. So G_4 is about 3.6-4.0 and ROOM via O_4 is nearly certain. NO-ROOM needs O_4 to fail as well, so it is practically unreachable. The rule that picks the family then sends G2 to W, a critic that cannot see the action-stream luck O_4 exploits. The same holds for the future-path screens K*map uses. The other weak points are below: calibration (censoring planted optimistically, a degenerate M2 rung), forking paths (family code written after G1), wording (ROOM, NO-ROOM, 'head start zero'), sandbox state, and kill-safe runs. Fix them before day 0. Method: I read files and used grep and awk only. I ran no Python, emulator or GPU job and changed no project file. I wrote one scratch input file of rung constants for awk: C:\\Users\\jeremie\\AppData\\Local\\Temp\\claude\\F--dev-ragnarok\\826ba484-cdf9-442a-89b7-2049c18871fc\\scratchpad\\rungs.txt. Note: ARC4_VISION.md is at F:\\dev\\ragnarok\\ARC4_VISION.md, not under docs\\.",
  "issues": [
   {
    "title": "G1 is nearly certain to say ROOM, through a foresight row (O_4) that licenses nothing a carried object can hold",
    "severity": "blocking",
    "problem": "O_4 picks the best of 4 complete rollouts. A candidate fixes the cell, the process, the first slot and the random action stream. The winner is chosen with unbilled look-ahead on archive-best S, which is the yardstick's own quantity. In the G1b records, about 0.2-3% of expansions between M1 and M_top raise the archive-best S, counting bursts of records more than 2k frames apart. Examples: kung_fu 1%, qbert 0.5%, zaxxon 0.2-0.8%, chopper 0.9%. Counting every record gives an upper bound of 0.3-16%. So S0.4 will predict G_4 = (1-(1-p)^4)/p of about 3.6-4.0 on nearly every unit, and the plan itself expects R(O_4) <= G_4. ROOM-O_4 is therefore close to foregone; the plan's 0.55 odds contradict its own prediction. G1 fails only if O_4 fails, and NO-ROOM needs all three rows, so the 'no' branch the brief asks about is practically unreachable. Passing via O_4 also does not show room for carryable knowledge. O_4 mostly exploits which random action stream gets lucky, and no critic over the 44 search-sense features (family W's object) can see that before the rollout. The family rule (largest point log R picks the family) then sends G2 to W almost regardless of K*mac's result. This is the plan's most likely ending (about 0.40: ROOM, then KC-dev). O_4 is also about 270M of G1's 372M emulated frames, and at equal emulated frames it is slower than V2s.",
    "fix": "Before day 0, make O_4 a reference row. Report it, or run it on fewer units or seeds, but exclude it from the ROOM/NO-ROOM verdict, from the family rule and from the ceiling-to-label rule. Commit S0.4's G_4 as its predicted value. If a ceiling for the selection channel is wanted, use one a critic could reach. Option 1: an expected-value oracle that ranks m (cell, process, first slot) candidates by success estimated from r unbilled rollouts on independent streams, then executes the winner on a fresh stream outside those r. Option 2: run the same-game critic K_S live (see the next issue). Every O row sentence must say 'with 3 unbilled look-ahead rollouts per expansion; at equal emulated frames it is slower', and R must also be reported on total emulated frames."
   },
   {
    "title": "Channel-to-family mapping is unsound, and no same-game version of the carried object is tested before G2",
    "severity": "major",
    "problem": "K*map's power comes from the future lineage's screens. W's critic sees no screen content: S-rank, depth, yield, rarity, process, clock. So ROOM-K*map does not show that W can capture anything. For family M the mechanism matches (Q5 plus a library), but K*mac is game-specific content while X is a cross-game vocabulary. The same-game objects (K_S, K_mine) and their learners from scratch (F_S, the empty-library learner) first run in G2. R(K:F) is only reported, as WORTH-LIVE or the ANOMALY caption, and never gates. v84 note 20 shows the learner from scratch catching up within a run. The plan can therefore spend G2's roughly 0.42-0.48G frames on an object whose same-game ceiling was never checked live.",
    "fix": "Write both families' code before G1. Add an object-room gate before X, the controls and the placebos. Either run K_S/F_S and K_mine/F_M as G1 rows, or stage G2: G2a runs K and F of the chosen family against V2s, with thresholds from the frozen I-stage rule, and G2b runs only if R(K:F) and R(K:V2s) meet ROOM. A family is licensed only by a row whose information that family can hold: K_S or the expected-value oracle for W; K*mac or K_mine for M. K*map alone licenses no current family. It points to the deferred representation channel."
   },
   {
    "title": "ROOM, NO-ROOM and 'head start zero' sentences claim more than the rules test",
    "severity": "major",
    "problem": "(a) The ROOM sentence says 'at least tau_pt x faster'. The rule only requires point R >= tau_pt, which can be 1.6, and LB90 >= tau_lb (1.15-1.35). P(ROOM | true 1.5x) is not constrained, so ROOM does not mean 'about 2x' as the summary says. (b) NO-ROOM excludes 2x and nothing smaller. With the noise implied by G1b (per-run sigma^2 about 0.2, x1.5 for steered arms, tau = 0.2, U = 12, n = 4), the SE of log R is about 0.11. A true 1.5x then gives UB95 < 2 about 90% of the time. Yet section 11 reads NO-ROOM as 'rewinding and retrying substitute for knowing' and changes substrate, although room for FASTER-PARTIAL is not excluded. (c) 'Head start is zero by construction' holds only for state, archive and entry. In G2 and B1, a carried prior is exactly an initial competence advantage. ARC4_VISION's counterexample applies: a frozen X that learns nothing on the new game passes a time-to-threshold rule. The R_late clause does not separate a uniformly better default from faster adaptation.",
    "fix": "(a) Word ROOM as 'point estimate >= tau_pt, 90% lower bound >= tau_lb'. Report P(ROOM | D1.5) from the calibration, and either constrain it or rename the label ROOM(>tau_lb). (b) Word NO-ROOM as '2x excluded at 95%; room up to UB95 not excluded'. Report UB95 per channel, and state in section 11 that room of PARTIAL size remains possible. (c) Say 'no head start in state or archive'. Add an X_frozen arm to G2 and B1: for family M, the library fixed with no online mining; for family W, the prior with no in-run refits beyond the intercept. Report the interaction contrast [lambda(F_M) - lambda(V2s)] - [lambda(X_on) - lambda(X_frozen)]. Use 'adapts faster' wording only if X_on beats X_frozen; otherwise add the caption ADVANTAGE-FROM-PRIOR."
   },
   {
    "title": "Calibration is optimistic about censoring and about a degenerate first rung, so NO-ROOM is less safe than stated",
    "severity": "major",
    "problem": "(a) The spike-ins halve T. For a V2s run censored at 4M, halving gives 2M, as if the true time were exactly the cap. k* is the minimum over two seeds, so M_top sits at V2's 4M frontier: alien's best was 3,570 against M4 = 3,528; bank_heist 360 against 335; phoenix 10,710 against 10,064; zaxxon reached M_top at 3.56M. Fresh V2s seeds will miss M_top fairly often. Planting from censored runs overstates the measured R of a true 2x arm and understates P(NO-ROOM | D2). Counting misses at the cap for every arm pulls the real R toward 1, which makes UB95 < 2 easier. The 2x-cap sensitivity line applies to claim arms only, not to NO-ROOM. (b) The first rung from s1 is often floor-bound or degenerate. In G1b, T(M2) - T(M1) is under 5,000 frames on 10 of 32 runs (bank_heist 604, yars_revenge 320 and 2,712, phoenix 2,092-3,980, amidar 2,040-4,928, zaxxon 4,516, up_n_down 4,852). qbert 9712's first M1 record already has cum 775 >= M2 = 562.5. That term is the same constant for every arm and dilutes R: with k* = 4, a true uniform 2x shows as 1.59 on that unit. The start-identity check does not catch S(s1) >= M2. (c) The x1.5 variance and tau = 0.2 heterogeneity are assumed and never checked against the steered arms actually run.",
    "fix": "(a) In S0.3 and in the I-stage, a planted arm drawn from a censored comparator run stays censored at the cap. That is pessimistic for NO-ROOM; report an imputed-tail variant beside it. NO-ROOM-K must also hold when V2s misses are counted at 2x the cap. (b) Before S0, define each start's rungs as the M_k above S(s1), and drop or relabel any start with S(s1) >= M2. Tie the floor to the reference lineage's physics floor F_k instead of a flat 5,000, or drop from lambda the rungs whose natural median gap is under about 20k frames. (c) Pre-register a check after G1: if an arm's realised variance ratio to V2s exceeds 1.5, or the heterogeneity exceeds 0.4, rerun the frozen calibration with the realised values. If that label's operating characteristics then fail, downgrade it to INCONCLUSIVE; never upgrade. Centre the I-stage residuals per (unit, start) with the sqrt(3/2) correction, not per unit with sqrt(6/5)."
   },
   {
    "title": "Forking paths: family code, kappa and the family's oracle are settled after G1, on the units G2 reuses",
    "severity": "major",
    "problem": "Only the family that G1 selects gets code, and it is written after G1 results on the same U development units that G2 then reuses. v84 needed 20 build notes of implementation latitude. Kappa is specified for F_S only, not for X_S, P_S or U_S. 'The family's G1 oracle' in the ceiling-to-label rule and in phi is ambiguous for W (O_4 or K*map); if it means O_4, the rule is toothless. The ceiling rule also uses a point estimate (R >= 2.0) and compares against V2s, not F. G1-P's GENERIC captions fire when LB90 <= 1, which is the likely outcome at n = 4 even when the oracle beats its placebo by 1.3x, so a 'generic' label is predictable. Yet G1-P is caption-only and cannot veto the family choice.",
    "fix": "Before G1, write, test and hash both families' code: macros.py, critic_s.py, the miners, DM, P, and the kappa rule for every W arm. Put the hashes in the F1 commit. Name each family's oracle now. Use a lower bound in the ceiling rule (for example, pursue FASTER-2x only if the oracle's LB90 >= 1.5). Make GENERIC an equivalence criterion (UB of R(K:placebo) < 1.25), with NOT-SEPARATED otherwise, and let a GENERIC result remove that row from the family rule."
   },
   {
    "title": "O_m sandbox state is incomplete, so the stated identities can fail by construction",
    "severity": "major",
    "problem": "The rollout and the following draws depend on state the plan neither saves nor restores: Explorer2.best_S, best_S_billed and best_S_exp (they drive the +50 extension rule and two features); any_rise, hist and coarse_count; A.chosen, which pick_cell increments, so propose() cannot reuse pick_cell unchanged; MeTracker.size, which update() changes and restore does not reset, plus pos and miss; the trace's on_rise lists; AtariEnv.frames and loads; and evictions at the 20k cell cap (G1b runs reached it, with about 15k evictions), which the overlay ignores. With that leak, O_1 run through the sandbox is not identical to V2s. Test 5 as stated can then fail, or pass only if O_1 skips the sandbox, in which case it tests nothing. K*map's 0.5 coin is drawn from the child stream, so K*map with an empty map consumes an extra draw and cannot equal V2s byte for byte. The plan does not say what happens when the re-execution assertion fails during a G1 run.",
    "fix": "Snapshot and restore the full explorer and tracker state listed above in the sandbox, with the archive read-only plus an overlay that simulates eviction. Run O_1 through the sandbox path in G-build test 5. Draw steering coins from a separate stream, default_rng([seed, 85, e, 'steer']). Declare that a failed assertion aborts the run as an Interruption, rerun after one dated fix, and report the count."
   },
   {
    "title": "O runs will not survive routine supervisor kills; compute is undercounted",
    "severity": "major",
    "problem": "G1b logged 31 TERMINATE and 186 SUSPEND actions over about 82 runs. O_4 emulates about 5x its billed frames, so a 1-4M billed run takes roughly 1-4 h of wall time, and a censored one far longer. Runs restart from zero, so expected wall time grows like (e^{kL}-1)/k: about 7 h for a 3 h run at 0.5 kills per hour, which threatens the day 4-5 verdict. Checkpointing triggers only if the median O_4 wall time in G-build smoke runs exceeds 90 min, and smoke runs are not censored runs. Body.enter (100-500k frames per run, repeated for every arm) is missing from the compute table. The W-family G2 needs 36 runs per unit, not 32, and B1 needs 34-38 per unit, not 30.",
    "fix": "Make checkpointing mandatory (every 500 expansions, with a resume-identity test) for all O runs and for any run projected over 60 min. Cache the entry result per (unit, start), or bill it in the table. Recompute G2 and B1 for the larger family and check the totals against the 1.3G and 0.6G caps before day 0."
   },
   {
    "title": "Family W's tuned defaults miss the obvious fixed heuristic (greedy frontier)",
    "severity": "minor",
    "problem": "G1-P names greedy-frontier steering (GF) as the generic explanation for 'where' room, but G2's W arms have only T_clock and S as tuned defaults. An X_S whose cross-game prior amounts to 'prefer high-S cells' could beat F_S (through an early jump) and V2s, and no fixed rule in G2 tests that.",
    "fix": "Add GF as a tuned default for W in G2 and B1, with the same LB90 > 1 clause."
   },
   {
    "title": "Seeds, names and wording slips",
    "severity": "minor",
    "problem": "The F2 rule moving X and F from 6 to 8 seeds has no seeds: 9853-9858 hold 3 per start and 9859-9861 are taken. Seed 9801 serves both the unit draw and the K~mac derangement. 'W' names both a family and family M's learner from scratch. The summary's 'G1 ROOM about 0.6' is conditional on G1 running (0.75). The 'never-opened' sealed games are opened first by B0's V2 eligibility runs.",
    "fix": "Allocate 9873-9876 now for the extra B1 seeds. Use default_rng([9801, purpose]) for each draw. Rename family M's learner from scratch, for example F_M. Mark the odds as conditional. Say 'not opened before B0 eligibility'."
   }
  ]
 }
]
```
