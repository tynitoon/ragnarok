# v85 judges (workflow wf_8db018d6-25c, unedited)

---

**Ranking.** wildcard > hard-games > worth-live > representation > procedural-live

**Recommended combination.** Use wildcard as the backbone, applied where hard-games says knowledge should matter, with worth-live's calibration.

(1) Stage 0, 0 frames. worth-live's exact spike-ins (D2, D_late, D_head, D1) on the G1b records, plus a frozen design and threshold rule taken from representation's C0. Pairwise intersection-union contrasts (procedural-live) replace the 'fastest of' comparator.

(2) First decisive gate, about 3 days. All arms start from SHARED SAVESTATES with an empty archive (wildcard), at two kinds of start:
- s1 at M1 on about 8 room games;
- s_stall, the plateau state that ends a G1b or failure-game run (wildcard option (a)), on genuine search-failure games. The ladder-artifact games are screened out first: the sports games with compressed ladders (bowling, ice_hockey, fishing_derby) and the k*=0 cases (assault, krull).
Arms:
- V2;
- D, demonstration replay, the planted positive control;
- K*map ('where');
- K*mac ('how');
- O_4, the foresight dose, as the steering-channel ceiling (worth-live);
- a teleport head-start control H (hard-games).
Scoring: the matched-rung lambda from the shared start, plus hard-games' MRR with exposure from each arm's own arrival on the stall panel. NO-ROOM requires a 90% upper bound below 2.

(3) Carried object, only if ROOM-MAC on stall states: the hindsight-mined motor vocabulary Q5. It is tested against designer macros, within-game mining, the best single process and a non-advancing placebo. Where it fits, X_on vs F_on serves as the 'learns faster, not head start' contrast.

(4) Sealed test, once: the failure stratum, admitting hard-exploration games first, scored with MRR and the prior-alone line. The headline is the learning-set dose curve over 0/4/12/41 source games, with the controls drawn as flat lines.

(5) If NO everywhere, v86 takes worth-live's rewind-price dial and hard-games' rewind-limited search toward live play, plus a report-only live probe now. representation's carried eye is kept for the NES rung, where vision §1.3 says seeing becomes expensive.

Every claim states SEARCH WITH BILLED REWINDS from a shared savestate.

### wildcard — 7.5

- Strengths: Best fit with the owner vision. (1) The headline it plans, a learning-set curve over 0/4/12/41 source games with designer macros, within-game mining and a placebo drawn as flat lines, is the accumulation figure of ARC4_VISION claim (4). It is also the most direct picture of 'each new game learned faster the more games came before'. (2) Its carried object, a motor vocabulary, is one of the expensive shared things named in vision §1.1 (reach/avoid/hit/jump) and in the §2.6 carry list (hold-length priors). It is also what NES will need (hold-length jumps, §3.2), and V2 genuinely cannot reacquire it within a run, because V2 has no way to invent processes. (3) Starting every arm from a shared savestate s1 with an empty archive removes head start in state, archive and perception by construction. This is a cleaner answer to 'a head start is not faster learning' than any statistical correction. (4) The where/how split of a demonstration's value (D vs K*map vs K*mac), done live from a common state, is a new analytic device. Its cousins (Salimans & Chen, backward Go-Explore, mu-resets) use demonstration states, not this decomposition. (5) It has the cheapest decisive gate: about 150M frames, about 10 h, verdict by day 3. (6) NO-ROOM requires the 90% upper bound to be below 2, so a negative is a real negative and not just a failure to pass. (7) A report-only live probe without savestates gives the project its first live-play number, which is vision claim (2). Facts checked: 16 room dev games have k* >= 4, and depth at M3 ranges from 429 to 9670 steps.
- Flaws: (1) The decisive gate sits on room games where V2 already climbs. By the proposer's own reading the most likely ending is NO-ROOM or INCONCLUSIVE, so the vision-aligned parts (G2, the learning-set curve) are reached with P about 0.3 x 0.25. (2) The 'how' oracle is weak by construction: macros without an index, mined from the same segment, may add little where random held processes already find the next rise. A NO-ROOM-MAC then speaks about this miner more than about motor knowledge. (3) Starting from one state with an empty archive changes V2's regime (no frontier diversity at M1), so the result concerns a modified V2. The cumulative S at s1 must also be seeded from run A, which is implementation detail left unstated. (4) X is likely to tie DM (cf. temporally extended epsilon-greedy), giving GENERIC-VOCABULARY. (5) The dose curve comes from unsteered V2 play, not a sequential lineage, which is declared. P(ACCUMULATES) is below 1%. (6) Macros mined from rows logged every 4th step have reduced resolution. (7) INCONCLUSIVE at 0.25 is high for the 'decisive' gate.
- Graft: Shared-savestate starts (zero head start by construction), the live where/how decomposition of a demonstration as an advance ceiling, NO-ROOM defined by an upper bound below 2, the learning-set dose curve with flat-line controls as the headline figure, and the report-only live probe without savestates.

### hard-games — 7

- Strengths: Puts knowledge where vision §1.1 says it pays: where a fresh learner is stuck, as humans are in Montezuma. It makes the 7 unopened hard-exploration sealed games the one-shot test, which is the most presentable possible positive ('carried knowledge gets the agent unstuck on never-opened Montezuma/Pitfall/Venture'). The instrument is genuinely novel for this setting. (1) A Mantel-Haenszel rung-transition rate ratio with exposure counted from each arm's own arrival, censored time counted as exposure and no imputed cap, which improves on v84's cap-imputation censoring. (2) A prior-alone line, with search off, that makes Taylor & Stone's jumpstart vs time-to-threshold separation testable. (3) A live teleport head-start negative control H, which certifies the yardstick cannot be gamed. Also good: the K_where/K_how diagnostics, and the logic of the KC-ceiling consequence, which leads to rewind-limited search ('continues' rather than savestates), a step toward the vision's live-play ceiling. Panel facts checked: the 10 failure games failed (b); demon_attack, road_runner and ms_pacman failed (d) only; tennis failed (a) only; T(M1) on room games is 0.10 to 0.51M.
- Flaws: (1) The failure panel is heterogeneous, and much of it is not hard exploration. assault and krull have k*=0. star_gunner's seeds split 37,800 vs 800, which is seed luck. bowling (bests 66/72), ice_hockey (21/24) and fishing_derby (46/62) are sports games with compressed ladders. centipede has random-play bars near skilled scores. 'Failure' partly reflects ladder geometry (M_k = B' + 2^k u'), not a stuck search. (2) K_trail is capped by the 12M teacher's reach: where the teacher is also stuck, the 'oracle' carries nothing. (3) Seeds count as events, with only 1 to 2 strata per game and 2 to 3 seeds, so per-game MH ratios rest on 0 to 3 events. The 0.5 correction and the [1/8, 8] clip will dominate, and G-sim's exponential fits on such sparse records are fragile. NOT-MEASURABLE is a real risk. (4) Under the max(0, r) sense, montezuma and pitfall may fall into the zero stratum, leaving too few sealed hard units. (5) Matching within 8 of 88 blocks is ad hoc. (6) The Phase B objects (W critic, Q5 motifs) are generic. (7) The cost is high: 1.7G frames over 10 to 12 days. (8) It skips the brief's room-game question and keeps only a report-only contrast there.
- Graft: MRR exposure counted from each arm's own arrival, with censoring as exposure; the prior-alone line, which drops strata the object reaches without search; the live teleport head-start control H; and the failure-stratum sealed design that admits hard-exploration games first. Ideally applied from shared stall savestates.

### worth-live — 6.5

- Strengths: The most direct and best-calibrated answer to the brief's first question. (1) The Stage-0 spike-ins (D2, D1.5, D_late, D_head, D1) are planted on real G1b runs from other seeds, so noise, the 5k floor, clipping and censoring are all real. D_head tests directly that the yardstick ignores a head start, which is exactly the v84 lesson. (2) The best-of-m foresight oracle O_m is a clean live ceiling for the steering channel: O_1 equals V2, the dose is graded, its gain is predicted in advance from exp_meta, and its frames are reported unbilled. Applying 'oracle-verifier best-of-N' to calibrate a search-speed yardstick is real methodological novelty. (3) A wrong-game placebo K-tilde at every step. (4) The ladder of foresight, same-game recall, generalisation beyond the frontier, online learner and cross-game carry is a sound scientific structure; WORTH-LIVE (K_early against F_live) is a smart test of whether there is room left once the run learns by itself. (5) The rewind-price dial in the 'if no' section is the most vision-relevant path to live play. Facts checked: 14 room dev games have k* >= 5, and sigma^2 = 0.376. The first gate takes about 2 to 3 days.
- Flaws: (1) As a vision vehicle it is mainly a diagnostic. The cross-game object X (KL-3) is underspecified ('the carrier made game-agnostic'), and the modal ending (p about 0.6) is 'dense-reward Atari search is not knowledge-limited', which is informative but not presentable progress. (2) Overclaim in §1: 'if even K_full does not give 2x, no cross-game object can'. K_full acts only through re-ranking 4 V2 candidates and biasing slot draws, so its ceiling is about O_4. Representation and action-generation channels are untested; only KL-0b covers that, and only after a NO. (3) There is no live demonstration-replay positive control. If O is CAPPED, the 'any knowledge' question is answered only for the selection channel. (4) The waypoint term is a +1 tie-break, a weak use of route knowledge, yet the arm is labelled 'most favourable'. (5) The t=4 row approximation for the start-cell screen. (6) O_m adds engineering weight: buffered archive writes, m-way clone/restore and a resume identity test. (7) Novelty sits in the instrument, not in anything carried.
- Graft: Stage-0 exact time-dilation spike-ins including D_head, the dose-graded foresight oracle O_m as a live channel ceiling with its gain predicted in advance, the capture ratio phi against foresight, the K_early vs F_live 'worth-live' test, and the rewind-price dial for v86.

### representation — 6

- Strengths: Aligned with the vision's own architecture: the parser, 'a way of seeing', is THE carried module in ARC4_VISION §2.6, and seeing is what becomes expensive at NES (scrolling, tiles). Its claim arm, X_on (carried init plus online learning) against F_on (random init plus the same online learning), is the right operational definition of 'learns faster, not a head start'. The mean-var placebo P_on (Raghu et al.), the random eye R and hand-coded avatar cells H_me together form an excellent control set. Its live pipeline control O carries a sharp logical point: replaying a same-game route trivially passes, so 'any knowledge' must be asked per carriable type. C0's frozen design and threshold choice rule (P(pass|2x) >= 0.8, P(pass|1.25x) <= 0.1, P(pass|head start) <= 0.05) is well specified. Facts checked: alien reaches M3 at depth 1209/1397.
- Flaws: (1) The first gate is placed where the vision itself predicts no effect. §1.3 says Atari seeing is cheap and the intra-Atari reading is expected flat, and on room games V2 already climbs with downscaled frames (P(SEEING-ROOM) = 0.15). (2) K*_see is not an oracle: an eye trained self-supervised on the same game's frames is not guaranteed to be a better cell key than frame codes. SEEING-NO-ROOM is therefore only weakly informative about seeing knowledge in general. (3) Novelty is low to moderate: learned Go-Explore cells (Latent/cell-free Go-Explore), SSL pretraining for RL (ATC, SGI, Proto-RL) and pretrained-vs-random-init tests are all established. (4) It is the heaviest build: online torch inside workers, deterministic CPU training, re-embedding and re-keying, harvest re-runs of G1b (64M + 132M frames), and 5 GPU-h with a CUDA context that commits host memory. Up to 1.4 GB per online worker (3.6 GB for 3) sits at the edge of the 3 to 4 GB Ragnarok share. (5) The online eye's compute is not billed, which favours F_on's catch-up but also hides a real cost. (6) Hard games are deferred, though that is where learned cells are known to matter.
- Graft: Make the claim contrast 'carried-init online learner vs the same learner from scratch' (X_on:F_on), use the mean-var weight-statistics placebo for any learned-weight object, keep the random-feature arm R in the no-carry field, and adopt the principle that 'any knowledge' is trivially yes via route replay, so the ceiling must be measured per carriable type. Keep the eye for the NES rung.

### procedural-live — 5

- Strengths: A clever reading of vision §1.1, cheap vs expensive to reacquire. The fast bandit has a half-life of about 140 expansions (0.995 discount, checked), so a prior there would only be a head start. Slow posture knobs, with about 2 to 33 epochs between M1 and M_top, are where a prior could act inside the measured window. The certified common prefix covers epochs 0-1, and on 14/22 room games both G1b seeds reach M1 before 200k frames (checked), so the arms share M1 exactly there. The K*_route lineage-cell boost is a live content positive control that answers 'any knowledge'. The pairwise intersection-union contrasts replace v84's 'fastest of {V2, T, S}' comparator, a valid statistical correction (a minimum over noisy arms biases log R against X). The controls separate constant defaults S, a schedule T, a context-swapped placebo P and a label-permuted placebo U. The proposer is candid that TUNED-DEFAULTS is the main risk.
- Flaws: Weakest fit with the owner vision. Carrying 'how to search' (resolution target, rollout length, frontier share, depth exponent) is algorithm configuration or AutoML warm-start, not an agent that learns games faster thanks to what it learned about games. Even a positive would read as 'earlier games tuned the hyperparameters', which is neither presentable nor tied to the console ladder. Novelty is low: the Agent57/MEME meta-controller, meta-Thompson sampling, DAC/SATzilla and Feurer warm-starting are all prior work, as the proposer admits. The same-game ceiling K*_proc is estimated from only 2 F runs per game, over 9 postures x 3 clock bins with a few epochs each, so it is noisy. A KC-proc-worth would therefore risk repeating v84's error of letting a capped instrument close a route. The first gate is the heaviest (0.49G frames, verdict day 4) and sequential, because F must finish before K*_proc exists. K*_route runs on 1 seed only. Process: a 'python -' invocation on an empty heredoc breached the read-only hard rule in letter (disclosed, nothing ran).
- Graft: Pairwise intersection-union contrasts instead of a minimum over noisy comparator arms, a byte-identical certified common prefix, a lineage-cell route boost as a live positive control alongside any 'how' oracle, and the ceiling-to-label rule (2x labels declared out of reach if the measured ceiling is below 2) written before any carried run.

---

**Ranking.** wildcard > worth-live > procedural-live > representation > hard-games

**Recommended combination.** Build v85's first gate on wildcard's shared-savestate knowledge room, verified with worth-live's instrument checks.

**0 frames, before the freeze.**
- worth-live's exact spike-ins (D2, D1.5, D_late, D1) on real runs, under a frozen size rule.
- wildcard's physics floor and the rebuild of s1 by replay in either ALE context (build note 16).
- s1 drawn from both G1b seeds.
- Note: D_head is redundant here, because a shared start removes head starts by construction.

**G1, live.** 10 room dev units with k* >= 4, 4-6 seeds, about 10-15 h. Arms:
- V2;
- D, the demonstration replay (trivial positive control);
- a graded positive control, either D at replay probability 0.25 or O_2/O_4 foresight on 4 units, to show the yardstick reads about 2x;
- K*map ("where" oracle; procedural's coverage flag and ceiling-to-label rule);
- K*mac ("how" oracle);
- optionally K*_see from representation, as one extra row.

Labels are three-way (ROOM / NO-ROOM / INCONCLUSIVE). Every comparison is pairwise, from procedural (no minimum over noisy arms). The claim header says SEARCH WITH BILLED REWINDS from a shared savestate. Verdict around day 3.

**Afterwards.**
- **Only after ROOM:** carry the matching object (Q5 motor vocabulary, or a learned progress representation). Rehearse it against designer macros, within-game mining, the best single process, tuned defaults and a placebo, then run the sealed test once.
- **If NO-ROOM:** the cheapest substrate switch is stall savestates on the failure games, with hard-games' teleport control H as the head-start negative control and calibration by resampling real runs. The sealed hard-exploration games stay reserved for that test.

**Not recommended for v85's first gate.** Procedural's 0.49G posture ceiling, the GPU eye pipeline, and the new Mantel-Haenszel yardstick on thin data.

### wildcard — 8

- Strengths: The cheapest decisive first gate in the set: about 150M billed frames, about 10 h on 3 workers, about 1.5 days of code, verdict by day 3, 0 GPU, at most 0.6 GB per worker. Starting every arm from the same savestate s1 (with the same entry and an empty archive) removes head starts in state, archive and entry by construction. That answers the brief's "matched rungs, not a head start" more cleanly than any "counted from own M1" rule, and it also removes T1 and entry noise, which raises power. The arms split a demonstration's value into "where" (K*map, future lineage codes steering cell choice) and "how" (K*mac, unindexed future motor patterns). So the gate asks the brief's real question ("any knowledge short of a demonstration") at oracle strength. It has a live planted control whose outcome is declared in advance: replaying the demonstration (D) must give R >= 4 on 8/10 units. It also has a physics-floor check, and a power simulation on real seed pairs with a rule that raises seeds from 4 to 6 if needed. The labels are three-way (ROOM, NO-ROOM, INCONCLUSIVE), so a miss is not over-read. The pool facts check out: 16 room dev games have k* >= 4, and the stored rec_actions and base_state make s1 rebuildable. The whole chain (0.93G frames, about 60 h of machine time) fits two weeks with slack.
- Flaws: 1. INCONCLUSIVE is 0.25 by the author's own estimate. NO-ROOM needs a 90% upper bound below 2, so a point estimate around 1.5-1.75 settles nothing.
2. D is a trivial-pass control: it checks the pipeline but does not show that the yardstick reads intermediate effects (1.5x-2x). The power simulation is modelled on λ residuals from full runs counted from their own M1, not from s1 starts, so its noise model may be off.
3. Each game uses one s1 from one reference run (9712), and the oracle is the exact continuation of that same trajectory. This narrows the sample of states and may flatter K*map.
4. An empty archive at s1 changes V2's dynamics (no earlier cells to fall back on). The result describes V2 started mid-game, not V2's natural climb.
5. The macro miner works from rows logged every 4th step, so step-level motor patterns are approximated. K*mac's "how" is not cleanly separated from route content.
6. Censoring is symmetric (cap for every arm); it is only acceptable because it is an oracle ceiling.
7. ROOM-MAP-only ends v85 without a carried object.
- Graft: Add worth-live's 0-frame exact spike-ins (D2, D1.5, D_late, D1), built on real runs, plus its frozen size rule. Add a graded live positive control between trivial D and nothing, for example D with replay probability 0.25, or an O_2/O_4 foresight dose on 4 units, so the yardstick is shown to read about 2x and not only 8x. Draw s1 from both G1b seeds to widen the states.

### worth-live — 7.5

- Strengths: The strongest instrument verification of the five.
- At 0 frames, exact time-dilation spike-ins on real runs of other seeds (D2, D1.5, D_late, D_head, D1), so floor, clipping and censoring are all real, with a frozen size rule: P(YES|D2) >= 0.8, P(YES|D1 or D_head) <= 0.05, recovered R(D2) within [1.8, 2.2].
- Then a live, dose-graded foresight oracle (O_2, O_4) whose gain is predicted in advance from exp_meta, with an instrument-fidelity clause.
- This is exactly the v84 lesson (simulate the achievable ceiling before freezing), applied to both the statistics and the live channel.

Other strengths:
- Labels are crisp (PASS/NARROW/CAPPED, then YES/GENERIC/PARTIAL/NO/HARM).
- A wrong-game placebo (K̃) separates steering from content.
- The reported capture φ = log R(K)/log R(O_4) is informative.
- Cost is modest: KL-0 about 230M emulated frames (15-24 h at the G1b rate), KL-1 6-9 h, decisive about 2-3 days after the freeze, 0 GPU.
- Data needs check out: G1b rows (8x8 screen, it, t), exp_meta and rec_actions all exist on disk for K_full.
- Flaws: 1. The engineering risk is the highest of the cheap gates:
   - Best-of-m needs buffered archive writes. In V2, puts happen inside the rollout, A.better() reads state that earlier puts in the same rollout changed, clones and frees are interleaved, and row-log writes must be undone.
   - It also needs per-candidate RNG streams while keeping O_1 byte-identical to V2.
   - It adds a new checkpoint/resume of the whole ALE archive. The 4x-wall O runs are more exposed to the routine supervisor kills (31 during G1b).
2. K_full is a weak learned object: a 2-run critic, an action lift, and a waypoint bonus used only to re-rank 4 V2 proposals. The likely NO is therefore overclaimed as "the most favourable same-game knowledge" and "not knowledge-limited". No route oracle at full strength is run live, so the brief's "any knowledge, even oracle-ish" is answered only for the selection channel.
3. The bounds 1 <= R(O_2) <= R(O_4) <= m are not guaranteed by construction under path dependence and noise.
4. Counting from each arm's own M1 still lets a richer archive at M1 leak a head start.
5. 6 units x 3 seeds is a small panel.
- Graft: Adopt wildcard's shared-savestate start, which removes archive and head-start leakage, shortens runs and removes most of the checkpointing need. Add a full-strength route oracle arm (K*map, or procedural's K*_route) to KL-1, so that a NO can actually mean "knowledge short of a demonstration does not help".

### procedural-live — 6

- Strengths: Several careful design pieces:
- A certified common prefix: every non-V2 arm plays exactly like V2 for 200k frames, and 14/22 games reach M1 by then, so on those games the arms share M1 exactly.
- A live pipeline positive control (K*_replay, R >= 3 on 3/4 calibration games); a miss is a build fix, never a kill.
- G-SIM with a planted 2x, a planted head start and power under heterogeneity, plus a frozen rule to add seeds.
- A ceiling-to-label rule (2x labels declared out of reach if the ceiling is below 2).
- Pairwise intersection-union contrasts, which avoid the min-over-noisy-arms bias the author quantifies (about 0.25-0.3 in log).

The ROUTE-ROOM arm on 22 games gives a direct answer to "does any knowledge help". The reasoning on which knobs are slow to reacquire, and therefore worth carrying, is sound. The 22-game panel is broad, with 0 GPU and little RAM.
- Flaws: 1. The first gate is not cheap: 0.49G frames, and it is sequential. F must run to 5M on 44 game-seeds before K*_proc can be built, so the verdict comes around day 4-5.
2. K*_proc is not a ceiling. It is a noisy table estimated from 2 bandit runs plus G1b. The kill sentence "carrying it across games cannot reach 2x" therefore overclaims; a brute-force best fixed posture per game would be the real same-game ceiling.
3. The prior on posture effects is low (PROC-WORTH 0.2): the knobs vary by about 2x around V2 values that already work. On fast-climbing games only about 1-5 decisions of 100k frames fall between M1 and M_top, so even a good prior has little room to act.
4. ROUTE-ROOM runs on only one seed.
5. The total of 1.95G frames is the largest of the five; the plan fits two weeks only if it dies early, which is its own most likely ending (about 0.8).
- Graft: Replace K*_proc with a per-game oracle best fixed posture (a small grid run once), so the ceiling is real. Use wildcard's shared savestate so posture choices act exactly in the measured window. Keep its common-prefix certificate, its ceiling-to-label rule and its pairwise intersection-union contrasts as standard parts of whichever design is chosen.

### representation — 5.5

- Strengths: The yardstick calibration (C0) is well specified:
- planted uniform, late-only and head-start-only effects on real G1b records;
- explicit design and threshold choice under P(pass|2x) >= 0.8, P(pass|1.25x) <= 0.1, P(pass|head start) <= 0.05 and P(pass|null) <= 0.02;
- a physical-floor check and a live open-loop replay control.

The author states the honest point that route knowledge trivially passes, so "any knowledge" must be asked per carriable type. The first gate is moderate in cost (0.32G frames, about 2 GPU-hours, day 3-4). The side readings (random eye R, hand-coded avatar cells H_me) are useful, and the comparator set, including the same eye learning from scratch online and a mean-var placebo, is well chosen for a later transfer claim.
- Flaws: 1. It asks about one narrow knowledge type on the substrate where the literature predicts the least room: downscaled cells are a strong Go-Explore default on dense-reward room games. SEEING-ROOM is 0.15 by the author's estimate, so the likely result is a predictable negative that says little about "any knowledge".
2. Feasibility friction is the highest:
   - an eye training pipeline on a shared GPU, where the CUDA context commits host memory within a 3-4 GB RAM budget, so GPU work is limited to when at most one explorer runs;
   - passive harvest re-runs that must reproduce the G1b runs exactly;
   - PCA-quantile re-keying of cells;
   - later, deterministic online torch training on the CPU and 1.4 GB workers.
3. The +20% throughput assumption for eye inference is unverified.
4. Comparing against the per-game best of {V2, R} costs power, and C0 must model that bias correctly.
- Graft: Run K*_see (and R) as extra arms inside wildcard's shared-savestate G1, the combined K* panel the author suggests, harvesting only those units. That turns a standalone narrow gate into one cheap row of a broader knowledge-room table.

### hard-games — 5

- Strengths: The substrate fits the vision best: it measures knowledge where fresh search fails, as humans do with priors. The panel is correctly derived: the 10 games that fail criterion (b), with freeway excluded and the 4 games that fail other criteria set apart. The live head-start teleport control (H) and the prior-alone line are a clever way to make head starts testable. A rescue probability ("unstuck in a% versus b%") is presentable. Using the sealed hard-exploration games once as the test set is principled.
- Flaws: 1. The yardstick is new and unvalidated: a Mantel-Haenszel rate ratio over rung strata, on data that are thin by construction. Failure games have k* of 0-2 and at most 3 events per stratum, with 0.5 continuity corrections and clipping.
2. G-sim fits exponential rates, which is model-based rather than resampling real runs, so it may misstate heavy-tailed climb times. NOT-MEASURABLE is a real risk (0.25 by the author's estimate).
3. 12M teacher runs on failure games may reach only 1-2 rungs beyond V2, which gives short trails and few strata. The teachers are also long runs exposed to supervisor kills, with reruns from zero.
4. When the prior-alone line is ℓ_g = -1, the reset is treated as the matched rung. That credits faster route recall to the first rung as search speed, which re-imports the head start that v84's own-M1 rule excluded. H tests only a teleport.
5. Beyond the simulation there is no guaranteed live positive control: K_trail is both the ceiling and the thing under test.
6. The first verdict comes around day 5, the total is 1.7G frames and 10-12 days, so two weeks is tight.
- Graft: Use wildcard's stall-savestate variant: start all arms from the shared plateau state where V2 stalls, and score P(climb one rung within 2M). Calibrate by resampling real runs rather than with an exponential model, add a demonstration-replay positive control, and keep H as the negative control. That makes it a cheap v86 substrate if the room gate says NO-ROOM.

---

**Ranking.** worth-live > wildcard > procedural-live > representation > hard-games

**Recommended combination.** Use worth-live as the backbone and repair it with grafts.

Calibration (Stage 0):
- Keep the exact spike-ins (D2, D1.5, D_late, D_head, D1).
- Choose units x seeds and the thresholds by representation's joint frozen rule, adding the constraint P(pass|1.25x)<=0.10.
- Inflate sigma^2 by 1.5 for steered arms, as in procedural-live.

KL-0:
- Keep the dose-graded foresight oracle O_1/O_2/O_4 with its predicted gain. Buffer Trace84 records for unkept candidates. Define PASS vs NARROW when R>=2.5 but the fidelity clause fails.
- Add wildcard's demonstration replay D as a live pipeline control, with its outcome declared in advance (R>=4), so the instrument's ceiling is checked live.

KL-1:
- Fresh V2 seeds.
- Run K_rank and K_act as separate arms beside K_full, and run K_act even if O is CAPPED, because O_4 bounds only re-ranking.
- Label with wildcard's symmetric rule: ROOM needs point R>=1.75 with lower bound >=1.25; NO-ROOM needs a 90% upper bound < 2; anything else is INCONCLUSIVE.
- Restate the negative so it names the object and its channels: not 'most favourable knowledge', not 'no cross-game object can'.
- Report a shared-savestate (s1) climb on a subset as a secondary, head-start-free yardstick.

KL-2: keep it unchanged (K_early beyond its frontier; K_early vs F_live).

KL-3:
- Use procedural-live's pairwise intersection-union contrasts against V2, F_live, T, S, P and U instead of a minimum over noisy comparators.
- Keep the TUNED-DEFAULTS sentence.
- Use X vs the same learner from scratch (representation's X_on:F_on template) as the transfer contrast.
- Add wildcard's learning-set dose curve as a caption.

Hard-games elements stay report-only: the prior-alone line, and the failure panel scored with fresh V2 seeds. If KL-1 comes out NO-ROOM, choose the next substrate (rewind price, or games where search fails) in a later version, with fresh comparator seeds.

Judge's disclosure: one shell command I ran included an accidental no-op 'python -c 1'. It imported and executed nothing beyond starting the interpreter. Everything else was read and grep only: no emulator, no GPU, no edits.

### worth-live — 8

- Strengths: The best instrument check on this lens, with two independent layers.
- Stage 0 plants exactly known time-dilation effects in real G1b runs (D2, D1.5, D_late, D_head, D1). Noise, the 5,000 floor, censoring and clipping are therefore real. A frozen size rule is applied before any threshold is set: P(YES|D2)>=0.8, P(YES|D1)<=0.05, P(YES|D_head)<=0.05, mean R-hat(D2) in [1.8, 2.2], or the gate is UNDERPOWERED.
- KL-0 is a live, dose-graded ceiling on the steering channel. O_1 = V2 <= O_2 <= O_4 holds by construction. The gain is predicted from exp_meta before KL-0, with a fidelity clause. This is the v84 note-14 lesson applied both to the live yardstick and to the channel K uses.

Head start is handled three ways: lambda from each arm's own M1, lambda_late, and lambda_abs as a guard. Comparators run on fresh seeds 9802-9804, so there is no selection on G1b outcomes.

Placebos and the learning-speed decomposition:
- K-tilde, a wrong-game placebo, runs at KL-1 (GENERIC-STEERING label).
- KL-2 separates recall from generalisation (K_early beyond its own frontier, scored on R_late) and from the run's own online learner (K_early:F_live) before any cross-game object is built. This is the cleanest within-game way to tell 'already knows' from 'learns faster'.

Every gate can fail: UNDERPOWERED, CAPPED, NO-KNOWLEDGE-LIMIT, KC-worth-live.

The first answer is cheap: about 0.3G emulated frames over 2-3 days.

Facts check: the 14 room games with k*>=5 match room_verdict_v84.json.
- Flaws: Overreach in the negative.
- K_full is not 'the most favourable same-game knowledge'. A replay of its own lineage, used as a control in other proposals, would trivially exceed 2x.
- 'If even that fails, no cross-game object can' does not follow. Another channel (cells or representation) or a pooled cross-game fit could do better. ARC4_VISION section 1.2 says WORTH 'is not a proven upper bound on all reuse'.
- YES fires at R>=1.6, yet its sentence says 'about 2x'.

CAPPED skips all of KL-1, but O_4 bounds only re-ranking among V2's own proposals. The action lift changes the proposals themselves and is not bounded by best-of-4.

The K-tilde placebo is not sharpness-matched in the waypoint channel: another game's codes never match, so that term is inert. K_full:K-tilde therefore partly measures 'waypoints on vs off'.

The critic is trained on the screen of the t=4 row but applied at decision time to the start cell (declared, but a train/serve mismatch).

O_m implementation risks:
- Trace84 records rungs at clone time (senses_v84.py on_clone). Unkept candidates' trace records must also be buffered, or rungs reached in unbilled rollouts leak into T(M).
- The predicted gain ignores that kept rollouts are longer (+50 steps per new best), so the 0.5 x log-gain fidelity clause may misfire.
- PASS vs NARROW is undefined when R>=2.5 but the fidelity clause fails.

The D-simulation uses sigma^2=0.376 from unsteered runs, with no inflation for steered arms.
- Graft: Exact time-dilation spike-ins (D2, D1.5, D_late, D_head, D1) with a frozen size rule before thresholds are set. A dose-graded best-of-m foresight oracle with an a priori predicted, bounded gain as the live channel ceiling. The KL-2 split (K_early beyond its frontier vs the online learner F_live) separating recall, generalisation and online learning.

### wildcard — 7

- Strengths: The strongest head-start control: every arm starts from the same savestate s1 (M1 of run A) with an empty archive and an identical entry. Head start in states and archive is therefore zero by construction, not just by the aggregator.

The value of a demonstration is decomposed live: D (replay, outcome declared in advance as R>=4 on >=8/10 units) against K*map ('where') and K*mac ('how'), with log R_K / log R_D reported. These are channel-specific oracle ceilings, not one trivial replay.

It is the only proposal whose negative needs evidence of absence. NO-ROOM requires a 90% upper bound < 2, and anything else is INCONCLUSIVE. My hand check with 10 units x 4 seeds, sigma^2=0.376 and tau~0.3 gives se~0.17, P(ROOM|2x)~0.8, and NO-ROOM shown when the true ratio is about 1.3x or less.

Further controls: a V2-vs-V2 instrument line, a physics floor >=8, and fresh V2 seeds.

G2 controls are well chosen:
- W, the within-game miner, as the no-carry learner;
- DM, designer macros, as the tuned default;
- S, the best single process;
- P, a matched non-advancing placebo.

A learning-set dose curve (0/4/12/41 source games, Page's L) separates a better default (flat line) from accumulation.

The cheapest decisive gate: about 150M frames, verdict by day 3.
- Flaws: The yardstick from s1 with an empty archive is a new regime, not V2's natural climb, and it has never run. The G0 power and false-positive simulation uses G1b natural-run seed pairs (sigma^2=0.376), so its noise model may not match. The D control only checks a large positive; the null rests on the live instrument line.

K*mac is unindexed and mined from slots logged every 4th step (rows log one every 4th step), so the 'how' oracle is weak. NO-ROOM-MAC bounds this miner, not motor knowledge.

ROOM-MAP-only and INCONCLUSIVE (about 0.25 by its own odds) both end v85 with no carried test.

Reaching M1 is never measured, so knowledge that speeds early discovery is invisible.

The dose sources are unsteered V2 play, a declared stand-in for a real sequential lineage.

s1 comes from run A's lineage, which is known to climb. This affects all arms equally but limits generality.
- Graft: The shared-savestate start as a secondary, head-start-free yardstick. The demonstration decomposition (D, 'where', 'how') as channel-specific ceilings. NO-ROOM defined by a 90% upper bound < 2, with an explicit INCONCLUSIVE band. The learning-set dose curve with flat tuned-default lines.

### procedural-live — 6.5

- Strengths: The best control design for tuned defaults and placebos:
- S, the best constant posture;
- T, a clock schedule that is the same on every game;
- F, the fresh posture learner;
- P, context sequences swapped between games;
- U, labels permuted;
- an X_8 dose arm;
- the honest TUNED-DEFAULTS sentence.

Contrasts are pairwise in an intersection-union test, replacing v84's per-game 'fastest of {V2, T, S}'. The proposal correctly notes the bias of taking a minimum over noisy arms.

The common prefix is certified: P0 runs byte-identically for 200k frames. On both G1b seeds 14/22 room games reach M1 before 200k (verified in room_verdict_v84.json), so head start is excluded by construction there.

G-SIM covers a planted 2x, a planted head start, and power with tau and sigma^2 x 1.5 for bandit arms, with a frozen rule to move to 3 seeds. The 22-game panel gives power of about 0.8 at 2x (my hand check: se~0.14).

K*_route answers the brief's 'any knowledge' question live.

The case for slow knobs is correct: a 0.995 discount gives a half-life of about 140 expansions, so a process prior would only be a head start.
- Flaws: K*_proc is not a ceiling. It is a posture x 3-clock-bin table estimated from 2 F runs: about 96 epochs over 27 cells, with y = the epoch rise of archive-best s_bin, mostly 0 or 1. That is a noisy estimate. X is context-conditional and pools 41 games, so it can exceed K*_proc.

As a result, two inferences are invalid:
- the ceiling-to-label rule, 'R(K*_proc)<2 makes 2x unreachable';
- the kill sentence 'carrying it across games cannot reach 2x'.

A real same-game posture ceiling would be the per-game best fixed posture, measured by running the 9 postures.

The K*_replay pipeline control can never fail as a gate, and the route arms say little about the posture channel.

The most expensive first gate: about 0.49G, verdict around day 4. The effect is small a priori around near-tuned V2 constants (its own odds: 0.2), so the likely output is a weakly informative negative.

A flat meta-bandit with a 10% floor will explore extreme postures (rollout 50, c=12) that may harm; this is only flagged (KC-harm).
- Graft: Pairwise intersection-union contrasts instead of a minimum over noisy comparators. The best-constant S and tuned-schedule T controls with the TUNED-DEFAULTS label. A certified common prefix. Inflating sigma^2 for steered arms in the power simulation.

### representation — 6

- Strengths: C0 is the most rigorous rule for freezing thresholds.
- The design and the thresholds are chosen jointly by a frozen rule over planted patterns: uniform 1, 1.25, 1.5 and 2; late-only; head-start-only.
- Constraints: P(pass|2x)>=0.8, P(pass|1.25x)<=0.10, P(pass|head start)<=0.05, P(pass|null)<=0.02.
- The bias of the best-of-two null comparators is simulated exactly as G1 uses it.
- YARDSTICK-UNDERPOWERED is a possible outcome.

The live pipeline control is an open-loop replay of the M_top record, which must read at the clip (8). The proposal makes the correct and important point that 'any knowledge' is trivially yes through route replay, so room must be asked per carriable type.

X_on vs F_on (the same eye learning online from scratch) is the cleanest 'learns faster thanks to earlier games' contrast. Controls: R (random eye), P_on (mean-var init placebo), H_me (hand-coded avatar cells), the R-HELPS flag, and fresh V2 seeds.
- Flaws: K*_see is not an oracle. A self-supervised same-game eye with a PCA-quantile grid is one learned representation, so SEEING-NO-ROOM's 'seeing cells is not V2's bottleneck' overreaches. A representation ceiling needs an evaluator-only oracle cell key in G1: RAM bytes that move along reference lineages, or at least H_me.

The prior is low and the question narrow. Room games are where downscaled cells already work; Go-Explore's representation payoff is on hard exploration. Its own odds are 0.15, and the likely output answers only 'this eye on room games'.

C0 may freeze tau_pt=1.5 with a lower bound of 1.1, below the brief's ~2x.

Online eye compute is unbilled. This is declared, but it means X_on vs F_on equalises billed frames, not compute.

Resources sit at the edge: 3.6 GB for 3 online workers against a 3-4 GB share, and the GPU must run while at most one explorer runs.

The harvest re-runs add a passivity dependency that must be certified.
- Graft: C0's joint design-and-threshold rule, with a false-pass constraint at 1.25x and the comparator-maximum bias simulated. The live replay pipeline control read at the clip. X_on vs F_on (the same learner, carried init vs from scratch) as the template for the transfer contrast. The R-HELPS flag.

### hard-games — 5

- Strengths: It matches ARC4_VISION section 1.1 by measuring knowledge where fresh search fails. The panel facts are correct:
- 10 development games fail (b);
- demon_attack, road_runner and ms_pacman fail (d) only;
- tennis fails (a) only;
- freeway is excluded.

The yardstick (MRR) counts exposure from each arm's own arrival at a rung, over matched strata. Combined with a 'prior-alone' line (the object with search off), it separates a jumpstart from search speed (Taylor and Stone), and the rescue probability is easy to read.

A live route ceiling (K_trail) and a live head-start control (H) run before any carried object. G-sim can return NOT-MEASURABLE. The zero stratum is reported apart, and skiing is declared ahead.
- Flaws: Selection bias (regression to the mean):
- The panel was chosen because V2 failed on G1b seeds 9712 and 9713, and G1 reuses exactly those runs as 2 of its 3 V2 comparator seeds.
- In the sealed test, V2 is the eligibility runs that defined the failure stratum plus one seed.
- Both bias V2 low and inflate R(K:V2) and R(X:V2). Comparators must be fresh seeds.

The instrument is the least calibrated of the five:
- k* runs from -1 to 2 on this panel, leaving few strata, with at most 3 events per stratum per arm.
- 0.5 corrections and the clip at 8 make r_g lumpy.
- Failure games are bimodal by seed. star_gunner: one seed reaches M3, the other never reaches M0. beam_rider: T1 of 3.9M vs 0.54M. An exponential-rate G-sim with Gamma(0.5) shrinkage does not represent this, and upper-stratum rates are unobserved, so the prior drives the simulated power.

G-sim sizes for 70% power at 2.5x while the pass threshold is 2.0, so power at the target effect is lower by design.

H is confounded. The teacher's mid-rung clone lies on a lineage known to climb further, so faster climbing from it is route knowledge, not a pure head start, and the expected R(H:V2) is not 1. The fallback amendment (a 100k settling window, re-scoring the same runs) changes the yardstick after data have been seen.

The sealed failure stratum may be thin (its own odds: 0.7 for >=6 units).

Phase B's W is close to v84's route-S critic and to the content that did not transfer in v83.
- Graft: The prior-alone line (the carried object with search off) as a report-only head-start bound. The failure panel as a report-only contrast stratum, with fresh V2 seeds. The rescue-probability readout as a readable secondary number.
