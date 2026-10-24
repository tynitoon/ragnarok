# v85 design proposals (workflow wf_8db018d6-25c, unedited)

---

## Proposal 1: v85 "Carry a way of seeing": a learned eye for V2's cells, gated by a same-game room test on a calibrated live yardstick

**One line.** First, check with 0 frames that the matched-rung climb can read a 2x gain. Then ask whether even an eye trained on the same game's own frames, used as V2's cell key, makes V2 climb at least ~2x faster than V2 and a random eye. Only if it does, test a carried eye that keeps learning online against the same eye learning from scratch, a mean-var placebo and hand-coded avatar cells, on the sealed games, once.

**First gate.** G1, SEEING ROOM (live, development, after C0 and G-build; verdict on about day 3–4).

Setup:
- Units: the design D chosen by C0 (6–10 units × 3–4 seeds), drawn with seed 9802 from the 22 G1b room development games. k* and the bars come from v84.
- Arms, paired on seeds 9810–9813, cap 6M, evaluator stop at M_top:
  - V2;
  - R: random-init eye, frozen, used as the cell key;
  - K*_see: an eye trained self-supervised (temporal InfoNCE + inverse dynamics on V2's own canonical slots) on frames from the same game's G1b runs, frozen, used as the cell key through an adaptive PCA-quantile grid.

SEEING-ROOM requires all of:
- R(K*_see : per-game best of {V2, R}) ≥ τ_pt with bootstrap lower bound ≥ τ_lb. The thresholds come from C0's frozen rule: P(pass | planted 2×) ≥ 0.8; P(pass | 1.25×) ≤ 0.1; P(pass | head-start-only) ≤ 0.05.
- Lower bound of R_late > 1.
- Point R_abs ≥ 1.

Otherwise SEEING-NO-ROOM: "even an eye trained on the same game's own frames does not make V2 climb ≥ 2× faster at matched rungs". The carried-seeing route closes and v85 ends with the "if no" plan.

Flags:
- R-HELPS: R(R:V2) ≥ 1.5 with lower bound > 1. Random-feature cells then join every later no-carry field.
- HARM: R(K*_see:V2) < 0.8.

Cost: ≈ 0.32G billed frames (harvest 64M, checks 20M, G1 ≤ 240M) and about 2 GPU-h. G1's V2 and R runs can double as shared comparators for other angles' same-game K* arms.

**Ceiling check.** C0 uses no emulator frames. It is committed before any v85 arm frame and before any threshold is set.

1. Planted-speed simulation on the real G1b records (22 room development games × 2 seeds).
   - Planted patterns: uniform ρ ∈ {1, 1.25, 1.5, 2}; late-only 2× (after M2); head-start-only (T(M1) halved, later gaps unchanged).
   - Each game's shift in λ is computed exactly from its records, so the 5,000-frame floor and the 6M cap are honoured.
   - Run noise is resampled from the pooled within-game residuals (σ̂² = 0.376). The comparator is the best of two null arms, as in G1. The full decision rule (two-level bootstrap) is applied.
   - Designs: (6,3), (8,3), (8,4), (10,4). Thresholds: τ_pt ∈ {1.5, 1.6, 1.75} and τ_lb ∈ {1.1, 1.2, 1.3}.
   - Frozen choice rule: the cheapest design, then the highest thresholds, such that P(pass | 2×) ≥ 0.80, P(pass | 1.25×) ≤ 0.10, P(pass | head start) ≤ 0.05 and P(pass | null) ≤ 0.02. If nothing qualifies: YARDSTICK-UNDERPOWERED, G1 does not run, and Jérémie is asked.
   - The same machinery reports the power of the fixed G2 and B1 thresholds. Rule: if P(GO | 1.5×) < 0.5, seeds go from 4 to 6.
2. Physical floor per unit: V2's T(M_k*) − T(M1) divided by (4 × lineage-depth gap + 60 per rung). This is the speed-up a route-knowing searcher could show. Expected 20–200× (alien: M3 at depth ~1.3k steps against 0.2–1.4M billed frames). It shows the instrument has room for 2×.
3. Live pipeline positive control O on 2 units, seed 9712: V2 whose first rollout replays open-loop the G1b M_top record's actions. It must read r_g at the clip (8), with all replay checks passing in either ALE context. A failure is an instrument defect, fixed and rerun.
   - What O shows: same-game route knowledge trivially passes. "Any knowledge" must therefore be asked per carriable type, and G1 asks it for "seeing".

**Odds.** My estimates:

| Event | P |
|---|---|
| C0 finds a valid design at ≤ (8 units, 4 seeds) | 0.85 |
| G-build passes | 0.85 |
| G1 SEEING-ROOM | 0.15 (0.08–0.25) |
| R-HELPS flag | 0.2 |
| GO, given room | 0.25 |
| B0 measurable (≥ 6 of 17 non-hard sealed games) | 0.8 |
| B1 PARTIAL or better, given measurable | 0.35 |
| B1 2×, given measurable | 0.12 |

- Why SEEING-ROOM is unlikely: V2 already keys on S and lives; downscaled frames are a strong Go-Explore default on room games; random features often match learned ones.
- Why GO is unlikely: learning to see costs compute, not billed frames, so the fresh online eye F_on catches up. This is ARC4_VISION §1.3's "Atari seeing is cheap".
- Overall, with gates correlated (×1.5): sealed SEARCH-CLIMB FASTER-PARTIAL or better ≈ 1% (0.3–2%); FASTER-2× ≈ 0.4%.
- Most likely ending (≈ 0.6): SEEING-NO-ROOM on day 3–4. That would be a mechanism-level negative for carried seeing, measured on a yardstick whose ceiling has been verified, plus the side readings: does a random eye or hand-coded avatar cells change V2?

**Compute.** Planning basis: G1b-realised throughput, about 15M billed frames per hour on 3 workers, plus about 20% for eye inference (numpy, ~0.3 ms per step). Average arm run assumed at 2.5M (cap 6M, stop at M_top), re-measured at G1. Compute is not billed; the yardstick counts emulator frames only.

| Stage | Billed frames | Wall | GPU |
|---|---|---|---|
| C0 | 0 | 1–2 h CPU | 0 |
| Harvest, 8 units × seeds 9712/9713 (deterministic re-runs with a passive clip sampler) | 64M | ~4.5 h | 0 |
| Smokes, machinery check, pipeline control O | ~20M | ~2 h | 0 |
| K*_see eyes (8) | 0 | — | ~2 h |
| G1 (≤ 96 runs) | ≤ 240M | ~16–19 h | 0 |
| Harvest of the other 33 games (in parallel with G1) | 132M | ~9 h | 0 |
| X folds (2) and sealed eye | 0 | — | ~2.5 h |
| G2 (160 runs; online arms slower) | ~400M | ~32 h | 0 |
| B0 (17 sealed games × 2 seeds) | 136M | ~9 h | 0 |
| B1 (8 units × 28 runs) | ~560M | ~40 h | 0 |
| **Total** | **≈ 1.55G** | **≈ 125 h** (12–15 days of calendar) | **≈ 5 h** |

- **First gate:** ≈ 0.32G billed frames, about 2 GPU-h, verdict on day 3–4.
- **Caps:** development 0.95G, sealed 0.75G. Cut ladder, applied by calendar only: G2 units 8→6; drop X; sealed units 8→6; B1 comparators 4→3 seeds.
- **RAM per worker:** frozen-eye ≤ 0.7 GB; online ≤ 1.4 GB (torch CPU with 1 thread, a 4,000-clip reservoir of about 160 MB, cell inputs of about 160 MB); harvest ≤ 0.7 GB. Three workers stay within 3.6 GB; the supervisor guard is unchanged.
- **GPU training** runs only while at most one explorer is running, because the CUDA context commits host memory. VRAM ≤ 3 GB.
- **Determinism:** online training runs on the CPU with deterministic torch, so a supervisor kill reruns from zero with an identical result.
- **Disk:** about 10 GB of compressed clips (10 consecutive 64×64 luminance frames + slots, one every 64 steps).

**Cousins.** From memory; a literature check is due before any novelty claim.

Cousins:
- Go-Explore (Ecoffet et al. 2019; Nature 2021): downscaled-frame cells versus domain-knowledge cells; the adaptive downscaling that V2 reuses.
- Cell-free / latent Go-Explore (Gallouédec & Dellandréa 2023): online learned latent cells. This is essentially the F_on arm.
- Time-myopic Go-Explore (2023).
- Intelligent Go-Explore (Lu, Hu & Clune 2024): uses foundation models, excluded here.
- NGU and Agent57 (Badia et al. 2020): inverse-dynamics embedding with kNN episodic novelty; meta-controller.
- RND (Burda et al. 2019) and the large-scale curiosity study (Burda et al. 2019): random features were competitive in-game, while IDF features generalised better to new Mario levels. This motivates the R arm.
- Representation pretraining for RL: ATC (Stooke et al. 2021), SGI (Schwarzer et al. 2021), Proto-RL (Yarats et al. 2021), APT (Liu & Abbeel 2021). Hansen et al. 2022 find from-scratch baselines competitive.
- Transfusion (Raghu et al. 2019): a mean-var init recovers much of the transfer benefit. This is the P_on placebo.
- Meta-learned exploration: MAESN (Gupta et al. 2018), E-RL² (Stadie et al. 2018), Alet et al. 2020, First-Explore (Norman & Clune 2023).
- Options and skills (DIAYN, option-critic): not used.

Not new: SSL encoders, inverse-dynamics embeddings, learned Go-Explore cells, random-feature novelty, mean-var placebos.

New in the test:
- a way of seeing learned only from the agent's own savestate play on earlier games, from pixels and its own canonical button classes, with no downloaded weights;
- judged by billed frames at matched rungs on never-opened games;
- against the same eye learning from scratch online, a weight-statistics placebo, a random eye and hand-coded avatar cells;
- preceded by a same-game room gate and a planted-speed calibration of the yardstick.

The contribution is mainly an honest test, not a new algorithm.

# v85: carry a way of seeing for search

**Claim header (every verdict JSON):** "Every v85 claim concerns a SEARCH WITH BILLED REWINDS (savestates, 60 frames per restore, every emulated frame billed). It uses the senses granted on 22 Oct 2026 (score rises, lives). It is not live play." RAM is evaluator-only. No downloaded weights. Reserved games stay closed and enduro is excluded.

## 1. Mechanism

V2 keys its archive on (lives, s_bin(S), code_r), where code_r is a block-averaged luminance frame. In Go-Explore, the cell representation is the lever: domain-knowledge cells beat downscaled frames by large factors on Montezuma and Pitfall. Downscaled frames confuse "an enemy moved" with "I moved" and can miss small sprites. v85 asks one question: does a learned, carried **eye** give V2 better cells, and therefore a faster climb?

**The eye E**
- About 0.1M parameters, numpy inference on the CPU (~0.3 ms per step, measured at G-build).
- Input: the 64×64 block-mean luminance frame plus its difference from the previous step.
- Layers: conv 32@8/4, 64@4/2, 64@3/1, then linear to z ∈ ℝ³², L2-normalised.
- Self-supervised, with no reward and no RAM:
  - (a) temporal InfoNCE: positives are 1–8 steps apart in the same rollout, with ±4 px shifts and τ = 0.1;
  - (b) inverse dynamics from (z_t, z_{t+1}) to V2's own canonical slot class (sx, sy, extra). The classes come from the agent's entry adapter. This is the NGU/IDF bias toward what the agent controls.
- Loss (a) + 0.5·(b). Adam 3e-4, batch 256, 40k steps, equal weight per game, seeded, trained on the RTX 4080.
- No game ids, no palette, no button names.

**Use in V2: cells only, one frozen usage, no usage pilot**
- Key = (lives, s_bin(S), g(z)).
- g projects z onto the top-p principal axes of the archive's stored z and splits each axis into d archive-quantile bins, with (p, d) ∈ {(2,4), (3,4), (4,4), (4,8), (6,8)}.
- Every 250 expansions, V2's own adaptive rule picks the level whose distinct-key count is closest in log to clip(30√n, 64, 8000). Axes and bins are then recomputed (deterministic eigh, fixed sign convention) and the archive is re-keyed.
- Everything else is V2 unchanged. With eye = None, the run is byte-identical to V2 (certified).

**Online variant**
- The eye keeps training on the run's own frames: a reservoir of 4,000 clips (one every 64 steps).
- Every 500 expansions it takes 200 Adam steps on the CPU (torch, 1 thread, deterministic). All cells are then re-embedded (each cell stores its 2×64×64 input) and the archive is re-keyed.
- Compute is not billed. Billed frames are the yardstick; wall time is reported.

## 2. Arms

| Arm | Eye | Role |
|---|---|---|
| V2 | none (frame code) | no-carry baseline |
| R | random init, frozen | no knowledge (RND-style random features) |
| K\*_see | same-game eye, trained on frames from that game's G1b runs (9712/9713 re-run with a passive sampler), frozen | room ceiling; development only, never a claim |
| X | carried (other games' frames), frozen | mechanism readout |
| **X_on** | carried init + online learning | **claim arm** |
| F_on | random init + online learning (same steps) | the fresh learner |
| P_on | mean-var init (each layer drawn i.i.d. with the carried layer's mean and variance; Raghu et al. 2019) + online learning | placebo: weight statistics without learned features |
| H_me | hand-coded cells: (lives, sbin, grid of "me" positions from Body.track at 4/8/16, chosen adaptively); coarse code when "me" is lost | engineer's tuned default |

- No-carry field **N** = the best of {V2, R, F_on, H_me}, chosen per game and per resample.
- Placebo **Π** = P_on.

## 3. Yardstick (v84 §7, unchanged)

- Bars B′ (seed 9706 for development); rungs M_k; k\* from G1b on both seeds.
- Cap 6M; the evaluator stops a run at M_top.
- λ is counted from each arm's own M1 over k = 2..k\*; λ_late from M2; the absolute guard λ_abs runs from frame 0.
- r_g clipped to [1/8, 8]; two-level bootstrap (2,000 resamples); one-sided permutation p; leave-one-game-out minimum; HS caption.
- Replay checks accept either ALE context (notes 15–16).

## 4. Ceiling check first: C0 (0 frames; committed before any v85 arm frame and before any threshold is set)

**1. Planted-speed simulation on the G1b records** (22 room development games × 2 seeds)
- Planted patterns:
  - uniform speed-ups ρ ∈ {1, 1.25, 1.5, 2};
  - late-only 2× (after M2);
  - head-start-only (T(M1) halved, gaps unchanged).
- Each game's shift is computed exactly from its records, so the 5,000-frame floor and the 6M cap are honoured.
- Run noise is drawn from the pooled within-game residuals (σ̂² = 0.376). The comparator is the best of two null arms, exactly as in G1.
- Designs D ∈ {(6 units, 3 seeds), (8,3), (8,4), (10,4)}; thresholds τ_pt ∈ {1.5, 1.6, 1.75} and τ_lb ∈ {1.1, 1.2, 1.3}.
- **Rule, frozen now:**
  - Pick the cheapest D, then the highest τ_pt, then the highest τ_lb, such that:
    - P(pass | 2×) ≥ 0.80;
    - P(pass | 1.25×) ≤ 0.10;
    - P(pass | head start) ≤ 0.05;
    - P(pass | null) ≤ 0.02.
  - If no combination qualifies, the label is YARDSTICK-UNDERPOWERED, G1 does not run, and Jérémie decides on seeds.

**2. Physical floor**
- Per unit: V2's T(M_k\*) − T(M1), divided by (4 × lineage-depth gap + 60 per rung).
- This is the speed-up available to a searcher that already knew the route.
- Alien reaches M3 at a depth of about 1.3k steps (≈ 5k frames), against 0.2–1.4M billed frames, so I expect 20–200×. This is checked, not assumed.

**3. Live pipeline positive control O** (2 units, seed 9712)
- V2 whose first rollout replays, open-loop, the actions of the G1b M_top record.
- It must read r_g at the clip (8), with every replay passing. Otherwise it is an instrument defect: fix and rerun, never a kill.
- What it shows: route knowledge trivially gives far more than 2×. "Any knowledge" therefore has to be asked per carriable *type*. Route memory is not carriable.

## 5. Gates

**G-build** (two fix rounds, then KC-build)
- v84 isolation tests extended to agent_v85: no evaluation import, no RAM, no game names.
- Byte identity with eye = None.
- Determinism of frozen and online arms: two 50k smokes give identical output.
- Harvest passivity: the sampler re-run of (game, 9712) reproduces the G1b JSON records exactly.
- Leak test: X's source manifest excludes its fold's units, and no sealed frame enters any development eye.
- Machinery check: R-cells reach M1 within 1.5M on 2 non-unit room games.
- RAM and fps are engineering checks only.

**G1, SEEING ROOM** (first decisive gate; live; development)
- Units: D from C0, drawn with seed 9802 from the 22 room development games.
- Arms: V2, R and K\*_see, paired on seeds 9810–9813.
- **SEEING-ROOM** if all of these hold:
  - R(K\*_see : best{V2, R}) ≥ τ_pt with lower bound ≥ τ_lb;
  - lower bound of R_late > 1;
  - point R_abs ≥ 1.
- Otherwise **SEEING-NO-ROOM**: *"even an eye trained on the same game's own frames does not make V2 climb ≥ 2× faster at matched rungs; on these games, seeing cells is not V2's bottleneck."* The route closes and v85 ends here.
- Flags:
  - R-HELPS: R(R:V2) ≥ 1.5 with lower bound > 1. Random-feature cells then join every later no-carry field.
  - HARM: R(K\*_see:V2) < 0.8.
- G1's V2 and R runs can serve as the shared comparators for other angles' same-game K\* arms, in one combined K\* panel.

**G2, transfer rehearsal** (development; single shot)
- Same units and seeds. New arms: X_on, F_on, P_on, X, H_me. V2, R and K\*_see are reused.
- X sources: 2 folds, each eye trained on all 41 dev+cal harvests minus the fold's units.
- **GO** if all of these hold:
  - R(X_on:F_on) ≥ 1.4 with lower bound > 1;
  - R(X_on:N) ≥ 1.2 with lower bound > 0.9;
  - lower bound of R(X_on:P_on) > 1;
  - lower bound of R_late(X_on:F_on) > 1;
  - point R_abs ≥ 1.
- Otherwise KC-dev, and the sealed pool stays closed.
- Capture is reported: φ = log R(X_on:N) / log R(K\*_see:N).
- The C0 machinery reports the power of these fixed thresholds. If P(GO | 1.5×) < 0.5, seeds go from 4 to 6 (rule fixed now).

**Sealed freeze**
- 3-agent adversarial review.
- Hashes committed: the 41-game eye, the code and the thresholds; power and censoring forecasts.

**B0, sealed eligibility**
- Bars on seed 9830. V2 on 9831–9832 at 4M (sampler on), on the **17 non-hard sealed games**.
- G1b room rule (a)–(d).
- At least 6 units are needed. Otherwise one second round on 9 games (seeds 9840–9842); still fewer than 6 gives NOT-MEASURABLE.
- The 7 hard-exploration games stay sealed for v86.

**B1, sealed test** (once)
- X_on and F_on on 9833–9838; V2, R, P_on and H_me on 9833–9836.
- Instrument line: F_on on 9833–35 against F_on on 9836–38. INCONCLUSIVE if its 99% interval excludes 1.
- NEGATIVE-TRANSFER is tested first: upper bound of R(X_on:F_on) < 1.
- **FASTER-2×** if all of these hold:
  - R(X_on:F_on) ≥ 2.0 with lower bound ≥ 1.3;
  - R(X_on:N) ≥ 1.5 with lower bound > 1;
  - lower bound of R(X_on:Π) > 1;
  - lower bound of R_late > 1;
  - lower bound of R_abs ≥ 0.9.
- **FASTER-PARTIAL** if all of these hold:
  - R(X_on:F_on) ≥ 1.25 with lower bound > 1;
  - lower bound of R(X_on:N) > 0.9;
  - lower bound of R(X_on:Π) > 1;
  - lower bound of R_late > 1;
  - point R_abs ≥ 1;
  - leave-one-game-out minimum > 1.
- Other labels:
  - PLACEBO-EQUIVALENT ("weight statistics, not learned seeing");
  - BELOW-NO-CARRY-FIELD;
  - EARLY-RUNGS-ONLY;
  - NOT-SIGNIFICANT.
- The HEAD-START caption appears whenever HS > 1 but the λ ratio is not above 1.

**Positive sentence:** "On never-opened Atari games, a savestate explorer whose cells come from an eye that keeps learning on the new game climbs matched rungs faster when that eye starts from weights learned self-supervised on 41 earlier games' frames (the agent's own play) than when it starts from random or mean-var weights, and faster than the no-carry field (frame code, random eye, hand-coded avatar cells). It carries a way of seeing, not game consequences. This is search with billed rewinds."

## 6. Pools and seeds

- **Development units:** the 22 G1b room games.
- **Eye sources:** all 41 dev+cal games, using harvests of their G1b seeds. These are deterministic re-runs, so V2 produces no new behaviour.
- **Sealed:** 17 non-hard games. The second round (9 games) is a top-up only. The 7 hard games stay closed.
- **Seed block 9800–9849:** a grep is clean in three worktrees; it will be repeated over all of them at the freeze.

| Seed | Use |
|---|---|
| 9800 | smoke runs |
| 9801 | C0 simulation |
| 9802 | unit draw |
| 9804 | eye init |
| 9805 | training shuffle |
| 9806 | development bootstrap and permutation |
| 9810–9813 | G1 and G2 arms |
| 9814–9829 | reserve |
| 9830 / 9831–9832 / 9833–9838 / 9839 | sealed bars / eligibility / arms / bootstrap |
| 9840–9848 | second round |
| 9849 | sealed unit draw |

## 7. Compute and RAM

Throughput basis: G1b-realised ≈ 15M billed frames per hour on 3 workers, plus about 20% for the eye. Average arm run 2.5M (re-measured at G1).

| Stage | Billed frames | Wall | GPU |
|---|---|---|---|
| C0 | 0 | 1–2 h CPU | 0 |
| Harvest, 8 units × 2 seeds | 64M | ~4.5 h | 0 |
| Smokes, machinery check, O | ~20M | ~2 h | 0 |
| K\*_see eyes (8) | 0 | — | ~2 h |
| **G1** (≤ 96 runs) | ≤ 240M | ~16–19 h | 0 |
| Harvest of the other 33 games (in parallel) | 132M | ~9 h | 0 |
| X folds + sealed eye | 0 | — | ~2.5 h |
| G2 (160 runs) | ~400M | ~32 h | 0 |
| B0 (34 runs) | 136M | ~9 h | 0 |
| B1 (224 runs) | ~560M | ~40 h | 0 |
| **Total** | **≈ 1.55G** | **≈ 125 h** | **≈ 5 h** |

- **Caps:** development 0.95G, sealed 0.75G.
- **Cut ladder, applied by calendar only:** G2 units 8→6; drop X; sealed units 8→6; B1 comparators 4→3 seeds.
- **First gate:** ≈ 0.32G billed frames, about 2 GPU-h, verdict on day 3–4.
- **RAM per worker:** frozen-eye ≤ 0.7 GB; online ≤ 1.4 GB (torch, reservoir, cell inputs); harvest ≤ 0.7 GB. Three workers stay within 3.6 GB.
- **GPU training** runs only while at most one explorer worker is running, because the CUDA context commits host memory. VRAM ≤ 3 GB.
- **Supervisor kills:** a killed run is restarted from zero and gives an identical result.
- **Disk:** about 10 GB of clips.

## 8. Odds (mine)

| Event | P |
|---|---|
| C0 finds a valid design at ≤ (8,4) | 0.85 |
| G-build passes | 0.85 |
| **SEEING-ROOM** | **0.15** (0.08–0.25) |
| R-HELPS flag | 0.2 |
| GO, given room | 0.25 |
| B0 measurable | 0.8 |

- Why GO is unlikely: learning to see costs compute, not billed frames, so F_on catches up. This is the "Atari seeing is cheap" prediction in ARC4_VISION §1.3.
- Sealed FASTER-PARTIAL or better: ≈ 1% (0.3–2%). FASTER-2×: ≈ 0.4%.
- Most likely ending (≈ 0.6): SEEING-NO-ROOM on day 4. That would be a mechanism-level negative on a verified yardstick, with the R and H_me side readings.

## 9. If no

SEEING-NO-ROOM, together with v84's capped clocks and v83's non-transferring consequences, would mean this:
- V2 on room games is limited by stochastic discovery and lineage depth, not by what it sees or predicts.
- V2 learns almost nothing during a run except its archive, so carried knowledge has little to accelerate.

Knowledge should matter where a fresh learner pays a lot. Options for v86:

**(a) Hard-exploration games** (the 7 sealed ones)
- Learned cells are Go-Explore's known bottleneck there.
- Rungs would be defined evaluator-side (true score or rooms decoded from RAM, never shown to the agent).
- Speed would be measured only where fresh V2 reaches at least two rungs; otherwise the reading is "reach", stated as possibly a head start.

**(b) The NES rung**
- Scrolling, tiles and long horizons make seeing expensive to reacquire.

**(c) A learner that must learn from frames**
- For example, distilling the archive into a live policy (the wall from v72 to v74).
- The carried eye would initialise that learner. This is where SGI- and ATC-style pretraining has shown sample-efficiency gains.

## 10. Cousins and novelty (from memory; a literature check is due before any novelty claim)

- Go-Explore (Ecoffet et al. 2019, 2021): downscaled cells versus domain-knowledge cells.
- Cell-free / latent Go-Explore (Gallouédec & Dellandréa 2023): online learned latents. F_on is essentially this.
- Time-myopic Go-Explore (2023).
- Intelligent Go-Explore (Lu, Hu & Clune 2024): foundation models, which are excluded here.
- NGU / Agent57 (Badia et al. 2020): inverse-dynamics embedding and kNN novelty; meta-controller.
- RND (Burda et al. 2019) and the large-scale curiosity study (Burda et al. 2019): random features were competitive in-game, while IDF generalised better to new Mario levels. This is why R is an arm.
- Representation pretraining: ATC (Stooke et al. 2021), SGI (Schwarzer et al. 2021), Proto-RL (Yarats et al. 2021), APT (Liu & Abbeel 2021); Hansen et al. 2022 on from-scratch baselines.
- Transfusion (Raghu et al. 2019): the mean-var init behind P_on.
- Meta-learned exploration: MAESN, E-RL², Alet et al. 2020, First-Explore.
- Options and skills (DIAYN, option-critic): not used.

**Not new:** SSL encoders, inverse-dynamics embeddings, learned Go-Explore cells, random-feature novelty, mean-var placebos.

**New in the test:**
- a way of seeing learned only from the agent's own savestate play on earlier games;
- judged by billed frames at matched rungs on never-opened games;
- against the *same eye learning from scratch online*, a weight-statistics placebo, a random eye and hand-coded avatar cells;
- preceded by a same-game room gate and a planted-speed calibration of the yardstick.

The contribution is mainly an honest test.

---

## Proposal 2: v85 "Rescue": measure knowledge where fresh search fails

**One line.** Measure knowledge on the 10 development games where V2 fails. The yardstick is a matched-rung rescue rate: a Mantel–Haenszel hazard ratio with exposure counted from each arm's own arrival at a rung, and rungs the knowledge reaches on its own, without search, excluded. It is certified by a 0-frame simulation, then by a live same-game route oracle (K_trail) and a live head-start teleport control (H), before any carried object runs. The sealed hard-exploration games become the one-shot test set.

**First gate.** **G1 RESCUE-CEILING (live, burned games only, ≈ 0.48G billed frames).** It runs after G-sim, the 0-frame statistical preflight.

- **Panel.** The 10 development games where V2 fails G1b (b): assault, beam_rider, berzerk, bowling, centipede, fishing_derby, ice_hockey, krull, name_this_game, star_gunner. A report-only contrast of 4 room games is added.
- **Teacher.** One 12M V2 run per game (seed 9801). Its best lineage gives a trail of frame codes and slots, built from pixels, buttons and senses only.
- **Arms.** Cap 4M; G1b runs 9712 and 9713 are reused.
  - V2 (3 seeds).
  - K_trail, the same-game route oracle: trail-cell boost ×8 plus the teacher's button prior with probability 0.5 (3 seeds).
  - K_where and K_how, diagnostic arms that pick the object family (2 seeds each).
  - H, a head-start teleport control: V2 rooted at the teacher's clone at rung ⌈k/2⌉ (2 seeds).
- **Measure.** The Mantel–Haenszel rung-transition rate ratio. Exposure is counted from each arm's own arrival at a rung. Strata at or below the prior-alone line are dropped.
- **PASS (CEILING-2×).** All of:
  - R(K_trail:V2) ≥ 2.0 with a lower 90% bound ≥ 1.3;
  - r_g ≥ 1.5 on at least 5 of the 10 games;
  - point R_abs ≥ 1;
  - point R(H:V2) ≤ 1.3 with a lower bound ≤ 1.
- **Other labels.**
  - CEILING-PARTIAL: R between 1.5 and 2. The sealed label is then capped at PARTIAL.
  - KC-ceiling: no carried object goes live on V2; v86 moves to a rewind-limited search.
  - INSTRUMENT-GAMEABLE: H fails. One dated amendment is allowed; a second failure stops all live claims in v85.

**Ceiling check.** The instrument is checked in three layers before any carried object is trusted.

1. **G-sim (0 frames).**
   - Per-game, per-stratum exponential rates are fitted to the 82 G1b and 12 G1a records.
   - The exact estimator is then run on simulated arms with true multipliers of ×1, ×2 and ×2.5 above the line, and with a pure head start (the first ⌈k\*/2⌉ rungs reached at frame 0).
   - Seeds per arm: the smallest n in {3, 4, 6} for which:
     - the pass rule fires in ≥ 70% of simulations at 2.5×;
     - it fires in ≤ 5% at 1×;
     - the median R̂ under the head start lies in [0.9, 1.15].
   - If no n qualifies, the verdict is NOT-MEASURABLE before any frame is spent.
2. **Live oracle ceiling, K_trail.**
   - The trail comes from a 12M same-game teacher run, so it is route knowledge from the agent's own pixels and senses, never RAM.
   - It must reach R ≥ 2.0 with a lower bound ≥ 1.3. Otherwise no carried object can be expected to show 2×.
   - The capture φ = log R(X:V2) / log R(K_trail:V2) is reported later.
3. **Live head-start negative control, H.**
   - H teleports V2 to the teacher's state at a middle rung.
   - It must not inflate the measure: point R(H:V2) ≤ 1.3 and lower bound ≤ 1.
   - This certifies that matched-rung exposure, together with the prior-alone line, cannot be gamed by a head start. One dated amendment is allowed (a 100k-frame settling window).

**Odds.** Mine; the gates are correlated.

| Event | P |
|---|---|
| G-sim yields a usable seed count (≤ 4) | 0.75 |
| H control passes | 0.8 |
| CEILING-2× | 0.35 |
| CEILING-PARTIAL | 0.25 |
| KC-ceiling | 0.4 |
| G2 GO, given a ceiling | ≈ 0.17 |
| ≥ 6 sealed failure-stratum units | 0.7 |
| Sealed SEARCH-RESCUE FASTER-PARTIAL or better | ≈ 2% |
| Sealed FASTER-2× | ≈ 0.6% |

Most likely endings:
- A verified live ceiling followed by KC-dev (≈ 0.5): knowledge can matter, but no carried object captures it.
- KC-ceiling (≈ 0.4): a clean live negative. Under billed savestates, information is not the bottleneck, which points v86 to a rewind-limited search.

**Compute.** **Billed frames.**

| Stage | Frames |
|---|---|
| G-sim | 0 (CPU minutes on existing records) |
| G1 | ≈ 0.48G, plus 0.12G for the optional contrast |
| Phase B object fits | offline CPU, under 2 h |
| G2 rehearsal | ≈ 0.44G |
| B0 eligibility on 24 sealed games, plus B1 on ≤ 10 failure-stratum units | ≈ 0.67G |
| **Total** | **≈ 1.7G** |

The G1 figure breaks down as: teachers 10 × 12M, V2 1 new seed, K_trail 3 seeds, K_where 2, K_how 2, H 2, prior-alone 0.5M runs.

**Time.**
- About 55–75 h nominal on 3 workers (3k fps unsteered, about 2k fps steered).
- About 80–120 h at G1b's realised 14–22M frames/h.
- Roughly 10–12 calendar days including reviews.

**Resources.**
- RAM: V2 workers peaked at 0.23–0.56 GB commit in G1a, so 3 workers stay at or under 2 GB, within the 3–4 GB Ragnarok share. Promise-row logging is off for the steered arms.
- Runs are deterministic and rerun from zero after supervisor kills.
- GPU: 0 h. Disk: about 20 GB.

**Cousins.** **Go-Explore (Ecoffet et al. 2019, 2021).**
- Its exploration is fresh for each game; V2 reuses its adaptive downscaling.
- Its hard-game results took hundreds of millions to billions of frames, and Pitfall needed domain-knowledge cells.
- The learned-policy and robustified variants learn per game. The backward algorithm learns from demonstrations (Salimans & Chen 2018).
- K_trail is a demonstration-like oracle in the spirit of the checkpoints of Aytar et al. (2018), used only as an instrument ceiling.
- Latent Go-Explore learns its representation online for each game. Intelligent Go-Explore relies on foundation models, which are excluded here.

**Agent57 and NGU.** Per-game novelty over a controllable embedding, with a bandit meta-controller (V2's process bandit is a cousin). Nothing is carried across games, and they use billions of frames.

**Meta-learned exploration.** MAESN, E-RL², First-Explore, meta-learned curiosity (Alet et al. 2020). These are usually tested within the training distribution.

**Options and skills.** Portable options (Konidaris & Barto 2007), eigenoptions (Machado et al.), temporally-extended ε-greedy (Dabney et al. 2021). These are the cousins of the motif process Q5 and of the tuned defaults T.

**Measurement.** Taylor & Stone (2009) give jumpstart and time-to-threshold. Mantel–Haenszel rate ratios come from survival analysis.

**Human priors.** Dubey et al. 2018; Tsividis et al. 2017.

**Honest novelty: the instrument, not the algorithm.** No cited work combines all of the following:
- a live same-game oracle ceiling;
- a live head-start teleport control;
- a prior-alone line that separates a head start from search speed;
- all applied to the games where fresh search fails, before a carried object is trusted;
- the sealed hard-exploration titles used once as the test set.

# v85 "Rescue": measure knowledge where fresh search fails

**Claim header (unchanged).** Every v85 claim concerns a SEARCH WITH BILLED REWINDS using the senses granted on 22 October (pixels, own buttons, max(0, r_step), lives). It is not live play without savestates. RAM, true score and game names are evaluator-only. No downloaded weights. Reserved games are never opened; enduro is excluded. "Live" below means emulator runs, as opposed to offline clocks. *Provenance: I only read and grepped files; I ran no Python, emulator or GPU job.*

## 1. Thesis, from the files
Knowledge pays only where it is expensive to reacquire (ARC4_VISION §1.1).

On its 22 room games, V2 reaches M1 within 0.10 to 0.51M billed frames (room_verdict_v84.json). Both offline clocks were capped at 1.16 to 1.62 because the fresh learner was already fast.

Where V2 fails there is room by construction: once V2 is stuck, the clock keeps running until the cap. On 10 development games, criterion (b) failed, meaning they missed M3 on at least one seed:
- assault, beam_rider, berzerk, bowling, centipede, fishing_derby, ice_hockey, krull, name_this_game, star_gunner;
- krull, centipede and bowling also failed in G1a.

Four other games are not search failures. demon_attack, road_runner and ms_pacman failed (d) only, because they climb too fast. tennis failed (a) only.

Humans lean on priors most in exactly such games (Dubey et al. 2018). v85 therefore asks how fast a stuck search gets unstuck, with and without knowledge.

## 2. Yardstick: matched-rung rescue rate (MRR)
Inputs are the arrival times T(M_k) of each run: billed frames from frame 0, replay-confirmed in either ALE context (build note 16).

- **Exposure from each arm's own arrival.** In rung stratum k:
  - **events** = the number of seeds that reached M_{k+1};
  - **exposure** = Σ over the seeds that reached M_k of [min(T(M_{k+1}), cap) − T(M_k)], with a floor of 5,000 frames.
  - Censored time is counted as exposure, so no cap time is imputed.
- **Matched strata only.** Stratum k enters a comparison only if both arms reached M_k on at least one seed.
- **Per-game ratio.** r_g is the Mantel–Haenszel incidence-rate ratio over strata: Σ_k d_Xk·E_Ck/E_k ÷ Σ_k d_Ck·E_Xk/E_k, where d = events and E = exposure.
  - Add 0.5 to an arm's event total when either arm has none.
  - Clip r_g to [1/8, 8].
- **Aggregate.** R = exp(mean over games of log r_g). Uncertainty comes from a two-level bootstrap (games, then seeds; 2,000 resamples). p comes from permuting arm labels within each game.
- **Head-start line.** Each carried arm also runs **prior-alone**: search off (no restores), episodes from reset with the object's button parts active, 0.5M billed frames, 2 seeds. V2 runs prior-alone too.
  - The highest rung any prior-alone run reaches is the line ℓ_g.
  - Strata at or below ℓ_g are dropped for every arm in that comparison.
  - If ℓ_g = −1 (the object alone reaches nothing), the reset is the matched rung for all arms.
  - This is how "failure turned into success" counts honestly: when the object alone finds nothing, finding the first rise sooner is search speed, not a head start.
- **Guards and readouts.**
  - Guard: point R_abs ≥ 1, where R_abs is the time from frame 0 to the highest matched rung.
  - Readable number: the rescue probability P(reach M_{k+1} within 2M frames of reaching M_k), reported as "unstuck in a% versus b% of cases".
  - Also reported: a per-rung profile and a "without near-siblings" line.
- **Sparse games.** If the bars leave u′ undefined, or give fewer than 1 rise event per bar episode, the rungs are lineage counts of rise events, n_k ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}, counted by evaluator replay of the record actions.
  - Games with zero rises in both the bars and the eligibility runs form a **zero stratum**. It is reported with a Fisher test on "any rise" and never enters R.
  - Written now: skiing has no positive rewards and should land in the zero stratum.

## 3. G-sim: 0 frames, before seed counts are frozen
1. Fit per-game, per-stratum exponential rates to the 82 G1b and 12 G1a records, with Gamma(0.5) shrinkage.
2. Run the exact G1 estimator on simulated arms with the planned panel, caps and seeds, under three scenarios:
   - (a) every rate above the line multiplied by 2 or by 2.5 (oracle);
   - (b) rates multiplied by 1 (null);
   - (c) a pure head start: the first ⌈k\*/2⌉ rungs reached at frame 0.
3. Apply the rule, fixed now. Use the smallest n ∈ {3, 4, 6} seeds per arm for which:
   - the G1 pass rule fires in ≥ 70% of simulations at 2.5×;
   - it fires in ≤ 5% at 1×;
   - the median R̂ under (c) lies in [0.9, 1.15].

   If no n qualifies, the verdict is **NOT-MEASURABLE**, before any live frame.

G-sim certifies the statistics. G1 then measures the live ceiling.

## 4. G1, the first decisive gate: RESCUE-CEILING
**Question:** does any knowledge, here same-game route knowledge (oracle-ish), make V2 climb at least 2× faster on matched strata in games where V2 fails?

- **Panel.** The 10 failure games above. freeway is excluded: no run reached M0, so its ladder sits at the game's ceiling.
- **Contrast (report-only, first to be cut).** hero, jamesbond, wizard_of_wor, kung_fu_master. Written now: R is expected to be higher on the failure games than on these.
- **Teacher.** One V2 run per game, 12M billed frames, seed 9801. Its best lineage gives the **trail**: for each step, the (11,8,8) code and the slot pressed, mapped to the learner's canonical slots by the learner's own adapter.
  - It is built from pixels, buttons and senses only.
  - It is an instrument, never a claim arm, and never used on sealed games.
- **Arms.** Cap 4M, with an evaluator stop at the teacher's top rung, so the G1b runs can be reused.

| Arm | Seeds | What it is |
|---|---|---|
| V2 | 9712, 9713 (reused from G1b), 9802 | No knowledge |
| K_trail (ceiling) | same seeds as V2 | Trail cell boost plus teacher button prior (see below) |
| K_where | 9712, 9713 | Cell boost only. Diagnostic; picks the object family |
| K_how | 9712, 9713 | Button prior only. Diagnostic; picks the object family |
| H (head-start negative control) | 9712, 9713 | V2 whose root is the teacher's clone at rung j = ⌈k_teacher/2⌉ (billed 60 frames). Its line is j |

- **How K_trail steers.**
  - A cell matches trail index i if its (11,8,8) code differs from the trail's code in at most 8 of 88 blocks. The tolerance is needed because score digits differ between runs.
  - Cells that match indices in [p − 5, p] get 8× selection weight, where p is the deepest matched index.
  - At a matched cell, the first slot of the Q2–Q4 processes is the teacher's slot, with probability 0.5.
  - Frontier draws are left untouched.
- **PASS (CEILING-2×).** All of the following must hold:
  - R(K_trail:V2) ≥ 2.0, with a lower 90% bound ≥ 1.3;
  - r_g ≥ 1.5 on at least 5 of the 10 games;
  - point R_abs ≥ 1;
  - H does not inflate: point R(H:V2) ≤ 1.3, with a lower bound ≤ 1.
- **Other labels.**
  - **CEILING-PARTIAL:** 1.5 ≤ R < 2, or the bound is below 1.3, with the lower bound still > 1. Phase B runs, but the sealed label is capped at PARTIAL.
  - **KC-ceiling:** "Even same-game route knowledge, with this trail form, does not make this savestate search climb ≥ 1.5× faster from matched rungs where it fails." No carried object goes live on V2 (§7).
  - **INSTRUMENT-GAMEABLE:** H fails. One dated amendment is allowed: exposure starts after a 100k-frame settling window, and the same runs are re-scored. A second failure ends live claims in v85.
  - **Thin-archive penalty (caption):** R(H:V2) < 0.7. The bias is conservative.
- **Family rule (fixed now).** The object family is W if R(K_where:V2) ≥ R(K_how:V2), and H otherwise.

## 5. Phase B: carried, game-agnostic objects
Each object is learned only on the 41 burned games, with leave-one-game-out in development.

- **W, "rescue sense over places".**
  - A logistic critic predicting "an expansion from this cell reaches the next rung within 200 expansions".
  - Inputs: V2's 12 cell features plus 4 place features:
    - code rarity at each resolution;
    - share of blocks changed versus the root;
    - visit rarity of the "me" position when it is tracked.
  - Fitted on the G1b and G1 V2 logs, with equal weight per game.
  - Steering: draw 32 cells with V2's weights, pick one with probability ∝ exp(1.5·clip(z, −2, 2)), and mix 0.5 with V2's own draw.
- **H, "motor motifs".**
  - Run-length-encoded sequences of canonical slot classes, 8 to 64 steps long, taken from rollouts on the 41 games.
  - Each motif is weighted by its ADV lift over its base rate.
  - The library is added as a fifth bandit process, Q5.
- **Controls.**
  - V2.
  - T, the tuned search defaults: the cross-game best frontier probability, re-key target and process mix, with no content.
  - P, the same object fitted on labels permuted within each run.
  - U, the same object fitted without labels: the input-PCA direction for W, motif frequency for H.
  - Comparator sets: N = the best of {V2, T}; Π = the better of {P, U}.
- **G2, development rehearsal** (failure panel and contrast; X on seeds 9810–9813, T, P and U on 9810–9812; V2 reused from G1).
  - **GO** requires all of:
    - R(X:V2) ≥ 1.4 with lower bound > 1;
    - lower bound of R(X:N) > 0.9;
    - lower bound of R(X:Π) > 1;
    - R_abs ≥ 1;
    - ℓ(X) ≤ ℓ(V2).
  - Reported: the capture φ = log R(X:V2) / log R(K_trail:V2).
  - Otherwise **KC-dev**, and the sealed pool stays closed.

## 6. Sealed test (once)
- **Freeze.** A 3-agent review, then the freeze commits:
  - the hashes of the object, the code and the thresholds;
  - the G-sim power rerun with the measured variances.
- **B0.** Bars (seed 9830) and V2 eligibility runs (9831–9832, 4M) on all 24 sealed games.
  - **Units = the failure stratum:** k\* < 3 on a seed, and at least 1 sensed rise.
  - Hard-exploration games that meet this rule enter first. There are 6 candidates once skiing is set aside: montezuma_revenge, pitfall, venture, private_eye, solaris, gravitar. Others are drawn with seed 9849, up to 10 units.
  - The sealed room games are a report-only contrast.
- **B1.**
  - Seeds: X on 4 seeds; T, P and U on 3 each (9833–9836); V2 = the eligibility runs plus 9833. Cap 4M.
  - Instrument line: V2 against V2 on fresh seeds. INCONCLUSIVE if its 99% interval excludes 1.
- **Labels.**
  - **SEARCH-RESCUE FASTER-2×** requires all of:
    - R(X:V2) ≥ 2 with lower bound ≥ 1.3 and p ≤ 0.025;
    - lower bound of R(X:N) > 1;
    - lower bound of R(X:Π) > 1;
    - lower bound of R_abs ≥ 0.9;
    - leave-one-game-out minimum ≥ 1.5.
  - **FASTER-PARTIAL** requires all lower bounds > 1, p ≤ 0.025 and R_abs ≥ 1.
  - Otherwise the v84 ladder applies: NOT-SIGNIFICANT, BELOW-NO-CARRY-FIELD, GENERIC-STEERING, NEGATIVE TRANSFER (upper bound of R(X:V2) < 1), HEAD-START ONLY, NULL.
  - The zero stratum is reported apart.

## 7. If the answer is no (KC-ceiling)
A KC-ceiling would mean that, with billed savestates, information is not this search's bottleneck: restoring and trying again substitutes for knowing. Humans cannot rewind, and that is why priors decide their success in Montezuma.

The substrate where knowledge should matter is therefore a **rewind-limited search**: restores only at points the agent chose, at most one per N frames (N = 2,000, then 8,000). That is "continues" rather than savestates, a step toward the vision's live-play ceiling.

v86 would rerun G-sim and G1 unchanged in that regime, since the yardstick and the H control do not depend on the regime.
- If route knowledge then gives ≥ 2×, knowledge matters as it does for humans, and Phase B moves to that regime.
- If not, the honest conclusion is that this pixel-code search has no level of representation on which knowledge can act. A learner with object-level state would come next.

## 8. Pools, seeds, compute, odds
- **Pools.**
  - 35 + 6 burned games: G-sim, G1, G2 and the object sources.
  - Sealed: B0 and B1 only.
  - The 9 second-round games are kept as a top-up.
- **Seeds.** A fresh block, 9800–9849. A grep of the tracked .py files is clean today and is repeated at freeze.
- **Billed frames.**

| Stage | Billed frames |
|---|---|
| G1 | ≈ 0.48G (+ 0.12G contrast) |
| G2 | ≈ 0.44G |
| B0 + B1 | ≈ 0.67G |
| **Total** | **≈ 1.7G** |

- **Wall time and resources.**
  - About 55–75 h nominal on 3 workers. At G1b's realised rate (about 14–22M frames/h) that is about 80–120 h, or roughly 10–12 calendar days with reviews.
  - Runs are deterministic and rerun from zero after supervisor kills.
  - V2 workers peaked at 0.23–0.56 GB commit (g1a_report.json), so 3 workers fit in ≤ 2 GB. Promise-row logging is off for the steered arms.
  - GPU: 0 h.
- **Odds (mine; the gates are correlated).**

| Event | P |
|---|---|
| G-sim gives n ≤ 4 | 0.75 |
| H passes | 0.8 |
| CEILING-2× / CEILING-PARTIAL / KC-ceiling | 0.35 / 0.25 / 0.4 |
| G2 GO, given a ceiling | ≈ 0.17 |
| Sealed failure stratum has ≥ 6 units | 0.7 |
| Sealed FASTER-PARTIAL or better | ≈ 2% |
| Sealed FASTER-2× | ≈ 0.6% |

- **Most likely endings.**
  - A verified live ceiling, then KC-dev (≈ 0.5). Knowledge can matter, but nothing carried captures it.
  - KC-ceiling (≈ 0.4). A clean, informative negative about knowledge under savestate search.

## 9. Cousins and honest novelty
- **Go-Explore (Ecoffet et al. 2019, 2021).**
  - Its exploration is fresh for each game.
  - Its hard-game results took hundreds of millions to billions of frames, and Pitfall needed domain-knowledge cells.
  - The learned-policy and robustified variants learn per game. The backward algorithm learns from demonstrations (Salimans & Chen 2018).
  - K_trail is that kind of demonstration oracle, close to the checkpoints of Aytar et al. (2018), but used here only as an instrument ceiling.
  - Latent Go-Explore learns its representation online for each game. Intelligent Go-Explore needs a foundation model, which is excluded here.
- **Agent57/NGU.** Per-game novelty over controllable embeddings, with a bandit over exploration modes (V2's process bandit is a cousin). Nothing is carried across games.
- **Meta-learned exploration.** MAESN, E-RL², First-Explore, Alet et al. 2020. Usually tested within the training distribution, not on sealed, never-opened games.
- **Options and skills.** Portable options (Konidaris & Barto 2007), eigenoptions, temporally-extended ε-greedy (Dabney et al. 2021). Q5 and T are their cousins.
- **Measurement.** Taylor & Stone (2009) separate "jumpstart" from "time-to-threshold". The prior-alone line and the matched strata make that separation testable. Mantel–Haenszel rate ratios are standard.
- **What is new is the instrument, not the algorithm:**
  - a live same-game oracle ceiling;
  - a live head-start teleport control and a prior-alone line, run before any carried object is trusted;
  - all aimed at the games where fresh search fails, with the sealed hard-exploration titles as the one-shot test set.


---

## Proposal 3: v85: is V2's climb knowledge-limited? A knowledge ladder calibrated by a foresight oracle

**One line.** First check the yardstick with exact time-dilation spike-ins, then check the steering channel with a live best-of-m foresight oracle whose gain is bounded and predicted in advance. Only then ask whether the most favourable same-game knowledge (critic + action lift + waypoints from the game's own 8M G1b frames) makes V2 climb about 2x faster from matched rungs, with a wrong-game placebo. If it does, separate recall from generalisation and from online learning before building any cross-game object.

**First gate.** The first decisive gate has two parts, run back to back with thresholds frozen at the day-0 commit.

**KL-0, live channel ceiling (instrument only).** V2 against O_4 on 6 room development units × 3 fresh seeds (9802–9804); O_2 on 2 seeds as a dose point.
- O_m is a best-of-m foresight oracle: m candidates drawn from V2's own laws, buffered rollouts, and only the candidate with the largest rise in archive-best S is billed and committed. O_1 ≡ V2.
- PASS: R(O_4:V2) ≥ 2.5 with 90% lower bound ≥ 1.6, and log R ≥ 0.5 × the predicted log gain.
- CAPPED (below 1.6): stop with "the climb is not decision-limited even with 4-way foresight".

**KL-1, the knowledge question.** K_full against V2 (reused) and against K̃, the wrong-game placebo.
- K_full is fitted only on the game's own G1b runs: a cell/process critic (ADV and S heads over 44 search features plus screen PCs), an action lift from record lineages, and frame-code waypoints. It re-ranks 4 V2 candidates and biases button draws.
- YES-KNOWLEDGE: R(K_full:V2) ≥ 1.6, lower bound ≥ 1.15, p ≤ 0.05, r_g ≥ 1.5 on at least half the units, and lower bound of R(K_full:K̃) > 1.
- Other labels: GENERIC-STEERING if the gain does not beat K̃; PARTIAL for 1.25–1.6; NO-KNOWLEDGE-LIMIT below 1.25; HARM below 0.8.
- Yardstick: v84's matched-rung λ, counted from each arm's own M1 (head start excluded), with λ_late and λ_abs reported.
- Cost: about 2–3 days.
- Claim scope: a SEARCH WITH BILLED REWINDS.

**Ceiling check.** **Stage 0 (0 frames, before the freeze, on the 44 G1b room runs plus G1a 9710).**
- Spike-ins with exactly known effects, built on real runs of other seeds, so between-seed noise, censoring, the 5,000 floor and the [1/8, 8] clipping are all real:
  - D2: times after the run's own M1 halved (truth 2.00);
  - D1.5;
  - D_late: halved only after the run's own T(M2) (truth R_late 2);
  - D_head: M1 twice as early, climb unchanged (truth 1.00; checks immunity to a head start);
  - D1: null.
- A frozen size rule takes the smallest of (6, 3), (8, 3), (8, 4), (10, 4) units × seeds with:
  - P(YES | D2) ≥ 0.8;
  - P(YES | D1) ≤ 0.05 and P(YES | D_head) ≤ 0.05;
  - mean R̂(D2) in [1.8, 2.2].
  - Otherwise the gate is declared underpowered and does not run.
- Also written before any live run:
  - the predicted gain of O_m from the per-segment S-rise rate p in `exp_meta`, under independence: 1/(1 − (1 − p)^m) vs 1/p;
  - bounds known by construction: 1 = R(O_1) ≤ R(O_2) ≤ R(O_4), at most about m;
  - a lineage-depth floor per unit; no knowledge can beat it, and units below a ratio of 4 are flagged.

**KL-0 (live channel ceiling).** The O_4 dose arm must show R ≥ 2.5 (lower bound ≥ 1.6) and at least half the predicted log gain before the knowledge thresholds are applied. Otherwise the result is CAPPED or NARROW, labelled as such, and is never read as a negative on knowledge carried through other channels.

**Odds.** - Qualifying size of at most (8, 4) in the D-simulation: 0.9.
- O_4 PASS / NARROW / CAPPED: 0.6 / 0.25 / 0.15.
- KL-1, given it runs: YES-KNOWLEDGE 0.17, GENERIC-STEERING 0.1, PARTIAL 0.25, NO-KNOWLEDGE-LIMIT 0.48. Overall YES ≈ 0.15.
- Given YES: GENERALISES (K_early beyond its frontier) and WORTH-LIVE (beats F_live) both pass ≈ 0.2, so ≈ 0.03 overall.
- KL-3 development GO ≈ 0.3 given both.
- Sealed SEARCH-CLIMB FASTER-2× ≈ 0.2%; PARTIAL or better ≈ 0.5%.
- Most likely ending (≈ 0.6): within about 3 days, an instrument-verified NO, either from O CAPPED or from NO-KNOWLEDGE-LIMIT. It would say that V2's billed-rewind climb on dense-reward development games is search-limited, not knowledge-limited, and would send v86 towards the rewind-price dial and hard-exploration analogues.

**Compute.** Basis: a V2 run stopped at M_top averages about 2M billed frames, about 15–25 minutes on one core (G1b: 4M took 30–50 minutes).

| Stage | Runs | Cost |
|---|---|---|
| Stage 0 | — | About 2 h of CPU, 0 frames; lineage replays under 1M evaluator frames |
| KL-0 | 18 V2 + 18 O_4 (≈ 4× emulation) + 12 O_2 (≈ 2×) | About 230M emulated frames, about 30–48 core-hours, about 10–16 h on 3 workers (15–24 h at G1b's realised rate) |
| KL-1 | 36 K_full and K̃ runs | About 72M frames, 4–6 h (6–9 h realised) |
| KL-2 | 72 runs | About 7–12 h |
| KL-3 | Development plus sealed B0 and B1 | About 5–7 days |

- The first decisive answer arrives about 2–3 days after the freeze.
- Totals: about 0.5G billed frames in development plus about 0.25G of unbilled oracle frames; about 0.7G billed frames sealed.
- RAM: at most 0.8 GB per worker (V2 peak commit 0.23–0.56 GB; O's buffered clones under 50 MB; K objects under 20 MB), so 3 workers need at most 2.4 GB. The supervisor guard stays at 14 GB free. GPU: 0 h.
- O runs write a checkpoint every 500 expansions; a test that a resumed run equals an uninterrupted one is added to G-build.

**Cousins.** All from memory; a literature check is due before any novelty claim.
- Go-Explore (Ecoffet et al. 2019, Nature 2021): domain-knowledge cells beat downscaled frames on hard exploration, so representation knowledge pays there; policy-based Go-Explore.
- Intelligent Go-Explore (Lu, Hu, Clune 2024): foundation-model knowledge selects archive states, the closest cousin of K's re-ranking.
- Latent Go-Explore (Gallouédec and Dellandréa 2023).
- NGU and Agent57 (Badia et al. 2020): a bandit meta-controller over exploration policies, like V2's process bandit.
- Meta-learned exploration: MAESN (Gupta et al. 2018), First-Explore (Norman and Clune 2023), AdA (2023).
- Options and skills: Sutton, Precup and Singh 1999; option-critic (Bacon et al. 2017); DIAYN (Eysenbach et al. 2018).
- AlphaZero and ExIt (Anthony et al. 2017): learned priors speed tree search.
- Levin tree search and policy-guided heuristic search (Orseau et al. 2018; Orseau and Lelis 2021): search cost bounded by the policy's quality.
- Hindsight optimisation (Yoon et al. 2008).
- Best-of-N with an oracle verifier ("Large Language Monkeys", Brown et al. 2024): O_m is exactly this, and φ is a learned verifier's capture of it.
- Learned node selection in branch and bound (He, Daumé, Eisner 2014).

What I believe is new is the combination, not any part:
- exact time-dilation spike-ins plus a dose-graded foresight oracle, used to calibrate a live search-speed yardstick before thresholds are frozen;
- a gated ladder: foresight ≥ same-game recall ≥ generalisation beyond the knowledge frontier ≥ online learner ≥ cross-game carry;
- a wrong-game placebo at each step.

# v85 proposal: is the climb knowledge-limited at all?

**Claim header (unchanged from v84).** Every v85 claim concerns a SEARCH WITH BILLED REWINDS: savestates, 60 frames per restore, every emulated frame billed. The agent's senses are those granted on 22 October: pixels, its own buttons, max(0, r_step) and lives. Nothing here is live play without savestates. The oracle arms (O, D) are evaluator instruments and never count as agents.

*Editor's note.* I only read and grepped. I ran no Python, emulator or GPU job, and edited no file.

## 1. Why this question comes first
v84 closed two carried-knowledge routes with an offline clock whose ceiling was below its own 2× bar: 1.16 to 1.62 even with near-oracle priors. One question came before those routes and was never asked: **does V2's climb have room for knowledge at all?**

V2's time is (expansions needed) × (frames per expansion). Knowledge can act on that time in only three ways:
- re-rank which (cell, process, buttons) proposal gets billed;
- bias the proposals themselves (action priors);
- change the cell representation.

The most favourable knowledge any learner could hold is everything extractable from 8M frames of its own earlier play of the same game. If even that does not make V2 about 2× faster from matched rungs, no cross-game object can. v85 should then say so cheaply and change the substrate. If it does, we learn which channel carries the gain, and whether the gain is recall or generalisation, before building anything cross-game.

## 2. Arms (a knowledge ladder)
Every arm is V2 at its frozen v84 constants plus a steering object. The object plugs into the existing hooks (`pick_cell`, `pick_process`, `on_put`, `on_iter`) and one new hook, the in-rollout slot draw. An object with no content reduces to V2 byte for byte (prefix-identity test).

- **V2.** No knowledge.
- **O_m (evaluator foresight, instrument only).**
  - Each expansion draws m candidates (cell, process, first slot, rollout RNG stream) from V2's own laws. Candidate 1 is V2's own draw.
  - All m rollouts run through clone/restore, with their archive writes buffered.
  - The evaluator keeps the candidate with the largest rise in archive-best sensed S, then ADV, then new cells. Ties go to candidate 1.
  - Only the kept rollout is billed (its frames plus 60) and committed to the archive, the bandit and the row log. The other rollouts' frames are reported as unbilled oracle frames.
  - With sticky actions at 0 the emulator is deterministic, so the only randomness is the agent's own button draws. O_m therefore means perfect knowledge of which of m proposals succeeds.
  - O_1 ≡ V2. Doses m ∈ {2, 4}.
- **K_full (the most favourable legitimate same-game knowledge).** Fitted only on game g's two G1b runs (9712, 9713), then hashed and frozen before KL-1. It has three parts:
  1. **Cell/process critic.** Two L2-logistic heads (ADV, and a rise of archive-best S) on the 44 logged search features plus the top 32 PCs of the 8×8-pooled foreground screen of each expansion's first logged row. That row is at t = 4, a declared approximation of the start cell. Ridge chosen by 2-fold cross-seed validation.
  2. **Action lift.** For each canonical slot × process: the log-ratio of the slot's frequency on record lineages (`rec_actions`) to its frequency among all logged rollout buttons, clipped to [−2, 2].
  3. **Waypoints.** RES[0] frame codes along both runs' record lineages, replayed on the evaluator side from `base_state` and tagged with the rung reached.
  - **Live use.** Draw 4 candidates from V2, re-rank them by s = logit_ADV + logit_S + 1·[code is a waypoint at a rung ≥ the current one], take the argmax (ties go to candidate 1). In Q2–Q4, draw the first slot and each button change from 0.5·uniform + 0.5·softmax(lift). Q1 is unchanged.
- **K̃ (wrong-game placebo).** The K_full of another unit, assigned by a fixed derangement (seed 9801). Same machinery, same sharpness.
- **KL-2 only.**
  - K_early: fitted only on reference data billed before T_ref(M2).
  - F_live: the same machinery, learned online from the run's own data, refit every 250 expansions from flat.
  - K_rank: re-ranking only.
  - K_act: action lift only.

## 3. Yardstick and ceiling checks (all before any threshold is frozen)
**Yardstick.** v84 §7 unchanged:
- bars B′ (seed 9706), rungs M_k, and k\* taken from G1b's two seeds;
- λ = mean over k = 2..k\* of log max(5000, T(M_k) − T(M1)), counted from each arm's own M1;
- λ_late counted from the arm's own T(M2) over k = 3..k\*; λ_abs as a guard;
- 6M cap, stop at M_top, v84 censoring rules, r_g clipped to [1/8, 8], R = exp(mean ℓ_g);
- two-level bootstrap and within-game permutation;
- replay checker with the two-context rule (build note 16).

**Stage 0 (0 frames, on the 44 G1b room runs, plus G1a seed 9710 where it exists).**
- **Spike-ins with exactly known effects.**
  - D2: every time after the run's own M1 halved (true R = 2.00).
  - D1.5 (true R = 1.5).
  - D_late: halved only after the run's own T(M2) (true R_late = 2).
  - D_head: M1 reached twice as early, climb unchanged (true R = 1.00; checks immunity to a head start).
  - D1: null.
  - Each spike-in is built from runs of other seeds, so between-seed noise, censoring, the 5,000 floor and clipping are all real. Per-game means come from G1b; residuals are resampled from the pooled empirical set (σ̂² = 0.376).
- **Size rule** (written in the text, applied before the freeze commit). Take the smallest of (6 units, 3 seeds), (8, 3), (8, 4), (10, 4) such that:
  - P(YES | D2) ≥ 0.8;
  - P(YES | D1) ≤ 0.05 and P(YES | D_head) ≤ 0.05;
  - the mean recovered R̂(D2) lies in [1.8, 2.2].
  - If no size qualifies, the gate is declared underpowered and does not run.
- **Predicted gain of O_m**, written down before KL-0. From `exp_meta`, per game and per segment between records, estimate the per-expansion probability p of an S rise. Under independence, best-of-m needs 1/(1 − (1 − p)^m) expansions per rise instead of 1/p. Bounds known by construction: 1 = R(O_1) ≤ R(O_2) ≤ R(O_4), and R(O_m) is at most about m.
- **Depth floor.** (T(M_k) − T(M1)) / (4·(d(M_k) − d(M1))). No knowledge can beat lineage depth; units with a ratio below 4 are flagged.
- **Readout (not a gate).** K's same-game, cross-seed AUC.

**KL-0 (live channel ceiling).** V2 and O_4 on 3 seeds; O_2 on 2 seeds as a dose point.
- **PASS:** R(O_4:V2) ≥ 2.5 with 90% lower bound ≥ 1.6, and log R(O_4) ≥ 0.5 × the predicted log gain (instrument fidelity).
- **NARROW** (point estimate in [1.6, 2.5)): KL-1 runs, captioned with the channel ceiling.
- **CAPPED** (below 1.6): "even 4-way foresight does not speed this climb 1.6×; it is not decision-limited". KL-1 does not run.

## 4. Gates (frozen at the day-0 commit, after Stage 0)
**KL-1 — the first decisive knowledge gate.** Paired seeds 9802–9804; the V2 runs are reused from KL-0.
- **YES-KNOWLEDGE.** All of:
  - R(K_full:V2) ≥ 1.6, with 90% lower bound ≥ 1.15 and p ≤ 0.05;
  - r_g ≥ 1.5 on at least half the units;
  - lower bound of R(K_full:K̃) > 1.
  - Sentence: "same-game knowledge from 8M frames of the agent's own earlier play makes the billed-rewind climb about 2× faster from matched rungs on these development games."
- **GENERIC-STEERING.** The conditions against V2 hold, but the lower bound of R(K_full:K̃) is ≤ 1. The gain comes from steering, not from game content: tune V2 instead of carrying knowledge.
- **PARTIAL.** 1.25 ≤ R < 1.6, or the lower-bound condition fails.
- **NO-KNOWLEDGE-LIMIT.** R < 1.25. Sentence: "even the most favourable same-game knowledge does not speed V2's matched-rung climb by 1.25×; on dense-reward Atari this search is not knowledge-limited." Flag HARM if R < 0.8.
- **Reported:** capture φ = log R(K_full) / log R(O_4); late and absolute-time ratios; HS with its interval; a per-unit table.

**KL-2** (only after YES, or PARTIAL with R ≥ 1.4). Same units and seeds.
- **RECALL-ONLY:** R_late(K_early:V2) < 1.4. The knowledge works only inside its own frontier.
- **GENERALISES:** R_late(K_early:V2) ≥ 1.6 with lower bound ≥ 1.15.
- **WORTH-LIVE:** R(K_early:F_live) ≥ 1.4 with lower bound > 1. This is the room left for any carried object once the run learns by itself. If it fails (KC-worth-live), no cross-game object is built.
- **Carrier:** whichever of K_rank and K_act gives the larger gain.

**KL-3 (cross-game; only after GENERALISES and WORTH-LIVE).**
- **Object X:** the carrier made game-agnostic and learned leave-one-game-out on the development games, then refined online exactly like F_live. Two forms:
  - action lift → a map from canonical slot class × ego-centric context to lift;
  - critic → a zero-mean prior on the critic's slopes.
- **Controls:**
  - F_live;
  - tuned search defaults T (frozen grid on the calibration games);
  - the single best process S;
  - P (labels permuted within each run) and U (input-statistics prior with the same spectrum);
  - N = best of {V2, T, S, F_live}, chosen per game and per resample;
  - K_early as the same-game anchor.
- **Development rehearsal** on the KL units:
  - R(X:F_live) ≥ 1.4 with lower bound > 1;
  - R(X:N) ≥ 1.2;
  - lower bound of R(X:Π) > 1;
  - lower bound of R_late > 1;
  - point R_abs ≥ 1.
- **Sealed test, once:** 3-agent review and sealed freeze; then B0 (V2 on 2 seeds, eligibility as in G1b); the D spike-ins re-run on the B0 runs at 0 frames; then B1 once, scored with v84's ladder (FASTER-2×, PARTIAL, EARLY-RUNGS-ONLY, BELOW-NO-CARRY-FIELD, GENERIC-STEERING, NEGATIVE TRANSFER; HEAD-START and LATE-START captions).

## 5. If NO: what it means and what would make knowledge matter
A NO after O passes is a finding. V2's frontier rule, score bins and bandit already extract what knowledge offers at 4M frames. The remaining cost is search luck, and only foresight (free compute) buys that. "Learns faster thanks to earlier games" then cannot show up as a 2× faster billed-rewind climb on dense-reward Atari. v86 should stop building carried objects for this substrate. Three options, for Jérémie to decide (each changes the substrate):
1. **Rewind-price dial.** V2, K_full and O_4 on the same units, with restores priced at 600 frames and a restore budget. If R(K:V2) rises with the price, knowledge belongs to the low-rewind regime, which is the vision's live-play claim.
2. **Hard-exploration analogues on development games.** The non-room development games where V2 stalls (k\* ≤ 2), scored as the probability of reaching M2 or M3 within 4M. Knowledge comes from the O_4 runs, which reach further. The sealed hard-exploration games stay sealed.
3. **KL-0b representation oracle.** Cells keyed on evaluator RAM bytes that move along the reference lineages; report-only. Go-Explore's ablations put the payoff of domain knowledge in the representation. Only a 2× result here would justify a learned progress embedding.

## 6. Pools and seeds
- **Units:** 6 (or the size rule's number), drawn with seed 9800 from the 14 room development games with k\* ≥ 5: amidar, asterix, bank_heist, chopper_command, gopher, jamesbond, kung_fu_master, phoenix, pooyan, qbert, up_n_down, video_pinball, wizard_of_wor, yars_revenge.
- **Knowledge sources:** G1b seeds 9712/9713 (already burned).
- **Untouched until KL-3's sealed freeze:** the 24 sealed and 9 second-round games. Reserved games are never opened; enduro is excluded.
- **Seeds 9800–9849.** The `.py` files are clean today; the "no seed use" grep is repeated at the freeze.

| Seed | Use |
|---|---|
| 9800 | Unit draw and D-simulations |
| 9801 | Placebo derangement |
| 9802–9804 | KL-0, KL-1 and KL-2 arms |
| 9805–9806 | O_2 |
| 9807 | Bootstrap and p |
| 9808–9811 | KL-3 development |
| 9812–9819 | Reserve |
| 9820 / 9821–9822 / 9823–9828 / 9829 | Sealed bars / eligibility / arms / bootstrap |
| 9830–9839 | Second round |
| 9840 | Sealed unit draw |

## 7. Compute and RAM
Basis: a V2 run stopped at M_top averages about 2M billed frames, about 15–25 minutes on one core.

| Stage | Runs | Cost |
|---|---|---|
| Stage 0 | — | About 2 h of CPU, 0 frames |
| KL-0 | 18 V2 + 18 O_4 (≈ 4× emulation) + 12 O_2 | About 30–48 core-hours, about 10–16 h on 3 workers (15–24 h at the rate G1b actually achieved) |
| KL-1 | 36 | About 4–6 h (6–9 h realised) |

- The first decisive answer arrives about 2–3 days after the freeze.
- KL-2: about 7–12 h. KL-3 (development plus sealed B0 and B1): about 5–7 days.
- Totals: about 0.5G billed frames in development plus about 0.25G of unbilled oracle frames; about 0.7G billed frames sealed.
- RAM: workers stay at or below 0.8 GB (V2 peaks 0.23–0.56 GB; O's buffered clones add under 50 MB), so 3 workers need at most 2.4 GB. The supervisor guard stays at 14 GB free. GPU: 0 h.
- O runs write a checkpoint every 500 expansions. G-build tests that a resumed run equals an uninterrupted one.

## 8. Odds (mine)

| Event | P |
|---|---|
| A qualifying size of at most (8, 4) | 0.9 |
| O_4 PASS / NARROW / CAPPED | 0.6 / 0.25 / 0.15 |
| KL-1, given it runs: YES / GENERIC / PARTIAL / NO | 0.17 / 0.1 / 0.25 / 0.48 |
| GENERALISES and WORTH-LIVE, given YES | ≈ 0.2 |
| Sealed FASTER-2× | ≈ 0.2% |
| Sealed PARTIAL or better | ≈ 0.5% |

**Most likely ending** (p ≈ 0.6): an instrument-verified NO within about 3 days. That is the information v85 needs.

## 9. Cousins and novelty (from memory; a literature check is due before any claim)
**Cousins:**
- Go-Explore (Ecoffet et al. 2019/2021): domain-knowledge cells and its policy-based variant.
- Intelligent Go-Explore (Lu, Hu, Clune 2024): foundation-model knowledge chooses archive states. This is the closest cousin of K's re-ranking.
- Latent Go-Explore (Gallouédec and Dellandréa 2023).
- NGU and Agent57 (Badia et al. 2020): a bandit meta-controller over exploration policies, like V2's process bandit.
- Meta-learned exploration: MAESN, First-Explore, AdA.
- Options and skills: Sutton, Precup and Singh 1999; option-critic; DIAYN.
- AlphaZero and ExIt: priors that speed tree search.
- Levin tree search and policy-guided heuristic search (Orseau et al. 2018; Orseau and Lelis 2021): search cost bounded by the policy's quality.
- Hindsight optimisation (Yoon et al. 2008).
- Best-of-N with an oracle verifier ("Large Language Monkeys", Brown et al. 2024): O_m is exactly this, and φ is the capture of a learned verifier.
- Learned node selection in branch and bound (He et al. 2014).

**What I believe is new is the combination**, not any single component:
- exact time-dilation spike-ins plus a dose-graded foresight oracle, used to calibrate a live search-speed yardstick before thresholds are frozen;
- a ladder from foresight down to cross-game carry: foresight ≥ same-game recall ≥ generalisation beyond the knowledge frontier ≥ online learner ≥ cross-game carry;
- a frozen gate with a wrong-game placebo at each step.


---

## Proposal 4: v85: Carry how to search, measured live. First a live same-game ceiling, then search-allocation priors tested against tuned defaults

**One line.** Carry slow, game-agnostic search-allocation knowledge: a context-conditional prior for a bandit over V2 "postures" (resolution target, rollout length, frontier share, depth weighting). First verify live, on the 22 G1b room games, that same-game posture knowledge (K*_proc) can make V2 climb at least 1.75x faster from matched rungs. Same-game route memory (K*_route) is run as a positive control. Only then test X against V2, F, the best fixed posture S, the tuned schedule T and a context-swapped placebo, sealed once. Every claim concerns a search with billed rewinds.

**First gate.** G-CEIL is live and decisive. It runs on the 22 G1b room development games, with k* and the bars fixed from G1b and a 5M cap for every arm.

**Runs.**
- F (flat-prior posture meta-bandit) on seeds 9852–9853, run to 5M as source data.
- V2 and K*_proc on seeds 9854–9855.
- K*_route on 9854.

**Arms.**
- K*_proc is the meta-bandit primed with a posture-by-clock-bin table. The table comes from the same game's own F runs plus its G1b runs, counted as P0 epochs.
- K*_route is V2 with ×8 selection weight on cells whose (s_bin, coarse frame code) lie on the same game's best G1b lineage. These codes come from an evaluator replay of the agent's own frames.
- Every non-V2 arm plays V2 byte-identically for the first 200k billed frames, a certified common prefix.

**Yardstick.** v84's λ: the mean over k = 2..k* of log max(5000, T(M_k) − T(M1)), counted from each arm's own M1. λ_late and λ_abs are also computed. The bootstrap resamples games, then seeds (seed 9856).

**PROC-WORTH** requires:
- R(K*_proc:V2) ≥ 1.75, one-sided 95% lower bound ≥ 1.25;
- lower bound of R_late(K*_proc:V2) > 1;
- r_g ≥ 1.5 on at least a third of the games.

**ROUTE-ROOM** answers "does any knowledge help": R(K*_route:V2) ≥ 2.0 with lower bound ≥ 1.3.

**Kill.** If PROC-WORTH fails, the label is KC-proc-worth: "with two runs of same-game evidence, posture knowledge does not make V2 climb 1.75x faster; carrying it across games cannot reach 2x." v85 then ends, with the ROUTE answer and the §9 consequences.

**Flags and invalidation.**
- KC-harm if R(F:V2) < 0.8.
- ROUTE-WEAK if lineage-code coverage is below 30%.
- INCONCLUSIVE if the V2 C-vs-D line's 99% interval excludes 1, if 40% or more of runs are censored, or if the prefix is violated.

**Cost.** About 0.49G billed frames, 22–35 h on 3 workers. Verdict around day 4.

**Power (by hand, σ² = 0.376, τ = 0.2).** About 0.8 at a true 2x and about 0.15 at 1.5x. G-SIM replaces these figures before freeze.

**Ceiling check.** There are four layers, all completed before any carried arm runs.

**(1) G-SIM (offline, 0 frames, before freeze).** It uses the 44 G1b V2 runs:
- (a) the seed-vs-seed null gives the instrument line;
- (b) planted 2x: halve one seed's T(M_k) − T(M1). The frozen aggregator (5k floor, clip [1/8, 8], 5M censoring) must recover R in [1.8, 2.2];
- (c) planted head start: shift only T(M1). R must stay 1.00 ± 0.05 while HS = 2, which proves head starts are not credited;
- (d) power: resample λ residuals with heterogeneity τ in {0, 0.2, 0.4} and σ² ×1.5 for bandit arms.

The rule is frozen now: if G-CEIL power is below 0.75 at a true 2x, or the false-pass rate exceeds 0.10 at 1.25x, every G-CEIL arm goes to 3 seeds before freeze.

**(2) Live pipeline positive control (pilot, 4 calibration room games).** K*_replay replays the stored same-game lineage's next 100 actions, with probability 0.5, at matched cells. It must give R ≥ 3 against V2 on at least 3 of 4 games. Failure is a build fix, not a kill.

**(3) Knowledge ceiling measured live (G-CEIL).** Two same-game oracles are run:
- K*_proc, the ceiling of the carried object class;
- K*_route, a content ceiling that also answers "does any knowledge help".

**(4) Ceiling-to-label rule, written now.** If the measured R(K*_proc:V2) < 2.0, the FASTER-2x labels are declared unreachable before any X run, and only PARTIAL is pursued. The capture ratio φ = log R(X:V2) / log R(K*_proc:V2) is reported.

This implements the v84 lesson: simulate the achievable ceiling with an oracle before trusting thresholds.

**Odds.** These odds are mine; the gates are correlated, so I multiply the product by about 1.5.

| Event | P |
|---|---|
| G-SIM passes | 0.85 |
| Pilot replay control works | 0.9 |
| ROUTE-ROOM (same-game route memory gives ≥ 2x) | 0.5 |
| PROC-WORTH (same-game posture prior gives ≥ 1.75x) | 0.2 |
| G-DEV GO, given worth | 0.25 (main risk: what transfers is the marginal best posture, which S and T already capture, giving TUNED-DEFAULTS) |
| Sealed measurable (≥ 6 of 17 units) | 0.9 |
| PARTIAL, given GO | 0.4 |
| 2x, given GO | 0.15 |
| **Overall sealed SEARCH-CLIMB FASTER-PARTIAL (search allocation)** | **about 2–3%** |
| **Overall 2x** | **about 0.7–1%** |

**Most likely ending (about 0.8):** KC-proc-worth at G-CEIL by about day 4, for about 0.5G billed frames. It would come with a clean answer on whether route knowledge can give 2x. That answer would close or redirect this object class, at this level of evidence, cheaply.

**Compute.** **Basis.** G1b ran at 14–22M billed frames/h on 3 workers. A run stopped at M_top costs about 2.5M; F runs go to 5M.

| Stage | Billed frames | Wall time |
|---|---|---|
| G-SIM | 0 | about 1 h CPU |
| G-BUILD and pilot | about 75M | about 5 h |
| G-CEIL | F 44 × 5M + V2 and K*_proc 88 × 2.5M + K*_route 22 × 2.2M ≈ 0.49G | 22–35 h |
| G-DEV (only if PROC-WORTH) | F sources on 19 other opened games 190M + X/T/S/P 176 × 2.5M + X_8 and U 110M ≈ 0.74G | 34–53 h |
| Sealed | B0 eligibility 17 × 2 × 4M = 136M + B1 about 10 units × 20 runs × 2.5M = 500M ≈ 0.64G | 29–46 h |
| **Total** | **≈ 1.95G** (caps: development 1.4G, sealed 0.8G) | about 4–6 days of compute; 10–12 calendar days with reviews if every gate passes |

**RAM.** V2 measured 0.23–0.56 GB peak commit per worker (G1a). The meta-controller adds less than 10 MB, and promise-row logging is off for the arms. 3 workers stay at or below 2.1 GB, with the supervisor guard at 14 GB free.

**GPU:** 0 h.

**Resumability.**
- A checkpoint every 1M frames saves the archive clone states, RNG states, bandits and ledger.
- On resume, the run plays one NOOP after reset before its first restore, reproducing the after-step ALE context (build note 15).
- Test: kill and resume at 3 points on 3 games, Q*bert included, and compare byte for byte with an uninterrupted run. If it fails, runs restart from zero as in v84. This is engineering, never a kill.

**Cousins.** All of this is from memory; a literature check is due before any novelty claim.

**Closest cousins:**
- **Agent57 (Badia et al. 2020).** A sliding-window UCB meta-controller picks among NGU's family of exploration/discount policies, fresh on each game; MEME (Kapturowski et al. 2022) extends it. These are the mechanism my posture bandit borrows. What I add is a cross-game, context-conditional prior on the bandit, and a test of that prior.
- **Go-Explore (Ecoffet et al. 2019, 2021).** Its hand-set selection weights and dynamic downscaling are V2's own. Its learned-policy, latent and intelligent variants (policy-based Go-Explore; Gallouédec & Dellandréa 2023; Lu, Hu & Clune 2024, which uses a foundation model and is not allowed here) do not carry search-allocation priors across games.
- **Online hyperparameter schedules:** PBT (Jaderberg 2017), meta-gradients (Xu et al. 2018), PB2 (Parker-Holder 2020).
- **Algorithm selection and configuration:** Rice 1976; SATzilla; DAC (Biedenkapp 2020; Adriaensen 2022).
- **Warm-started tuning:** Feurer 2015; Swersky 2013.
- **Meta-learned bandit priors:** meta-Thompson sampling (Kveton 2021); Basu 2021; Cella 2020.
- **Meta-learned exploration:** RL², E-RL², MAESN, meta-learned curiosity (Alet 2020), First-Explore.
- **Learned search and branching heuristics:** He, Daumé & Eisner 2014; Khalil 2016.
- **Human analogue:** strategy-selection learning and rational metareasoning (Lieder & Griffiths 2017; Russell & Wefald 1991).
- **Options and skills (Sutton, Precup & Singh 1999; option-critic; DIAYN)** carry action knowledge. This object deliberately does not; that becomes the next lever if both oracles fail.

**Honest novelty.** The algorithm pieces are not new. What is new is the test:
- search-allocation knowledge learned only from the agent's own savestate play, from pixels and its senses;
- a certified common prefix;
- a live same-game procedural ceiling and a route-memory positive control, run before any carried arm;
- pairwise controls that separate transfer from both kinds of tuned defaults (best constant S, tuned schedule T) and from context-swapped and label-permuted placebos;
- matched-rung climbs on never-opened games.

# v85: Carry how to search, measured live

**Editor's note.**
- Read and grep only. Sources: the brief; v84 protocol §0–13 and Part C; build notes 1–20; `room_verdict_v84.json`; `g1a_report.json`; `explorer2.py`, `framecode.py`, `v2_job_v84.py`, `room_verdict_v84.py`; the head of `bars_v84.py`; `pools_v83.py`; the last handoff sections; `ARC4_VISION.md` §0–1 and §2.6.
- One slip, on record: a shell pipeline started `python -` on an empty here-document. No code ran, nothing was imported or printed. No emulator, GPU job or edit.
- Seed block 9850–9899: the `*.py` grep is clean in `ragnarok` and `ragnarok_v84`. The 9-worktree grep is due at freeze.

## 1. Thesis
**V2 is the asset, and its yardstick is precise.** Its matched-rung climb varies little between seeds (σ̂² = 0.376 per run). v84 never ran a carried object live. Both of its routes died on an offline clock capped near 1.2–1.6×.

**I carry how to search, not what a game is.**
- Consequence content did not transfer (AUC 0.52).
- Budget allocation, cell coarseness and rollout length are game-agnostic by construction.

**Which knobs to carry.** The vision's law (§1.1) says reuse pays only for what is expensive to reacquire.
- V2's process bandit is cheap to reacquire: it gets feedback every expansion and its half-life is about 140 expansions. A process prior would be spent before M1, so it would only be a head start. I report the process-mix settling time from G1b `exp_meta`.
- Slow knobs are expensive. They get one feedback per 100k-frame epoch, and G1b leaves only about 2–33 epochs between M1 and M_top. A fresh learner cannot sort nine settings in that window.
- A prior therefore acts exactly where the yardstick measures.

**Every claim concerns a search with billed rewinds.** Nothing here tests live play without savestates.

## 2. Mechanism
**Postures.** Four slow V2 knobs, each with a low and a high level around V2's value:

| Knob | Low / high | V2 |
|---|---|---|
| A: resolution target c in clip(c√n, 64, 8000) | 12 / 75 | 30 |
| B: rollout (base, ext, max) | (50, 25, 200) / (200, 100, 800) | (100, 50, 400) |
| C: frontier share | 0.10 / 0.50 | 0.25 |
| D: depth exponent β in 2^(β·(sbin − sbin_max)) | 0.25 / 1.0 | 0.5 |

There are nine postures: the 2^(4−1) fraction (D = ABC) plus P0 = V2. Everything else is V2's.

**Meta-controller.**
- An epoch is 100k billed frames.
- **Common prefix:** every non-V2 arm, oracles included, plays P0 for epochs 0–1. This prefix is byte-identical to V2 (certified). On both G1b seeds, 14/22 room games reached M1 before 200k frames, so on those games the arms share M1 exactly.
- From epoch 2, a Normal Thompson bandit picks a posture: discount 0.9 per epoch, 10% uniform floor, propensity logged.
- Reward y = the rise of the archive-best sensed s_bin during the epoch. Every arm senses this quantity.
- A resolution change takes effect at V2's own re-key points.

**Context c** (agent-side, frozen, computed over the last epoch):
- log billed frames;
- best s_bin;
- ADV rate;
- life losses per rollout;
- sensed rises per 1k steps;
- new-cell rate;
- distinct coarse codes per cell;
- realised/planned rollout length;
- adapter signature.

**Carried object X.**
- For each posture p, a ridge regression m_p(c) of y.
- It is fitted on the epoch logs of F runs (the flat-prior meta-bandit) on source games, with equal game weight. The ridge is chosen by leave-one-game-out (LOGO) validation.
- On a new game, m_p(c_t) is the prior mean.
- Its strength is n0 = noise variance / LOGO prediction-error variance, clipped to [2, 20]. It is computed per arm from sources only, with no live tuning.
- X is a hashed numeric table: no game names, no RAM, byte-identical under permutation of the source order.

## 3. Yardstick and ceiling checks, before freezing
**Yardstick (as in v84 §7.2–7.4).**
- λ over k = 2..k\*, counted from each arm's own M1.
- λ_late, counted from M2.
- λ_abs, counted from frame 0.
- k\* and the bars come from G1b. Replays follow the rule of build note 16.
- Cap of 5M frames for every arm, with an evaluator stop at M_top. F runs on to 5M because its epochs are source data; this does not change its λ.

**Contrasts are pairwise**, combined in an intersection-union test: each lower bound (LB) must be > 1. I drop v84's per-game "fastest of {V2, T, S}". Taking the minimum over noisy arms biases log R against X by about 0.25–0.3.

**1. G-SIM** (0 frames, about 1 h of CPU, before freeze), on the 44 G1b runs:
- (a) a seed-vs-seed null gives the instrument line;
- (b) planted 2× (halve one seed's T(M_k) − T(M1)): the recovered R must lie in [1.8, 2.2];
- (c) planted head start (shift T(M1) only): R must be 1.00 ± 0.05 while HS = 2;
- (d) power, by resampling λ residuals with heterogeneity τ ∈ {0, 0.2, 0.4} and σ² ×1.5 for bandit arms.

Frozen rule: if G-CEIL has power < 0.75 at a true 2×, or a false-pass rate > 0.10 at 1.25× (τ = 0.2), every G-CEIL arm gets 3 seeds before freeze. By hand, the power is about 0.8 at 2×, 0.15 at 1.5× and 0.01 at 1.25×.

**2. Pipeline positive control (pilot, exploratory): K\*_replay** on the four calibration room games.
- At a cell matching the same game's best G1b lineage (s_bin and coarse code), the rollout replays the lineage's next 100 actions with probability 0.5.
- It must give R ≥ 3 against V2 on at least 3 of 4 games.
- A miss is a build fix, never a kill.

**3. Knowledge ceiling = G-CEIL** (§5): live, with two same-game oracles.

**4. Ceiling-to-label rule (frozen now).** If the measured R(K\*_proc:V2) < 2.0, the 2× labels are declared out of reach before any X run.

## 4. Arms
| Arm | Definition | Separates |
|---|---|---|
| V2 | v84 constants | incumbent |
| F | meta-bandit, flat prior | fresh posture learner; source data |
| S | single best posture (equal-game mean y on sources; LOGO in development), fixed, no bandit | best constant |
| T | meta-bandit; prior = posture × 3 clock bins (< 0.5M, 0.5–1.5M, > 1.5M), shrunk to the posture mean | tuned schedule, the same on every game |
| X | meta-bandit; prior m_p(c) | carried, game-conditional |
| P | X's recipe fitted on sources whose context sequences are swapped between games (derangement, seed 9859) | same capacity and marginals, broken content |
| U (report, development, 1 seed) | X's prior with the posture labels permuted per game | confidence without content |
| X_8 | X from 8 source games (draw 9860) | dose caption |
| K\*_proc | T-shaped table built from the same game's own F runs (seeds A, B), plus its G1b runs counted as P0 epochs | same-game procedural ceiling |
| K\*_route | V2, with ×8 selection weight on cells whose (s_bin, coarse code) lie on the same game's best G1b lineage. The codes come from an evaluator replay of the agent's own frames; development only | same-game content ceiling; live positive control |

**Separating transfer from tuned defaults:**
- S beating V2 means V2's constants were wrong.
- T beating V2 means a better schedule, the same on every game.
- P ≈ X means that any game-varying prior helps.
- Only X beating all of V2, F, S, T and P is transfer.

## 5. Gates and kill criteria
| Gate | Content | Pass | Kill (sentence) |
|---|---|---|---|
| G-SIM | §3.1 | the rules hold | redesign before freeze |
| G-BUILD | P0 equals V2 byte for byte; determinism; resume identity; isolation | tests pass | KC-build after 2 fix rounds |
| Pilot | calibration room games, seeds 9857–9858 | exploratory | build fixes only |
| **G-CEIL** (decisive) | 22 room games. F on seeds A, B; V2 and K\*_proc on C, D; K\*_route on C | **PROC-WORTH:** R(K\*_proc:V2) ≥ 1.75 with LB ≥ 1.25; LB of R_late > 1; r_g ≥ 1.5 on ≥ 1/3 of units. **ROUTE-ROOM** (answers "any knowledge"): R(K\*_route:V2) ≥ 2.0 with LB ≥ 1.3 | **KC-proc-worth:** "With two runs of same-game evidence, posture knowledge does not make V2 climb 1.75× faster; carrying it across games cannot reach 2×." v85 ends. Flags: KC-harm if R(F:V2) < 0.8; ROUTE-WEAK if lineage-code coverage < 30% |
| G-DEV (single shot) | X, T, S, P on C, D; U on C; F and V2 reused | **GO:** R(X:V2) ≥ 1.4 with LB > 1; LBs of R(X:T), R(X:S), R(X:P) and R(X:F) > 1; LB of R_late(X:V2) > 1; point R_abs ≥ 1 | **KC-dev**, with the TUNED-DEFAULTS caption if the LB of R(S:V2) or of R(T:V2) is > 1 |
| Freeze | 3-agent review; hashes; power with the measured σ² | — | — |
| B0 | 17 non-hard sealed games: bars 9870; V2 on 9871–9872 at 4M; G1b room rule | ≥ 6 units (if more than 10, draw 10 with 9880) | one second round; otherwise NOT MEASURABLE |
| B1 (once) | X and F: 4 seeds each; T, S, P and V2: 3 seeds each | sealed labels below | — |

**Instrument line.** V2 C vs D in development; F 2 seeds vs 2 seeds on the sealed games. The result is INCONCLUSIVE if any of these holds:
- the line's 99% interval excludes 1;
- ≥ 40% of runs are censored;
- the prefix certificate is violated.

**Sealed labels.** The headers state SEARCH WITH BILLED REWINDS and the senses.
- **FASTER-2× (search allocation).** All of:
  - R(X:V2) ≥ 2.0 with LB ≥ 1.3;
  - R(X:T) and R(X:S) ≥ 1.4, each with LB > 1;
  - LBs of R(X:P) and R(X:F) > 1;
  - R_late(X:V2) ≥ 1.5 with LB > 1;
  - LB of R_abs ≥ 0.9;
  - r_g > 1 on ≥ 2/3 of units;
  - LOGO minimum of R(X:T) > 1;
  - p ≤ 0.025 for X against T.
- **PARTIAL.** Every LB > 1, p ≤ 0.025, and point R_abs ≥ 1.
- **TUNED-DEFAULTS.** S or T beats V2, and X fails its T or S clause. Sentence: "Earlier games tuned the search; a human tuning on them gets the same setting; this is not faster learning."
- **Then:** GENERIC-ADAPTATION, EARLY-RUNGS-ONLY, HEAD-START ONLY (never called faster), NEGATIVE, NULL.

## 6. Pools and seeds
**Pools.**
- Development units: the 22 G1b room games.
- Pilot: pong, breakout, seaquest, asteroids.
- Sources: F runs on all 41 opened games (seeds A, B).
- Sealed: the 17 non-hard games. **The 7 hard-exploration sealed games stay unopened**, kept for §9.
- Second round: top-up only.

**Seeds.**

| Seed | Use |
|---|---|
| 9850 | smoke runs |
| 9851 | G-SIM |
| 9852–9853 | A, B |
| 9854–9855 | C, D |
| 9856 | development bootstrap and permutation |
| 9857–9858 | pilot |
| 9859–9860 | placebo and dose draws |
| 9870 | sealed bars |
| 9871–9872 | sealed eligibility |
| 9873–9878 | sealed arms |
| 9879 | sealed bootstrap and p |
| 9880 | sealed unit draw |
| 9890–9898 | second round |
| Rest of 9850–9899 | reserve |

## 7. Compute and RAM
**Basis.** G1b ran at 14–22M billed frames/h on 3 workers. A run stopped at M_top costs about 2.5M.

| Stage | Billed frames | Wall time |
|---|---|---|
| G-SIM | 0 | about 1 h CPU |
| G-BUILD and pilot | about 75M | about 5 h |
| **G-CEIL** | F 44 × 5M + V2 and K\*_proc 88 × 2.5M + K\*_route 22 × 2.2M ≈ 0.49G | 22–35 h; verdict about day 4 |
| G-DEV | sources 190M + 176 × 2.5M + X_8 and U 110M ≈ 0.74G | 34–53 h |
| Sealed | B0 136M + B1 ≈ 200 × 2.5M ≈ 0.64G | 29–46 h |
| **Total** | **≈ 1.95G** (caps: development 1.4G, sealed 0.8G) | about 4–6 days of compute; 10–12 days with reviews |

**RAM and GPU.**
- V2 measured 0.23–0.56 GB per worker. The meta-controller adds less than 10 MB, and promise rows are off for the arms.
- 3 workers use ≤ 2.1 GB; the supervisor guard stays at 14 GB free.
- GPU: 0 h.

**Resume.**
- A checkpoint every 1M frames saves the archive states, RNGs, bandits and ledger.
- On resume, the run plays one NOOP after reset before its first restore (build note 15).
- Test: a kill and resume at 3 points on 3 games, Q\*bert included, must reproduce the uninterrupted JSON byte for byte. If it does not, runs restart from zero as in v84. This is engineering, never a kill.

## 8. Odds (mine)
| Event | P |
|---|---|
| G-SIM passes | 0.85 |
| Pilot control works | 0.9 |
| ROUTE-ROOM | 0.5 |
| PROC-WORTH | 0.2 |
| GO, given PROC-WORTH | 0.25 (main risk: what transfers is the marginal best posture, which S and T already hold) |
| Sealed measurable | 0.9 |
| PARTIAL, given GO | 0.4 |
| 2×, given GO | 0.15 |
| **Overall: sealed transfer PARTIAL** | **about 2–3%** |
| **Overall: sealed 2×** | **about 0.7–1%** |

**Most likely ending (about 0.8):** KC-proc-worth by day 4, for about 0.5G, together with a ROUTE answer. A KC-proc-worth closes this object class at this level of evidence, not every procedural carry.

## 9. If no
**ROUTE yes, PROC no.** Knowledge matters to this climber, but it is "where", not "how". Content did not cross unrelated games (v83). The next test should therefore stay within families where content is shared: the second-round near-siblings first, then NES.

**Both no.** Where fresh search already climbs, V2 spends its frames on random rollouts that find the next rise. Neither knowing where nor knowing how to search buys 2×. Knowledge can then matter only where fresh search fails, as it does for humans. Two options:
- **(i) A hard-exploration substrate.** The 13 non-room development games serve for development and the 7 unopened sealed hard games as the test. The yardstick is the rungs reached within budget, plus matched-rung times where both arms reach them.
- **(ii) A learner whose knowledge enters action generation:** cross-game motor priors or options in place of the four random processes.

## 10. Novelty (from memory; a literature check is due)
**Not new:**
- the posture bandit (Agent57's meta-controller over NGU policies; MEME);
- online schedules (PBT, meta-gradients, PB2);
- algorithm configuration (Rice; SATzilla; DAC);
- warm-started tuning (Feurer 2015; Swersky 2013);
- meta-learned bandit priors (Kveton 2021; Basu 2021);
- Go-Explore's selection weights and downscaling (Ecoffet 2019, 2021), and its policy, latent and LLM variants (Gallouédec 2023; Lu, Hu & Clune 2024);
- meta-learned exploration (RL², E-RL², MAESN, Alet 2020, First-Explore);
- learned search heuristics (He et al. 2014; Khalil 2016);
- human strategy-selection learning (Lieder & Griffiths 2017).

Options and skills carry action knowledge; this object does not.

**New is the test:**
- search-allocation knowledge learned from the agent's own savestate play;
- a certified common prefix;
- a live same-game ceiling and a route-memory positive control, run before any carried arm;
- pairwise controls that separate transfer from both tuned defaults (a constant and a schedule) and from content placebos;
- matched rungs on never-opened games.

---

## Proposal 5: v85: second attempt, then learning sets. Knowledge measured live from a savestate shared by all arms

**One line.** First, a cheap live test: start every arm from the same savestate and ask whether even same-game oracle knowledge short of a demonstration doubles V2's climb. "Where" is the screens along the game's future best path; "how" is the game's own future motor patterns, used unindexed. Only if "how" has room, carry a cross-game motor vocabulary mined by hindsight (a fifth bandit process) to the sealed games, and show a learning-set curve over 0/4/12/41 source games against designer macros, within-game mining, the best single process and a non-advancing placebo.

**First gate.** G1, the knowledge room, run live from a savestate shared by all arms (development pool, about 150M billed frames, about 10 h, verdict by day 3).

Units: 10 of the 16 G1b room dev games with k* ≥ 4, drawn with seed 9800. Every arm starts from s1, the state at M1 on the G1b run on seed 9712 (run A), with an empty archive and the same entry; the clock runs from s1. Seeds 9801-9804, cap 4M, the evaluator stops each run at M_top.

Arms:
- V2, with no knowledge.
- D, the planted control: replays run A's own lineage.
- K*map ("where" oracle): run A's future best-lineage coarse codes steer cell selection with probability 0.5.
- K*mac ("how" oracle): an unindexed motor vocabulary mined from run A's own rollouts above M1, with chunks that start at s1 excluded.
The oracles use the agent's own future pixels and buttons, never RAM, and carry no claim.

Yardstick: λ = mean over k=2..k* of log max(5000, T(M_k)) from s1; λ_late from each arm's own T(M2); λ_abs; two-level bootstrap.

Labels, per oracle:
- ROOM: point R ≥ 1.75, lower bound ≥ 1.25, late lower bound > 1, and r_g ≥ 1.5 on at least 4/10 units.
- NO-ROOM: 90% upper bound < 2.
- INCONCLUSIVE otherwise.

Outcomes:
- ROOM-MAC: go to the cross-game motor vocabulary (G2).
- ROOM-MAP only: v85 ends with a room result; designing a carried object for "where" is left to v86.
- NO-ROOM: knowledge short of a demonstration does not double this climber. Pivot to stall savestates, the 7 unopened hard-exploration games, or live play.
- INCONCLUSIVE: stop.

**Ceiling check.** G0, before any threshold is frozen (0 billed frames).
(1) Physics floor from the G1b records (billed, cum, depth): the frames a perfect searcher would need above M1 (4 per lineage step plus 60 per 400-step rollout), against V2's 0.1-3.3M. The median floor ratio must be at least 8.
(2) Power and false-positive simulation of the frozen G1, G2 and B1 rules from the 22 room games' seed pairs (sigma2 = 0.376 winsorised), with planted ratios 1, 1.5, 2 and 3 and heterogeneity tau in {0, 0.3, 0.5}. At tau = 0.3 it requires P(ROOM | 2x) ≥ 0.75 and P(ROOM | 1x) ≤ 0.05. If either fails, seeds go from 4 to 6 (the only change allowed), and thresholds are set below the target effect so they are not a coin flip at it.
(3) s1 is rebuilt by exact replay in the ALE context that reproduces the recorded rise (note 16), otherwise the next unit is drawn.
(4) Live planted control D, a demonstration replay, whose expected outcome is declared before running: R(D:V2) ≥ 4 on at least 8/10 units with replay-confirmed rungs, otherwise an Interruption with one fix. Instrument line: V2 against V2 on two seed blocks, INCONCLUSIVE if its 99% interval excludes 1.

**Odds.** Mine.
- G0 passes (possibly needing 6 seeds): 0.9. Demonstration control works: 0.9.
- G1: ROOM-MAC 0.3; ROOM-MAP 0.3; any room 0.45; NO-ROOM 0.3; INCONCLUSIVE 0.25.
- GO given ROOM-MAC: 0.25. Sealed pool measurable (6 or more units): 0.8. FASTER-PARTIAL or better given GO: 0.4; 2x given GO: 0.15.
- Overall: sealed FASTER-PARTIAL or better about 2.5%; sealed SEARCH-CLIMB FASTER-2x about 1%; ACCUMULATES caption below 1%. v84 wrote 0.5% and 0.15%.
- Clean G1 answer by day 3: about 0.75.
- Most likely ending: the demonstration shows a large gain while the "where" and "how" oracles stay below 2x or are inconclusive. That is a cheap, decisive statement that knowledge short of a demonstration does not double this savestate climber, and it points to stall savestates, hard-exploration games or live play.

**Compute.** Billed frames, with wall times at G1b's realised rate of about 16M frames/h on 3 workers:
- G0: 0 frames, about 2 h CPU.
- G1: 160 runs, about 150M, about 10 h.
- G2: 240 runs, about 290M, about 18 h.
- B0: bars plus 34 eligibility runs on 17 sealed games, about 140M, about 9 h.
- B1: about 290 runs, about 350M, about 22 h.
- Total: about 0.93G billed, about 60 h of machine time. G1 verdict by day 3; 8-10 days for a chain that stays alive.
- Code: about 1.5 days (start from s1 in SenseEnv84, Q5 process, miner, map and demo steering, verdict). The G-build tests are isolation, prefix identity with Q5 inactive, determinism and start-state replay.
- RAM: at most 0.6 GB per worker (G1a peak commit was 0.23-0.56 GB; the library is under 10 MB), so 3 workers stay at or under 1.8 GB. The supervisor guard stays at 14 GB free, and runs are resumable by deterministic rerun from zero.
- GPU: 0 h.
- Sources for the mined library come from the existing G1b logs (rows, rise lags, ADV flags), so they cost no new frames.

**Cousins.** Go-Explore family:
- Go-Explore (Ecoffet et al. 2019, 2021), the ancestor of V2 and its adaptive downscaling.
- Policy-based Go-Explore.
- Latent Go-Explore (Gallouédec & Dellandréa 2023), which learns cells per game.
- Intelligent Go-Explore, which uses a foundation model and is not allowed here.

Starting from shared or demonstration states, the cousins of the shared-savestate yardstick:
- Salimans & Chen 2018.
- The backward robustification in Go-Explore.
- Mu-resets (Kakade & Langford 2002).

Exploration-policy cousins of the arms:
- Agent57 and NGU (Badia et al. 2020): a bandit meta-controller over exploration policies. V2's process bandit is a small version, and Q5 adds a learned policy family to it.
- Temporally extended epsilon-greedy (Dabney et al. 2021): the designer-macro arm and Q2-Q4.

Macros and skills:
- McGovern & Sutton 1998.
- Durugkar et al. 2016.
- Chang et al. 2019, macros built on one Atari game and reused elsewhere: the closest prior work.
- Rosman & Ramamoorthy 2012, learned action priors across tasks.
- Lange & Faisal 2019, action grammars.
- Options (Sutton, Precup & Singh 1999; Bacon et al. 2017) and DIAYN.

Meta-learned exploration: MAESN (Gupta et al. 2018, structured noise across tasks), E-RL2, First-Explore and Alet et al. 2020.

Learning sets and human priors: Harlow 1949; Dubey et al. 2018; Tsividis et al. 2017.

Honest novelty: the mechanism is not new. What is new is the test: a live decomposition of a demonstration's value into "where" and "how" from shared savestates, used as an a-priori ceiling, then a learning-set curve on never-opened games against designer macros, within-game mining, the best single process and a non-advancing placebo. The agent's inputs contain no RAM, and no downloaded weights are used. These references are from memory; a literature check is due before any novelty claim.

# v85: second attempt, then learning sets (wildcard proposal)

**Claim header.** Every v85 claim concerns a SEARCH WITH BILLED REWINDS started from a savestate shared by all arms. It uses the senses granted on 22 October (score rises, lives). It is not live play, except the labelled report-only probe in §7. Sources read: the brief, the v84 protocol §5–§10, build notes 1–20, `room_verdict_v84.json`, `g1a_report.json`, `explorer2.py`, `senses_v84.py`, `rowlog.py`, `v2_job_v84.py`, the handoff, and ARC4_VISION §0–§1 and §4. I only read files and ran read-only greps; I ran no Python.

## 1. Thesis
- v84 left us a climber, V2 (M3 on 22/35 dev games), but no knowledge was ever run live.
- The offline clock could not show 2× because a low-capacity fresh learner reacquires knowledge almost at once. The same arithmetic holds live. Knowledge doubles a climb only if two things hold:
  - it changes the search a lot;
  - a fresh agent cannot reacquire it during the climb.
- V2's own learnable state is reacquired within ~200 expansions (a 4-arm bandit with discount 0.995, plus a resolution). So the first question is not "does object X transfer?" but "does any knowledge short of a demonstration double this climb at all?".
- **Reframed yardstick 1: shared-savestate segments.** Every arm starts at the same savestate s₁, the state at rung M1 on a reference V2 run. Each arm starts with an empty archive and the identical entry (body, adapter, me tracker; median entry 135k frames, off the clock). Head start in states, archive, perception and entry is zero by construction. What is measured is the climb above M1.
- **Reframed yardstick 2: a learning-set curve** (Harlow) for the transfer claim. It plots the frames to climb never-opened games against the number of earlier games the carried object was mined from (0, 4, 12, 41). Random subsets stand in for random orders.
  - A "better default" stops improving after a few games; accumulation keeps falling.
  - Tuned defaults and the placebo appear as flat lines on the same plot.
- **My bet: a motor vocabulary.** It is a set of temporally extended patterns over V2's canonical slots (movement class × extra bit, learned from pixels at entry). It is mined by hindsight from the agent's own advancing rollouts and offered as a fifth bandit process.
  - It is the one kind of knowledge V2 cannot reacquire within a game: V2 has no way to invent processes.
  - ARC4_VISION §1.1 names motor skills among the expensive shared things.
  - The bandit limits negative transfer to roughly the 20% floor's share of expansions.

## 2. G0: ceiling checks before freezing (0 billed frames, about 2 h CPU)
1. **Physics floor.** From the G1b records (billed, cum, depth), compute per unit the frames a perfect searcher needs above M1: 4·Δdepth plus 60 per 400-step rollout. Compare with V2's frames above M1.
   - On room games V2 spends 0.1–3.3M frames above M1; lineages are 430–9,700 steps deep at M3.
   - Pass if the median floor ratio is ≥ 8.
2. **Power and false-positive simulation** of the frozen G1, G2 and B1 rules.
   - Inputs: the 22 room games' seed pairs (σ̂² = 0.376 winsorised; within-game seed gaps up to 2.55 log units, carnival), resampled seed differences, heterogeneity τ ∈ {0, 0.3, 0.5}, planted ratios 1, 1.5, 2 and 3.
   - Required at τ = 0.3: P(ROOM | 2×) ≥ 0.75 and P(ROOM | 1×) ≤ 0.05.
   - If missed, seeds go from 4 to 6, the only change allowed, decided before any G1 frame.
   - Thresholds sit deliberately below the target effect: v84's thresholds placed exactly at the target gave about 50% power.
3. **Rebuilding s₁.** Replay the recorded actions from the base state evaluator-side, in whichever ALE context reproduces the recorded sensed rise exactly (note 16). A unit that does not reproduce is replaced by the next draw.
4. **Live planted control** (outcome declared now). Arm D replays run A's own lineage from s₁. Expected: R(D:V2) ≥ 4 on ≥ 8/10 units, with replay-confirmed rungs. Otherwise the instrument is defective: Interruption, one fix allowed.

## 3. G1: knowledge room from a shared savestate (dev, live, the decisive gate)
**Units.** 10 of the 16 room dev games with k\* ≥ 4 (so ≥ 2 late rungs), drawn with seed 9800. s₁ comes from the G1b run on seed 9712 (run A). Bars (9706), the ladder and k\* are as in v84.

**Arms.** Seeds 9801–9804; cap 4M billed frames from s₁; the evaluator stops a run at M_top.
- **V2**: no knowledge.
- **D**: the demonstration (where, how and when).
- **K\*map ("where", oracle)**: the coarse codes (8×6×4) along run A's best lineage above s₁, in order. With probability 0.5 per expansion, select the archive cell that matches (exactly, else nearest in L1) the furthest-along map code; otherwise use V2's own draw.
- **K\*mac ("how", oracle)**: an unindexed vocabulary built by the §4 miner from run A's own rollouts above M1, i.e. the same game, the same segment, its future. Chunks that started at s₁ are excluded.
- The oracles use the agent's own future pixels and buttons (no RAM). They are experimenter-side references, never carried, and carry no claim.

**Yardstick.**
- λ = mean over k = 2..k\* of log max(5,000, T(M_k)), with T in billed frames from s₁.
- λ_late = the same over k = 3..k\*, counted from each arm's own T(M₂).
- λ_abs = log T(M_top).
- A missed rung counts the cap for every arm. A sensitivity line counts carried-arm misses at twice the cap.
- R = exp(mean over games of [λ̄_V2 − λ̄_K]), with a two-level bootstrap (games, then seeds), 2,000 resamples, seed 9805.
- Instrument line: V2 on 9801–02 against V2 on 9803–04; INCONCLUSIVE if its 99% interval excludes 1.

**Labels (per oracle, captioned "with these oracle rules").**
- **ROOM-K**: point R ≥ 1.75, 90% lower bound ≥ 1.25, lower bound of R_late > 1, and r_g ≥ 1.5 on ≥ 4/10 units.
- **NO-ROOM-K**: 90% upper bound < 2.
- **INCONCLUSIVE-K**: anything else.
- Reported: log R_K / log R_D, the share of a demonstration's value that is "where" and the share that is "how".

**Consequences.**
- **ROOM-MAC** → G2.
- **ROOM-MAP only** → v85 ends on a positive room result. A carried "where" object (a learned progress representation) is v86 design work and is not improvised here.
- **INCONCLUSIVE** → stop.
- **NO-ROOM** → v85 ends with this sentence: "From the same savestate, neither the future path's screens nor this game's own future motor patterns make the climber 2× faster; only a full demonstration does."
  - Meaning: on room games, a savestate searcher's speed is set by the physics of the search and by luck, not by knowledge, so cross-game transfer cannot give 2× here.
  - Substrates where knowledge would matter, for Jérémie to choose:
    - (a) stall savestates from the plateaus that end every G1b run, scored as P(climb one rung within 2M) from the shared stall state;
    - (b) the 7 hard-exploration sealed games, deliberately kept unopened for this;
    - (c) live play without rewinds, where knowledge is the only lever.

## 4. The carried object: a mined motor vocabulary (frozen now)
- **Macro.** A run-length sequence of canonical slots, 8–64 steps in multiples of 4; for example, move-right ×12 then right+extra ×4. On a new game, an unavailable slot falls back to its movement class without extra, else to NOOP.
- **Data.** Agent-side only: its own slots (the rows log one every 4th step), its own sensed rises (`rise_lag`), and its own ADV flags (`exp_meta`, joined on `it`). All of these already exist for the 82 G1b runs.
- **Miner.** Candidates are the 32 steps before each sensed rise inside an advancing rollout.
  - Lift = (ADV count + 1) / (count in 32-step windows of non-advancing rollouts + 1), with equal game weight.
  - The library is the top 32 by lift, with edit distance ≥ 3 between macros and at most 4 macros from any one source game.
- **Q5.** A fifth process in V2's bandit. It draws macros by Thompson sampling (Beta prior from the lift, n₀ = 10) and chains them up to the rollout length.
  - With Q5 inactive, the run is V2 byte for byte (prefix-identity test).
  - The library's mean is not zero, so head start is controlled by the shared start and the late clause, not by a certificate.

## 5. G2: development rehearsal (single shot, only after ROOM-MAC)
Same units, s₁ and seeds; V2's runs are reused.

**Arms:**
- **X**: library mined from the 40 other opened games, leave-one-game-out.
- **W**: the same miner run on this game's own rollouts, refit every 250 expansions (the no-carry learner).
- **DM**: a designer library, frozen now. 32 macros: holds of 8 and 32 steps in 8 directions, the same with extra, extra bursts, left-right and up-down alternations, diagonal zigzags. This is the tuned default, a temporally extended ε-greedy.
- **S**: the single best process (leave-one-game-out).
- **P**: a placebo library mined from non-advancing windows of the same 40 games, matched on length and slot frequency (seed 9807).
- **X₄**: dose arm, 4 source games, 2 draws × 2 seeds (seed 9810).

**Rule.** N = the per-game, per-resample best of {V2, DM, W, S}.
- **GO** requires all of:
  - R(X:V2) ≥ 1.5 with lower bound > 1.1;
  - R(X:N) ≥ 1.25 with lower bound > 1;
  - lower bound of R(X:P) > 1;
  - lower bound of R_late(X:N) > 1;
  - point R_abs(X:N) ≥ 1.
- Otherwise **KC-dev**: "the cross-game vocabulary does not beat designer macros, within-game mining and the best process on development segments."
- Reported: capture φ = log R(X:V2) / log R(K\*mac:V2).

## 6. Sealed test (once)
**Before it.** A 3-agent review and a freeze that commits the hashes of the libraries, the code and the power simulation.

**B0 (eligibility).**
- Bars on 9820, then V2 on 9821–9822 (cap 4M), on the 17 sealed games that are not hard-exploration.
- Unit rule: the G1b room rule plus k\* ≥ 4. s₁ comes from 9821.
- At most 8 units (drawn with 9830 if more).
- Fewer than 6: the second round runs once (9840–9848). Still fewer than 6: NOT-MEASURABLE.

**B1 (arms, seeds 9823–9828).**
- X₄₁ and V2: 6 seeds each.
- DM, W, S, P and X₁₂: 4 seeds each.
- X₄: 2 draws × 2 seeds.
- On pacman and frogger the libraries exclude ms_pacman and freeway, and a "without near-siblings" line is reported.

**Labels.**
- **SEARCH-CLIMB FASTER-2× (carried vocabulary)** requires all of:
  - R(X:V2) ≥ 2 with lower bound ≥ 1.4;
  - R(X:N) ≥ 1.4 with lower bound > 1;
  - lower bound of R(X:P) > 1;
  - lower bound of R_late(X:N) > 1;
  - point R_abs(X:N) ≥ 1.
- **FASTER-PARTIAL**: R(X:N) ≥ 1.2 with lower bound > 1, lower bound of R(X:P) > 1, lower bound of R_late > 1.
- **GENERIC-VOCABULARY**: X beats V2 but not N.
- **NEGATIVE-TRANSFER**: upper bound of R(X:V2) < 1.
- **ACCUMULATES** (caption): Page's L test over the doses 0, 4, 12, 41 with p ≤ 0.05, and lower bound of R(X₄₁:X₄) > 1.
- Headline figure: the learning-set curve, with DM, W and P as flat lines.

**Limits.**
- The dose sources come from unsteered V2 play. That is a declared stand-in for a true sequential lineage.
- Reaching M1 is not measured.

## 7. Live probe (report-only, not a claim)
On the sealed units, without savestates: an online bandit over X₄₁'s library, against DM's library and the B′ bars, 20 episodes of 7,000 steps each. It would be the project's first live-play number for a carried object.

## 8. Pools and seeds
- **Development and calibration (35 + 6, burned):** G1 and G2 units, plus the mining sources (existing logs, no new source frames).
- **Sealed:** 17 games, opened only at B0. The 7 hard-exploration games stay sealed. The 9 second-round games are only a top-up.
- **Reserved games:** never opened. **Enduro:** excluded.
- **Seeds:**

| Seed | Use |
|---|---|
| 9800 | Unit draw |
| 9801–9804 | Development arms |
| 9805 | Development bootstrap |
| 9806 | Power simulation |
| 9807 | Placebo |
| 9810 | Dose draws |
| 9820 / 9821–22 / 9823–28 / 9829 / 9830 | Sealed bars / eligibility / arms / bootstrap / unit draw |
| 9840 / 9841–42 / 9843–48 | Second round: bars / eligibility / arms |

- No 98xx seed appears in the tracked `.py` files of ragnarok, ragnarok_v83 or ragnarok_v84; the grep is repeated over every worktree at the freeze.

## 9. Compute
Wall times assume G1b's realised rate of about 16M billed frames per hour on 3 workers.

| Stage | Runs | Billed frames | Wall |
|---|---|---|---|
| G0 | — | 0 | about 2 h CPU |
| G1 | 160 | about 150M | about 10 h |
| G2 | 240 | about 290M | about 18 h |
| B0 | 34 + bars | about 140M | about 9 h |
| B1 | about 290 | about 350M | about 22 h |
| **Total** | | **about 0.93G** | **about 60 h of machine time** |

- G1 verdict by day 3. A chain that survives every gate ends in 8–10 days.
- **Memory:** ≤ 0.6 GB per worker (G1a peak commit 0.23–0.56 GB), so 3 workers stay ≤ 1.8 GB. Supervisor guard at 14 GB free; deterministic restarts from zero.
- **GPU:** 0.
- **Code:** about 1.5 days (start from s₁, Q5, miner, map/demo steering, verdict). Tests: isolation, prefix identity, determinism, start-state replay.

## 10. Odds (mine)
| Event | P |
|---|---|
| G0 passes (possibly with 6 seeds) | 0.9 |
| D control works | 0.9 |
| ROOM-MAC | 0.3 |
| ROOM-MAP | 0.3 |
| Any room | 0.45 |
| NO-ROOM | 0.3 |
| INCONCLUSIVE | 0.25 |
| GO, given ROOM-MAC | 0.25 |
| Sealed measurable | 0.8 |
| PARTIAL or better, given GO | 0.4 |
| **Sealed FASTER-PARTIAL or better, overall** | **about 2.5%** |
| **Sealed 2×, overall** | **about 1%** |
| ACCUMULATES | < 1% |

**Most likely ending:** a cheap, decisive G1. The demonstration shows a huge gain; the "where" and "how" oracles fall short of 2×. That says knowledge short of a demonstration does not double this climber, and it points to stall states.

## 11. Cousins and honest novelty (from memory; a literature check is due before any claim)
- **Go-Explore family:**
  - Go-Explore (Ecoffet et al. 2019, 2021): V2 descends from it.
  - Policy-based Go-Explore.
  - Latent Go-Explore (Gallouédec & Dellandréa 2023): learns cells per game.
  - Intelligent Go-Explore: uses a foundation model, which is not allowed here.
- **Starting from shared or demonstration states:** Salimans & Chen 2018; Go-Explore's backward robustification; μ-resets (Kakade & Langford 2002).
- **Agent57 and NGU:** a bandit meta-controller over exploration policies, of which V2's process bandit is a small cousin.
- **Temporally extended ε-greedy** (Dabney et al. 2021): the DM arm and V2's held processes Q2–Q4.
- **Macros and skills:**
  - McGovern & Sutton 1998; Durugkar et al. 2016.
  - Chang et al. 2019: macros built on one Atari game reused elsewhere. This is the closest prior work.
  - Learned action priors across tasks (Rosman & Ramamoorthy 2012).
  - Action grammars (Lange & Faisal 2019).
  - Options (Sutton, Precup & Singh 1999; Bacon et al. 2017); DIAYN.
- **Meta-learned exploration:** MAESN (structured noise across tasks), E-RL², First-Explore, Alet et al. 2020.
- **Learning sets and human priors:** Harlow 1949; Dubey et al. 2018; Tsividis et al. 2017.

**Novelty, honestly stated.** The mechanism (mined macros as an extra exploration process) is not new. What is new is the test:
- a live decomposition of a demonstration's value into "where" and "how", from shared savestates, used as an advance ceiling;
- then a learning-set curve on never-opened games, against designer macros, within-game mining, the best single process and a non-advancing placebo;
- all with no RAM in the agent's inputs, no downloaded weights, and only the score-rise and lives senses.
