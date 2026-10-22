# v83 "Promise map": milestone definition, ready to freeze

I only read and grepped files. Nothing was run and nothing was edited. What I checked:

- `env.clone()` returns `(state, prev, last, _true_score)`, and every restore bills 60 frames (`F:\dev\ragnarok_v81\ragnarok\agent_v71\env.py:53-60`).
- `Archive.choose` weights each cell 4/sqrt(chosen+1) at the best score (1 elsewhere) and loops over the cells in Python.
- `SignatureAdapter(env.game, ...)` is called at `learner.py:46`.
- `Perceiver.warm` defaults to 1500 steps.
- `eligibility_v81.references` measures uniform random play but leaves it out of the bar (`scripts/eligibility_v81.py:102`).
- v81 Amidar result: 189 against a bar of 40.7, 1,635,540 billed frames, one seed. On Bank Heist, Crazy Climber and Wizard of Wor practice scored 0 because the agent read the wrong counter. Crazy Climber: uniform 7900 against random-with-repeat 2910. Wizard of Wor billed 2.89M frames against a 2M budget.
- The pools are those of v82 judge §2.
- Seeds: 9131-9145, 9147-9149, 9230-9238, 9241-9245 and 9250-9258 have no seed mention in any worktree. The only hits were 9146, 9239 and 9240 as percentages, 9248 as a frame count, and 9150 used as a test seed (`tests/test_response_envelope_equivalence_v70.py:228`). v82's own seeds (9101-9111, 9201-9211) are left alone.

---

## 1. Decision and why

The carried object comes from the **value** design (a promise map). The protocol structure is a merger of the best parts of all three designs. Exactly one carrier is ever run on the sealed games.

Why the promise map:
- It carries content learned from earlier games' experience: what the screen around me looks like before points and before a death. Its labels come from the new senses. That is exactly what killed v81's affordance prior, which had to read score and death from pixels. ARC4_VISION §2.6 and the contact-outcome priors (ARC4_VISION.md:223-231) already name this mechanism.
- It acts at every iteration, both in which cell to return to and in which buttons to try. So it can in principle change the rate of the climb, not only the starting point.
- It is reached through a minimal fresh explorer (arm V), which answers v81's open G-room question first and cheaply.

Why not the other two:
- **cells**: the carried object is about 60 search settings, so "tuned on dev games beats untuned defaults" is confounded with transfer. It is the heaviest build, its thumbnail keys break the palette-invariance test, and it predicts by its own account that θ/α only buy a head start.
- **skills**: the least plausible effect. Repeat 0.95 already holds buttons for about 20 steps, and Go-Explore is limited by which cell it picks. Its fresh arm also risks being crippled.

Corrections taken from the judges:
- **From cells:** room selection and eligibility come from separate fresh seeds and are committed with a hash before any arm runs. Also: 4-seed placebo, a fresh-vs-fresh instrument line, a projected detectable effect that includes between-game spread (τ), a leave-one-game-out robustness check, and a permutation p-value.
- **From value:** each arm's climb is counted from its own M1, with a late-climb clause from M2. X must beat V, F and P. X0 is used for attribution.
- **From skills:** the score sense carries only rises (the owner wrote "ses hausses"). Uniform random play enters the bar. The placebo keeps structure and confidence and loses only content.
- **Feasibility fixes:**
  - the budget is enforced by billed frames inside the loop;
  - handles are opaque and zlib-compressed;
  - the archive has a cap and a declared eviction rule;
  - `choose` is vectorised;
  - a longer warm-up replaces `fit_counters`;
  - probes are rate-limited and logged per arm;
  - stability is pooled across games, never a per-game filter on 4 seeds;
  - one rule links G-room and the power check;
  - censoring is resolved against the carried arm;
  - 2 workers, not 4.

---

## 2. Step A: room and development gates (development pool, with the senses)

### 2.1 Senses and explorer V (shared by every arm)

**`SenseEnv` proxy** over `agent_v71/env.AtariEnv`:
- `step(a)` returns `(obs, rise, lives, end)`:
  - `rise` = max(0, ALE reward summed over the 4 frames);
  - `lives` = `ale.lives()`;
  - `end` = game over. It is used **only** as an episode boundary (the rollout stops and the state is not archived). It is never a label, a feature or a veto (v82 judge, correction 4).
- `clone()` returns an int handle. A registry holds the ALE state plus zlib(obs). Only the max-pooled obs is needed, because the next step overwrites `prev` and `last`. There is also `free(h)`.
- `restore` bills 60 frames. The billed ledger never rewinds and includes probe frames and the adapter's sibling emulator.
- `sibling(seed)` builds the adapter's private emulator without exposing the game name.
- `ram()`, `true_score()`, a raw `lives()` and `.game` do not exist.
- On every clone, the proxy writes (billed frames, true cumulative score) to an **evaluator-only trace**. Agent code cannot import its reader, which an AST test checks.

**Entry** (identical for every arm):
- `Perceiver.warm` with 12,000 steps instead of 1500, to make up for the background samples that `fit_counters` used to provide;
- `SignatureAdapter` through the sibling factory;
- the v71/v74 seeding of me;
- the `lost_frac` check (1500 steps).

There is no `fit_counters` and no HUD key. Entry costs about 0.15-0.2M billed frames, measured and reported per game.

**Explorer V** is a port of `Agent.explore` (`learner.py:132-188`, with the v74 `_track` fix):
- **Cell key:** (me on a 16×16 grid, or (-1,-1) when I am not found; lives; floor(4·log2(1+S))), where S is the cumulative sensed rise along the cell's trajectory. A better cell has higher S, then shorter depth (the v71 rule).
- **Choice:** the v71 weights, vectorised.
- **Rollouts:** 100 steps. A uniform button is held with repeat 0.95. A lives decrease ends the rollout and its state is not archived.
- **Probes:** at most one every 8 steps while I am lost. This is a declared change from v71. Probe frames are billed and logged.
- **Cap:** 20,000 cells. When full, the archive evicts the most-chosen cell among those below the best S (ties: the oldest). The rule is the same in every arm.
- **Budget:** a hard stop at the billed cap, checked at every step. Early stop at M* happens evaluator-side: the queue reads the trace and the agent never sees it.
- **Logs:**
  - per iteration: billed frames, best S, cell count, probe frames, lost steps;
  - every 4th rollout step: a promise row (packed view, slot, labels from the senses).

### 2.2 Promise machinery (arms F, X, P, X0 and K*; absent in V)

**View:**
- A 64×64 window on me showing the colour-free foreground (pixels that differ from the agent's own background), with me masked out.
- Four channels: objects, pixels that appeared, pixels that vanished, and small blobs (at most twice my size).
- The window is rotated so that the candidate slot's move (sign dx, sign dy, from the adapter) points up. There are 4 rotations plus a diagonal flag. The stay slot and private slots are not rotated and are flagged.
- When I am not found, the window is centred on the screen and a "me lost" flag is set.

**Features** (about 430):
- pooling of each channel at 8×8, 4×4 and 2×2;
- forward-cone distance bins to the nearest object and to the nearest approaching object;
- slot flags;
- distance to the screen edge ahead;
- a lives = 1 flag;
- an unrotated 8×8 whole-screen foreground.

Fixed standardisation scales are part of W.

**Labels** come only from the senses:
- GOOD: a rise within 24 steps.
- BAD: a lives decrease within 40 steps.

Rows whose horizon is cut off and rows where I am lost are dropped. On a game whose lives never change, the BAD head is intercept-only.

**Model:**
- Two logistic heads, with intercepts free per game.
- Online refit every 250 iterations on a reservoir of at most 120k rows (float32).
- Loss = mean log-loss + (n0/(n0+n))·(κ0/2)·‖w − w0‖², with n0 = 50,000 and κ0 = 1.

**Steering:**
- (a) Cell choice: draw 16 candidates using V's weights, then pick in proportion to exp(β·clip(z, −2, 2)). z is the robust z-score of u(c) = max over slots of [P(GOOD) − λ·P(BAD)].
- (b) At the start of a rollout and at each action change (repeat 0.95 kept), the slot is drawn from (1−α)·uniform + α·softmax(logit GOOD − λ·logit BAD).
- Settings: β = 1, α = 0.5, λ = 1.
- With w = 0, both rules give exactly V's distribution, so F starts as V. A test checks this.

**Carried object (Π):**
- W = both heads' non-intercept weights plus the standardisation scales.
- W_dev is trained on the A1 logs of all 41 development and calibration games, each game weighted equally. LOGO W₋g is trained on the same logs without game g. W_same(g) is trained on game g's A1 logs only.
- **Placebo W_perm:** the same training, but labels are shifted in a circle within each game by at least 10 rollouts (seed 9149). Each head is then rescaled to the norm of W's head. This keeps base rates, how labels cluster in time and the prior's confidence, and breaks the state-to-outcome link.
- If only one head passes G-pred (§2.6), X, P, X0 and K* carry only that head; the other head's w0 is 0. This rule is mechanical.

| Arm | w0 | Refit | Role |
|---|---|---|---|
| V | (no promise model) | none | plain explorer with the senses |
| F | 0 | on | fresh: the same learner without earlier games |
| X | W₋g in dev, W_dev in sealed | on | carried |
| P | W_perm | on | placebo |
| X0 | W | off | attribution only |
| K* | W_same (dev only) | on | WORTH ceiling |

### 2.3 Pools and seeds

**Pools** (copied from the v82 judge into `pools_v83.py`):
- Development: the 35 games.
- Calibration: the 6 games (they count as sources and as units, but are reported separately).
- Sealed: the 24 games.
- Second round: the 9 games.
- Never opened: boxing, frostbite, riverraid, kangaroo. Enduro is excluded.

| Use | Seeds |
|---|---|
| Engineering and smoke runs (never a measurement) | 9130 |
| A1 room selection (V) and source logs | 9131, 9132 |
| Development bars | 9133 |
| Optional D1 diagnostic | 9134 |
| A3 WORTH and A4 freeze 1 (paired across arms) | 9135-9138 |
| A4 freeze 2 | 9141-9144 |
| Placebo shift and unit shuffle | 9149 |
| Reserve | 9139, 9140, 9145, 9147, 9148 |
| Sealed bars | 9230 |
| Sealed E runs | 9231, 9232 |
| Sealed arms (paired) | 9233-9236 |
| Live diagnostic (sticky 0.25) | 9241, 9242 |
| Second round: bars / E / arms / X0 | 9250 / 9251-9252 / 9253-9256 / 9253-9254 |

Re-grep all of these in seed context at A0 and write the result into the protocol.

### 2.4 Bars, score milestones and climbs (evaluator side)

- **Bar:** B_g = max(best constant button, uniform random, random with repeat 0.95), measured in practice conditions: sticky 0, 0-30 NOOPs, 7000 steps, 10 episodes each, with `eligibility_v81.references` plus uniform added to the max.
- **Step:** u_g = max(0.5·|B_g|, the median positive true increment over the reference episodes). If the reference episodes show no increment, the median comes from the selection or E runs' trace.
- **Ladder:** M_k = B_g + 2^k·u_g for k = 0..10. When B > 0 this gives M0 = 1.5B, M1 = 2B, M2 = 3B, M3 = 5B. The ladder still works when B ≤ 0.
- **T_s(M):** billed frames from frame 0 (entry included) at the first clone whose true cumulative score is at least M. It is confirmed by replaying the trajectory from the root; a mismatch is a defect and an Interruption.
- **Climb:** C = T(M*) − T(M1), counted from each arm's own M1. **Late climb:** C_late = T(M*) − T(M2).
- **Censoring always goes against the carried arm** (X, X0, or K* in WORTH):
  - a carried run that does not reach M* counts C = cap;
  - a comparator run (V, F, P) that does not reach M* counts C = cap − T(M1);
  - if it also misses M1, it counts the floor.
  - The floor is C ≥ 5,000 frames.
- **Statistics:**
  - ℓ_g = mean log C(comparator) − mean log C(X);
  - r_g = exp(ℓ_g), clipped to [1/8, 8];
  - pooled R = exp(mean over games of ℓ_g);
  - 95% interval from a two-level cluster bootstrap (games, then seeds within each arm; 2000 resamples, percentile).

### 2.5 A1, G-room (the question v81 left open)

- **Units:** 41 games. Arm V, seeds 9131 and 9132, cap 2M billed frames each.
- **A game is a room game if all of these hold:**
  - (a) each run has at least 10 sensed rises in its first 300k billed frames;
  - (b) M*_g, the highest rung reached by both seeds within 2M, is at least M3;
  - (c) median T(M1) ≤ 1M;
  - (d) median C ≥ 300k.
- **No per-game stability filter.** On 2-4 seeds it would mostly select noise. The spread is pooled instead: σ̂² = mean over room games of (log C₁ − log C₂)²/2.
- **G-room passes** if both hold:
  - at least 13 of the 35 development games are room games;
  - se_proj = sqrt((σ̂²/2 + 0.09)/G_proj) ≤ 0.23, with G_proj = 17·n_room/35.
  
  17 is the 24 sealed games minus 7 hard-exploration titles (montezuma_revenge, pitfall, venture, private_eye, solaris, gravitar, skiing), which are assumed never eligible. This pessimism is declared now. Thirteen room games gives G_proj ≈ 6.3, consistent with the sealed minimum of 6.
- **KC-room** otherwise. The recorded result: "with rewinds and the score/lives senses, fresh Atari practice by this explorer is too cheap, saturated or noisy for a 2× faster climb to be measurable on the sealed pool". The per-game cost map is kept.
- **Also reported:** σ̂, the per-game curves, the probe share of frames, the me-lost fraction, and evaluator-side accuracy of me against the `*_v70` RAM judges on the calibration games. `room_verdict.json` is committed before any F, X, P or K* frame.

### 2.6 A2, G-pred (offline, no frames)

- On each room game g, W₋g is scored on g's A1 rows. Clusters are blocks of 50 iterations.
- **A head passes on game g** if:
  - its AUC is at least 0.60 for GOOD or 0.65 for BAD; and
  - its AUC exceeds the better of two baselines by at least 0.03, with a cluster-bootstrap lower bound above 0 (2000 resamples). The baselines are the per-slot base rate fitted in-sample on g, and the placebo W_perm,₋g.
- **A head passes overall** if it passes on at least 50% of room games and on at least 6 of them. G-pred passes if at least one head passes. Otherwise **KC-pred**.
- **A2b (report only; prediction written now):** held-out log-loss of the prior plus in-game refit, against in-game only, at 2k, 10k and 50k rows. The prediction: the prior's advantage falls below 25% of its initial value by 50k rows. A head start is then the expected outcome unless the steering compounds.

### 2.7 A3, WORTH (live, development)

- **Units:** the room games. If there are more than 12, take the first 12 of a fixed shuffle (seed 9149).
- **Arms:** V, F and K*, 4 paired seeds (9135-9138), cap 4M, early stop at M*.
- **Pass:** R(K* vs the better of V and F) ≥ 2.0, with a game-bootstrap lower bound ≥ 1.25, and r_g ≥ 2 on at least 1/3 of the units.
- **KC-worth** otherwise: even a same-game promise prior cannot halve the climb.
- A KC-harm flag (F more than 1.25× slower than V) is reported. Freeze 2 may address it.

### 2.8 A4, development rehearsal (at most 2 freezes)

- **Freeze 1 (as frozen now):** X with W₋g and P with W_perm,₋g, 4 seeds 9135-9138, paired with A3's V and F. X0 with seeds 9135-9136. B's exact statistics are computed on these runs.
- **GO** if all of these hold:
  - R(X:F) ≥ 1.5 with lower bound > 1.1;
  - R(X:V) lower bound > 1.0;
  - R(X:P) lower bound > 1.0;
  - R_late(X:F) lower bound > 1.0.
- **Otherwise one declared freeze 2,** written before its runs:
  - It may change only the promise machinery: features, horizons, β, α, λ, n0, κ0, the refit schedule, or a small MLP head.
  - It may not change the senses, the explorer core, the cell key, the milestones or the thresholds.
  - F, X, P and X0 re-run on 9141-9144 (X0 on 9141-9142). V is reused.
  - Failing again gives **KC-dev**: the sealed games stay unopened.
- **Before the sealed run:**
  - an adversarial review (ragnarok-review);
  - commit the code, the protocol, the pools, and the SHA-256 of W_dev and W_perm;
  - `git commit --only`.
- Any change to the explorer core after A1 invalidates A1. It becomes an Interruption and A1 is re-run on reserve seeds.

---

## 3. Step B: transfer to the sealed games (once)

**B0, blind eligibility (independent of the arms).**
- After the freeze: sealed bars (seed 9230), then the E runs (V, seeds 9231-9232, cap 2M) on the 24 sealed games.
- Eligibility is the A1 room rule (a-d) applied to the E runs. This also fixes M*_g.
- `craft_v6_out/v83/sealed_eligibility.json` is committed with its hash before any arm runs, and the arm queue reads only that file.
- If fewer than 6 games are eligible, the second round is opened once under the same rules and the units are pooled. If there are still fewer than 6, the verdict is **NOT MEASURABLE**, which is neither a pass nor a fail.

**B1, arms.**
- X (W_dev), F, V and P (W_perm), 4 paired seeds 9233-9236.
- X0 with seeds 9233-9234.
- Cap 4M, early stop at M*.

**Metrics.**
- R(X:F), R(X:V), R(X:P) and R_late(X:F), as defined in §2.4.
- Head start: HS = the geometric mean of T_F(M1)/T_X(M1).
- A one-sided permutation p-value for R(X:F): swap the X/F labels within each game (4+4, 70 permutations per game; Monte Carlo with 10,000 draws if needed).
- Leave-one-game-out minimum of R(X:F).
- **Instrument line:** split F into seeds 9233-9234 against 9235-9236. The pooled ratio must lie in [0.7, 1.43].

**Verdicts** (fixed now; the first that matches applies; negatives are kept):
1. **NOT MEASURABLE:** fewer than 6 units after the second round.
2. **INCONCLUSIVE:** the instrument line fails, or at least 50% of X runs or of F runs are censored at M*.
3. **FASTER-2×**, "learns new Atari games faster thanks to earlier games, in practice with rewinds and with the score-rise and lives senses". All of these must hold:
   - R(X:F) ≥ 2.0 with lower bound ≥ 1.3, and p ≤ 0.025;
   - R(X:V) ≥ 1.5 with lower bound > 1;
   - R(X:P) ≥ 1.5 with lower bound > 1;
   - R_late(X:F) ≥ 1.5 with lower bound > 1;
   - r_g(X:F) > 1 on at least 2/3 of the units;
   - the leave-one-game-out minimum of R(X:F) ≥ 1.5.
4. **FASTER-PARTIAL:** the lower bounds of R(X:F), R(X:V), R(X:P) and R_late(X:F) are all above 1, and p ≤ 0.025.
5. **NEGATIVE TRANSFER:** the upper bound of R(X:F) or of R(X:V) is below 1.
6. **HEAD START ONLY:** HS ≥ 1.5 with lower bound > 1, and neither verdict 3 nor 4 holds.
7. **NULL:** anything else.

A pure head start cannot pass, for four reasons:
- every climb starts at the arm's own M1;
- a late clause starts at the arm's own M2;
- the placebo must also be beaten;
- HS is its own, lower verdict.

**Captions** (they never change the verdict):
- **GENERIC STEERING:** the lower bounds of R(X:F) and R(P:F) are both above 1, but R(X:P)'s lower bound is at most 1. A confident prior helps, but no content transfers.
- **Attribution:** if R(X0:F) ≥ 0.8·R(X:F), the result is called "a fixed search prior from earlier games"; otherwise, "a prior combined with in-game learning".

Written in advance: because the 2× clause is on the point estimate, a true 2× effect passes only about half the time.

**Also reported:**
- frames to every rung;
- the area under log score against log frames;
- censoring counts per arm;
- the probe share per arm, and the climb recomputed without probe frames (secondary);
- **D2 live diagnostic:** promise-greedy play without savestates, sticky 0.25, 10 games × 7000 steps, seeds 9241-9242. The X and F models at the end of practice are compared with the best constant under the per-seed criterion of the protocol skill.

**Kill and stop rules:**
- KC-room, KC-pred, KC-worth, KC-dev as defined above.
- KC-placebo is built into every FASTER verdict.
- Budget: §4.
- A worker above 3 GB is paused, never changed.
- A defect found mid-run stops the queue and becomes an Interruption; partial results are committed.
- A defect found after scoring is never re-scored on this pool. A revised version may only use pools that are still unopened.

---

## 4. Budget and calendar

**Throughput:** about 2.4k billed frames/s per worker (v81 measured 1.64M frames in 567 s on Amidar, including perception), plus at most 15% promise overhead. The ragnarok-run skill says "usually 1-2 jobs", so plan on 2 workers and 3 when Tavern is idle. With about 70% availability that is roughly 250-300M frames a day.

| Stage | Billed frames |
|---|---|
| A0: smoke runs and pre-checks (9130) | ≈ 15M |
| A1: bars, 41 games | ≈ 80M (evaluator only) |
| A1: V room runs, 41 × 2 × 2M | ≈ 165M |
| A2: G-pred | 0 (numpy, memory-mapped, ≤ 3 GB) |
| A3: up to 12 games × 12 runs × ≈ 2.2M | ≈ 320M |
| A4 freeze 1: 12 × 10 runs | ≈ 265M |
| A4 freeze 2 (if needed): 12 × 14 runs | ≈ 370M |
| B0: bars plus E runs, 24 games | ≈ 145M |
| B1: ≈ 8 units × 18 runs, plus D2 | ≈ 330M |
| Second round (if needed) | ≈ 150M |

- **Totals:** about 1.3G frames on the full path, 1.8G with freeze 2 and the second round.
- **Calendar:**
  - 2.5 days of engineering;
  - A1, about 1 day (the promise code is built during it);
  - A2, a few hours;
  - A3, about 1.2 days;
  - A4, 1 to 2.3 days;
  - review and freeze, 0.5 day;
  - B, about 2 days.
  
  That is about **9-11 days**.
- **Hard stop:** 14 calendar days or 2.0G billed frames. The development part is capped at 1.3G. Once the sealed run starts it is completed, within about 0.6G.
- **Most likely stops:** KC-room around day 4 (≈ 0.26G frames), KC-pred the same day, KC-worth around day 6 (≈ 0.6G).

**RAM:**
- No torch: `.venv-arc4` with numpy, scipy and ale_py.
- Target 1.5 GB per worker; pause above 3 GB.
- The archive is at most about 0.5 GB with zlib obs, and the row reservoir about 0.2 GB.
- `scripts/ram_supervisor.py --low 12 --high 14 --clow 12 --chigh 14`.
- Launch gates wait for 14 GB of RAM and of commit.
- Queues are resumable and every output carries `provenance_v81.stamp()`.

---

## 5. First 5 concrete steps (worktree F:\dev\ragnarok_v83)

1. **Worktree, protocol and pools.**
   - Run `git -C F:\dev\ragnarok worktree add F:\dev\ragnarok_v83 -b arc4-v83-promise eab9b9b`. eab9b9b is the head of arc4-v81-atlas; master lacks the v81 scripts.
   - Write `docs/ARC4_V83_PROTOCOL.md` in French with everything in this document: the owner's senses exception of 2026-10-22 (score rises and lives; `over` only as a boundary), the seed re-grep table and the "does not establish" list.
   - Add `ragnarok/evaluation/pools_v83.py` and `tests/test_pools_v83.py`: pools disjoint, reserved games and enduro excluded, sealed names appearing only in this module and the judge document.
   - Record the senses exception in the "Never" line of `F:\dev\ragnarok\.claude\skills\ragnarok-protocol\SKILL.md`.
2. **Senses and adapter.**
   - `ragnarok/agent_v83/senses.py` (SenseEnv, opaque handles, evaluator trace, sibling factory).
   - `ragnarok/agent_v83/adapter.py`, a copy of `agent_v71/adapter.py` taking a factory instead of the game name.
   - `tests/test_senses_v83.py`, with these checks:
     - over 10k steps on 3 calibration games, `rise` equals the positive part of the reward and `lives` equals `ale.lives()`;
     - a restore gives a byte-identical obs;
     - each load bills 60 frames;
     - the ledger never decreases;
     - sibling frames are billed.
   - `tests/test_isolation_v83.py` (AST): no ram, getRAM, true_score, `.game`, torch, learner or policy imports; no pool game names in `agent_v83`; the trace reader cannot be imported from `agent_v83`.
3. **Explorer V, views, logs and evaluator.**
   - `ragnarok/agent_v83/explorer.py`: entry, vectorised archive with cap and eviction, billed cap, probe rate limit.
   - `ragnarok/agent_v83/views.py` and `ragnarok/agent_v83/logs.py`. The A1 logs must already carry the promise rows.
   - `ragnarok/evaluation/milestones_v83.py`: ladder, T(M) from the trace, replay confirmation, censoring rule.
   - `scripts/bars_v83.py`.
   - `tests/test_explorer_v83.py`:
     - 20k cells under 600 MB;
     - replaying the best trajectory reproduces its true score;
     - the view is equivariant under rotation.
   - Smoke runs, seed 9130, on Amidar (V must reach at least 3× B in 2M) and Freeway.
4. **Queue, review and launch of A1.**
   - `scripts/room_v83.py`: resumable, one subprocess per job, waiting gates copied from slice_v74, provenance stamp. Outputs go to `craft_v6_out/v83/room_<game>_<seed>.json`.
   - `scripts/room_verdict_v83.py`.
   - Run ragnarok-review on the senses, explorer, milestones and bars, and fix what it finds.
   - `git commit --only` the code and protocol, then launch the bars and the room runs on 2 workers under the supervisor.
5. **During A1: the promise code.**
   - `ragnarok/agent_v83/promise.py`: heads, L2-SP, reservoir, steering.
   - `scripts/train_promise_v83.py`: W_dev, LOGO, W_same and the norm-matched W_perm, streamed from memory-mapped files.
   - `scripts/gpred_v83.py`.
   - `tests/test_promise_v83.py`:
     - w = 0 reproduces V's choice distribution exactly;
     - the placebo destroys a planted synthetic signal;
     - W is byte-identical under permutations of button order and palette in the source logs.
   - A second review before A3.

At the first milestone, also record in `docs/CODEX_HANDOFF.md`:
- v83 spends the v82 judge's 24 sealed games, and possibly its second round;
- a revived v82 would need a new sealed pool;
- v83's seeds.

---

## 6. What this milestone does not establish

- **No live competence.** Every claim is about practice with rewinds (savestates billed at 60 frames per load) on Atari, at sticky 0. The v72-v75 distillation wall is untouched; D2 is only a diagnostic.
- **No perception of score or death.** Score rises and lives are given as senses, and every result says so. `over` is used only as a boundary.
- **Not "any game".** Only games where a fresh climb is costly enough are units. A KC-room says nothing about transfer as such.
- **No head start presented as learning faster.** HS is reported separately and never counts as faster learning.
- **Nothing about NES, 16-bit or PC.** It also does not show that the representation of me works on new games; that is only measured on calibration games against the RAM judges.
- **A FASTER result has a narrow scope.** It would say only that a carried state-to-outcome prior speeds this explorer's climb. GENERIC STEERING would say only that a confident prior helps, not that content transferred.

---

## 7. Honest odds (my estimates, not measurements)

| Event | Probability |
|---|---|
| A0 harness and Amidar smoke test pass | ≈ 0.85 |
| G-room passes (at least 13 of 35 games, se_proj ≤ 0.23) | ≈ 0.45 |
| G-pred passes (at least one head), given room | ≈ 0.45 (BAD head more likely than GOOD) |
| WORTH, given room and pred | ≈ 0.35 |
| Development GO within 2 freezes, given WORTH | ≈ 0.4 |
| At least 6 sealed units (with the second round), given GO | ≈ 0.7 |
| FASTER-2×, given a measurable B | ≈ 0.35 |
| **Overall FASTER-2×** | **≈ 0.01** |
| Overall FASTER-PARTIAL or better | ≈ 0.02 |
| HEAD START ONLY, if B runs | ≈ 0.3 |
| A clean answer to G-room | ≈ 0.85 |
| The line ends at KC-room, KC-pred or KC-worth | ≈ 0.75 |

These odds are lower than the designs' own estimates (2-5%). The merged gates are stricter: X must beat F, V and P, the late climb is required, and censoring goes against X.

The most likely ending is a cheap, clean negative within about 4-6 days, which still gives:
- the first per-game map of fresh practice cost and spread with the senses;
- the answer to v81's G-room question;
- whether a state-to-outcome prior is even a lever (WORTH), before the sealed pool is spent.