# v82 "General perception": milestone definition (skeptical judge)

I only read and grepped files. Nothing was run and nothing was edited. I checked the seeds and game names by grep across all seven worktrees (`F:\dev\ragnarok`, `_v73`, `_v74`, `_v75`, `_v80`, `_v80c`, `_v81`). As a control, the same regex finds seed 8101 in 14 files.

## 0. Corrections to make before anything is frozen

1. **The seeds proposed in the designs were already used in ARC 3.** 8200–8203, 8210–8213 and 8220–8223 appear in `F:\dev\ragnarok\ARC3_PLAN.md:302-303` and `preregistration.md:7056`. 8300–8302, 8310–8312 and 8320–8322 appear in `docs\ARC3_V62_CONFIRMATION_RESULTS.md:33-35`. Use the empty 91xx/92xx range instead (§3).
2. **The evidence says the perception code is unchanged since the survey. That is only partly true.** Since b3d4f0f, `hud.py`, `perceiver.py`, `tracker.py`, `me_buttons.py` and `env.py` are unchanged. `learner.enter` still runs the same `warm` + `fit_counters(20000, 0.95)`. So the score-region flip really comes from the random number stream alone. However, `adapter.py` is new (109 lines) and `learner.py` changed by 519 insertions (v72–v74). So comparing survey and v81 on button effects or death compares different code.
3. **The agent can see the true score inside its savestates.** `env.clone()` returns `(state, prev, last, true_score)`. The v82 agent must get opaque handles.
4. **`over` is an emulator signal, not something read from the screen.** The death design uses `ale.game_over()` to name the death class, which turns it into a label for roughly one death in 3 to 5. In the gated variant, `over` may only mark the end of a game. The variant that uses `over` is reported separately.
5. **The death premise check only looks just before deaths.** The likely failure in maze and junction games is the reverse: identical frames while alive, because input is only read at junctions or every N frames. The check must also measure how often control looks lost while the agent is alive.
6. **The self-evaluation design's point-of-no-return search is too expensive.** It runs 3 branches for every action, about 100k frames per death. The death design's 8 shared-seed siblings cost about 20k frames per death, and they let agent and evaluator judge identical branches.
7. **v82's entry costs about 5 times v71's billed frames.** A v71 arm with the same frame budget is needed, otherwise "more random play" is not ruled out.
8. **The RCF thresholds (the counter design's proposed finder) were written after seeing the results.** They are only starting values for development and get frozen at the declared freeze.
9. **The RAM "me" judge is not fully independent of the agent.** It picks memory bytes by whether actions change them, which is the same principle as the agent's button-based self-detection. A blind visual audit is what anchors it.
10. **Two v81 defects, not fixed (read-only):**
    - `F:\dev\ragnarok_v81\docs\ARC4_V81_ATLAS.md` lines 92–95 contain stray shell lines (`EOF`, `cd …`, `git commit …`).
    - The output of `score_lag_v81` was never saved to a file.

## 1. Question and scope

**Question.** Can one frozen perception module that knows no specific game find three things on Atari games it has never seen, reliably across agent seeds:
- the score counter;
- life losses, with their onset;
- "me" (my avatar).

The module may use only screen bytes, the buttons and its own savestate branches. It is judged on the evaluator side only, against ALE reward, `lives()` and a RAM-probe judge.

This is a perception benchmark only: no learner, no practice, no policy. Each of the three components gives its own verdict.

## 2. Game pools (fixed now)

- **Never opened (reserved):** boxing, frostbite, riverraid, kangaroo.
- **Calibration (6 games):** freeway, pong, breakout, seaquest, ms_pacman, asteroids. These already have RAM judges (`ragnarok/evaluation/*_v70.py`). They are only used to validate the judges. They are free for development but never count in reported rates.
- **Development (35 games, free):** alien, amidar, assault, asterix, atlantis, bank_heist, beam_rider, berzerk, bowling, carnival, centipede, chopper_command, crazy_climber, demon_attack, fishing_derby, gopher, hero, ice_hockey, jamesbond, krull, kung_fu_master, name_this_game, phoenix, pooyan, qbert, road_runner, space_invaders, star_gunner, tennis, time_pilot, up_n_down, video_pinball, wizard_of_wor, yars_revenge, zaxxon.
- **Development core (20 games, for fast iteration):** amidar, bank_heist, crazy_climber, wizard_of_wor, assault, demon_attack, name_this_game, carnival, alien, chopper_command, road_runner, qbert, centipede, krull, up_n_down, kung_fu_master, ice_hockey, tennis, atlantis, hero.
- **Sealed test (24 games, scored once).** Each has zero mentions in code, docs or outputs in any worktree; all are bundled in `ale_py\roms`.
  - From the usual 57-game benchmark: battle_zone, defender, double_dunk, gravitar, montezuma_revenge, pitfall, private_eye, robotank, skiing, solaris, surround, tutankham, venture.
  - Other games: air_raid, donkey_kong, frogger, galaxian, journey_escape, kaboom, keystone_kapers, king_kong, mario_bros, mr_do, pacman.
- **Second sealed round (kept back, zero mentions):** atlantis2, crossbow, elevator_action, joust, laser_gates, sir_lancelot, trondead, turmoil, word_zapper.
- **Excluded:** enduro (a held-out candidate in `ARC4_VISION.md:338`), and the board, educational and two-player ROMs.

Whether a component applies to a game is decided mechanically from evaluator counts (§7), not from manuals or from agent results.

## 3. Seeds (no seed-context mention in any worktree)

Note: with sticky actions at 0, the ALE seed does nothing. A "seed" is the agent's random stream (entry play, sibling choices) and the tape stream. That is exactly where v81's fragility came from.

| Use | Seeds |
|---|---|
| Development iteration (agent entry) | 9101 |
| Checks at each declared freeze (agent entry) | 9102, 9103, 9104 (9104 is validation) |
| Development tapes | 9111 |
| Sealed agent entries | 9201, 9202, 9203 |
| Sealed tapes | 9211 |
| Kept unused | 9105, 9106, 9204–9206 |

## 4. What the agent must produce by itself (contract)

**Entry phase.** The agent plays its own game during entry (cap: 1.0 M billed frames per game and seed), then freezes. It never sees the tapes during entry. It then reads the tapes in lab mode: it may clone and restore at tape states, billed at no more than 3 extra frames per plain frame of the scored segments.

| Output | What it is |
|---|---|
| Score | A screen region plus an "increment" event stream, or ABSTAIN, or a TWO-COUNTERS flag |
| Death | Labels τ (the estimated onset of each life loss), or ABSTAIN for the whole game |
| Me | A position or "not visible" at every step. The button-effect table is diagnostic only |

**Inputs allowed:** screen bytes, the buttons, opaque savestate handles, and `over` used only as an episode boundary.

**Enforced by:** an environment proxy whose `ram()`, `lives()`, `true_score()` and `getRAM`/`setRAM` raise, plus a test that inspects the agent_v82 code (AST).

## 5. Arms, all on the same tapes and entry seeds

- **A0:** v71 perception exactly as it ran in v81.
- **A1:** v71 with longer fit and reader passes, matched to A2's median entry frames on development games. The length is fixed at the freeze.
- **A2:** v82, frozen. One shared sibling-branch primitive drives three parts:
  - RCF score finder, stages A–D;
  - death by the "input ignored" test (D1), bisection on a ring of savestates, and naming of no-control intervals without using `over`;
  - counterfactual "me" (the pixels my buttons change), with a template tracker that only re-finds me by appearance.
- **Floors, evaluator side:**
  - a random region of the same size;
  - random death labels at A2's rate;
  - "me" placed at the screen centre or on a random blob.

  If a floor passes on a game, that metric is uninformative there: it is excluded and reported.

## 6. Evaluator side (the agent never reads any of it)

**Tapes.** Per game and tape seed: 40 clusters in the sealed run, 20 during development. Each cluster is:
1. a reset, then U{0..30} NOOPs;
2. U(0, 2000) steps of random play with repeats (0.95);
3. a scored segment of 1500 steps: 750 steps of random play with repeats (0.8), then 750 steps of evaluator search (every 32 steps, 4 candidates of 32 steps each, keeping the one with the best reward − 1000 × lives lost).

Only actions are stored, with a RAM hash every 500 steps to check that replay is exact. Truth is logged at every step: reward, `lives()`, `game_over`.

**Judges.**
- **Score:** steps with reward > 0, merged when within 2 steps. A RAM-probe display region (`setRAM`, present in ale_py) is diagnostic only: it tells a wrong region from a display lag.
- **Death:** t_L = a drop in `lives()`, or a game over with no drop in the previous 60 steps.
  - The evaluator's point of no return, p_eval, uses 8 random-with-repeat siblings with seeds fixed per (game, cluster, death, offset, k).
  - doomed(p) means all 8 siblings lose a life within [p, t_L + 16].
  - The search goes backward exponentially, then by bisection, over [t_L − 150, t_L]. Non-monotone cases are counted.
  - At most 100 deaths per game in the sealed run, 50 in development.
- **Me:** level-1 RAM probing (bytes that branches change with a consistent sign; a ±δ perturbation must produce a compact, proportional shift).
  - Valid only if, on at least 5 of the 6 calibration games, it puts the centroid within 2 px of the `*_v70` judges on at least 95% of visible frames.
  - Then, per game, a blind audit: 60 frames showing only the judge's outline, by a reviewer agent, with Jérémie checking 10%. Accept at 57/60 or better.
  - All judge work is capped at 1.5 days (v81 §6). A game that is not validated in time is dropped from M, never extended. Sealed-game judges are built only after the agent is frozen.

## 7. Metrics and thresholds (fixed in advance)

All matching is one-to-one and greedy, nearest first.
- **Lift (chance correction):** shift the agent's events in a circle by at least 50 steps within each cluster, 100 times. Lift = (P − P0)/(1 − P0).
- **Confidence intervals:** bootstrap over clusters, 2000 resamples.

**S, the score.** Applies to a (game, seed) when the tapes contain at least 30 true rises and an evaluator replay of the seed's 20k-step random entry stream contains at least 10. The game counts if this holds on at least 2 of the 3 seeds.

| Class | Rule (event within ±4 steps of a true rise) |
|---|---|
| CORRECT | At least 10 agent events; P ≥ 0.9; R ≥ 0.8; lift ≥ 0.5; lower 95% bound of P ≥ 0.75 |
| WRONG | A region was chosen, at least 5 true rises, and P < 0.5 or lift < 0.3 |
| ABSTAIN | No region, or the TWO-COUNTERS flag |
| PARTIAL | Everything else |

A game passes S if it is CORRECT on every applicable seed and WRONG on none.

**D, the death events.** Applies when the tapes contain at least 30 true deaths per seed.
- A label τ is a hit if it falls in [p_eval − 3, t_L + 3], or in [t_L − 60, t_L + 3] when there is no p_eval.
- A seed passes with: at least 20 labels, P ≥ 0.9, R ≥ 0.7, lift ≥ 0.5, lower bound of P ≥ 0.75 and lower bound of R ≥ 0.55.
- A game passes if all 3 seeds pass.
- Games without lives: no more than 1 false label per 10k scored steps on all 3 seeds. Labels within [t_go − 60, t_go + 3] of a game over are excluded.

**D-onset (sub-claim, only on games that pass D).** |τ − p_eval| ≤ 3 on at least 70% of at least 30 matched deaths, on all 3 seeds. This is v81's G0 rule.

**M, "me".** Applies when the judge is validated and there are at least 2000 scored steps where the judge sees me, across at least 20 clusters.
- A seed passes when, in lab mode, the centroid is within 4 px on at least 90% of judge-visible steps (lower bound ≥ 0.8), and the agent claims a position on at most 10% of judge-invisible steps (counted when there are at least 200 such steps).
- WRONG: the agent claims me on at least 50% of visible steps but is more than 8 px off on at least 50% of those (it is tracking another object).
- Live mode is reported, not gated.
- A game passes if all 3 seeds pass.

**Pool criteria, sealed run, A2, per component.** A component passes if all of these hold:
1. at least 10 applicable games; otherwise the component is "not measurable", which is neither a pass nor a fail;
2. at least 60% of applicable games pass, with a Wilson 95% lower bound of at least 0.4 (so at least 10/15, 13/20 or 15/24);
3. A2 beats both A0 and A1 on discordant games, by a one-sided exact sign test with p ≤ 0.05;
4. a WRONG seed occurs on no more than 10% of applicable games;
5. for D only: the no-lives rule holds on at least 80% of the no-lives games where it can be measured.

**Verdict levels.**
- "General perception (events)" means S, D and M all pass.
- "With onset" additionally requires D-onset to pass on at least 50% of the games that pass D.
- The (game, seed) cases that pass S, D and M form the eligibility list for the next learning milestone.

## 8. Gates and stop criteria (negative results are kept; no re-run on the same pool)

**G-P, premises (day 2).** Development and calibration games, evaluator-side measurement of signals the agent can compute, seeds 9101–9103.

| Check | Pass | If it fails |
|---|---|---|
| P-S: candidate recall | The generous candidates (RCF stage A) contain a region with P ≥ 0.9 and R ≥ 0.5 on at least 70% of applicable (game, seed) cases. One generic redesign is allowed. | KC-S: the S line stops |
| P-D: control loss at death | On lives games with at least 30 deaths: an input-ignored stretch of at least 4 steps starting in [t_L − 40, t_L] for at least 70% of deaths, on at least 50% of games. **And** the false input-ignored rate on 2000 alive checks (at least 100 steps from any death or reset) is no more than 2%, on at least 2/3 of games. Frames-identical-but-RAM-differs cases are reported. | KC-D: the input-ignored test is dropped as the death signal; D is reported as a negative premise; no fallback is built in this milestone |
| P-M: counterfactual "me" | The counterfactual mask is within 4 px of the `*_v70` judge on at least 80% of 200 probe states, on at least 4 of the 6 calibration games | KC-M: A2 uses v71's "me" |

If all three premises fail, the milestone ends on day 2 with the A0 baseline map and the negative premises.

**G-dev, after at most 3 declared freezes.** Each freeze is scored on 9104. The final freeze is also scored on 9102 and 9103.
- A2 must beat A0 by at least 6 development games on at least one surviving component, without losing more than 2 games on another.
- Otherwise (KC-dev): the sealed games are not opened, the negative is reported, and the sealed pool stays intact.

**Sealed run (once).**
- A component that passes in fewer than 30% of applicable games is killed as a general method.
- Between 30% and the bar, the result is PARTIAL. A new version may only be tested on the second-round pool.
- A defect found after scoring is recorded as an "Interruption" and is never re-scored on this pool.

**Budget stop.** 6 calendar days or 1.2 G emulated frames, whichever comes first. Stop and report.

## 9. Budget (estimates, to be measured on 2 games in step 3)

| Item | Frames |
|---|---|
| Development tapes (41 games, 20 clusters) | ≈ 21 M |
| Development p_eval (≤ 50 deaths per game) | ≈ 41 M |
| Judges | ≈ 6 M |
| A0 on 3 seeds | ≈ 46 M |
| Premises | ≈ 20 M |
| A2 iteration (development core, 6 loops) | ≈ 190 M |
| 3 freezes on 9104 | ≈ 200 M |
| Final freeze on 9102 and 9103 | ≈ 130 M |
| A1 on 9104 | ≈ 40 M |
| Sealed run (24 games, 40 clusters, 3 seeds, 3 arms, p_eval ≤ 100 deaths) | ≈ 370 M |
| **Total** | **≈ 1.05 G** (cap 1.2 G) |

At about 2.2k frames/s per worker (survey speed) on at most 6 workers, that is about 20–30 hours of compute.
- Each worker stays under 3 GB and imports neither torch nor the learner. Runs go through the RAM supervisor, keeping at least 12 GB free.
- If the 2-game measurement projects more than 1.2 G, cut the freezes to 2. The sealed run is never cut.
- Calendar:

| Day | Work |
|---|---|
| 1 | Protocol, harness, calibration |
| 2 | Tapes, A0 baseline, G-P |
| 3–4 | Build v82, development iteration |
| 5 | Adversarial review, freeze, fix A1's length, build and audit the sealed judges |
| 6 | Sealed run and report |

## 10. First 5 steps (new worktree `F:\dev\ragnarok_v82`, branch `arc4-v82-perception` from `arc4-v81-atlas` @23fbe04)

`master` has no v81 scripts, which is why the branch starts from `arc4-v81-atlas`.

1. **Protocol before any frame.**
   - Create the worktree.
   - Write `F:\dev\ragnarok_v82\docs\ARC4_V82_PERCEPTION.md` (in French, as the project's protocol skill requires), with §§2–9 and the grep commands and outputs for seeds and games.
   - Add `ragnarok/evaluation/pools_v82.py` (pools, seeds, constants) and `tests/test_pools_v82.py`: pools must be disjoint, reserved games excluded, and sealed names must appear in no other file.
   - Commit.
2. **Honesty layer and output contract.**
   - `ragnarok/agent_v82/env_view.py`: the proxy, opaque handles, frame billing.
   - `tests/test_agent_v82_isolation.py`: AST check plus a runtime check.
   - `ragnarok/evaluation/perception_contract_v82.py`: the output contract of §4, with an adapter wrapping agent_v71 for A0 and A1.
3. **Evaluator.**
   - `ragnarok/evaluation/tapes_v82.py`.
   - `ragnarok/evaluation/judges_v82.py`: score, lives, p_eval, RAM-probe "me" and display region.
   - `scripts/calibrate_judges_v82.py` → `craft_v6_out/v82/judges_calib.json`.
   - Measure cost on amidar and freeway → `craft_v6_out/v82/cost_probe.json`, and update the budget table in the protocol.
4. **A0 baseline and G-P.**
   - `scripts/bench_v82.py --arm A0 --pool dev --seeds 9101 9102 9103 --tapes 9111` → `craft_v6_out/v82/bench_A0_dev.json`.
   - `scripts/premise_v82.py` → `craft_v6_out/v82/premise_{S,D,M}.json`.
   - Run through the project's run skill (`ragnarok-run`: supervisor, resumable queue). Append the results to the protocol and apply the stop criteria.
5. **Build v82 on the surviving premises.**
   - `ragnarok/agent_v82/siblings.py`, `control.py`, `selfseg.py`, `counters.py`, `perceiver.py`, with tests on synthetic frames.
   - Iterate on 9101 over the development core.
   - Declared freezes → `craft_v6_out/v82/bench_A2_dev_f{1,2,3}.json`.
   - Then run the adversarial review (`ragnarok-review`), commit the code and protocol, and only then open the sealed games.

## 11. What this milestone does NOT establish

- **Nothing about learning:** not learning, not learning faster, not transfer. v82 perception is built in, not learned.
- **Not that the explorer can practise with it.** v81's E and G-place stay untested.
- **Not perception in live play.** Death detection needs savestates. The cheap screen-only detector, and play with sticky actions at 0.25 (which breaks determinism), are not gated.
- **Not which counter is mine** in two-sided games. Abstaining there is not a pass.
- **Not the score's value**, only its increments.
- **Not the true point of no return.** p_eval is "doom under 8 random siblings", an operational definition.
- **Not late-game states.** Tapes reach about 2000 steps plus a shallow search.
- **Not anything beyond Atari** (NES and later), or beyond 24 test games.
- **"Me" is judged by a judge that shares the action-dependence premise**, anchored by the audit.
- **Even a full pass at 60% leaves about 40% of games without perception.**

## 12. Odds (my estimates, not measurements)

| Outcome | Estimate |
|---|---|
| At least one premise passes G-P | ≈ 0.75 |
| P-S passes | ≈ 0.7 |
| P-D passes | ≈ 0.55 (risk: identical frames while alive in maze and junction games) |
| P-M passes | ≈ 0.55 |
| KC-dev: sealed games never opened | ≈ 0.3 |
| Sealed S passes | ≈ 0.35 |
| Sealed D passes | ≈ 0.3 |
| Sealed M passes (M measurable ≈ 0.6) | ≈ 0.3 |
| S, D and M all pass | ≈ 0.1 |

The most likely outcome is a per-component map with S partly repaired, still valuable because it names what blocks the ladder.