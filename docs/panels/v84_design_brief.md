# v84 design brief (for the design panel, 23 October 2026)

## Goal (owner, non-negotiable)
Demonstrate that the agent **learns a new game faster thanks to what it learned on earlier games**
(the human-like property), honestly and presentably. Long ladder: Atari -> NES -> 16-bit -> PC, each
generation faster than from scratch. A head start is not faster learning. The owner wants genuine
innovation, not an adopted pipeline, but honesty limits are absolute: criteria frozen before results,
negatives kept, no oracle information in the agent's inputs.

## Rules
- Agent inputs: screen pixels, its own buttons, and (owner decision 2026-10-22) the score's RISES and
  the LIVES count as senses. RAM, true score, game names: evaluator only. Savestates allowed as an
  organ (each load billed 60 frames; every emulated frame billed, probes included).
- No downloaded pretrained weights. Reserved games never opened: boxing, frostbite, riverraid, kangaroo.
- Pools available (v82 judge): 35 development games, 6 calibration games (all used by v83 A1; their A1
  logs exist), 24 SEALED games and 9 second-round games still UNOPENED (usable once under a new protocol).
  Sealed: battle_zone defender double_dunk gravitar montezuma_revenge pitfall private_eye robotank skiing
  solaris surround tutankham venture air_raid donkey_kong frogger galaxian journey_escape kaboom
  keystone_kapers king_kong mario_bros mr_do pacman (7 of them hard-exploration).
- Machine: one Windows PC, 12 CPU threads, RTX 4080 16 GB (torch 2.11 CUDA works), shared with another
  project that needs >= 12 GB RAM free at all times (a supervisor suspends/kills jobs below that; budget
  ~2-4 GB RAM for Ragnarok, GPU shared). ALE 0.12, sticky 0 for practice. Typical ALE throughput here:
  a 2M-billed-frame explorer run takes 10-15 min on one core.
- Rigour proportionate: pre-registration, freeze and adversarial review only at milestones that carry a
  claim; exploratory work fast and labelled exploratory.

## What has been tried (ARC 4) and what it taught
- v71-v74: perception from pixels works on many games (me by controllability, buttons by signature,
  score/death readers - later replaced by senses). Savestate branch values are real, but distilling them
  into a reactive pixel policy failed: live policies only matched the best constant button.
- v80: a readable "theory" of the game (objects, motion, contacts) + imagination: correct readable rules,
  but predictions did not beat climatology robustly. Hand-added notion types fixed one game, broke another.
- v81-v82: generic score/death reading from pixels failed on 4 of 5 new games -> owner granted the senses.
- v83 "promise map": carried logistic heads predicting GOOD (score rise within 24 steps) / BAD (life loss
  within 40 steps) around me, used to steer a Go-Explore-style explorer (cells = me on 16x16 x lives x
  log-bin of cumulative rise; 100-step rollouts, uniform random button held with repeat 0.95; a life loss
  ends a rollout; probes to re-find me). Gate A1 ("room": does the FRESH explorer climb enough that a 2x
  faster climb is measurable?) = KC-room: only 6/35 dev games (alien, amidar, bank_heist, ice_hockey,
  jamesbond, qbert; + breakout, ms_pacman) reach 5x the bar in 2M billed frames on both seeds.
  Details: F:\dev\ragnarok_v83\docs\ARC4_V83_A1_RESULT.md and craft_v6_out/v83/room_verdict.json.
- A1 lessons (ranked, from an adversarial review): (1) a fresh learner must climb before any transfer
  test (4 games stayed below the random bar; 33/58 failing runs still improving after 1.5M);
  (2) bars were often set by per-step random play the explorer never does (B / repeat-random >= 2.3 on
  21 games, none with room); (3) probes cost 27% of frames, > 64% on 7 games; (4) some rungs exceed game
  maxima (fishing_derby, pong, freeway); bar plays all lives while the explorer stops at the first loss;
  (5) the cell needs a progress axis independent of me/score (runs with ~4,100 rollouts and zero rise);
  (6) seed spread is large (pooled sigma of log climb 1.08 with 2 seeds).
- Earlier arcs (symbolic worlds, ARC 2/3): frozen transfer was real but "learns faster" stayed inside
  noise; v62 showed the advantage concentrates on hard goals while experience can hurt easy ones.

## Assets on disk
- v83 code (F:\dev\ragnarok_v83): senses proxy with billed ledger and replayable evaluator trace,
  explorer V, views/features (64x64 egocentric window, ~427 features), promise heads with L2-SP, W
  training (LOGO, placebo), G-pred, climb statistics (censoring, two-level bootstrap, permutation p).
- A1 logs: 82 runs x ~60-100k promise rows (packed views, slots, outcome lags), bars for 41 games.
- Exploratory result (pending, no claim): cross-game prediction of GOOD/BAD on the 8 room games.

## The question for the panel
Propose v84: the design most likely to yield, within about two weeks of this machine, an HONEST positive
demonstration of "learns new Atari games faster thanks to earlier games", or else the most informative
negative. It must include: the mechanism of reuse (what is carried, why it should speed learning), the
fresh learner and why it will climb on most games (address lessons 1-6), the yardstick of "faster"
(matched levels, head start vs late climb, placebo control), gates in order with kill criteria, pools and
seeds, compute and RAM estimates, odds, and what is genuinely new versus the literature.
