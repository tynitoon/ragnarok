# ARC 4 — the ladder: pixels in, buttons out, one console generation after another

DRAFT until the STAGE-0 freeze (the commit that runs gate 0). Written 2026-09-08 by the project lead
from a 5-design / 3-judge panel (infant, neuro, curiosity, world-model, symbolic designs) whose synthesis
was then refuted by 3 auditors (wall: fatal; instrument, feasibility: major). This document keeps the
panel's architecture, adopts the wall auditor's structural fix, applies the instrument fixes, and cuts
the measurement apparatus to what ARC 3 taught: one primary, one null, one figure, cheap gates that can
say STOP. Versions of this arc are v70+. Section 10 is the audit ledger.

## Résumé pour Jérémie (le reste est le document technique, en anglais)

Le but, avec tes mots : une IA qui apprend n'importe quel jeu naturellement, qui voit l'écran et appuie
sur des touches, qui monte Atari, NES, Mega Drive, PC, et qui apprend chaque génération plus vite que si
elle partait de zéro. Ce document dit comment, et surtout comment on saura si c'est vrai.
- Elle naît avec très peu : « mourir c'est mal », « un compteur qui monte c'est bien », « ce qui bouge
  ensemble est un objet », « ce que mes touches font bouger, c'est moi ». Tout le reste s'apprend.
- Elle apprend à voir par ses PROPRES expériences (elle sauvegarde l'état, appuie sur une touche ou pas,
  et compare), pas avec un détecteur écrit à la main. C'est le point que les auditeurs ont imposé : si
  on lui donne un détecteur tout fait, un agent neuf voit aussi vite qu'un agent expérimenté, et
  l'échelle ne peut rien montrer.
- Elle tient un carnet de faits (« toucher ça tue ») écrit par ses expériences, et un souvenir de ce que
  les objets qui se comportent comme ça font en général : c'est ce qui monte d'une console à l'autre.
- Avant de dépenser sur une génération, un gate mesure si même une IA qui aurait tout retenu
  parfaitement apprendrait plus vite qu'une neuve. Si non, cette génération est un barreau, pas un test.
- Le premier vrai test de ta vision est l'entrée sur NES après Atari. Atari est prévu comme le barreau.
- Ce qui n'est pas promis : le niveau humain, « tous » les jeux, les jeux 3D et PC (une sonde, pas une
  promesse), et qu'elle « comprenne ».

## 0. The goal and the claim ceiling

Goal (owner, verbatim intent): an AI that learns ANY game naturally: input = the screen, output =
simulated keys/buttons, trained by playing on emulators with the clock accelerated; Atari, then NES,
then Mega Drive / SNES, then PC; each generation learned FASTER than from scratch. Second goal: see how
far a frontier model can design a novel AI on its own, grounded, honest about what has cousins.

CLAIM CEILING (if every gate through stage 3 passes, the figure shows):
  (1) from pixels and buttons alone, no reward, no labels, no pretrained weights, the agent learns to
      see (measured against emulator RAM, experimenter-side), finds which object it is, discovers what
      its buttons do, reads its own score and detects its own deaths, on Atari, NES and 16-bit;
  (2) reaches a fixed competence bar (0.25 of human) on a majority of held-out games per generation in
      LIVE play (savestates off, the keyboard-demo code path);
  (3) on held-out games of a NEW generation, an agent carrying only game-agnostic weights and tables
      reaches that bar in fewer emulated frames than an identical fresh agent, beyond fresh-vs-fresh
      noise, with the wording rule of 4.3 deciding "learns faster" vs "uses what it learned";
  (4) an accumulation curve Atari -> +NES -> +16-bit on the same held-out sets, descriptive, beside a
      frames-matched same-generation control;
  (5) an inspectable ledger of contact facts a non-specialist can read with thumbnails.
NOT shown: human-level play; "any game" (a declared family: 2D console games with a visible avatar,
life-loss resets, a digit-like score; exclusions counted); PC or 3D competence (stage 4 is a probe);
understanding; that from-scratch perception beats a pretrained encoder (only that it suffices).

## 1. Why the ladder can work, and the one way it fails by construction

1.1 The law paid for three times: reuse pays only when what is shared is EXPENSIVE for a fresh learner
to reacquire. In console games the shared things are real: segmenting sprites from tiles under flicker
and scrolling, finding oneself, an effect vocabulary and the motor skills reach/avoid/hit/jump, danger
priors ("fast things approaching me kill me"), reading a self-written store. Sprite identities, level
layouts and button maps are cheap and game-specific and are deliberately NOT carried.

1.2 THE FAILURE MODE THE WALL AUDITOR NAMED, now the arc's first design rule. If the expensive things
are hand-coded PROCEDURES handed to both arms, or if the fresh arm receives dense hand-coded supervision
at the new game (a classical background/flow extractor as the parser's teacher), then the fresh arm
reacquires everything in minutes and the ladder replays ARC 2: a head start, then catch-up, no slope.
RULE: nothing hand-coded that solves an expensive shared thing enters either arm's LEARNING loop. The
parser learns from the agent's own counterfactual experiments and from prediction; the classical
extractor exists only experimenter-side (pre-check and reference). Innate grants (section 2) are the
cheap core knowledge a child has; they are disclosed, and none of them is a perception, motor or
danger competence. The WORTH gate (4.4) measures, before any lineage is trained, whether a PERFECT carry
of the learned objects would show "learns faster" at all; if not, the generation is a rung.

1.3 Honest prediction before anything runs: on Atari, seeing may be cheap (a five-line white-centroid
extractor once dominated learned perception on blob-separable games) and the intra-Atari reading is
expected flat. The first real test of the claim is NES entry after Atari (scrolling, tiles, gravity,
hold-length jumps, 8 buttons). FRESH-COST (frames a fresh agent needs to see, find itself and read its
buttons on a new game) is printed at every generation entry and is the first number that says whether
anything is expensive to reacquire.

1.4 Where it is hard: the 2D -> 3D jump (ME becomes the viewpoint; the parser needs an ego-motion head;
stage 4 probes only that); forgetting (every checkpoint is tested on every earlier held-out set);
compute (one RTX 4080, 12 cores; 16-bit emulation is CPU-bound; budgets re-planned from measured
wall-clock per 100k frames at every stage entry); measurement (five voided verdicts: one primary, one
null, one figure, a frozen scorer, every extra number labelled reported-not-required).

## 2. The architecture

INNATE GRANTS, disclosed once, learned by nothing (the child's core knowledge; none is a competence):
  G1 a reset/death event has negative valence (a veto, never a reward)
  G2 a discrete, rare, monotone, action-contingent counter has positive valence
  G3 objecthood = connected stuff that moves coherently (common fate); learning refines it
  G4 self = what my own commands change (efference copy / controllability)
  G5 a motor-babbling reflex on every new controller
  G6 the ledger schema, the 12-slot effect vocabulary, the 10 contact outcomes, the 16-d behaviour
     signature, the archive cell rule, the arbiter constants
  G7 frame preprocessing for ALE: max-pool over the last 2 frames (flicker), frameskip 4
  G8 from stage 1: a global scroll estimate by phase correlation (code, both arms alike; disclosed as
     the one grant that touches perception, and measured as such: FRESH-COST with and without it)
Whole agent ~3M parameters: parser ~1.5M, policy ~1.1M, danger head ~0.2M. Tables are tensors with a
JSON export.

### 2.1 Perception: the counterfactual parser (`parser_v70`) — learns from its own experiments
Why not reconstruction again: six earlier attempts bound objects by reconstruction and never bound the
ball. A counterfactual can: the emulator's determinism is free, exact supervision, and it is the AGENT'S
experiment, not a hand-coded label.
INPUT: native frame, uint8 palette/RGB, 2-frame max-pool, 2-level pyramid (native + half) so 8 px Atari
sprites and 32 px Genesis sprites fall inside one ~40 px receptive field. Nothing in the network is
shaped by H, W, palette size, button count or category count.
NETWORK: fully convolutional U-Net-lite (6 conv blocks 32-64-96, stride-2 down x3, 3 up blocks with
skips, GroupNorm, SiLU, ~1.5M). Per-pixel heads: f(x) in R^32 instance embedding; m(x) foreground;
d(x) in R^2 action-conditioned displacement to t+1; c(x) controllability; s(x) solidity.
TARGETS — all from the agent's own branches and its own predictions; NO classical background model,
NO block-matching flow, NO hand-coded extractor anywhere in the loop (rule 1.2):
  P  PREDICTION: reconstruct frame t+1 from frame t, the effect taken, and the heads' displacement:
     warp(frame t, d) -> t+1, L1 on pixels. Foreground m is what the warp explains by MOVING and
     background is what it explains by STAYING; d is learned motion, not a label.
  C  COUNTERFACTUAL CONTROLLABILITY: from savestate S run effect a for 4 frames and NOOP for 4 frames
     (E1/E2 branches the agent runs anyway); the diff region between the two branches, projected onto
     the tracker's PREDICTED position of the component (not the overlap at t: fast balls move 4-8 px
     per step), is labelled controllable; BCE on c. Pixel-exact; guard: only moving components.
  G  GROUPING: pixels whose predicted displacement d agrees within a component are pulled together in
     f, different components pushed apart (discriminative loss, margin 1.0) — common fate from the
     parser's OWN motion estimate.
  I  IDENTITY: tracked instances across 8 frames pulled together in f (InfoNCE) — the tracker is the
     parser's own.
  S  SOLIDITY (stage 1 on): predicted me-displacement under the chosen effect but zero realised
     displacement and no reset -> the pixels in the predicted path are solid; walls become objects by
     bumping into them.
Data: 100k frames per game under the entry protocol's babbling and play, branches every 16 frames
(charged to the frame axis, 2.5). Compact storage (uint8 frames, bit-packed masks, LZ4 memmap clips)
so 10 games fit in ~20 GB. Training: batch 64, Adam 3e-4, per-game online fine-tuning 1 step per 64
env steps for every arm alike.
INSTANCES: connected components of m > 0.5 (scipy.ndimage.label), split by k-means on f when a
component's embedding variance exceeds a threshold. Descriptor per instance (48 floats): centre and
size as screen fractions, area, velocity, mean f (32), c, s, colour histogram (8). Tracker: Hungarian
on predicted position + embedding, 4-frame memory, coasting to 30 frames, re-match within 8 px logged
as a reappearance. CATEGORIES per game: nearest codebook entry over [f, log size, colour hist], new if
distance > 0.6, cap 64; MERGE when two categories' behaviour signatures and controllability coincide
within 0.2 (animation frames, palette swaps, power-up forms reunite; merge rate logged).
CLASSICAL REFERENCE EXTRACTOR (`ref_extract_v70`), EXPERIMENTER-SIDE ONLY: background = per-pixel
mode over a life, sprites = connected components of non-background. Used for (i) the CPU pre-check that
the RAM harness is right (recall@4 px >= 0.9 on RAM objects), (ii) the reference line printed beside
every perception number. Never a front-end of any arm unless gate 0-B declares perception failed, in
which case it becomes the front-end of BOTH arms as a disclosed grant and "learns to see" leaves the
claim ceiling (pre-declared outcome, 5).
MEASUREMENT THAT IT SEES (RAM, scorer process only, 4.1): recall@4 px (Atari) / 6 (NES) / 8 (16-bit)
of RAM objects that moved in the last 8 frames; precision; identity switches per 1000 frames;
me-accuracy (within 4 px of the RAM player); category purity (NMI vs RAM type); fragmentation ratio
(categories per RAM type, target <= 2). Nulls beside it: a random-init parser probed identically (3
seeds) and the reference extractor. Perception bar P: recall >= 0.8 and me-accuracy >= 0.9.
GENERATION CHANGES: FCN + screen-fraction descriptors absorb resolution; G8 scroll absorbs NES; at
stage 3 grouping uses similarity transforms (dx, dy, log scale) and descriptors gain scale and an
occlusion-order depth rank; parallax = dominant global shift, residual layers become slow solid objects.

### 2.2 Agency and motor (`me_buttons_v70`)
ME = the instance with the highest time-averaged controllability c over the last 60 frames, present at
life start, moving, with lag-1 divergence in the branches (the ball after a paddle hit diverges at lag
>= 2: downstream, not me). Several controllable instances: the longest-persisting is me, the others are
"mine" (a policy input). Hysteresis 20% for 32 frames. No controllable instance for 120 frames = a
non-play state: emit start/advance and wait. Measured: me within 4 px of the RAM player on >= 0.9 of
frames where the player is on screen. Games with no visible self are excluded by a rule written before
any result, exclusion rate reported.
BUTTON-TO-EFFECT DISCOVERY (G5), on entry and whenever controllability collapses: from 16 archived
savestates, every single button and pair (Atari 18 actions; NES 8 + 28 pairs; 16-bit pruned to ~60 by
sampling), 8 frames held vs NOOP; effect vector on me (mean dx, dy, the up-then-down jump signature,
size change, a spawned instance within 24 px, global background shift, nothing); clustered into the 12
EFFECT SLOTS carried across games (move +x -x +y -y, jump, spawn, size-change, start/advance, none, 3
free). The table effect -> buttons is the per-controller adapter, keyed from stage 1 by a coarse
me-state (grounded / airborne). THE POLICY NEVER SEES BUTTONS: its action space is the 12 effects; a
controller change is a babbling event. Cost ~9k frames (seconds). Measured: table agreement >= 0.9 with
a hand-written truth table per held-out game; frames-to-table printed and must FALL across generations
if the effect vocabulary transfers.

### 2.3 Drives (`drives_v70`): no reward; four signals; an arbiter, not a sum
  SURVIVAL, a veto. Reset detector: >= 60% of instances vanish and the object graph returns within 30
    frames to a configuration hashed at a previous life start, OR the lives cell decrements. Danger head
    (BCE, death within 30 frames, ~0.2M, trained on replay) vetoes options with p_death > 0.5. Veto-lift
    (the D3 absorbing-state repair): if every option is vetoed, lift the veto; in lab mode return to an
    earlier archive anchor, in LIVE mode act uniformly over effects. Death BUDGET: 5 intentional deaths
    per 100k frames, spent only by E3. Validated: F1 >= 0.9 vs ALE lives within +/-30 frames.
  SCORE, self-taught (G2): the ODOMETER READER. Candidate regions = static-position foreground cells
    whose glyph content changes in discrete, sparse steps; glyphs are pixel-exact so each gets an id by
    template match; digit ORDER is discovered from the odometer structure (the rightmost changing cell
    cycles with period <= 10; the cell to its left changes exactly when the rightmost wraps); value =
    sum digit_i base^i. Timers rejected (action-independent constant-rate change: identical across
    counterfactual branches). A small cell that decrements at reset is the lives counter. Enabled only
    after a self-consistency test (non-decreasing within a life on >= 95% of frames, >= 3 carries);
    switched off if violated. Validated experimenter-side against RAM (r >= 0.99, exact >= 0.95), NEVER
    corrected from it. Games without a valid reader run on survival + LP + novelty and are marked.
  LEARNING PROGRESS: per ledger cell, the decrease over the last 20 visits of the predictive entropy of
    its Dirichlet posterior; an intrinsically random cell stops paying (the noisy-TV closure).
  NOVELTY: 1/sqrt(count) over archive cells (category set present, me position on an 8x8 grid, score
    bucket, scroll bin, lives). Object-level, never pixels.
  ARBITER: options = (goal, target), goals in {reach, avoid, hit-with-mine, score-up, survive,
    experiment(cell), return(anchor)}; each drive's value per option z-scored over a 10k-frame window;
    salience = LP 1.0 + novelty 0.7 + score 1.0 (once the reader passed), survival veto first;
    winner-take-all with hysteresis (0.3 sigma) and minimum dwell (to termination or 64 frames); tonic
    temperature = smoothed LP (progress -> exploit, stagnation -> explore). Score-seeking emerges late.
    The activity sequence is logged: the agent's own curriculum, the demo's second panel.
  PATHOLOGY DETECTORS (three, printed at every gate, never auto-corrected mid-run): noisy TV (LP flat
    while novelty stays high on one cell), death loop (deaths per 1k frames must fall >= 50% from the
    first to the last 100k), stereotypy (fraction of bouts on one option <= 0.5). The POSITIVE noisy-TV
    validation (an injected 32x32 noise panel gets < 5% of option dwell under LP and > 30% under a
    raw-prediction-error ablation) runs at gate 1, where the ledger and options exist; the panel is a
    seeded screen-byte transform inside the harness so the leak test still holds.

### 2.4 Memory and skills (`ledger_v70`, `policy_v70`)
Per game, EMPTY at entry for every arm (nothing in these crosses a world):
  CODEBOOK[game]; LEDGER[game]: key (cat_A, cat_B, approach_dir in 8 bins) -> {n, outcome_counts[10],
  n_intervention, n_observation}; outcomes {A vanishes, B vanishes, both, A bounces, B bounces, A stops
  (solid), reset, score up, lives down, nothing}; written on every detected contact, with the
  intervention flag when produced by E2. EFFECT TABLE[game]; ARCHIVE[game] (Go-Explore cells -> best
  savestate, cap 20k cells / 2 GB, LRU by novelty); background, scroll, reader glyph map.
Carried across games (the lineage):
  ROLE PRIOR: a 16-d BEHAVIOUR SIGNATURE per category (speed mean/var, chases-or-flees correlation,
    periodicity, controllability, solidity, spawn rate, size fraction, vertical vs horizontal motion,
    contact-death rate, contact-score rate, vanish-on-contact rate, permanence) quantised by a
    64-entry learned VQ -> Dirichlet pseudo-counts alpha[10] over contact outcomes, updated by counting
    after every game; the prior of every new ledger row. Its worth is countable: contacts needed before
    a ledger row predicts at 0.9 accuracy, with vs without the prior. Its expected SATURATION (games
    until > 90% of VQ cells have n >= 10) is printed at gate 1 so "flat after X_1" is a prediction.
  POLICY pi(effect | tokens, goal): object-graph transformer, 4 layers, d 128, 4 heads, ~1.1M,
    permutation-invariant over <= 32 instance tokens. Token = geometry relative to me (dx, dy, dvx, dvy,
    w, h, scale, depth rank), controllability, solidity, mine, permanence, the role-prior vector (10),
    the ledger's predicted outcome vs me (10), log counts. NO category id, NO appearance embedding, NO
    button. Goal token: type one-hot + target signature (16) + target cell (2). Unit tests: invariant to
    permutation of category indices and of the button table; no parameter shaped to H, W, n_buttons,
    n_categories.
  DANGER HEAD; drive normalisers; arbiter constants; hold-length priors.
  PROCEDURES (code, both arms alike, disclosed as capabilities, never claimed as transfer): me-finder,
    button discovery, reader, reset detector, experiment selection, scroll estimate.
SKILLS by HINDSIGHT SELF-IMITATION (the only policy signal; no critic, no reward): every trajectory is
segmented at events (score up, death, new cell, vanish, contact); the last H = 32 steps before each
achieved event are relabelled with the goal achieved; credit ONLY under the commanded goal
(`relabel_commanded`, 0.7^lag); CE, buffer 500k uint8 rows, batch 512, 300 steps per 50k frames; 5%
uniform exploration over effects; eval with a context mask and fallback (D3). ON-POLICY CORRECTIONS:
when a commanded skill fails, or on any death, the E3 rewind's surviving sequence is written as a
DAgger-style label, so the policy is trained where it fails. Measured: option success at game ENTRY
(before in-game training) vs a fresh net; rising across generations = a transferred motor vocabulary.

### 2.5 Active experimentation (`lab_v70`): savestates as an organ, every frame paid
The DEFAULT agent does not plan by branching at act time (removes the 3-12x parser multiplier and the
"competence lives in the emulator" failure by construction). Four experiments, all charged:
  E1 BUTTON DISCOVERY (2.2), ~9k frames at entry.
  E2 CONTACT TEST: the ledger cell with the highest expected information gain (Dirichlet entropy x
     reachability x LP); load the anchor where cat_B is nearest; reach(cat_B) as INTERVENTION and
     avoid/NOOP as CONTROL from the same state, 2 branches each, <= 60 frames; write with the flag;
     restore. Measured: ledger accuracy vs RAM outcomes split by intervention vs observation; the
     intervention split must be more accurate or the engine is not earning its frames.
  E3 ONE DEATH, ONE LESSON: on death at t, the suspect = the fact written in [t-15, t] with the highest
     prior death rate; restore t-k for k in {8, 16, 32} and run avoid(suspect); survival in >= 2 of 3
     rewinds sets the fact's confidence to 1; else clear the suspect. The surviving rewind is a hindsight
     AVOID demonstration. <= 100 frames per death. Measured: deaths by the same fact before its death
     rate falls below 10%, target <= 3, vs a no-rewind ablation (the contemporaneous control).
  E4 RETURN TO FRONTIER: Go-Explore return to the least-visited cell when LP and novelty stall 5k frames.
Budget: experiments <= 20% of frames in the first 500k of a game, decaying to 5%, hard cap 30%; the
parser's C branches (every 16 frames) are inside this budget and are charged.
ACCOUNTING, frozen: every emulated frame counts on the primary's axis, branches and rewinds included,
plus a 60-frame charge per savestate load. COMPETENCE is read ONLY in live mode (fresh emulator,
savestates off, single life sequence, the demo code path, ALE's 108k-frame episode cap). The
savestate-assisted score is reported beside it; a live/lab success ratio < 0.5 is the savestate-
addiction detector.

### 2.6 Transfer: what crosses generations, what must not, and the mechanism stated so it can fail
CARRIED: parser; policy; danger head; role prior (the object that grows); drive normalisers; arbiter
constants; hold-length priors; procedures as code. EMPTY AT ENTRY for every arm: codebook, ledger,
effect table, archive, background, scroll, reader map, optimizer. ENTRY PROTOCOL, identical across arms:
(1) 20k frames of babbling -> effect table; (2) me-finder; (3) reader search; (4) category discovery,
ledger rows from the role prior (fresh: uniform); (5) play under the arbiter.
ENFORCEMENT: no nn.Embedding over games, buttons or sprites; no parameter shaped to resolution, button
count or palette; the leak test asserts the lineage export is byte-identical under permutation of the
source game's category ids, palette and button layout; the parser is DISCLOSED as the one carried
module that encodes appearance.
MECHANISM of "learns faster" on a new game: a parser that already segments (head start on frames-to-P
AND a faster online adaptation, measured); a role prior that predicts a hazard from its motion before it
is touched (contacts saved); a policy that already maps "high death-rate signature approaching" to
avoid and "contact-then-score signature" to reach. Mechanism PROBES on a burned game, frozen weights,
before any confirmatory (the ARC 3 4.3 lesson), each a 3-se rule vs random init: (a) first-1000-action
avoidance of RAM-lethal objects; (b) frames-to-P, carried vs fresh parser; (c) contacts-to-0.9-ledger
with vs without the role prior; (d) which component: S (parser only), R (knowledge only), P (placebo:
permuted signatures, shuffled pseudo-counts) run on the SAME burned game, caption only. A probe that
fails is an architecture result fixed before the confirmatory, never after.
PRETRAINED DOWNLOADS: against. An ImageNet/CLIP encoder is statistically mismatched to 8 px sprites
and would erase the largest expensive-shared thing from the accumulation curve; an LLM would inject
walkthrough knowledge (the recipe-oracle defect of ARC 1). Permitted, disclosed, never arms: a frozen
pretrained encoder as a CEILING reference at stage 2; an LLM as an offline reader of the ledger.

## 3. The ladder

3.0 STAGE 0 (instrument + gate 0): ale-py 0.12.1 (cp314 wheel verified), OCAtari RAM decoders in a
Python 3.12 venv scorer process (OCAtari pins numpy < 2; fallback AtariARI hand-port with per-game
pixel calibration), emulator pool with branch/restore, reference extractor, parser, me/buttons/reader/
reset, leak test. Produces the parser weights, the gate-0 numbers and FRESH-COST(0).
3.1 STAGE 1, Atari 2600 (160x210, 18 actions): train Pong, Breakout, SpaceInvaders, Freeway, Seaquest,
MsPacman, Asteroids, Kangaroo; held-out (6, burned, OCAtari-covered, named at the freeze, DISJOINT from
gate-0's held-outs): candidates Tennis, Krull, Enduro, Boxing, Frostbite, Riverraid minus whatever gate 0
opens. Gates 0-A, 0-B, 1, 2 (section 4.4); two lineages with different game orders. NO intra-Atari
confirmatory: the intra-Atari reading is the predicted rung and the claim's first test is NES; the ~90
GPU-h it would cost go to stage 2. Stage 1 reports FRESH-COST, the K* WORTH result and the probes.
3.2 STAGE 2, NES (256x240, D-pad + A/B/start/select, scrolling, tiles, gravity, hold-length jumps,
lives digits, levels). New for both arms: G8 scroll + world-coordinate object memory, 2-context effect
table, target S. Emulator: stable-retro is Linux-only -> the emulator pool runs under WSL2 with the
SAME pool API as stage 0 (designed for a socket boundary from week 1); ~1 week. Train (6): SMB,
BalloonFight, IceClimber, KungFu, Galaga, Excitebike; held-out (4-6): DonkeyKong, Castlevania, Contra
(level 1), AdventureIsland + two if decoders exist; RAM decoders (player x/y, score, lives; enemies for
2 games) committed BEFORE the freeze; the owner's human baseline (3 x 30 min per held-out, median)
recorded before the freeze. Gate 3 = gate 0 on NES + the jump effect discovered on 3/4 games. THE
FIRST CONFIRMATORY: X_1 (after Atari) vs Fa, Fb on the NES held-outs (4.3), plus the checkpoints of X_1
after 2 / 4 / 8 Atari games on the same games (does the head start grow with Atari games, before any
NES training). Prediction: P(DEMONSTRATED) ~0.35-0.45; INCONCLUSIVE modal.
3.3 STAGE 3, SNES / Genesis (parallax, sprite scaling, 6-8 buttons): similarity-transform grouping,
scale and depth rank, parallax layers. Train (4): Sonic, StreetsOfRage, SuperMarioWorld, GradiusIII;
held-out (4) fixed at the freeze from ROM availability; Mode-7 titles reported, never held-out. Arms:
X_2 (Atari+NES), X_1, X_1-ext (frames-matched: same total frames on more Atari), Fa, Fb. The
accumulation curve's third point; also tested on the Atari and NES held-outs (forgetting).
3.4 STAGE 4, PC 2D hooks (screen-grab + key injection; mouse = two continuous effects) and the 2D -> 3D
probe (ViZDoom-class, ~10 GPU-h): the parser gains an ego-motion head; controllability goes GLOBAL and
ME becomes the viewpoint; contact/approach re-grounded on (bearing, apparent-size rate). Outside the
ceiling; the sprite parser likely needs a depth-aware redesign, said now.

## 4. Measurement

4.1 GROUND TRUTH: RAM (OCAtari/AtariARI object lists, score, lives; stable-retro data.json + hand
decoders; 200 owner-annotated frames per NES/16-bit held-out for enemies) is read ONLY by a scorer
process on the same emulator; the agent process receives screen bytes and its own outputs, asserted by
a unit test that hashes the observation source (leak test). Allowed: perception scores, me/reader/reset
validation, competence scoring, ledger audit, gate numbers. Forbidden: any observation, loss, drive,
curriculum choice, or threshold set after seeing transfer data.
4.2 COMPETENCE: HN = (score - random) / (human - random), uncapped; Atari human from the published
Atari-100k tables (recorded WITHOUT sticky actions: disclosed; random and the RAM-object ceiling are
re-measured by us UNDER the test convention, sticky p = 0.25, and HN is reported relative to both);
NES/16-bit human = the owner's recorded play. Bars C1 0.10, C2 0.25 (the bar of record for gates and
the figure's line), C3 1.0. Live mode, 20 episodes per evaluation point; se_res per game is MEASURED at
G_b from the eval-episode sd, never asserted.
4.3 PRIMARY (monotone; pointwise dominance can never score lower), per held-out unit u:
    m(u, t) = clip(HN(u, t), 0, 1)   (capped at C3 = human, which the ceiling says is never reached, so
              no arm saturates inside the ceiling — the instrument auditor's fix; C2 is NOT the cap)
    t in T = {0.25, 0.5, 1, 2, 4} M frames on Atari (x2 NES, x3 16-bit), every emulated frame counted
    A_arm(u) = mean over t of m_arm(u, t);   Delta(u) = A_X(u) - 0.5 (A_Fa(u) + A_Fb(u))
    Delta_learn = mean over units;  se_null = sd_u[A_Fa(u) - A_Fb(u)] / sqrt(2N), A_Fa(u) = the MEAN
              over that unit's fresh inits (3-5; stated here to fix the unit); se = max(se_null, se_res)
    DEMONSTRATED iff Delta_learn >= 2 se AND Delta_s > 0 for each lineage s (per-lineage clause);
    NOT-DEMONSTRATED iff Delta_learn <= 0; INCONCLUSIVE otherwise. Instrument line: |t_init| >= 2 ->
    INCONCLUSIVE. THE UNIT SET IS THE WHOLE HELD-OUT SET, fixed at the freeze; "learns-faster game" vs
    "reach game" is a caption label that never changes N or the units entering Delta or se_null.
    WORDING RULE (caption only): "learns faster" iff the FRACTION of inits reaching C2 by checkpoint t,
    censored at budget, for X dominates fresh at >= 2 of the checkpoints t >= 0.5M by 2 se of its own
    fresh-vs-fresh null (not bounded by a head start); otherwise "uses what it learned". Delta_0 (after
    the entry protocol, frozen weights) is drawn and labelled "head start", never in the verdict.
    Reported: frames-to-C2 as a fraction of inits per checkpoint; uncapped log-frame AUC of HN.
4.4 GATES PER STAGE, each frozen in the commit that runs it:
    G_a INSTRUMENT: perception, me, reader, reset, throughput vs RAM (section 5 for stage 0). A failure
        is an instrument defect: fix, log, re-run, at most twice; then STOP with the numbers.
    G_b FEASIBILITY on the held-out set (3 fresh inits, full budget, live eval):
        REACHABLE     the RAM-object ceiling arm (parser replaced by RAM object lists) reaches C2 on
                      every gate game;
        FRESH-LEARNS  >= 2 of 3 fresh inits reach C2 within budget on >= 2/3 of gate games (a fraction,
                      never a median);
        WORTH (replaces the procedure-oracle ROOM; the wall auditor's fix): K* = the PERFECT-CARRY
                      ceiling: the carried objects (parser, policy, danger head, role prior) trained to
                      convergence on the held-out game ITSELF, ledger and prior pre-filled from
                      RAM-scored contacts, then FROZEN and run through the identical entry protocol vs
                      Fa/Fb on the frozen primary — it upper-bounds what any lineage could carry.
                      PROCEED iff A_K* - A_fresh >= 4 se_proj AND K*'s wording-rule fraction dominates
                      fresh (a perfect carry would be called "learns faster"); otherwise the generation
                      is a RUNG by arithmetic, captioned so, and the confirmatory does not run. ~1 GPU-h
                      per unit. FRESH-COST is printed beside it as a gate number: the frames a fresh
                      agent needs to pass G_a's bars must exceed the frames spanned by the first two
                      checkpoints, else the reading is a rung by construction.
        One pre-declared softening notch per stage (C1 instead of C2), then STOP.
    G_c VERDICT (4.3) with arms X, Fa, Fb only; K* and the RAM ceiling reported beside them.
4.5 ACCUMULATION: X_1, X_2, X_3 from each of 2 lineages tested on every generation's held-out set,
beside Fa/Fb and the frames-matched control X_k-ext. DESCRIPTIVE, no verdict label (ARC 3's rule:
the area cannot tell compounding from ceiling on its own); the reading rule is pre-declared: "rises"
iff Delta(X_k) increases in k beyond 2 se while Delta(X_k-ext) does not; "flat after X_1" is called
"does not compound" only if A_X1(u) < 0.5 on the majority of units (room exists), else "ceiling-limited".
4.6 SEEDS AND BURNS: per-unit init seeds by (stage, unit, arm); emulator seeds identical across arms;
held-out sets named at the freeze and never opened before the confirmatory; every game a human debugged
on is burned; gate-0 held-outs are disjoint from stage-1 held-outs. A frozen scorer (`score_v70.py`)
accepts only the (N, budget) pairs declared at the freeze, committed before any arm exists.

## 5. GATE 0, the first cheap gate, in full (<= 12 GPU-h, ~2-3 engineering weeks)

GATE 0-A "DRIVE PRIMITIVES ON THE REFERENCE EXTRACTOR" (no learning; < 2 GPU-h). Games: Pong,
Breakout, Freeway, Seaquest, MsPacman, Asteroids. Runs: 200k frames per game of babbling + arbiter play
from 16 savestates with branches; the reference extractor supplies instances (experimenter-side
front-end for THIS gate only, so drives and perception fail independently). Pre-check: the extractor
reaches recall@4 px >= 0.9 on RAM objects (else the RAM harness is wrong, not the learner). Measured,
thresholds frozen now:
    (a) THROUGHPUT >= 2000 agent-steps/s (frameskip 4, 16 envs, extraction, label branches and scorer
        included; the label harness is benchmarked separately); below 1000 the engineering is fixed
        before any GPU;
    (b) ME within 4 px of the RAM player on >= 0.9 of visible frames in >= 5/6 games (Asteroids may miss);
    (c) READER passes self-consistency and reads with r >= 0.99 vs RAM in >= 4/6 (Pong may miss);
    (d) DEATH detector F1 >= 0.9 vs ALE lives within +/-30 frames in >= 5/6;
    (e) BUTTONS: dominant effect agrees in sign with RAM delta-x/y on >= 0.9 of probe trials in 6/6;
        table stable within 20k frames;
    (f) LEAK: the observation tensor hashes to a pure function of screen bytes; the lineage export is
        byte-identical under palette / category / button permutation;
    (g) reported, not required: a stage-0-computable noisy-TV line (option dwell on an injected panel
        under the archive-novelty drive alone), the real test being gate 1's.
PASS = (a)-(f). A miss is an instrument fix and re-run, at most twice.
GATE 0-B "THE PARSER SEES, FROM ITS OWN EXPERIMENTS" (<= 10 GPU-h). Data: 100k frames per train game
(the six above) under babbling + play with C branches every 16 frames. Train on P, C, G, I (S is stage
1), 3 seeds, ~2 GPU-h per seed pooled. Held-out for THIS gate (burned for it, disjoint from stage 1's):
Boxing, Frostbite, Riverraid, Kangaroo. Measured, thresholds frozen now:
    (1) recall@4 px of RAM objects that moved in the last 8 frames >= 0.80 on >= 5/6 train games, AND
        the ball on Pong and Breakout >= 0.70 (the exact failure of the six earlier attempts);
    (2) me-accuracy >= 0.90 within 60 frames of each life start on 6/6;
    (3) NULLS: (1) exceeds a random-init parser probed identically by >= 3 sd over seeds; the reference
        extractor's recall is printed beside it (context, not a bar: rule 1.2 forbids it as a teacher,
        so matching it is not required);
    (4) fragmentation ratio <= 2 and identity switches < 5 per 1000 frames (reported at gate 0, gated
        from stage 1);
    (5) IS SEEING EXPENSIVE (reported; decides the claim's shape, not PROCEED): zero-shot recall on
        >= 3/4 held-out games >= 0.5; FRESH-COST(0) = frames for a fresh parser to reach recall 0.8 per
        held-out game (3 inits) beside the carried parser fine-tuned; "perception is expensive" iff
        the carried/fresh ratio <= 0.5 beyond the fresh-vs-fresh 95th percentile on >= 3/4 games.
Outcomes: (1)-(3) pass -> the parser is the front-end of stage 1; (1)-(3) fail on Pong/Breakout only ->
one pre-declared notch (motion-energy weighting of foreground), re-run once; (1)-(3) fail otherwise ->
learned perception closes a seventh time with its numbers, the reference extractor becomes the
front-end of BOTH arms as a disclosed grant, "learns to see" leaves the claim ceiling, and the
ledger / role-prior / policy claim carries alone; (5) fails while (1)-(3) hold -> seeing is cheap on
Atari, dropped from the Atari rung's transfer claim, re-read at NES entry.
PROCEED to stage 1 iff 0-A passes and 0-B has resolved into a declared outcome. Predicted: 0-A passes
on the first or second attempt (P ~0.7); 0-B (1)-(3) with P ~0.4 (six prior failures; one genuinely new
signal and no classical teacher this time); (5) "seeing is expensive on Atari" P ~0.3.

## 6. Implementation plan, stage 0 (nothing on GPU before step 6)

Runtime (2026-09-08): Python 3.14.3, torch 2.11.0+cu126, RTX 4080 16 GB, 12 cores; ale-py 0.12.1
cp314 wheel; OCAtari needs a 3.12 venv (numpy < 2) -> the scorer is a separate process anyway;
stable-retro Linux-only -> WSL2 for stage 2, the pool API socket-ready from week 1.
Files (new; frozen files of earlier arcs untouched; ARC 3's relabel, permutation tests, scorer
skeleton reused):
  ragnarok/environments/emu_pool.py   worker pool over ale-py; frameskip 4, 2-frame max-pool;
                                      clone/restore/branch API; per-env RNG; frame counter incl. charges;
                                      socket boundary so the pool can live in WSL2 later
  ragnarok/environments/scorer_v70.py separate process (3.12 venv): RAM decoders, HN, object lists,
                                      lives, score; refuses to run in the agent process; the leak hash
  scripts/ref_extract_v70.py          background mode over a life, CC sprites (experimenter-side)
  scripts/parser_v70.py               U-Net-lite, heads f/m/d/c/s, targets P/C/G/I/S, branch harness
  scripts/track_v70.py                CC instances, k-means split, Hungarian tracker, descriptors,
                                      codebook, behaviour signature, merge rule, fragmentation
  scripts/me_buttons_v70.py           me-finder; babbling; effect vectors; 12 slots; 2-context table
  scripts/reader_v70.py               odometer reader, timer rejection, lives cell, reset detector
  scripts/drives_v70.py               LP, novelty, danger head, arbiter, veto-lift, death budget,
                                      three detectors, noise-panel transform
  scripts/ledger_v70.py               ledger with intervention flag, role prior (VQ + Dirichlet),
                                      archive with byte cap, JSON export with thumbnails, leak test
  scripts/policy_v70.py               object-graph transformer, goal token, hindsight relabelling,
                                      DAgger buffer, context mask, permutation tests
  scripts/lab_v70.py                  E1-E4, EIG selection, frame accounting, live/lab split
  scripts/gate0a_v70.py, gate0b_v70.py, gate1_v70.py, gate2_v70.py (K*, FRESH-COST), lineage_v70.py,
  probe_v70.py, confirm_v70.py, score_v70.py (frozen), figure_v70.py, test_v70.py
  preregistration.md                  the v70 entry: gate-0 thresholds above, seeds, burned sets
Order: week 1 pool + scorer venv + reference extractor + throughput bench + leak test (0 GPU-h); week 2
parser + tracker + me/buttons + reader + drives primitives, gate 0-A (< 2 GPU-h); week 3 gate 0-B
(<= 10 GPU-h); weeks 4-6 ledger, role prior, policy, lab, gate 1 (~15 GPU-h); weeks 7-9 gate 2 with K*
and FRESH-COST (~15 GPU-h), two lineages (~16 GPU-h), probes (~2 GPU-h). Stage 0-1 total ~60 GPU-h.
Stage 2 ~120 GPU-h, stage 3 ~150, stage 4 probe ~10, all re-planned from measured wall-clock.

## 7. Risks and the observation that reveals each early
  R0 everything expensive turns out to be a procedure or cheap (the wall, at pixel level)    K* WORTH
     gate, FRESH-COST vs the first two checkpoints, gate 0-B (5)
  R1 the parser fails to bind small fast objects (a seventh time)      gate 0-B (1) on Pong/Breakout
  R2 seeing is cheap on Atari                                          gate 0-B (5); re-read at NES
  R3 contact knowledge is cheap (a fresh ledger fills in a few hundred contacts)     probe (c)
  R4 reader coverage (bars, icons, two-sided scores)                   gate 0-A (c); exclusions counted
  R5 category fragmentation (Mario small/big/fire)                     fragmentation ratio at every G_a
  R6 scrolling / parallax break background and memory                 stage-2 week-1 recall
  R7 drive pathologies                                                 three detectors; gate-1 panel test
  R8 competence lives in the emulator                                  live-only eval; live/lab ratio
  R9 throughput (v49 died at 15% GPU)                                  gate 0-A (a)
  R10 bimodal fresh inits                                              fractions over inits, N >= 6
  R11 role prior misaligned across generations                         refutation rate on the first NES game
  R12 forgetting                                                       every checkpoint on every earlier set
  R13 ground-truth cost (decoders, owner baselines)                    no decoder + baseline = not a held-out
  R14 platform (WSL2, ROMs, OCAtari on 3.12)                           week 1
  R15 measurement complexity (five voided verdicts)                    one primary, one null, one figure

## 8. What is novel and what has cousins, honestly
COUSINS: object discovery from motion/reconstruction (MONet, SPACE, SAVi; this repo's SpriteAE);
agent localisation by controllability (Bellemare, Veness, Bowling 2012; Choi et al. 2019); controllable
factors (Bengio); action-effect learning and motor babbling (Lungarella, Oudeyer, Chandak 2019);
learning-progress curiosity (Schmidhuber 1991; Oudeyer IAC; Colas IMGEP); count-based novelty and
savestate archives (Go-Explore); basal-ganglia arbitration (Gurney, Prescott, Redgrave 2001); schema
stores with counted outcomes (Drescher 1991; Schema Networks; OO-MDPs; this project's stores);
hindsight relabelling and self-imitation (HER, SIL; this project's v53-v55); DAgger; goal-conditioned
relational policies; backtracking from savestates (Backplay); RAM ground truth (AtariARI, OCAtari);
Atari-100k human baselines. DreamerV3, MuZero, Agent57 and Go-Explore-as-recipe are what this is NOT.
NEW IN THIS COMBINATION, each measured and refutable: (1) savestate counterfactuals as pixel-exact,
label-free targets for controllability and grouping, resolution-agnostic, with NO classical teacher in
the loop (gate 0-B); (2) one controllability statistic that yields ME in 2D and the viewpoint in 3D
(stage 4); (3) the effect vocabulary as the policy's entire motor interface with a per-controller table
rebuilt by a fixed experiment (frames-to-table falling across generations); (4) the odometer-carry
score reader without OCR or labels, validated against RAM and never corrected from it; (5) a
category-keyed ledger with an intervention flag written by branch pairs, plus one-death-one-lesson
rewind attribution vs a no-rewind ablation; (6) the role prior as the object that accumulates, its
worth counted in contacts; (7) the measurement package on console generations: live-only competence
with every frame charged, the PERFECT-CARRY ceiling before budget, a frames-matched control, a
contemporaneous fresh-vs-fresh null, RAM-probed perception, fraction-over-inits gates, a frozen scorer.
The closed loop pixels -> categories -> ledger by experiment -> identity-free skills -> next console,
gated cheaply at every rung, is what the second goal will be judged on.

## 9. What is NOT claimed
Not human-level; not state-of-the-art sample efficiency (Atari-100k SOTA is reward-trained, context
only); not "any game"; not a full game (first levels, 0.25-of-human bars, partial success counts); not
PC or 3D competence; not understanding; not that the parser is a general vision system or the role
prior a theory of physics; not that G1-G8 are learnable; not that hand-coded ceilings are beaten; not
that pretrained encoders would not score higher. A head start is "uses what it learned"; only the
censored-time fraction rule earns "learns faster". If gate 0 STOPs, the deliverable is a seventh,
sharper map of why learned perception on Atari is not the bottleneck; if a later gate STOPs, an agent
with an inspectable ledger and a rung-by-rung measured account of why reuse did not pay, published as
ARC2_PLAN 8/11/15 and ARC3_PLAN 10/11 were.

## 10. Audit ledger (first panel on the synthesis; this document's answers)
  wall (fatal): expensive things were procedures for both arms + a classical teacher for the parser
       -> rule 1.2; parser targets P/C/G/I from own branches and predictions only; classical extractor
       experimenter-side; K* WORTH gate; FRESH-COST vs the first two checkpoints; G8 measured as a grant
  instrument (major): primary saturated at C2, slope bounded by head start, unit set chosen after data,
       no per-lineage clause, se_res asserted, sticky-action mismatch, gate-0/stage-1 held-out overlap,
       live absorbing state -> 4.2-4.6 as written
  feasibility (major): noisy-TV STOP criterion needed week-7 machinery; OCAtari numpy < 2 on 3.14; data
       volume; T3 projection geometry; stable-retro Linux-only -> gate 1 for the panel test, 3.12 scorer
       venv, 100k frames compact, tracker-predicted projection, WSL2 pool with a socket API from week 1
  cuts from the synthesis: intra-Atari confirmatory, negative-control lineage, branching-planner
       ceiling, dynamics model, S/R/P as confirmatory arms (now probes), three of six detectors
