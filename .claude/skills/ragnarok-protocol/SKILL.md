---
name: ragnarok-protocol
description: Use when designing, freezing, running or interpreting an ARC 4 experiment — any claim that the Ragnarok agent plays well, learns, learns faster, or transfers — including offline benchmarks of policies and pilots.
---

# ARC 4 experiment discipline

Goal (owner): an agent that learns any game from the screen and buttons and learns each new game
FASTER thanks to what it learned before. Innovation is wanted; honesty is non-negotiable.

## Never
- Put oracle information in the agent's observations or labels: RAM, emulator reward, `env.true_score()`,
  `env.lives()`, the 4th element of `env.clone()` are evaluator-only. Exception (owner decision
  2026-10-22, v83 on): score increments and the lives count may be given as SENSES (like a human reading
  them); every result using them must say so; nothing else (RAM, positions) without asking Jérémie.
  Reserved games (boxing, frostbite, riverraid, kangaroo) are never opened.
- Change a criterion, threshold, seed set or arm after seeing results. Never drop a negative result.
- Decide from a pilot of a few games: a 10-game pilot gave 462, the frozen run 387 for the same idea.
- Call a head start "faster learning" (compare climb costs between matched levels; use a placebo arm).

## Before a live measurement
1. Write `docs/ARC4_VNN_PROTOCOL.md` (French): what changed and why (evidence), method, games, arms,
   FRESH training and evaluation seeds (never reuse seeds seen in development), evaluation conditions
   (sticky 0.25, 10 games per eval seed, 7000 steps max), baselines, criterion, resources.
2. Baselines on the same evaluation games: every constant button, uniform random, random with repeat,
   the untrained policy through the agent's own interface. The bar is the BEST CONSTANT BUTTON.
3. Criterion (v74 and later): an arm beats the best constant on a game if EACH training seed's 20 games
   exceed the constant's mean by > 2·sqrt(se_seed² + se_const²), complete data required; success needs
   >= 2 games of 6. Pooled tests are secondary only.
4. Run the pre-checks that validate any new agent-side signal against the evaluator on separate seeds
   (example: `scripts/check_death_v73.py`), write their numbers into the protocol.
5. Get an adversarial review of the code (skill `ragnarok-review`), fix, smoke-test end to end on
   1-2 games, then commit code + protocol BEFORE launching (`git commit --only` the relevant files,
   footer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).

## While it runs
- A defect found mid-run: stop the queue, append an "Interruption" section to the protocol (cause,
  affected outputs), commit the partial results as they are, fix in the next version.
- Resource changes (gates, lazy imports) are allowed mid-run; anything touching the agent is not.

## Offline benchmarks (distillation, transfer)
- Group folds by shared start state or identical root screen (deterministic emulator: rollouts from one
  start coincide); measure gain over the BEST CONSTANT DEVIATION, not over the base (Assault: two thirds
  of an apparent gain were a state-independent "pause once"); cluster-bootstrap intervals; several seeds.
- Check that the inputs can see what decides the action (Assault's heat gauge was outside the global view).
- Check branch values are not selection noise (`scripts/check_branch_noise_v74.py`).
- Offline results are development evidence only; competence is established live under a frozen protocol.

## Reporting
Short, plain French for Jérémie: what was established, what failed, what is next. Distinguish
feasibility, transfer and faster learning. Update `docs/CODEX_HANDOFF.md` at milestones (keep it short).
