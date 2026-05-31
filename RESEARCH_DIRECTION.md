# Ragnarok — Research Direction (authoritative anchor)

*This file is the single source of truth for WHERE WE ARE and WHERE WE GO. Keep it
honest and current. Read it first every session.*

Last updated: 2026-05-31 (after phase-gate review #2).

---

## NORTH STAR (the goal, unchanged)
An agent that can **learn anything and reuse what it learned RELIABLY in new
contexts**. Concretely: drop it on ANY game, it learns to win **from pixels**;
it **accumulates notions/skills**; it **uses prior knowledge to solve new things
faster**; eventually beyond games (maths, language). The bar the owner set:
**without reliable reuse-in-new-contexts, accumulating skills is pointless.**

## THE ONE TEST THAT MATTERS (definition of "it works")
From **pixels**, on a **real game/task**, show that a **pre-learned notion** makes
a **NEW** game/task reach competence in **measurably fewer episodes** than from
scratch — **reliably** (≥3 seeds), ideally with **recognition** (pick the right
notion) and **learn-when-novel**. A positive here = come back to the owner.

---

## HONEST STATE (what is actually true, post-review)

### Solid, reproduced, survives review
- Learn games from **pixels** (Pong/Breakout/Snake/Tetris win/score). (v15)
- A developmental **recognise-or-learn loop** runs in the abstract/concept domain
  and on known games (recognise which known game from pixels -> reuse its skill).
  (v21-v25). Cost grows sublinearly with DISTINCT games. (v26)
- **Concept-granularity compounding** is the project's strongest "more knowledge ->
  faster" evidence: learn a FACTORED concept on a hard task -> big speed-up. (v17b
  ~5-15x on Tetris; v13b impossible->~10 iters). FACTORED, not monolithic.
- **variety -> generalisation** holds in concepts (v19) and symbolic maths (v33,
  clean curve). It is classic meta-learning; value = the thesis spans domains.

### Honest negatives / retractions (these make the rest trustworthy)
- Blind **cross-game representation transfer** is weak/unreliable (v31, v35b).
- A stored notion does **NOT** help when the task is **easy to solve from scratch**
  (v36/v37 nulls). PRINCIPLE: **reuse pays only when re-deriving the notion in the
  new context is HARD** (pixels / partial-obs / deep composition).
- The Pong "variety -> anticipation" mechanism was **FALSIFIED** (v32); that arc was
  mundane domain randomisation.
- **RETRACTED after phase-gate review #2 (2026-05-31):** v38 "~18x / DECISIVE" and
  v39 "novelty lock CLOSED". Reasons (all accepted): the v38/v39 mechanisms were
  **edited between prereg and result** (tuned-to-pass); the committed v39 JSON says
  **novel_detected=FALSE** (headline led with a favourable seed); v38's "18x" is a
  **strawman** (planner handed the reward geometry, planning compute hidden, single
  seed, siloed PPO baseline that actually wins on quality); the work is a
  **re-derivation of MBRL (PETS) + multiple-model adaptive control (MMAC/MOSAIC)**,
  uncited; and it was a **DETOUR into a 1-D toy** that avoids the real problem.
- **The real wall is known:** v12-C already tried acting **from pixels via a learned
  world-model + planning** and it was an honest NEGATIVE, PARKED. The mechanism is
  UNTESTED-to-failing one rung up (pixels + horizon).

### What honestly survives from the model-based arc
The **recognise-then-reuse PATTERN** works in a small, low-dim, well-separated
regime (= classical multiple-model adaptive control). Modest and known. The
**learn-when-novel** trigger (novelty detection) is **unsolved** (absolute residual
threshold conflates "novel" with "uncontrollable"; needs a relative/conformal test).

---

## THE BET (still reasonable, must be tested where it's hard)
Learn **factored NOTIONS** (reusable abstractions / world-model factors) → **recognise**
which applies to a new context → **reuse** it → **learn anew** when none fits.
This is the right family (world-models + skill library). It is ONLY validated in
easy/low-dim regimes. **It must be proven from PIXELS or it doesn't count.**

## PLAN (ordered; do these, review between each)
1. **PIXEL NOTION → FASTER NEW TASK (the decisive test).** Pretrain a *factored*
   notion from pixels SELF-SUPERVISED (e.g. "where will the moving object be" /
   a predictive latent) on VARIED dynamics (no task reward). Then measure
   episodes-to-competence on a NEW pixel control task, WARM (reuse the notion) vs
   SCRATCH, ≥3 seeds. Per the v36 principle, the notion SHOULD help here (extracting
   dynamics from pixels is hard). Decisive either way. *Avoid v12-C's trap: start
   reactive (notion as a perception feature), not full model-based planning.*
2. **Recognition from pixels, under OVERLAP + noise.** Once (1) works, add a small
   library of pixel-notions and a *calibrated/relative* novelty detector; report a
   confusion matrix + novelty AUROC over a graded sweep (not a single pass/fail).
3. **Faithful "varied practice" test (owner's insight), hard input.** Diversity-vs-
   repetition at matched compute, hard/perceptual input, weak downstream probe,
   OOD compositional transfer target. (v37 dodged this with an easy probe.)
4. **Scale toward a diverse LEARNABLE game suite** (needs better exploration than
   vanilla PPO — Flappy/Catcher didn't train; use shaping / v7 curiosity).

## STANDING RULES (added 2026-05-31, per owner)
- **FREEZE the mechanism at preregistration.** If the smoke shows the registered
  design fails, that is a NEGATIVE to report — NOT a licence to edit the mechanism
  and re-run under the same prereg. Disclose any post-prereg change explicitly.
- **Review regularly:** spawn 3-4 adversarial agents (rigour / ML / strategy) at
  every milestone and after any "positive". Dissent > consent. Act on findings.
- **Honesty over hype:** the JSON of record must match the headline; report the
  worst seed, the hidden compute, the unfair baseline. The project's credibility is
  its willingness to kill its own claims.
- **Fair baselines:** match information and compute (e.g. reward-matched, multi-task,
  goal-conditioned) before claiming a win. Count ALL compute (planning included).
- **Seed everything** (torch + cuda + env); ≥3 seeds for any headline; log seeds +
  hyperparameters in the result JSON.
- **Come back to the owner ONLY with a convincing, positive, reviewed result.**


## UPDATE 2026-05-31 (after v40 null): the SHARPENED requirement for a positive
v36 (state, easy), v37 (probe, easy), v40 (pixels, but easy + noisy notion) all
NULL for the SAME reason. For a reuse-positive, BOTH must hold:
1. **Scratch must genuinely STRUGGLE** on the task in the budget — not learn it in
   ~40 iters. Achieve via: long horizon / sparse reward / a consequence that is
   HARD to extract from the raw input (e.g. a MULTI-BOUNCE landing, or a
   delayed/occluded outcome).
2. **The notion must be ACCURATE** on exactly that hard-to-extract quantity (low
   prediction error), or WARM is handicapped by a noisy feature.
Next concrete experiment (v41): a HARD pixel task (multi-bounce projectile or
sparse-reward long-horizon intercept) where scratch <0.3 at the budget, an
ACCURATE notion (bigger CNN / more data / predict the multi-bounce landing), and
WARM vs SCRATCH. If WARM>>SCRATCH reliably -> the positive; then review, then report.
