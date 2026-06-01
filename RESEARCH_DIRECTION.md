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


## UPDATE 2026-05-31 (after v41 null #4): STRATEGIC PIVOT — stop raw-pixel reactive reuse
FOUR nulls (v36 state-easy, v37 probe-easy, v40 pixel-easy, v41 pixel-hard) share a
STRUCTURAL tension: a notion useful enough to help (hard-to-extract quantity) is hard
to LEARN ACCURATELY from raw pixels, and small pixel games are learnable-enough that
scratch doesn't need it. So raw-pixel REACTIVE notion-reuse does not pay.
The project's ONE clean reuse-positive (v17b ~5-15x) had BOTH conditions raw-pixels
lacks: (a) a STRUCTURED representation (Tetris board) where the notion (landing) is
learned ACCURATELY; (b) a task genuinely HARD from scratch (deep Tetris).
=> NEW DIRECTION for the positive (replaces "raw-pixel reactive" as plan item 1):
GENERALISE v17b. Take a FACTORED notion learnable accurately from a STRUCTURED (not
raw-pixel) representation, on a GENUINELY HARD task, and show it learned on task A
accelerates a DIFFERENT hard task B that uses the same notion — reliably (>=3 seeds),
fair baseline, reviewed. Candidate substrates: Tetris/CraftWorld (structured, hard),
or a structured-symbolic input. Pixels remain the eventual rung, but only once the
notion can be made accurate there (frame-stack/higher-res/learned-latent) AND the
task is hard enough — not before.

## UPDATE 2026-05-31 (deepest diagnosis + owner's RSSM hint): GO LEAN, AMORTISE SEARCH
Unifying pattern across ALL results: **reuse pays exactly when the notion AMORTISES
an expensive SEARCH/SIMULATION the task REQUIRES.**
- v17b WORKS: Tetris is hard because choosing a placement needs simulating the drop;
  the "landing" notion replaces that simulation -> planning becomes cheap -> 5-15x.
- v36/v40/v41 NULL: reactive catch requires NO costly search (tracking suffices), so
  the notion has nothing to amortise.
Corollary (matches the owner's hint that RSSM may be the wrong path): the WORKING
mechanism (v17b) is a LIGHT FACTORED PREDICTOR (one quantity), NOT a heavy
world-model/RSSM. Both times we went heavy (RSSM-from-pixels v12-C; model-based
planning v38/v39) it failed or was retracted. **Bet LEAN + FACTORED, not big-model.**

### v42 (next concrete experiment, frozen-when-built): notion-amortises-search, FAIR
A SEARCH-HEAVY task (candidate: Tetris-placement, or an abstract "pick the option
whose value needs a rollout"). TWO model-free agents, identical except WARM's obs
includes a pre-learned FACTORED notion (the rollout/landing outcome) as a feature;
SCRATCH gets the raw obs and must learn the search implicitly. Both model-free ->
FAIR (addresses the v38 strawman critique). Measure trials-to-competence, >=3 seeds.
HYPOTHESIS: WARM >> SCRATCH *because the task needs the search the notion amortises*.
If it nulls even here, the "amortise-search" theory is wrong too -> report and rethink.
Also worth exploring (per owner): NON-model-based reuse — reusable OPTIONS/skills
(hierarchical), or symbolic/program reuse — anything that makes a NEW hard task cheaper.


## UPDATE 2026-05-31 (after v42 null #5): "notion-as-feature" REFUTED; go MODEL-BASED, FAIRLY
5 fair nulls (v36/37/40/41/42) settle it: a learned notion as an RL FEATURE does NOT
speed a new task, because the agent learns the task from its raw observation anyway.
The notion only helps via PLANNING (avoid RL trial-and-error) = model-based RL.
=> THE positive to pursue (replaces "notion-as-feature"): MODEL-BASED SAMPLE-EFFICIENCY,
done FAIRLY. Learn a LEAN, FACTORED model (e.g. v17b's landing / a 1-step dynamics),
PLAN with it, and show it reaches competence in fewer ENVIRONMENT interactions than a
FAIR model-free baseline (multi-task / goal-conditioned, NOT siloed), COUNTING planning
compute, >=3 seeds, frozen, reviewed. This is the owner's "fewer trials with prior
knowledge"; it is real (v17b) but was overstated in v38 -- so the job is a RIGOROUS,
review-proof re-demonstration, not a new mechanism. Lean model (per owner's RSSM hint),
not heavy world-model. Substrate: Tetris (placement planning via learned landing) or
the inertial reach (planning under inertia). v43 = that, fairly.


## UPDATE 2026-05-31 (v43 reviewed -> retracted): the ORACLE is the strawman; v44 = LEARNED dynamics
v43 (model-based Tetris) looked positive (MB 56 vs MF 15 lines) but the adversarial
review (re-ran the code) killed it: MB was handed the ANALYTIC dynamics (the hard
part), the "learned value" is cosmetic (a hand-coded heuristic matches it), "3 seeds"
was a deterministic-eval N=1, planning compute was hidden, and MF was truncated at
~10% of its known regime. = the v38 strawman again. The review caught it BEFORE it
was reported -> discipline working.
v44 (the real test): MB scores placements from a LEARNED landing model (not the
oracle), so BOTH pay to learn the dynamics; randomised eval seeds + mean/std;
compute on a 2nd axis (count the lookahead); MF run to its v15 regime or matched
compute. Question: is learning a LEAN FACTORED model more env-sample-efficient than
end-to-end, AFTER counting planning cost? Honest either way. THIS is the bar.


## UPDATE 2026-05-31 (v44 — DECISIVE fair negative; closes the reuse arc)
v44 (MB LEARNS its model, no oracle, eval fixed): MB reaches 10 lines @ 153.6k
interactions; MF (end-to-end) @ ~92-123k and ends higher. => model-free is MORE
sample-efficient. At FAIR accounting, "prior knowledge -> fewer trials" does NOT
hold on learnable substrates: learning an accurate model costs interactions that
aren't repaid when MF can learn directly. Every prior "win" (v38/v43 oracle, v17b
given-model+hand-scoring) hid this. The capability is real ONLY when the knowledge
is expensive-to-rederive AND the task is too hard to learn directly (v17b's narrow
regime). The honest boundary of the whole reuse investigation.
ONLY remaining place a FAIR win could live: a task model-free genuinely CANNOT learn
(sparse / very-long-horizon) + a strong planner. That, or accept the bounded honest
conclusion. Decide with the owner.


## UPDATE 2026-06-01 (deep literature survey — the MISSED angle: in-context RL / Algorithm Distillation)
Ran a verified multi-agent literature survey (2022-2026; 28 primary sources; 25 claims
3-vote adversarially verified, 16 confirmed). Three conclusions:

1. **Our v44 boundary IS the field consensus, not our failure.** At fair accounting,
   cross-task forward transfer in deep RL is near-zero/negative (CORA / Powers 2022),
   frozen visual pretraining (PVR / R3M / VC-1) is NOT better than from-scratch OOD
   (arXiv 2411.10175), and every reported "win" (XTRA +23%, DUSDi, CKA-RL +8%) HIDES
   uncounted pretraining / reward-free data — existence-proofs, NOT fair refutations.
   Large reliable forward transfer is an OPEN problem. We re-derived the real frontier.
2. **Heavy world-models are NOT the reuse mechanism (owner's RSSM hint corroborated).**
   DreamerV3 trains each agent FROM SCRATCH (generality + a compute scaling law, not
   transfer); TD-MPC2 multi-task is CO-TRAINING, held-out only ~2x. Bet LEAN for REUSE.
3. **THE missed angle, with real (within-domain) evidence: in-context RL / Algorithm
   Distillation (AD).** AD distils ACROSS-EPISODE LEARNING HISTORIES (where competence
   visibly improves) into a causal transformer; on a NEW task from the same distribution
   it improves IN-CONTEXT, gradient-free. Why it escapes ALL our nulls: the reusable
   object is neither a notion-as-feature (v36-42 nulls) nor a model-to-plan (v43/44) — it
   is THE LEARNING ALGORITHM ITSELF, amortised over a task DISTRIBUTION. KEY REFRAME of
   our whole arc: we tested reuse on SINGLE source->target pairs (exactly where the field
   ALSO nulls); AD's wins live in reuse AMORTISED over a distribution. Gated by
   LEARNING-PROGRESS in the data (expert-only distillation fails — Gato-style). Within-
   domain ONLY (cross-domain deferred/unsolved) -> a tech-tree-family positive does NOT
   claim cross-game yet; it is the honest stepping stone.

### NEW PLAN (replaces "accept the bounded conclusion"; ordered, review between each)
- **v45 (PRIMARY): Algorithm Distillation on PROCEDURAL tech-trees.** Use v10's
  DeviceVecTechTree.gen_tree(seed) to make a DISTRIBUTION of random tech-trees. Log
  from-scratch PPO learning histories on TRAIN trees; distil into a causal transformer;
  test gradient-free IN-CONTEXT mastery on HELD-OUT trees (unseen seeds) vs from-scratch
  PPO. FAIR: count distillation as a ONE-TIME cost AMORTISED over the held-out set;
  per-tree compare in-context episodes vs from-scratch PPO episodes; control the
  "amortising-parallel-actors" artifact (single-stream fair accounting); >=3 seeds;
  FROZEN at prereg. Falsifiable null = no in-context speedup. The developmental thesis
  made concrete: the agent's own history of getting better becomes the reusable knowledge
  that makes the next tree faster.
- **v46 (characterise the boundary): model-based vs model-free crossover** in the regime
  reuse already paid (deep sparse tech-tree) vs how much expensive sub-structure is SHARED
  across tasks — maps WHERE the v44 boundary sits.
- **v47 (rigour harness): adopt the Continual-World forward-transfer protocol** as the
  standard metric; re-test a factored representation with pretraining COUNTED.
- **AVOID (disproven by us AND the field):** blind PVR / encoder transfer across different
  games; a static predicted-dynamics feature handed to model-free RL; oracle-dynamics planning.

### DEFINITION OF AN AD POSITIVE (the bar for v45)
On HELD-OUT procedural tech-trees, the distilled transformer reaches mastery in measurably
fewer ENVIRONMENT EPISODES in-context (gradient-free) than from-scratch PPO, reliably
(>=3 seeds), with distillation compute reported and amortised over the held-out set, and
the parallel-actor artifact controlled. Positive + reviewed = come back to the owner.
