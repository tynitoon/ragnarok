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

## UPDATE 2026-06-01 (v45 + v48: two more matched-compute deflations; the honest boundary, consolidated)
- v45 (Algorithm Distillation / in-context RL — the survey #1 idea): RETRACTED as a reuse result.
  3 adversarial reviews + independent checks: held-out task COLLAPSED to a contextual bandit over
  cell-types (prereqs granted), memorisation leak (corr +0.69), the speedup an artifact of
  asymmetric measurement, "overfitting" within noise. STRUCTURAL: AD only amortises PPO-solvable
  tasks -> it RELOCATES the v44 boundary, does not break it.
- v48 (compositional reuse vs flat, MATCHED compute, 3 seeds): reuse 0.73+/-0.02 vs flat 0.54+/-0.02
  on iron_pickaxe. REAL but MODEST reliable edge. The M-series/v7 "flat 0.11" was UNMATCHED-COMPUTE;
  at equal compute flat reaches 0.54. NULL at the decisive bar (fails on flat side).

CONSOLIDATED HONEST BOUNDARY (what 48 versions establish): under FAIR matched-compute accounting,
every LARGE reuse advantage shrinks to modest-or-null. Surviving "wins" gave the hard part for free
(v17b: given landing + hand-scoring) or used unmatched compute (M-series). Reuse pays only in the
narrow regime where the knowledge is EXPENSIVE-to-rederive AND the task too hard to learn directly —
and that regime is hard to hit FAIRLY: grant the decomposition and you gave the answer; withhold it
and flat is competitive. Matches the field consensus (survey: cross-domain reuse unsolved).
GENUINELY WORKS (fair): learn from pixels (v15); recognise known games + sublinear accumulation
(v25/26); AUTONOMOUS curriculum discovery + skill-library construction (v7) — value = autonomy/open-
endedness, not beating flat on one target.
BEST REMAINING GRAIL BET (untested fairly): does a learned SKILL LIBRARY let the agent solve NOVEL
deep targets (held-out goals it was NOT built for) faster/at-all than flat-from-scratch, at FAIR
AMORTISED accounting? Only untested regime combining what AD cannot (reach depth) with the North-Star
core (reuse on the NEW), amortising the library ("childhood") over many novel goals.

## UPDATE 2026-06-01 (DEFINITIVE session conclusion — both original architectures explored, grail MAPPED not cracked)
Owner directed building our OWN AI after the survey showed known methods deflate (our v44 negative =
field consensus). Two genuinely-different original architectures, built + rigorously reviewed:
- OPTION 1 NOTION GRAPH (forced reuse via composing a self-growing notion library): v0.1 predicts
  from pixels (works); v0.2 cross-world reuse RETRACTED by 3 reviews (warm-start, strawman baseline,
  ~RIMs/MoE); v0.3 control emerged but the growing mixture COLLAPSES (dead-expert = known MoE problem).
- OPTION 2 REFUTATION ENGINE (knowledge = refutable invariants, discovered by falsification): r0.1
  stable law-control; r0.2b discovers the gravity FORM by falsification (state); r0.2c "reuse 56-144x"
  RETRACTED (hardcoded rollout not reuse; artifact; = SINDy+RLS+MPC); r0.3 pixel-control works but
  precise discovery gated by perception noise; r0.4 falsification perfect (15/15) but BEHAVIORALLY
  INERT (reliability gives zero control advantage — closed-loop robust to the wrong law).
DEFINITIVE HONEST CONCLUSION: neither original architecture cracks the grail. The grail (general
reliable reuse from pixels) is gated by TWO genuine OPEN PROBLEMS — (1) learned PERCEPTION of the
right quantities, (2) INDUCTION of expensive structure/laws — confirmed from ~10 independent angles,
each time killing our own overclaim via adversarial review. The real contribution = a PRECISE MAP of
why reliable developmental reuse is hard + two coherent original architectures that reveal where the
wall is + genuine bounded positives (pixels v15; recognise+accumulate v25/26; autonomous
curriculum+composition v7). Further progress requires attacking the two open problems directly
(frontier research), not more mechanism variations.

## PHASE GATE (2026-06-02) — from-scratch perception arc CLOSED; 3 adversarial reviews converge

The from-scratch pivot started with the perception core (owner: "perception first, build nothing on
top until solid"). Six iterations (percept v0.1-v0.6): slot-attention, sprite-AE, +motion, recurrent
tracker, channel-indexed keypoints, K=8 reliability. Net: NEGATIVE for the grail. Honest reasons:
(1) it never reliably/generally bound the ball (1/5 Pong seeds at the frozen bar; 0/3 Breakout ~random;
low error only via oracle channel selection); (2) MORE IMPORTANTLY, perception is a NON-BOTTLENECK for
these blob-separable games — a 5-line white-centroid (already deployed in REE r0.3, pixel control
~0.87-of-oracle) dominates anything learned. We solved a self-imposed problem.

THREE adversarial reviews (perception / strategy / missed-angles), unanimous:
- STOP perfecting perception; use the existing reliable object extractor; pivot effort to REUSE.
- The object-centric -> relational-world-model -> cross-game reuse BET will most likely REPRODUCE our
  prior nulls (#2 handed-features redundant, #4 closed-loop robust to wrong models, #6 shared-dynamics
  deflated) because those failure modes are REPRESENTATION-INVARIANT. ~15-20% it gives a fair positive.
- The cross-game-from-pixels grail is, at this lab's scope, most plausibly a WALL. The project's real,
  defensible contribution is the PRECISE MAP of why + the bounded honest positives (pixels v15,
  recognize+accumulate v25/26, autonomous curriculum v7).
- HIGHEST-EV remaining shots (both deliver WITHIN-FAMILY reuse, not the cross-game grail):
  1) v49 DEPTH-SCALING crossover (preregistered, NEVER RUN; substrate ready today via v10/tech_tree).
     Find depth D* where a FAIR strong-flat collapses (<=0.2) while library-composition holds (>=0.8),
     with the library AMORTISED over MANY held-out targets (v48 amortised over ONE — its key flaw).
     HARDEN (or it retracts like v43/v45): (a) compose arm must NOT be handed the DAG/decomposition;
     (b) STRONG flat baseline (tuned exploration), so a collapse isn't a PPO artifact; (c) leak-fixed
     generator (shuffle cell<->item) + memorization-correlation gate checked BEFORE reading results.
     ~30-40% convincing positive; ~55-60% clean NULL that completes the boundary map (still valuable).
  2) "CHILDHOOD AMORTISATION": distribution-level skill-library reuse over a STREAM of novel deep goals,
     scored by the integral of (flat - warm) cost over the held-out distribution (the one regime where
     the field's FAIR wins live). ~30-35%. Successor if v49 nulls.

DECISION REQUIRED FROM OWNER: this reverses the most-recent explicit steer (from-scratch model, pixels,
perception-first). Options: (A) pivot to v49 hardened; (B) push an original PIXEL reuse idea anyway
(honours the pivot, ~15-20%); (C) accept the "likely a wall" map and consolidate the bounded positives.

## v49 STATUS (2026-06-03) — mechanism BUILT + reusable primitive DE-RISKED, but COMPUTE-BOUND
The hard part of the depth-scaling experiment is solved: the reusable nav-collect skill WORKS reliably
(0.96-0.99 success, generalises across procedural trees at depths 3 AND 7) once the target type is
BROADCAST as spatial channels into the CNN (the fix that took it 0.25->0.99). Manager + strong-flat +
fair matched-step accounting are implemented. BUT the full sweep is COMPUTE-INTRACTABLE to iterate on
this single GPU: the manager's nested option-rollout (macro-steps x option inner-steps x iters, each a
sequential CNN forward + Python env.step) runs at ~15% GPU util, >20 min per config. Could not extract
a single (compose, flat) data point in-session. The real multi-seed run (depths 3,5,7,9 x seeds 0,1,2)
is now cooking in the BACKGROUND (~4-6h); results -> craft_v6_out/v49_depth_scaling.json. A proper run
needs either a vectorised rollout (no Python per-step loop) or many GPU-hours. Honest odds (per reviews):
~35% a crossover appears (bounded within-family positive), ~55% null (fair flat keeps pace) — neither
is the cross-game-from-pixels grail, which the 3 reviews judge most plausibly a WALL at this scope.

## v49 RESULT (depth 3, seed 0) — directional signal, but NOT a clean/convincing positive
First (and only, in-session) data point: COMPOSE 0.64 vs FLAT 0.00 at depth 3 (7.1M matched primitive
steps; nav-skill 0.91). Timing: skill 100s / mgr 387s / flat 553s = ~18 min/config (manager option
rollouts consume ~5.9M steps -> the matched flat trains long). DIRECTIONALLY reuse-pays (compose >>
flat), BUT fails the bar honestly:
- Compose 0.64 < the frozen 0.8 (manager not reliable even shallow).
- FLAT 0.00 at depth 3 is NOT a credible "strong fair flat" -> flat can't discover deep crafting AT
  ALL, so there is no honest shallow-succeeds/deep-fails DEPTH crossover (flat fails at the shallowest
  craft depth, independent of D). This is exactly the weak-baseline trap that deflated v48->v44/v48; a
  0.00 flat inflates the gap and a reviewer rejects it.
=> v49 does NOT yield a clean preregistered crossover without slow flat-hardening (count-based
exploration / target-directed shaping), each iteration ~18 min, a full fair sweep ~4-6h, and per all
prior evidence most likely landing at "bounded within-family at best." Combined with the perception
arc + the 3 reviews: the cross-game-from-pixels GRAIL is, at this lab's scope, most plausibly a WALL.
The project's defensible value remains the precise BOUNDARY MAP + the bounded honest positives.

## v50 RESULT (childhood amortisation, depth 7, 8 train / 5 held-out) — NEGATIVE; manager is the blocker
The CHILDHOOD SKILL transfers cleanly (nav 0.94 on held-out trees — a real positive on the skill).
BUT the per-tree composition MANAGER is UNRELIABLE: warm masters held-out trees only ~1/4
(tree0 d5: 0.76 YES; tree1 d6, tree2 d5, tree3 d4: 0.00 NO), and SCRATCH fails identically — so a
PPO-over-options manager trained on a SINGLE procedural tree (~100 iters, sparse reward) gets STUCK
regardless of target depth (fails even d4). Amortisation is moot when the agent can't reliably master
the new task. The skill-only "childhood" saving (the ~1.6M skill cost) is real but small vs the
manager cost, and break-even (~9 trees) is never reached because mastery fails. **v50 = NEGATIVE: the
COMPOSITION (manager), not perception/skill, is the reliability blocker.**

-> v51: a TRANSFERABLE ROUTER (meta-manager) trained across the childhood DISTRIBUTION of trees, on
TREE-AGNOSTIC observable per-item features (in_inv/unlocked/craftable_now/collectable_now/is_goal/
is_resource), permutation-invariant. Bet: distribution-training escapes the single-tree local optima
(robust) AND the strategy TRANSFERS zero-shot to held-out trees -> the agent's WHOLE policy
(skill+composition) reuses -> dramatic amortisation. (Also = the owner's brain-inspired router idea.)

## v51 / GREEDY (night 2026-06-03) — composition-RELIABILITY barrier precisely located
After v50 (skill transfers 0.94 but per-tree manager unreliable), tested whether composition is
learnable/easy:
- LEARNED ROUTER (tree-agnostic per-item features, permutation-invariant, trained on the distribution):
  masters only ~1% even on TRAIN trees at 150 iters — severely under-trained / hard credit assignment.
- GREEDY forward-chaining over OBSERVABLE affordances (collect-what's-collectable / craft-what's-
  craftable / pursue-goal; a fixed tree-agnostic planner, no DAG granted) + the reused nav skill:
    * depth 3 (shallow): masters HELD-OUT trees ~0.69 (range 0.31-0.98) — REUSE WORKS for new shallow tasks.
    * depth 5: ~0.13.  depth 7: ~0.01 — FAILS, even with ample option budget (option_timeout 35).
- Diagnosis: the barrier is COMPOSITIONAL EXECUTION RELIABILITY. Per-step skill success (~0.94, lower
  on some resource-types) COMPOUNDS over the deep chain (0.94^~10 plus extra failures -> ~0). It is NOT
  the composition LOGIC (greedy is correct) nor the option budget; it is that reliably executing a LONG
  chain needs near-perfect per-step skills, which a GENERAL (transferred) skill doesn't have.

### Honest synthesis (the precise reuse boundary, refined)
- PERCEPTION/SKILL reuse: WORKS — one childhood skill transfers zero-shot to held-out procedural trees (0.94).
- COMPOSITIONAL reuse into NEW tasks: works for SHALLOW targets (transferred skill + general forward-
  chaining masters held-out depth-3 ~0.69), BREAKS for DEEP targets (compounding per-step unreliability).
- This refines the project-wide "reuse is real but bounded": a reusable skill + a general (even fixed)
  planner solves NEW shallow tasks cheaply; deep tasks are gated by per-step execution reliability that
  general skills lack. The grail (reliably master arbitrary NEW tasks from reuse) is blocked here by
  compositional execution reliability, not by perception or by composition LOGIC.

## v51 — ATTACKED the composition-reliability barrier (stochastic skill + ample budget) — CONFIRMED genuine
Attacked the deep-composition barrier directly:
- stochastic skill in options (escape deterministic loops) + option_timeout 60: depth 7 held-out still ~0.01.
- ample macro_budget 50 + fixed instrumentation (#items unlocked): SHARP cliff confirmed — depth<=5 masters
  (train d4-d5: 0.95-1.0), depth>=6 collapses. One train d6 reached 0.73 (so deep CAN work with a good
  skill+budget), but ALL held-out deep (d5-8) fail. Greedy makes PROGRESS (unlocks ~5-12 of ~14 items)
  then gets STUCK partway. Failures track per-tree SKILL WEAKNESS on specific resource types (held-out
  tree with nav 0.496 -> 1.36 items, 0.0 master).
- CONCLUSION (firm): the barrier is PER-RESOURCE-TYPE SKILL RELIABILITY on held-out trees. A general
  (transferred) skill is ~0.87-0.94 on AVERAGE but weak on SOME types on SOME trees; deep targets that
  need a weak-type collect fail, and the failure stops the whole chain. Neither stochastic execution nor
  ample budget cracks it. To crack deep compositional reuse one needs a skill UNIFORMLY reliable across
  ALL resource types (a perception-robustness problem), not just high-on-average.

### NIGHT SYNTHESIS (option 2 fully explored)
- CLEAN POSITIVE: one childhood skill transfers zero-shot to held-out procedural trees (nav 0.94).
- BOUNDED POSITIVE: reused skill + general forward-chaining masters SHALLOW (depth<=4-5) held-out tasks (~0.7).
- HARD WALL: DEEP held-out tasks fail on per-type skill reliability (attacked directly; doesn't crack).
- This precisely locates the grail blocker at COMPOSITIONAL EXECUTION RELIABILITY (uniform per-step
  skill reliability), distinct from perception (works), composition logic (works), and the reuse
  economics (real-but-modest). Consistent with the 3 reviews: grail is a genuine research wall here.

## *** v51 — DEEP COMPOSITIONAL REUSE WORKS (deadlock fix) — needs verification/review ***
The whole deep-composition failure was a GREEDY BUG, not a fundamental barrier: the planner used
"craftable & ~UNLOCKED" -> once an intermediate craft was made then CONSUMED as an input to two items,
it was never re-crafted -> second consumer/goal stalled ("makes ~all items but not the target"). Fix:
"craftable & ~IN_INV" (re-craft consumed intermediates).
RESULT (depth 7, 8 train / 6 held-out, skill 0.95, stochastic options, macro 45): greedy held-out
master jumped 0.16 -> **0.98** — EVERY held-out tree (depths 4-8) mastered (0.87-1.0), unlocking ~all
items incl. the deep target. So: ONE childhood skill (transfers zero-shot 0.95) + GENERAL
forward-chaining over OBSERVABLE affordances (no DAG granted) MASTERS arbitrary NEW DEEP procedural
tasks with ZERO adulthood learning = the developmental-reuse thesis, demonstrated.
HONEST CAVEATS / TODO before claiming: (1) the composition planner is FIXED (general, observable-only,
not tree-specific) — the LEARNED+reused part is the skill; a learned router should now also work since
the strategy is known-correct. (2) VERIFY robustness on fresh seeds. (3) ADVERSARIAL REVIEW (env too
easy? forward-chaining trivial on any DAG? fair vs from-scratch?). (4) Re-test the v50 AMORTISATION
(warm reused-skill+planner MASTERS held-out cheaply now -> clean childhood amortisation). DO NOT
overclaim before (2)-(4).

## *** RETRACTION of the v51 "0.98 breakthrough" — 2 adversarial reviews DEMOLISHED it ***
I over-claimed "deep compositional reuse works (0.98)". Two adversarial reviews (correctly) showed it
is largely an ARTIFACT, not a grail positive:
1. Forward-chaining is LOGICALLY GUARANTEED on this substrate (reviewer proved 1000/1000 with unlimited
   budget): gen_tree builds ACYCLIC DAGs with NON-SCARCE resources (cells re-collectable), so
   topological forward-chaining cannot fail. The 0.98 = "we implemented forward-chaining correctly",
   not reuse. The 0.16->0.98 "fix" was de-deadlocking the planner, not a reuse discovery.
2. The composition is HAND-CODED. The LEARNED router (the interesting version) FAILED: 0.003 zero-shot
   (1% train) in the SAME file. 0.98 stands in for a failed learned composer.
3. NO reuse/efficiency baseline exists (no from-scratch, no relearn-skill; v50 amortisation never
   completed; v49 flat=0.00 is degenerate). Mastery-rate is a CAPABILITY metric, not a REUSE metric.
   With a fixed planner + transferred skill, per-tree adulthood cost ~0 in BOTH warm & scratch -> the
   only possible saving is the one-time skill cost = the smallest/least-interesting reuse claim.
4. Substrate too EASY to be grail-relevant: full-obs, no scarcity/irreversibility/search; greedy over
   observable affordances (which ARE the DAG re-expressed) is trivially optimal. Real difficulty is
   per-step skill reliability, NOT planning.

### Honest status (corrected)
- CLEAN-ish POSITIVE: a nav-collect skill transfers zero-shot across procedural trees (~0.95). REAL but
  MODEST and ALREADY KNOWN (v49/v50); "transfer across trees" is thin (nav task identical per tree).
- The 0.98 "masters new deep tasks" is NEAR-TRIVIAL (guaranteed forward-chaining + dense-grid incidental
  collection + standing in for a FAILED learned composer). NOT the grail.
- LEARNED composition (the real developmental question) FAILED again (0.003) — the open problem.
- The grail-relevant test (per reviews): a HARDER substrate (scarcity/irreversibility/partial-obs, no
  affordance oracle, 20+ chains) where greedy is NOT trivially optimal, and ask whether a LEARNED
  composer transfers + beats from-scratch on COST-TO-MASTER. That is the honest next experiment; the
  learned composer already fails on the EASY substrate, so expectations are low.
- 5th discipline-caught overclaim this session. The pattern holds: reliable LEARNED reuse of
  composition is the wall; perception/skill transfer is the one real (modest) positive.

## v52 — FAIR amortisation (reviews' #1 ask) — the HONEST option-2 positive (modest)
WARM (childhood skill once + greedy) vs SCRATCH (fresh per-tree skill + SAME greedy), depth 7, 4 held-out:
- Both arms MASTER held-out trees: WARM 0.99, SCRATCH 0.99 (so it's a fair COST comparison, not capability).
- C_lib = 3.28M (childhood, 8 trees, once) vs C_skill = 2.46M PER fresh tree -> warm cumulative crosses
  below scratch after ~1.3 tasks. So childhood skill-training AMORTISES over a stream of new tasks.
- HONEST framing: REAL but MODEST. Childhood saves the one-time skill-relearning cost; it does NOT
  enable anything scratch can't (scratch masters too); composition is a FIXED trivially-correct planner
  (not learned). This is "reuse of perception/skill amortises", the smallest honest reuse claim — NOT
  "the agent learns to reuse composition" (that failed: learned router 0.003).

## OPTION 2 — CONCLUDED (honest)
Net result of the night's option-2 exploration:
+ Childhood nav-collect skill transfers zero-shot to held-out procedural trees (~0.95).
+ That skill amortises (train once, reuse over a stream; break-even ~1.3 tasks) — fair, both arms master.
- Deep "mastery 0.98" was an ARTIFACT (trivial forward-chaining on a no-scarcity DAG) — retracted.
- LEARNED composition reuse FAILED (0.003) — the real grail blocker, unchanged.
=> Modest, fair, real reuse positive (skill amortisation). The grail (reliable LEARNED reuse of
composition on tasks where it's actually hard) remains a wall. Next honest grail test = a HARDER
substrate (scarcity/irreversibility/partial-obs, no affordance oracle) where greedy is NOT optimal and
a LEARNED composer is REQUIRED — owner's strategic call (big build, low odds per the learned-router 0.003).

## v53 — REVIEW STATISTIQUE (confound de difficulté) — verdict: PARTIELLEMENT confondu, MAIS l'effet tient
Reviewer reproduced the stream trees exactly and computed true difficulty (prerequisite-CLOSURE size,
not depth). Findings (honest, integrated):
- The headline early-vs-late contrast (2.73M->0) is INFLATED: task 1 (27 ops) is STRUCTURALLY
  INFEASIBLE within the ~19-production effective macro budget (its 5.73M = guaranteed budget
  exhaustion, 70% of the early mean); late bucket happens to contain small closures. Cost tracks
  closure size within the feasible envelope (t2, 13 ops, was already FREE at position 2).
- BUT the effect SURVIVES difficulty control: feasible-early vs late still falls (2.46M -> 0+0+0), and
  the CLEANEST evidence is the SAME-TASK A-vs-B deltas (difficulty cancels by construction):
    t9 (d6, 16 ops, hardest mastered): B fresh zs 0.000 + 1.64M to master | A experienced zs 0.734, FREE
    t6: A 2.46M vs B 4.92M | t3: A 0.82M vs B 1.64M | (t0 calibrates seed noise: A 2.46 vs B 1.64)
- ENVELOPE LIMIT (the important negative): the flywheel generalizes WITHIN the feasible envelope but
  never EXPANDS it — nothing above the ~19-production cliff was ever solved (t1 27 ops, t5/ho0).
  "Late tasks are free" = TRUE. "Compounds into progressively DEEPER competence" = NOT shown.
- ho3 warning: easiest held-out tree (7 ops) scores 0.121 goal-conditioned vs 0.605 goal-ABLATED ->
  the goal head can actively hurt; what transfers is largely a goal-agnostic unlock-everything habit.
- Recommendations adopted: lead with same-task A-vs-B; flag t1/t5/ho0 as budget-infeasible by
  construction; seeds 1-2 stay FROZEN (replication of the prereg protocol, no mid-replication change);
  a post-hoc EXPLORATORY envelope diagnostic (macro_budget raised) only AFTER seeds, clearly labeled.

## v53 — VERDICT FINAL (3 seeds, frozen prereg): NULL — compounding does NOT robustly replicate
seed0 POSITIVE (8/10, compounding T, sep T) | seed1 PARTIAL (7/10, comp F, sep T) | seed2 NULL (7/10,
comp F, sep F: A 1.64M > B 1.23M on late tasks — experienced agent cost MORE than amnesic). => 1/3 on
the frozen criterion; even the confound-free same-task A-vs-B separation is 2/3 with one REVERSAL.
The seed-0 "flywheel compounds" headline was LUCK (which random trees landed early vs late). RETRACTED.

### What IS real (honest, salvageable)
- Self-imitation (hindsight CE) makes the composer LEARNABLE where sparse-reward RL gave 0.003: it
  masters 7-8/10 stream tasks and transfers zero-shot to IN-ENVELOPE held-out trees (~0.9). Real,
  clean finding about the LEARNING SIGNAL (CE-on-own-successes >> RL for this composition) — the thing
  predicted. But this is "self-imitation learns + transfers to similar tasks", NOT a developmental
  COMPOUNDING loop.
### What is NULL
- The thesis "accumulated experience makes each NEW task progressively cheaper, reliably" does NOT
  replicate. Seed-to-seed difficulty variance (closure-size, the ~19-production feasibility cliff)
  SWAMPS the compounding signal at this scale/stream length. The effect is real-but-fragile, not robust.

### The honest scientific conclusion (the actual result of the night)
At THIS scale (10-task stream, single GPU), the compounding signal exists but is DOMINATED by task-
difficulty noise -> not robustly measurable. Averaging it out needs a much longer/broader task stream
(more compute) -- i.e. the magnitude-needs-scale conclusion, now QUANTIFIED: the per-task difficulty
variance is larger than the per-task compounding gain on a 10-task procedural stream. Consistent with
the whole project: mechanism plausible, magnitude gated by breadth/compute we do not have. v53 closes
the option-2 arc honestly. No clean grail positive; the defensible deliverables remain the bounded
positives + this precise map of why compounding is fragile at small scale.

---

## v54 — CLEAN MEASUREMENT: verdict PARTIAL, and the mechanism is NOT compounding (2026-06-13/14)

Prereg frozen before code (4cd880f); interpretation locked before any seed finished, after a 4-agent
adversarial design audit (5b21e2d). All 3 seeds ran UNINTERRUPTED (no resume -> counterfactually clean).
Design: uniform-feasible tasks only (target production-count band [10,16], macro_budget 40 so nothing is
budget-infeasible), stream N=16, amnesic control B on EVERY task, mechanism identical to v53.

### Result per the FROZEN rule: PARTIAL (primary yes 3/3, secondary 1/3)
- PRIMARY (confound-free, same-task Delta_k = cost_B - cost_A): POSITIVE on 3/3 seeds, NO reversal.
  mean Delta = +1.95M / +1.95M / +2.15M primitives; B>A on 13/16, 14/16, 15/16 tasks.
  This is a genuine improvement over v53 (1/3 with a reversal) -- the confound removal WORKED.
- SECONDARY (cost_A declines over the stream): 1/3 seeds only -> frozen rule gives PARTIAL, not POSITIVE.

### Why the secondary failed -- and why that is the REAL finding
The data is a STEP, not a curve. On all 3 seeds: task 0 costs A full price (1.64M, zero-shot 0.00);
from task 1 onward A zero-shots almost everything (cost 0) while B, cold each time, pays 1.64-3.28M.
The secondary fails precisely BECAUSE there is no gradual decline: A is already at ~0 by task 1.
=> The mechanism is ONE-SHOT GENERALISATION across a narrow family of procedural trees, NOT the
   lifelong "flywheel" accumulation the v53/v54 framing assumed. The 16-task accumulation appears to
   contribute nothing beyond task 0. The "flywheel / lifelong buffer" narrative is NOT supported.

### Goal-ablation diagnostic (post-hoc, locked in advance): GOAL-AGNOSTIC 3/3
Final Arm-A composer, held-out + stream, conditioned vs goal-ABLATED:
  seed0 held 0.683 vs 0.727 (abl) | seed1 0.861 vs 0.861 | seed2 0.996 vs 0.997; stream ~identical.
The composer IGNORES the goal. Interpretation lock #1 CONFIRMED: the positive Delta is a warm-started,
goal-BLIND "unlock-everything" reflex converging in fewer rounds -- NOT goal-directed experience transfer.

### What is honestly TRUE
- A ENABLES, not merely accelerates: on several tasks B (cold) FAILS to master within the 3M budget while
  A zero-shots them (s0 t12,t15; s1 t9,t13,t14,t15; s2 t15).
- Not trivial: hard trees exist where BOTH arms fail (t3, t9 at pc15), so the band is not "all easy".
- Defensible claim, and the ONLY one: "a composer trained by hindsight self-imitation on ONE procedural
  tech-tree task generalises zero-shot to a family of same-band trees, mastering trees a from-scratch
  composer cannot crack in 3M steps -- via a goal-blind unlock-everything policy."
- NOT claimable: developmental compounding, accumulated experience making new tasks progressively
  cheaper, goal-directed transfer. (Would be the 7th overclaim of a family retracted 6 times.)

### Open questions a v55 would settle (NOT run)
1. Is task 0 sufficient? Arm A with the buffer FROZEN after task 0 -- if Delta survives, accumulation is
   provably irrelevant and the honest headline is one-shot generalisation, full stop.
2. Foreign-buffer control: warm-start on equal-volume demos from a DIFFERENT tree family. Delta surviving
   => the gain is generic data, not task-relevant experience.
3. Envelope width: how far out of the [10,16] band does the zero-shot reflex hold?

### Status of the option-2 arc
v53 closed it as NULL; v54 re-opens a NARROWER, cleaner positive: not compounding, but robust one-shot
in-family generalisation of a goal-blind composition reflex. Real, modest, and correctly framed.
