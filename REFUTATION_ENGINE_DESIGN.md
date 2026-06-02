# Ragnarok — OPTION 2: the REFUTATION ENGINE (radically different bet)

*Designed 2026-06-01 after option 1 (Notion Graph) showed control can emerge but stabilising a
growing hard-mixture is known MoE territory. This is deliberately NOT a gradient-trained
statistical predictor/policy/mixture — it is a developmental "little scientist".*

---

## WHY THIS, derived from our own 48-version + this-session map
Two hard-won facts:
1. **Every reuse attempt reused SURFACE STATISTICS** (features, weights, experts, dynamics-models)
   — which transfer poorly or only as a warm-start, because they are tied to the source task's
   surface, not to what is TRUE across tasks.
2. **Reuse only PAYS (v44 boundary) when the knowledge is EXPENSIVE to rederive AND the task is
   too hard to learn directly.** Cheap-to-rederive knowledge (e.g. "right moves paddle right") gives
   only a warm-start, because from-scratch learning is also cheap.

> **The radical bet: knowledge = REFUTABLE INVARIANTS (laws), discovered by actively SEEKING TO
> BREAK them, and kept only while they survive. Reuse is reliable BY CONSTRUCTION — an invariant
> that survived thousands of refutation attempts is TRUE across contexts, so it holds in a new one.
> And we target invariants that are EXPENSIVE to find (require search / long horizon / many
> samples), so reusing them genuinely pays.**

This is the opposite of statistical averaging: a law is not "usually right", it is **either
unrefuted or dead**. Survivors are context-invariant; that is exactly what reliable reuse needs.

---

## WHAT KNOWLEDGE IS
A **law** = a refutable relation among QUANTITIES, their changes, and ACTIONS, with parameters and
a confidence that is earned by survival, e.g.:
- controllability: `Δq_i ≈ θ · a`  (action a changes quantity q_i)
- kinematics / conservation: `q_i(t+1) ≈ q_i(t) + v_i`, `q_i + q_j ≈ const`
- predicates / preconditions: `event ⟺ φ(q)`  (e.g. catch ⟺ |paddle−fruit| < tol)
- COMPOSITE / EXPENSIVE laws: multi-step or conditional invariants that take long experience to
  verify (these are the ones whose reuse pays).
A law lives in an abstract quantity-space, so it is **detached from pixels' surface** — it can hold
in a totally different-looking world if that world has the same underlying mechanism.

## THE ENGINE (developmental, Popperian)
1. **Perceive** quantities q from the frame (v0: simple colour-blob centroids — a stepping stone;
   learned perception later).
2. **Hold** a population of candidate laws (from a small generative grammar of relations).
3. **Test & refute**: every step, score each law's residual on the new data. Confidence rises while
   it holds; a law that fails beyond tolerance is **refuted** (killed or split into conditional laws).
4. **REFUTATION-SEEKING CURIOSITY** (the original exploration engine): the agent acts to go where its
   current laws are MOST UNCERTAIN / most likely to break — it hunts its own theory's boundaries,
   not generic novelty. This finds invariants (and their limits) fast.
5. **Act** by applying surviving laws: predict each action's effect on the goal-relevant quantity,
   pick the action that achieves the goal. Control flows through trusted laws.
6. **Reuse**: in a new task, re-test existing laws; the ones that still hold are TRUSTED immediately
   (no relearning); only the genuinely-new part of the world needs new laws.

## WHY REUSE IS RELIABLE *AND* PAYS (the two things that killed everything before)
- **Reliable**: survivors are unrefuted-across-contexts -> they hold in the new context if the
  mechanism is shared (true, not similar). No warm-start-that-fades.
- **Pays**: we deliberately target EXPENSIVE-to-derive invariants (the v44 regime). Re-deriving a
  hard invariant from scratch costs much search; reusing a trusted one is free.

---

## DECISIVE TEST (must beat the v44 boundary, fairly)
A task family where the goal-relevant invariant is EXPENSIVE to find from scratch (long horizon /
hidden / needs many interventions) but SHARED across the family. Measure: a refutation-agent that
already holds the invariant solves a NEW family member in far fewer interactions than from-scratch
(which must re-derive it) — AND than a strong model-free baseline at MATCHED compute, >=3 seeds.
If a reused invariant makes a too-hard-to-learn-directly task solvable where from-scratch fails ->
that is the North Star, on the regime where reuse genuinely pays, with reliability by construction.

## MINIMAL BUILD PLAN (incremental, tested before scaling)
- **r0.1** — perception (colour-blob quantities) + a few hand-grammar laws (controllability,
  kinematics) tested+refuted online on Catcher; the agent ACTS via surviving laws and catches
  reliably (stable, unlike NG v0.3a's collapse — laws don't "collapse", they're refuted or kept).
- **r0.2** — REUSE: carry the laws to Pong/Breakout; the controllability/kinematics laws survive
  re-testing -> immediate control; only new laws (bricks/scoring) are added. vs from-scratch + a
  fair model-free baseline, on an EXPENSIVE-invariant task.
- **r0.3** — refutation-seeking curiosity replaces hand-exploration; the agent discovers law
  boundaries autonomously.
- **r0.4** — learned perception + law-grammar induction (remove the hand-started scaffolding).

## HONEST NOVELTY & RISKS (no overclaiming this time)
- Relatives exist: symbolic regression / equation discovery (Eureqa, AI-Feynman), causal discovery,
  optimal-experiment-design/active learning, Popperian/"artificial scientist" agents. The original
  integration is: an EMBODIED DEVELOPMENTAL CONTROL agent whose representation IS a refuted-or-kept
  invariant set, with REFUTATION-SEEKING curiosity, explicitly aimed at the expensive-invariant
  reuse regime. We will cite relatives and not claim more than the integration.
- Risks: perception/grammar are hand-started in r0.1 (a stepping stone, removed by r0.4); law
  induction can be brittle; the "expensive-invariant" task must be designed honestly (not rigged).
- Rigor kept: fair baselines, matched compute, >=3 seeds, adversarial review before any claim.

---
## r0.2a RESULT + honest caveat (2026-06-01)
The agent discovers g (0.00393 vs true 0.004) and intercepts at 1.00 (random 0.19, oracle 0.91),
STABLE. BUT the LAW FORM (discrete kinematics + reflective bounce) is HAND-CODED in predict_landing;
the agent only fits the parameter g. So r0.2a is model-based control with a GIVEN model + 1 fitted
param = the v36/v43 "given-the-hard-part" trap, and reusing g (one scalar) would be trivial.
=> The genuine Refutation test (r0.2b): the agent must DISCOVER THE LAW FORM itself, by falsifying
candidates from a small grammar (constant-velocity vs constant-acceleration vs damping) AND
discover the reflective-bounce conditional from where the main law breaks. Only then is the reused
invariant non-trivial. Honest deep pattern to watch: reliable reuse pays only for EXPENSIVE
invariants; discovering expensive invariants = program/theory induction (the field's open problem);
whenever the form is given, reuse is trivial. r0.2b tests how far genuine form-discovery gets.

---
## r0.2c RESULT — 2026-06-01: reused law >> model-free (robust), self-bounded honestly
Reused (falsification-verified) const-acc+bounce law reaches catch>=0.8 in 5,120 interactions on
EVERY gravity (0.003/0.0045/0.006) and EVERY seed (9/9), vs model-free PPO-from-scratch 287k-737k
interactions -> speedup 56-144x, every cell. RELIABLE by construction (form refuted-or-kept).
SELF-BOUNDED (before any review, per discipline): (a) it is model-based (rollout-to-landing largely
HAND-CODED) vs model-free = the KNOWN "having the right model crushes model-free on hard exploration"
= our own v36; (b) the reused invariant (the const-acc FORM) is CHEAP to discover (3-candidate
grammar); (c) velocity perception is STATE, not pixels. So the 56-144x is real but mostly the known
model-helps effect; the GENUINELY-novel contribution is the falsification-discovery + reliability-by-
construction framework, NOT the speedup magnitude. r0.3 = do it FROM PIXELS (remove the state hand-
engineering) -> the real North-Star test.

---
## r0.3 / r0.3b (FROM PIXELS) — 2026-06-01: control works, precise discovery gated by perception
From raw 48px frames (colour-blob perception) the agent perceives the ball, predicts landing, and
INTERCEPTS at ~0.85-0.89 (random 0.19, oracle 0.90) — control from pixels WORKS. BUT precise
law-discovery is gated by perception sub-problems:
- r0.3: per-step falsification picks the WRONG form ('damp' not 'const_acc') because pixel
  quantisation noise (~0.02) swamps the true per-step Delta(vy) (~0.004).
- r0.3b: windowed parabola-fit DETECTS acceleration (quad/lin residual 0.365) but g-extraction
  cancels to ~0 because reflective BOUNCES fold the trajectory (positive curvature at the fold
  cancels the negative curvature of smooth arcs) -> needs trajectory SEGMENTATION at bounces.
Control is robust to the imperfect law; precise discovery needs real perception/signal-processing.
=> HONEST META (whole session, all angles): the Refutation Engine works CLEANLY on clean quantities
(state: discover + reuse laws, 56-144x vs model-free) and DEGRADES on the perception sub-problem
(pixels). It genuinely advances the FRAMING (reliability-by-construction via falsification; stable,
no collapse) but the grail remains gated by the SAME open problems mapped all session: learned
PERCEPTION of the right quantities + INDUCTION of expensive structure. Defensible positive = the
STATE-mode developmental scientist (discovers physical laws by falsification, reuses them reliably
across a task family, vastly out-samples model-free RL). Pixels-precise-discovery is future work
(segmentation/denoising), not a grail blocker we can hand-wave.

---
## r0.2c RETRACTED + 3 adversarial reviews (2026-06-01)
3 reviews (methodology / novelty / strategy) DEMOLISHED the r0.2c "56-144x reuse" headline:
- r0.2c GravInterceptor does NOT reuse a discovered law: const-acc+bounce is HAND-CODED in
  landing(); only scalar theta is re-fit from scratch each world. "Reuses its discovered law"
  is FALSE at the code level = our own retracted v36/v43 (model-based-given vs model-free).
- "5,120 interactions" = chunk*num_envs (eval-cadence floor); theta=0 (NO law) passes 2/3 cells
  -> the invariant is barely load-bearing. The 56-144x is an artifact + info-unmatched baseline.
- NOVELTY: it is SINDy + recursive-least-squares + adaptive MPC + RANSAC inlier gate + ballistic
  LS, in Popperian language. "Reliability by falsification" = re-label of model-selection-by-residual.
  No guarantee/version-space/PAC bound. Refutation-seeking curiosity is named but NOT implemented.
- The SOLE potentially-novel claim (reliability by construction) is UNTESTED: no law-violating task
  is ever on trial; where the selector ran for real (pixels r0.3/r0.3b) it picked the WRONG law.
=> RETRACT r0.2c. THE decisive next test (unanimous): LAW-VIOLATING tasks. Carry a const-acc law
that held on gravity; on a violator (damping), does the agent REFUTE + adapt (vs a BLIND ablation
that mis-applies the stale law and fails)? Report refute-detection + post-refute recovery, >=3 seeds.
If clean refute-and-adapt-where-blind-fails -> first genuine contribution. Else -> falsified on
state, stop. (r0.4 below.)
