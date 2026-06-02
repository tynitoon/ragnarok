# Ragnarok — Original Architecture: the NOTION GRAPH (NG)

*Our own design, from first principles. Not a re-implementation of a known method.
Inspired by concepts (compression/MDL, predictive coding, composition) but the
architecture and the algorithms are ours to invent, test, and iterate.*

Started: 2026-06-01, after 48 versions mapped where known bricks fail.

---

## THE PRINCIPLE (derived from our own 48-version map)
Every reuse attempt failed for ONE structural reason: **reuse was BOLTED-ON** — a
stored notion / model / skill the agent could ignore. When a task was learnable, the
agent ignored the stored thing and learned directly. Reuse was never *forced*.

> **Original design choice: make reuse INTRINSIC.** The agent's ONLY representation is
> a single, growing, self-compressing, compositional library of NOTIONS. Perceiving,
> predicting and acting are ALWAYS done by composing notions — there is no raw-pixel
> shortcut. A new context is therefore solved by REUSE (compose existing notions) plus
> the MINIMUM of new notions. A compression drive forces notions to be general/reusable.

This is the opposite of everything we tried: not "store a notion and hope it helps,"
but "the agent literally cannot represent anything except as a composition of notions."

---

## WHAT A NOTION IS (grounded in pixels)
A notion is a small learned module with two parts:
- a **predictor**: given a local spatio-temporal context (a patch + its recent change +
  the action), predict that patch's next state. = a reusable "this behaves like this".
- a **detector / binding**: where/when does this notion apply (its responsibility map).

Notions compose: a frame is explained as a tiling/overlay of active notions; the next
frame is the composition of their predictions. Higher notions can be abstractions of
frequently co-occurring lower notions (hierarchy emerges from compression, not by hand).

---

## THE AGENT LOOP (developmental)
1. **Perceive** — decompose the frame; for each region pick the notion that best
   predicts it (composition by binding). This is the agent's *understanding* of now.
2. **Predict** — compose active notions -> predicted next frame.
3. **Act** — choose actions that (a) achieve a goal via the predicted consequences, or
   (b) reduce prediction uncertainty (curiosity). Control flows THROUGH the notions.
4. **Learn from surprise** — where prediction fails (high local error):
   - **re-bind**: maybe an existing notion explains it under a different binding -> REUSE;
   - else **mint** a new minimal notion fit to the surprising dynamics -> LEARN.
5. **Compress (consolidation / "sleep")** — merge near-duplicate notions, prune rarely-
   used ones, and abstract frequent compositions into one higher notion. Library size is
   penalized. This FORCES the library toward few, general, reusable primitives.

---

## WHY THIS GIVES RELIABLE REUSE (where all 48 versions failed)
- A new world is perceived through the SAME notions -> if covered, understood at once.
- Compression makes a notion that fires in many contexts cheap and a one-off expensive
  -> the library *evolves toward general primitives* on its own.
- Reuse is not optional: the agent can only act through its notion predictions, so prior
  notions are ALWAYS in the loop. There is no "ignore the notion and learn flat" escape.

---

## THE DECISIVE TEST (forced-reuse acceleration, from pixels)
On a STREAM of pixel worlds that share sub-dynamics (ball moves, object bounces, paddle
tracks...), measure:
- (a) **new notions minted per world** — must DECREASE as the library grows (reuse);
- (b) **prediction error** — stays low (competence is kept while reusing);
- (c) **cross-world**: a library built on worlds A,B,C understands a NEW world D with
  FEWER new notions and lower initial error than building from scratch.
If new-notions-per-world falls while competence holds -> reliable reuse, FORCED, from
pixels = the North Star mechanism, *by construction* rather than by hope.

FAIR BASELINE (rigor kept): a fixed-capacity predictor with the SAME parameter/compute
budget but NO compositional-notion structure (e.g. one monolithic CNN world-model).
If the notion-library reuses across worlds where the monolith does not, the STRUCTURE
(not the capacity) is what reuses.

---

## BUILD PLAN (incremental, each tested before the next)
- **v0.1** — the notion-library next-frame predictor on ONE pixel world: does it predict
  by composing a SMALL, growing-then-plateauing set of reused notions? (perception+pred core)
- **v0.2** — the forced-reuse test: stream of worlds with shared dynamics; new-notions-per-
  world must fall vs the monolith baseline. THE decisive representation result.
- **v0.3** — control: act through notion predictions to reach goals (curiosity + goals).
- **v0.4** — open-ended developmental stream; autonomous notion growth + consolidation.

## HONEST RISKS (named up front)
- Minting/pruning can be unstable (notion explosion or collapse) — needs careful drives.
- Grounding the detector+predictor in pixels is hard.
- Control-through-notions is a second hard step after the representation works.
We accept these — it is a hard *original* bet that targets the exact failure we mapped,
not a re-derivation of known work. Rigor (>=3 seeds, fair baselines, no leaks, review)
is kept — to keep our OWN ideas honest.

---
## v0.2 RESULT — honest verdict after 3 adversarial reviews (2026-06-01)
The 3-seed cross-world result (library warm-gain +31% vs monolith +17%) was put to adversarial
review. VERDICT: the "forced compositional reuse works" framing is RETRACTED. Findings:
- It is a WARM-START, NOT durable reuse: at the ASYMPTOTE, warm final error >= scratch on 2/3
  seeds (s1 0.00446 vs 0.00372; s2 0.00610 vs 0.00490). The +31% AUC is the first-window head-
  start (Pong notions already render Breakout's ball/paddle); the from-scratch library erases it
  in ~2k steps. AUC rewards head-starts -> wrong metric.
- The monolith baseline is UNDERTRAINED (a single per-patch MLP that never converges, 2-2.7x
  worse) = a strawman, the mirror of the retracted v38/v43 oracle. The design doc promised a CNN
  monolith; the implemented one is an MLP. So "structure reuses" is only shown vs a weak flat MLP.
- "Reuse 80-92%" is mostly APPEARANCE similarity (near-identical white-ball sprite), not
  compositional DYNAMICS reuse. The genuinely-new dynamics (brick collisions) = the 5-7 minted
  notions, small and unexamined.
- NOVELTY: the prediction core is close to KNOWN work — hard-assignment Mixture-of-Experts
  (Jacobs&Jordan'91), VQ codebook dynamics (van den Oord'17), Resource-Allocating Networks
  (Platt'91 = mint-on-residual/prune), and especially Recurrent Independent Mechanisms (Goyal'19)
  which ALREADY claim modular dynamics primitives transfer better than a monolith. ~10-15% is ours
  (the forced-reuse framing + the patch-tiled mint/prune-on-pixels integration), and the genuinely
  original part (agent acts ONLY through composed notions) is UNTESTED (no control in v0.1/v0.2).
=> HONEST MINIMAL CLAIM: a hard-binding mixture transfers shared-appearance primitives as a
WARM-START Pong->Breakout; the edge is an init head-start that does NOT persist to the asymptote.
True, small, known. The decisive test (next): CONVERGED/CNN baseline + NOVEL-dynamics target +
metric = new-notions-to-reach-error-epsilon and FINAL error (not AUC), >=3 seeds. And the original
bet (forced reuse via ACTING through notions) must actually be built+tested, or it is just RIMs.

---
## DECISION (owner, 2026-06-01): pursue OPTION 1 (forced-acting), fallback OPTION 2 (radical-originality)
Owner chose to TRY option 1 = build v0.3 (the agent ACTS only through composed notions: control +
intrinsic drives + open-ended growth), with the reviewers' rigor (metric = new-notions-to-competence
+ FINAL competence, NOT AUC; CONVERGED baseline; NOVEL dynamics; >=3 seeds; adversarial review).
*** IF v0.3 FAILS -> OPTION 2 = design something MORE radically original (the recurring failure,
flagged by owner + 2 reviews this session, is re-deriving known methods: AD, then RIMs). REMEMBER. ***
v0.3 enabler: notions need a context DETECTOR (predict which notion applies from context, with no
future target) so the agent can ACT by planning forward through notions. Added: per-notion context
centroids + NotionLibrary.predict(ctx) (context-bound forward prediction). Substrate: Catcher.
