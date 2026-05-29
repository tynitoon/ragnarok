# Ragnarok — Roadmap (living doc)

Owner directive: build the AI as described (childlike developmental learner);
conclusive results over a paper; full autonomy; keep this roadmap current.

## Objective (the vision)
An agent that (1) learns basic notions, (2) REUSES them to learn complex
notions faster ("de plus en plus vite" / compounding), (3) DISCOVERS what to
learn next, (4) learns anew when there is no link. Demonstrated conclusively,
on a non-toy substrate, with honest controls + recorded failures.

## DONE (validated, preregistered, on GitLab)
- v3 gated reuse, v4 compositional reuse, v5 learned O(1) relevance gate
  (2-D point-mass; mechanisms validated but substrate is toy).
- v6 substrate PIVOT -> DeviceVecCraftWorld (depth-6 crafting tech-tree DAG,
  egocentric partial obs). Discrete PPO learner.
  - M3 (decisive): reuse => FLAT per-node learning cost vs depth; no-reuse
    fails past depth 2 (deep mastery 1.00 vs 0.00).
  - M5: learned composition — manager discovers a DAG-valid order, reaches
    iron_pickaxe ~6x flat.
  - M6: option semantics; end-to-end completion ceilings ~0.71.
- v7 (capstone): AUTONOMOUS DISCOVERY — no goals given, frontier-novelty +
  reuse discovers own curriculum, masters full tree bottom-up (9/9, DAG-valid,
  reaches iron_pickaxe); curiosity-flat baseline 0.11 / depth<=4.

## TRIED / dead-ends (don't redo)
- v3.x frozen-representation transfer + per-step action-blending composition:
  modest/null; abandoned for hierarchy + model-based.
- Fixed-K macro-steps (M5) and run-until-+1 options (M6): end-to-end plateaus
  ~0.66-0.71 — ROOT CAUSE = manager under-collects resource QUANTITY (needs
  2 stone, ~4 wood; collects each ~once). Not skill reliability, not option
  length, not undertraining (180-iter run = 100-iter run).
- Curiosity-only flat PPO: stalls at depth <=4 (can't reach deep via novelty
  alone) — this is the v7 baseline.

## NOW
- (idle) v10 generality + play_world demo done; v11 (universal learned nav)
  PARKED as honest negative. Pick next research when resuming.

## DONE (recent)
- play_world: runnable random-world demo (learns recipes -> plans -> builds);
  testable locally on any seed.
- v11 universal learned nav: PARKED — env/reward correct (scripted nav 1.0)
  but PPO can't bootstrap nav goal-conditioned over ~13 target types
  (hard exploration). v10 generality (scripted-nav primitive) stands.

## DONE (recent)
- v10 GENERALITY: 10/10 RANDOM unseen tech-tree worlds — rule recovery
  precision/recall 1.00, planned 10/10, execution 1.00. The agent develops
  in worlds nobody hand-built (not memorising one tree). Closes the #1
  "toy/hand-built" gap. (Nav scripted for tree-agnosticism; learned nav in
  v6/v7/v9.)
- v9 MODEL-BASED: learned the recipe DAG from interaction (precision/recall
  1.00), BFS-plans to any target zero-shot, executes 0.72 (~7x flat).
  Retires the "recipes hand-given" caveat.

## DONE (recent)
- v7 SOLIDIFIED N=5: 5/5 seeds full tree + DAG-valid + iron_pickaxe.
- v8 ROBUSTNESS across world sizes: grid 13 (3/3) and grid 17 / very sparse
  (2/2) still reach 9/9 + iron_pickaxe. Discovery does not break with harder
  navigation; not an artifact of the tiny 9x9 world. (DAG-order metric is a
  slight over-counter: a deep skill can subsume a prerequisite collect.)
- M7 reliability: end-to-end iron_pickaxe 0.65->0.71->0.77; PLATEAUED ~0.77
  (in-chain skill-compounding). PARKED (diminishing returns).

## FUTURE RESEARCH (parked — kept for later exploration, prioritized)
1. Curiosity-GUIDED frontier exploration (vs random) — push the robustness
   limit to even sparser/bigger worlds (random frontier held to grid 17).
2. SCALE depth / CROSS-WORLD via a data-driven configurable tree spec
   (default = current 9-node tree, regression-test with craft_sanity):
   deeper trees + a second world sharing primitives (cross-task transfer).
   Incremental — M3 already shows the reuse advantage grows with depth.
3. RSSM latent world-model + imagination/CEM planning on CraftWorld (the
   pixels/latent version of v9's symbolic model — richer, harder).
4. Visual render of the agent crafting (tangible demo, not scientific).
5. Quantity-aware / retry control to push end-to-end execution 0.77 -> ~1.0.

## Conventions
- Preregister design in preregistration.md BEFORE each run (chronology).
- Commit+push often; record failures honestly; controls + N seeds where
  stochastic. Update this file as state changes.
