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
- [CROSS-WORLD via configurable tree] v9: refactor CraftWorld to a
  DATA-DRIVEN tree spec (items/recipes/resources as a config), default =
  current 9-node tree (regression-test with craft_sanity). Unlocks BOTH a
  deeper tree and a SECOND world (different recipes sharing primitives) ->
  the cross-world transfer test (skills from A accelerate B). The highest-
  value remaining frontier.

## DONE (recent)
- v7 SOLIDIFIED N=5: 5/5 seeds full tree + DAG-valid + iron_pickaxe.
- v8 ROBUSTNESS across world sizes: grid 13 (3/3) and grid 17 / very sparse
  (2/2) still reach 9/9 + iron_pickaxe. Discovery does not break with harder
  navigation; not an artifact of the tiny 9x9 world. (DAG-order metric is a
  slight over-counter: a deep skill can subsume a prerequisite collect.)
- M7 reliability: end-to-end iron_pickaxe 0.65->0.71->0.77; PLATEAUED ~0.77
  (in-chain skill-compounding). PARKED (diminishing returns).

## NEXT (prioritized)
1. SCALE depth: with the configurable tree, extend to depth ~8-10; show
   discovery still climbs it and the reuse advantage grows with depth.
2. Curiosity-GUIDED frontier exploration (vs random) — would extend the
   robustness limit to even sparser/bigger worlds.

## BACKLOG / ideas
- Discover the RECIPES/physics themselves (not just the sub-goal curriculum) —
  deepest autonomy, hard/open-ended.
- Visual render of the agent crafting (tangible demo).
- World-model / planning in CraftWorld latent (reuse RSSM) instead of model-free.

## Conventions
- Preregister design in preregistration.md BEFORE each run (chronology).
- Commit+push often; record failures honestly; controls + N seeds where
  stochastic. Update this file as state changes.
