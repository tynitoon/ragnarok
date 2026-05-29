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
- [SOLIDIFY v7] v7 capstone was N=1. Re-run autonomous discovery over N=5
  seeds to show it RELIABLY reaches 9/9 + DAG-valid order (not a fluke).
  Highest-value, lowest-risk (no env surgery); firms up the headline.

## NEXT (prioritized)
0. SCALE depth: extend the tree (diamond/diamond_pickaxe -> depth ~8) via a
   deep=True flag (keep old env default intact); show discovery still climbs
   it and the advantage GROWS with depth. (Moderate env surgery — do after
   solidifying v7.)

## DONE since last update
- M7 reliability: end-to-end iron_pickaxe 0.71 -> 0.77 (quantity-aware
  options); PLATEAUED ~0.77 (in-chain skill-compounding, not quantity).
  PARKED — diminishing returns (0.65->0.71->0.77 across M5/M6/M7); learning
  claims don't depend on 1.0. A working autonomous builder ~7x flat.

## NEXT (prioritized)
2. CROSS-WORLD transfer: skills learned in world A accelerate mastery in a
   related world B (shared primitives, different layout/recipes) — the
   ultimate reuse test (cross-task, not just cross-depth).
3. Harder navigation (bigger/sparser/obstacles) so learned low-level skills
   separate decisively from random nav end-to-end.

## BACKLOG / ideas
- Discover the RECIPES/physics themselves (not just the sub-goal curriculum) —
  deepest autonomy, hard/open-ended.
- Visual render of the agent crafting (tangible demo).
- World-model / planning in CraftWorld latent (reuse RSSM) instead of model-free.

## Conventions
- Preregister design in preregistration.md BEFORE each run (chronology).
- Commit+push often; record failures honestly; controls + N seeds where
  stochastic. Update this file as state changes.
