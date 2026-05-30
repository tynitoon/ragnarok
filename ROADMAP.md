# Ragnarok — Roadmap (living doc)

Owner directive: build the AI as described (childlike developmental learner);
conclusive results over a paper; full autonomy; keep this roadmap current.

## Objective (the vision)
An agent that (1) learns basic notions, (2) REUSES them to learn complex
notions faster ("de plus en plus vite" / compounding), (3) DISCOVERS what to
learn next, (4) learns anew when there is no link. Demonstrated conclusively,
on a non-toy substrate, with honest controls + recorded failures.

## NORTH STAR refined (2026-05-30, owner): GENERAL GAME MASTERY
"L'IA finale parfaite: la mettre sur N'IMPORTE QUEL jeu et qu'elle arrive à le
MAÎTRISER pour GAGNER." The developmental learning (reuse/discovery) is the
MEANS; the END is: drop the same agent on an arbitrary game, from pixels, and
it learns to WIN. This spotlights the exact piece v12-C struggled with —
acting from pixels to maximize a game's score. New program v15: prove the agent
WINS real, recognizable games from pixels (GPU-batched), then GENERALITY (same
agent masters multiple distinct games), then layer the developmental reuse
(cross-game skill transfer). Honest framing: PPO-from-pixels winning Pong is
established RL; the project-aligned novelty is generality across a game suite +
combining it with the validated reuse/discovery machinery.

### Game curriculum (owner's plan, 2026-05-30) — teach win/lose, then climb
Each rung teaches a REUSABLE capability; the agent should learn the CONCEPT of
winning/losing (from the on-screen outcome) and carry it across games.
- P0 — reactive control + "more points = good": **Pong**. DONE: win-rate 0.97
  from pixels (random 0.00). [v15 M1]
- P1 — clear a goal + lives: **Breakout**. DONE: same agent, return +25.6 vs
  random -23.5, clears the wall ~31x/eval, ~1 life lost. Generality across 2
  distinct games from pixels. [v15 P1]
- P2 — endless score-max + survival (no fixed end): **Snake**. NOW: same agent;
  env validated (greedy +48 food vs random +2.6). The "maximize points, don't
  die" concept.
- P3 — GENERALIZE win/lose (the core scientific milestone): a SHARED outcome-
  recognizer trained across P0-P2, then dropped on a NEW game where the agent
  seeks "win"/avoids "lose" with little/no explicit reward. = "it understood
  winning/losing." This is the reuse/compounding thesis on the most basic
  notion.
- P4 — placement + delayed reward (bridge to Tetris): **Tetris-lite**, placement
  as a MACRO-action (which column + rotation), reusing options (M6/M7) +
  planning (v9) + world model (v12-B).
- P5 — **full Tetris**: placement+rotation+line-clearing, shaped reward
  (height/holes/lines) + model-based lookahead.
Honest: full pixel Tetris is hard for the field (sparse/delayed reward, long
horizon); the macro-action placement framing + shaping + reuse is the
tractable path. Don't skip P3-P4 — that's the real bridge.

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

## BIG DIRECTION (2026-05-30): PERCEPTION + WORLD-MODEL (one GPU, unlimited time)
Reframe from the owner: only the local GPU, but time is NOT a constraint.
=> stop toy symbolic minutes-long runs; USE the GPU fully (CNNs, RSSM,
hours/days of training). Move from "given clean symbolic state" to "the agent
SEES the world (pixels) and learns a world model" — the project's RSSM
foundation, finally at real (single-GPU) scale. The honest path from
validated toy MECHANISMS to an AI that learns like we wanted.
  Phase A — perception: learn skills from PIXELS (CNN encoder).        [NOW]
  Phase B — RSSM world-model from pixels (predict/dream/plan).
  Phase C — developmental loop (skills/discovery/composition) on the LEARNED
            latent, on progressively richer worlds.
Discipline unchanged: preregister, milestones, honest negatives, commit/push,
occasional plain-language explanations. Expect failures; report them.

## NOW (v12 perception program — status)
- [v12 Phase A] DONE/POSITIVE — collect_wood learned FROM PIXELS (CNN encoder,
  no cell-types given) reaching 1.00. The agent learned to SEE.
- [v12 Phase B] DONE/POSITIVE — RSSM world model from pixels: one-step recon
  + open-loop k-step prediction beats a persistence baseline. It predicts.
- [v12 Phase C] acting via the learned pixel model — CONCLUDED: NOT cracked
  (honest negative, parked). Dreamer (imagination actor-critic): degenerate
  0.00 on sparse AND dense. Random-shooting MPC: ~random (H6 0.58 vs 0.39
  was within the random baseline's own 0.39->0.61 run-to-run variance; H15
  0.59 vs 0.61 tied). The model PREDICTS (Phase B) but isn't actionable enough
  over a planning horizon for competent control in this budget. PARKED: needs
  a bigger/better world model, known Dreamer stabilization tricks, or stronger
  planning + much more compute. A (perception 1.00) + B (predicts, beats
  persistence) stand as the program's wins.
- [v13] DONE/NEGATIVE (robust N=3) — naive perceptual-encoder reuse HURTS:
  warm-starting the collect_wood CNN encoder to learn stone/coal/iron was
  ~1.7x SLOWER than scratch (speedup 0.60x), FROZEN failed outright (0/3).
  Low-level features specialized to one target colour are the wrong bias for
  a sibling target. Sharpens the story: reuse must be at the SKILL/PREREQUISITE
  level (M3/v7), not raw features -> motivates v13b.
- [v13b] DONE/STRONG POSITIVE (N=3): M3 compounding holds FROM PIXELS. With
  prerequisites reused, a CNN policy masters every skill up to make_iron_pickaxe
  (depth 6) at 1.00 from raw pixels, each in ~10 iters regardless of depth; the
  FLAT agent (no reuse) fails at depth >= 1 (can't even chain wood->table under
  sparse reward). Reuse decomposes an unlearnable deep problem into shallow
  learnable steps -> the project's HEART, validated on pixels.
- [v14] DONE/STRONG POSITIVE (N=5) — CAPSTONE: SELF-DIRECTED MASTERY FROM PIXELS.
  Given pixels and NO goals, 5/5 seeds autonomously sequence + master the FULL
  9-skill tree (DAG-valid, reaching iron_pickaxe) learning every skill from
  pixels, ~10 min. Unifies v7 (sequencing) + v12-A (perception) + v13b
  (compounding). CAVEAT (phase-gate review): the novelty/frontier signal reads
  inventory (a reachability ORACLE) -> the curriculum ORDERING is env-gated, not
  pixel-discovered; the from-pixels result is the perceptual skill-mastery +
  sequencing. True pixel-based novelty = future work.
- [v13c] DONE/POSITIVE (N=3) — fair shaped-flat baseline (phase-gate review
  response): a flat agent with a DENSE per-achievement reward, no reuse, 2.46M
  steps from pixels, reaches ONLY depth 0 (collect_wood) and fails every deeper
  target. Refutes "v13b flat failed only due to sparse reward" -> reuse, not
  shaping, unlocks deep skills from pixels. v13b claim survives the fair test.
- [DELIVERABLE] one-command local demo showcasing the VALIDATED developmental
  loop (compounding + autonomous discovery) so Jeremie can run/watch/test it.

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
