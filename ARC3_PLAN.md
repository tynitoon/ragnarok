# ARC 3 — "ALCHEMY" (v61): a family of worlds that share hidden laws

DRAFT until the STAGE-0 freeze (section 4.0). Written 2026-09-06 after a 20-agent design panel (4 designs,
3 judges, 3 adversarial auditors, 2 revisions) whose final draft was still refuted 3/3 at 68k characters;
this document keeps its substrate, which every lens rated top-2 and no auditor attacked, and cuts the
measurement apparatus to what survived. A second panel of 4 auditors then reviewed THIS document
(section 9 is the ledger). Complexity in the measurement is where five ARC 1-2 verdicts died; it is
treated here as a defect in its own right.

## 0. Mission and north star

Show that the agent learns and USES what it learned to LEARN FASTER in a world it has never seen — both
arms learning, an unseen world, a learning curve. Longer aim: competence that keeps GROWING as worlds are
added. Partial success counts. The deliverable is one honest figure a non-specialist can read.

## 1. The wall, and why the substrate changes

ARC 1 (v49-v57): reuse pays only when knowledge is EXPENSIVE TO REDERIVE; resetting worlds made every
recipe cheap; three frozen NULLs. ARC 2 (v58-v60): frozen weights do transfer to an unseen world (3/6 vs
0/6 random), but once both arms learn, a fresh agent catches up in 1-3 rounds: Delta +0.069 vs se_null
0.076 (ARC2_PLAN section 15). Same wall from the other side: what is SHARED between two random-DAG worlds
("resources vs crafts, tools gate resources, try and see") is reacquired in ~2 rounds, and the
hand-designed store does the rest. Sixty versions changed the agent and the instrument; none changed the
one thing the wall is made of. ARC 3 changes the substrate and nothing else about the protocol.

What a substrate needs so that "learns faster" is even possible: (i) something shared across worlds that
a fresh agent needs MANY rounds to reacquire; (ii) a skin that varies enough that nothing item-specific
can be memorised; (iii) the shared thing only PARTIALLY exposed per world, so living in more worlds keeps
adding knowledge (otherwise the accumulation curve is flat after world 1, the negative the north star
warns about). And one design law the panel wrote and its auditors re-derived on this document: a memory
that PERSISTS across the goals of a world hands a fresh agent the shared law before the law is ever
costly. Section 2.5 draws the consequence.

## 2. The substrate

### 2.1 Laws — sampled ONCE (laws seed 31337), shared by every world, never observed

    K = 14 elements, T_MAX = 4 tiers. An ITEM is (element, tier). Raws are tier 1.
    Bond[K,K]   symmetric bool, diagonal False, density 0.5 off-diagonal, rejection-sampled until every
                element bonds with >= 2 partners.                      MEASURED: 44 true cells of 91
    Out[K,K]    symmetric element index, defined where Bond holds, sampled as a PROPER EDGE COLOURING of
                the bond graph (for every element a the products Out[a,b] over its partners are all
                distinct) — so no two recipes of one element collide.
    TIER law    combine(x, y) is LAWFUL iff tier[x] == tier[y] = t < T_MAX and Bond[el x, el y].
                A lawful combine yields the item (Out[el x, el y], t+1).
    OUTCOME     product : lawful and the product item exists in this world -> inputs consumed, +1 product
                inert   : lawful and the product does not exist here     -> inputs returned, nothing made
                botch   : not lawful                                     -> BOTH inputs destroyed
    GATE law    a raw is FREE or GATED; a gated raw can be collected only while holding some item of
                tier >= 2 (any element).
    QUOTA       each raw type yields at most 4 units per 48-step episode (a counter in the env; the grid
                is never touched, so the frozen nav skill sees exactly the grids it was calibrated on).

Why this is expensive without the law: the only way to learn a Bond/Out cell is to hold two items of the
right elements and tier and PROPOSE them. A botch at tier >= 2 destroys inputs that cost 4 / 8 raw units
to rebuild; a botch at tier 1 burns the episode's quota. 91 cells; a world exposes ~38% of the true ones.
Why three outcomes: a Bond-true pair whose product is absent in this world must not teach "this cell
fails", or the cross-world table gets poisoned. INERT keeps Bond exact.

### 2.2 Skin — per world seed

    (a) raws: 6 elements drawn from K without replacement -> cells 1..6 (nav skill range 1..9; n_cells 7).
        Four raws FREE, two GATED (assigned before the permutation).
    (b) roster: for t = 2..4, the products the law admits from the REAL items present at t-1; then FIXED
        SLOT COUNT 6 per tier (random drops logged; empty slots filled with DECOYS = elements producible
        neither as a real nor as a dropped product at that tier, never admitted as goals, inputs = {}).
        n_items = 24 always, so counts leak nothing.
    (c) item indices permuted (permute_spec rule); cell IDs permuted within the used set (v58 rule);
        grid positions re-rolled every episode.
    Stable across worlds: the (element, tier) of an item and the law. Not stable: index, cell, position,
    existence, partners, which cells the world exercises.

MEASURED on our implementation (scripts/mc_v61.py, 200 seeds; the panel's independent implementation gave
the same numbers to two decimals): real items per tier 6.00 / 5.66 / 5.26 / 4.64; 2.1 products dropped
per world; P(world admitted) 0.95; bond cells exposed per world 0.38 (sd 0.10); cumulative coverage of a
random admitted 4-world lineage 0.38 / 0.62 / 0.76 / 0.85 (sd 0.06 at 4).

### 2.3 Generator and env — ragnarok/environments/law_world.py

    sample_laws(seed, K=14, density=0.5) -> dict(K, T_MAX, Bond, Out)
    gen_law_world(seed, laws, n_raw=6, n_free=4, slots=6, quota=4) -> spec with the v58 fields (n_items,
        n_cells, kind, cell, tool=-1, inputs={}, craft_actions=[]) plus el, tier, gate, decoy and the
        EXPERIMENTER-SIDE tensors (execution and scoring only, never observed): pair_out[24,24] (product
        slot | INERT | BOTCH), chains (cheapest recipe tree per item), plans (the law-knower's plan),
        admitted (per tier), world_ok.
    permute_law_spec(spec, seed): every tensor permuted consistently (test (5) asserts equivariance).
    ADMISSION of a world (mechanical, logged, never inspected): world_ok = at least one tier-2 product
        from the free raws AND >= 1 admitted goal at each tier 2, 3, 4, where an ADMITTED GOAL is a real
        item whose plan (below) fits the quota and 48 macro-steps. At run time the nav gate (>= 0.85 per
        cell type) is added; a world failing it is skipped and logged. Test (10): the plan executes every
        admitted goal of 200 seeds from an EMPTY inventory on a symbolic executor (3030/3030).
    The PLAN (law_world.plan_for): gather-then-craft. If the chain needs a gated raw, a tier-2 tool from
        two FREE raws is built first — the chain's own tier-2 item when it has one (only its FIRST
        production moves to the front; a reused intermediate is produced again where the chain needs it,
        the defect test (10) caught), else an extra one; gated units are collected while the tool is held,
        then the free units, then the chain bottom-up.
    LawVecTechTree(DeviceVecTechTree): no craft actions (action 5 = NOOP); collect requires the gate
        satisfied AND quota left, and decrements the quota; combine(i, j) -> outcome in {PRODUCT, INERT,
        BOTCH, INVALID}. Egocentric view, wall sentinel, nav mode untouched. SMOKE (16 envs, world 8100):
        nav gate min 0.898; L masters tiers 2/3/4 at 1.00/1.00/1.00.

### 2.4 Observation — nothing computed from pair_out, Bond, Out or gate tensors

    per slot i (28 slots, 24 valid), 32 floats in [0,1]:
      base 8    in_inv, unlocked, tried_ep, succ_last, is_goal, is_resource, is_valid, quota_frac
      attrs 16  element one-hot (14), tier/4, gate/2
      store 8   n_succ_log, n_fail_log, obtained_ever, since_succ_log, frac_minctx_held, ctx_new,
                known_recipe, needed
    per unordered pair i<j (378), 5 floats: pair_succ_log, pair_nonprod_log, pair_tried_ep,
                pair_succ_ep, pair_yields_needed_unheld
    Row = 28*32 + 378*5 = 2786 bytes uint8 (single quantisation, test (7)).

DISCLOSED GRANTS, stated once: is_resource, is_valid, quota_frac (the agent's own counter); element
labels (the law's shared vocabulary; the first layer learns a 14-symbol table and this plan says so);
tier; gate flag; and the ROSTER (which (element, tier) items exist here is a consequence of the law; the
count is fixed, the element SET per tier carries some information, and test (14) measures how much: the
write-up must say the transferred object may be roster-to-cell inference as well as memorised cells).
None of them says "will this combine succeed". Identity-freeness is claimed over ITEM SLOTS only.

### 2.5 Store — scripts/pair_store_v61.py, PairStore, one per parallel env, never pooled

    write_pair(i, j, outcome): PRODUCT -> pair_succ += 1, pair_out = product slot; INERT ->
        pair_inert += 1; BOTCH -> pair_fail += 1. Exact: deterministic law, both inputs held, no nav
        skill in the path, so failure-elimination is SOUND for pairs (unlike raws).
    write_raw(g, held_before, obtained): the ARC 2 rule for raws (failures only count — nav timeouts and
        quota refusals must never eliminate a true gate); the gate-context candidates C[g] intersect on
        success; frac_minctx_held and ctx_new are what the policy reads.
    derived: known_recipe[k] = any(pair_out == k); needed = closure from the goal over KNOWN recipes;
        pair_yields_needed_unheld. state_dict / load_state_dict; eval envs get a COPY (test (13)).
    LIFETIME — the design law of section 1, decided: the store is PER-GOAL working memory. It is EMPTY at
        every goal entry for every arm and persists across the episodes and rounds of that goal only.
        The second panel measured what a store persisting across the stream does: at tier-3 entry a
        fresh arm's per-env store already holds ~0.99 of the world's tier-2 recipes and ~0.7 of its tier-3
        recipes from incidental proposals alone; at tier-4 entry ~0.84 of tier-3 and ~0.4 of tier-4
        recipes, so the tier-4 goal is handed over before the law is ever costly and the law's worth
        collapses to a flat offset on one panel (Delta ~0.07, H 0.07-0.10, a STOP-too-easy at the gate).
        With a per-goal store, everything that crosses a goal — this world's earlier recipes for the
        fresh arm, four worlds of chemistry for M — crosses in the WEIGHTS, which is the object the
        claim is about. Buffers are per-world (2.6), so the weights keep learning from earlier goals.

### 2.6 Policy — scripts/pair_net_v61.py

PairNet scores an ACTION MATRIX: diagonal (i,i) = "pursue raw i" (valid iff is_resource & valid &
quota_frac > 0), off-diagonal i<j = "combine(i,j)" (valid iff both held & valid); 28 + 378 = 406
actions. Pair cells are encoded SYMMETRICALLY: [x_i + x_j, x_i * x_j, mean over valid slots, goal
attrs, pair feats, is_diag] = 118 -> 128 -> 128 -> heads: policy logit; OUTCOME head 3-way
(botch / inert / product) + product element (14-way). No nn.Embedding over slots, no parameter shaped
to a slot count, permutation-equivariant over slots (tests (5), (6)).

Act: stochastic = softmax over valid cells + 5% uniform over valid. Deterministic (eval) = argmax under
the context-aware mask: a diagonal slot is masked iff (tried_ep & ~succ_last & ~ctx_new) or quota_frac
== 0; a pair is masked iff pair_nonprod > 0 or (tried this episode and not productive this episode). If
every valid cell is masked the argmax falls back to the unmasked valid set (no absorbing state; symmetric
for every arm). Identical for every LEARNED arm (M4, Fa, Fb). The references L and G' act by their own
rules and are unmasked (a nav timeout would otherwise block L's legitimate retry); the smoke measured L
at 1.00 unmasked. (Review correction: the first draft said "L with the mask on".)

Training per round, identical for every arm: 300 steps, bs 512, Adam 3e-4 constructed fresh at every
world entry; buffers (policy 200k rows, outcome 100k rows, uint8) are per-WORLD, empty at world entry,
and eval episodes never enter them (test (13));
    loss = CE(policy, relabelled commanded action)   relabel_commanded_v58's rule on this row layout:
                                                     one commanded goal per episode, credit only under
                                                     it, 0.7^lag (D4)
         + CE(outcome 3-way, observed) + CE(product element | product)   on the outcome rows of every
                                                     training combine
RULING on the outcome head (the panel's open question 1): it is supervised learning of what the agent
OBSERVED — the same bit the store writes — with no goal credit in it. Constraint 2 governs the policy's
credit and is untouched. Every arm has it. It is the place the shared law can live in weights; without
it the table would have to be learned implicitly through policy imitation, which is the weak signal ARC
2 measured. Evidence dropout: none.
Initialisation: every fresh net is constructed under a PER-UNIT torch seed (4.7), so init variance
enters the contemporaneous null by construction. M loads lineage weights; only its sampling stream is
per-unit. Nav-skill draws come from the global torch RNG under the arm's stream: arms are paired on
GRIDS and eval seeds, not on nav outcomes (symmetric noise that lands in se_null).

### 2.7 Hand-coded references (ceilings, never arms of the claim) — scripts/hand_v61.py

    L   LawKnower: attributes + the TRUE tables (experimenter-side); executes the plan of 2.3 per env
        with a plan pointer that advances on the observed inventory change. The ceiling WITH the law.
    G'  StoreSweeper: the store, the tier law and the gate law, NO Bond/Out. ONE rule shared by the
        CPU model and the GPU reference (scripts/sweeper_v61.py): build the goal if its recipe is known;
        else build toward the CHEAPEST untried equal-tier pair among PRODUCIBLE items (raws with quota,
        or known recipes whose inputs are producible), random ties; else build a known tier-2 tool to
        unlock gated raws; else refill. The ceiling WITHOUT the law. (The first draft's sweeper only
        tried pairs among items it happened to hold and stalled at tier 3-4 — a reference that cannot
        climb is not a ceiling; the CPU model caught it before any GPU was spent.)
Both expose act(obs, env) / no-op train, scored by the same run_goal on the same criterion and schedule
(the ARC 2 section-14 lesson).
MEASURED, CPU model with perfect nav, per-goal store from empty, cap 576 attempts (= 3 rounds):
tier-2 goals ~27 attempts (0% censored); tier-3 goals 70-350 attempts (0% censored: within 1-2 rounds);
tier-4 goals censored in 67-97% of runs on 6 of 7 worlds tried (8301: 166). WITHOUT THE LAW, tier 4 is
mostly out of reach in 3 rounds even for an ideal sweeper; WITH it, L takes 15 macro-steps.

### 2.8 The unit: a world and its 3-goal stream

A UNIT is one admitted world, entered by every arm with weights only: buffers and optimizer empty, the
store empty at EVERY goal entry (2.5). Inside it, the same CHAIN-BLIND stream for every arm:

    goals = [ admitted goal of tier 2, of tier 3, of tier 4 ], each the admitted real item of that tier
            with the LOWEST permuted index (a coin flip the experimenter never touches — never "the
            chain of the tier-4 goal", which would be a curriculum computed from the true recipes)
    each goal: B_max rounds at FIXED budget, no early break; eval before the first round and after
            every round -> master_per_round[0..B_max]
    credit: relabel toward the commanded goal of each episode only

Why a stream: an untrained net is uniform over valid actions and never reaches a tier-4 item; the fresh
arm needs rungs. The tier-2 goal IS the rung: a uniform policy unlocks it in ~0.5 of episodes (0.23 on
the ~40% of units whose tier-2 goal needs a gated raw; min 0.055 per episode over 24 units), so every arm
gets credited rows in round 1. The tier-2 panel is expected FLAT-IDENTICAL across arms from b = 1 and is
captioned as the rung, not as evidence. The law is read on the tier-3 and tier-4 positions.
Why the fresh arm's tier-4 goal is costly under a per-goal store: from an empty store it must rebuild
the tier-2 items it knows only in its weights, then sweep tier-2 and tier-3 pairs at 4-8 raw units per
botch; the ideal store-only sweeper (2.7) mostly does not reach tier 4 in 3 rounds, and a learner is
slower. The three positions therefore play three roles, declared here: tier 2 = the RUNG (every arm
~1.0 from b = 1); tier 3 = the LEARNS-FASTER position (a fresh agent reaches it inside the budget, an
experienced one sooner); tier 4 = the REACH position (with the law it is a 15-step plan; without it,
mostly not solved in the budget). The gate checks the tier-3 position has dynamic range (4.1).

## 3. What transfers, the mechanism, and the predictions

What M carries (weights only): (1) an outcome model over element PAIRS — the Bond/Out cells it exercised
in earlier worlds, plus the tier law — i.e. a partially observed, memorised LAW TABLE keyed by the
disclosed element vocabulary (or, in part, a roster-to-cell inference; test (14) bounds it); (2) a policy
that reads it; (3) the generic evidence-reading and exploration skills of ARC 2. Mechanism on entry to a
new world: M maps the roster to (element, tier) from the attributes, proposes pairs its table says are
lawful, and skips the sweep; the fresh arm sweeps. Accumulation: each world exposes ~38% of the cells;
coverage after 1 / 2 / 3 / 4 worlds 0.38 / 0.62 / 0.76 / 0.85, so M4 knows more cells of the test world
than M1 — a rising curve has a MECHANISM here, not a hope. No composition claim is made.

Predictions stated before the gate, on THIS protocol (per-goal store, chain-blind stream):
    tier 2   every arm ~1.0 from b = 1; Delta ~0 (the rung)
    tier 3   fresh climbs within 1-3 rounds from its weights' tier-2 knowledge (the ideal sweeper needs
             70-350 attempts); M4 ahead at b = 0..1, converging by B_max -> the learns-faster panel
    tier 4   fresh mostly does not reach it inside B_max (ideal sweeper censored in ~80% of worlds at
             576 attempts); M4 with the cells covered plans it in ~15 steps -> the reach panel, a large
             flat offset that is "uses what it learned", NOT "learns faster" (the wording rule of 4.4)
    Delta_learn (mean over the 3 positions, b = 1..B_max): 0.15-0.35 for M4 (mostly tier 4); tier-3
             Delta 0.1-0.3; ~0 for M1 at tier 3
    accumulation: flat at M1, rising at M2 and M4
The second panel's stream simulations (PERSISTENT store) gave Delta 0.00 at c = 0.37 and 0.07-0.10 at
c = 0.83 against an ideal fresh arm, H 0.07-0.10: that is the number the per-goal store is designed to
lift, and the gate measures whether it did. INCONCLUSIVE is the modal outcome at these magnitudes and is
accepted as one (4.4 says what it licenses). Numbers that the gate replaces: none of the above gates.

## 4. Measurement

### 4.0 STAGE-0 FREEZE
Frozen in the commit that runs gate_v61.py, BEFORE K0: K 14, density 0.5, T_MAX 4, quota 4, n_raw 6,
n_free 4, slots 6, laws seed 31337, the goal rule and the stream (2.8), the per-goal store (2.5),
cfg_v61 (= cfg_v58 values: 64 envs, macro_budget 48, 4 episodes per round, 300 train steps, thresh 0.6),
B_max(gate) = 4, the seed table (4.7), the notch list (4.1). Nothing about the substrate changes
afterwards except by the one notch, which is itself pre-declared here.

### 4.1 THE CHEAP GATE — gate worlds = the first two admitted seeds >= 8100 (8100, 8101; burned)
K0 — hand-coded, ~1.5 GPU-h. Nav gate. L on every goal of both units, one deterministic eval each,
mask on. G' from an empty store on both units, B_max(gate) rounds of episodes per goal, no training,
attempts-to-first-obtain recorded per goal.
    REACHABLE   iff nav_min >= 0.85 AND m_L(g) >= 0.85 on every goal of both units. A REACHABLE failure
                is an INSTRUMENT defect (L already passed test (10) on this world): fix, log, re-run K0.
    CONSISTENT  (reported) G' attempts-to-first-obtain per goal — the MEDIAN over the 64 envs, each
                censored at the goal's budget (B x 4 x 48 = 768 at the gate) — beside the CPU model's
                prediction of the SAME statistic (median over runs, same cap, same rule, from an empty
                per-goal store). A miss by > 2x on a non-censored goal halts K1 for a bug hunt. (Review
                correction: the first draft compared the min over envs against a single-run median.)
K1 — learned, ~4.4 GPU-h. Both units, three fresh arms Fa, Fb, Fc (per-unit init seeds), B_max(gate) =
4 rounds per goal, identical grid/eval seeds across arms.
    Measured: curves m_a(g, b); b*(g) = first round at which the median fresh arm reaches 0.6 (inf if
              never), printed for every tier; A_a(u) = mean over the unit's 3 goals of the mean over b =
              1..B of m_a(g, b) — THE SAME OBJECT the confirmatory scores;
              sd_init = sd of the pairwise per-UNIT differences {A_Fa-A_Fb, A_Fa-A_Fc, A_Fb-A_Fc}(u)
              over the 2 units (6 values, df printed; the per-goal sd printed beside it as a diagnostic
              only — the first draft took the per-goal sd here, which is not the confirmatory's unit:
              the ARC2_PLAN section-8 defect in the false-STOP direction, caught by all four auditors);
              H = mean_u[A_L(u) - mean(A_Fa,A_Fb,A_Fc)(u)], printed per tier as well;
              H' = mean_u[A_L(u) - A_G'(u)]; outcome-head AUC of the fresh arms on the gate world's true
              Bond table after each round; credited rows per round per arm; wall-clock per round.
    FRESH-LEARNS  iff b*(tier-2) <= 2 on both units AND b*(tier-3) <= 3 on both units
                  (the learns-faster position must have dynamic range for a fresh agent; tier 4 is the
                  reach position and is NOT required — b*(tier-4) is printed)
    ROOM          iff H >= 4 * se_proj, se_proj = max(sd_init / sqrt(2N), se_res), N from 4.4
                  (DEMONSTRATED needs Delta >= 2 se; assuming M captures at most half the headroom, the
                  headroom must be 4 se. sd_init has ~5 df: a ROOM decision near the line is a coin
                  flip, and the prereg says so.)
PROCEED iff REACHABLE and FRESH-LEARNS and ROOM.
NOTCHES — at most ONE in the whole arc, then STOP:
    not FRESH-LEARNS  -> soften: the store PERSISTS across the stream (2.5's alternative, measured by
                         the second panel to hand the fresh arm the rungs) and B_max(test) = 4; K1 is
                         re-run on the same units (+4.4 h); PROCEED then requires FRESH-LEARNS and ROOM
                         on the re-run; else STOP. (The tier-3 position is what this notch must open.)
    not ROOM          -> harden: quota 4 -> 3 (a sweep-cost lever; the panel measured it does not starve
                         the uniform bootstrap: min 0.040 per episode); regenerate the gate worlds (next
                         admitted seeds), re-run K0+K1 once. A ROOM failure whose cause is sd_init alone
                         (H above 0.10 but 4 se above H) is reported as such; the notch still applies.
    not REACHABLE     -> instrument defect, not a notch.
STOP is published as a GATE RESULT with its numbers ("the substrate as built is too easy / too hard /
unresolvable at this N for this instrument"), never as a wall measurement. Cost of a STOP: <= ~6 GPU-h
(<= ~11 with a notch).
After PROCEED the prereg is appended to preregistration.md (v61) with B_max(test), N, the notch if any,
the seed table, and every K0/K1 number; then it is frozen.

### 4.2 Pretraining — three independent lineages s = 0, 1, 2
Lineage s lives in the first 4 admitted seeds of 8200+10s .. 8200+10s+9, in order (mc_v61: s0 = 8200-8203,
s1 = 8210-8213, s2 = 8220-8223, unless the run-time nav gate skips one). Per world: the stream of 2.8
extended to 4 goals (tier 2, 3, and the two lowest-index admitted tier-4 goals), B = 3 rounds per goal,
per-goal store, buffers and optimizer rebuilt at entry, weights carried. Checkpoints M1..M4 per
lineage. Recorded for free: per-world learning-curve areas (world k's area given k-1 prior worlds, with
the world's L and G' areas beside it as difficulty context), exercised-cell coverage per checkpoint.
Cost ~8.8 GPU-h. Pretraining rounds may differ from the test budget: fairness binds only at test.

### 4.3 Mechanism probe — world 8150 (burned), after pretraining, before the confirmatory, ~0.5 GPU-h
Frozen checkpoints M1..M4 of lineage 0 and a random-init R, weights frozen, store empty:
    (a) FIRST-PROPOSAL LAWFULNESS: over 256 episodes, the fraction of each arm's first combine per
        episode that is Bond-true. PROCEED iff r(M4) - r(R) >= 3 * sqrt(r(1-r)/256) pooled. If M4's
        first proposals are no more lawful than a random net's, the weights carry no law: STOP, and the
        finding is an architecture result ("the outcome head did not put the table into the policy"),
        fixed before any confirmatory compute, never after. (At first proposals every held item is a
        raw, so the tier law cannot pass this on its own; a roster-to-cell prior can, and the caption
        says "law-carrying or roster-inferring".)
    (b) reported: outcome-head AUC on 8150's true Bond table, split into cells exercised in pretraining
        vs unexercised, beside the roster-only ceiling of test (14); r(M1..M4) as the accumulation of the
        law in the weights.

### 4.4 Confirmatory — per lineage s: the first N/3 admitted seeds of 8300+10s .. 8300+10s+9
N = 12 units (4 per lineage) at B_max(test) = 3; N = 9 if B_max(test) = 4 or the smoke measures > 260 s
per round (declared at the freeze, never after). Arms per unit, identical grid/eval seeds, weights only
at entry:
    M4    lineage s weights            Fa, Fb   fresh, per-unit init seeds
    L     one eval per goal (flat)     G'       lineage-0 units only (budget), no training
    B_max(test) = clamp(b*(tier-4, median over gate units) + 1, 3, 4), from K1.
PRIMARY (monotone; pointwise dominance can never score lower):
    A_a(u) = mean over the unit's 3 goals of the mean over b = 1..B_max of m_a(g, b)
             — b = 0 is EXCLUDED from the verdict statistic: at every goal entry the store is empty, so
             b = 0 is a weights-only head start, reported separately (below), never part of the verdict
    Delta_learn = mean over units of [ A_M4(u) - 0.5 (A_Fa(u) + A_Fb(u)) ]
    se_null     = sd over units of [ A_Fa(u) - A_Fb(u) ] / sqrt(2N)      (contemporaneous, includes init
                  variance by 2.6; never a transplanted number)
    se_res      = 0.0625 * sqrt(2 / (3 * B_max * N))                     (the 1/64 eval resolution floor)
    se          = max(se_null, se_res)
    DEMONSTRATED      iff Delta_learn >= 2 se AND Delta_s > 0 for each lineage s
    NOT-DEMONSTRATED  iff Delta_learn <= 0
    INCONCLUSIVE      otherwise
    Instrument line: t_init = mean_u[A_Fa - A_Fb] / (sd/sqrt(N)); |t_init| >= 2 -> INCONCLUSIVE with the
    reason printed (the two fresh arms are exchangeable by construction; a systematic difference is a
    seeding or code defect). Under a valid null this fires in ~6% of runs; stated here.
REPORTED, never in the verdict: Delta per tier position; Delta_0 = mean_u[m_M4(g,0) - 0.5(m_Fa+m_Fb)(g,0)]
per tier — on the tier-2 goal it is the ARC 2 frozen-transfer replicate (12 units, weights only, no
in-world experience); on tiers 3/4 it is the weights-only head start after this world's earlier goals;
the endpoint gap m(g, B_max) per arm and unit (so an M4 that starts ahead, learns nothing and is
overtaken is visible: the b = 1..B_max area alone can reward a head start that the fresh arms are slow
to erase); Delta_s with se_s = sd_s/sqrt(N/3) per lineage; the robustness null sd_u[A_M4 - 0.5(A_Fa +
A_Fb)]/sqrt(N) beside se_null; the treatment side has three checkpoint draws (one per lineage) and its
init variance is not in se_null — the per-lineage clause is what covers it; A_L and A_G' beside every
arm; credited rows per round per arm; the full 0..B_max area as a secondary.
WORDING RULE (pre-declared, caption only): the verdict sentence says "learns faster" only if the
tier-3 position alone satisfies Delta(tier-3) >= 2 se(tier-3) (its own contemporaneous null over the 12
units); otherwise a DEMONSTRATED verdict is worded "solves with prior knowledge what a fresh agent does
not learn in the budget" — the reach panel is "uses what it learned", and the area statistic cannot
tell a head start from faster learning on its own (the auditors' accepted residual).
WHAT INCONCLUSIVE LICENSES: the curves, Delta_learn with its se, the per-tier and endpoint lines, and
the sentence "not distinguishable from zero at this N"; no verdict wording, no mechanism label.
MECHANISM label (caption only, never the verdict): "consistent with a carried bonding table" iff the
probe passed AND the outcome-head AUC on the test worlds' exercised cells >= 0.8, with the roster-only
ceiling printed beside it; otherwise the caption attributes the advantage to prior experience only.

### 4.5 Accumulation — DESCRIPTIVE, no verdict label
(i) within pretraining: per-world learning-curve area of world k given k-1 prior worlds, lineage by
lineage, normalised by the world's L area (difficulty) and with G' beside it (free); (ii) the probe's
r(M1..M4) on the fixed held-out world 8150 (free) — the default second panel: "the law in the weights
accumulates with worlds"; (iii) lineage 0 only, first on the cut list: the fresh arm Fa on the four
pretrain worlds of lineage 0 (2.9 GPU-h), a paired M_{k-1}-vs-fresh point per world with no band.
Worlds-lived-in is collinear with gradient steps and this is said in the caption; no "ACCUMULATION
SUPPORTED" wording; "competence grows with worlds" is not claimed from (i)-(iii).

### 4.6 Budget (220 s/round assumed; re-measured by the 64-env smoke, test (12), before the freeze)
    K0 1.5 | K1 4.4 | pretraining 8.8 | probe 0.5 | confirmatory at B_max 3: 12 units x 3 goals x 3 arms
    x 11 min = 19.8 + G' on 4 units 1.6 | accumulation (iii) 2.9          total ~39.5 GPU-h
    at B_max(test) = 4: confirmatory 9 units x 3 x 3 x 14 min = 18.9 (N = 9)
    CUT LIST, in order, applied at the freeze if the measured round time demands it: (1) accumulation
    (iii); (2) G' to 2 units; (3) N = 9 at B_max 3. Never cut: Fb, a lineage, the probe. Hard ceiling
    40 GPU-h.

### 4.7 Seeds and burned worlds
    laws 31337 | gate: first two admitted >= 8100 | probe 8150 | pretrain 8200-8229 by lineage |
    test 8300-8329 by lineage, never inspected before the run (admission is mechanical)
    grid seed per (world w, goal position p) = w + 11 p, eval seed +9, identical across arms
    init seed per (lineage s, world w, arm a) = 100000 + 1000 s + 10 (w - 8000) + a; fresh arms a in
    {1,2,3}; the lineage's own initial net a = 4 at its first world; gate arms use s = 9. Distinct
    across gate / pretrain / test by construction (w - 8000 differs by role).
    sampling stream per (unit, arm) = init seed + 500
    Burned: 8100-8199 (gate, probe) and any world a human looked at. 6100/6101 stay burned.

## 5. Implementation plan (ordered; nothing runs on GPU before step 7)

    1  ragnarok/environments/law_world.py — laws, generator, permutation, admission, plan,
       LawVecTechTree.                                                                  DONE (3a526e9)
    2  scripts/hand_v61.py — L (planner) and G' (sweeper) as composers.                 DONE (03fee1d)
    3  scripts/mc_v61.py — CPU Monte Carlo: generator table, admission lists for every named seed window
       (chain-cost clauses; the nav gate is added at run time), lineage coverage, ideal-sweep attempts
       per admitted goal from EMPTY (--sweep). Printed by test (9).                      DONE / --sweep
    4  scripts/pair_store_v61.py — PairStore, PairEnv, masks.                            DONE
    5  scripts/pair_net_v61.py — PairNet, ComposerV61, BufferV61, collect / relabel / eval /
       run_goal_v61 / run_unit_v61 (per-goal store), cfg_v61, seed table.                DONE
    6  scripts/test_law_v61.py — (1)-(7), (9), (10), (13) green; (11) nav gate and (12) the timed 64-env
       smoke run from gate_v61.py; (14) roster-only ceiling (scripts/roster_v61.py, CPU).
    7  scripts/gate_v61.py (K0 + K1, JSON + log, --resume, _smoke suffix), STAGE-0 FREEZE commit, run.
    8  preregistration.md v61 entry (frozen after PROCEED); scripts/score_v61.py written and frozen
       BEFORE pretraining, with a synthetic self-test that a pointwise-dominating arm never scores lower.
    9  scripts/pretrain_v61.py, probe_v61.py, confirm_v61.py (per-unit JSON checkpoints, --resume,
       detached launch); scripts/figure_v61.py (the deliverable, built from the JSON only).
    Dev: 2-3 more sessions. GPU: section 4.6.

## 6. Risks and the observation that reveals each early

    R1 fresh arm never climbs past tier 2 from a per-goal store (bootstrap fails at tier 3/4)
                                                                    K1 b*(tier-3), b*(tier-4) = inf
    R2 fresh arm sweeps the whole table in one round (too easy; ARC 2 again)   K1 H ~ 0, ROOM fails
    R3 the outcome head learns the table but the policy ignores it   probe (a) fails while (b) AUC high
    R4 roster grant predicts unexercised cells well (M's edge is the roster)   test (14) AUC high
    R5 M4 starts ahead, learns nothing, is overtaken                 endpoint gap line; per-round curve
    R6 round time balloons with the 406-action matrix                smoke (12); auditors' arithmetic
                                                                     says the env dominates, not the net
    R7 lineage coverage far below 0.38/world (4 goals exercise few cells)   pretrain coverage log
    R8 unit-level noise far above v60                                K1 sd_init; ROOM
    R9 M4 proposes fewer pairs than a uniform fresh arm and enters each goal with a SMALLER store
       when its prior misses the cell — a deflating asymmetry, stated not fixed
    R10 credited rows = 0 in a fresh arm's round (relabel returns None)   rows-per-round line

## 7. What is NOT claimed

Not composition; not "the agent understands chemistry"; not sim-to-real; not that the agent finds every
law (it exposes ~38% of the table per world and that is the point). The transferred object is a
partially observed memorised law table keyed by disclosed labels — or a roster-to-cell inference the
labels permit — plus tier/gate laws, plus exploration skill, and this is stated on the figure. The store
is per-goal by design: nothing in it crosses goals or worlds, and a positive result is about the
WEIGHTS. A negative or inconclusive result is published with its numbers, as sections 8, 11 and 15 of
ARC2_PLAN were.

## 8. The figure

One panel per tier (2, 3, 4): MEAN MASTERY (the verdict's own object) vs practice rounds b = 0..B_max,
for M4, fresh (mean of Fa/Fb with their spread as the band), L (dashed ceiling), G' (dotted); the tier-2
panel captioned as the rung. The b = 0 point is drawn and labelled "head start (weights only)", with
Delta_0 printed on the figure. Caption states the verdict (DEMONSTRATED / NOT / INCONCLUSIVE),
Delta_learn with se, Delta per tier, the endpoint gap, and the mechanism label with the roster ceiling.
A second, smaller panel: 4.5 (ii) by default, with the collinearity caveat.

## 9. Audit ledger (second panel, 4 lenses, on this document's first draft)

    all four   sd_init on the per-goal unit vs se_null on the per-world unit (false-STOP)   -> 4.1 fixed
    northstar  store persisting across the stream hands the fresh arm the rungs (measured)  -> 2.5 per-goal
               store; predictions restated for this protocol (3); CONSISTENT on the same protocol
    fresh      FRESH-LEARNS untested on the stream unit; gated tier-2 goals                 -> 2.8 numbers;
               b* printed per tier; notch = persistent store
    history    Delta_0 mislabelled; endpoint gap dropped; Delta_s by sign only; seed table
               collisions; eval-mask absorbing state; decoy/dropped overlap; rows = 0        -> 4.4, 4.7,
               2.6, test (9), R10
    leak       figure plots counts, verdict scores means; accumulation caption; power statement
               -> 8, 4.5, 3
    Cuts endorsed by every auditor: the chain ladder, evidence dropout, Mz/Mperm/Mdeg, N = 36, M1-M3 as
    test arms, COSTLY-0/1, the 68k apparatus. Residual accepted, not fixed: the b = 1..B_max area can
    reward a head start the fresh arms are slow to erase (the endpoint line makes it visible).

## 10. GATE RESULT, first run (2026-09-06 04:05-07:35, 3.5 GPU-h, per-goal store) -> STOP, notch applied

    REACHABLE   True   nav min 0.898 / 0.879; L = 1.00 on all six goals
    CONSISTENT  held   G' median attempts-to-first-obtain (cap 768) vs CPU model:
                       8100: 33 / 104 / 768   vs 29 / 103 / >768 (94% censored)
                       8101: 27 / 123 / 688   vs 29 / 123 / >768 (62% censored)
    K1 fresh    A(u) over b = 1..4:   8100  F1 0.499  F2 0.358  F3 0.517  | L 1.00  G' 0.658
                                      8101  F1 0.496  F2 0.345  F3 0.341  | L 1.00  G' 0.663
                b* (median over the three fresh arms): tier 2 [1, 1]; tier 3 [3, inf]; tier 4 [inf, inf]
                per arm, tier 3: 8100 {3, inf, 3}; 8101 {3, inf, inf}. Tier 4: 0 of 6 fresh arms, zero
                credited rows in every round (R10 as predicted).
    sd_init     0.126 (6 per-unit values); per-goal sd 0.211 (diagnostic)
    H           0.574 (per unit 0.542 / 0.606; per tier 0.005 / 0.717 / 1.000); H' = 0.340
    se_proj     0.030 under B_test 4, N 9  -> 4 se_proj = 0.119
    FRESH-LEARNS False (tier 3 on 8101)   ROOM True   -> STOP -> the one pre-declared softening notch
    wall-clock  K0 ~26 min per world; K1 8.4 min per goal-run at B 4 (~126 s per round incl. eval)

Reading, stated once: with a per-goal store the substrate is REACHABLE and has ROOM in abundance (the law
is worth 0.72 of mastery-area on tier 3 and all of it on tier 4), but a fresh agent climbs to tier 3
inside three rounds in only 3 of 6 inits. The learns-faster position needs a fresh agent that can learn
it; the notch (4.1) is the pre-declared way to give it one: the store persists across the stream, so
the tier-2 goal's incidental discoveries carry into tier 3, and B_max(test) = 4. The cost, measured by
the second audit panel before the run, is dilution of the law's worth on tier 3; the gate re-run
measures what remains. K0's sweeper and K1 are re-run on the same two units under the notch (~5.5 h);
PROCEED requires FRESH-LEARNS and ROOM on the re-run, else the arc STOPs at the gate.
Budget note for the freeze after PROCEED: at the measured 126 s/round a goal-run at B 4 costs 8.5 min,
so the confirmatory at N = 12 costs ~15.3 h and fits under the 40 h ceiling; N is declared at the freeze.

## 11. GATE RESULT, notch run (2026-09-06 07:36-10:48, 3.2 GPU-h, store persisting) -> STOP. ARC 3 closes at the gate.

    REACHABLE   True   L = 1.00 on all six goals (twelve of twelve over both runs)
    CONSISTENT  held   G' medians 8100: 33 / 9 / 768; 8101: 28 / 25 / 403 (tier 3 falls from 104/123 to
                       9/25 with the inherited store — the dilution the second panel predicted, measured)
    K1 fresh    A(u):  8100  F1 0.556  F2 0.398  F3 0.443   8101  F1 0.539  F2 0.463  F3 0.331
                b* tier 3 per arm: 8100 {2, inf, 4}; 8101 {4, 4, inf}  -> medians [4, 4]
                tier 4: 1 of 6 fresh arms reaches 0.09; the rest 0.00
    sd_init 0.121 | H 0.547 (per tier 0.005 / 0.645 / 0.990) | H' 0.333 | 4 se_proj 0.114
    FRESH-LEARNS False   ROOM True   -> STOP. The one notch is spent. The confirmatory does not run.

### What the gate established (both runs, 6.7 GPU-h)
1. The substrate is REACHABLE and the instrument is calibrated: the law-knower masters every goal; the
   CPU model predicts the store sweeper's cost to within a few percent on every non-censored goal.
2. The law is worth a lot here — this was the wall ARC 1 and ARC 2 could not build: against fresh
   LEARNERS the headroom is 0.65-0.72 of the mastery area on tier 3 and ~1.0 on tier 4; against the
   ideal law-free sweeper 0.33-0.34. Without the law, tier 4 is out of reach in four rounds for eleven
   of twelve fresh arms and for the sweeper on both worlds.
3. Fresh learners are BIMODAL on the learns-faster position: over twelve fresh arms (two protocols),
   about half reach 0.6 on tier 3 within 2-4 rounds and half stall below 0.35 for the whole budget.
   Per-goal store: 8100 {3, inf, 3}, 8101 {3, inf, inf}; persisting store: 8100 {2, inf, 4},
   8101 {4, 4, inf}. The persisting store moves the sweeper's clock by 10x and the learner's by ~0.
4. The frozen feasibility rule — median b*(tier 3) <= 3 on both units — was missed by one round on
   both units in the notch run, and two of the twelve tier-3 curves crossed 0.5625 (two envs of 64
   under the 0.6 line) at exactly the deciding round. The second audit panel wrote before the run that
   a gate decision near the line is a coin flip at these degrees of freedom; it was. That is a fact
   about the criterion's resolution, recorded here, and not a licence to move it after the fact.

### What this is, and is not
It is a FEASIBILITY result: on this substrate at this budget the fresh baseline does not reliably learn
the position where "learns faster" would be read, so the pre-registered comparison cannot be made as
designed. It is NOT a null on the claim (no experienced arm was ever run), not a wall measurement, and
not a defect of the substrate's design goal — the law's headroom is the largest effect this project has
ever had in front of it.

### Options after a gate STOP (the owner's decision, not the lead's — spending after a frozen STOP is
### exactly the line this project does not cross on its own)
A. Close ARC 3 at the gate. Publish sections 10-11 as the result. Cost so far 6.7 GPU-h of 40.
B. v62: the same substrate and code, a NEW preregistration whose feasibility criterion is set from these
   twelve measured curves, gated on NEW burned worlds (8102, 8103) so nothing is fitted on the data that
   produced the STOP: e.g. B_max 6 per goal at gate and test (the measured b* of the learning half is
   2-4), and FRESH-LEARNS = "at least two of three fresh arms reach 0.6 on tier 3 within B_max on both
   units" (a fraction, not a 3-arm median). The verdict rule, the primary, the null, the wording rule and
   the probe are unchanged. Budget: gate ~8 h (B 6), pretraining ~9 h, probe 0.5 h, confirmatory at
   N = 9 and B 6 ~17 h -> ~35 h more; total ~41 h, i.e. the confirmatory needs N = 9 and G' on two
   units, or a raised ceiling. Labelled everywhere as a redesign after a feasibility STOP.
C. Run the confirmatory anyway under the wording rule. Rejected by the lead: it changes the gate after
   seeing it fail.

## 12. v62 — the re-gated arc (owner's decision B, 2026-09-06 11:30). A redesign after a feasibility STOP.

What changes, and only this: the FEASIBILITY criterion and the budget per goal, both set from the twelve
fresh curves the v61 gate measured; the gate runs on NEW burned worlds. The substrate, the code, the
primary, the contemporaneous null, the verdict rule, the wording rule, the probe rule, the seed table
roles and the test windows are the v61 ones, untouched.

### 12.1 Protocol
    store       PERSISTS across the goals of a world for every arm (the v61 notch protocol). Chosen for
                feasibility: under it 2 of 3 fresh arms reached 0.6 on tier 3 within 4 rounds on BOTH
                v61 units (per-goal store: on one). The cost was measured before this choice: tier-3
                law headroom 0.65 instead of 0.72, tier 4 unchanged (0.99), sweeper's tier-3 clock 10x
                faster, learner's ~unchanged. Symmetric across arms; the second panel's verdict stands:
                it dilutes, it does not bias.
    B_max       6 rounds per goal at the gate. B_max(test) = 6 if the confirmatory fits the ceiling at
                the freeze (12.4), else 5. Pretraining keeps B = 3 (fairness binds at test only).
    unit/stream unchanged (tier 2, 3, 4 lowest-index admitted items; fixed budget; per-world buffers)

### 12.2 The gate (K0 + K1 on the first two admitted seeds >= 8102: 8102, 8103; burned)
    REACHABLE     unchanged
    CONSISTENT    the CPU STREAM model (scripts/sweeper_v61.py simulate_stream: persisting store, full
                  budget per goal, first-obtain scored), VALIDATED on the v61 notch run before this text
                  was written: predicted 33 / 9 / 768 and 29 / 24 / 352 vs measured 33 / 9 / 768 and
                  28 / 25 / 403. Predictions at cap 1152: 8102 = 57 / 7 / >1152 (100% censored);
                  8103 = 23 / 29 / >1152 (92%). A miss by > 2x on a non-censored goal halts K1.
    FRESH-LEARNS  iff ALL 3 fresh arms reach 0.6 on tier 2 within 2 rounds on both units, AND at least
                  2 OF 3 fresh arms reach 0.6 on tier 3 within B_max on both units.
                  A fraction over arms, not a 3-arm median: v61's median rule let one stalled init
                  decide a unit. Tier 4 printed, not required (the reach position).
    ROOM          unchanged: H >= 4 se_proj, sd_init per unit, se_res, under B_test and N = 9.
    PROCEED iff all three. NO softening notch exists in v62: a FRESH-LEARNS failure ends the ARC 3
    line for good, published as the gate result. The one hardening notch (not ROOM -> quota 3, new
    gate worlds) is kept; ROOM was 0.55-0.57 vs 0.11-0.12 in v61, so it is not expected to fire.

### 12.3 Predictions, stated before the gate
From the twelve v61 curves: half the fresh inits reach 0.6 on tier 3 by round 2-4; of the stalled
half, those rising at round 4 (0.31-0.33) plausibly cross by round 6, the flat ones (0.03-0.06) do not.
Expected: 2 of 3 arms on each unit with probability ~0.6-0.7, so P(FRESH-LEARNS on both) ~0.4-0.5.
This gate is a coin flip too, and it is the last one. Tier 4: 0-1 of 6 fresh arms. H ~0.5.

### 12.4 Budget (126 s/round measured; 13.5 min per goal-run at B 6, 8.5 at B 3)
    v61 spent 6.7 | v62 gate K0 ~1.5 + K1 ~4.0 | pretraining 3 x 4 x 4 x 8.5 min = 6.8 | probe 0.5 |
    confirmatory N 9 x 3 goals x 3 arms x 13.5 min = 18.2 + G' on 2 units 1.4      total ~39.1 GPU-h
    Ceiling for the whole ARC 3 line, owner-approved with option B: 42 GPU-h. Cut list at the freeze,
    in order: (1) B_max(test) 5 (confirmatory 15.2 h); (2) G' to 1 unit. Never cut: Fb, a lineage,
    the probe, N = 9.

### 12.5 What is said about this everywhere it is reported
"v62 is a redesign after a feasibility STOP: its feasibility criterion was set from the twelve curves
that failed v61's, and it was gated on worlds those curves never touched. The claim's verdict rule
was frozen before any of it and did not move."
