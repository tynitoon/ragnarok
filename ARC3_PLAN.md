# ARC 3 — "ALCHEMY" (v61): a family of worlds that share hidden laws

DRAFT until the STAGE-0 freeze (section 4.0). Written 2026-09-06 after a 20-agent design panel (4 designs,
3 judges, 3 adversarial auditors, 2 revisions; transcript under the session's workflow directory). The
panel's final draft was still refuted 3/3 and had grown to 68k characters with nine open questions; this
document keeps its substrate, which every lens rated top-2 and no auditor attacked, and cuts its
measurement apparatus down to what survived the audits. Complexity in the measurement is where five ARC
1-2 verdicts died; it is treated here as a defect in its own right.

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
warns about).

## 2. The substrate

### 2.1 Laws — sampled ONCE (laws seed 31337), shared by every world, never observed

    K = 14 elements, T_MAX = 4 tiers. An ITEM is (element, tier). Raws are tier 1.
    Bond[K,K]   symmetric bool, diagonal False, density 0.5 off-diagonal, rejection-sampled until every
                element bonds with >= 2 partners.
    Out[K,K]    symmetric element index, defined where Bond holds, sampled as a PROPER EDGE COLOURING of
                the bond graph (for every element a the products Out[a,b] over its partners b are all
                distinct) — so no two recipes of one element collide.
    TIER law    combine(x, y) is LAWFUL iff tier[x] == tier[y] = t < T_MAX and Bond[el x, el y].
                A lawful combine yields the item (Out[el x, el y], t+1).
    OUTCOME     product : lawful and the product item exists in this world -> inputs consumed, +1 product
                inert   : lawful and the product does not exist here     -> inputs returned, nothing made
                botch   : not lawful                                     -> BOTH inputs destroyed
    GATE law    a raw is FREE or GATED; a gated raw can be collected only while holding some item of
                tier >= 2 (any element).
    QUOTA       each raw type yields at most 4 units per 48-step episode (counter in the env; the grid is
                never touched, so the frozen nav skill sees exactly the grids it was calibrated on).

Why this is expensive without the law: the only way to learn a Bond/Out cell is to hold two items of the
right elements and tier and PROPOSE them. A botch at tier >= 2 destroys inputs that cost 4 / 8 / 16 raw
units to rebuild; a botch at tier 1 burns the episode's quota. 91 cells; a world exposes ~37% of them.
Why three outcomes: a Bond-true pair whose product is absent in this world must not teach "this cell
fails", or the cross-world table gets poisoned. INERT keeps Bond exact.

### 2.2 Skin — per world seed

    (a) raws: 6 elements drawn from K without replacement -> cells 1..6 (nav skill range 1..9; n_cells 7).
        Four raws FREE, two GATED (assigned before the permutation).
    (b) roster: for t = 2..4, the products the law admits from the items present at t-1; then FIXED SLOT
        COUNT 6 per tier (random drops logged; empty slots filled with DECOYS = elements not producible
        at that tier, never admitted as goals, inputs = {}). n_items = 24 always, so counts leak nothing.
    (c) item indices permuted (permute_spec rule); cell IDs permuted within the used set (v58 rule);
        grid positions re-rolled every episode.
    Stable across worlds: the (element, tier) of an item and the law. Not stable: index, cell, position,
    existence, partners, which cells the world exercises.

Panel's Monte Carlo on its own implementation (200 seeds; RE-MEASURED by test (9) on ours): real items
per tier 6.0 / 5.7 / 5.2 / 4.6; ~2.6 products dropped per world; P(>= 1 admitted tier-4 goal) 0.92;
bonding-cell coverage of a random 4-world lineage 0.37 / 0.59 / 0.74 / 0.83 cumulative.

### 2.3 Generator and env — ragnarok/environments/law_world.py

    sample_laws(seed, K=14, density=0.5) -> dict(K, T_MAX, Bond, Out)
    gen_law_world(seed, laws, n_raw=6, n_free=4, slots=6) -> spec dict with the v58 fields (n_items,
        n_cells, kind, cell, ...) plus el[i], tier[i], gate[i], decoy[i], and EXPERIMENTER-SIDE tensors
        (execution and scoring only, never observed): pair_out[24,24] (item | INERT | BOTCH), chain
        cost per item (cheapest chain, raw units by type), true_pre.
    permute_law_spec(spec, seed): every tensor permuted consistently (test (5) asserts equivariance).
    ADMISSION of a world (mechanical, logged, never inspected): nav gate >= 0.85 per cell type; >= 1
        tier-2 product from the free raws; >= 1 admitted goal at each tier 2, 3, 4. An ADMITTED GOAL is a
        real (non-decoy) item whose cheapest chain fits the quota (max raw units of any type <= 4) and
        which the scripted law-knower L executes from an empty inventory within 48 macro-steps in >= 0.85
        of 64 envs. A world that fails is skipped and the next seed in order is taken.
    LawVecTechTree(DeviceVecTechTree): craft action is a no-op; collect requires gate satisfied AND quota
        left, decrements the quota; combine(i, j) -> outcome in {PRODUCT, INERT, BOTCH} applied to the
        inventory. Egocentric view, wall sentinel, nav mode untouched.

### 2.4 Observation — nothing computed from pair_out, Bond, Out or gate tensors

    per slot i (28 slots, 24 valid), 32 floats in [0,1]:
      base 8    in_inv, unlocked, tried_ep, succ_last, is_goal, is_resource, is_valid, quota_frac
      attrs 16  element one-hot (14), tier/4, gate/2
      store 8   n_succ_log, n_fail_log, obtained_ever, since_succ_log, gate_certified_now,
                frac_minctx_held, known_recipe, needed
    per unordered pair i<j (378), 4 floats: pair_succ_log, pair_nonprod_log, pair_tried_ep,
                pair_yields_needed_unheld
    Row = 28*32 + 378*4 = 2408 bytes uint8 (single quantisation, as BufferV58).

DISCLOSED GRANTS, stated once: is_resource, is_valid, quota_frac (the agent's own counter); element
labels (the law's shared vocabulary; the first layer learns a 14-symbol table and this plan says so);
tier; gate flag; and the ROSTER (which (element, tier) items exist here is a consequence of the law; the
count is fixed, the element SET per tier carries some information, and test (14) measures how much).
None of them says "will this combine succeed". Identity-freeness is claimed over ITEM SLOTS only.

### 2.5 Store — scripts/pair_store_v61.py, PairStore, one per parallel env, never pooled

    write_pair(i, j, outcome): PRODUCT -> pair_succ += 1, pair_out[i,j] = product slot; INERT ->
        pair_inert += 1; BOTCH -> pair_fail += 1. Exact: deterministic law, both inputs held, no nav
        skill in the path, so failure-elimination is SOUND for pairs (unlike raws).
    write_raw(g, held_before, obtained): the EvidenceStore rule for raws (failures only count — nav
        timeouts and quota refusals must never eliminate a true gate); min_ctx on success;
        gate_certified_now = n_succ[g] > 0 and held ⊇ min_ctx[g].
    derived: known_recipe[k] = any(pair_out == k); needed = closure from the goal over KNOWN recipes;
        pair_yields_needed_unheld[i,j]. state_dict / load_state_dict; eval envs get a COPY (test (13)).
    Lifetime: EMPTY at world entry for every arm; persists across the goals of a world (2.8); never
        carried between worlds. What crosses worlds is weights only.

### 2.6 Policy — scripts/pair_net_v61.py

PairNet scores an ACTION MATRIX: diagonal (i,i) = "pursue raw i" (valid iff is_resource & valid &
quota_frac > 0), off-diagonal i<j = "combine(i,j)" (valid iff both held & valid); 28 + 378 = 406
actions. Input per cell [x_i(32) || x_j(32) || mean over valid slots (32) || goal attrs (16) || pair
feats (4) || is_diag (1)] = 117 -> 128 -> 128 -> heads: policy logit; OUTCOME head 3-way
(botch / inert / product) + product element (14-way). No nn.Embedding over slots, no parameter shaped
to MAX_ITEMS, permutation-equivariant over slots (test (5), (6)).

Act: stochastic = softmax over valid cells + 5% uniform over valid. Deterministic (eval) = argmax under
the context-aware mask: a diagonal slot is masked iff (tried_ep & ~succ_last & ~ctx_new) or quota_frac
== 0; a pair is masked iff pair_fail > 0 or pair_inert > 0 or (tried this episode and not productive
this episode). Identical for every arm; L must reach >= 0.85 with the mask on (test (10)).

Training per round, identical for every arm: 300 steps, bs 512, Adam 3e-4 constructed fresh at every
world entry;
    loss = CE(policy, relabelled commanded action)   relabel_commanded_v58 VERBATIM: one commanded goal
                                                     per episode, credit only under it, 0.7^lag (D4)
         + CE(outcome 3-way, observed) + CE(product element | product)   on an OUTCOME buffer of every
                                                     training combine's (obs at decision, pair, outcome)
RULING on the outcome head (the panel's open question 1): it is supervised learning of what the agent
OBSERVED — the same bit the store writes — with no goal credit in it. Constraint 2 governs the policy's
credit and is untouched. Every arm has it. It is the place the shared law can live in weights; without
it the table would have to be learned implicitly through policy imitation, which is the weak signal ARC
2 measured. Evidence dropout: none (p = 0).
Initialisation: every fresh net is constructed under a PER-UNIT torch seed (4.7), so init variance
enters the contemporaneous null by construction (the panel's r1 fatal). M loads lineage weights; only
its sampling stream is per-unit.

### 2.7 Hand-coded references (ceilings, never arms of the claim)

    L   the law-knower: attributes + the TRUE tables, experimenter-side; a gather-then-craft planner
        (expand the known tree against the multiset held; if a gated unit is missing and no tier >= 2
        item is held, build the cheapest tool from free raws; while holding it collect every gated unit;
        then the free units; then combine bottom-up). The ceiling WITH the law.
    G'  the store-only sweeper: needed-closure over KNOWN recipes; certified chain first; else the
        cheapest-to-rebuild untried equal-tier pair among producible items; collects to refill; retries
        a gated raw after the max held tier rises; no law access. The ceiling WITHOUT the law.
Both wrapped as composers with act() and a no-op train, scored by the same run_goal on the same
criterion (the ARC 2 section-14 lesson: never score a reference on a different criterion).

### 2.8 The unit: a world and its 3-goal stream

A UNIT is one admitted world, entered by every arm with the store, buffers and optimizer EMPTY. Inside
it, the same CHAIN-BLIND stream for every arm:

    goals = [ admitted goal of tier 2, of tier 3, of tier 4 ], each the admitted real item of that tier
            with the LOWEST permuted index (the index is a per-world permutation, so this is a coin flip
            the experimenter never touches — never "the chain of the tier-4 goal", which would be a
            curriculum computed from the true recipes)
    each goal: B_max rounds at FIXED budget, no early break; eval before the first round and after
            every round -> master_per_round[0..B_max]; the store persists to the next goal of the stream
    credit: relabel_commanded_v58 verbatim on the commanded goal of each episode

Why a stream and not one tier-4 goal: an untrained net is uniform over valid actions and reaches a tier-4
item in 0 of 9,600 simulated episodes (panel r2); the fresh arm needs rungs to climb. A tier-ordered
stream IS the climb, evaluated at every tier, with the same budget per tier for every arm, and it is what
the north-star figure shows ("goals mastered vs practice, by difficulty"). Why not the panel's chain
ladder: it commands the intermediates of the goal's cheapest chain, computed from the true tables — a
curriculum grant the auditors could not clear. The stream uses tier (observable) and a random index.
Why the fresh arm's tier-4 goal is still costly: its chain may need tier-2/3 items that are NOT the
commanded tier-2/3 goals; discovering them is the sweep the experienced arm skips.

## 3. What transfers, the mechanism, and the predictions

What M carries (weights only): (1) an outcome model over element PAIRS — the Bond/Out cells it exercised
in earlier worlds, plus the tier law — i.e. a partially observed, memorised LAW TABLE keyed by the
disclosed element vocabulary; (2) a policy that reads it; (3) the generic evidence-reading and
exploration skills of ARC 2. Mechanism on entry to a new world: M maps the roster to (element, tier)
from the attributes, proposes pairs its table says are lawful, and skips the sweep; the fresh arm sweeps.
Accumulation: each world exposes ~37% of the cells; coverage after 1 / 2 / 3 / 4 worlds ~0.37 / 0.59 /
0.74 / 0.83, so M4 knows more cells of the test world than M1 — a rising curve has a MECHANISM here,
not a hope. No composition claim is made (Bond is tier-independent by construction).

Panel's learner Monte Carlo (its implementation, 16 units, perfect pooling — optimistic for the fresh
arm), curves m(b) for b = 0..5:
    UNIFORM fresh   0.00 0.06 0.13 0.10 0.18 0.19     lower bound on the fresh arm
    IDEAL fresh     0.00 0.63 0.88 0.88 0.88 0.88     upper bound (G'-style sweep + perfect pooling)
    M at c = 0.37   0.00 0.81 0.88 0.88 0.94 0.94     "M1"
    M at c = 0.59   0.17 0.94 0.94 0.94 1.00 1.00     "M2"
    M at c = 0.83   0.52 0.94 0.94 0.94 0.94 0.94     "M4"  (a perfect planner over its prior; a
                                                            per-pair scorer will do less)
Read honestly: against the IDEAL fresh arm, Delta over b = 1..3 is ~+0.14 at c = 0.83 and ~0 at c =
0.37; the real fresh arm lies between UNIFORM and IDEAL and only the gate can say where. Predictions,
stated before the gate: accumulation flat at M1, rising at M2 and M4; Delta_learn at M4 in the range
0.05-0.15 against a good fresh arm; INCONCLUSIVE is a live outcome and is accepted as one.

## 4. Measurement

### 4.0 STAGE-0 FREEZE
Frozen in the commit that runs gate_v61.py, BEFORE K0: K 14, density 0.5, T_MAX 4, quota 4, n_raw 6,
n_free 4, slots 6, laws seed 31337, the goal rule (2.8), the stream rule, cfg_v61 (= cfg_v58 values:
64 envs, macro_budget 48, 4 episodes per round, 300 train steps, thresh 0.6), B_max(gate) = 4, the seed
table (4.7), the notch list (4.1). Nothing about the substrate changes afterwards except by the one
notch, which is itself pre-declared here.

### 4.1 THE CHEAP GATE — gate worlds = the first two admitted seeds >= 8100 (burned)
K0 — hand-coded, ~0.6 GPU-h. Nav gate. L on every goal of both units, one deterministic eval each,
mask on. G' from an empty store on both units, B_max(gate) rounds of episodes, no training.
    REACHABLE   iff nav_min >= 0.85 AND m_L(g) >= 0.85 on every goal of both units. A REACHABLE failure
                is an INSTRUMENT defect (L already passed test (10) on this world): fix, log, re-run K0.
    CONSISTENT  (reported) G' attempts-to-first-obtain per goal beside test (9)'s prediction; a miss by
                > 2x halts K1 for a bug hunt.
K1 — learned, ~4.2 GPU-h. Both units, three fresh arms Fa, Fb, Fc (per-unit init seeds), B_max(gate) =
4 rounds per goal, identical grid/eval seeds across arms.
    Measured: curves m_a(g, b); b*(g) = first round at which the median fresh arm reaches 0.6 (inf if
              never); A_a(g) over b = 1..B; sd_init = sd of the 3 pairwise per-goal differences
              {A_Fa-A_Fb, A_Fa-A_Fc, A_Fb-A_Fc} (df printed); H = mean_g[A_L - mean(A_Fa,A_Fb,A_Fc)]
              (the headroom a perfect law transfer could buy); H' = mean_g[A_L - A_G'] (the law's worth
              against the ideal sweeper); outcome-head AUC of the fresh arms on the gate world's true
              Bond table after each round; wall-clock per round.
    FRESH-LEARNS  iff b*(tier-2 goal) <= 2 on both units AND b*(tier-4 goal) <= 4 on at least one
                  (a fresh agent CAN learn this substrate inside a budget the arc can pay)
    ROOM          iff H >= 4 * se_proj, se_proj = max(sd_init / sqrt(2N), se_res), N from 4.4
                  (DEMONSTRATED needs Delta >= 2 se; assuming M captures at most half the headroom, the
                  headroom must be 4 se. It excludes a substrate on which even a perfect law transfer
                  could not be detected at N; it cannot exclude a small real effect.)
PROCEED iff REACHABLE and FRESH-LEARNS and ROOM.
NOTCHES — at most ONE in the whole arc, then STOP:
    not FRESH-LEARNS  -> soften: B_max(test) = 5 and K1 extended by 2 rounds on the same units (+1.4 h);
                         PROCEED then requires b*(tier-4) <= 6 on one unit; else STOP.
    not ROOM          -> harden: quota 4 -> 3 (the panel measured this raises the ideal-sweep cost 372 ->
                         562 attempts without starving the uniform bootstrap: min P(tier-2 unlock per
                         episode) 0.040); regenerate the gate worlds (next admitted seeds), re-run K0+K1.
    not REACHABLE     -> instrument defect, not a notch.
STOP is published as a GATE RESULT with its numbers ("the substrate as built is too easy / too hard /
unresolvable at this N for this instrument"), never as a wall measurement. Cost of a STOP: <= ~5 GPU-h
(<= ~10 with a notch).
After PROCEED the prereg is appended to preregistration.md (v61) with B_max(test), N, the notch if any,
the seed table, and every K0/K1 number; then it is frozen.

### 4.2 Pretraining — three independent lineages s = 0, 1, 2
Lineage s lives in the first 4 admitted seeds of 8200+10s .. 8200+10s+9, in order. Per world: the
stream of 2.8 extended to 4 goals (tier 2, 3, and the two lowest-index admitted tier-4 goals), B = 3
rounds per goal, store empty at entry, buffers and optimizer rebuilt at entry, weights carried.
Checkpoints M1..M4 per lineage. Recorded for free: per-world learning-curve areas (the within-lineage
accumulation signal: world k's area given k-1 prior worlds), exercised-cell coverage per checkpoint.
Cost ~8.8 GPU-h. Pretraining rounds may exceed the test budget: fairness binds only at test.

### 4.3 Mechanism probe — world 8150 (burned), after pretraining, before the confirmatory, ~0.5 GPU-h
Frozen checkpoints M1..M4 of lineage 0 and a random-init R, weights frozen, store empty:
    (a) FIRST-PROPOSAL LAWFULNESS: over 256 episodes, the fraction of each arm's first combine per
        episode that is Bond-true. PROCEED iff r(M4) - r(R) >= 3 * sqrt(r(1-r)/256) pooled. If M4's
        first proposals are no more lawful than a random net's, the weights carry no law: STOP, and the
        finding is an architecture result ("the outcome head did not put the table into the policy"),
        fixed before any confirmatory compute, never after.
    (b) reported: outcome-head AUC on 8150's true Bond table, split into cells exercised in pretraining
        vs unexercised, beside the roster-only ceiling of test (14); r(M1..M4) as the accumulation of the
        law in the weights.

### 4.4 Confirmatory — per lineage s: the first N/3 admitted seeds of 8300+10s .. 8300+10s+9
N = 12 units (4 per lineage) at 220 s/round; N = 9 if the smoke measures > 260 s/round (declared at the
freeze, never after). Arms per unit, identical grid/eval seeds, store/buffers/optimizer empty at entry:
    M4    lineage s weights            Fa, Fb   fresh, per-unit init seeds
    L     one eval per goal (flat)     G'       lineage-0 units only (budget), no training
    B_max(test) = clamp(b*(tier-4, median over gate units) + 1, 3, 4), from K1.
PRIMARY (monotone; pointwise dominance can never score lower):
    A_a(u) = mean over the unit's 3 goals of the mean over b = 1..B_max of m_a(g, b)
             — b = 0 is EXCLUDED from the verdict statistic: at world entry the fresh arm's b = 0 is 0 by
             construction (empty store, untrained net), and an area that includes it manufactures
             "learns faster" out of "starts ahead" (the panel's r2 fatal)
    Delta_learn = mean over units of [ A_M4(u) - 0.5 (A_Fa(u) + A_Fb(u)) ]
    se_null     = sd over units of [ A_Fa(u) - A_Fb(u) ] / sqrt(2N)      (contemporaneous, includes init
                  variance by 2.6; never a transplanted number)
    se_res      = 0.0625 * sqrt(2 / (3 * B_max * N))                     (the 1/64 eval resolution floor)
    se          = max(se_null, se_res)
    DEMONSTRATED      iff Delta_learn >= 2 se AND Delta_s > 0 for each lineage s
    NOT-DEMONSTRATED  iff Delta_learn <= 0
    INCONCLUSIVE      otherwise (a likely honest outcome at a true Delta ~0.10; the rule is not widened)
    Instrument line: t_init = mean_u[A_Fa - A_Fb] / (sd/sqrt(N)); |t_init| >= 2 -> INCONCLUSIVE with the
    reason printed (the two fresh arms are exchangeable by construction; a systematic difference is a
    seeding or code defect). Under a valid null this fires in ~6% of runs; stated here.
REPORTED, never in the verdict: Delta_0 = mean_u[m_M4(g,0) - 0.5(m_Fa+m_Fb)(g,0)] over the tier-3/4 goals
(the ARC 2 frozen-transfer replicate on 12 units); per-tier curves; A_L and A_G' beside every arm; the
full 0..B_max area as a secondary.
MECHANISM label (caption only, never the verdict): "LAW-CARRYING" iff the probe passed AND the
outcome-head AUC on the test worlds' exercised cells >= 0.8; otherwise the caption says the advantage is
not attributed to the law.

### 4.5 Accumulation curve — DESCRIPTIVE, no verdict label
(i) within pretraining: per-world learning-curve area of world k given k-1 prior worlds, lineage by
lineage, with the world's L and G' areas beside it as difficulty context (free); (ii) the probe's
r(M1..M4) (free); (iii) lineage 0 only, first on the cut list: the fresh arm Fa on the four pretrain
worlds of lineage 0 (2.9 GPU-h), giving a paired M_{k-1}-vs-fresh point per world. Worlds-lived-in is
collinear with gradient steps and this is said in the caption; no "ACCUMULATION SUPPORTED" wording.

### 4.6 Budget (220 s/round; re-measured by the smoke, test (12), before the freeze)
    K0 0.6 | K1 4.2 | pretraining 8.8 | probe 0.5 | confirmatory 12 units x 3 goals x 3 arms x 11 min
    = 19.8 + G' on 4 units 1.6 | accumulation (iii) 2.9      total ~38 GPU-h
    CUT LIST, in order, applied at the freeze if the measured round time demands it: (1) accumulation
    (iii); (2) G' to 2 units; (3) N = 9. Never cut: Fb, a lineage, the probe. Hard ceiling 40 GPU-h.

### 4.7 Seeds and burned worlds
    laws 31337 | gate: first two admitted >= 8100 | probe 8150 | pretrain 8200-8229 by lineage |
    test 8300-8329 by lineage, never inspected before the run (admission is mechanical)
    grid seed per (world w, goal position p) = w + 11 p, eval seed +9, identical across arms
    init seed per (lineage s, world w, arm a) = 100000 + 1000 s + 10 (w mod 100) + a, a in {1,2,3};
    sampling stream per (unit, arm) = init seed + 500
    Burned: 8100-8199 (gate, probe) and any world a human looked at. 6100/6101 stay burned.

## 5. Implementation plan (ordered; nothing runs on GPU before step 6)

    1  ragnarok/environments/law_world.py — laws, generator, permutation, admission, LawVecTechTree.
    2  scripts/hand_v61.py — L (planner) and G' (sweeper) as composers.
    3  scripts/mc_v61.py — CPU Monte Carlo: generator table (2.2), admission lists for every named
       seed window, coverage per lineage, ideal-sweep attempts per admitted goal. Printed verbatim by
       test (9); these are the PREDICTIONS the gate is checked against.
    4  scripts/pair_store_v61.py — PairStore, PairEnv (option execution over the action matrix with the
       frozen nav skill; one unit per collect option; combine once), features.
    5  scripts/pair_net_v61.py — PairNet, ComposerV61, BufferV61 (policy + outcome), collect / relabel
       (v58 verbatim) / eval / run_goal_v61 / run_unit_v61, cfg_v61.
    6  scripts/test_law_v61.py — (1) unexercised-edit leak test on store and net input; (2) gate edit on
       an uncollected raw; (3) law-break; (4) counterfactual law with roster forced equal; (5) slot
       equivariance; (6) no MAX_ITEMS-shaped parameter; (7) uint8 round-trip; (8) hidden asserted;
       (9) generator invariants + the MC tables; (10) env unit tests and L from empty on 20 seeds;
       (11) nav gate on three law worlds, degradation < 0.10 vs v58; (12) timed smoke of one round;
       (13) eval isolation (buffers and store bit-identical after an eval); (14) roster-only ceiling:
       a classifier from (test roster, raw set, exercised table at coverage c) to unexercised Bond
       cells, trained on 2,000 sampled law tables, held-out AUC per c.
    7  scripts/gate_v61.py (K0 + K1, JSON + log), STAGE-0 FREEZE commit, run.
    8  preregistration.md v61 entry (frozen after PROCEED); scripts/score_v61.py written and frozen
       BEFORE pretraining, with a synthetic self-test that a pointwise-dominating arm never scores lower.
    9  scripts/pretrain_v61.py, probe_v61.py, confirm_v61.py (per-unit JSON checkpoints, --resume,
       detached launch, _smoke suffix excluded by the scorer).
    Dev estimate: 3-4 sessions at this project's measured velocity. GPU: section 4.6.

## 6. Risks and the observation that reveals each early

    R1 fresh arm never climbs past tier 2 (uniform bootstrap fails at tier 3)       K1 b*(tier-3) = inf
    R2 fresh arm sweeps the whole table in one round (too easy; ARC 2 again)         K1 H ~ 0, ROOM fails
    R3 the outcome head learns the table but the policy ignores it                  probe (a) fails while
                                                                                    (b) AUC is high
    R4 roster grant predicts unexercised cells well (M's edge is the roster, not     test (14) AUC high
       memory)                                                                       -> caption says so
    R5 goal-3 chain needs pairs neither arm was commanded (both arms stall at tier   K0: L reaches it,
       4; no headroom)                                                               G' does not, in a
                                                                                    budget of B rounds
    R6 round time balloons with the 406-action matrix                               smoke (12)
    R7 lineage coverage far below 0.37/world (4 goals exercise few cells)            pretrain coverage
                                                                                    log; M4 c < 0.6
    R8 unit-level noise far above v60 (3 goals per unit are correlated)              K1 sd_init; ROOM
    R9 M4 loses its policy in round 1 of a new world (drift on a sharp checkpoint)   M4 curve shape
                                                                                    reported per round

## 7. What is NOT claimed

Not composition; not "the agent understands chemistry"; not sim-to-real; not that the agent finds every
law (it exposes ~37% of the table per world and that is the point). The transferred object is a
partially observed memorised law table keyed by disclosed labels, plus tier/gate laws, plus exploration
skill — and this is stated on the figure. The store is per-world by design: nothing in it crosses
worlds, and a positive result is about the WEIGHTS. A negative or inconclusive result is published with
its numbers, as sections 8, 11 and 15 of ARC2_PLAN were.

## 8. The figure

One panel per tier (2, 3, 4): goals mastered (fraction of units) vs practice rounds b = 0..B_max, for
M4, fresh (mean of Fa/Fb with their spread as the band), L (dashed ceiling), G' (dotted). Caption states
the verdict (DEMONSTRATED / NOT / INCONCLUSIVE), Delta_learn with se, Delta_0 separately, and the
mechanism label. A second, smaller panel: the accumulation curve of 4.5 with the collinearity caveat.
