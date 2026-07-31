# ARC 2 — EvidenceNet: change world, keep skills

**Handoff document.** Written 2026-07-30 for the implementing agent. The design was arbitrated by a
4-agent adversarial workflow; Step 1 is DONE and committed (`db9a8ba`). Execute Steps 2-6 IN ORDER.
A separate verification pass will audit the implementation against this document before any
confirmatory GPU is spent — deviations must be either justified in commit messages or absent.

## 0. Context you must not re-litigate

- ARC 1 (v49-v57, "is memory necessary at depth") is CLOSED: three frozen NULLs, honestly published.
  Its files are FROZEN — do not modify `hidden_recipe_v55.py`, `credit_fix_v57.py`, `run_v57.py`,
  `score_v57.py`, or any `v5x` result JSON. New code goes in new `*_v58.py` files.
- Four instrument defects were found across ARC 1 and must never recur:
  D1 oracle features (craftable_now computed from true recipes) — excluded by the leak test;
  D2 saturating metrics (cost hits 0 → learning invisible);
  D3 argmax absorbing state at eval (bit-identical observation after repeated failure);
  D4 hindsight credit leak (99.69% of gradient steps at depth were about non-commanded goals) —
     commanded-only credit (`relabel_commanded`) EVERYWHERE, including pretraining.
- Three consecutive verdicts died to symbolically predicted strata/thresholds (v54 band, v55
  blind-cost, v57 pc>=10). REPAIR, non-negotiable: no symbolic strata in any verdict gate; all
  thresholds are FIT ONCE from a calibration phase, then hard-frozen with a committed scorer BEFORE
  any confirmatory run.

## 1. The claim being built toward

> Meta-trained across 4 same-family hidden-recipe worlds, a frozen-weight agent that writes its own
> per-world evidence store discovers and masters a NEW world of that family X% cheaper
> (attempts-to-master, discovery included) than an identical fresh agent — same architecture, same
> store, same credit rule, equal in-world budget — beyond what generic exploration grammar (Mdeg)
> and memorized generator constants (param-shifted worlds) explain, at a disclosed pretraining cost.

NEVER claimed: enablement (F6 guards), cross-family/cross-domain transfer, symbolic DAG induction,
zero-shot mastery. Granted interface, disclosed: frozen nav skill, item->cell / item->craft-action /
is_resource / is_valid bindings, the one-of-each-input family invariant.

## 2. Architecture (Step 1 half already committed)

- `EvidenceStore` (`scripts/evidence_store_v58.py`, DONE): per-env slot-indexed fast memory; success
  contexts intersect candidate masks S (crafts) / C (resource gates) — exact under multiplicity-1;
  failures only count (the draft's failure-elimination was unsound under stochastic nav, see module
  docstring). Persists across truncations; zeroed on world entry. 10 read features per slot, frozen
  list in `EvidenceStore.features`. Tests in `scripts/test_evidence_v58.py` (leak / equivariance /
  persistence) MUST keep passing.
- `EvidenceNet` (Step 2, to build): PerItemRouter skeleton (`meta_manager_v51.py:39-60`) with **NO
  nn.Embedding** — shared per-item MLP (hidden 128, ~50-100k params) over
  `[7 base feats || 10 evidence feats || mean-pooled global context]`, masked per-slot logits.
  ONE pre-registered fallback if the K1 gate fails: a single attention round over slots. Nothing else.
- Training loop: the v57 pipeline verbatim — hindsight CE over a uint8 buffer, `relabel_commanded`
  (gamma 0.7), INSTR_MASK at eval, per-world buffers, per-env stores. The store snapshot at decision
  time is PART of the stored observation (buffer rows widen 28x7 -> 28x17 uint8; evidence features are
  in [0,1] — quantize to uint8 levels {0..255} consistently at write AND read).
- At test on held-out worlds, arm M's weights are FROZEN — zero gradient updates; all adaptation is
  store writes.

## 3. Environment hardening (Step 2, before the gate)

- Extend `permute_spec` (new v58 wrapper, do not edit the frozen v55 file) to ALSO permute cell IDs
  into 1..MAX_CELLS-2 (closes the depth-order shortcut: gen_tree assigns cells in creation order).
- macro_budget 26 -> **48** for all arms (probe: 72/75 mid-discovery failures at 26 recover by 60).
- Per-env stores across the 256 parallel envs (a POOLED store would be a 256x discovery artifact).
- Per-world buffers only (buffer rows are slot-indexed, hence world-locked).
- Declare in the prereg: item->cell, item->craft-action, is_resource, is_valid are action-interface
  grants read from the spec, equalized across all arms.

## 4. Arms

| arm | what | role |
|---|---|---|
| M | EvidenceNet pretrained on 4 worlds (seeds 4000-4003), weights frozen at test | treatment |
| F | identical arch INCLUDING identical store, random init, trained in-world, equal env budget | primary control |
| Mdeg | pretrained at matched volume on degenerate family n_items=8/max_inputs=1 | cheap-transfer isolator |
| Z | arm M with store zeroed at every eval step (eval-only) | store-ignoring detector |
| G | hand-coded `evidence_policy`, no learning (CPU) | headroom reference |
| F6 | fresh agent, 6x r_max, on <=2 cells where F failed | enablement guard |
| D | oracle arm (hidden=False), ceiling reference only | never in a meta arm |

Worlds: pretrain 4000-4003; calibration 5000-5001 (F only); held-out test 6000-6002 (primary) +
7000-7001 param-shifted (n_items=20; p_resource=0.15/p_tool=0.8) (secondary). All disjoint from skill
seeds 1000-1007 and ARC-1 worlds 3002/3003/3016. Both arms share the per-seed frozen nav skill
(`craft_v6_out/v55_skill_s{seed}.pt`) with the per-world nav gate; 3 run seeds.

## 5. Steps, gates, kills

- **Step 2 (CPU)**: EvidenceNet + hardening above + widened buffer. Unit test: forward pass
  permutation-equivariance of the net (permute slots+features -> logits permute).
- **Step 3 — GATE K1 (<=1 GPU-h)**: train F on ONE world under commanded-only credit. PASS = shallow
  mastery ~ MemoryNet level AND store-zeroed eval drop >=30% (the net demonstrably READS the store).
  FAIL -> one shot for the attention fallback, one more 1 GPU-h gate; if that fails too, ARC 2 ends
  before any confirmatory spend.
- **Step 4 — CALIBRATION (~4-6 GPU-h)**: F alone on worlds 5000-5001. Fit P1/P2 thresholds from
  measured variance. KILL K2: if F's stream cost varies >2x between the two calibration worlds, stop
  and redesign the metric — do not proceed on pencil numbers. Then FREEZE the prereg + write
  `scripts/score_v58.py` and COMMIT BOTH before any confirmatory run.
- **Step 5 — CONFIRMATORY (~15-20 GPU-h)**: pretrain M and Mdeg; evaluate M/F/Z on the 5 held-out
  worlds; G on CPU; F6 on <=2 cells. Detached runs + resume + per-goal JSON checkpoints (run_v57.py
  patterns). Pre-committed trim order at ONE halfway checkpoint if over budget: drop 4th pretrain
  world, then 1 param-shifted world, then r_max 4->3. Hard ceiling 30 GPU-h.
- **Step 6**: verdict from the frozen scorer; amortization ledger (break-even worlds =
  C_pretrain / per-world saving); writeup committed whatever the result.

Pencil predictions (to be re-fit ONCE in Step 4, then frozen): P1 pooled M/F cost ratio <= 0.65;
P2 paired attempts-to-first-demo, M wins >=60% of non-tied goals and wins >= 2x losses; P3 saving
(F-M) >= 2x (F-Mdeg); P4 Z/M >= 1.5; P5 param-shifted M/F <= 0.80; P6 M worse than F on <= 25% of
paired goals. SUPPORTED iff P1 & P2 & P3 & P4. REFUTED iff pooled M/F >= 0.85 OR wins < 1.5x losses.
Downgrade labels are pre-specified in the design record — use them verbatim.
KILLS: K1/K2 above; K3 = REFUTED per frozen rule -> published NULL, arc closed; K4 = M ~= Z ->
"store-ignoring collapse", published regardless of P1; K5 = 30 GPU-h ceiling.

## 6. What the verification pass will check (implementer: self-audit against this list)

1. Frozen ARC-1 files untouched; all new code in `*_v58.py`.
2. No identity embeddings anywhere in the policy; no oracle features; leak test still passes and now
   also covers the net's INPUT (edit unobserved recipe -> net input bit-identical).
3. Commanded-only credit in EVERY training call, including M/Mdeg pretraining.
4. Per-env stores, per-world buffers, store persists across truncation, zeroed on world entry.
5. uint8 quantization of evidence features is consistent between buffer write and net read.
6. Cell-ID permutation actually applied in every world construction (pretrain, calibration, test).
7. Equal in-world env budgets M vs F, per test world; eval cost counted; INSTR_MASK at every eval.
8. Thresholds in the committed scorer match the calibration fit; no number changed after Step 4.
9. Gate/calibration/confirmatory artifacts saved as JSON with resume support; runs detached.
10. The seven arms match the table; D never contaminates a meta arm; Z is eval-only.

---

## 7. HANDOFF — status at the end of Step 4 (2026-07-31, Opus 5)

Steps 2, 3 and 4 are DONE and committed. **No confirmatory GPU has been spent.** Step 5 is deliberately
NOT started: it is gated on the verification pass, as agreed.

| step | state | commit |
|---|---|---|
| 1 EvidenceStore + tests + demo | done | `db9a8ba` |
| 2 EvidenceNet + hardening + tests | done | `2be71e8` |
| 3 GATE K1 | **PASSED** | `5f69ad3` |
| 4 calibration, thresholds fitted, prereg frozen | done | `d215dae` |

### Results
GATE K1 (world 4000): mastery 4/4, including pc 11 and 12 in ONE round — the depths at which ARC 1's
amnesic arm starved outright. Zeroing the evidence half of the observation drops the SAME trained weights
from ~0.98 to ~0.00 (relative drop 0.99): the policy is almost nothing without its store, which is the
separation portability requires.
CALIBRATION (arm F only, worlds 5000/5001 x 3 inits, 9/9 mastered everywhere): stream costs
[3072,3840,2880] and [2496,2688,2880]; between-world median ratio 1.14 so **K2 did not fire**; null F/F
ratio min 0.750 / median 1.002 / max 1.333. Headroom: F sits at 1.78x and 1.56x the metric's hard floor,
so the best M/F physically reachable is 0.562-0.643 — P1 is demanding but NOT saturated.
FROZEN in `scripts/score_v58.py`: `P1_MAX_RATIO = 0.712`, `REFUTE_RATIO = 0.750`. Measured, not chosen.

### THE TWO DECISIONS LEFT TO THE VERIFIER

**(a) P2 is BLOCKED and must be ruled on.** Its specified statistic (attempts-to-first-demo) had ZERO
resolution: all 54 calibration values were exactly 48, one episode, because with 256 parallel envs some
env always reaches the goal inside the first episode. Freezing it would have put a permanently-false
conjunct inside SUPPORTED, making success unreachable by construction — the D2 defect class, in the very
prediction meant to be P1's fine-resolution backup. It was NOT silently redefined: picking a statistic
after seeing calibration data is exactly what cost v54, v55 and v57 their verdicts.
Evidence already gathered so the ruling needs no new GPU:
  - candidate "round-1 demo count" (already recorded): pooled null win rates over 18 goals are
    0.846/0.667/0.154/0.467/0.333/0.533 -> any valid threshold must exceed ~0.90; and 16/54 goals (30%)
    sit at the 1024/1024 ceiling at first exposure, i.e. tied by saturation.
  - INSTRUMENTATION HAS BEEN FIXED (logging only, no statistic chosen): every goal now records a
    `discovery` dict — fraction of envs reaching the goal at first exposure (resolution 1/num_envs) and
    the min/median within-episode macro-step — and `first_demo_attempt` now has within-episode
    resolution. Verified on a live smoke: values 24 and 26 (not the old constant 48), fracs 0.44 and
    0.63, median steps 32 and 37. A future calibration can fit a threshold on these without the ceiling.
  - NOTE for the ruling: with 256 envs `min_step` will itself saturate low (some env gets lucky early);
    `median_step` and `frac` are the ones that keep resolution.
The scorer reports P1/P3/P4/P5/P6 and can return REFUTED, but REFUSES to return SUPPORTED while
`P2_MIN_WINRATE is None` — verified by running it.

**(b) Is the experiment adequately powered?** Disclosed before the run, not after: arm F — a FRESH agent
— already masters 9/9 goals on both calibration worlds and reaches the commanded goal in 700-1024 of 1024
env-episodes at FIRST exposure, on goals up to pc 16. There may be too little headroom for a genuine
cross-world transfer effect to clear P1. If the verifier judges it underpowered, the fixes to weigh are
harder held-out worlds, fewer parallel envs (so discovery is not nearly free), or restructuring the
primary around discovery quality rather than cost.

### One documented deviation, measured not asserted
Section 3 of this plan says permute cell IDs into 1..MAX_CELLS-2. The frozen childhood nav skill has only
ever seen cell IDs 1..9 (measured across its 8 training trees) and consumes the target cell type as a
one-hot, so the literal instruction would feed it positions it never trained on and collapse navigation
for EVERY arm, treatment and controls alike. `permute_spec_v58` permutes WITHIN each world's used set,
which still decorrelates cell ID from gen_tree's creation order — the stated purpose. `test_net_v58.py`
test [5] measures both ways on three worlds: max degradation +0.032, all still clearing the 0.85 nav
gate. The composer never observes cell IDs at all, so the shortcut was dormant at policy level anyway.
The verifier should accept or reject this explicitly.

### Self-audit against section 6
1 frozen ARC-1 files untouched (all new code in `*_v58.py`) — OK. 2 no embeddings, no MAX_ITEMS-shaped
parameter, leak test extended to the NET INPUT — OK, tested. 3 commanded-only credit in every training
path — OK (`relabel_commanded_v58` is the only relabel used). 4 per-env stores, per-world buffers, store
persists across truncation and resets on world entry — OK, tested. 5 uint8 quantisation identical on
write and read, binary half exact — OK, tested. 6 `permute_spec_v58` used in gate and calibration; the
Step-5 runner does not exist yet and MUST use it. 7 equal budgets / eval cost counted / INSTR_MASK at
every eval — implemented in `run_goal_v58` and `ComposerV58.act`; the M-vs-F budget equality is a Step-5
runner property and is NOT yet enforced in code. 8 scorer constants match the calibration JSON — OK.
9 JSON artifacts + resume — calibration has both; the Step-5 runner will need them. 10 arms — Z is
eval-only via `zero_store`; G exists as `evidence_policy`; M/Mdeg/F6/D are Step-5 work, not yet written.

**Remaining to build in Step 5:** the confirmatory runner itself (pretraining M and Mdeg, the held-out
evaluation with frozen weights, arms Z/G/F6/D, detached execution with resume, per-goal JSON). Nothing of
it exists yet, by design.

---

## 8. VERIFICATION VERDICT: **NO-GO — REDESIGN REQUIRED**. The frozen v58 prereg is VOID. Do not run it.
(3-agent adversarial audit, 2026-07-31; the fatal item independently re-derived and CONFIRMED below.)

### FATAL — censoring is asymmetric and lands only on the treatment
Arm M is FROZEN at test (zero gradient steps) while F trains in-world. A goal M's frozen weights cannot
crack burns the full censored budget (r_max x 192), which exceeds M's ENTIRE above-floor allowance.
Re-derived here on the exact held-out set (worlds 6000/6001/6002 admit 10/8/9 = 27 goals; floor 5184;
F ~ 8928 from the calibration's 331 attempts/goal):
    M masters 27/27 in one round each  -> M/F 0.581  -> P1 PASSES
    M masters 26/27, censored on 1     -> M/F 0.774  -> **published REFUTED**
    M masters 25/27, censored on 2     -> M/F 0.968  -> REFUTED
So a frozen-weight agent that mastered 26 of 27 unseen goals in a SINGLE round each — an extraordinary
result — would be published verbatim as "indistinguishable from another fresh run. Arc closed."
**P1 as frozen is not a cost test; it is "M must be flawless on 27/27", and the arithmetic already
predicts a fourth frozen NULL.**

### MAJOR — the threshold was fitted on the wrong unit
P1_MAX_RATIO 0.712 = 0.95 x the SINGLE-world null minimum (0.750), but score_v58.py applies it to a
3-world POOLED ratio. Pooling averages out world noise, so the pooled null is materially tighter
(audit's recomputation: pooled min 0.800-0.853). REFUTE_RATIO 0.750 therefore sits BELOW the pooled
null's own minimum: a genuine 20% saving (M/F 0.800), clearly outside the null, would be published as
REFUTED. With only 2 calibration worlds the 3-world pooled null cannot be estimated at all.

### MAJOR — half the cost metric is dead weight
26 of 54 calibration goal-runs had zero_shot >= thresh on arrival and STILL paid a mandatory full round
(run_goal_v58 computes zero_shot then enters the loop unconditionally). On those goals M cannot win by
any amount. NOTE: the mandatory round is LOAD-BEARING and must not be naively deleted — v53/v54 returned
cost 0 on zero-shot success, the buffer stopped growing, and compounding became unobservable
(hidden_recipe_v55.py documents this). The repair must be elsewhere.

### MAJOR — my own discipline slip, caught by the audit
The first-demo statistic was redefined INSIDE the freeze commit (`git diff 5f69ad3 d215dae --
scripts/evidence_net_v58.py`). Calibration measured the episode-granular version; the committed code
emits a within-episode statistic whose null has NEVER been measured. It was documented as "logging only",
and while P2 is None it cannot reach the verdict path — but it silently re-arms the moment anyone sets
P2_MIN_WINRATE. The artifact's null and the code's statistic no longer refer to the same quantity.

### The P2 block was LOAD-BEARING, and blocking it was correct
Had the calibration's fitted P2_MIN_WINRATE = 0.6 been copied as calibrate_v58.py instructed, the scorer
would have emitted REFUTED on EVERY possible outcome (all goals tie -> wins = losses = 0 -> win/loss 0.0
< REFUTE_MIN_WIN_LOSS 1.5). That 0.6 was derived from calibrate_v58.py's zero-non-tied-goals SENTINEL
(`return wins/n if n else 0.5`), not from a distribution. It must never be copied.
RULING on P2: do not freeze it, and do not fit it from existing data. Re-measure in calibration v2 with a
CONTINUOUS paired statistic (mean paired log-ratio of round-0 demo counts, or paired difference in
`discovery.median_step` / `discovery.frac`) — a win rate is dead on arrival because its own null already
reached 0.846 and 26% of pairs tie from ceiling saturation. `min_step` must NOT be used (it saturates low
at 256 envs). Delete P2_MIN_WIN_LOSS and REFUTE_MIN_WIN_LOSS with it. Dropping P2 from SUPPORTED by
amendment was considered and REJECTED: it is P1's fine-resolution backup, and P1 is the conjunct being
ruled underpowered.

### Required redesign, in the audit's leverage order (~5-7 GPU-h for calibration v2)
1. **Reduce num_envs below 256.** Highest leverage: it attacks the root cause — a FRESH agent currently
   reaches the goal in 700-1024 of 1024 env-episodes at first exposure, so discovery is nearly free and
   there is almost nothing for transfer to buy — and it simultaneously fixes the P2 degeneracy (the
   min-over-256-envs trigger is why all 54 first-demo values were exactly 48).
2. **Cap per-goal censoring** (e.g. 3 rounds), or exclude censored goals from the cost sum and report
   "M failed k goals" as a separate PRE-REGISTERED line, so one frozen-arm miss cannot manufacture a
   REFUTED.
3. **Harder / deeper held-out worlds**, so difficulty tracks ability.
4. **Re-fit P1 on the POOLED unit** over 3 calibration worlds x 3 inits, and replace calibrate_v58.py's
   bare binary saturation guard (`p1_thr < best_reachable`, which passed by only 0.069 on a statistic
   quantised at 0.0215 per round) with an explicit margin requirement.

### Status
Steps 1-3 stand: the EvidenceStore, the identity-free EvidenceNet, and GATE K1 (mastery 4/4, store-zeroed
drop 0.99) are unaffected and remain valid. What is void is the Step-4 MEASUREMENT DESIGN — P1's metric,
its threshold's unit, and P2. Step 5 must not run until calibration v2 lands. No confirmatory GPU has
been spent, which is exactly what the gate-before-spend discipline exists to achieve.
