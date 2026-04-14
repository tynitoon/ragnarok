# Ragnarok — Research Preregistration (v3)

**Date committed:** 2026-04-14
**Revision:** v3 — addresses second-round stress-test feedback on v2 (power, mechanism, compute, pilot consistency)
**Target venue:** RLC 2026 short paper or NeurIPS 2026 workshop
**Target reviewer score:** 7.5–8.0/10 (workshop tier, not main track)
**Timeline:** 14–16 weeks from commit date (revised up from 11–12)
**Primary author:** Jérémie (solo dev), Claude Code in the loop
**Compute:** 1× RTX 4080; investor compute optional if pilot warrants

This document is a commitment. Changes to hypotheses, envs, baselines, or metrics
after a pilot run require a dated amendment in §13.

---

## 1. Research claim

> **H1-primary (confirmatory):** A shared RSSM latent policy trunk over
> `cat(h, z)`, combined with nearest-centroid skill retrieval, enables positive
> forward transfer from a discrete-action source to a continuous-action target
> with different observation dimensionality. On the **single primary
> endpoint** — MountainCarContinuous-v0 sourced from CartPole-v1 — at N=20
> seeds per arm, Ragnarok-transfer achieves shorter RMST for samples-to-mastery
> than the best-performing baseline at one-sided log-rank `p < 0.05`.
>
> **H1-secondary (descriptive):** Same mechanism, secondary target envs
> (Acrobot from CartPole; DMC-cartpole-swingup from Pendulum), N=10 per arm.
> Holm-Bonferroni across the two secondary envs; secondary results reported
> with 95% bootstrap CIs but do not gate the headline claim.
>
> **H2 (scaling, exploratory):** Sample efficiency on a held-out target task
> improves with skill-library size `k`. Reported as a descriptive log-linear
> regression with confidence band; **not** preregistered as a confirmatory test
> (only 5 k-points, underpowered). H2 is reframed as exploratory and will be
> labeled as such in the paper. A confirmatory restatement requires a separate
> preregistration with `k ∈ {1,2,3,5,8}` and N≥8 seeds per level (§7 A5b).

### 1.5 Novelty delta vs prior work

H1's load-bearing novelty is the conjunction of three properties simultaneously
in one architecture: (a) a shared RSSM trunk that operates on a *task-agnostic*
`cat(h,z)` representation, (b) nearest-centroid skill retrieval keyed on
encoded observations of the new task, and (c) demonstrated positive transfer
across a discrete→continuous action-space change. Closest prior work and how
H1 differs:

- **Continual-Dreamer (Kessler 2023):** lifelong RSSM transfer, but action
  space is held constant across tasks and there is no skill-retrieval layer.
- **Choreographer (Mendonca 2023):** discovers latent skills in dream space,
  but does not retrieve them across heterogeneous-action-space targets and
  trains skills in a single environment.
- **CoWorld (Wang 2024):** cooperative world models for transfer, but
  homogeneous action spaces.
- **SPiRL / OPAL (Pertsch 2020 / Ajay 2021):** skill-prior retrieval, but no
  Dreamer-style RSSM trunk and no cross-action-space transfer.
- **Hypernetwork policies (Rezaei-Shoshtari 2022):** address dim-mismatch via
  generated weights, not via shared latent features.

H1's claim is dead if literature review (week 2 — moved up from §11) finds a
paper that already conjoins all three of (a), (b), (c) on the same target env
class.

H1-primary alone is the workshop-tier contribution. H1-secondary and H2
support the narrative but are not load-bearing for acceptance.

## 2. Environments

Three target envs (was four — DMC cheetah-run moved to appendix per
compute-budget review). Two source envs unchanged.

| Role   | Env                          | Obs dim | Action space      | Notes                      |
|--------|------------------------------|--------:|-------------------|----------------------------|
| Source | CartPole-v1                  |       4 | Discrete(2)       | Dense reward               |
| Source | Pendulum-v1                  |       3 | Continuous(1)     | Dense negative cost        |
| Target | Acrobot-v1                   |       6 | Discrete(3)       | Sparse negative            |
| Target | MountainCarContinuous-v0     |       2 | Continuous(1)     | Sparse terminal            |
| Target | DMC cartpole-swingup         |       5 | Continuous(1)     | Same task, new engine      |
| Appdx  | DMC cheetah-run              |      17 | Continuous(6)     | Reported but not headline  |

Rationale for cut: cheetah-run alone consumes ~40% of estimated compute under
the v1 plan and adds a third heterogeneity axis (scale shift) that confounds
H1's discrete↔continuous narrative. Appendix-only reporting preserves honesty;
headline claim is over the 3 envs above.

DMC tasks require `dm_control` → Python 3.11 venv (§6.2). Walker-walk,
hopper-stand, finger-spin held in reserve for appendix; NOT headline.

## 3. Baselines

Cut from eight to five (per compute-budget review). Each baseline keeps the
target env's identical step budget (500 k env-steps) and seed schedule.
Hyperparameters from authors' published defaults; any tuning documented in
`benchmark_notes.md`.

| Baseline                              | Purpose                                       |
|---------------------------------------|-----------------------------------------------|
| SB3 PPO (discrete) / SAC (continuous) | Sample-efficiency floor                       |
| DreamerV3 from scratch                | World-model-from-scratch ceiling              |
| Continual-Dreamer (Kessler 2023)      | Lifelong world-model baseline (cross-dim ref) |
| Ragnarok from scratch                 | Our own from-scratch floor                    |
| **Ragnarok transfer (ours)**          | The headline                                  |

Controls (run as ablations in §7, not headline baselines): source-reward-shuffle
(A3), random-skill-retrieval (A7), equal-parameter Continual-Dreamer (A8).

Hypernetwork (Rezaei-Shoshtari 2022) and DreamerV3 + naive fine-tune are
**dropped from the headline table** to free compute. Hypernetwork remains in
the related-work discussion. If reviewers demand a hypernet number, it goes
into the rebuttal addendum, not the preregistered table.

If DreamerV3 official JAX repo cannot install on our stack within 3 days,
fall back to `NM512/dreamerv3-torch`. Document which in `benchmark_notes.md`.

## 4. Metrics

Per Agarwal et al. 2021 ("Deep Reinforcement Learning at the Edge of the
Statistical Precipice"), with censoring per Kaplan & Meier 1958.

1. **Headline (H1-primary):** Restricted Mean Survival Time (RMST) of
   "samples-to-mastery" on **MountainCarContinuous (primary endpoint only)**,
   computed via Kaplan-Meier estimator. Comparison: one-sided log-rank test
   at α=0.05 (no multiplicity correction needed — single primary endpoint).
   Truncation horizon τ = 500 k env-steps. Implementation:
   `lifelines.KaplanMeierFitter` and `lifelines.statistics.logrank_test`.
2. **Secondary (H1-secondary):** Same RMST procedure on Acrobot and
   DMC-cartpole-swingup, Holm-Bonferroni across the two secondary envs at
   α=0.05.
3. **IQM of normalized return** (interquartile mean across seeds×envs), 95%
   stratified bootstrap CI (10 000 resamples), performance profiles.
   Normalization: `(return - scratch_mean) / (max_return - scratch_mean)`.
   `max_return` per env is pinned in §4.5 — no analyst freedom.
4. **AULC (area under learning curve, normalized)**, secondary, with CIs.
5. **Probability of improvement:** `P(Ragnarok-transfer > best baseline)` via
   stratified bootstrap rank test across seeds, per env.

Mean-of-final-return is **not** a reported metric.

### 4.6 Censoring sensitivity

Right-censoring on MountainCarContinuous is expected to be high (40–60% of
runs may not reach 0.8×SB3-final within 500k env-steps; SAC routinely fails
to solve MCC on ~30–40% of seeds). To pre-empt the "your result depends on
τ" reviewer question:

- Primary RMST reported at τ = 500k (as preregistered).
- Sensitivity analysis: same RMST recomputed at τ ∈ {300k, 750k, 1M*}.
  Primary claim is robust **only if** the sign of the RMST difference and the
  qualitative log-rank significance hold across all four τ values.
- Per-env censoring rate reported alongside RMST.
- *(τ = 1M only if compute permits — gated on §12 budget.)*

### 4.5 Pinned analyst degrees of freedom

To eliminate post-hoc tuning of metric thresholds:

| Quantity                          | Value (frozen at commit time)                          |
|-----------------------------------|--------------------------------------------------------|
| Mastery threshold per env         | 80% of SB3 final return at 500 k steps, **median over 10 SB3 seeds, computed once before headline runs and committed to `thresholds.json`** |
| `max_return` per env              | Env's documented optimal (CartPole 500, Acrobot −80, MountainCarCont +95, DMC cartpole-swingup 1000)  |
| Truncation horizon τ              | 500 000 env-steps                                      |
| "Convergence" criterion           | Eval-return moving average over last 50 episodes within 5% of best-50-ep window of run; otherwise censored |
| Eval frequency                    | Every 5 000 env-steps, 10 deterministic eval episodes  |
| RNG seed schedule                 | `{42, 43, …, 42+N-1}` per (env, method)                |

These values are committed in `thresholds.json` alongside this document.

## 5. Statistical rigor

- **N=20 seeds** per arm (Ragnarok-transfer + best baseline) on the **primary
  endpoint** (MountainCarContinuous). Other baselines on the primary endpoint
  at N=10. Rationale: prospective power calc under exponential survival, HR =
  1.5, α=0.05 one-sided log-rank, censoring rate 40% → power ≈ 0.72; at N=10
  per arm, power collapses to ~0.40 (devil's advocate review, G0 round 2).
- **N=10 seeds** per (env, method) on the two secondary endpoints (Acrobot,
  DMC-cartpole-swingup). Holm-Bonferroni across the two secondary envs.
- **N=5 seeds** for ablations (§7), with any ablation re-run at N=10 if the
  initial result is borderline (`p ∈ [0.01, 0.05]`). The re-run **replaces**
  the original (not pooled — pooling would be a garden-of-forking-paths).
- Pooled IQM test across envs (when reported as a tertiary descriptive
  statistic): stratified bootstrap CI excludes 1.0 at the 95% level.
- Pre-run: `preregistration.md` and `thresholds.json` committed *before* the
  headline benchmark starts (git hash referenced in paper).
- All reported confidence intervals are **stratified bootstrap** (Agarwal §4),
  not Gaussian. Stratification key: env.
- **Prospective power transparently reported** in §5 of the paper itself, not
  hidden in supplementary.
- **Monte Carlo power simulation** (Weibull + immortal-mass) committed to run
  before Phase 5 locks. Spec in `thresholds.json` →
  `prospective_power_assumption.monte_carlo_spec`. If simulated power < 0.65,
  §5 is amended in §13 with revised N or honest underpowered framing **before**
  any Ragnarok headline seed runs.
- **Null-result framing pre-declared:** if H1-primary returns `p > 0.05` at
  N=20, the paper reports "underpowered to reject the null at α=0.05 with
  observed effect size X" — *not* "no effect found." This framing is
  preregistered to prevent retrospective misclassification.

## 6. Architecture — must-do fixes before benchmark

### 6.1 Code fixes (Phase 1, 5–6 days)

1. **Wire `LatentPolicyHead` into acting path.** Currently trained (commit
   `efe1410`) but never called to produce actions. Add `acting_policy_mode` ∈
   `{"obs", "latent"}`; default `"obs"`; set `"latent"` after cross-task
   transfer load. In `collect_episode`, branch on mode. Add regression test
   `test_cross_dim_acts_from_latent` asserting `latent_policy.forward` is
   called during a CartPole→Acrobot rollout. **This is publication-blocking:
   without it, every previous "transfer number" is meaningless.**
2. **Consolidate GAE.** One `ragnarok/learning/advantages.py`;
   `real_experience`, `dream_augmenter`, `dreamer` all import from it. Sign
   convention and continues-handling identical.
3. **Strip env-name reward shaping from default path.** Move shapers to opt-in
   `RewardShapingConfig` (default off). All benchmark numbers reported without
   shaping unless explicitly marked `+shape`.
4. **Freeze checkpoint schema.** Remove `ckpt.get("policy", ckpt.get("actor_critic"))`
   backward-compat shims. Migrate old ckpts once; error loudly after.
5. **Smoke benchmark before refactor.** 3 envs × 2 seeds × 100 eps, pinned
   RNG, final-return snapshot committed as regression fixture.

### 6.2 Python 3.11 venv for DMControl (Phase 2, 1–2 days)

`dm_control` has no Python 3.14 wheel. Separate venv at `venv311/`. Document
setup in `README.md`. CI matrix: existing unit tests on 3.14, DMC integration
tests on 3.11. `lifelines` (for K-M analysis) installed in both venvs.

### 6.3 Module decomposition (Phase 6, deferred)

Decomposition of god-object `agent.py` into `learning/trainers/`, `acting/`,
`core/transfer.py` is **deferred until after pilot succeeds**. If pilot kills
the project, we don't reorganize files we're about to throw out.

## 7. Ablations

Five seeds each, **on the primary endpoint only** (MountainCarContinuous from
CartPole). Secondary-env ablations are appendix-only and only run if compute
allows (§12).

| Ablation                            | Null it kills                                  |
|-------------------------------------|------------------------------------------------|
| A1: Frozen trunk                    | Trunk must carry transferable features         |
| A2: Randomly-initialized trunk      | Heads alone aren't enough                      |
| A3: Source-task reward shuffle      | Source *task structure* matters                |
| A4: ObsEncoder-only transfer        | RSSM (not just encoder) matters                |
| A5: Latent-dim sweep {64,160,512}   | 160 isn't cherry-picked                        |
| A5b: H2 confirmatory (deferred)     | Library-size scaling — separate prereg if H2   |
|                                     | descriptive trend justifies the run            |
| A6: No-retrieval (fixed skill)      | Centroid retrieval matters                     |
| A7: Random-retrieval control        | Centroid retrieval beats random-skill pick     |
| A8: Equal-FLOP-source Continual-Dreamer | Source-training compute, not architecture, drives result |
| **A9: Shuffled-dynamics RSSM**      | RSSM features generalize, not just init        |

A9 (added in v3) is the mechanism-isolation test: retrain the RSSM on a
permuted-transition version of the source env (same observations,
**cross-trajectory shuffle of next-state targets** — pinned in
`thresholds.json` to prevent post-hoc choice between within- and
cross-trajectory shuffles), keep the trunk weights and the centroid, run
transfer. If A9 matches A1/A2 performance, the "trunk transfers" claim
collapses to "any decent initialization works" — H1 dies. If A9 is materially
worse than the headline, it isolates the *learned RSSM dynamics* as the
load-bearing mechanism, not weight-init alone. This is the single ablation a
hostile reviewer is most likely to demand. **Executability flagged for G1
review:** the trainer must accept a transition-shuffling wrapper; if it does
not, A9 wiring is part of Phase 1's must-do fixes (§6.1).

A8 (revised in v3 from "equal-parameter" to "equal-FLOP-source"): control on
*source-training compute*, not just parameter count, so the "Ragnarok had
more compute upstream" critique is closed.

Ablation runs scheduled after the main benchmark is compute-committed.

## 8. Pilot experiment (Phase 3 — the gate)

**Duration:** 7–10 days after Phases 1 & 2 finish (≈ week 3 of project).

**Setup:** Pilot is a small-N rehearsal of the **headline test**, not a
different test (v3 fix per G0 round-2 review). 5 seeds × 3 source→target
pairs (one of which IS the primary endpoint) × {scratch, transfer} = 30 runs.
Each run 200 k env-steps or convergence (per §4.5).

| Pair                                  | Role                            |
|---------------------------------------|---------------------------------|
| **CartPole → MountainCarContinuous**  | **Primary-endpoint rehearsal**  |
| CartPole → Acrobot                    | Secondary-endpoint rehearsal    |
| Pendulum → DMC-cartpole-swingup       | Secondary-endpoint rehearsal    |

**Pass criteria (ALL must hold):**
- On the primary-endpoint rehearsal pair (CartPole→MCC): RMST ratio
  (scratch/transfer) ≥ 1.3× with one-sided log-rank `p < 0.10` (relaxed α
  for pilot; headline uses `p < 0.05`)
- On at least one secondary pair: RMST ratio ≥ 1.3× directionally (no p
  threshold — pilot is too small)
- No pair shows anti-transfer (RMST ratio < 0.9×)
- Mechanism check: `acting_policy_mode == "latent"` confirmed in logs for the
  transfer arm of every run (closes the "still half-wired" failure mode)

**Fail → activate Plan B (§10).** No negotiation. We do not massage the pilot.

## 9. Review gates

Multi-agent review mandatory at each gate. ≥2 agents in parallel, distinct
lenses. Dissent is the goal.

| Gate | When                       | Agents                                   |
|------|----------------------------|------------------------------------------|
| G0   | Before commit of this file | Compute auditor + Statistician (DONE)    |
| G1   | End of Phase 1             | Architecture + Testing                   |
| G2   | End of Phase 3 (pilot)     | Research strategist + Devil's advocate   |
| G3   | End of Phase 5 (benchmark) | Experimental design + Statistics + DA    |
| G4   | Pre-submission draft       | Research strategist + Writing            |

Each gate produces a written verdict in `reviews/gate_N.md` committed to repo.
G0 already executed; verdicts informed v2 of this document.

## 10. Plan B (if kill criterion triggers)

Re-ranked in v3: B2 promoted to first because it requires zero new
infrastructure and uses the runs already produced. B1 demoted because the
sequential-training harness + forgetting-metrics infrastructure does not
exist; building it inside ~10 weeks after a week-4 pilot fail is high risk.

- **B1: "When does world-model transfer fail" negative-result paper
  (formerly B2).** Re-uses every benchmark run already produced. Honest
  workshop fit; these get accepted if the failure analysis is rigorous and
  identifies the breakdown mode (e.g., "transfer fails when source action
  space is discrete and target requires multi-step continuous control").
- **B2: Sequential-crystallization catastrophic-forgetting paper (formerly
  B1).** Requires a sequential-training harness + retention-metric pipeline
  that does not currently exist. Estimated additional infrastructure cost: ~3
  weeks of build before any new results. Only viable if pilot fails by week 4
  (leaving ~10 weeks).
- **B3: JOSS software paper.** Open-source Ragnarok as a modular
  Dreamer-based skill-transfer research toolkit. No novelty bar. Secondary
  asset regardless of which A/B path we take.

## 11. Kill criteria (non-negotiable)

| Week | Trigger                                                     | Action            |
|-----:|-------------------------------------------------------------|-------------------|
|    1 | Lit review finds ≥3 direct prior works conjoining (a)+(b)+(c) of §1.5 | Reformulate claim (moved up from week 2 in v2; threshold tightened from ≥10 to ≥3 — even one direct overlap is publication-fatal) |
|    4 | Pilot (§8) fails any pass criterion                         | Switch to Plan B  |
|    8 | Full benchmark: primary endpoint shows < 1.3× RMST ratio OR `p > 0.05` after censoring sensitivity sweep | Write B1 negative-result paper |

## 12. Phases & timeline

Revised from v2 per compute-feasibility re-audit. **Phase 1 now produces a
measured wall-clock table** before Phase 5 budget is locked.

| Phase | Deliverable                                | Duration | Cumulative |
|------:|--------------------------------------------|---------:|-----------:|
|    0  | This preregistration committed (v3)        |    3 d   |       3 d  |
|    1  | Architectural must-do fixes (§6.1) **+ smoke-bench wall-clock table** (§12.5) |    7 d   |      10 d  |
|    2  | Python 3.11 DMC venv + CI (§6.2)           |    2 d   |      12 d  |
|    3  | **Pilot** (§8) — gate G2                   |   10 d   |      22 d  |
|    4  | Baseline implementations & SB3 threshold pre-runs (fills `thresholds.json`) |   16 d   |      38 d  |
|    5  | Full benchmark — primary endpoint at N=20 + secondaries at N=10 |   28 d   |      66 d  |
|    5b | Ablations on primary endpoint (5 seeds × 9 ablations) |   12 d   |      66 d  |
|    6  | Module decomposition (optional, post-pilot)|   10 d   |      76 d  |
|    7  | Paper draft, figures, README, OpenReview   |   18 d   |      94 d  |

Total: ~13–14 weeks compute+work, ~16 weeks with slack. Slippage log in §13.

### 12.5 Compute budget gating

Phase 1 produces a measured wall-clock per (method, env) on the actual 4080
via a 3-seed × 50k-step smoke run. The smoke-bench numbers are committed to
`compute_budget.json` and Phase 5 only proceeds if extrapolation fits in 28
days at 90% GPU duty cycle. The 28-day wall-budget is compute-device-
independent; the throughput-derived projection from the 4080 smoke
determines whether the claim budget fits. If it does not, §12.5 cut order
applies.

If extrapolation exceeds budget, **pre-declared cuts apply in this priority
order** (no on-the-fly negotiation):
1. Drop DMC ablations (A1–A9 on DMC) → ablations only on MountainCarContinuous
2. Drop secondary endpoints from headline → MountainCarContinuous-only paper
   with weaker title: "Discrete-to-continuous skill transfer in latent
   actor-critic with shared RSSM trunk: a single-environment study". **If cut
   #2 fires, the paper additionally runs A1+A2+A8+A9 as an expanded
   mechanism-isolation panel on MCC** (4 ablations × N=5 = 20 extra runs ~ 25
   GPU-hr) — single-environment papers survive workshop review only on
   mechanism depth, so this is preregistered as a non-negotiable companion to
   cut #2.
3. Drop primary cell N=20 → N=15 (power drops to ~0.62 at HR=1.5; flag in §5
   and §13)

Continual-Dreamer source-training (Kessler 2023 multi-task setup) explicitly
budgeted: ~15 GPU-hr per source seed × 5 source seeds = 75 GPU-hr included
in Phase 4 baselines, not Phase 5.

## 13. Amendments

- **2026-04-14 (v2):** Pre-commit revision after G0 round-1 stress-test review.
  Changes from v1:
  - §1: H2 demoted to exploratory; H1 restated in RMST terms
  - §2: 4 target envs → 3 (DMC cheetah-run to appendix)
  - §3: 8 baselines → 5
  - §4: 500k-imputation censoring → Kaplan-Meier + RMST + log-rank
  - §4.5: Pinned analyst DoF in `thresholds.json`
  - §5: Bonferroni → Holm-Bonferroni
  - §7: Added A7, A8; A5b H2-confirmatory deferred
  - §9: Added gate G0
  - §12: 11–12 weeks → 14–16 weeks

- **2026-04-14 (v3 patches):** Round-3 G0 review (methodology 8.6/10 PASS,
  devil's advocate 8.2/10 PUBLISHABLE). Three trivial fixes applied
  pre-commit:
  - `thresholds.json` synced with v3 prose (version bumped, `fwer_correction`
    rewritten to "none on primary; Holm on 2 secondaries", `headline_seeds_N`
    renamed `secondary_seeds_N`, `censoring_tau_sweep_env_steps` and `ablations`
    block added including pinned A9 shuffle spec)
  - §5: Monte Carlo power simulation pre-committed; null-result framing
    pre-declared to prevent retrospective reframing
  - §7: A9 shuffle type pinned (cross-trajectory) and executability flagged for G1
  - §12.5: Cut option #2 (MCC-only paper) now non-negotiably bundles
    A1+A2+A8+A9 mechanism panel (mechanism depth required for single-env
    workshop survival)

- **2026-04-14 (v3):** Pre-commit revision after G0 round-2 stress-test review
  (devil's advocate scored v2 at 5.8/10 — well below 8.5 bar). Changes from v2:
  - §1: H1 split into **H1-primary** (single-endpoint confirmatory at N=20)
    and **H1-secondary** (descriptive on 2 other envs at N=10). Power
    calculation in §5 — N=10 was underpowered (~0.40); N=20 with single
    primary endpoint reaches ~0.72 at HR=1.5 with 40% censoring.
  - §1.5: Added explicit novelty-delta paragraph vs Choreographer, CoWorld,
    SPiRL, OPAL, Continual-Dreamer.
  - §4: Single-primary-endpoint headline test (no Holm needed for primary);
    Holm only on secondaries.
  - §4.6: Added censoring sensitivity sweep at τ ∈ {300k, 500k, 750k, 1M}.
  - §5: Power calculation transparently reported in paper §5, not buried.
    Ablation re-run policy clarified: replaces (not pools) the original.
  - §7: Added **A9 shuffled-dynamics RSSM** (mechanism isolation — answers
    "trunk transfers vs init-effect" critique). A8 revised from
    "equal-parameter" to "equal-FLOP-source" (closes "more upstream compute"
    critique).
  - §8: Pilot now rehearses the headline test (CartPole→MCC is the primary
    endpoint and the primary pilot pair); pilot pass criterion aligned with
    headline RMST framing; mechanism check on `acting_policy_mode == "latent"`
    added.
  - §10: Plan B re-ranked — B2 (negative-result, formerly B2) promoted to B1;
    sequential-crystallization (formerly B1) demoted to B2 because
    infrastructure does not exist.
  - §11: Week-1 lit-review kill criterion (was week 2); threshold tightened
    from ≥10 prior works to ≥3 conjoining (a)+(b)+(c) of §1.5; week-8
    criterion re-anchored to primary endpoint + censoring sweep.
  - §12: Phase 1 now produces measured wall-clock smoke-bench
    (`compute_budget.json`); §12.5 pre-declares cut order if compute overruns;
    Continual-Dreamer source-training compute explicitly budgeted in Phase 4.
- **2026-04-14 (v3.1 hardware correction):** §header and §12.5 referenced
  "RTX 5090" as the compute device. Corrected to RTX 4080 (the actual
  hardware). No methodology change — the 28-day wall-budget is
  device-independent, and the §12.5 cut order is keyed off measured
  throughput from `compute_budget.json` rather than the device name.
  Smoke-bench ground truth on RTX 4080 is the load-bearing number; device
  name in prose is cosmetic.

- **2026-04-14 (v3.2 week-1 lit-review result):** §11 kill criterion
  discharged. Independent lit-review agent searched ICLR/NeurIPS/ICML/RLC
  proceedings 2021–2026, arXiv cs.LG/cs.AI, and Google Scholar for works
  conjoining all three of §1.5's (a) RSSM cat(h,z) trunk, (b)
  nearest-centroid skill retrieval on encoded obs, (c) discrete→continuous
  action-space transfer. **Result: 0 works satisfy all three.** Ten
  candidates examined; closest partial matches:
  - **LEGION (Nature MI 2025):** DPMM-clustered skill memory — partial on
    (b), but SAC backbone (not RSSM) and continuous throughout. Closest
    single prior; cite prominently in related work.
  - **SRSA (NVlabs 2025):** skill-retrieval-for-assembly library with
    learned success predictor — partial on (b), both (a) and (c) absent.
  - **XSkill (CoRL 2023):** prototype-clustered skill embeddings —
    partial on (b), both (a) and (c) absent.
  - **Cross-Embodiment Latent Space Alignment (arXiv 2406.01968):**
    continuous↔continuous dim-mismatch, not action-type change.
  - **TrajWorld (ICML 2025):** transformer heterogeneous-env world
    model — no RSSM, no skill library, no action-type change.
  - **Dreamer 4 (arXiv 2509.24527, 2025):** moves away from RSSM toward
    transformer dynamics; confirms RSSM-centric claim is still the
    minority branch in late 2025–early 2026.

  **Novelty-delta clarification to §1.5** (to be reflected in paper
  related-work prose, does NOT change hypothesis): the discriminating
  axis for H1 is **action-space type change** (Discrete → Box), not
  **dim mismatch** (Box_n → Box_m). The latter is a crowded subfield in
  2025 (hypernetwork policies, latent alignment, unified action spaces);
  the former remains unoccupied in the conjunction with (a)+(b). Paper
  must keep this distinction sharp so reviewers don't conflate H1 with
  the dim-mismatch line.

  **No hypothesis change. No endpoint change. No power recalculation.**
  Project proceeds with H1 as preregistered; new references added to
  related-work bibliography. Amendment timestamped pre-execution of
  Phase 2 so the paper's "week-1 lit review" narrative is verifiable
  against git history.

- (Subsequent amendments timestamped here before execution.)
