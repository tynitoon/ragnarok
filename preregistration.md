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

### 1.0 Elevator pitch (one paragraph for reviewers / abstract seed)

> Existing skill-transfer methods (PEARL, MAML, Continual-Dreamer,
> Choreographer, LEGION) require **homogeneous action spaces** or
> **task-specific adapter networks** (hypernetworks, per-task heads,
> meta-training). Ragnarok demonstrates that a shared RSSM latent trunk
> over `cat(h, z)` plus nearest-centroid skill retrieval enables positive
> forward transfer **across a discrete→continuous action-space change**
> on the same obs dimensionality class, without per-task hypernetworks
> or meta-training. The load-bearing novelty is the *action-space type
> change* (Discrete → Box), not the *dim mismatch* (Box_n → Box_m) that
> hypernetwork and latent-alignment work already addresses. The primary
> confirmatory test is CartPole-v1 → MountainCarContinuous-v0 (§8 / §5);
> two secondary pairs (CartPole→Acrobot, Pendulum→DMC-cartpole-swingup)
> and one adversarial-negative pair (Pendulum→Reacher, §7 A10) test
> generality and falsification.

### 1.1 Formal hypotheses

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
| **A10: Adversarial-negative pair**  | Effect is not "transfer succeeds on any target"; pendular-physics cherry-pick |
| **A11: GRU-shuffled transferable trunk** | Learned GRU dynamics, not just spectral-norm initialization |

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

A10 (v3.5): CartPole-v1 → DMC finger-spin. Same action-space type change as
the primary pair (Discrete → Box; isolates the H1 axis), but non-pendular
physics class on the target (rotational-forced finger, no gravity well).
Predicted outcome under H1: no transfer or anti-transfer. Reported in the
headline table regardless of direction; if A10 shows transfer parity with
the primary pair, the paper must reframe the claim scope as "transfer works
on any cross-action-type pair, not only on pendular-class physics". N=5
seeds per arm, ~15-20 GPU-h total. Runs via
`scripts.pilot_run --run-adversarial`.

A11 (v3.5): GRU-shuffled transferable trunk. After `try_transfer()` loads
the RSSM-core subset (`core.gru.*` + prior/posterior), permute rows and
columns of `core.gru.weight_ih_l0` / `weight_hh_l0` and reshuffle the
corresponding bias entries. Row-column permutation preserves the full
singular-value spectrum (Frobenius + spectral norm unchanged), total
parameter count, and the weight-magnitude distribution — but destroys any
learned temporal correlation. If A11 ≈ real transfer on the primary metric,
the "learned recurrent dynamics transfer" mechanism claim dies and the
paper reduces to "transferring *any* initialization with the right spectral
properties works." N=2 seeds on the primary pair (cartpole_mcc), ~5 GPU-h.
A11 is lighter than A9 (A11 permutes already-trained GRU weights, A9
re-trains the RSSM on shuffled trajectories) and complementary — A9 tests
"RSSM features generalize", A11 tests "the specific GRU recurrent structure
matters, not just init properties". Runs via
`scripts.pilot_run --ablation shuffled-gru`.

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

**v3.5 adds Plan B0** between the §8 PASS path and Plan B1 so a modest
but statistically reliable transfer effect does not fall into the §8
binary's null gap. Previously the decision tree was
`PASS | Band B rescue | Plan B1/B2/B3`; a 1.15–1.30× ratio at p < 0.10
that survives the Band B single-cell HP sweep without clearing 1.30×
would have collapsed into B1, which framing-wise overclaims the failure.
Plan B0 pre-declares the "modest but reliable" framing so the paper
narrative matches the evidence.

- **B0: "Modest but reliable cross-action-space transfer" companion
  paper (new in v3.5).**

  *Trigger:* activates if and only if all of the following hold after
  the Band B HP sweep (per Bug E v4 amendment):
    1. Primary-pair RMST ratio ∈ [1.15, 1.30) at one-sided log-rank
       `p < 0.10` AND permutation `p < 0.10` (both — v3.5 analyzer
       upgrade; asymptotic log-rank alone is untrustworthy at N=5)
    2. No anti-transfer on any pair (ratio ≥ 0.9 on all 3)
    3. Mechanism check passes: `acting_policy_mode == "latent"` on
       all cross-dim transfer runs AND the §7 A11 GRU-shuffled
       ablation (v3.5) shows a ≥ 0.10 ratio gap vs real transfer
       on the primary pair. A11 is the load-bearing mechanism
       filter for B0: if shuffled ≈ real, B0 is NOT available —
       fall through to B1.
    4. Sign-test 4/5 seed-direction filter passes (Bug E v2
       amendment — unchanged).

  *Framing (preregistered, not post-hoc):* paper title and abstract
  frame the result as "a reliable but modest forward-transfer
  effect from discrete to continuous action spaces via shared RSSM
  latent trunk", NOT as "strong transfer" or "significant speedup".
  The honest-magnitude constraint is load-bearing: the paper MUST
  report the observed ratio as the headline number, NOT the Band B
  lower edge, NOT a rescued cell's ratio, NOT an AUC derivative.
  RMST ratio and permutation-p remain the §4 headline metrics.

  *Mandatory companion analyses (pre-committed, all required):*
    - Early-step descriptive panels (v3.5 §4 secondary): mean
      return at 2k / 5k / 10k env-steps with bootstrap 95% CIs on
      both arms; AUC over [0, 50k] env-steps. These are where a
      modest effect is more visible; reporting them is not
      cherry-picking, because §4 pre-committed them as descriptive
      secondaries regardless of outcome.
    - Per-seed scatter (all seeds, no selection) on primary-metric
      steps-to-mastery.
    - §7 A10 (adversarial-negative pair Pendulum→Reacher) and A11
      (GRU-shuffled) results REPORTED in the headline table, not
      appendix. The scope-bounding honesty of "look, it fails here
      and here" is what lets B0 survive review at workshop tier.
    - Explicit "Why we don't claim more" section: the ratio is
      reliably positive but modest; the mechanism check passed; we
      don't know whether a larger-N study would lift the ratio or
      the Band B HP sweep already caught the optimum.

  *Why B0 is not a back-door around §8:* B0 does NOT replace the
  §8 headline test. If the pilot lands in Band A, §8 PASS proceeds
  to Phase 5 headline N=20 and a strong-claim paper. B0 is only
  available **after** the Band B single-cell rescue fails AND the
  mechanism filters (4) pass. A reviewer can verify chronology via
  git history: B0 was committed at `4f8bb11` on 2026-04-15 13:28,
  **while pilot #2 was in progress** — 2 of 5 primary seeds (42, 43)
  had completed showing a partial ratio of ~1.10, and seeds 44-46 had
  not yet produced data (see `reviews/chronology_audit.md` for the
  full reconstruction). The B0 band edges `[1.15, 1.30)` were committed
  unchanged from the v3.4 Bug E v3 amendment (commit `e24832c`,
  2026-04-15 03:28, **pre-pilot**); they were NOT tuned to the partial
  observed signal, which sat below the band's floor at the time of B0
  commit. The §8 primary threshold `1.30` remained fully pre-pilot
  (v3, 2026-04-14) and did not move. B0 is therefore *pre-outcome*
  (before the N=5 ratio could be computed) but NOT *pre-pilot*; the
  integrity claim is that no N=5 RMST ratio existed anywhere in the
  project when B0's band was committed, which is the load-bearing
  unblinding event for the §8 primary decision. B0 is a
  *framing-honesty* mechanism, not a *pass-bar-relaxation* one.

  *Why B0 is not "Band B success renamed":* Band B runs a single
  warmup-LR HP cell at N=3 to see if the first-cut HP was wrong.
  Band B success (ratio ≥ 1.20 at p < 0.10 in the rescue cell)
  means "HP rescue found — rerun Phase 5 headline at the rescued
  HP". B0 activates when **no HP cell rescued the signal** but the
  original pilot still shows a reliable modest effect. Band B says
  "try harder"; B0 says "this is the real effect size, just smaller
  than we'd hoped". Different claims, different remedies.

  *Falsification surface:* if, during the B0 companion analyses,
  any of (A10 shows transfer parity / A11 shows shuffled ≈ real /
  per-seed 4-of-5 direction filter fails) emerges, the paper
  converts to Plan B1 (negative-result) — B0 is *conditional* on
  the mechanism filters holding, not a guaranteed fallback.

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
|    4 | Pilot (§8) fails any pass criterion                         | Switch to Plan B (B0 if modest-reliable effect + mechanism filters hold; else B1) |
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

- **2026-04-14 (v3.3 Python 3.11 → 3.10 substitution, operational):**
  §6.2 specified a **Python 3.11** venv for DMControl. On the target
  workstation the only installable minor versions from
  python.org/releases that are available are 3.14 (main) and 3.10;
  3.11 is not available and installing it would require an unrelated
  system change. Python 3.10 is substituted because:
  - `ragnarok/pyproject.toml` declares `requires-python = ">=3.10"`,
    so the project officially supports 3.10.
  - `dm_control==1.0.38` supports Python 3.8–3.12; 3.10 is inside
    that range.
  - `mujoco==3.7.0` and `lifelines==0.30.0` both install cleanly on 3.10.
  - No ragnarok module uses 3.11-only syntax (verified by static grep:
    no `match` statements, no `except*`, no `tomllib` imports, no PEP
    695 generic syntax).

  **Venv directory renamed:** `venv311/` → `venv310/`. References in
  §6.2 prose and §12 timeline entry remain correct in spirit (isolated
  venv for DMC deps); only the minor version differs. Tests that run
  in this venv are identical.

  **No methodology change.** DMC envs only enter H1-secondary (Acrobot,
  DMC-cartpole-swingup) and H2 (exploratory). Primary H1 endpoint
  (CartPole → MountainCarContinuous) runs entirely in the main 3.14 env
  and is unaffected. Amendment timestamped pre-execution of Phase 2
  smoke on DMC.

  Operational artifacts produced by this amendment:
  - `venv310/` with torch+cu126, dm_control 1.0.38, mujoco 3.7.0,
    lifelines 0.30.0, ragnarok (editable)
  - `pyproject.toml` build-backend corrected from legacy placeholder
    to `setuptools.build_meta`; `[tool.setuptools.packages.find]`
    pinned to `ragnarok*` (setuptools would otherwise pick up
    `logs/`, `checkpoints/`, `skills_data/`, `venv310/` as top-level
    packages and refuse to build)
  - `SETUP.md` with reproducible install steps for both envs

- **2026-04-14 (v3.4 — Bug E discovered mid-pilot; pilot #1 killed; co-transferred RSSM core + LR warmup):**
  Phase 3 pilot #1 was launched per §8 (5 seeds × 3 pairs × {scratch, transfer}).
  At N=2 of the primary-endpoint rehearsal (CartPole→MCC) the transfer
  arm produced an RMST ratio of ~0.98 — well below the 1.3× pass
  criterion and indistinguishable from no-transfer despite
  `acting_policy_mode == "latent"` being confirmed in logs and the
  expected source skill being loaded. A devil's-advocate review agent
  pointed at the `try_transfer` cross-dim branch and asked what
  exactly was being transferred. Investigation found:
  - **Bug E (architectural, not plumbing).** The "transferable trunk"
    saved with each Skill consisted only of the latent policy MLP
    weights (`shared.*` + `critic_head.*`). The RSSM that produces
    the `(h, z)` features the trunk consumes was NOT serialized and
    NOT loaded on the target env. The cross-dim transfer therefore
    moved a policy that reads `cat(h, z)` features but left the
    target env with a fresh-random RSSM that emits noise. The trunk
    saw garbage and the §8 mechanism check trivially failed even
    though `acting_policy_mode == "latent"` was structurally true.
    This is consistent with the observed 0.98 ratio: random features
    in, random behaviour out.

  **Pilot #1 killed at N=2/5 on the primary pair.** Artifacts
  preserved at `pilot_results.json.broken_trunk` and
  `pilot_run.log.broken_trunk` for the post-mortem section of the
  paper, but excluded from any §4 metric. Killing mid-pilot is not a
  garden-of-forking-paths violation — Bug E was a code defect, not a
  result the run produced; the §8 pass criteria were never evaluated
  on the broken pipeline.

  **Fix scope (one atomic commit, all Phase A–F below).**
  - Phase A (RSSM API): split RSSM into env-agnostic transferable
    subset (`core.gru`, `core.prior`, `core.posterior`) vs per-env
    IO (encoder, `core.pre_gru`, decoder, reward + continue
    predictors). New methods `transferable_state_dict()`,
    `load_transferable_state_dict(strict=True)`,
    `transferable_params()`, `non_transferable_params()`. Strict
    load raises on shape mismatch — silent acceptance would have
    re-introduced Bug E.
  - Phase B (Skill schema): added `rssm_core_state_dict` field
    (`default_factory=dict` for backward-compat with pre-Bug-E
    skills); `SkillLibrary.save_skill` serializes it; the existing
    meta-test `test_every_skill_dataclass_field_is_serialized`
    catches any future omission.
  - Phase C (optimizer): `WorldModelTrainer` now uses two named Adam
    param groups (`transferable`, `io`) so the transferable subset's
    LR can be scaled independently of the per-env IO that needs full
    LR to learn the target's obs/action layout. New methods
    `set_transferable_lr_scale(scale, warmup_episodes)`,
    `step_episode()`, `get_transferable_lr()`. Defaults pinned in
    `RagnarokConfig.transfer`: `rssm_transfer_lr_scale = 0.1`,
    `rssm_transfer_warmup_episodes = 200`.
  - Phase D (agent wiring): crystallization saves the RSSM core;
    `try_transfer` cross-dim branch loads the core BEFORE the trunk
    (the trunk's behaviour depends on the core producing consistent
    features), flips `acting_policy_mode = "latent"`, and applies
    the LR warmup. Failure during cross-dim load (shape mismatch on
    `hidden_dim` / `stoch_dim` / `encoder_hidden`) returns `None`
    cleanly rather than pretending transfer succeeded. Trust region
    is now gated on `acting_policy_mode == "obs"` — capturing the
    obs policy as a KL reference in latent mode would pull a
    randomly-initialized policy toward random init, which is wrong
    and irrelevant. `wm_trainer.step_episode()` wired at all 6
    episode-end sites.
  - Phase E (regression suite, `tests/test_rssm_transfer.py`,
    24 tests): partition correctness; cross-env load preserves IO
    layers; strict-vs-nonstrict shape-mismatch behaviour; skill
    serialization round-trip including empty-default backward-compat;
    LR-scaling param-group disjointness + countdown semantics;
    end-to-end cross-dim transfer flips acting mode; skipped when
    `rssm_core_state_dict` is empty (pre-Bug-E artifact); trust
    region not activated in latent mode. One behavioural smoke marked
    `@pytest.mark.slow`, run manually before pilot relaunch. Plus 3
    updates to `tests/test_latent_policy.py` mock skills to set the
    new `rssm_core_state_dict` attribute.
  - Phase F (this amendment): timestamped pre-relaunch.

  **Decision rule UNCHANGED at primary threshold.** Pilot #2 will run
  on the fixed pipeline with the same primary configuration as pilot #1:
  §8 pass criteria (RMST ratio ≥ 1.3 on primary, p < 0.10, no
  anti-transfer pair, `acting_policy_mode == "latent"` confirmed in
  logs) and §11 week-4 kill criterion still apply. Number of seeds (5),
  number of pairs (3), and pair identities (`cartpole_mcc` primary,
  `cartpole_acrobot` and `pendulum_dmc_cartpole` secondary) all
  unchanged. No metric, threshold, or analysis pipeline was relaxed.
  This amendment documents an implementation defect and its fix; it
  does not weaken any criterion that the broken pipeline would have
  failed.

  **Pre-relaunch checklist (gating pilot #2 launch):**
  1. `pytest tests/` green (achieved: 357 passed, 1 skipped after
     Bug E v2 fixes; was 338 / 15 at v3.4 commit).
  2. Multi-agent code review on the Bug E fix (G1.5 review — extends
     standing G1 gate per §9). At minimum: an architecture agent on
     RSSM partition correctness + a testing agent on regression-suite
     coverage. Verdicts committed to `reviews/bug_e_fix.md` (NEW)
     before launch.
  3. Behavioural smoke (now **2 seeds**, not 1, per devil's-advocate
     review — see "Bug E v2" amendment below): CartPole → MCC over the
     first ~200 episodes. Required signals: (a) `acting_policy_mode`
     flips to `"latent"`, (b) loaded RSSM core weights survive the LR
     warmup window (||Δθ|| on `core.gru.*` < 30% of initial norm by
     ep 200), (c) `KL(posterior‖prior)` trajectory shows the prior
     becoming relevant (decreasing trend over training).
  4. Pilot #2 launch only after items 1–3 pass.

- **2026-04-15 (v3.4 amendment "Bug E v2" — 3-agent code review on
  the Bug E fix; review-driven hardening):**
  After the v3.4 fix landed (commit `f0c9155`), a 3-agent G1.5 review
  was run per checklist item #2 (architecture / testing / devil's
  advocate). Verdicts and full responses are stored at
  `reviews/bug_e_fix.md`. None of the three reviewers found a
  launch-blocking defect in the fix as committed; two raised a
  partition-emptiness concern that turned out to be a misreading of
  `EnsembleRSSMCore` (it is *additive*, not a replacement, so
  `self.core` and the transferable subset stay intact under the
  default `ensemble_cores=2`); the regression suite has been
  extended to lock that invariant. The remaining reviewer concerns
  produced six review-driven hardenings in this amendment, all
  landed before pilot #2 launch:

  *Code/test hardenings (committed in the same atomic Bug E v2 commit):*
  - **Adam-state reset on transferable group post-load.** New
    `WorldModelTrainer.reset_transferable_optimizer_state()` clears
    `exp_avg` / `exp_avg_sq` for every transferable param at
    `try_transfer` time. Without this, the LR-scale = 0.1 nominal
    cap is meaningless — the bias-corrected first-step magnitude
    depends on stale second-moment estimates.
  - **`encoder_hidden` mismatch raises with explicit guidance.**
    Posterior shape `(64, hidden_dim + encoder_hidden)` silently
    pinned `encoder_hidden` as a project-wide invariant; the new
    error message names it explicitly so any future per-env tuning
    surfaces immediately at skill-load time, not 200 episodes later.
  - **Real LR-drift behavioural test.** `test_lr_warmup_actually_
    dampens_param_drift` runs identical-seed train_steps with and
    without warmup and asserts the warmed group drifts at least 2×
    less. The previous tautological `.lr` field-check tests are
    retained for fast-failure but no longer load-bearing.
  - **Default-config non-empty subset regression test.** Locks the
    `RagnarokConfig()` invariant the reviewers explored — any future
    refactor that empties `transferable_state_dict()` under default
    config now breaks a fast unit test instead of a 20-hour pilot.

  *Decision-rule additions (do NOT relax §8; only add side-rails):*
  - **Three-band post-pilot decision rule (devil's-advocate
    concern #3).** The §8 binary `ratio ≥ 1.3, p < 0.10` still
    decides launch-vs-Plan-B. But a borderline outcome between
    "mechanism dead" and "mechanism alive but first-cut HP wrong"
    is now resolved by a pre-declared band:
      - **Band A — pass:** ratio ≥ 1.3 AND p < 0.10 → proceed to
        Phase 5 headline run.
      - **Band B — diagnostic:** ratio ∈ [1.05, 1.30) at any p, OR
        ratio ≥ 1.30 at p ∈ [0.10, 0.20) → run a single warmup-LR
        sweep at N=3 per cell over `rssm_transfer_warmup_episodes
        ∈ {50, 200, 500}` (~10 GPU-h). If any cell hits Band A,
        proceed with that HP; if no cell does, treat as Band C.
      - **Band C — Plan B:** ratio < 1.05 OR anti-transfer on
        primary OR `acting_policy_mode != "latent"` → activate Plan
        B (§10) immediately.
    The Band-B sweep is bounded (one HP, 3 cells) and cannot be
    extended post-hoc. This is not goalpost-moving: the §8 primary
    threshold is unchanged; Band B distinguishes a fixable
    first-cut HP from a dead mechanism, and any HP rescue must
    clear the same 1.3× / p<0.10 bar.
  - **Sign-test seed-direction filter (devil's-advocate concern
    #5).** Even if Band-A criteria are met, the primary pair must
    show transfer ≥ scratch on at least 4/5 seeds (per-seed wall
    median return after 200k env steps). A 5/5 ratio of 1.3×
    driven by one outlier seed and four ties does not pass — the
    paper claim is "consistent transfer benefit", not "lucky
    seed under a one-sided test".
  - **Smoke pre-check (devil's-advocate suggestion).** The
    behavioural smoke (checklist item #3) is upgraded from 1 seed
    to 2 seeds (~3 GPU-h total instead of ~1h) and now logs three
    diagnostic series: `||Δθ||` on transferable params,
    `||Δθ||` on the latent trunk, and `KL(posterior‖prior)`
    trajectory. If transferable `||Δθ|| > 50%` of initial norm by
    episode 100, abort and investigate before launching pilot #2 —
    the LR warmup is not actually working and pilot #2 will repeat
    pilot #1's failure mode for a different reason.

  *Concerns deferred (documented for transparency, NOT acted on
  before pilot #2):*
  - **Latent trunk has no LR warmup symmetric with the RSSM core**
    (devil's-advocate concern #8). The trust region is gated off
    in latent mode (correctly), but no replacement constraint
    protects the trunk from early noisy PG updates. Decision: rely
    on the 2-seed smoke to catch trunk drift; if `||Δtrunk||`
    exceeds 50% by ep 100, add a symmetric trunk-LR warmup before
    relaunch. Cheap to add post-hoc; not worth pre-emptive scope
    creep.
  - **MCC censoring crushes effective N at 5 seeds** (devil's-
    advocate concern #6). Real concern; sampling variance of
    RMST at N=5 with 30–40% censoring may exceed the 1.3× margin
    in either direction. Decision: do NOT bootstrap-validate the
    SE before pilot #2 (would require the headline-scale SAC
    runs we don't have yet); rely on the Band-B / sign-test
    filters above to catch underpowered positives. If pilot #2
    lands cleanly in Band A, the headline N=20 run will resolve
    any residual SE concern.
  - **Honest mechanism reporting (devil's-advocate concern #2).**
    The transferred prior may act as a marginal regularizer
    rather than a true dynamics carrier. The pilot already logs
    `KL(posterior‖prior)`; the post-pilot analysis will report
    its trajectory in the paper alongside the RMST number. No
    mechanism-rescue claims will be made if KL stays flat over
    training.

  All review-driven changes preserve the §8 / §11 decision rules
  and only add stricter filters. None weaken the pass criterion.

- **2026-04-15 (v3.4 amendment "Bug E v3" — 2nd-round 3-agent code
  review on the v2 hardenings; review-driven hardening, supersedes
  the relevant v2 clauses below):**
  After the v3.4 "Bug E v2" hardenings landed (commit `88dbe8c`), a
  2nd-round G1.5 review was run on the v2 changeset (architecture /
  testing / devil's advocate). Verdicts: architecture
  LAUNCH-READY, testing SUFFICIENT, devil's advocate
  LAUNCH-WITH-MODIFIED-CRITERION (2 blockers). Verdicts and
  dispositions appended to `reviews/bug_e_fix.md`. Three changes
  follow, all landed before pilot #2 launch:

  *Decision-rule edits — these SUPERSEDE the v2 clauses they refer
  to (Band B band edges + Band B winner-promotion rule):*
  - **Band B lower edge raised: 1.05 → 1.15** (architecture review,
    devil's-advocate concern reinforcing). Rationale: at N=5 with the
    expected 30–40% MCC censoring, RMST sampling SE is on the order
    of 0.15–0.25; a 1.05 lower edge is below the noise floor and
    triggers a Band-B HP sweep on null-noise outcomes. Raising to
    1.15 keeps Band B as "weak but real-looking signal" and pushes
    pure noise into Band C (Plan B). The upper Band B condition
    (ratio ≥ 1.30 at p ∈ [0.10, 0.20)) is unchanged because the
    1.30 cutoff is the §8 primary; only the noise-floor edge moves.

    **Effective Band B (supersedes v2):** ratio ∈ [1.15, 1.30) at
    any p, OR ratio ≥ 1.30 at p ∈ [0.10, 0.20).

  - **Bonferroni correction on the Band B HP sweep** (devil's-
    advocate review #2, BLOCKER). The v2 amendment specified a
    3-cell sweep (`rssm_transfer_warmup_episodes ∈ {50, 200, 500}`
    at N=3 each) with the original §8 per-cell α = 0.10. Under the
    null this gives FWER ≈ 1 − (1 − 0.10)³ ≈ 27%: a 1-in-4 chance
    that a "Band B rescue" cell hits Band A by chance alone with
    zero true effect. That's not a rescue, that's regression to the
    mean dressed up as a result.

    **Per-cell criterion (supersedes v2):** each Band B cell must
    clear ratio ≥ 1.30 AND p < 0.0333 (= 0.10 / 3, Bonferroni FWER
    bound at α = 0.10 across the 3 cells) to qualify as a Band B
    rescue winner. The §8 primary threshold (ratio ≥ 1.30, p < 0.10)
    is unchanged for the headline N=20 run; only the underpowered
    N=3 rescue-cell test gets the multiplicity correction. The
    headline N=20 (Phase 5) confirms any Band B winner at the §8
    bar — Bonferroni only protects the *promotion* decision, not
    the eventual claim.

    **Why Bonferroni and not Holm-Bonferroni** (which the rest of
    §5 uses for paired secondary envs): Holm requires sorted
    p-values across the family and is more powerful, but with
    only 3 cells and N=3 per cell the power gain is marginal,
    while the implementation footprint (sorted-p tracking across
    cells in the analyzer) is non-trivial. Plain Bonferroni is
    conservative in the right direction.

  *Code edits (committed atomically with this amendment):*
  - **Smoke telemetry now actually logged** (devil's-advocate
    review #2, BLOCKER). The v2 amendment committed to logging
    `||Δθ||` on transferable params, `||Δθ||` on the latent trunk,
    and `KL(posterior‖prior)` trajectory during the smoke pre-check
    — but no code in `scripts/pilot_run.py` actually emitted them,
    making the prereg's "abort if drift > 50% by ep 100" criterion
    unenforceable from the smoke output. **Fixed:**
    `_train_to_step_budget` now snapshots the transferable subset
    immediately after `try_transfer()` succeeds and captures a
    telemetry record at every eval checkpoint with
    `transferable_drift_max`, `transferable_drift_per_param`, and a
    `kl_posterior_prior` probe (single-batch, no-grad,
    ~few-ms cost). The series is serialized as
    `PilotRun.telemetry` in the output JSON. A real-time
    `[TELEMETRY ALERT]` line is printed the first time
    transferable drift crosses 50% so the operator sees it without
    scraping JSON. **Trunk drift logging deferred** to a follow-up
    commit if pilot #2 needs it; the v2 amendment's deferred
    "trunk LR warmup" decision (concern #8) hinges on trunk drift,
    so this is not strictly required for the pilot launch decision.

  *Testing edits (same atomic commit):*
  - LR-drift threshold tightened from 2× to 4×: the v2
    `test_lr_warmup_actually_dampens_param_drift` only required the
    warmed group to drift half as much as the unwarmed baseline. The
    nominal LR scale is 0.1× (10× expected dampening), so 2× passes
    a "half-broken warmup" mutant. The 4× threshold rejects the
    obvious mutants while staying safely above the natural variance
    of identical-seed Adam runs.
  - Reset-state lazy-init verification: a new assertion runs one
    `train_step` after `reset_transferable_optimizer_state()` and
    confirms that Adam re-creates `exp_avg`/`exp_avg_sq` on the
    next step (closes the gap between "state was deleted" and
    "Adam actually re-initializes correctly").
  - `encoder_hidden`-only mismatch test extended to the
    `hidden_dim`-only confusion case so the error-message guidance
    doesn't accidentally fire on the wrong root cause.
  - `try_transfer` integration test that asserts the call ordering
    `reset_transferable_optimizer_state → set_transferable_lr_scale`
    (reset must precede scale; reverse order is silently wrong but
    type-checks fine).

  All v3 changes preserve the §8 / §11 primary decision rules
  unchanged at the headline N=20. v3 only tightens v2's
  rescue-cell (Band B) and smoke-precheck side-rails. None
  weaken any pass criterion.

- **2026-04-15 (v3.4 amendment "Bug E v4" — 3rd-round 3-agent code
  review on the v3 hardenings; review-driven hardening, supersedes
  the relevant v3 clauses below):**
  After the v3 hardenings landed (commit `e24832c`), a 3rd-round
  G1.5 review was run on the v3 changeset (architecture / testing /
  devil's advocate). Verdicts: architecture FIX-ONE-MAJOR (raw KL
  vs free-nats clamped KL — fixed before any further review),
  testing INSUFFICIENT-WITHOUT-FIX (closure-extracted telemetry had
  zero unit-test coverage — fixed with 7 new tests in
  `TestComputeTransferTelemetry`), devil's advocate
  LAUNCH-WITH-MODIFIED-CRITERION (1 BLOCKER, 3 MAJORs on Band B
  power, smoke flag, lower edge). Verdicts and dispositions
  appended to `reviews/bug_e_fix.md`. Five changes follow, all
  landed before pilot #2 launch:

  *Decision-rule edits — these SUPERSEDE the v3 clauses they refer
  to (Band B sweep design + lower edge):*
  - **Band B sweep collapsed: 3 cells → 1 cell at warmup_episodes=200,
    N=5** (devil's advocate v3 BLOCKER). Power analysis on the v3
    Bonferroni-corrected design at α = 0.0333, df = 2, ratio = 1.5,
    σ = 0.25 yields power ≈ 7.4% — Band B was statistically dead.
    The single-cell rescue at the same warmup_episodes=200 anchor
    used by the §8 primary recovers per-cell α = 0.10 (no
    multiplicity correction needed for a 1-cell test) and lifts
    power on the same ratio/σ to ≈ 50%. Rationale for keeping
    warmup_episodes=200 specifically (not the v3 grid {50, 200, 500}):
    it's the only cell with prior architectural justification (the
    LR warmup horizon argued for in the v2 amendment); the others
    were exploratory. If pilot #2 lands in Band B at warmup=200, a
    follow-up sweep with proper N can refine; if it lands in Band C,
    the prereg's Plan B is the answer, not a wider sweep.

    **Effective Band B (supersedes v3):** single-cell rescue with
    `rssm_transfer_warmup_episodes = 200`, N = 5, ratio ≥ 1.20 at
    p < 0.10 (ratio threshold raised — see next bullet).

  - **Band B lower edge raised: 1.15 → 1.20** (devil's advocate v3
    MAJOR). Even at the v3-tightened 1.15 edge, with σ = 0.25
    (upper of the 0.15–0.25 noise range estimated in v3) the null
    p-value for a 1.15 ratio is ≈ 0.17 — above the 10% bar that
    §8 primary uses. Raising to 1.20 yields a null p ≈ 0.10 at the
    same σ, matching the §8 α exactly and pushing the noise floor
    out of Band B. The §8 primary 1.30 cutoff is unchanged; Band B
    only loses its bottom slice.

    **Effective Band B (final, supersedes both v2 and v3):** single
    cell at `rssm_transfer_warmup_episodes = 200`, N = 5, ratio ∈
    [1.20, 1.30) at p < 0.10 OR ratio ≥ 1.30 at p ∈ [0.10, 0.20).

  *Code edits (committed atomically with this amendment):*
  - **Smoke flag now matches the prereg-committed 2-seed protocol**
    (devil's advocate v3 BLOCKER). The v2 amendment committed to a
    2-seed smoke pre-check, but `scripts/pilot_run.py:--smoke` was
    still hardcoding `args.seeds = 1`, silently producing
    single-seed smokes that violated the prereg. **Fixed:**
    `--smoke` now sets `args.seeds = 2` and `args.max_steps =
    40_000` (the v2 default of 20k didn't leave headroom past the
    `||Δθ|| > 50% by ep 100` abort criterion when an episode runs
    long). Help text and usage docstring updated accordingly.

  - **Raw KL probe (no free-nats clamping)** (architecture v3
    MAJOR — fixed pre-amendment, redocumented here for the record).
    The v3 telemetry implementation initially called
    `rssm.loss(...)["kl_loss"]` to get the KL probe, but that path
    applies free-nats clamping (`max(kl, free_nats/stoch_dim)`) and
    averages over stoch dims — the floor exactly matches the
    expected value early in training, so the probe was structurally
    incapable of detecting the "prior crushed" failure mode it
    claimed to monitor. **Fixed:** the probe now calls
    `rssm.observe(obs, actions)` and computes
    `kl_divergence(Normal(post_m, post_s.exp()),
    Normal(prior_m, prior_s.exp())).sum(-1).mean()` directly. The
    telemetry function was extracted from a `_train_to_step_budget`
    closure to module level so it can be unit-tested
    (`TestComputeTransferTelemetry`, 7 tests including the
    load-bearing `test_kl_probe_is_unclamped_raw_kl`).

  *Smoke aggregation rule (pre-declared, not in code):*
  - **2-seed smoke abort logic.** With `seeds = 2` per the BLOCKER
    fix above, the prereg pre-declares: smoke aborts (and pilot #2
    is held) if EITHER seed shows `transferable_drift_max > 0.50`
    at any telemetry checkpoint up to ep 100. The "either" rule
    (not "both" or "mean") is intentionally pessimistic — a single
    seed showing catastrophic drift is sufficient evidence that the
    LR warmup is not doing its job; demanding both seeds confirm
    the failure would risk launching a 20-GPU-h pilot with one
    known-broken arm.

  All v4 changes preserve the §8 / §11 primary decision rules
  unchanged at the headline N=20. v4 collapses an underpowered
  rescue sweep (Band B 3-cell → 1-cell), tightens its lower edge
  (1.15 → 1.20), fixes a code/prereg drift on smoke seeds (1 → 2),
  and replaces a structurally-broken KL probe with a raw KL
  probe — all strictly tightening filters or fixing instrumentation
  bugs. None weaken any pass criterion.

- **2026-04-15 (v3.5 mid-pilot #2 review — 4-agent review landed 6
  decisions on the running pilot):** While pilot #2 was running on
  the Bug E v5.3 pipeline (8/30 runs complete at review time), a
  4-agent review (RL-methodology, code-review, strategy, architecture)
  was commissioned and independently surfaced six convergent concerns.
  All six actions were approved ("tu peux faire du 1 2 3 4 5 6") and
  are being executed DURING pilot #2 because four are analysis-only
  (§1, §10, analyzer code, analyzer metrics) and two consume ~20 new
  GPU-h that runs in parallel. None changes the §8 / §11 headline
  decision rule; all only add explanatory text, explicit side-rails,
  or additional falsification surface.

  *Narrative edits (no methodology change):*
  - **§1.0 elevator pitch added** (all 4 reviewers independently:
    "lead with action-space mismatch, not 'skill transfer'"). Framing
    the load-bearing axis as Discrete→Box (action-type change), not
    Box_n→Box_m (dim mismatch), which is already a crowded subfield
    per the v3.2 lit review. §1.0 is prose; §1.1 formal hypotheses
    unchanged.

  *Decision-rule additions (do NOT weaken §8; only disambiguate the
  middle zone and add a falsification lever):*
  - **§10 Plan B0 added** — pre-declares the modest-but-reliable
    transfer outcome path. See §10 for the exact ratio / p band, the
    required honest framing, and the mandatory "why we don't claim
    more" companion analyses. Plan B0 sits BETWEEN §8 PASS and Plan
    B1 negative-result in the decision tree. Adding Plan B0 does NOT
    move the §8 threshold; it only replaces the old implicit "small
    positive → crash into B1" fallthrough with an explicit preregistered
    framing so the reader can verify we didn't post-hoc invent a third
    claim tier after seeing the numbers.

  *Analyzer upgrades (same atomic commit series):*
  - **Permutation test** added to `scripts.pilot_analysis` alongside
    the asymptotic log-rank. At N=5 per arm the lifelines log-rank is
    an asymptotic chi-sq approximation whose coverage is known to
    drift under small-sample + heavy-censoring. The permutation test
    (10k label shuffles preserving per-arm sample sizes, computing the
    signed O-E numerator on each shuffle, one-sided p from the empirical
    tail) is exact under exchangeability and adds <1s to analyzer wall
    time. The asymptotic log-rank remains the §8-declared primary
    p-value; the permutation p is reported alongside as a robustness
    check. If they disagree by more than 0.05 at the headline N=20
    scale, the robustness disagreement itself is reported in the paper
    and the more conservative of the two is used for any post-hoc
    inference.
  - **Early-step return descriptors + AUC** added to the analyzer.
    §4 primary metric remains samples-to-mastery RMST. But the review
    raised a real concern: mastery threshold on MCC (~90/100) is
    plateau territory; the real transfer signal shows up in the first
    2–5k steps when the transferred prior is still load-bearing and
    the target trunk hasn't adapted out of distribution. New descriptive
    secondaries (bootstrap 95% CI, NOT §8-gating): (a) mean return at
    2k, 5k, 10k env-steps; (b) AUC(return, [0, 50k env-steps]). These
    are reported in the paper panel alongside RMST regardless of §8
    outcome. They are NOT post-hoc primary endpoints; the §8 gate is
    still RMST+log-rank.

  *Falsification / mechanism (more GPU, parallel to pilot #2):*
  - **§7 A10 adversarial-negative pair added** (Pendulum → Reacher):
    continuous→continuous, different physics class (pendular→robotic
    arm, no gravity-well dynamics). Predicted outcome: no transfer or
    anti-transfer. Reported regardless of direction. If A10 shows
    transfer parity with the primary pair, the "pendular physics
    cherry-pick" critique is real and the paper must reframe the
    claim scope. +15–20 GPU-h; runs as a separate `pilot_adversarial_
    run.json` during pilot #2 wind-down.
  - **§7 A11 GRU-shuffled ablation added** (2 seeds on primary pair):
    shuffles the transferred RSSM GRU weights before transfer, preserving
    spectral norm and total parameter mass but destroying the learned
    recurrent structure. If shuffled-GRU ≈ real-GRU on the primary
    metric, the "learned dynamics transfer" mechanism claim dies and
    the paper reduces to "transferring *any* initialization with the
    right spectral properties works." +5 GPU-h; runs atomically with
    A10.

  *What is explicitly NOT changed in v3.5:* §8 primary (RMST ≥ 1.3,
  p < 0.10, no anti-transfer pair, latent mode); §11 kill criteria
  at weeks 1/4/8; §5 Holm-Bonferroni on secondaries; §6 RSSM
  transferable-subset design; Band A/B/C post-pilot decision rule
  from Bug E v2/v3/v4. All review actions either clarify narrative
  or add falsification levers in directions the paper is already
  committed to report honestly.

  *Post-pilot backlog committed (NOT executed now, tracked to
  completion):* The 4-agent review also surfaced 5 items that are
  deferred until after pilot #2 completes but are committed work,
  not drop-ons. They live at `reviews/post_pilot_backlog.md` (POST-001
  through POST-006) with source-review attribution, blocking-phase
  gate, and effort estimate per item. The backlog file is the single
  source of truth for deferred post-pilot work; losing track of any
  entry there is a preregistration-integrity defect. Each item must
  complete, re-triage, or retire-with-rationale before the phase it
  gates begins.

- **2026-04-16 (v3.6 — post-pilot #2 chronology correction; self-audit
  triggered by devil's-advocate review):**
  Pilot #2 completed 2026-04-16 (40 runs: 3 pairs × 5 seeds × 2 arms +
  source pre-trainings). Post-completion review by 3 parallel reviewer
  agents (RL-methodology, devil's-advocate, paper-strategy) surfaced
  one integrity defect in the v3.5 preregistration text that must be
  corrected before any paper submission. This amendment resolves that
  defect. No pass-bar changes.

  *Defect:* The v3.5 §10 B0 clause "Why B0 is not a back-door around
  §8" contained the sentence: *"A reviewer can verify chronology via
  git history: B0 was committed pre-data (v3.5, before pilot #2
  unblinding), and the §8 threshold did not move."* The phrase
  "pre-data, before pilot #2 unblinding" is factually inaccurate. B0
  was committed at `4f8bb11` on 2026-04-15 13:28, approximately 6
  hours after pilot #2 launched (pilot log reconstructs start at
  07:34 the same day). At B0 commit time, the primary pair had 2 of
  5 seeds complete (42, 43) with a partial observed ratio of ~1.10;
  seeds 44, 45, 46 had not yet produced data. The seed that drives
  the final 1.238 ratio per leave-one-out analysis (46, LOO drop =
  1.049) was run ~5 hours after B0 commit.

  *Correction:* §10 B0 sentence has been rewritten to accurately
  reflect the chronology and the distinction between *pre-pilot*
  (§8 primary threshold, v3 on 2026-04-14, unchanged) and
  *pre-outcome* (B0 band edges committed before the N=5 RMST ratio
  was computable from 5 complete seeds). See `reviews/chronology_audit.md`
  for the full timeline reconstruction and adjudication.

  *Why this does NOT invalidate B0:* The B0 band edges [1.15, 1.30)
  were not tuned to the partial observed signal — at commit time the
  observed partial ratio was ~1.10, **below** the band's floor. The
  1.15 floor comes from commit `e24832c` (2026-04-15 03:28, before
  pilot launch) per the v3.4 Bug E v3 architecture review, based on
  noise-floor reasoning (RMST sampling SE at N=5 with MCC censoring,
  independent of observed data). The chronology breach is one of
  *phrasing integrity*, not of *data-driven band fitting*. The
  correction above makes the weaker-but-accurate integrity claim
  explicit.

  *Why §8 is unaffected:* §8's primary `ratio ≥ 1.30 AND p < 0.10`
  threshold was committed in v3 on 2026-04-14 (commit `28603ce`) —
  one day before pilot #2 launched, with no seed-level data of any
  kind in existence. The §8 threshold's pre-pilot status is
  intact. Only B0's fallback framing was committed mid-pilot.

  *v3.6 changes to preregistration content:*
  1. §10 B0 "Why B0 is not a back-door around §8" paragraph updated
     with accurate chronology and the pre-pilot-vs-pre-outcome
     distinction.
  2. This v3.6 amendment entry added to §13.
  3. New file `reviews/chronology_audit.md` committed with the full
     timeline reconstruction, intended for the paper's supplementary
     materials.

  *What is explicitly NOT changed in v3.6:*
  - §8 primary threshold (1.30 ratio, p < 0.10) — unchanged.
  - §10 B0 trigger clauses 1-4 — unchanged.
  - §10 B0 band edges [1.15, 1.30) — unchanged (the ∆ vs v3.5 is the
    paragraph's integrity phrasing, not the bands themselves).
  - Post-pilot backlog POST-001..POST-007 — unchanged.
  - §11 kill criteria — unchanged.

  *Corrective actions for paper submission* (now committed):
  - Include `reviews/chronology_audit.md` in supplementary materials.
  - If a reviewer raises the chronology: acknowledge directly, cite
    the audit, do not defend the old "pre-data" phrasing.
  - Surface the pre-outcome claim (not pre-pilot) in the methods
    section honestly.

  *Amendment trigger:* devil's-advocate agent review 2026-04-16;
  findings adjudicated in `reviews/chronology_audit.md`.

- **2026-04-17 (v3.7 — Band C extension N=10 on primary pair;
  pre-registered BEFORE launch of seeds 52-56).**

  *Context:* Band B rescue (seeds 47-51, warmup=200, primary pair
  cartpole→mcc) completed 2026-04-17 10:07. Results, from
  `pilot_bandb_results.json` via `scripts/pilot_analysis.py`:
  - RMST ratio (scratch/transfer) = **1.605** — above Band A
    threshold (≥ 1.30) and all Band B cells.
  - Log-rank p (one-sided, asymptotic) = **0.2402**.
  - Log-rank p (permutation, N=10,000) = 0.2585.
  - Mechanism: 5/5 transfer runs on `latent` mode, 5/5 loaded a
    crystallized skill. PASS.
  - Per-seed ratios (scratch_stm / transfer_stm): seed 47 = 0.98,
    seed 48 = 1.50, seed 49 = 3.29, seed 50 = 0.97, seed 51 = 1.57.
    3/5 positive, 2/5 neutral.
  - Leave-one-out: min ratio = 1.435 (drop seed 49), max = 1.671
    (drop seed 47). **No outlier-driven effect** — contrast with
    pilot #2 where LOO drop of seed 46 collapsed ratio from 1.238
    to 1.049.

  The ratio is strong and LOO-robust, but variance is high enough
  that p exceeds §8 threshold (0.10) and all Band B cells
  (max p = 0.20 in v4 cell 2). The §8 primary FAIL verdict holds
  strictly. Plan B0 as currently specified (ratio ∈ [1.15, 1.30))
  does NOT match the observation either — observed ratio 1.605 is
  above the B0 band.

  The finding is directionally consistent with transfer but
  statistically underpowered at N=5.

  *What changes in v3.7:* §10 gains a new **Band C** specification
  for a pre-registered N=10 extension on the primary pair only.
  The extension adds seeds 52–56 (5 fresh seeds) to the existing
  seeds 47–51 Band B pool, evaluated as a single N=10 analysis.

  **Band C pass spec (must satisfy ALL three):**
  1. RMST ratio (scratch/transfer) ≥ **1.30** computed on N=10
     pooled (seeds 47–56).
  2. Log-rank p (one-sided, BOTH asymptotic AND permutation
     N=10,000) < **0.10**.
  3. Leave-one-out minimum ratio ≥ **1.15** across all N=10 LOO
     draws (no single seed drives the result below the Band B
     floor).

  If all three pass → **§8 primary recovers as PASS** for the
  workshop paper via N=10 pre-registered extension. This is a
  stronger finding than Plan B0 (which expects modest-but-reliable
  1.15–1.30 ratio): Band C would land at Band-A-equivalent ratio
  with strict significance at N=10.

  **Band C kill spec (any triggers abandonment of workshop paper
  via primary pair):**
  - RMST ratio N=10 < **1.20**, OR
  - Log-rank p N=10 ≥ **0.20** (both asymptotic and permutation),
    OR
  - LOO minimum ratio < **1.00** (a single seed's removal inverts
    the direction).

  If any kill triggers → accept that the primary pair cannot
  support the workshop claim. Pivot to Post-1 horizontal scale
  (5–10 new tasks, §14 research_plan.md) and abandon the workshop
  submission, as committed in `reviews/research_directions.md` §6
  branch C.

  **Intermediate outcome** (ratio ∈ [1.20, 1.30) at p < 0.10, or
  ratio ≥ 1.30 at p ∈ [0.10, 0.20), or LOO min ∈ [1.00, 1.15)):
  triggers **Plan B0 modest paper** with Band C's N=10 numbers
  substituted for the original B0 analysis. Mechanism filters of
  §10 B0 (A11 ≥ 0.10 ratio gap, A7 scan, A9 cross-trajectory
  shuffle) remain required.

  *Chronology assertion (critical for integrity):*
  This amendment is committed **before** seeds 52–56 are launched.
  The seeds 47–51 data is fully unblinded at commit time (see
  ratio/p/LOO numbers above), but seeds 52–56 have NOT been
  trained. Reviewer can verify via:
  - `git log -1 preregistration.md` SHA and timestamp at the
    moment of this commit;
  - file mtime of `pilot_bandb_results.json` (2026-04-17 08:58,
    pre-amendment) containing only 5 seeds;
  - `run_overnight_bandc.sh` (created in same commit batch)
    specifying seeds 52–56 with base_seed=52.

  This is a weaker integrity claim than full pre-pilot (§8 at v3,
  2026-04-14, was pre-pilot for ALL seeds): Band C is **pre-
  seeds-52-56 but post-seeds-47-51**. The distinction matters for
  honest reporting and is the same kind of pre-outcome vs
  pre-pilot distinction as the B0 chronology correction in v3.6.

  *What is NOT changed in v3.7:*
  - §8 primary threshold (1.30 / p<0.10) — unchanged; Band C
    satisfies §8 strictly if it passes.
  - §10 B0 and §10 B1 — unchanged; Band C sits above B0 as a
    strengthening path.
  - §11 kill criteria — unchanged; Band C kill triggers activate
    §11's Post-1 pivot clause.
  - Post-pilot backlog POST-001..POST-007 — unchanged.
  - Mechanism requirements (A11, A7, A9 per §10 B0 clause 3) for
    intermediate Plan B0 path — unchanged.

  *Budget:* ~10 GPU-hours wall (5 seeds × ~2 hours/seed incl.
  cartpole source crystallization + scratch mcc + transfer mcc).
  Comparable to pilot #2 (12.65 GPU-hr at N=5). Executed as
  `run_overnight_bandc.sh`, logged to `pilot_bandc.log`, results
  merged into `pilot_bandc_results.json` separately from existing
  `pilot_bandb_results.json` to preserve audit trail.

  *Corrective actions for paper submission:*
  - Report Band C result (pass, intermediate, or kill) honestly
    with full N=10 per-seed ratios, LOO table, p-values from both
    asymptotic and permutation tests.
  - Include this amendment text and the timestamped commit SHA
    in supplementary materials so the pre-seeds-52-56 chronology
    is independently verifiable.
  - Do NOT cherry-pick: the N=10 analysis reported is the ONLY
    analysis reported for the primary pair in the paper's headline
    table. No re-analysis with different seed subsets.

  *Amendment trigger:* Band B rescue underpowered at N=5
  (ratio 1.605 strong, p=0.24 fails §8); decision to proceed via
  pre-registered N=10 extension over either accepting weak B0 path
  or abandoning paper. See `reviews/research_directions.md` §6 for
  full branch A/B/C decision tree.

- **2026-04-18 (v3.8 — Band C KILL: all three kill criteria triggered;
  primary-pair workshop path officially abandoned; branch C pivot to
  Post-1 exploration program activated per §10 / §11 / `reviews/
  research_directions.md` §6).**

  *Context:* Band C N=10 extension (seeds 47–56 pooled on primary pair
  cartpole→mcc) completed 2026-04-18 ~00:10. Results, from merged
  `pilot_bandc_n10_merged.json` via `scripts/pilot_analysis.py`:

  - RMST ratio (scratch/transfer, N=10 pooled) = **1.036**
  - Log-rank p (one-sided, asymptotic) = **0.510**
  - Log-rank p (permutation, N=10,000) = **0.516**
  - Mechanism: 10/10 transfer runs on `latent` mode, 10/10 loaded
    a crystallized skill. Mechanism check PASS.
  - Leave-one-out minimum ratio = **0.871** (dropping seed 51).
  - Per-seed ratios: seed 47 = 0.979, 48 = 1.503, 49 = 3.285,
    50 = 0.970, 51 = 1.573, 52 = 1.020, 53 = 1.000, 54 = 1.913,
    **55 = 0.328 (anti-transfer)**, 56 = 0.995. Distribution:
    4 positive / 5 neutral / 1 strongly negative.

  *Verdict against v3.7 pre-registered criteria:* all three Band C
  kill criteria are satisfied simultaneously:

  | Kill criterion | Threshold | Observed | Triggered |
  |---|---|---|---|
  | Ratio < 1.20 | < 1.20 | 1.036 | YES |
  | Log-rank p ≥ 0.20 (both tests) | ≥ 0.20 | 0.510 / 0.516 | YES |
  | LOO minimum ratio < 1.00 | < 1.00 | 0.871 | YES |

  *Decision per v3.7 pre-registered clause:* "If any kill triggers →
  accept that the primary pair cannot support the workshop claim.
  Pivot to Post-1 horizontal scale (5–10 new tasks, §14
  research_plan.md) and abandon the workshop submission." All three
  triggered, therefore: **workshop-paper-on-primary-pair path is
  officially abandoned as of this commit.** Branch C of the decision
  tree (`reviews/research_directions.md` §6) is activated.

  *Scientific reading.* The specific mechanism tested — shape-checked
  transferable-subset loading of a Dreamer-RSSM's dynamics modules
  across the discrete↔continuous action-space-type boundary, with
  the policy switched to `acting_policy_mode=latent` — does not
  produce a reliable transfer benefit on the primary pair at N=10.
  Band B's N=5 signal (ratio 1.605) was high-variance seed lottery,
  exactly the failure mode documented in Henderson et al. 2018 and
  Agarwal et al. 2021. **The hypothesis is falsified at N=10 on the
  most favorable pair in the preregistered matrix.** The primary
  pair (CartPole→MountainCar-Continuous) shares physics class,
  observation dim, and action semantics more closely than any other
  pair imaginable within gym classic-control + MCC; the null result
  here is strong evidence that naive subset loading is insufficient,
  not merely that adversarial pairs are harder.

  *What remains scientifically valuable.*
  - **The negative result itself** is publishable as a rigorous
    falsification in a negative-results report or workshop negative-
    results track.
  - **The research program (Q1/Q2/Q3)** of `reviews/
    research_directions.md` is not invalidated; if anything, it is
    strengthened. Q1 (physics-grounded world models via contrastive /
    disagreement-weighted objectives) becomes higher-priority because
    the reconstruction-based RSSM has now empirically failed the
    cross-action-type transfer test. Q2 (contextual skill selection)
    and Q3 (transfer acceleration beyond `load_state_dict`) remain
    open and are justified by this null baseline.
  - **The methodology** (preregistration + amendment chronology +
    multi-agent reviews + chronology audit + tests) is a publishable
    contribution independent of scientific outcome, per the
    unconditional blog-post commitment in `docs/compute_application/
    research_proposal.md` §5.

  *What is NOT changed in v3.8:*
  - §8 primary threshold (1.30 / p<0.10) — not changed, it was
    simply not met.
  - §10 B0 / B1 fallback paths — B0 was above the observed ratio
    anyway; B1 (negative-results framing) is now the active path
    for any publication mentioning the primary pair.
  - §11 kill criteria — not changed; v3.7 Band C kill triggers are
    a refinement, not a relaxation, of §11.
  - Post-pilot backlog POST-001..POST-007 — unchanged; some items
    (notably POST-005 A7/A9/A10/A11 ablations) are now deferred
    rather than workshop-blocking, because no workshop is being
    written on the primary pair.

  *Next steps committed as of v3.8 amendment:*
  1. `docs/compute_application/research_proposal.md` §2 updated
     with N=10 final numbers and branch-C-active narrative
     (same commit batch as this amendment).
  2. `reviews/research_directions.md` updated: branch C active,
     Q1/Q2/Q3 exploration becomes the program's operational path.
  3. TRC application submitted in the honest-pivot framing: the
     compute is requested not to prove the falsified hypothesis
     but to explore the three-question research program with the
     larger skill library, architectural variants, and ablations
     that this negative result makes scientifically necessary.
  4. Blog post "Preregistering against yourself" drafted within
     two weeks, now with the added concrete case study of a
     preregistered kill criterion actually triggering and being
     honored.

  *Amendment trigger:* Band C N=10 analysis 2026-04-18 ~00:30;
  verdict matches v3.7 kill specification; decision follows the
  pre-committed branch-C clause without deviation.

- **2026-04-18 (v3.9 — preregister Q1/Q2/Q3 operational thresholds
  before any TPU-hour is spent on branch-C exploration).**

  *Context:* v3.8 activated branch C (Q1/Q2/Q3 exploration program
  per `reviews/research_directions.md` §6). A 4-agent adversarial
  review on 2026-04-18 before TRC submission flagged that Q1/Q2/Q3
  exist as prose in `reviews/` but not as pre-registered experiments
  with numerical thresholds — a legitimate critique. This amendment
  promotes the Q1/Q2/Q3 thresholds from `reviews/research_directions.md`
  §6 prose to official preregistration status, *before* any
  TRC-allocated TPU compute is spent on them.

  **Q1-C (contrastive RSSM + disagreement-weighted ensemble) — Sprint 1.**
  - Setup: replace reconstruction loss with contrastive latent prediction
    (InfoNCE or BYOL-style, decoder weight ≤ 0.1) on existing
    `EnsembleRSSMCore`. Re-run primary pair CartPole→MountainCar-
    Continuous at N=5.
  - Pass: RMST ratio ≥ **1.20** AND log-rank one-sided p < **0.20**
    AND LOO min ≥ **1.05**. Interpretation: contrastive RSSM produces
    a detectable improvement over the falsified Band C baseline
    (ratio 1.036).
  - Kill: RMST ratio ≤ **1.05** OR LOO min < **0.95**. Interpretation:
    contrastive objective does not help; Q1-C is rejected as a path
    and the next sprint pivots to Q1-B (Hamiltonian) or Q3.
  - Intermediate (ratio ∈ (1.05, 1.20) or p ∈ (0.20, 0.40)):
    requires multi-agent review before deciding whether to extend
    to N=10 or abandon.
  - Budget: ~15 TPU-hours (Month 1 allocation).

  **Q3-B (EWC-protected RSSM subset loading) — Sprint 2.**
  - Setup: repair the broken API of `ragnarok/learning/ewc.py`
    (currently un-imported, signature misaligned with RSSM trainer);
    compute Fisher diagonal on crystallized skill; apply EWC penalty
    during target-task RSSM subset training. Re-run primary pair at
    N=5 with EWC-protected transfer.
  - Pre-check (Day 1 of sprint): Fisher max/median on crystallized
    cartpole skill ≥ **10**. If Fisher is flat (max/median < 10),
    EWC cannot distinguish important weights; **skip** this sprint
    and reallocate budget to Q1-B or horizontal scale.
  - Pass: RMST ratio ≥ **1.20** AND log-rank p < **0.20** AND LOO
    min ≥ **1.05** over the falsified Band C baseline.
  - Kill: RMST ratio ≤ **1.05** OR LOO min < **0.95**.
  - Budget: ~15 TPU-hours (Month 1-2 allocation).

  **Q2 horizontal scale (Post-1 skill-library extension) — Sprints 3-4.**
  - Setup: add 7 source-target pairs spanning DMControl (cheetah,
    walker, hopper, quadruped) and MetaWorld (reach, pick-place,
    button-press). Each pair run at N=5 scratch + N=5 transfer with
    nearest-centroid skill selector from expanded library.
  - Pass condition for "cross-action-type transfer works": **at
    least 3 of 7 new pairs** must exhibit RMST ratio ≥ **1.15**
    directionally (log-rank p < 0.30 at N=5).
  - Kill condition for the entire research thesis: **0 or 1 of 7**
    new pairs exhibit ratio ≥ 1.10. In that case, "skill reuse via
    shared RSSM trunk" is considered empirically unsupported as a
    family of approaches, not just on the primary pair. Project
    then either (a) pivots to a completely different skill-
    representation paradigm, or (b) enters wind-down.
  - Intermediate: 2 of 7 positive — extend to N=10 on those 2 pairs
    before decision; requires multi-agent review.
  - Budget: ~40 TPU-hours (Months 2-3 allocation).

  **Q2 contextual selection (PEARL-style encoder) — Sprint 5, conditional.**
  - Runs **only if** Q2 horizontal scale yields ≥ 3 positive pairs
    (providing a non-trivial skill library to select from).
  - Setup: compare nearest-centroid skill selection (current
    baseline) against a PEARL-style context encoder trained
    post-hoc on the pooled pilot data.
  - Pass: context encoder reduces episodes-to-mastery by ≥ **15%**
    averaged over the positive pairs, at N=5 per pair.
  - Kill: context encoder performs within ±5% of centroid baseline.
  - Budget: ~10 TPU-hours (Month 3 allocation, conditional).

  **Q3-A kickstarting (decaying-coefficient distillation) — Sprint 6,
  conditional.**
  - Runs **only if** Q3-B EWC sprint shows directional signal
    (ratio ≥ 1.10) or if horizontal scale yields positive pairs to
    test kickstarting on.
  - Thresholds identical to Q3-B structure but with kickstarting
    coefficient schedule as the intervention.
  - Budget: ~10 TPU-hours (Month 3+, conditional).

  **Multi-skill composition (POST-007) — deferred.**
  - Requires ≥ 10 skills in library to be empirically tractable.
  - Runs only if Sprints 3-4 deliver at least 7 additional skills.
  - Full specification deferred to a future v3.10 or v3.11
    amendment written before the sprint launches. Listed here only
    to document the intended research trajectory.

  **Meta-kill criterion for the entire Ragnarok research thesis.**
  If, after Sprints 1-4 complete (Q1-C + Q3-B + horizontal scale),
  **none** of the three paths produces directional signal per the
  thresholds above, the thesis that "skills can be transferred
  across tasks via shared neural modules in the Dreamer-RSSM family"
  is considered not-supported by the empirical evidence. Decision
  at that gate:
  - (a) pivot to a radically different skill-representation
    paradigm (e.g. program synthesis, language-grounded skills,
    foundation-model-based agents) via a new preregistration
    amendment that declares the prior framing dead; OR
  - (b) enter project wind-down and publish a comprehensive
    negative-results report without continuing compute use.
  The multi-agent review at this gate must explicitly ask "are we
  ignoring data?" and "are we persisting out of sunk cost?". A
  minimum of 2 of 4 reviewers must vote for option (b) or the
  decision defaults to continuation via (a).

  *Chronology assertion (critical for integrity):*
  This amendment is committed **before** any TRC TPU-hour has been
  spent and **before** any of the Q1/Q2/Q3 sprints launch. The
  thresholds above are therefore pre-outcome for every Sprint
  whose kill/pass they govern. The commit SHA of v3.9 will be
  cited in every future sprint-result amendment as "thresholds
  pre-registered at SHA X".

  *What is NOT changed in v3.9:*
  - §8 primary threshold — remains as originally committed; the
    primary-pair hypothesis is already falsified at N=10.
  - §10 B0 / B1 — B0 is moot (ratio observed was above the band);
    B1 (negative-results framing) remains available for any
    publication mentioning the primary pair.
  - §11 kill criteria — not changed; the meta-kill above is a
    *program-level* kill, layered on top of §11's project-level
    kill.
  - v3.8 branch-C activation — not changed; v3.9 preregisters the
    operational details of the already-activated branch C.

  *Amendment trigger:* 4-agent pre-TRC-submit review 2026-04-18
  (`reviews/pre_trc_submit_4agent_review_2026-04-18.md`) explicitly
  flagged the absence of preregistered Q1/Q2/Q3 thresholds as a
  credibility gap. This amendment addresses the gap before
  submission and before compute is spent.

- **2026-05-19 (v3.10 — device-path transfer confound identified; the
  transfer mechanism is corrected; the scratch-vs-transfer comparison must
  hold the learner fixed).**

  *Context.* A 3-agent adversarial review on 2026-05-19 challenged the
  Phase-2 device-path "recalibration" (re-running cartpole→MCC transfer on
  the accelerator-resident reimplementation). The challenge was verified
  directly against the code (`device_agent.py`, `agent.py`, `pilot_run.py`).

  *Finding 1 — the device recalibration is confounded; abandoned.* In the
  device path, `DeviceAgent.load_snapshot` flips `acting_mode="latent"`:
  the transfer arm then collects via `collect_rollout_latent`, trains only
  the latent policy + world model, and is evaluated through the latent
  policy — a single-pass, single-minibatch, unclipped on-policy A2C. The
  scratch arm trains and is evaluated through SAC (off-policy, replay,
  ~512 updates/rollout). The two arms run different RL algorithms, so the
  scratch/transfer ratio cannot isolate the transfer effect from the
  SAC-vs-A2C learner gap. The device recalibration is abandoned (no
  further seeds).

  *Finding 2 — the v3.8 gym falsification is NOT confounded and stands.*
  Verified separately: the gym pilot (`pilot_run.py`) trains both arms via
  `train_policy_real → _train_sac`; `acting_policy_mode` affects only
  `collect_episode`, which the pilot loop never calls. The gym
  cartpole→MCC arms both ran SAC (collect, train, eval). v3.8's N=10
  falsification is therefore a clean SAC-vs-SAC negative and is not
  reopened by this amendment. What it falsified, restated precisely:
  *warm-starting the env-agnostic RSSM core from the source skill produces
  no transfer benefit on cartpole→MCC.*

  *Finding 3 — both transfer mechanisms tested to date are weak channels.*
  The gym mechanism warm-starts the RSSM core, but SAC reads raw
  observations — the transferred core reaches the measured learner only
  indirectly (dream training, latent curiosity). The device mechanism
  routes transfer through a separate weak latent A2C. Neither has tested
  the strong-learner / strong-channel combination: the task's real learner
  directly consuming the transferred representation.

  *Corrected transfer mechanism (preregistered).* From this amendment
  onward, transfer experiments hold the learner fixed and give transfer a
  direct channel:
  - Both arms (scratch, transfer) use the same task learner — SAC for
    continuous control, the standard discrete learner for discrete
    control. The transfer arm never swaps to the latent policy as its
    actor.
  - The learner's actor and critic read an augmented observation
    `[obs, h, z]` — the raw observation concatenated with the RSSM latent
    state.
  - The only difference between the two arms is the RSSM env-agnostic
    core's initialisation: the transfer arm warm-starts it from the source
    skill; the scratch arm initialises it fresh. Architecture, learner,
    optimiser, eval — identical. This isolates the transfer effect.
  - The transferred core trains under the existing post-transfer LR warmup
    (`set_transferable_lr_scale`).
  - `acting_policy_mode="latent"` / the latent-policy-as-actor path is
    retired as a transfer vehicle (code retained, no longer used for
    transfer).

  *Corrected metric (preregistered).* Single-episode "first eval ≥
  threshold" is replaced by a ≥10-episode averaged eval; the primary
  endpoint is the sample-efficiency AUC of the smoothed return-vs-env-steps
  curve over the fixed budget (transfer vs scratch). RMST / log-rank
  remains a secondary endpoint where mastery is reached.

  *Corrected task pair (preregistered).* cartpole→MCC is retired as a
  transfer pair: v3.8 cleanly established it has no transferable structure
  (physics-dissimilar — balance / dense reward vs energy-pumping / sparse
  reward). Transfer experiments use pairs that share dynamics —
  graded-difficulty within one dynamics family, where a world-model core
  is genuinely transferable. The first corrected experiment validates the
  mechanism end-to-end on one such pair at N ≥ 3 on the available GPU; the
  exact pair is named in the implementing commit.

  *Relation to the v3.9 Sprint program.* v3.9's Sprints (Q1-C, Q3-B,
  horizontal scale) are not cancelled. Every Sprint inherits the corrected
  mechanism, metric, and pair-selection principle above; v3.9 Sprint
  thresholds are reinterpreted against the AUC endpoint and restated at
  each Sprint's launch amendment. The v3.9 meta-kill criterion is unchanged
  and now governs the corrected Sprints.

  *What is NOT changed in v3.10:* §8 / §11; the v3.9 meta-kill; v3.8's
  falsification (which stands, restated above); the device-path
  infrastructure (built and validated — retained and parked, not deleted;
  the intended compute path for the horizontal-scale Sprints once a working
  method exists to scale).

  *Chronology assertion.* This amendment is committed **before** the
  corrected mechanism is implemented and **before** any corrected-mechanism
  run. The corrected mechanism, metric, and pair-selection principle above
  are pre-outcome.

  *Amendment trigger:* 3-agent adversarial review 2026-05-19; the device
  confound verified in `device_agent.py`; the gym non-confound verified in
  `pilot_run.py` + `agent.py`.

- **2026-05-19 (v3.11 — concurrent RSSM training is incompatible with
  off-policy SAC replay; the representation is frozen for the SAC arm).**

  *Flaw found.* The v3.10 corrected mechanism was implemented (commit
  b1d60ae) and a run started. Its first phase — training the SOURCE agent
  on standard MountainCarContinuous, SAC reading the 162-d augmented
  observation [obs, h, z] — FAILED to learn. Over 60 rollouts / 1.97M
  env-steps the eval return never rose above ~6 and ended negative; a
  learned MCC agent evals ~+90. The blip-then-collapse signature is a
  training instability, not slow learning.

  *Cause.* SAC is off-policy: it learns from a 200k-transition replay
  buffer. v3.10 stores the augmented observation — which contains the
  RSSM latent (h, z) — directly in that buffer, while the RSSM trains
  concurrently. A transition collected at rollout 5 carries a latent
  produced by the rollout-5 RSSM; a transition from rollout 40 carries a
  rollout-40 RSSM's latent; they are sampled into the same minibatch. The
  same physical state therefore maps to different 162-d vectors depending
  on when it was collected, so the critic's bootstrap target
  Q(s,a) <- r + gamma*Q(s',a') is regressed across a non-stationary
  representation and cannot converge. (DreamerV2/V3 avoid exactly this by
  never replaying stored latents — the world model re-encodes raw
  observations with its current weights.) The proven device baseline
  confirms the locus: device_recalibration.py's scratch arm — the same
  SAC, same curiosity, same concurrently-trained RSSM, but reading the
  RAW 2-d observation (DeviceAgent constructs SAC with obs_dim equal to
  the env's raw obs_dim and collects on raw obs) — masters standard MCC
  at 590k env-steps. The only change between the working baseline and the
  failing run is SAC's input: a stationary 2-d obs vs a non-stationary
  162-d [obs, h, z].

  *Diagnosis discipline.* The diagnosis was put to a 3-agent adversarial
  review. One agent challenged it, asserting the recalibration baseline
  also read the augmented obs (which would make the failure mere
  variance). That assertion was checked directly against device_agent.py
  and found false — DeviceAgent's SAC is constructed with obs_dim equal
  to the env's raw obs_dim and collects on raw obs; it is a genuine
  raw-obs baseline. The diagnosis stands.

  *Corrected mechanism (preregistered).* The RSSM representation is
  FROZEN during the SAC arm. The world model is trained first (for the
  source skill, by the proven raw-obs DeviceAgent path); its env-agnostic
  core is then frozen, and SAC trains on the now-stationary
  [obs, h_frozen, z_frozen]. With a frozen RSSM the replay buffer is
  representation-consistent and the off-policy critic is well-posed. This
  supersedes exactly one clause of v3.10 — "the transferred core trains
  under the post-transfer LR warmup"; the core no longer trains during
  the SAC arm, so that LR-warmup machinery is unused. Every other v3.10
  element is retained: same learner (SAC), same augmented [obs, h, z]
  interface, same optimiser and eval, and the sole inter-arm difference
  remains the RSSM core initialisation.

  *Arms.* scratch — RSSM core fresh-initialised, frozen. transfer — RSSM
  core warm-started from the source snapshot, frozen. The latent z fed to
  SAC is the posterior mean (deterministic), so the frozen representation
  is a fixed function of the observation history.

  *Scope — what a Stage-1 result does and does not show (preregistered).*
  The scratch arm's core is fresh-initialised — v3.10's "scratch
  initialises fresh", unchanged. This first experiment therefore compares
  a SOURCE-trained representation against a fresh/random one. It is not a
  tautology: SAC reads the raw observation in BOTH arms, and raw-obs SAC
  alone masters MCC (the 590k baseline), so the scratch arm is a fully
  capable learner and its frozen random core is at worst ignorable noise.
  A positive Stage-1 result (transfer AUC > scratch AUC) establishes that
  a transferred world-model core measurably accelerates SAC through the
  augmented-obs channel. It does NOT, by itself, separate "the source
  skill specifically transferred" from "any trained world model would
  help". That separation is the preregistered Stage-2 follow-up, run only
  if Stage 1 is positive: a control whose RSSM is pretrained on the
  TARGET task to convergence and then frozen, with its target-task
  pretraining env-steps charged to the sample-efficiency AUC budget. Stage
  1 is run first because it is the cheapest falsification and carries no
  representation-pretraining chicken-and-egg or budget-accounting
  complexity; a null Stage 1 ends the line and makes Stage 2 moot.

  *Pair, metric, budget.* Unchanged from v3.10: source = standard
  MountainCarContinuous, target = MountainCarContinuous-Hard (weaker
  engine, shared dynamics family); endpoint = sample-efficiency AUC of the
  >=10-episode-averaged eval-return vs env-steps curve, transfer vs
  scratch, N>=3 seeds. Both Stage-1 arms spend their entire target
  env-step budget on SAC (neither does target-task representation
  pretraining), so the AUC x-axis is directly comparable.

  *What is NOT changed in v3.11:* §8 / §11; the v3.9 meta-kill; v3.8's
  falsification; the device-path infrastructure; v3.10's Findings 1-3 and
  its corrected-pair principle. Only v3.10's "core trains under LR warmup"
  clause is superseded.

  *Chronology assertion.* This amendment is committed BEFORE the v3.11
  implementation and BEFORE any v3.11 run. The frozen-representation
  mechanism and the Stage-1/Stage-2 staging are pre-outcome.

  *Amendment trigger:* the v3.10 source-run learning failure; 3-agent
  adversarial review 2026-05-19; the off-policy / non-stationary-latent
  diagnosis verified in sac.py + device_agent.py.

- **2026-05-19 (v3.12 — the v3.11 Stage-1 result is confounded by latent
  scaling; the augmented observation is normalized and a permuted-core
  control is added).**

  *Flaw found (adversarial review of the v3.11 run).* The v3.11 Stage-1
  run completed — 3 seeds, transfer AUC > scratch AUC in all 3, results
  committed — and was put to a 3-agent adversarial review. The review
  found a genuine confound. SAC reads the augmented observation
  cat([obs, h, z]); only the raw `obs` block is scale-normalised
  (DeviceRunningNormalizer) — the 160-d RSSM latent block [h, z] is fed
  to SAC's MLP un-normalised. A frozen *trained* RSSM core and a frozen
  *random* core emit [h, z] of materially different magnitude and
  conditioning. SAC's plain-MLP actor/critic optimise an un-normalised
  input at an effective rate that depends on that scale, so the v3.11
  "transfer > scratch" gap is confounded: it cannot be separated into
  "the transferred core carries useful learned structure" vs "the
  transferred core merely emits better-scaled features for an MLP". The
  v3.11 Stage-1 result therefore does NOT cleanly establish
  representational transfer and is superseded.

  *Secondary review findings (also corrected here).* (i) The v3.11
  script reported a transfer/scratch AUC *ratio*; with a near-zero,
  high-variance denominator (one scratch seed had AUC 0.77) the ratio is
  a pathological estimator (per-seed ratios 3.25 / 8.02 / 32.09). The
  registered endpoint is the AUC *difference*; v3.12 reports the
  difference (the v3.11 per-seed transfer-minus-scratch gap, 26.7 / 24.2
  / 24.0, was in fact strikingly consistent). (ii) The registered metric
  is the *smoothed* return-vs-env-steps curve; v3.11's AUC applied no
  smoothing. v3.12 pre-specifies the smoothing: a 3-point moving average
  of the per-iteration eval curve before trapezoidal integration.
  (iii) N=3 is underpowered for the observed bimodal scratch baseline
  (it mastered in 1/3 seeds, stayed flat in 2/3); v3.12 uses N=8.

  *Corrected design (v3.12).* Retained from v3.11: the frozen-RSSM
  mechanism — validated, frozen-RSSM augmented SAC learns MCC-Hard,
  which the v3.10 concurrent-RSSM version could not — SAC as the fixed
  learner, the RSSM-independent ICM curiosity, the standard-MCC ->
  MCC-Hard pair. Changed:
  - The augmented observation cat([obs, h, z]) is passed through a
    running normaliser before SAC (every arm), so all arms feed SAC a
    comparably-scaled input and the scale channel is closed.
  - A third control arm — `permuted` — is added: the source RSSM core
    with the weights of each parameter tensor randomly permuted. This
    preserves the trained core's weight distribution, per-layer norm and
    rank statistics (hence its [h,z] scale) while destroying its learned
    structure. The decisive test of representational transfer is
    transfer vs permuted: a transfer>permuted gap is attributable to
    learned structure, not scale. transfer vs scratch (fresh-random
    core) is retained as the v3.11-comparable baseline.
  - N=8 seeds; primary endpoint = the per-seed AUC *difference*
    transfer-minus-permuted (and transfer-minus-scratch), reported as a
    mean with a CI over the 8 seeds.

  *Claim ceiling (unchanged from v3.11).* Even a clean v3.12 positive
  (transfer > permuted) establishes only that the source-trained core's
  learned structure accelerates SAC — it still does not separate
  source-skill-specific knowledge from generic target-usable world-model
  structure. The target-pretrained-core control (Stage 2) remains the
  preregistered conditional follow-up, gated on a positive de-confounded
  v3.12.

  *Chronology assertion.* This amendment is committed BEFORE the v3.12
  implementation and BEFORE any v3.12 run. The augmented-obs
  normalisation, the permuted-core control, N=8, and the difference
  endpoint with the specified smoothing are pre-outcome.

  *Amendment trigger:* 3-agent adversarial review of the v3.11 Stage-1
  run, 2026-05-19 — the latent-scaling confound; the ratio-statistic and
  curve-smoothing deviations; the N=3 power shortfall.

- **2026-05-20 (v3.13 — the latent must be load-bearing for transfer to
  be detectable; SAC reads latent-only and the full RSSM is transferred).**

  *Flaw found (interpretation of the v3.12 null).* v3.12's clean null
  (transfer-minus-permuted AUC +2.68 +/- 13.87, N=8) is consistent with a
  simple mechanistic explanation: SAC reads cat([obs, h, z]) and the raw
  2-d obs is a complete state for MountainCarContinuous-Hard, so SAC
  solves the task using just the obs and learns to weight the 160-d
  latent block toward zero. The transferred representation has nothing
  load-bearing to do, so its quality cannot be measured. Compounding
  this, the encoder (per-env, fresh-random in every arm under
  v3.11/v3.12) feeds the core a random input regardless of core
  training, further degrading any signal the transferred core could
  carry.

  *Corrected mechanism (v3.13).*
  - SAC reads ONLY the latent cat([h, z]) — no raw observation in the
    actor or critic input. The latent is therefore load-bearing: a
    learner that cannot use it cannot solve the task. This is the
    "strong channel" the v3.10 amendment called for — but with the same
    learner (SAC) across all arms, not the retired latent-A2C path.
  - The transferable subset is broadened from the env-agnostic core to
    the FULL RSSM state dict. MCC and MCC-Hard share obs_dim, action_dim
    and reward structure, so the encoder, decoder, reward-predictor and
    continue-predictor are all shape-compatible. Transferring everything
    lets the transferred representation produce a meaningful target
    latent (a fresh-random encoder + a transferred core would still feed
    the core noise, denying the core's learned dynamics any real state
    to operate on).

  *Controls and metric (retained from v3.12).* Three arms — transfer
  (the full source RSSM, frozen), permuted (each parameter tensor of the
  source RSSM randomly permuted: scale/rank/distribution-matched,
  learned structure destroyed, frozen), scratch (fresh random RSSM,
  frozen). The augmented-obs normaliser is over the 160-d latent only.
  Same RSSM-independent ICM curiosity. N=8 seeds; primary endpoint =
  the per-seed AUC difference transfer-minus-permuted (and
  -minus-scratch), mean +/- Student-t 95% CI; curve smoothed by 3-point
  moving average before trapezoidal integration.

  *Heterogeneous-dim relation.* v3.13 is preregistered for SAME-obs-dim
  transfer (MCC -> MCC-Hard). The heterogeneous-dim transfer claim
  (e.g. cartpole -> mcc) cannot transfer the encoder and is the
  conditional follow-up: only if v3.13 shows working transfer is it
  worth probing whether the env-agnostic core alone transfers across
  obs_dim boundaries.

  *Decisive interpretation.* If v3.13 is positive (transfer > permuted,
  CI excludes zero), the representation-transfer mechanism is alive and
  the program proceeds to probe the heterogeneous-dim and core-only
  cases. If v3.13 is also null under this maximally-favourable design
  (forced latent dependence + full-RSSM transfer + the proven freeze
  fix + the v3.12 scale control), the representation-transfer line on
  these toy continuous-control tasks is concluded null and the project
  pivots its mechanism class.

  *Chronology assertion.* This amendment is committed BEFORE the v3.13
  implementation and BEFORE any v3.13 run. The latent-only mechanism,
  the full-RSSM transferable scope, the decisive interpretation, and the
  same-vs-heterogeneous staging are pre-outcome.

  *Amendment trigger:* the v3.12 clean null (commit dacb4ff) and the
  observation that the augmented [obs, h, z] mechanism gives SAC a
  raw-obs fallback that nullifies any transferred-latent benefit on
  fully-observable tasks.

- **2026-05-20 (v3.14 — does the env-agnostic core ALONE carry the v3.13
  transfer effect? Required stepping stone toward heterogeneous-dim).**

  *v3.13 outcome (positive, marginal).* The v3.13 run (commit e8bd009)
  came out positive on the decisive comparison: transfer-minus-permuted
  AUC mean +15.67 +/- 15.65 (N=8, CI excludes zero by ~0.02), with 7/8
  seeds favouring transfer and end-of-run mastery rates of 6/8 (transfer)
  vs 2/8 (permuted) vs 0/8 (scratch). The hypothesis that v3.12's null
  was caused by SAC's raw-obs fallback rendering the latent irrelevant
  is supported: forcing latent dependence (no raw obs) and giving the
  latent real content (full-RSSM transfer including encoder) makes the
  transfer effect detectable for the first time in the project.

  *Question.* v3.13's transferable scope was the FULL RSSM state dict —
  encoder + env-agnostic core + decoder + heads. The workshop claim is
  about transferable representational structure in the env-agnostic core
  (gru + prior + posterior), which is the only subset that survives a
  heterogeneous-dim transfer (where source/target obs_dims differ, so
  the encoder cannot transfer). v3.14 probes the necessary preliminary:
  on the same-dim MCC -> MCC-Hard pair, does the core ALONE carry the
  v3.13 effect, or was the encoder transfer essential?

  *v3.14 design.* Identical to v3.13 except the transferable scope
  reverts to the v3.11/v3.12 subset — the env-agnostic core only
  (gru, prior, posterior). The encoder, decoder, reward and continue
  predictors are fresh-random in EVERY arm (transfer, permuted, scratch).
  Mechanism (latent-only SAC, freeze, aug-normalizer, ICM curiosity,
  3-arm controls) and metric (per-seed AUC difference, 3-point smoothed,
  Student-t 95% CI) are unchanged. N=8. The source RSSM core is the
  v3.11 source snapshot (`transfer_v311_out/source_snapshot.pt` — the
  same source training as v3.11/v3.12).

  *Decisive interpretation.*
  - v3.14 positive (transfer > permuted, CI excludes 0): the env-agnostic
    core ALONE carries transferable knowledge. Heterogeneous-dim transfer
    (cartpole -> mcc with core-only transfer) becomes the natural next
    test — this is the workshop's heterogeneous-dim claim.
  - v3.14 null: the encoder transfer was essential to v3.13. The
    heterogeneous-dim mechanism (which cannot transfer the encoder) is
    unlikely to work as currently designed; the project then either
    accepts the same-dim positive as the workshop claim, or pivots to
    mechanisms that carry encoder-equivalent information across obs_dim
    boundaries (e.g. shared abstract state, dimension-aligning probes).

  *Chronology assertion.* This amendment is committed BEFORE the v3.14
  implementation and BEFORE any v3.14 run. The core-only transferable
  scope is pre-outcome.

  *Amendment trigger:* the v3.13 marginal positive (commit e8bd009) and
  the question of whether the env-agnostic core alone carries the
  effect — the necessary preliminary to the workshop's heterogeneous-dim
  claim.

- **2026-05-20 (v3.13/v3.14 extension to N=16 — firm up both marginal
  results before pivot decisions).**

  *Why.* v3.13 (full-RSSM transfer) came out marginally positive at N=8:
  transfer-minus-permuted mean +15.67, 95% CI +/-15.65, lower bound +0.02.
  v3.14 (core-only transfer) came out marginal/null at N=8: mean +7.64,
  95% CI +/-9.61, lower bound -1.97. Both CIs are wide enough that
  doubling N could meaningfully tighten the verdict on either result.
  Crucially, the two outcomes drive different downstream decisions
  (v3.13 firm-positive => workshop has a clean same-dim result; v3.13
  null => the project's only positive collapses; v3.14 firm-positive =>
  heterogeneous-dim becomes viable; v3.14 null => the core-only
  transferable subset is too weak and the workshop's heterogeneous-dim
  claim is dead) — so investing in tighter CIs is high-leverage before
  any pivot.

  *What.* Extend both v3.13 (full-RSSM) and v3.14 (core-only) from N=8
  to N=16 seeds. The scripts are resume-aware: they append seeds 8-15 to
  the existing seeds 0-7 in results.json without re-running the first
  eight. All other design parameters (latent-only SAC, freeze,
  aug-normalizer, 3 arms, 3-point smoothing, AUC-difference endpoint)
  are unchanged. The source snapshots are the same cached files used at
  N=8.

  *Optional-stopping note.* The decision to extend was made AFTER seeing
  the N=8 results, which is a known statistical concern. This amendment
  is the discipline: the N=16 extension is committed BEFORE the
  additional seeds run, and the analysis at N=16 uses the same primary
  endpoint (AUC difference + Student-t 95% CI). N=16 is committed as
  the final stopping point for both — no further seed extensions are
  permitted on these two designs without another amendment.

  *Chronology assertion.* This amendment is committed BEFORE the N=16
  runs. The endpoint, the stopping rule (N=16, no further extensions
  without a new amendment), and the unchanged-design constraint are
  pre-outcome.

  *Amendment trigger:* the v3.13 marginal positive (commit e8bd009) and
  the v3.14 marginal null (commit 0cf55ca) — both CIs at N=8 are too
  wide for the downstream decisions they drive.

- **2026-05-20 (v3.15 — heterogeneous-dim core-only transfer
  (CartPole -> MCC-Hard) under the v3.13/v3.14 mechanism).**

  *Trigger.* The v3.14 N=16 result (commit 511527b, mean +11.00, 95% CI
  +/-5.80, lower bound +5.20) confirmed that the env-agnostic core
  alone — without encoder transfer — carries transferable structure on
  the same-obs-dim MCC -> MCC-Hard pair. The core (gru + prior +
  posterior) is the ONLY subset of the RSSM shareable across an obs_dim
  mismatch, so v3.14's positive directly authorizes the
  heterogeneous-dim follow-up that the v3.13 and v3.14 amendments
  preregistered as conditional.

  *Design.* Source = CartPole-v1 (obs_dim 4, action_dim 2, discrete);
  target = MountainCarContinuous-Hard (obs_dim 2, action_dim 1,
  continuous). The source RSSM is trained inside a DeviceAgent on the
  proven raw-obs path (PPO + WM + latent policy, no curiosity for
  discrete — the existing DeviceAgent code path for discrete envs); its
  env-agnostic core (gru + prior + posterior) is snapshotted. The target
  arms run the v3.14 mechanism exactly: SAC reads cat([h, z]) only
  (latent-only, load-bearing), the RSSM is frozen, the augmented vector
  is run through a running aug-normalizer, curiosity is the fresh ICM
  module per arm, three arms (transfer = load_transferable_state_dict
  from CartPole / permuted = the CartPole core with each parameter
  tensor's elements randomly permuted / scratch = fresh random core).
  Metric and stopping rule: per-seed AUC difference, 3-point-smoothed,
  Student-t 95% CI, N=8 first; if marginal (CI lower bound within +/-3
  of zero) extend to N=16 in a follow-up amendment.

  *Relation to v3.8.* v3.8 (cartpole -> mcc, gym pilot) was a clean
  falsification of cartpole -> mcc transfer under a DIFFERENT mechanism
  — the gym pilot trained both arms via _train_sac (SAC reading raw
  obs) and used a dream-training transfer channel. v3.15 retests
  cartpole -> mcc transfer under the v3.13/v3.14 latent-only mechanism
  (SAC reads ONLY the frozen RSSM latent, with the augmented vector
  scale-normalized and a permuted-core structure control). v3.15 is NOT
  a re-run of v3.8; it is a new test under a stronger transfer channel,
  authorized by the v3.14 positive.

  *Decisive interpretation.*
  - v3.15 positive (transfer > permuted, CI excludes 0): the workshop's
    heterogeneous-dim transfer claim is alive — the env-agnostic core
    carries dynamics structure that helps SAC even on a different
    dynamics family.
  - v3.15 null: the v3.13/v3.14 positive holds within a shared-dynamics
    family (MCC family) but does NOT generalise across families
    (CartPole pole-balancing -> MCC energy-pumping). Workshop story
    becomes "transfer works within a dynamics family", a meaningful but
    narrower claim.

  *Chronology assertion.* This amendment is committed BEFORE the v3.15
  implementation and BEFORE any v3.15 run. The source/target pair, the
  unchanged-from-v3.14 mechanism, the metric, and the N=8 -> N=16
  stopping rule are pre-outcome.

  *Amendment trigger:* the v3.14 N=16 positive (commit 511527b) and the
  conditional heterogeneous-dim follow-up preregistered in v3.13 and
  v3.14.

- **2026-05-20 (v3.16 — heterogeneous-dim core transfer with a
  shared-physics source: Pendulum -> MCC-Hard).**

  *Trigger and rationale.* v3.15 N=8 (commit ea1df5d, CartPole -> MCC)
  came in marginal-null: mean +7.24, 95% CI +/-14.55, lower bound -7.31
  (outside the prereg's +/-3 marginal band, so NOT extended to N=16 —
  the prereg discipline holds). The honest read of v3.15: the env-
  agnostic core does carry a positive-direction signal across an
  obs_dim mismatch (mean is positive, 5/8 seeds favour transfer), but
  CartPole's pole-balancing dynamics and MCC's energy-pumping dynamics
  are physically very different, so the shared structure available to
  transfer is thin and the signal cannot be resolved at N=8. v3.16
  re-tests the heterogeneous-dim claim with a closer-physics source.

  *Source choice.* Pendulum (gymnasium Pendulum-v1, here implemented as
  ``DeviceVecPendulum``: obs_dim 3 (cos theta, sin theta, theta_dot),
  action_dim 1 continuous (torque)). Pendulum and MCC share a
  one-degree-of-freedom nonlinear mechanical system with gravity, a
  velocity clamp, and continuous-torque/force actuation — both require
  energy-pumping behaviour to reach a high-potential goal state. This
  is much closer-physics than CartPole and is precisely the kind of
  shared-dynamics-family-different-obs-dim transfer the workshop claim
  is about.

  *Design.* Identical to v3.15 except the source env:
    - SOURCE: DeviceAgent on Pendulum (continuous: SAC + WM + latent +
      curiosity, same path the MCC source uses). The env-agnostic core
      (gru + prior + posterior) is snapshotted.
    - Three target arms on MCC-Hard, latent-only SAC, frozen RSSM,
      aug-normalizer over the 160-d latent, ICM curiosity fresh per
      arm. Transfer / permuted / scratch — identical to v3.14/v3.15.
  Metric and stopping: per-seed AUC difference, 3-point smoothed,
  Student-t 95% CI. N=8 first; extend to N=16 only if CI lower bound
  is within +/-3 of zero (the SAME rule v3.15 used; respected even when
  unfavourable). No further extensions without another amendment.

  *Decisive interpretation.*
  - v3.16 positive: the workshop's heterogeneous-dim claim is alive,
    refined as "across-obs-dim transfer works when source and target
    share dynamics family". This is a defensible workshop result.
  - v3.16 also null: even with shared-physics, cross-obs-dim transfer
    of the env-agnostic core alone does not produce detectable
    acceleration. The honest conclusion would then be that this RSSM
    transfer mechanism delivers within a fixed obs_dim only — and the
    project pivots to a different substrate (the user's planned next
    branch: alternative model classes — policy distillation,
    successor features, or markovian dynamics models).

  *DeviceVecPendulum.* Newly added in this amendment's implementation
  commit (next). Physics matches gymnasium's Pendulum-v1 exactly
  (g=10, m=l=1, max_speed=8, max_torque=2, dt=0.05, max_steps=200,
  reward = -(theta_n^2 + 0.1 theta_dot^2 + 0.001 u^2)). Episode is
  truncation-only (no termination — Pendulum runs for 200 steps).

  *Chronology assertion.* This amendment is committed BEFORE the
  DeviceVecPendulum implementation lands, BEFORE the v3.16 script is
  written, and BEFORE any v3.16 run. Source/target pair, mechanism,
  metric and the N=8 -> N=16 stopping rule are pre-outcome.

  *Amendment trigger:* the v3.15 marginal-null on CartPole -> MCC and
  the question of whether a closer-physics heterogeneous-dim source
  enables the transfer the v3.13/v3.14 same-dim positives suggest is
  possible.

- **2026-05-20 (v3.17 — solidify v3.15's CartPole -> MCC null at N=16
  to match v3.16's budget).**

  *Honest motivation.* v3.15 N=8 (CartPole -> MCC, commit ea1df5d) was
  concluded null per the strict preregistered rule (CI lower bound
  -7.31, outside the +/-3 marginal band). The contrast with v3.16's
  N=16 positive (Pendulum -> MCC, commit 98ff2ca, lower bound +1.70)
  carries the project's "physics-matters" interpretation, but the N
  mismatch (8 vs 16) is a methodological asymmetry: the
  CartPole-vs-Pendulum contrast should rest on comparable budgets.
  v3.17 extends v3.15 to N=16 to firm up the null at the same N as
  v3.16's positive.

  *Optional-stopping discipline.* This extension is post-hoc relative
  to the v3.15 amendment. The honest framing: v3.17 IS optional
  stopping (the rule said don't extend; we are extending), motivated
  by the user's request for additional confirmatory tests. The risk
  guarded against is "extend until you get a positive". The discipline:
  N=16 is committed as the FINAL stopping point — no further
  extensions on v3.15 / v3.17 without another amendment, regardless
  of where the N=16 verdict lands. If the N=16 verdict surprises us by
  going positive (becoming a true positive that N=8 missed), that
  itself is reported as a finding, not buried.

  *Design unchanged.* All v3.15 parameters identical (latent-only SAC,
  frozen RSSM, aug-normalizer, 3 arms transfer/permuted/scratch, ICM
  curiosity, 3-point smoothing, same source snapshot at
  `transfer_v315_out/source_cartpole_core.pt`). The resume-aware
  script appends seeds 8-15 to the existing seeds 0-7. AUC difference
  + Student-t 95% CI is the same endpoint.

  *Chronology assertion.* This amendment is committed BEFORE the
  additional 8 seeds run; the FINAL stopping point at N=16 is
  pre-outcome.

  *Amendment trigger:* the user's request for more confirmatory tests
  to solidify the cross-experiment "transfer works iff physics is
  shared" interpretation, and the N-mismatch between v3.15 (N=8 null)
  and v3.16 (N=16 positive).

- **2026-05-20 (v3.18 — reverse-direction heterogeneous-dim:
  MCC -> Pendulum, symmetry check on the v3.16 positive).**

  *Trigger.* v3.16 N=16 (commit 98ff2ca, Pendulum -> MCC) was solidly
  positive: heterogeneous-dim transfer works when source and target
  share dynamics structure. The user requested confirmatory tests; a
  natural symmetry check is whether transfer works in the REVERSE
  direction with the same task pair. If the v3.16 effect reflects
  shared dynamics structure (as opposed to a directional asymmetry
  specific to Pendulum-source / MCC-target), MCC -> Pendulum should
  also be positive.

  *Design.* Identical to v3.16 except source/target swap:
  - SOURCE: standard MountainCarContinuous via the cached snapshot at
    `transfer_v311_out/source_snapshot.pt` (the same source used by
    v3.11/v3.12/v3.14 — already a core-only snapshot, no retraining).
  - TARGET: Pendulum-v1 (DeviceVecPendulum, obs_dim 3, action_dim 1
    continuous). Eval = mean per-episode return; episode length 200
    steps (Pendulum-v1 truncation cap); a perfect Pendulum agent
    scores near 0 (negative cost summed), random near -2000.
  - Same mechanism: latent-only SAC, frozen RSSM, aug-normalizer,
    ICM curiosity, three controls (transfer / permuted / scratch).
  - Metric: per-seed AUC difference, 3-point smoothed, Student-t 95%
    CI. N=8 first; extend to N=16 only if CI lower bound within +/-3
    of zero (same rule as v3.15 / v3.16).

  *Decisive interpretation.*
  - v3.18 positive: the v3.16 effect is symmetric — shared-physics
    transfer works in both directions. Strengthens the workshop claim.
  - v3.18 null (with v3.16 positive): the transfer effect is
    directional, MCC -> Pendulum specifically failing. This would be
    a surprising finding worth investigating (asymmetry diagnostics).
  - Both positive at comparable magnitudes => symmetric, robust
    heterogeneous-dim transfer with shared physics.

  *Chronology assertion.* This amendment is committed BEFORE the
  v3.18 implementation script and BEFORE any v3.18 run.

  *Amendment trigger:* the user's request for more confirmatory tests
  and the natural reverse-direction symmetry check on v3.16's positive.

- **2026-05-20 (v3.19 — multi-skill composition: do TWO transferred
  cores compose into a skill neither carries alone?).**

  *Question.* Phase A established: a single transferred RSSM core
  accelerates SAC iff the source and target share physics. The next
  question is the project's stated long-term goal — "learn a new skill
  faster by reusing OLD skillS, plural." Can two source cores, each
  carrying a partial skill, COMBINE into a target skill that requires
  both? The user's analogy: "I know how to use a violin, I know
  solfege, therefore I can play a piece of sheet music on violin" —
  neither source alone is sufficient; the composition is.

  *Composite target task.* DeviceVecCartPoleOnHill (added in this
  amendment's implementation commit) — a cart on the MCC hill with a
  pole balanced on top. obs (4) = [cart_pos, cart_vel, pole_angle,
  pole_angvel], action (1) = continuous engine force, sparse reward
  +100 only at goal-reach with pole still upright (theta < pi/4),
  early termination on pole-fall. Cart dynamics follow MCC (hill
  gravity, velocity clamp); pole dynamics follow CartPole (cart force
  determines pole accel). Aggressive driving (needed to climb) swings
  the pole; gentle driving preserves the pole but cannot climb. The
  composite genuinely requires BOTH MCC's energy-pumping skill AND
  CartPole's balancing skill.

  *Composition mechanism.* Element-wise WEIGHT AVERAGING of two
  source cores. For each parameter tensor in the env-agnostic core
  (gru + prior + posterior), the composition is
  ``avg[k] = (mcc[k] + cp[k]) / 2``. Same architecture / state_dim
  / SAC input as v3.14 (160-d latent, latent-only SAC). The simpler
  composition mechanism is tested first; concatenation (320-d latent,
  two parallel cores) is the contingent next test if averaging fails.

  *Five arms, N=8, same mechanism as v3.14.*
  - scratch: fresh random core.
  - permuted: MCC core with each parameter tensor's elements
    randomly permuted (scale/rank-matched, structure destroyed).
  - transfer_mcc: MCC core (single-skill source A, energy pumping).
  - transfer_cp: CartPole core (single-skill source B, balancing).
  - transfer_avg: (MCC + CartPole) / 2 — the composition.

  *Decisive interpretation.*
  - transfer_avg > max(transfer_mcc, transfer_cp), 95% CI excludes
    0 on the per-seed pairwise difference => COMPOSITION WORKS:
    averaging two partial-skill cores produces something stronger
    than either alone. This is the "violin + solfege" claim.
  - transfer_avg ~ max(transfer_mcc, transfer_cp): the composition
    has no additive benefit — the best single source already does
    everything it can.
  - transfer_avg < either single: averaging actively destroys
    structure. Pivot to the concatenation mechanism (v3.20 follow-up).
  - All arms null at the budget: composite too hard for the budget,
    follow-up tunes difficulty (looser THETA_THRESH or nearer goal).

  *Sources used (cached, no retraining):* MCC source =
  `transfer_v311_out/source_snapshot.pt` (the env-agnostic core from
  the same MCC source as v3.11/v3.12/v3.14). CartPole source =
  `transfer_v315_out/source_cartpole_core.pt` (the env-agnostic core
  from the v3.15 CartPole DeviceAgent — same network architecture).
  Both are `transferable_state_dict` outputs; their keys/shapes match
  exactly, making element-wise averaging well-defined.

  *Stopping rule.* N=8 first; extend to N=16 only if the decisive
  comparison (transfer_avg minus max(transfer_mcc, transfer_cp)) has
  per-seed-mean CI lower bound within +/-3 of zero.

  *Chronology assertion.* This amendment is committed BEFORE the
  v3.19 implementation script and BEFORE any v3.19 run.
  DeviceVecCartPoleOnHill is committed in the same step.

  *Amendment trigger:* the user's request for multi-skill composition
  experiments (the violin + solfege analogy), building on Phase A's
  same-physics-transfer-works results.

- **2026-05-20 (v3.20 — composition mechanism test on a target where
  single-skill transfer is established to work: MCC + Pendulum cores
  averaged into MCC-Hard).**

  *Why the pivot.* v3.19 (commit a4ff48b) came in null on
  cart-pole-on-hill: BOTH transfer_avg failed to beat the best single
  AND single-skill transfer (transfer_mcc, transfer_cp) failed to beat
  scratch. The two failures are confounded — they could mean (a) the
  averaging mechanism is broken, OR (b) cart-pole-on-hill is too
  DOF-mismatched with the sources for any transfer to help. Without
  separating them, v3.19 cannot answer the composition question.
  v3.20 isolates the composition mechanism by moving to a target where
  single-skill transfer is solidly established to work.

  *Design.* Target = MountainCarContinuous-Hard (the v3.13/14/16
  target where single-skill transfer from MCC or Pendulum produces a
  robust positive effect). Sources = MCC core + Pendulum core (both
  shown to transfer positively to MCC-Hard alone). Composition
  mechanism = element-wise weight averaging of the two cores
  (avg[k] = (mcc[k] + pen[k]) / 2). Same v3.14 mechanism otherwise
  (latent-only SAC, frozen RSSM, aug-normalizer, ICM curiosity).

  *Five arms, N=8.*
  - scratch:          fresh random core
  - permuted:         MCC core permuted
  - transfer_mcc:     MCC core alone (the v3.14 positive arm)
  - transfer_pen:     Pendulum core alone (the v3.16 positive arm)
  - transfer_avg:     (MCC + Pendulum) / 2 — the composition

  *Decisive interpretation.*
  - transfer_avg > max(transfer_mcc, transfer_pen), CI excludes 0
    positive => averaging COMPOSES the two skills into something
    better than either alone. The clean composition positive the
    project is after.
  - transfer_avg ~ max(transfer_mcc, transfer_pen): averaging
    preserves but doesn't add; composition mechanism is neutral on
    this pair.
  - transfer_avg < either single (CI excludes 0 negative): averaging
    actively DESTROYS structure between sources whose dynamics
    families overlap (MCC and Pendulum are both 1-DOF nonlinear
    energy-pumping). Then v3.21 tests CONCATENATION (the original
    v3.19 fallback, reframed as a v3.20 fallback).
  - Both single-skill transfers null on this target (unlikely given
    v3.14/v3.16 positives but possible at this seed sample): would
    indicate a regression in our infrastructure; investigate before
    iterating.

  *Cached sources.* MCC core: `transfer_v311_out/source_snapshot.pt`
  (the same source as v3.11/v3.12/v3.14/v3.19). Pendulum core:
  `transfer_v316_out/source_pendulum_core.pt` (the same source as
  v3.16). No retraining.

  *What the original v3.20 preregistered (concatenation on
  cart-pole-on-hill) becomes.* That test is REORDERED, not cancelled:
  if v3.20-here (averaging on MCC-Hard) shows averaging works as a
  composition mechanism, we then return to the cart-pole-on-hill
  composition question with confidence the mechanism is sound (so a
  null there would diagnose the target, not the mechanism). If
  v3.20-here shows averaging is broken, then we test concatenation
  next — and a positive there would justify revisiting both
  cart-pole-on-hill and the more general composition story.

  *Stopping rule.* N=8 first; extend to N=16 only if CI lower bound on
  the decisive comparison within +/-3 of zero (same rule as
  v3.13/14/15/16).

  *Chronology assertion.* Committed BEFORE the v3.20 implementation
  script and BEFORE any v3.20 run.

  *Amendment trigger:* the v3.19 null with confounded interpretation
  (composition mechanism vs target amenability), the need to isolate
  the composition mechanism on a target where single-skill transfer
  is solidly established.

- **2026-05-21 (v3.21 — composition via LATENT CONCATENATION (two
  parallel cores), the v3.19/v3.20 preregistered fallback).**

  *Trigger.* v3.20 N=8 (commit 9fc3816, MCC+Pendulum -> MCC-Hard, both
  single-skill transfers solidly positive) showed weight-averaging
  COMPOSITION HURTS: transfer_avg - max(transfer_mcc, transfer_pen)
  = -11.31 +/- 7.32, 0/8 seeds positive, CI entirely below zero. The
  averaging mechanism is killed; per the v3.19/v3.20 preregistered
  fallbacks, the next composition mechanism to test is latent
  concatenation.

  *Mechanism: latent concatenation.* Two frozen RSSM cores in parallel,
  each producing its own [h, z]; SAC reads cat([h_a, z_a, h_b, z_b])
  (= 2 * state_dim = 320-d) instead of the v3.13/14's single 160-d
  latent. Each core retains its learned structure intact (no weight
  averaging); SAC's first-layer Linear learns to weight the relevant
  half of the concatenated input. This is the principled composition
  mechanism — preserves each source's representation, lets SAC
  selectively use either or both.

  *Design.* Five arms, two cores per arm, same target as v3.20
  (MountainCarContinuous-Hard, where single-skill transfer is robustly
  positive). N=8.
    - scratch_dual:        2 fresh random cores
    - permuted_dual:       MCC core permuted + Pendulum core permuted
                           (scale/rank-matched, structure destroyed,
                           ensures the dual-core architecture itself
                           doesn't carry transfer signal)
    - transfer_mcc_only:   trained MCC core + 1 fresh random core
                           (single-skill in the concat architecture)
    - transfer_pen_only:   1 fresh random core + trained Pendulum core
                           (single-skill in the concat architecture)
    - transfer_both:       trained MCC core + trained Pendulum core
                           (THE COMPOSITION TEST)

  *Decisive interpretation.*
  - transfer_both > max(transfer_mcc_only, transfer_pen_only), CI
    excludes 0 positive: latent concatenation COMPOSES two skill cores
    into something stronger than either alone — the clean composition
    result the project is after.
  - transfer_both ~ max(transfer_mcc_only, transfer_pen_only): concat
    is neutral; SAC uses one core and ignores the other.
  - transfer_both < max(single): adding a second skill HURTS, even via
    concatenation. Would imply RSSM-core skill composition is broken
    as a general approach with this substrate; pivot to a different
    substrate (per the user's "if RSSM doesn't deliver, try other
    model classes").

  *Implementation note.* Requires new dual-latent rollout/eval
  functions (collect_rollout_dual_latent, evaluate_dual_latent) added
  to rollout.py — same structure as collect_rollout_augmented but
  threads two RSSMs in parallel and produces a cat of both latents
  for SAC. Sources cached as in v3.20 (no retraining). Stopping rule:
  N=8 first; extend to N=16 only if CI lower bound on the decisive
  comparison within +/-3 of zero (same rule as v3.20).

  *Chronology assertion.* This amendment is committed BEFORE the
  rollout.py changes, the v3.21 script and any v3.21 run.

  *Amendment trigger:* the v3.20 clean negative on averaging
  composition (commit 9fc3816); concatenation is the preregistered
  fallback as stated in v3.19 and v3.20.

- **2026-05-21 (v3.22 — pivot substrate: composition via POLICY weight
  transfer instead of RSSM-core weight transfer).**

  *Trigger.* v3.20 (averaging) and v3.21 (concatenation) both failed to
  produce additive composition on the RSSM-core substrate (averaging
  CLEAN NEGATIVE, concatenation NEUTRAL). The user's plan triggered:
  "if the RSSM substrate doesn't deliver composition, test alternative
  substrates." v3.22 swaps the substrate from "RSSM core weights" to
  "SAC policy weights" (actor + critics), keeping everything else as
  parallel to v3.20 as possible.

  *Mechanism.* No RSSM at all in the target — SAC reads raw obs
  directly (the proven raw-obs DeviceAgent setup). Source training:
  raw-obs DeviceAgent trains MCC and Pendulum sources; at end, save
  the env-agnostic subset of each source's SACPolicy (shared layer 2 +
  mean_head + logstd_head) and each source's two QNetwork critics
  (net.2 + net.4). These are the layers that DON'T depend on obs_dim
  (all action_dim=1 continuous, so the heads are shape-compatible).

  *Arms (5, N=8, target = MCC-Hard).*
  - scratch_pol: fresh random SAC (the v3.13-comparable null baseline)
  - permuted_pol: MCC source policy weights permuted per-tensor
  - transfer_mcc_pol: env-agnostic subset of MCC's SAC loaded
  - transfer_pen_pol: env-agnostic subset of Pendulum's SAC loaded
  - transfer_avg_pol: average of MCC + Pendulum env-agnostic subsets

  *Single-skill check (within the same experiment).* This v3.22 design
  also tests SINGLE-skill policy weight transfer (a question never
  asked before — Phase A used RSSM-core transfer to a target SAC
  reading the LATENT). If transfer_mcc_pol > scratch_pol with CI
  excluding 0, single-skill policy transfer works on this substrate.
  This contextualises any composition result.

  *Decisive interpretations.*
  - transfer_mcc_pol > scratch_pol AND transfer_avg_pol > max(
    transfer_mcc_pol, transfer_pen_pol), both CIs excluding 0: the
    policy-weight substrate supports single-skill transfer AND
    additive composition where the RSSM substrate did not. This is
    the cleanest possible positive on the violin claim.
  - Single-skill policy transfer works, but averaging composition
    doesn't: substrate doesn't matter for composition; the mechanism
    (averaging) is the issue regardless of substrate. Concatenation
    on policies would be a contingent v3.23.
  - Neither single-skill nor composition policy transfer works:
    the policy substrate is worse than RSSM-core for transfer. Try
    a different substrate (successor features, markovian dynamics)
    in v3.23.

  *Sources are re-trained for this experiment* (the original v3.11 +
  v3.16 source caches only saved the RSSM core, not the SAC actor +
  critics). Source training adds the policy-weight save step at the
  end and caches it; subsequent runs reuse the cache. SAC trainer
  gets two new methods: policy_transferable_state_dict (extract the
  env-agnostic subset) and load_policy_transferable_state_dict (load
  it). Implementation note: action_dim=1 across MCC / Pendulum /
  MCC-Hard, so the action-head layers (mean/logstd_head, Q-output)
  are shape-compatible and included in the transferable subset.

  *Stopping rule.* N=8 first; extend to N=16 only if the decisive
  composition CI lower bound is within +/-3 of zero (same rule as
  v3.20 / v3.21).

  *Chronology assertion.* Committed BEFORE the sac.py changes, the
  v3.22 script, and any v3.22 run.

  *Amendment trigger:* v3.20 + v3.21 exhausted RSSM-substrate
  composition mechanisms (averaging negative, concat neutral); user
  chose option 4 (pivot substrate) on the post-v3.21 fork.

- **2026-05-21 (v3.23 — composition via LEARNED soft gating over two
  frozen cores).**

  *Trigger.* v3.20 (averaging RSSM cores: CLEAN NEGATIVE), v3.21
  (concatenating RSSM latents: NEUTRAL) and v3.22 (averaging SAC
  policy weights: DEEPLY NEGATIVE) all failed to produce additive
  composition. Each used a STATIC composition rule. The user's plan
  triggered: try a more sophisticated, LEARNED composition. v3.23
  tests learned soft gating over two frozen cores — SAC trains a
  small gate jointly with its policy/critics, deciding how to mix
  the two source latents based on context.

  *Mechanism.* Two frozen RSSM cores (MCC and Pendulum source, same
  as v3.20/v3.21) produce two latents [h_a, z_a] (160-d) and
  [h_b, z_b] (160-d). A small `Gate` MLP takes the 320-d concat and
  outputs a scalar w in [0, 1] via sigmoid. The combined latent is
  ``mixed = w * latent_a + (1 - w) * latent_b`` (160-d). SAC's actor
  and critics read `mixed` (same architecture / dim as v3.14). The
  gate is SHARED between actor and critics (and their target nets);
  its parameters get gradients from the SAC loss via the
  differentiable path mixed -> actor/critic -> loss. Off-policy
  staleness note: the buffer stores aug_obs (the 320-d concat); at
  each SAC update the CURRENT gate re-computes mixed from stored
  aug_obs, so gate-improvement re-uses old data without
  importance-sampling — accepted as a standard SAC off-policy
  approximation.

  *Five arms, N=8, target = MCC-Hard, same v3.21 buffer/aug_obs setup
  (320-d dual latents).*
  - scratch_gated:        2 fresh random cores + gate
  - permuted_gated:       MCC core permuted + Pendulum core permuted + gate
  - transfer_mcc_gated:   trained MCC core + 1 fresh random core + gate
  - transfer_pen_gated:   1 fresh random core + trained Pendulum core + gate
  - transfer_both_gated:  trained MCC core + trained Pendulum core + gate
                          -- THE GATED COMPOSITION TEST

  *Decisive interpretation.*
  - transfer_both_gated > max(transfer_mcc_gated, transfer_pen_gated),
    CI excludes 0 positive: learned gating COMPOSES the two skill
    cores into something better than either alone. The first positive
    composition result the project achieves. Validates that the
    earlier nulls (averaging, concat) were a MECHANISM problem (static
    rules), not a fundamental limitation.
  - transfer_both_gated similar to best single: gating doesn't extract
    additive value from the second skill (SAC's gate learns w near
    0 or 1, picking the better single — composition still null).
  - transfer_both_gated < either single: gating actively hurts. Would
    indicate the gate adds optimisation difficulty without value.

  *Implementation.* No changes to SACTrainer or rollout.py;
  collect_rollout_dual_latent (v3.21) re-used. v3.23 script builds the
  SACTrainer with obs_dim = state_dim (160) and WRAPS its actor and
  critics with a shared Gate module (320 -> 1) that internally
  compresses 320-d aug_obs to 160-d mixed before forwarding. Gate
  parameters are added to the policy optimizer.

  *Stopping rule.* N=8 first; extend to N=16 only if the decisive
  composition CI lower bound is within +/-3 of zero (same rule as
  v3.20/21/22).

  *Chronology assertion.* Committed BEFORE the v3.23 script and any
  v3.23 run.

  *Amendment trigger:* the three static-composition nulls
  (v3.20/v3.21/v3.22) and the user's direction to push the
  composition question with a sophisticated mechanism (option 3
  on the post-v3.22 fork).

- **2026-05-22 (v3.24 — hierarchical composition: a learned MANAGER
  orchestrates two frozen skill policies on a target that genuinely
  requires both).**

  *Trigger and rationale.* Four composition mechanisms (v3.20 averaging
  RSSM cores, v3.21 latent concat, v3.22 averaging policy weights,
  v3.23 learned soft gate over cores) all failed to produce additive
  composition. The diagnosed root cause is NOT only the mechanism: in
  every case the target task (MCC-Hard) is solvable by a SINGLE skill,
  so there is no composition to detect — the v3.23 gate correctly
  learned to SELECT one core. v3.24 fixes BOTH problems: a target that
  genuinely requires two distinct skills, and a hierarchical mechanism
  (the user-chosen option B).

  *Composite target with isolable sub-skills.* DeviceVecCartPoleOnHill
  (the v3.19 composite — a cart on the MCC hill with a balanced pole)
  is paired with two NEW single-skill variants that share its exact
  4-d obs / 1-d action / goal / reward:
    - DeviceVecCartPoleOnHillClimbOnly: the pole is RIGID (never
      rotates, never falls). Isolates the CLIMBING skill — only the
      hill must be solved.
    - DeviceVecCartPoleOnHillBalanceOnly: the ground is FLAT (no
      hill-gravity term). Isolates the BALANCING skill — only the pole
      must be kept up while reaching the goal.
  The full composite requires climbing the hill WHILE keeping the pole
  upright; aggressive driving (needed to climb) swings the pole, gentle
  driving (preserves the pole) cannot climb. Neither sub-skill alone
  solves the composite. Because all three share obs/action dims, a
  policy trained on a sub-skill is dimensionally drop-in for the
  composite (fixing the v3.19 DOF mismatch).

  *Hierarchical mechanism (option B).* Two source SAC policies are
  trained, one on ClimbOnly and one on BalanceOnly, then FROZEN. A
  learned MANAGER — itself a SAC agent with obs the 4-d composite
  state and a 1-d continuous action w in [0, 1] — orchestrates them:
  at each step the action applied to the composite env is
  ``a = w * a_climb + (1 - w) * a_balance`` where a_climb / a_balance
  are the two frozen policies' greedy actions on the current obs. Only
  the manager is trained on the composite. This is the violin-analogy
  structure: the manager (the "musician") composes two fixed
  competences by deciding, per state, how much of each to apply.

  *Four arms, N=8, target = DeviceVecCartPoleOnHill.*
    - scratch_mgr:           manager + 2 fresh random policies
    - transfer_climb_only:   manager + (trained climb policy, random)
    - transfer_balance_only: manager + (random, trained balance policy)
    - transfer_both:         manager + (trained climb, trained balance)
                             -- THE HIERARCHICAL COMPOSITION TEST
  Each source policy is applied through its own saved obs-normaliser
  so it sees inputs distributed as in its training.

  *Decisive interpretation.*
  - transfer_both > max(transfer_climb_only, transfer_balance_only),
    95% CI on the per-seed difference excludes 0: hierarchical
    composition WORKS — a manager orchestrating two frozen skills
    solves a task neither skill solves alone. The project's first
    positive composition result and a direct realisation of the
    violin+solfege analogy.
  - transfer_both ~ best single: the manager cannot extract joint
    value; it collapses to one skill. Composition still null.
  - transfer_both < best single: the second option distracts the
    manager. Composition hurts.
  - All arms fail (no arm solves the composite): the composite is too
    hard for the manager-blend action class at the budget; a follow-up
    loosens difficulty (THETA_THRESH) or grants the manager a residual
    action term.

  *Metric.* Per-seed sample-efficiency AUC (3-point-smoothed eval
  return vs env-steps), Student-t 95% CI on transfer_both minus
  max(single). N=8 first; extend to N=16 if the decisive CI lower
  bound is within +/-3 of zero.

  *Implementation.* Two env variants added to device_env.py (committed
  with this amendment). A v3.24 script trains the two source policies
  via the proven raw-obs DeviceAgent path, then runs the four manager
  arms with a custom blend-action rollout. No changes to SACTrainer.

  *Chronology assertion.* This amendment, and the DeviceVecCartPoleOnHill
  Climb/Balance-Only env variants, are committed BEFORE the v3.24
  script and BEFORE any v3.24 run.

  *Amendment trigger:* the four static/gated composition nulls
  (v3.20-v3.23) and the user's directive to pursue composition via
  hierarchical RL / the options framework.

- **2026-05-22 (v3.25 — SEQUENTIAL composition: a manager switches
  between two frozen skills used in different phases).**

  *Trigger.* v3.24 (hierarchical manager on CartPoleOnHill) hit the
  preregistered "all arms fail" branch — 0/8 AUC>1 in every arm. The
  diagnosed cause: that composite's two skills CONFLICT over a single
  shared actuator (climbing needs aggressive oscillation; that same
  oscillation drops the pole), so a convex action blend
  w*a_climb + (1-w)*a_balance can only produce a washed-out compromise.
  The violin+solfege analogy assumes COMPLEMENTARY skills (separate
  channels); v3.24's were ANTAGONISTIC over one channel. v3.25 fixes
  the composite design: a SEQUENTIAL task where the two skills are used
  at DIFFERENT TIMES — no simultaneous conflict — which is the regime
  where hierarchical / options composition is established to work.

  *Composite target (DeviceVecNavigateThenBalance, mode='composite').*
  obs (5) = [cart_pos, cart_vel, pole_angle, pole_angvel, phase].
  Phase 0: drive the cart up the MCC hill to the goal (the pole is
  rigid). On reaching the goal the cart freezes and Phase 1 begins:
  the pole activates and must be held upright; +1 reward per phase-1
  step with the pole up, episode ends on pole-fall. To score well an
  agent must navigate FAST (more time left for phase 1) AND balance
  LONG. The two skills are needed at disjoint times.

  *Single-skill source tasks (share the 5-d obs / 1-d action).*
    - mode='nav': phase-0 task only, +100 on reaching the goal — the
      NAVIGATE skill.
    - mode='balance': starts in phase 1, cart frozen at the goal,
      +1/step pole-up — the BALANCE skill.

  *Mechanism — unchanged from v3.24.* Two source SAC policies (nav,
  balance) trained then FROZEN; a learned SAC MANAGER (1-d action =
  blend weight w in [0,1]) applies a = w*a_nav + (1-w)*a_balance to the
  composite. Because the skills are used in disjoint phases, the
  manager's job reduces to a TEMPORAL switch — learn w near 1 in phase
  0, near 0 in phase 1 (the phase is in the obs, so this is learnable).

  *Four arms, N=8, target = composite.*
    - scratch_mgr:           manager + 2 fresh random policies
    - transfer_nav_only:     manager + (nav policy, random)
    - transfer_balance_only: manager + (random, balance policy)
    - transfer_both:         manager + (nav policy, balance policy)

  *Decisive interpretation.*
  - transfer_both > max(transfer_nav_only, transfer_balance_only),
    95% CI on the per-seed difference excludes 0: SEQUENTIAL
    composition WORKS — a manager orchestrating two frozen skills in
    sequence solves a task neither solves alone. The project's first
    positive composition result.
  - transfer_both ~ best single, or worse: composition still fails
    even in the sequential regime. Combined with v3.20-v3.24 that
    would be a strong, well-bounded negative — simple skill
    composition does not work in this RL setup — and the project
    consolidates the Phase-A single-skill positives.

  *Metric.* Per-seed sample-efficiency AUC (3-point-smoothed
  mean-completed-episode return vs env-steps), Student-t 95% CI on
  transfer_both minus max(single). N=8 first; extend to N=16 if the
  decisive CI lower bound is within +/-3 of zero.

  *Implementation.* DeviceVecNavigateThenBalance added to device_env.py
  (one class, mode in {composite, nav, balance}), committed with this
  amendment. The v3.25 script reuses v3.24's manager + blend-rollout
  machinery unchanged.

  *Chronology assertion.* This amendment and DeviceVecNavigateThenBalance
  are committed BEFORE the v3.25 script and any v3.25 run.

  *Amendment trigger:* the v3.24 all-arms-fail outcome, the
  conflicting-skill diagnosis, and the user's directive to pursue
  composition via hierarchical RL — applied now to the sequential
  regime where it is sound.

- **2026-05-24 (v3.26 — DISCRETE manager via straight-through hard
  selection, addressing v3.25's soft-blend pathology).**

  *Trigger.* v3.25 (sequential composition with a SAC manager outputting
  a continuous blend weight w in [0,1]) was null with a striking
  signature: transfer_both AUC was 16.12 +/- 0.12 across N=8 (std
  ~0.7% of mean) — the manager converged to the SAME local minimum
  every seed, around w ~ 0.4-0.6, splitting the action 50/50 between
  the two skills instead of hard-switching per phase. The single-skill
  arms forced commitment (the random option leaves only one useful
  signal) and outperformed. The diagnosed pathology: soft-blend with
  two real options cannot learn a phase-conditional hard switch — it
  settles into "split the baby." The user-directed next test is a
  DISCRETE manager.

  *Mechanism — straight-through hard selection.* The manager's SAC
  architecture is unchanged (1-d continuous action w in [0,1]), but
  the blend is replaced with a hard one-of-two selection with
  straight-through gradient flow:

      hard = (w > 0.5).float()                 # 0 or 1
      w_eff = hard + w - w.detach()            # forward = hard;
                                               # d(w_eff)/dw = 1
      a = w_eff * a_opt_a + (1 - w_eff) * a_opt_b
                                               # forward: one of two
                                               # actions, period.

  Forward pass: the env action is exactly a_opt_a if w > 0.5 else
  a_opt_b — a true discrete selection. Backward pass: the manager's
  gradient on w flows as in the soft-blend, so SAC's policy still
  trains on a continuous w. This is the standard straight-through
  estimator and it directly addresses the v3.25 pathology by removing
  the smooth-mixing local minimum at w=0.5.

  *Design — unchanged from v3.25.* Target = DeviceVecNavigateThenBalance
  (composite). Sources = the same cached nav and balance policies
  (`transfer_v325_out/source_nav.pt`, `source_balance.pt`). Four arms
  (scratch_mgr / transfer_nav_only / transfer_balance_only /
  transfer_both). N=8. Endpoint = per-seed AUC, decisive
  transfer_both - max(singles), Student-t 95% CI.

  *Decisive interpretation.*
  - transfer_both > max(singles), CI excludes 0: discrete switching
    works — the project's first sequential composition positive,
    validates the v3.25 diagnosis (the pathology was soft-blending,
    not composition itself).
  - transfer_both ~ singles: even with hard selection the manager
    doesn't extract additive value from two real skills. The 8th
    composition null bounds the negative further.
  - transfer_both < singles: hard selection introduces enough critic
    instability (Q-function across a discontinuity at w=0.5) to hurt.
    Would point to a value-based discrete manager (DQN-over-options)
    as a follow-up if pursued.

  *Stopping rule.* N=8 first; extend to N=16 only if the decisive CI
  lower bound is within +/-3 of zero (same rule as v3.20-v3.25).

  *Chronology assertion.* This amendment is committed BEFORE the v3.26
  script and any v3.26 run.

  *Amendment trigger:* the v3.25 manager pathology (transfer_both
  stuck at 16.12 +/- 0.12 across 8 seeds) and the user's directive
  to try a discrete manager — option 2 on the post-v3.25 fork.

============================================================================
v4.0 — PARADIGM PIVOT: from representation transfer to model-based
developmental learning.
============================================================================

- **2026-05-29 (v4.0 — pivot rationale and the developmental program).**

  *Why pivot.* The v3.x program rigorously established two things:
  (1) a transferred RSSM world-model core gives a MODEST, CONDITIONAL
  single-skill speed-up to SAC, only when source and target share
  dynamics (v3.13/14/16 positive; v3.15/17 null contrast); (2) simple
  additive skill COMPOSITION does not work — 8 experiments across 5
  mechanism families (averaging, concat, gate, soft/hard hierarchical),
  all null/negative. The deeper diagnosis: across the ENTIRE v3.x
  program the world model was never used to ACT — SAC learned from real
  experience and the RSSM was only an auxiliary observation / curiosity
  signal. We transferred an organ we never used; hence the weak gains.
  Representation warm-starting is also the wrong paradigm for the
  project's stated goal — it gives a one-shot bump, it does not COMPOUND.

  *The goal, restated precisely (per the project owner).* A child-like
  developmental learner: first absorb many BASIC notions, then acquire
  COMPLEX notions that REUSE the basic ones, getting faster at each new
  notion that links to prior knowledge — and, when a new notion has no
  link, simply learning it fresh (no forced negative transfer). The
  measurable signature is a learning-to-learn curve: env-steps to master
  the K-th new task DECREASES as the library of mastered notions grows.

  *Architecture mapping (the v4 thesis).*
  - "basic notions" = (a) a world model of how the environment behaves
    (dynamics), and (b) elementary goal-reaching skills.
  - "complex notion reusing basics" = a new goal/reward solved by
    PLANNING in the reused world model; later, by sequencing skills.
  - "uses what it knows to go faster" = a trained world model makes a
    new goal in the same dynamics near-zero-shot (plan, don't relearn);
    a skill library makes a complex task a short search over skills.
  - "if no link, learn a new notion" = unseen dynamics -> extend/learn
    a new world model; no applicable skill -> primitive fallback +
    learn a new skill. Additive, relevance-gated, no forced reuse.

  *Phased plan.*
  - Phase 1 (preregistered below): make control MODEL-BASED — solve new
    goals by planning in a reused world model. The foundational pillar:
    "understanding the world makes new goals cheap." Fixes the v3.x
    defect (model never used to act).
  - Phase 2: a growing library of goal-conditioned skills used as
    TEMPORALLY-EXTENDED actions (not the per-step blending that failed
    in v3.24-26), on a compositional curriculum — the "complex uses
    basic" pillar.
  - Phase 3: the full developmental loop — growing library + relevance
    gating (reuse vs learn-fresh) — measured by the learning-to-learn
    curve over a curriculum.

  *What carries over from v3.x:* the device-resident batched-env infra,
  the RSSM (now load-bearing for control), and the preregistration /
  controls / Student-t-CI discipline. What is retired: the
  frozen-representation-transfer + action-blending-composition line
  (concluded; not reopened).

  *Honest scale caveat.* A single GPU + toy continuous-control tasks
  will not produce an impressive "real AI". v4 develops and validates
  the RIGHT MECHANISMS (model-based planning, hierarchical skill reuse,
  curriculum) at small scale with rigour — these are exactly the methods
  that scale; the result is "a validated mechanism", not an AGI.

- **2026-05-29 (v4.0 Phase 1 — a reused world model solves new goals by
  planning; the "understanding -> cheap new goals" demonstration).**

  *Environment (new).* DeviceVecPointMass2D — a 2-D point mass with
  momentum. obs = [x, y, vx, vy] (4-d), action = [fx, fy] (2-d
  continuous force), bounded arena, drag. A GOAL (gx, gy) defines a
  task; reward = goal-reach (within radius eps) with a small control
  cost. One DYNAMICS, infinitely many goals — the cleanest substrate
  for "reuse the dynamics model across tasks", and the natural base for
  Phase-2 compositional navigation. Short-horizon MPC suffices for
  point-to-point navigation (unlike MCC's long-horizon energy pumping),
  so planning is well-posed.

  *Mechanism.* (1) Train an RSSM world model (dynamics + decoder) on the
  point-mass dynamics, goal-AGNOSTIC (random-goal / exploratory data).
  (2) FREEZE it. (3) For a NEW goal: solve by CEM-MPC planning purely in
  the frozen model — sample action sequences, roll them via RSSM.imagine,
  decode latent -> predicted obs, score by the ANALYTIC new-goal reward
  (distance to the new goal; requires NO reward-head learning), refit the
  CEM distribution, execute the first action (receding horizon). The
  agent does ZERO policy/reward learning on the new goal — it only plans
  in its reused understanding of the world.

  *Arms / controls.*
  - mpc_trained:  CEM-MPC in the world model trained on the dynamics.
  - mpc_untrained: CEM-MPC in a fresh RANDOM RSSM (control — isolates
    that the LEARNED dynamics, not the planner, enable solving).
  - sac_scratch:  from-scratch SAC trained per goal (the warm baseline
    for "cost to master a new goal without reuse").

  *Endpoints.*
  - Primary: per-goal success rate and env-steps-to-goal of mpc_trained
    vs mpc_untrained (CI excludes parity => the learned model is what
    enables zero-shot goal solving).
  - The "faster and faster" curve: TOTAL env-steps to master K new goals.
    mpc_trained = (one-time model-training steps) + ~0 per goal -> the
    per-goal cost amortizes to ~0 as K grows; sac_scratch = K x
    per-goal-training -> linear. The crossover and the ->0 asymptote ARE
    the demonstration of compounding.
  - Secondary: mpc_trained must succeed across a distribution of goals
    (generalisation of the single dynamics model across the goal space).

  *Decisive interpretation.* mpc_trained solves new goals at near-zero
  marginal env-cost and beats mpc_untrained decisively => the
  model-based foundation works and "understanding the world makes new
  goals cheap" is demonstrated; proceed to Phase 2. If planning in the
  trained model fails (e.g. model not accurate enough for multi-step
  rollouts, or CEM horizon too short), diagnose model quality / planner
  horizon before proceeding.

  *Chronology assertion.* This amendment is committed BEFORE
  DeviceVecPointMass2D, the CEM-MPC planner, and any v4.0 run.

  *Amendment trigger:* the v3.x conclusion (representation transfer
  weak, composition null, world model unused for control) and the
  project owner's restated developmental-learning goal.

- **2026-05-29 (v4.0 Phase 2 — composition by sub-goals: reuse the
  "reach a point" skill to learn a sequential task fast).**

  *Trigger.* Phase 1 succeeded: a reused world model solves new goals by
  planning, and the compounding curve is demonstrated (marginal cost per
  new goal -> 0; crossover vs from-scratch at k=8; mpc success 1.00 on
  12 goals). Phase 2 is the project owner's "complex notion reuses basic
  notions, learned faster" pillar — with the CORRECT mechanism
  (temporal abstraction over sub-goals), not the per-step action-blending
  that v3.24-26 falsified.

  *Task (DeviceVecOrderedVisit).* obs (5) = [x, y, vx, vy, progress],
  same point-mass dynamics as Phase 1. Visit 3 fixed zones in a FIXED
  order; SPARSE reward (+1 per correct-next zone, +10 on completing all
  three; nothing for a wrong zone). Sanity-checked: an oracle that heads
  to the next-required zone completes reliably (371 completions / 300
  steps / 64 envs); RANDOM primitive actions complete ZERO times. So the
  task is solvable only by COMPOSING "reach a zone" moves — flat
  primitive exploration cannot find it.

  *Mechanism — hierarchical, temporal abstraction.*
  - Low level = the REUSED Phase-1 world model (4-d point-mass dynamics,
    UNCHANGED) + CEM-MPC: given a sub-goal point, plan to reach it. The
    "basic notion" — reused as-is, zero relearning.
  - High level = a SAC agent whose action is a 2-D SUB-GOAL point. Each
    high-level decision triggers a MACRO-STEP: the low-level MPC pursues
    the sub-goal for up to K low-level steps; the high level then
    re-decides. The high level trains on macro-transitions (obs, sub-goal,
    summed reward, next obs) — a semi-MDP. Temporal abstraction collapses
    the decision horizon from ~150 primitive steps to ~10 macro-decisions,
    which is exactly what made credit assignment intractable for the
    per-step managers of v3.24-26.
  - The high level learns ONLY the composition (which zone next / the
    order); the navigation is reused. "Complex = sequence of basics."

  *Arms.*
  - hierarchical_reuse: high-level SAC + low-level MPC in the TRAINED
    (reused) world model.
  - hierarchical_untrained: high-level SAC + low-level MPC in a fresh
    RANDOM world model (control — the low level cannot reach sub-goals,
    so composition has no working primitive). Isolates that the REUSED
    dynamics knowledge is what enables fast composition.
  - flat_scratch: flat SAC over primitive [fx, fy] actions, from scratch.

  *Endpoint.* Env-steps to MASTER the ordered visit (completion success
  >= 0.8 from random starts), per arm, N seeds. Decisive:
  hierarchical_reuse masters in DRAMATICALLY fewer env-steps than
  flat_scratch — or flat_scratch never masters (random completes 0, so a
  flat learner may fail entirely), in which case the result is the
  stronger "composition makes possible what flat RL cannot". And
  hierarchical_reuse >> hierarchical_untrained (the reused world model,
  not just the hierarchy, is what enables it).

  *Graceful-fallback note (Phase 3 hook).* The high level can only
  compose sub-goals the low-level world model can actually reach; here
  all zones are reachable by the reused free-space model, so reuse
  suffices. When a sub-task is NOT reachable by the current model (novel
  dynamics), the developmental loop must extend the model / learn a new
  skill — the "if no link, learn a new notion" branch, deferred to
  Phase 3.

  *Chronology assertion.* This amendment and DeviceVecOrderedVisit are
  committed BEFORE the Phase-2 hierarchical-controller script and any
  Phase-2 run.

  *Amendment trigger:* Phase 1 success and the project owner's
  clarified developmental-composition vision (complex reuses basic ->
  faster), realised with temporal-abstraction hierarchy.

- (Subsequent amendments timestamped here before execution.)
