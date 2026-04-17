# Ragnarok — Research Proposal

**Principal Investigator:** Jérémie Mortier (independent researcher, France)
**Project:** Modular reinforcement learning with cross-action-space skill transfer
**Repository:** https://gitlab.com/mortier.jeremie/ragnarok (mirror: https://github.com/tynitoon/ragnarok)
**Contact:** mortier.jeremie@gmail.com
**Date:** 2026-04-17

---

## 1. Project summary

Ragnarok is a **research program on modular skill learning in RL agents**, not a single-hypothesis study. The program addresses three open questions about *how skills should actually be represented and reused*, each with concrete falsifiable tests planned (detailed in `reviews/research_directions.md`):

- **Q1 — Pure skill learning.** Can an agent learn the *physics and causal interactions* of an environment rather than just predicting the next pixel? This addresses a documented limitation of Dreamer-family world models (Zhang 2021, DBC; Robine 2023, TWM): reconstruction-loss-based world models learn statistical patterns, not causal dynamics.
- **Q2 — Contextual skill selection.** Given a library of crystallized skills, which one (or which combination) should an agent use in a new situation? Current skill-selection methods are static nearest-neighbor on learned embeddings. This touches MoE, options frameworks, PEARL-style context encoders, and multi-skill composition.
- **Q3 — Transfer acceleration.** How can an agent use prior skills to learn faster on new tasks, beyond simple `load_state_dict` initialization? Kickstarting (Schmitt 2018), EWC-protected backbones (Kirkpatrick 2017), imagination-priming — all open engineering and empirical questions.

The **first falsifiable test** in this program is a narrow concrete claim that anchors the broader work:

> **Can a Dreamer-style RSSM's latent trunk (GRU core + prior + posterior distributions) transfer across an action-space-type boundary (discrete ↔ continuous) via shape-checked `load_state_dict`, and measurably accelerate learning on the target task?**

This first test matters because it concerns embodied-agent skill libraries that must span discrete-choice primitives (mode switches, grippers, tool selection) and continuous-control primitives (joint torques, wheel velocities). Progressive Networks (Rusu 2016), Options-Critic (Bacon 2017), SPiRL (Pertsch 2020), Gato (Reed 2022), RT-X (2024) each address related questions but are distinct from this specific mechanism. See §2 for full positioning.

**Why TRC compute for exploration, not just a single study.** The first test (cross-action-type transfer) is mostly complete at the time of writing — Band B rescue gave ratio 1.605 with underpowered p=0.259 at N=5, and Band C N=10 is running. The next six months' value will come from *exploring Q1, Q2, and Q3* across a larger skill library (~10 skills instead of 3) and multiple architectural variants, not from rerunning the first test. The compute enables breadth, not just statistical power.

**Preregistration-grade methodology.** Every hypothesis, threshold, and analytic choice is committed to a public `preregistration.md` document **before** the data it evaluates exists, with git-history-verified chronology and timestamped amendments for any revision. A solo-initiated chronology audit (`reviews/chronology_audit.md`) corrected one integrity defect in the preregistration before external review — this level of self-scrutiny is the project's methodological signature, and it applies to every future experiment under the TRC allocation.

## 2. Scientific contribution

**The architecture.** Ragnarok uses a Recurrent State-Space Model (RSSM) world model in the Dreamer family (Hafner et al., 2019–2023). The transferable subset — GRU core + prior + posterior distributions — is shape-compatible across tasks with different observation and action dimensions because it operates on `cat(h, z)` latent features, upstream of task-specific encoders and downstream of task-specific action heads. At transfer time, the agent loads this subset via `load_state_dict` with strict shape checking on the transferable subset only, then switches its policy to operate on latent features (`acting_policy_mode = latent`) rather than raw observations.

**Explicit positioning relative to adjacent literature:**
- **Gato (Reed et al., 2022)** — single-transformer multi-task policy handling mixed action spaces via tokenization. Differs from Ragnarok in that Gato is *one model* trained on many tasks, not a shape-checked subset transferred from one skill into a new-task agent.
- **RT-X / Open X-Embodiment (2024)** — cross-embodiment transfer via shared transformer backbone across robot morphologies. Shares the spirit of cross-platform transfer but operates at the scale of 1M+ episodes and uses a unified transformer, not a subset-of-RSSM transfer.
- **SPiRL (Pertsch et al., 2020)** — skill priors from offline data, KL-regularized during downstream RL. Assumes homogeneous action space across source and target.
- **Options-Critic (Bacon et al., 2017)** — options framework with learned terminations. Discrete-only action spaces.
- **Progressive Networks (Rusu et al., 2016)** — lateral connections across task columns. Fixed action space, parameters scale O(n) with skill count.

The specific mechanism Ragnarok tests — *shape-checked transferable-subset loading of Dreamer-RSSM's dynamics modules across discrete↔continuous action-type boundary, with the policy switched to latent mode* — is not resolved in the published record to the best of my search.

**The preregistered hypothesis (§8 of `preregistration.md`, committed 2026-04-14, pre-pilot).** On the primary pair CartPole-v1 → MountainCar-Continuous-v0, the project predicted a restricted mean survival time (RMST) ratio ≥ 1.30 (scratch/transfer) with log-rank one-sided p < 0.10, conditional on a mechanism check: acting policy on latent mode, crystallized skill loaded into the RSSM subset.

**Current empirical state (2026-04-18) — the hypothesis is falsified at N=10.**

Three preregistered runs were executed on the primary pair CartPole-v1 → MountainCar-Continuous-v0:

- **Pilot #2 (N=5, original)** — ratio 1.238, fragile under leave-one-out (dropping seed 46 → 1.049). A single lucky seed was driving the headline number.
- **Band B rescue (N=5 fresh seeds, corrected warmup)** — ratio 1.605, LOO min 1.435, log-rank p=0.259. Signal appeared stronger but remained statistically underpowered.
- **Band C extension (pooled N=10, seeds 47–56)** — the preregistered N=10 pooled analysis. Ratio 1.036, log-rank p=0.510 (asymptotic), 0.516 (permutation N=10,000). LOO min 0.871 (drop seed 51). Verdict: **indistinguishable from null hypothesis.**

All three pre-registered Band C kill criteria (committed at SHA `a0c1140`, 2026-04-17, before seeds 52–56 were launched) are triggered:

| Kill criterion | Threshold | Observed | Triggered |
|---|---|---|---|
| Ratio < 1.20 | < 1.20 | **1.036** | ✅ |
| Log-rank p ≥ 0.20 (both tests) | ≥ 0.20 | **0.510 / 0.516** | ✅ |
| LOO minimum ratio < 1.00 | < 1.00 | **0.871** | ✅ |

Per-seed breakdown (10 seeds): 4 positive (seeds 48, 49, 51, 54 with ratios 1.50, 3.29, 1.57, 1.91), 5 neutral (47, 50, 52, 53, 56 with ratios 0.97–1.02), **1 actively anti-transfer** (seed 55, ratio 0.33 — transfer arm 3× slower than scratch). The mean collapses to 1.036 because positive seeds are offset by neutrals and the one large negative.

**What this means scientifically.** The specific mechanism tested in this preregistration — shape-checked transferable-subset loading of a Dreamer-RSSM's dynamics modules across the discrete↔continuous action-space-type boundary, with the policy switched to latent mode — **does not produce a reliable transfer benefit on the primary pair at N=10.** Band B's N=5 signal (ratio 1.605) was high-variance seed lottery, exactly the failure mode documented in Henderson et al. 2018 (*Deep RL That Matters*) and Agarwal et al. 2021 (*Statistical Precipice*). This preregistration is the kind of rigorous falsification the field needs more of — publishable as a negative-results report even without a positive finding, because the methodology and preregistration-to-kill-verdict chain are themselves contributions.

**What this means for the research program.** Branch C of the preregistered decision tree activates as specified. The key chronology a skeptical reviewer should verify: the three decision branches (A pass, B intermediate, C kill → pivot) were committed at SHA `a0c1140` on 2026-04-17 10:15, **before** the N=10-deciding seeds 52–56 were launched. The Q1/Q2/Q3 exploration direction itself was committed at SHA `742299a` on 2026-04-16 at 23:50, when only Band B (N=5) data existed and before Band C began. Both can be independently verified with `git show a0c1140:preregistration.md | grep -A5 "Band C"` and `git log --follow reviews/research_directions.md`. The branch-C pivot is therefore a pre-registered response to an observed outcome, not a post-hoc reframe. What is honestly post-hoc is only the *elevation* of Q1/Q2/Q3 from "stretch research backlog" to "active roadmap" — a change of priority, not of substance.

The workshop-paper-on-primary-pair path is abandoned. The program pivots to the three-question exploration, where this negative result becomes informative:

- **Q1 (physics-grounded world models) is strengthened.** If the reconstruction-based RSSM trunk does not carry transferable causal dynamics across action boundaries, moving to contrastive / disagreement-weighted objectives that *should* carry causal structure becomes a higher-priority direction. The negative result provides a baseline against which improvements can be measured.
- **Q2 (contextual skill selection) becomes more tractable with more skills.** Post-1 horizontal scale to ~10 skills is the substrate needed for this question to be empirically answerable.
- **Q3 (transfer acceleration beyond `load_state_dict`) is justified.** Kickstarting, EWC-protected backbone, imagination-priming — each addresses the now-visible fact that *raw subset loading is insufficient*.

**What TRC compute specifically enables.** The bottleneck is no longer statistical power on the primary pair (that question is answered). The bottleneck is now **breadth of exploration** across Q1/Q2/Q3, which requires (a) a larger skill library built from 7 additional source-target pairs, and (b) architectural variants (contrastive RSSM, kickstarting, EWC-protected loading) tested against the newly-established negative baseline. With a single RTX 4080, this is 200+ GPU-hours of sequential computation. TRC compute removes the execution bottleneck at the exact moment the research program pivots from hypothesis-testing to exploration.

**Honest positioning on pair selection.** The primary pair (CartPole→MountainCar-Continuous) is, the author acknowledges after adversarial review, the most favorable cross-action-type pair imaginable — both pendular-class systems, both ~4D observations, with CartPole's discrete action semantically close to a discretized MountainCar-Continuous force. The null result on this pair is therefore evidence that naive `load_state_dict`-subset transfer does not work *even in the most favorable setting*, not just on adversarial pairs. This strengthens rather than weakens the motivation for Q1/Q2/Q3 research: easier approaches would have sufficed on this pair if they existed.

## 3. Methodology as rigor signal

Beyond the scientific claim, the methodological artifacts of the project are themselves a contribution:

- **Preregistration `preregistration.md`** with 11 timestamped amendments, every one citing a specific commit SHA and rationale.
- **Multi-agent peer review** at every milestone (3–6 specialized LLM agents: RL methodology, code review, strategy, devil's advocate, architecture). Dissent is logged in `reviews/` and resolved before execution.
- **Chronology audit** (`reviews/chronology_audit.md`) — solo-initiated audit that found and corrected a phrasing defect in the B0 fallback plan's pre-data claim.
- **444 tests passing** (pytest).
- **Seed-level JSON artifacts tracked in git**: `pilot_results.json`, `pilot_bandb_results.json`, `pilot_bandc_results.json` contain every run's evaluation curve, wall-clock time, git SHA, and provenance.
- **Kill criteria at every decision gate** (`preregistration.md` §11). Conditions under which the project is explicitly abandoned are pre-specified, not redefined post-hoc.

This methodology is the project's primary differentiator and is proposed as a reusable blueprint for other solo-dev or small-lab RL research. See §6 for a single consolidated disclosure of the LLM-assisted workflow under which it was produced.

## 4. Proposed use of TPU compute

### 4.0 Compute scaling rationale

The full pilot #2 + Band B + Band C sequence consumed **35 GPU-hours** on a single RTX 4080 (observed wall-clock, recorded in `pilot_results.json` provenance fields). The RSSM training step is the dominant cost; at batch size 16, sequence length 50, latent 32, GRU width 200, the step is arithmetically near the RTX 4080's peak for fp16. A TPU v3-8 delivers 2–3× the RTX 4080's effective throughput on RSSM-class workloads per published benchmarks (JAX-XLA comparison, comparable fp16 batches). Thus:

- 1 primary-pair N=5 run ≈ 7 GPU-hours (RTX 4080) ≈ **3 TPU v3-8 hours**
- 1 full 3-pair × 5-seed pilot ≈ 35 GPU-hours ≈ **15 TPU v3-8 hours**
- Post-1 horizontal scale (7 new pairs × N=5) ≈ 49 GPU-hours ≈ **20 TPU v3-8 hours**

These are conservative upper bounds; with JAX/XLA tuning they should shrink further. A calibration run in Month 1 (§4.1) will produce actual GPU-hr → TPU-hr conversion benchmarks and be reported in the first monthly update.

### 4.1 Month 1 (~40 TPU-hours) — validate pipeline + begin branch C exploration

With the primary-pair hypothesis falsified at N=10, Month-1 priorities shift from "confirm §8 via A10/A11 generality ablations" to "calibrate the TPU pipeline and begin Q1/Q2/Q3 exploration." The A10/A11 ablations are de-prioritized because they would generalize a claim that has been rejected; they remain valuable for a future negative-results report but are not blocking Month-1 work.

- **Replicate Band C verdict on TPU** to validate the GPU→TPU pipeline and produce the GPU-hr↔TPU-hr calibration benchmark. ~10 TPU-hours.
- **Sprint 1 begin — Q1-C contrastive RSSM + disagreement-weighted ensemble** (see `reviews/research_directions.md` §2 and §6 for full design). Swap reconstruction loss for contrastive latent prediction + ensemble disagreement weighting, re-run the primary pair at N=5 as a head-to-head against the Band C baseline (ratio 1.036). Pre-registered success = ratio ≥ 1.20 with LOO min ≥ 1.05; pre-registered kill = ratio ≤ 1.05 (no improvement detectable over falsified baseline). ~15 TPU-hours.
- **A10 adversarial pair** (CartPole → DMC-finger-spin, non-pendular target) — N=5, run opportunistically to bound the negative-result paper. Not blocking Q1/Q2/Q3 sprints. ~5 TPU-hours.
- **Buffer / debug / unexpected rerun budget.** ~10 TPU-hours.

### 4.2 Months 2–3 (~60 TPU-hours) — research program exploration

**Priority order and allocation subject to Month-1 verdicts.** The research program's three questions (Q1/Q2/Q3 in §1) are explored in parallel, with compute allocated based on which thread shows the strongest signal. The plan below represents an upper-bound commitment; actual allocation is adjusted each month in the monthly TRC report.

- **Post-1 horizontal skill-library scale** (~30 TPU-hours) — extend the skill library from ~3 skills to ~10 skills via 7 additional source-target pairs spanning DMControl (cheetah, walker, hopper, quadruped) and MetaWorld (pick-place, reach, button-press). This is the empirical substrate required to make Q2 (skill selection) and Q3 (multi-skill composition) *testable*: with only 3 skills, statistical power for these questions is insufficient.

- **Q1 exploration — contrastive RSSM + ensemble disagreement** (~15 TPU-hours) — replace reconstruction loss with disagreement-weighted contrastive loss on the existing `EnsembleRSSMCore` (see `reviews/research_directions.md` §2 for full mechanism and pre-registered success metric). Tests whether physics-grounded latent representations transfer better than pixel-prediction-grounded ones, both within-task-type and cross-action-type.

- **Q3 exploration — transfer acceleration** (~15 TPU-hours) — initial screening of Kickstarting (distillation with decaying coefficient) and EWC-protected backbone loading against the current `load_state_dict` baseline. Both methods are well-motivated but unvalidated in the specific context of RSSM subset transfer. See `reviews/transfer_acceleration_review.md` for full design space.

**Q2 exploration** (skill selection) is deferred to Months 3+ because it requires the 10-skill library from Post-1 to be empirically tractable.

### 4.3 Month 4+ (compute-tapered) — dissemination

Publication decisions depend on discovery quality, not on a predetermined schedule. The author commits to:

- **Public release of all seed-level data, code, preregistration, and amendment history** under Apache 2.0 regardless of discovery outcome.
- **A methodology blog post** within 2 weeks of Band C verdict (unconditional commitment).
- **Workshop or conference submission** *only if* one or more of the three questions produces a result that is (a) robust under adversarial review, (b) rare enough in the published literature to warrant attention, and (c) falsifiable against its own null.
- **A public exploratory report** documenting negative and null results from the program, with enough detail to save future researchers from rediscovering the same dead ends.

**Initial ask: 1 TPU v3-8 on-demand + preemptible overflow for 30 days (~40 TPU-hours),** renewable monthly on production of Month-1 deliverables. This is intentionally sober — smaller first allocation, renewed with evidence, per TRC's standard workflow. Renewal decisions will be driven by which of Q1/Q2/Q3 are showing empirical traction, not by a predetermined publication pipeline.

## 5. Deliverables committed

Three hard commitments, regardless of Band C outcome:

1. **Open-source code and data under Apache License 2.0.** All code, preregistration, seed-level JSON result artifacts, and amendment history remain in the public repository (GitLab primary, GitHub mirror). Reproducibility script (`scripts/reproduce_headline.py`) reads seed-level JSONs and regenerates the paper's headline table and figures, runnable on a single CPU in under 2 minutes.
2. **Monthly TRC progress report** — TPU-hours spent, experiments completed, outcomes against the preregistered kill/pass criteria, next-month plan. Supports TRC's renewal decisions with full information.
3. **Blog post on preregistration-grade methodology** for solo-dev RL — a reusable blueprint independent of the scientific result. Published within 2 weeks of Band C verdict.

One conditional commitment:

4. **Workshop paper submission** to RLC 2026 or NeurIPS 2026 workshop — *only if* Band C passes pre-registered thresholds AND A10 + A11 ablations support the mechanism claim. If these conditions are not met, the workshop submission is explicitly skipped in favor of a stronger main-track submission 3–6 months later from Post-1 horizontal-scale data.

Stretch items (explicitly not committed, listed only to describe the research roadmap):
- Q1-C contrastive RSSM experiment (see `reviews/research_directions.md` §2)
- Q3-A kickstarting transfer acceleration (see `reviews/transfer_acceleration_review.md`)
- POST-007 multi-skill composition (see `reviews/post_pilot_backlog.md`)

These are deferred to Month 3+ if TRC compute and results permit.

## 6. Principal investigator

**Jérémie Mortier** — independent researcher based in France. Seven years of professional software engineering since graduating with an MSc in IT Engineering from Epitech. Entered Epitech without prior programming background and advanced on the accelerated track (year 1 → year 2 early entry), completing the normally 5-year curriculum in 4 years. Spent the fourth year abroad at Epitech's Daegu (South Korea) campus, specializing in game development. Hired by Epitech as a teaching assistant during studies.

Engineering track record relevant to this application:

- **Lead developer at Piepacker / Jam.gg** (French gaming startup, 2020–2022) — architected the studio's title "Arsène Bomber" (currently available on Steam) and wrote a significant portion of its codebase. Lead architect responsibility on a shipped commercial product.
- **Indie mobile game self-publishing** (pre-2020) — shipped a mobile game that reached 50,000+ organic installs with under €200 paid acquisition, two-person team. Established the self-directed shipping pattern now applied to Ragnarok.
- **Three years of contract engineering at Stormshield** (Airbus Defence and Space subsidiary, French cybersecurity, 2023–present) — current paid engagement, providing both financial self-sufficiency for unfunded research and the engineering-rigor context reflected in Ragnarok's preregistration-grade methodology.

Gaming background is worth surfacing explicitly for a reinforcement-learning research application: games are the canonical RL environment, DeepMind built its reputation on Atari → Go → StarCraft, and the trajectory from shipping games to studying agent transfer learning is a coherent one, not a pivot.

What this background establishes for a compute-grant reviewer: (a) a seven-year track record of shipping complex systems solo or in small teams, (b) comfort with architecture decisions at scale, and (c) a pattern of self-directed work that succeeds without institutional backing — which is precisely the mode of research Ragnarok represents.

No current academic affiliation; no publication track record yet. What exists as of this application:

- **Full preregistered pilot** (40-run primary dataset + Band B rescue N=5 + Band C N=10 extension in flight), all seed-level data tracked in git for reviewer inspection.
- **11 timestamped preregistration amendments**, every one with commit SHA, rationale, and — critically — a solo-initiated chronology audit that identified and corrected one integrity defect (v3.5 → v3.6) before any external reviewer saw it.
- **444 passing tests** covering RSSM transferable subset, skill crystallization, SAC policy, curiosity, world-model trainer, pilot pipeline.
- **Four adversarial multi-agent reviews at pre-submission gates** (`reviews/pre_trc_4agent_review_2026-04-17.md`), with all critical corrections integrated before submission.

**Transparency on repository provenance.** The Git repository was initialized in January 2023 for an unrelated game-development project (multiplayer C/C++, archived March 2025 and dormant for 13 months before the April 2026 repurposing). The tag `rl-project-start` marks commit `3cf847d` on 2026-04-12, which begins the RL research era and wipes the prior codebase clean; reviewers can isolate the RL-era history with `git log rl-project-start..HEAD`. The old commits are preserved rather than deleted because rewriting history to hide them would be inconsistent with the project's stated integrity norms.

**Transparency on LLM-assisted workflow.** Implementation (code and documentation drafting) and multi-agent reviews are executed with Anthropic's Claude under sustained human review. The research question, hypothesis choices, preregistration thresholds, kill criteria, result interpretation, and arbitration of every factual claim are the PI's work; the multi-agent reviews approximate peer review at solo-dev scale but do not substitute for the external peer review that workshop submission will provide. This disclosure is present once, here, and once in §3 of this proposal, and once in the README — three places, because that kind of workflow deserves transparency at each entry point into the project.

**Funding and time commitment.** The project is self-funded, pursued alongside paid contract engagements at Stormshield (Airbus Defence and Space subsidiary, French cybersecurity sector). No external research grants, no institutional backing.

## 7. Why now, why TPU compute

The Band C N=10 extension currently running will, by end of 2026-04-17, give a clean first-test verdict. That verdict conditions the *framing* of the follow-up work but does not change the TRC ask: regardless of outcome, the next six months' value comes from exploring Q1/Q2/Q3 across a larger skill library and multiple architectural variants.

Current hardware is a single RTX 4080 GPU in a home workstation. Pilot #2 + Band B + Band C consumes approximately 35 GPU-hours end-to-end. Post-1 horizontal scale alone is estimated at 120+ GPU-hours of N=5 runs across 7–10 new pairs, before considering Q1 contrastive world models, Q3 transfer acceleration, or Q2 skill selection. The GPU is the hard bottleneck between ideas and verdicts.

TPU access would **not** accelerate the scientific thinking (that's what preregistration forces to the front) — it would directly remove the execution bottleneck, at the exact moment when the research program's design space has been mapped and the priority question is *which of these threads yields a discovery worth pursuing further*.

---

## 8. Research philosophy and output contract

**The primary output of this project is scientific discovery, not a specific publication.** A workshop or conference paper is a natural byproduct *if* discoveries warrant it — but the deliverable pitched to TRC is the exploration itself, not the production of a paper against a submission deadline.

Concretely, this means:

- **Publication decisions are discovery-driven, not schedule-driven.** If one of Q1/Q2/Q3 yields a robust, falsifiable, and genuinely new result, that result drives the paper timing and venue choice (possibly including venues higher than workshop tier). If no thread yields a publishable positive result, the program publishes a thorough negative-and-null-results report rather than forcing a marginal paper.
- **Transparent methodology is non-negotiable, always.** Regardless of which thread produces results, the preregistration amendments, seed-level data, multi-agent reviews, chronology audits, and reproducibility artifacts are committed to public git in real time, not staged for a submission bundle.
- **Open-source exploration, with the option to keep a specific trained artifact reserved.** All code, experimental logs, and methodology documents are open under Apache 2.0. If a specific combination of methods yields a trained model (e.g., a multi-skill library with demonstrably useful transfer properties), the *artifact itself* may be reserved for further development into a product or platform — but the reasoning, methodology, and underlying algorithms enabling it are always published openly, so the community can reproduce the science even if not the exact artifact.
- **The author chooses honesty over narrative convenience.** When a result is fragile, it is reported as fragile (as with Band B's p=0.259). When a claim needs to be narrowed after adversarial review, it is narrowed (as with §2's positioning against Gato, RT-X, SPiRL). When a preregistration defect is detected, it is corrected transparently (as with v3.5 → v3.6 chronology audit).

- **Existence of a mechanism matters more than its magnitude.** A small confirmed effect (e.g. 5–15% reduction in episodes-to-mastery, rigorously established with N≫10 across multiple pairs) is treated as a scientifically important result. The reason: existence confirms that a transferable mechanism crosses the action-space-type boundary, which is the prerequisite for *any* future research on amplifying that mechanism — through physics-grounded world models (Q1), contextual skill selection (Q2), transfer acceleration via kickstarting or EWC (Q3), or directions not yet imagined. A rigorous small-effect result is scientifically stronger than an inflated large-effect one, and is treated as such in publication venue choice and narrative framing.

**Why this framing is appropriate for TRC specifically.** TRC's mission statement explicitly values sharing research "through peer-reviewed publications, open source code, blog posts, or other means." The "other means" is load-bearing here: a thorough exploratory program with reproducible methodology and public negative results has independent scientific value, even absent a specific paper. The author commits to the full spectrum of TRC's valued output modes — not just paper-and-done.

---

*This proposal is committed to the public repository at
[`docs/compute_application/research_proposal.md`](../../docs/compute_application/research_proposal.md).
Any changes after TRC submission are timestamped via git and referenced here.*
