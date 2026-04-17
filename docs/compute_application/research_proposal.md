# Ragnarok — Research Proposal

**Principal Investigator:** Jérémie Mortier (independent researcher, France)
**Project:** Modular reinforcement learning with cross-action-space skill transfer
**Repository:** https://gitlab.com/mortier.jeremie/ragnarok (mirror: https://github.com/tynitoon/ragnarok)
**Contact:** mortier.jeremie@gmail.com
**Date:** 2026-04-17

---

## 1. Project summary

Ragnarok investigates a bottleneck in embodied-agent skill reuse: **real-world agents need libraries of both discrete-choice primitives (mode switches, gripper open/close, tool selection) and continuous-control primitives (joint torques, wheel velocities) — yet existing RL transfer methods almost all assume a fixed action-space type within a given skill library.** If a robot's "stabilize a cart" skill is discrete but its new task requires continuous control, the skill is wasted.

Ragnarok tests whether the **latent trunk of a Dreamer-style RSSM world model** — specifically, the GRU core, prior, and posterior distributions, loaded via shape-checked `load_state_dict` without the task-specific encoder or action head — can carry *dynamics knowledge* across this action-type boundary. The research question:

> **Given a skill crystallized on task A (one action-space type) and a new task B (different action-space type), does loading the RSSM transferable subset and switching the agent to act on latent features yield measurably faster mastery than training from scratch?**

This is narrower than "cross-embodiment transfer" (Gato 2022, RT-X 2024) which uses single-model tokenization, and narrower than "skill priors" (SPiRL 2020) which assume homogeneous action spaces. Progressive Networks (Rusu 2016) and Options-Critic (Bacon 2017) handle one action-space type at a time. **The specific gap Ragnarok tests is: shape-checked subset transfer across action-type boundary in the Dreamer-RSSM family.**

The project is conducted with **preregistration-grade methodology**: every hypothesis, threshold, and analytic choice is committed to a public `preregistration.md` document **before** the data it evaluates exists, with git-history-verified chronology and timestamped amendments for any revision. A solo-initiated chronology audit (`reviews/chronology_audit.md`) corrected one integrity defect in the preregistration before external review — this level of self-scrutiny is the project's methodological signature.

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

**Current empirical state (2026-04-17).** Pilot #2 (N=5 seeds, 3 pairs × 2 arms) showed primary-pair RMST ratio 1.238, fragile under leave-one-out (dropping seed 46 → 1.049). Band B rescue (N=5 fresh seeds on primary with a corrected warmup parameter) yielded ratio **1.605** (robust under leave-one-out, minimum LOO ratio = 1.435, above the §8 Band A threshold of 1.30). However, log-rank p-value at N=5 is 0.259 — directionally consistent with the hypothesis but statistically underpowered.

A pre-registered **Band C N=10 extension** (seeds 47–56 pooled) is running at the time of writing and will complete ~2026-04-17 evening (Band C pre-spec committed at SHA `a0c1140`, before seeds 52–56 launched). Pass, intermediate, and kill criteria are defined numerically; see `preregistration.md` §13 v3.7 for details.

**Honest positioning:** the primary pair (CartPole→MountainCar-Continuous) is, the author now acknowledges after adversarial review, the most favorable cross-action-type pair imaginable — both pendular-class systems, both ~4D observations, with CartPole's discrete action semantically close to a discretized MountainCar-Continuous force. The preregistration's secondary pairs (acrobot→cartpole-swingup, pendulum→cartpole-swingup) and the preregistered A10 adversarial pair (Pendulum→DMC-finger-spin, non-pendular target) are intended to test generality; A10 has not yet been executed and is explicitly a blocking deliverable for any workshop submission making a generality claim. Full honesty on this limitation is surfaced both in the paper's eventual methods section and in §10 B0 (the modest-but-reliable fallback path) of the preregistration.

## 3. Methodology as rigor signal

Beyond the scientific claim, the methodological artifacts of the project are themselves a contribution:

- **Preregistration `preregistration.md`** with 11 timestamped amendments, every one citing a specific commit SHA and rationale.
- **Multi-agent peer review** at every milestone (3–6 specialized LLM agents: RL methodology, code review, strategy, devil's advocate, architecture). Dissent is logged in `reviews/` and resolved before execution.
- **Chronology audit** (`reviews/chronology_audit.md`) — solo-initiated audit that found and corrected a phrasing defect in the B0 fallback plan's pre-data claim.
- **444 tests passing** (pytest).
- **Seed-level JSON artifacts tracked in git**: `pilot_results.json`, `pilot_bandb_results.json`, `pilot_bandc_results.json` contain every run's evaluation curve, wall-clock time, git SHA, and provenance.
- **Kill criteria at every decision gate** (`preregistration.md` §11). Conditions under which the project is explicitly abandoned are pre-specified, not redefined post-hoc.

This methodology is, to my knowledge, more rigorous than most single-author workshop submissions. The preregistration + audit + multi-agent review pattern is proposed as a reusable blueprint for other solo-dev or small-lab RL research.

**Disclosure of LLM-assisted workflow.** Ragnarok is developed using LLM-assisted workflows with Anthropic's Claude: code generation, documentation drafting, and the multi-agent reviews cited above (which are Claude-agent-based reviews rather than external human peers). This arrangement is declared openly to match the transparency norms of NeurIPS 2025 / ICML 2025. **All scientific decisions — the research question itself, hypothesis choice, preregistration thresholds, kill criteria, result interpretation, chronology audit initiation, and final arbitration of truth — are made and validated by the human principal investigator, who retains sole scientific and ethical responsibility.** The LLM executes drafts and proposals under sustained human review. The multi-agent review process is presented for what it is: a tool for approximating institutional peer review at solo-dev scale, not a substitute for external human peer review (which the workshop submission itself will provide).

## 4. Proposed use of TPU compute

### 4.0 Compute scaling rationale

The full pilot #2 + Band B + Band C sequence consumed **35 GPU-hours** on a single RTX 4080 (observed wall-clock, recorded in `pilot_results.json` provenance fields). The RSSM training step is the dominant cost; at batch size 16, sequence length 50, latent 32, GRU width 200, the step is arithmetically near the RTX 4080's peak for fp16. A TPU v3-8 delivers 2–3× the RTX 4080's effective throughput on RSSM-class workloads per published benchmarks (JAX-XLA comparison, comparable fp16 batches). Thus:

- 1 primary-pair N=5 run ≈ 7 GPU-hours (RTX 4080) ≈ **3 TPU v3-8 hours**
- 1 full 3-pair × 5-seed pilot ≈ 35 GPU-hours ≈ **15 TPU v3-8 hours**
- Post-1 horizontal scale (7 new pairs × N=5) ≈ 49 GPU-hours ≈ **20 TPU v3-8 hours**

These are conservative upper bounds; with JAX/XLA tuning they should shrink further. A calibration run in Month 1 (§4.1) will produce actual GPU-hr → TPU-hr conversion benchmarks and be reported in the first monthly update.

### 4.1 Month 1 (~40 TPU-hours) — validate + execute preregistered ablations

- **Replicate Band C verdict on TPU** to validate the GPU→TPU pipeline and produce the calibration benchmark. ~10 TPU-hours.
- **A10 adversarial pair** (CartPole → DMC-finger-spin, non-pendular target) — preregistered in §12.5 and **blocking any cross-action-type generality claim in the paper**. N=5. ~10 TPU-hours.
- **A11 GRU-shuffled weights ablation** — preregistered mechanism filter for §10 B0 path. N=5. ~5 TPU-hours.
- **Buffer / debug / unexpected rerun budget.** ~15 TPU-hours.

### 4.2 Months 2–3 (~60 TPU-hours, stretch) — Post-1 horizontal scale

- **Post-1 horizontal scale**: extend the skill library from 3 to ~10 skills via 7 additional source-target pairs across DMControl (cheetah, walker, hopper) and selected MetaWorld tasks. This is the empirical backbone of a future main-track submission and the only item truly compute-bounded (vs. idea-bounded) today. ~40 TPU-hours.
- **Buffer + analysis compute**. ~20 TPU-hours.

### 4.3 Month 4 (negligible compute) — dissemination

- Final analysis, paper draft (if Band C + A10 + A11 pass the pre-registered thresholds), code release, seed-level data release, blog post publication.

**Initial ask: 1 TPU v3-8 on-demand + preemptible overflow for 30 days (~40 TPU-hours),** renewable monthly on production of Month-1 deliverables. This is intentionally sober — smaller first allocation, renewed with evidence, per TRC's standard workflow. A post-workshop re-application would expand the ask for a main-track paper's compute budget.

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

**Jérémie Mortier** — independent researcher based in France. MSc in IT Engineering from Epitech (French engineering school with international campuses in San Francisco, Los Angeles, Berlin, Strasbourg). Three years of contract engineering at **Stormshield**, a French cybersecurity vendor subsidiary of Airbus Defence and Space — a context that demands the kind of engineering rigor reflected in the Ragnarok methodology.

No current academic affiliation; no publication track record yet. What exists as of this application:

- **Full preregistered pilot** (40-run primary dataset + Band B rescue N=5 + Band C N=10 extension in flight), all seed-level data tracked in git for reviewer inspection.
- **11 timestamped preregistration amendments**, every one with commit SHA, rationale, and — critically — a solo-initiated chronology audit that identified and corrected one integrity defect (v3.5 → v3.6) before any external reviewer saw it.
- **444 passing tests** covering RSSM transferable subset, skill crystallization, SAC policy, curiosity, world-model trainer, pilot pipeline.
- **Four adversarial multi-agent reviews at pre-submission gates** (`reviews/pre_trc_4agent_review_2026-04-17.md`), with all critical corrections integrated before submission.

**Transparency on repository provenance.** The Git repository was initialized in January 2023 for an unrelated game-development project (multiplayer C/C++, archived March 2025 and dormant for 13 months before the April 2026 repurposing). The tag `rl-project-start` marks commit `3cf847d` on 2026-04-12, which begins the RL research era and wipes the prior codebase clean; reviewers can isolate the RL-era history with `git log rl-project-start..HEAD`. The old commits are preserved rather than deleted because rewriting history to hide them would be inconsistent with the project's stated integrity norms.

**Transparency on LLM-assisted workflow.** Implementation (code and documentation drafting) and multi-agent reviews are executed with Anthropic's Claude under sustained human review. The research question, hypothesis choices, preregistration thresholds, kill criteria, result interpretation, and arbitration of every factual claim are the PI's work; the multi-agent reviews approximate peer review at solo-dev scale but do not substitute for the external peer review that workshop submission will provide. This disclosure is present once, here, and once in §3 of this proposal, and once in the README — three places, because that kind of workflow deserves transparency at each entry point into the project.

**Funding and time commitment.** The project is self-funded, pursued alongside paid contract engagements at Stormshield (Airbus Defence and Space subsidiary, French cybersecurity sector). No external research grants, no institutional backing.

## 7. Why now, why TPU compute

The Band C N=10 extension currently running will, by end of 2026-04-17, give a clean verdict: either §8 primary passes and I write a solid workshop paper with a confirmed novel contribution, or I pivot to Post-1 horizontal scale and aim for a stronger main-track submission 3–6 months later. **Either path is compute-bounded, not idea-bounded.**

Current hardware is a single RTX 4080 GPU in a home workstation. Pilot #2 + Band B + Band C consumes approximately 35 GPU-hours end-to-end. Post-1 horizontal scale alone is estimated at 120+ GPU-hours of N=5 runs across 7–10 new pairs, before considering Q1-C, Q3-A/B, or any follow-up. The GPU is a hard bottleneck.

TPU access would **not** accelerate the scientific thinking (that's what preregistration forces to the front) — it would directly remove the execution bottleneck between ideas and empirical verdicts, at the exact moment (first publishable result landing within weeks) when the marginal value of compute is highest.

---

*This proposal is committed to the public repository at
[`docs/compute_application/research_proposal.md`](../../docs/compute_application/research_proposal.md).
Any changes after TRC submission are timestamped via git and referenced here.*
