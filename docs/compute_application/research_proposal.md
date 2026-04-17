# Ragnarok — Research Proposal

**Principal Investigator:** Jérémie Mortier (independent researcher, France)
**Project:** Modular reinforcement learning with cross-action-space skill transfer
**Repository:** https://gitlab.com/mortier.jeremie/ragnarok (mirror: https://github.com/tynitoon/ragnarok)
**Contact:** mortier.jeremie@gmail.com
**Date:** 2026-04-17

---

## 1. Project summary

Ragnarok is an open-science reinforcement-learning (RL) research project investigating a specific, published-gap question: **can an RL agent crystallize a learned skill from one task and transfer its latent trunk to a new task with a *different action space type* (discrete ↔ continuous), and learn measurably faster than from scratch?**

The claim matters because the mainstream RL transfer-learning literature (Progressive Networks, Modular RL Policies, Soft Modularization, SPiRL, Options-Critic) almost exclusively studies transfer within a fixed action-space type. Cross-action-type transfer with a shared latent trunk — a mechanism-level question about what part of a Dreamer-style world-model truly generalizes — is a gap I have not been able to find resolved in the published record.

The project is conducted with **preregistration-grade methodology**: every hypothesis, threshold, and analytic choice is committed to a public `preregistration.md` document **before** the data it evaluates exists, with git-history-verified chronology and timestamped amendments for any revision. A solo-initiated chronology audit (`reviews/chronology_audit.md`) already corrected one integrity defect in the preregistration before external review — a level of self-scrutiny uncommon in solo-dev research.

## 2. Scientific contribution

**The architecture.** Ragnarok uses a Recurrent State-Space Model (RSSM) world model in the Dreamer family. Ragnarok identifies a *transferable subset* — GRU core + prior distribution + posterior distribution — that is shape-compatible across tasks with different observation and action dimensions, because it operates on `cat(h, z)` latent features upstream of task-specific input encoders and action decoders. At transfer time, the agent loads this subset via `load_state_dict` with strict shape checking on the transferable subset only, then switches its policy to operate on latent features (`acting_policy_mode = latent`) rather than raw observations.

**The preregistered hypothesis (§8 of `preregistration.md`).** On the primary pair CartPole-v1 → MountainCar-Continuous-v0, the project predicted a restricted mean survival time (RMST) ratio ≥ 1.30 with log-rank one-sided p < 0.10, conditional on passing a mechanism check (acting policy must be on latent mode with a crystallized skill loaded).

**Current empirical state (2026-04-17).** Pilot #2 (N=5 seeds) showed RMST ratio 1.238 fragile under leave-one-out (dropping seed 46 collapses the ratio to 1.049). Band B rescue (N=5 fresh seeds on primary with corrected warmup parameter) yielded ratio **1.605**, **robust under leave-one-out** (minimum LOO ratio = 1.435, well above the Band A threshold of 1.30). However, p-value at N=5 is 0.259 — directionally consistent but statistically underpowered.

A pre-registered **Band C extension** (N=10 pooled, seeds 47–56) is currently running and will complete ~2026-04-17 evening. The extension is specified **before** seeds 52–56 are launched, with kill criteria (ratio < 1.20, p ≥ 0.20, or LOO min < 1.00 triggers project pivot) and pass criteria (ratio ≥ 1.30, p < 0.10 in both asymptotic and permutation tests, LOO min ≥ 1.15 triggers §8 full pass).

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

### 4.1 Immediate needs (Month 1, ~60 TPU-hours)

- **Replicate Band C verdict on TPU** to establish confidence that the GPU-to-TPU pipeline produces consistent results. ~15 TPU-hours (1 paired primary-pair sweep).
- **Run the A10 adversarial pair ablation** (CartPole → DMC-finger-spin) already pre-registered in §12.5 but not yet executed due to GPU-hour budget. ~10 TPU-hours.
- **Run the A11 GRU-shuffled weights ablation** confirming mechanism (§10 B0 clause 3). ~5 TPU-hours.
- **Buffer/debug budget**. ~30 TPU-hours.

### 4.2 Scale-up phase (Months 2–3, ~100 TPU-hours)

- **Post-1 horizontal scale**: extend the skill library from 3 skills to 10 skills via 7 additional source-target pairs across DMControl and MetaWorld. This is the empirical backbone of any follow-up main-track paper. ~60 TPU-hours.
- **Q1-C contrastive RSSM** experiment (replace reconstruction loss with disagreement-weighted contrastive loss on the existing `EnsembleRSSMCore`). Ablation with on/off switch and OOD distractor benchmark. ~30 TPU-hours. (See `reviews/research_directions.md` §2 for full rationale.)
- **Q3-A kickstarting and Q3-B EWC** transfer-acceleration experiments (see `reviews/transfer_acceleration_review.md`). ~10 TPU-hours for initial screening.

### 4.3 Publication & dissemination (Month 4, negligible compute)

- Final N=20 seeds analysis for workshop submission (primary pair + 1–2 scaled secondaries).
- Paper draft, code release, seed-level data release, blog post.

**Total requested for initial 30-day allocation: 1 TPU v3-8 on-demand + preemptible overflow, renewable on production of the Month-1 deliverables.**

## 5. Deliverables committed

Per TRC's participation norm ("expected to share research through peer-reviewed publications, open source code, blog posts, or other means"):

1. **All code remains open source under Apache License 2.0** on the public repository (GitLab primary, GitHub mirror), with seed-level JSON result artifacts tracked in git for audit-level reproducibility.
2. **Workshop paper submission** to RLC 2026 workshop track or NeurIPS 2026 RL workshop, conditional on Band C outcome (full pass → §8 PASS paper; intermediate → B0 modest paper; kill → pivot to Post-1 and submit main-track instead).
3. **Blog post / technical write-up** on preregistration-grade methodology as a blueprint for solo-dev research rigor. This stands even if the scientific result is negative — the methodology itself is a publishable artifact.
4. **Reproducibility bundle** (POST-006 in `reviews/post_pilot_backlog.md`): `scripts/reproduce_headline.py` that reads seed-level JSONs and regenerates the paper's headline table and figures.
5. **Monthly progress reports** to TRC summarizing TPU-hours spent, experiments completed, and deliverables produced, enabling renewal decisions with full information.

## 6. Principal investigator

**Jérémie Mortier** — independent researcher based in France. MSc in IT Engineering from Epitech (French engineering school with international campuses in San Francisco, Los Angeles, Berlin, Strasbourg). Three years of contract engineering at **Stormshield**, a French cybersecurity vendor subsidiary of Airbus Defence and Space — a context that demands the kind of engineering rigor reflected in the Ragnarok methodology.

No academic affiliation; no publication track record yet. Ragnarok is a **5-day-old intensive project** as an RL research effort: the pivot commit `3cf847d` ("new projet") is dated 2026-04-12, verifiable via `git log 3cf847d`. For full transparency: the underlying Git repository dates back to 2023 as an unrelated game-development project (multiplayer C/C++, archived March 2025 and dormant for 13 months before the April 2026 pivot). A tag `rl-project-start` marks the pivot commit so reviewers can isolate the RL-era history with `git log rl-project-start..HEAD`.

The short RL-era timeline is not a qualifier; it is a signal. Reaching preregistered-pilot stage with full chronology audit, multi-agent review, 444 passing tests, and three complete pilot pairs (40 runs + Band B rescue + Band C N=10 extension in progress) in five days required sustained focus, long evening sessions, and heavy use of LLM-assisted development under ongoing human supervision. The research question — whether a skill's latent trunk transfers across action-space types — is the PI's own; the LLM writes the implementation under direction, and the PI arbitrates every scientific decision. See the disclosure paragraph at the end of §3 for the full workflow declaration.

The work is self-funded, pursued alongside paid contract engagements at Stormshield. The preregistration, chronology audit, and multi-agent review processes were adopted specifically to compensate for the lack of institutional peer review available to solo researchers — and they proved effective at catching one integrity defect (the B0 chronology phrasing in v3.5, corrected in v3.6) before any external reviewer saw it.

## 7. Why now, why TPU compute

The Band C N=10 extension currently running will, by end of 2026-04-17, give a clean verdict: either §8 primary passes and I write a solid workshop paper with a confirmed novel contribution, or I pivot to Post-1 horizontal scale and aim for a stronger main-track submission 3–6 months later. **Either path is compute-bounded, not idea-bounded.**

Current hardware is a single RTX 4080 GPU in a home workstation. Pilot #2 + Band B + Band C consumes approximately 35 GPU-hours end-to-end. Post-1 horizontal scale alone is estimated at 120+ GPU-hours of N=5 runs across 7–10 new pairs, before considering Q1-C, Q3-A/B, or any follow-up. The GPU is a hard bottleneck.

TPU access would **not** accelerate the scientific thinking (that's what preregistration forces to the front) — it would directly remove the execution bottleneck between ideas and empirical verdicts, at the exact moment (first publishable result landing within weeks) when the marginal value of compute is highest.

---

*This proposal is committed to the public repository at
[`docs/compute_application/research_proposal.md`](../../docs/compute_application/research_proposal.md).
Any changes after TRC submission are timestamped via git and referenced here.*
