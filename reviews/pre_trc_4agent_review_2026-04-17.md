# Pre-TRC 4-Agent Objective Review

**Date:** 2026-04-17 (afternoon, during Band C execution)
**Purpose:** Adversarial / objective review of the compute-application package (research proposal + README + preregistration + methodology) before submission to Google TPU Research Cloud and before workshop paper drafting. Commissioned by the PI at the decision gate between "submit now" vs "submit after Band C verdict" and between "write workshop paper" vs "blog post methodology".

**Methodology:** Four specialized agents spawned in parallel, each with access to the full repository artifacts. Instructions: be severe, opinionated, non-flattering. Red flags over politeness.

**Agents:**
1. **RL Methodology + Scientific Rigor Reviewer** (NeurIPS Area-Chair level)
2. **TRC Grant Application Reviewer** (ex-Google researcher, TRC panel experience)
3. **Devil's Advocate** (hostile Reviewer 2 simulation)
4. **Strategy / Science Communication Reviewer** (career + narrative + timing)

**Commissioning rationale:** after Band B rescue landed ratio 1.605 with p=0.259 (underpowered), and with Band C N=10 extension running, the PI wanted independent critique of the dossier before irreversible decisions (TRC submission, paper drafting). The multi-agent review pattern is a standing practice in this project (`calibration_velocity.md`, standing instruction "dissent > consent at every phase gate"). This instance is documented here for:
- Audit trail reproducibility (any reviewer can verify what feedback existed at what commit SHA)
- Paper supplementary materials (if this review is cited in methodology discussion)
- Future learning about what the LLM-agent review process actually flags vs misses

---

## 1. RL Methodology + Scientific Rigor Reviewer — Verdict

**Global:** weak acceptance for workshop — methodology above-median for solo-dev, but current signal (N=5, p=0.259) not workshop-ready without Band C N=10 landing in Band A, and amendment cadence raises reviewer flags needing proactive defense.

**Strengths:**
1. Preregistration discipline unusually strong for solo-dev; §8 threshold genuinely pre-pilot and git-verifiable.
2. Self-initiated chronology audit is credibility-positive. Most papers don't do this even when called out.
3. Mechanism-isolation ablations (A9, A11) are the right shape.
4. Kill criteria real, not theater.
5. Permutation + LOO robust min 1.435 on Band B is the strongest empirical signal.

**Weaknesses:**
- **CRITICAL.** Amendment fatigue (v3 → v3.7 in 4 days) is itself a p-hacking surface. Band C added *after* full unblinding of N=5. Hostile reviewer will allege seed-hacking even if none occurred.
- **MAJOR.** Power at N=5 and even N=10 insufficient. Prereg §5 concedes N=10 power ≈ 0.40.
- **MAJOR.** Primary pair CartPole→MCC is the most favorable cross-action-type pair imaginable (both pendular-class, both ~4D obs).
- **MAJOR.** A11 (GRU-shuffled) mechanism ablation unexecuted. Submitting with A11 "planned" is reviewer-bait.
- **MINOR.** Mechanism check verifies plumbing not mechanism ("acting_policy_mode=latent" only confirms flag is set).

**Required corrections before submission:**
1. Run A11 and A10 *before* writing paper. ~25 GPU-h total.
2. Write amendment chronology as a single supplementary figure (timeline plot).
3. Pre-declare Band C interpretation matrix as a table.
4. Reframe TRC proposal to de-emphasize "5-day". Same facts, better framing.
5. Add prospective-power paragraph to the paper itself.

**Open question reviewer will ask:** *"If Band C lands at ratio=1.4, p=0.08, LOO_min=1.20, what prevents me from reading this as 'you kept adding seeds until p crossed 0.10'?"* No clean answer in current docs.

---

## 2. TRC Grant Application Reviewer — Verdict

**Probabilities:**
- Initial allocation (30-day v3-8 preemptible): **55–65%**
- Full 3–4 month allocation as requested: **30–40%**
- Typical applicant baseline (PhD + advisor): ~70% initial / ~50% full. Ragnarok is ~10–15 points below.

**Key assessments:**
- Profile (freelance + Stormshield + Epitech + solo) = net neutral-to-mild-liability. Stormshield/Airbus mention is the best asset.
- LLM disclosure dosed roughly correctly but too prominent (3 places reads defensive). One clean declaration is enough.
- Repo recycling preempted cleanly with `rl-project-start` tag. Don't change.
- "5-day-old" framing is biggest unforced error. Remove; let methodology speak.

**Proposal weaknesses:**
- "Cross-action-type = published gap" partially defensible, partially vulnerable. Reviewer with 5-min Scholar search finds Gato (Reed 2022), RT-X (2024), SPIRL, Options-Critic, PLAS. Need to narrow claim to "shared-RSSM-trunk cross-action-type transfer in Dreamer family, shape-checked subset" — that IS an unoccupied gap.
- TPU-hour chiffrage amateur (no scaling analysis). Show the GPU-hr → TPU-hr conversion with a benchmark.
- Deliverables over-committed (workshop + blog + reproducibility + monthly + Q1-C + Q3-A/B in 4 months solo). Cut 40%.

**Strategic recommendation: apply in two stages.**
- **Stage 1 (this week, after Band C lands):** small initial TRC ask (30-day v3-8 preemptible, ~60 TPU-h) with specific framing "replicate Band C on TPU + run A10+A11 ablations". Tight, verifiable.
- **Stage 2 (after workshop paper acceptance, ~3 months out):** re-application citing TRC-compute credit on the paper, for Post-1 horizontal scale.

Applying now for full 160 TPU-hr with current evidence is the worst option.

---

## 3. Devil's Advocate (Hostile Reviewer 2) — Verdict

**Reviewer 2 score: 3/10**

**Priority-ordered attacks:**

1. **CRITICAL.** *"Pre-registered is a lie, and the audit proves it."* B0 committed at 13:28 while pilot running with 2/5 seeds complete. Self-auditing after the fact does not cleanse an integrity breach — it documents it. The existence of `chronology_audit.md` is itself the smoking gun.
2. **CRITICAL.** *"The signal is one seed deep."* Pilot #2 ratio 1.238 → 1.049 without seed 46. Band B ratio 1.605 with p=0.259 — indistinguishable from noise. Presenting 1.605 as headline while p=0.259 is advocacy, not science.
3. **MAJOR.** *"Task pair is trivially transferable by construction."* CartPole's discrete "push left/right" is exactly the discretized form of MCC's continuous "force ∈ [-1,1]". Same action semantics, different dtypes. Without A10 data, the claim "cross-action-type transfer in general" has zero empirical support beyond one nearly-isomorphic pair.
4. **MAJOR.** *"Mechanism check does not check mechanism."* `acting_policy_mode=latent` verifies the plumbing. A11 (the real mechanism test) deferred to after the positive-result-seeking work is done — canonical garden-of-forking-paths.
5. **MAJOR.** *"Literature review is gerrymandered."* Three-condition conjunction defined in a way that guarantees zero matches. Successor features, options frameworks, hypernetwork policies, Progressive Networks, cross-embodiment (Zakka 2022, Yang 2023 Robot Parkour) not seriously engaged.
6. **MAJOR.** *"5-day intensive framing is a liability."* 11 prereg amendments + 444 tests + 3 pilot bands + multi-agent review + chronology audit + 220-line research_directions.md in 5 days = LLM-driven artifact generation without external validation.
7. **MAJOR.** *"TRC application leans on a non-result."* Ratio 1.605 at p=0.259 does not meet §8 pass criteria. "Band C tonight" is the fig leaf.
8. **MINOR.** Repo provenance cosmetic but sloppy. Most research repos scaffolded fresh.
9. **MINOR.** CartPole threshold 450/500 = 90%, not 80% as prereg §4.5 specifies. Discrepancy may be a display issue; verify.
10. **CRITICAL.** *"What is the actual scientific contribution?"* One 1.6x ratio on one near-isomorphic task pair, two ablations unexecuted, mechanism unchecked — not a workshop contribution.

**Verdict:** *"Come back with Band C + A10 + A11 all executed at N≥10 with pre-frozen thresholds, on at least one genuinely non-pendular pair, and the floor rises to 5-6/10."*

---

## 4. Strategy / Narrative Reviewer — Verdict

**Thesis:** Methodology is a stronger asset than the scientific result at this stage. Pivot the narrative accordingly.

**Storyline critique:**
- 30-second pitch "cross-action-type skill transfer via shared RSSM trunk" lacks "so what". No explicit connection to embodied agents, foundation models for RL, or robot skill reuse.
- Mechanism without story. Fix: reframe §1.0 elevator pitch around "embodied agent skill reuse bottleneck: real-world robots need both discrete-choice (grippers, mode switches) and continuous-control primitives."

**Competitive positioning:**
- Dreamer V3 / TD-MPC2: downstream of them, OK — assume explicitly.
- Meta-RL / MAML / PEARL: good contrast ("no meta-training required") but must explicitly cite-and-compare or reviewer assumes evasion.
- Cross-action-type USP: defensible if framed as embodied-skill-reuse need. Lost if framed as literature-table gap.

**Timing recommendations:**
- **TRC: submit MAINTENANT, pas plus tard** (stratège's position — NOTE: diverges from TRC reviewer who recommends waiting for Band C; see synthesis).
- Workshop paper: *after* Band C verdict. Not before.
- Blog post methodology: *before* the paper. This weekend.
- Twitter/HN: 2 weeks out, not now.

**Three scenarios post-Band C:**
- **A — Band C pass:** workshop draft with headline + ablation panel, submit RLC workshop, blog post simultaneous. ~7.5/10 workshop outcome.
- **B — intermediate / B0:** resist rescue urge. Commit B0. Workshop paper "modest but reliable" — often better received than forced pass.
- **C — Band C fail:** don't write paper. Pivot Post-1 horizontal scale (6 months), main-track submission. Blog post methodology still publishable.

**Actions within 24h recommended by strategist:**
1. Submit TRC application (proposal is ready).
2. Write blog post "Preregistering against yourself" — 1500 words, this weekend.
3. Draft 3 cold emails (Kessler / Pertsch / Rakelly), send post-verdict.
4. Rewrite §1.0 elevator pitch with embodied-skill angle, commit before Band C verdict.
5. Do NOT touch Twitter/HN for 2 weeks.

---

## Synthesis — Points of Convergence (PI-level truth)

All 4 agents agree on:

1. **Current empirical claim is too fragile** (1 pair, N=5, p=0.259, A10/A11 not executed) to carry a workshop paper as-is.
2. **Methodology is the most stable asset** — possibly more publishable independently than the scientific result.
3. **"5-day intensive project" framing is a liability** in the TRC proposal, even when presented as a signal. Remove or reframe.
4. **Novelty claim must be narrowed** to "shared RSSM trunk in Dreamer family, shape-checked subset loading, Discrete↔Continuous" and explicitly cite Gato, RT-X, SPIRL, Options-Critic as adjacent-not-overlapping.
5. **Mechanism check is insufficient** without A11 executed.
6. **CartPole→MCC is too favorable a primary pair.** A10 (Pendulum→Reacher) is required for the cross-action-type claim to generalize.

## Synthesis — Points of Divergence (PI must arbitrate)

- **TRC submission timing:** strategist says "now"; TRC reviewer says "wait 24-48h for Band C"; RL methodology implicitly supports "wait until A10+A11 run". **PI arbitration 2026-04-17:** wait for Band C verdict (tonight), submit within 48h post-verdict. Reasoning: +15-20 percentage points probability for asymmetric 24-48h cost.
- **Workshop paper submission:** all 4 agents converge on "not today", "only if Band C pass AND A10 + A11 executed". PI accepts.
- **PI priority (declared 2026-04-17):** GPUs > paper. The paper serves the TRC, not the inverse. This reframes all downstream decisions.

## Corrections to execute before TRC submission

### Critical (blocking submission)

1. Narrow novelty claim in `research_proposal.md` §2 with explicit citations to Gato / RT-X / SPIRL / Options-Critic as adjacent-not-overlapping.
2. Remove "5-day-old" framing from `research_proposal.md` §6. Replace with fact-based enumeration.
3. Add compute scaling rationale box in §4 (GPU-hr → TPU-hr benchmark).
4. Cut deliverables §5 by ~40%.
5. Add angle "embodied agent skill reuse" to §1 project summary.
6. Reduce repetition of LLM-assisted disclosure (currently 3 places → 1 clean declaration).

### Important (strengthens but not blocking)

7. Add supplementary chronology figure (timeline plot of amendments with commit SHAs) as `reviews/amendment_chronology.png` or markdown table.
8. Execute A10 + A11 ablations before workshop paper drafting (~25 GPU-h, deferred until TRC compute is granted or run locally over weekend).
9. Write blog post methodology this weekend (independent of Band C outcome).

---

## Changelog / Commit history of this review

- **2026-04-17 ~12:15** — 4 agents spawned, instructions issued.
- **2026-04-17 ~12:30** — all 4 reviews delivered.
- **2026-04-17 ~12:45** — this synthesis document committed at (commit SHA to be inserted at commit time).
- Subsequent integrations of the corrections into `research_proposal.md` and other compute-application docs will reference this document.

---

*This review is part of the public repository's audit trail. Any subsequent revision of the `research_proposal.md` or compute-application package should reference this document by SHA and explicitly note which of the 6 critical corrections were integrated.*
