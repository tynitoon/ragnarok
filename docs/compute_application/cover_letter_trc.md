# Cover Letter — TPU Research Cloud Application

**To:** TPU Research Cloud team, Google Research
**From:** Jérémie Mortier, independent researcher, France
**Re:** Ragnarok — Modular RL with cross-action-type skill transfer
**Repository:** https://gitlab.com/mortier.jeremie/ragnarok (mirror: https://github.com/tynitoon/ragnarok)

---

Dear TRC team,

I am applying for a TPU Research Cloud allocation to validate and extend a preregistered RL research project at a stage where TPU compute specifically removes the execution bottleneck between ideas and empirical verdicts.

**The research question.** Can a Dreamer-style RSSM's latent trunk (GRU core + prior + posterior distributions) transfer across an action-space-type boundary (discrete ↔ continuous) via shape-checked `load_state_dict` loading, and measurably accelerate learning on the target task? This is narrower than cross-embodiment transfer (Gato, RT-X), narrower than skill priors (SPiRL), and narrower than options frameworks (Options-Critic) — but to my knowledge is not resolved in the published record. Importantly, it matters for embodied-agent skill libraries that must span discrete-choice primitives (mode switches, grippers) and continuous-control primitives (joint torques).

**The current empirical state** (2026-04-17): Band B rescue (N=5, primary pair) yielded RMST ratio 1.605, robust under leave-one-out (min 1.435), mechanism check passed, but log-rank p=0.259 — directionally strong, statistically underpowered at N=5. A pre-registered N=10 extension ("Band C", amended into `preregistration.md` §13 v3.7 *before* seeds 52–56 were launched) is completing tonight. The kill criteria and pass criteria are numerical, public, and git-anchored.

**What makes this application distinct from typical solo-researcher applications.** I have no academic affiliation, no prior publications, no advisor. What I have instead is a 5-day intensive research effort with a methodological footprint that is — by design — over-engineered for the data it protects:
- 1364-line preregistration with 11 timestamped amendments, each tied to a commit SHA
- A solo-initiated chronology audit that caught and corrected one integrity defect before any external reviewer saw it
- Four adversarial multi-agent reviews, most recent one conducted before this application was drafted, with all critical corrections integrated
- 444 passing tests covering the RSSM transferable-subset invariants
- Seed-level JSON artifacts with full provenance (git SHA, env versions, GPU) tracked in public git for every run

None of this substitutes for external peer review. It exists because external peer review is unavailable to me at this stage, and I believe compute grants should still be investable in careful solo research with verifiable methodology.

**Transparency on LLM-assisted development.** Implementation and documentation are produced using Anthropic's Claude under sustained human supervision. The research question, hypothesis thresholds, kill criteria, and arbitration of every factual claim are my own. This is declared in the research proposal (§3 and §6), the README, and this letter because I think that kind of workflow deserves transparency at each entry point.

**What I am asking for.** A 30-day allocation of 1 TPU v3-8 on-demand with preemptible overflow, approximately 40 TPU-hours of expected usage in Month 1. The plan is:
- Validate the GPU→TPU pipeline on the Band B dataset
- Execute the two preregistered ablations (A10 adversarial pair for generality, A11 GRU-shuffled for mechanism) that are blocking any workshop claim
- Report results publicly within 48 hours of month-end

If Month-1 results pass the preregistered thresholds, I will re-apply for a ~60 TPU-hour Month 2–3 allocation to execute Post-1 horizontal scale (7 additional source-target pairs). If they fail, I will publish a public negative-result report and methodology blog post, and defer the paper to a stronger future submission.

**Three hard commitments** regardless of scientific outcome:
1. Open-source code, data, and preregistration under Apache License 2.0
2. Monthly TRC progress report with TPU-hour accounting
3. Public blog post on preregistration-grade methodology for solo-dev RL

**What I want to avoid.** The worst outcome for both TRC and me is burning TPU quota on experiments that drift from their preregistered plan. The methodology is explicitly designed to prevent that — every Month-1 experiment has a preregistered go/no-go gate, and failure modes trigger honest reporting rather than rescue attempts.

I have been using a single RTX 4080 GPU from a home workstation, which has carried the project to a preregistered pilot verdict but is the hard ceiling on any scale-up. TPU access would not accelerate the scientific thinking — the preregistration forces that work to the front — but would directly remove the months-long rerun cycles between preregistered-amendment and empirical verdict that are the current bottleneck.

I am happy to answer any questions, provide additional materials, or iterate on the proposal before a decision. The entire project is public and every claim above is independently verifiable in under 5 minutes via the repository (see `docs/compute_application/reproducibility.md`).

Thank you for considering this application.

Sincerely,
**Jérémie Mortier**
Independent researcher, France
Contact: mortier.jeremie@gmail.com
Public repository: https://gitlab.com/mortier.jeremie/ragnarok
Application package: `docs/compute_application/` in the repository
