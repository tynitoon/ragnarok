# Compute Application Package — Ragnarok

**Purpose:** complete materials supporting the application to Google TPU Research Cloud (TRC) and backup applications (Lambda Labs research credits, potentially CoreWeave).

**Status:** drafted 2026-04-17. Final submission scheduled within 48 h of Band C N=10 verdict (expected 2026-04-17 evening).

---

## How to read this package

| # | File | Purpose | Length |
|---|---|---|---|
| 1 | [`cover_letter_trc.md`](cover_letter_trc.md) | Short pitch, TRC-specific voice, ~1 page | Short |
| 2 | [`research_proposal.md`](research_proposal.md) | Main technical proposal (7 sections) | ~3 pages |
| 3 | [`methodology_showcase.md`](methodology_showcase.md) | Verifiable rigor artifacts, for reviewer trust | ~1 page |
| 4 | [`compute_roadmap.md`](compute_roadmap.md) | TPU-hour budget + pre-registered gates | ~2 pages |
| 5 | [`reproducibility.md`](reproducibility.md) | 5-minute verification instructions | ~1 page |

**Recommended reading order for a TRC reviewer with 10 minutes:**
1. `cover_letter_trc.md` (2 min) — what, why, who
2. `reproducibility.md` (2 min) — gut check on claimed verifiability
3. `research_proposal.md` §1 + §2 + §4 (5 min) — the science and the compute ask
4. Skim `methodology_showcase.md` if curious about the rigor signal

**Recommended reading order for a workshop reviewer:**
1. `reviews/pre_trc_4agent_review_2026-04-17.md` in the repo root (shows what was flagged and addressed before submission)
2. `research_proposal.md`
3. Supporting artifacts in the repo itself (preregistration.md, reviews/ directory)

---

## Integrity notes

All documents in this folder are committed to public git before or simultaneous with any TRC application submission. Any revisions after submission are timestamped and documented in the git log. This is the same standard applied to `preregistration.md` itself — no post-submission rewriting of claims or numbers.

The application leverages Band B rescue results (ratio 1.605, p=0.259, LOO min 1.435, mechanism passed) plus a pre-registered Band C N=10 extension currently running. If Band C passes, the proposal's empirical section will be updated with N=10 numbers in a separate commit prior to submission. If Band C fails, the proposal will be submitted with the honest pivot narrative toward Post-1 horizontal scale + methodology contribution.

## Multi-agent review sign-off

This package was reviewed before finalization by four adversarial LLM agents (RL-methodology, grant-application, devil's-advocate, strategy). Six critical corrections were identified and integrated:

1. Narrowed novelty claim with explicit citations to adjacent work (Gato, RT-X, SPiRL, Options-Critic, Progressive Networks)
2. Removed "5-day-old intensive project" framing from §6 — replaced with fact-based enumeration
3. Added GPU-hr → TPU-hr conversion rationale in §4
4. Cut deliverables from 5 to 3 hard + 1 conditional + stretch (stretch items explicitly not committed)
5. Added "embodied agent skill reuse" framing to §1 for a clearer "so what"
6. Reduced LLM-assisted disclosure repetition (consolidated declaration)

See `reviews/pre_trc_4agent_review_2026-04-17.md` for the full review record.

## What is *not* in this package (intentionally)

- **A workshop paper draft.** Workshop submission is *conditional* on Band C passing AND A10 + A11 ablations supporting the claim. The paper is not the deliverable being pitched here — the TRC compute is what enables the paper (or justifies its absence honestly).
- **Overclaimed deliverables.** The `deliverables committed` list in `research_proposal.md` §5 was cut by ~40% after adversarial review flagged overcommitment risk.
- **Blog post on methodology.** Written in Week 1 post-verdict, committed to the repo, linkable in the month-1 TRC report.
