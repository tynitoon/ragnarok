# Pre-TRC-Submit 4-Agent Review (post-Band-C-kill)

**Date:** 2026-04-18, right before TRC application submission.
**Purpose:** re-adversarial review of the compute-application package after the Band C N=10 verdict (primary hypothesis falsified, all 3 kill criteria triggered, branch C activated per prereg v3.7 → v3.8). The 1st 4-agent review (`pre_trc_4agent_review_2026-04-17.md`) happened BEFORE the kill; this one happens AFTER, to verify the pivot narrative holds under new adversarial scrutiny.

**Agents:**
1. **TRC Reviewer Simulator** — ex-Google researcher, TRC panel experience
2. **Hostile Skeptic** — reject-seeking adversary specifically focused on the post-kill pivot
3. **Form Completion Specialist** — identified the real TRC form structure
4. **Narrative Coherence Reviewer** — cold-reader first-impression test

---

## Cross-agent convergence

All 4 agents independently flagged:

1. **README + cover letter are stale** (still say "Band C in progress, ratio 1.605"). Blocking.
2. **LLM disclosure over-handled** (3-4 places). Compress to 1 place in proposal + 1 in README.
3. **Q1/Q2/Q3 are prose, not preregistered with numerical thresholds.** Critical credibility gap.
4. **§4.1 still allocates 15 TPU-h to A10/A11 ablations of falsified hypothesis** — internally inconsistent with the pivot announced in §2.
5. **The reframe "research program" was committed ~10h pre-kill** (SHA `680dfe6` at 2026-04-17 14:18 vs kill verdict 2026-04-18 00:10). Git-verifiable, thus a vulnerability to hostile review.

## Cross-agent divergence

- **TRC Reviewer:** 72-78% initial approval probability, kill is mildly positive signal
- **Hostile Skeptic:** "Conditional approve" with critical attacks on pivot timing
- **Narrative:** "Coherent but tense" — data is honest, framing is defensive
- **Form Specialist:** reveals the form has no research-description field; the ~17k-char proposal is for the follow-up email thread, not the form itself

## The Form Specialist's major discovery

TRC's 2026 application is a **3-page Google Form** (short-answer intake, not a proposal submission):
- Page 1: name / email / organization / country / job title
- Page 2: ML experience calibration (frameworks, GCP experience, TPU experience, how-heard-about)
- Page 3: legal checkboxes

There is **no long-form field** for the research proposal. The 17k-char proposal finds its home in the **follow-up email thread** after TRC responds (~3-4 days). At submit time, the only bandwidth to signal rigor is:
- Embed repo URL in `Organization Name` field
- Compress engineering credibility into `Job title / role` (Piepacker + Airbus = 2 proper nouns)
- Select truthful (low) TPU experience + (strategic) JAX framework interest

## Key critical fixes identified

1. Update README with Band C verdict (stop claiming "in progress")
2. Update cover letter with Band C verdict
3. Rebalance §4.1 Month 1: less A10/A11 (ablations of dead hypothesis), more Q1-C contrastive (actual pivot direction)
4. Delete §8 "transistor / penicillin / Higgs" analogy (overreach)
5. Compress LLM disclosure to 1 place in proposal + 1 in README
6. Add sentence in §2 that pre-empts the "pivot post-hoc" attack by explicitly citing the pre-kill v3.7 amendment SHA (`a0c1140`, 2026-04-17 10:15) containing the branch-C clause
7. Preregister Q1/Q2/Q3 thresholds properly in preregistration.md §13 v3.9 (promote them from `reviews/` prose to actual prereg)

Estimated work: ~1h total before submit-ready.

## Integration status

See commit `bd9a7fd` for the integration of the above fixes.

---

## Individual agent reports (abridged)

Each agent's full response is in the conversation thread; key excerpts here for audit trail.

### TRC Reviewer Simulator — verdict
> *"I would approve this for 30 days at v3-8 preemptible. It's a net-positive application: methodological rigor is above median, the ask is sober, the honest pivot is a rare positive signal. The hesitation isn't 'will this person be honest' (they will) — it's 'is the research program still well-defined post-pivot.'"*

### Hostile Skeptic — verdict
> *"Conditional approve because the kill-honored-on-N=10 chain is genuinely rare and the methodology is real, but approval should be contingent on (a) preregistering Q1/Q2/Q3 thresholds in preregistration.md — not in reviews/ — before any TPU-hour is spent, and (b) rewriting §1 of the proposal to stop asserting Ragnarok was 'always a research program, not a single-hypothesis study,' which the git history does not support."*

### Form Specialist — key finding
> *"The form has no long-form research-description field. Reviewers read: one-line Organization Name, one-line Job title, multiple-choice experience signals. The dossier's depth becomes relevant only after approval (via the follow-up email thread)."*

### Narrative Coherence Reviewer — key finding
> *"The per-seed breakdown in §2 — 4 positive / 5 neutral / 1 actively anti-transfer — is the single most credibility-building sentence in the package. That sentence alone is the reason this application reads as honest science instead of a post-hoc narrative. Protect it, build around it, and let the rest of the document be less busy defending itself."*

---

*This review exists in the public git history and will be cited in the paper's supplementary materials if the project produces one. Any subsequent revision of the compute-application package should reference this document by commit SHA and note which of the critical fixes were integrated.*
