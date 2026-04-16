# Post-Pilot Action Backlog

Persistent registry of **deferred-but-committed actions** surfaced by agent
reviews during pilot #2 (Phase 3). These items are NOT urgent enough to block
the pilot, NOT load-bearing for §8 pass criteria, and NOT in scope for the
review-driven mid-pilot hardenings landing alongside. But they ARE committed
work — every item here is something the project has implicitly promised a
reviewer, a preregistered audit trail entry, or an agent review, and losing
track of them would be a slow accumulation of integrity debt.

**This file is the single source of truth for post-pilot TODOs.** If an item
is not here, it is not preregistered work; if it IS here, it must be either
(a) completed before Phase 5 headline run starts, (b) re-triaged by the next
multi-agent review, or (c) explicitly retired with written rationale
referenced from this file.

**Invariants:**
- Every entry has: `status` (open / in-progress / done / retired) +
  `source-review` date + reviewer type + `blocking-phase` (which phase this
  item gates if any) + one-paragraph rationale.
- New items appended at the bottom with timestamps. No deletion; retirement
  annotated in-place.
- Audit commit referenced from `preregistration.md` §13 at the amendment
  that created the item, so git history can verify "was X promised before
  result Y?".

---

## Source review: 4-agent mid-pilot review, 2026-04-15

Reviewers: RL-methodology, code-review, strategy, architecture. Pilot #2 was
at 8/30 runs complete. Review surfaced 6 urgent (now executing), 5 post-pilot
(here), and 2 non-blocking (dropped). The 6 urgent items are in prereg §13
v3.5 amendment; the 5 post-pilot items follow.

### POST-001 — Typed pilot schema (pilot_schema.py)

- **Status:** open
- **Source:** code-review agent (2026-04-15), "pilot_run.py is a 1186-line
  god-module; results schema is dict-shaped and brittle"
- **Blocking-phase:** Phase 5 (headline run) — data contract must be stable
  before N=20 seeds land
- **Rationale:** `PilotRun` / `PilotResults` are currently shaped as
  dictionaries passed through atomic-write JSON. Schema drift between
  `pilot_run.py` producers and `pilot_analysis.py` / `smoke_verdict.py`
  consumers is a Bug-F-in-waiting class. Introduce `scripts/pilot_schema.py`
  with Pydantic or `@dataclass` types, serialize via `asdict()`, validate on
  load. Migration path: additive fields with defaults keep backward-compat
  with existing `pilot_results.json`.
- **Effort:** 1–2 days. Touches `pilot_run.py` writers, `pilot_analysis.py`
  readers, `smoke_verdict.py` readers, and a fresh test module. Existing
  `pilot_results.json` / `.broken_trunk` artifacts must remain parseable
  (backward-compat layer).

### POST-002 — In-process smoke integration test

- **Status:** open
- **Source:** code-review + testing agents (2026-04-15), "smoke_bug_e_v5 was
  run manually; no test pins the contract that pilot_run --smoke produces
  the expected telemetry shape"
- **Blocking-phase:** Phase 5 (before next major `pilot_run.py` refactor)
- **Rationale:** The smoke pre-check is a load-bearing safety rail (aborts
  pilot if drift > 50% by ep 100). Its contract today is "reads like it
  works on the log Jeremie sees"; there's no regression test that simulates
  a run end-to-end and asserts the telemetry JSON has the right shape.
  A single `tests/test_pilot_run_smoke_integration.py` running a 2-seed 500-step
  smoke in-process (fake env, reduced budget) pins the producer-consumer
  contract and would have caught Bug E v3's closure-extraction miss.
- **Effort:** 1 day. Needs a tiny fake env with shapes matching the MCC
  transfer pair so `try_transfer` path actually exercises. ~2-min wall time
  per CI run, acceptable.

### POST-003 — Extract SmokeProfile constant

- **Status:** open
- **Source:** code-review agent (2026-04-15), "smoke mode parameters
  (seeds=2, max_steps=40_000, source_cap=100_000) are scattered magic
  numbers in `--smoke` argparse branch"
- **Blocking-phase:** Phase 5 (before next smoke round)
- **Rationale:** Bug E v4 amendment pinned the 2-seed / 40k-step smoke
  contract in prose (`preregistration.md` §13 v4 "Smoke flag now matches
  the prereg-committed 2-seed protocol"). The code enforces it via literal
  numbers in an argparse branch, which is how prereg-vs-code drift creeps
  back in. Extract to `SmokeProfile = {seeds: 2, max_steps: 40_000,
  source_cap: 100_000}` at module top of `pilot_run.py`, reference from
  argparse branch, assert in a unit test that matches prereg text.
- **Effort:** 30 min. Low-risk refactor; primarily a drift-prevention
  artifact.

### POST-004 — Appendix table for prereg amendments

- **Status:** open
- **Source:** strategy agent (2026-04-15), "reviewer who opens the
  preregistration lands on a 900-line file with 11 timestamped amendments
  in one §13 dump; impossible to extract decision history at a glance"
- **Blocking-phase:** Phase 7 (paper draft)
- **Rationale:** The amendment trail is a credibility asset — it's how we
  prove "decisions were made pre-data, not after seeing Y". But a 900-line
  linear log is unreadable. A compact table at the top of §13 summarizes
  each amendment in one row (date, version, affected section, one-line
  change, commit SHA), with the full prose below. Reviewer can triage the
  decision history in 30 seconds, cite the commit SHA to verify git-time
  ordering vs data-collection events.
- **Effort:** 2–3 hours. Mechanical extraction; opportunity to also
  canonicalize commit SHAs that are currently inline-prose.

### POST-005 — A7 / A9 / A10 / A11 ablations on headline figure

- **Status:** open
- **Source:** RL-methodology + devil's-advocate agents (2026-04-15),
  "mechanism-depth ablations are listed in §7 but not committed to a
  figure; a reviewer can't see at a glance whether the mechanism claim
  has been isolated"
- **Blocking-phase:** Phase 5b (ablations panel) → Phase 7 (paper)
- **Rationale:** The headline figure (Fig. 1 in the paper draft) will show
  per-seed RMST and learning curves for the primary pair. Adding a
  companion panel with A7 (trust-region scan), A9 (cross-trajectory
  shuffle), A10 (adversarial-negative pair Pendulum→Reacher), A11
  (GRU-shuffled weights) **side-by-side** on the primary metric converts
  the paper from "we transferred a skill" to "we transferred a skill, and
  here are four interventions that kill the effect in predictable ways".
  §12.5 already commits A1+A2+A8+A9 as a non-negotiable panel if cut #2
  fires; the same panel should appear in the 3-env headline paper too.
- **Effort:** Compute is in §5b budget; presentation is 1 day of figure
  work in Phase 7.

### POST-006 — Reproducibility artifacts bundle

- **Status:** open
- **Source:** strategy agent (2026-04-15), "paper claim is infalsifiable
  from outside without seed-level artifacts and a recipe"
- **Blocking-phase:** Phase 7 (OpenReview submission)
- **Rationale:** OpenReview reviewers now routinely ask for: (a) seed-level
  result JSONs (we have these in `pilot_results.json` already), (b) the
  exact git SHA per run (we have `git_sha` in the payload), (c) a
  reproducibility script that reads the JSON and regenerates the headline
  table + figure from scratch (we don't have this yet), (d) a
  `reproduce.sh` that takes a seed list and runs a subset on the reviewer's
  own GPU. (c) is load-bearing — without it the claim is a `.json` blob
  and a figure with no verifiable path between them.
- **Effort:** 2–3 days. Primarily `scripts/reproduce_headline.py` + a
  smoke-scale `reproduce.sh` that runs 1 seed end-to-end in under an hour.

---

## Non-blocking items (dropped, retained for provenance)

Two review items were triaged as "not worth preregistering":

- **NB-001:** Rename `pilot_run.py` to `pilot_phase3.py` for clarity
  (strategy agent). Decision: cosmetic; post-Phase-7 if ever. Does not
  appear in POST-### numbering.
- **NB-002:** Switch `json.dumps(indent=2)` to `msgpack` for
  `pilot_results.json` (code-review agent). Decision: human-readability
  of the JSON artifact is load-bearing for the review process and
  paper appendix; binary formats hurt more than they help here.

---

## Append log

Append new items below with timestamp and source-review reference. Do not
delete; retire in-place with "status: retired" + rationale reference.

---

### POST-007 — Multi-skill composition for new-task learning

- **Status:** open (design-space item; no implementation chosen yet)
- **Source:** user question during pilot #2 execution (2026-04-16). Jeremie
  asked "est-ce possible pour notre IA d'utiliser plusieurs skills cristallisés
  pour en apprendre un nouveau ?" after seeing paire 3 partial results.
  Explicitly deferred: "on en reparlera quand on y sera et on fera challengé
  les différentes implémentations à des agents avant de sélectionner la piste
  qu'on explorera."
- **Blocking-phase:** Post-1 (scale horizontal, 5–10 new tasks per
  `project_research_plan`) — NOT workshop paper. With only 3 skills in the
  library, multi-skill composition has no empirical traction; the probability
  of 2+ relevant skills existing for a new task is too low. Becomes interesting
  at ~10 skills and load-bearing at ~20+.
- **Rationale:** Current `try_transfer()` (`agent.py:742`) loads exactly one
  skill per new task: either exact env_name match, or latent-nearest neighbor
  via `skill_selector.select()`. `MultiSkillAgent` (`skills/multi_agent.py`)
  exists but is an **execution-time** router (dynamic skill switching during
  an episode), not a **learning-time** composer. The workshop paper's causal
  claim is cleanest with one-skill transfer, so single-skill init is the right
  choice for the headline experiment. Post-workshop, the library will grow
  and the question "can N skills accelerate learning more than the best 1
  skill?" becomes a legitimate research direction worth multi-agent debate
  before commitment.
- **Design space (to be challenged by agents at decision time):**
  1. **Ensemble init** — weighted average of top-k skills' transferable
     state_dicts (weights ∝ 1/latent_distance). Low dev cost (~1–2 days),
     risk of mode-cancellation when averaging weights encoding different
     behaviors. Works because transferable subset shape is identical across
     skills.
  2. **Mixture of Experts RSSM** — load k RSSM cores in parallel, learn a
     gating net that mixes their (h, z) outputs per step. Each skill keeps
     integrity, no cancellation. Cost: k× memory, k× forward pass, gate
     training. 2–3 weeks dev.
  3. **Progressive Networks** (Rusu et al. 2016) — freeze all prior skills,
     add new network with lateral connections to each frozen skill. Zero
     catastrophic forgetting, but parameter count grows linearly with skill
     library size. ~1 month dev; doesn't scale past ~10 skills.
  4. **Seed-buffer hybrid** — keep mono-skill weight init (current), but
     populate replay buffer with trajectories from top-k skills before
     training starts. Init unchanged; exploration enriched. Compatible with
     SAC off-policy. 3–5 days dev.
  5. **Multi-teacher distillation** — top-k skills act as teachers; student
     imitates confidence-weighted consensus, then fine-tunes on new task.
     ~2 weeks dev.
- **Decision protocol when we arrive at Post-1:** spawn 3–4 reviewer agents
  (RL-methodology, architecture, strategy, devil's-advocate) with this
  design-space list + current library state (expected ~10 skills post-Post-1
  start). Collect dissent on which option(s) to test first. Prefer cheap
  empirical answers (options 1 + 4) before committing to expensive
  infrastructure (options 2 + 3 + 5). Re-triage here with decision record.
- **Effort:** TBD at decision time. Each option's dev estimate above;
  compute cost depends on how many options we race empirically vs. pick
  upfront.
