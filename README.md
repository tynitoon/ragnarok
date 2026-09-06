# Ragnarok

**A solo-dev research project on developmental agents: can an agent learn skills and reuse them to learn
new things faster?** Run with preregistration-grade discipline — every hypothesis, threshold and
analysis rule is frozen in `preregistration.md` and evaluated by a scorer committed *before* the data
exists. Negative results are published with their numbers; overclaims are retracted in the record
rather than quietly dropped.

> **Honest headline, updated 2026-07-31.** Earlier versions of this README presented *compounding*
> — reuse making deeper skills no more expensive to learn — as a validated result. Those measurements
> were real, but a later audit found the regime was too easy: the agent was handed the recipe DAG for
> free in its observation every step, so nothing was expensive to re-derive and no memory could pay.
> Re-tested honestly under frozen rules, the strong claim returned **three NULLs (v53, v55, v57)**.
> What is genuinely established is narrower, and is listed below. The retraction trail is in
> `RESEARCH_DIRECTION.md`.

---

## What is established (survived adversarial audit)

| result | evidence |
|---|---|
| A childhood navigation skill **transfers zero-shot to unseen procedural worlds** | ~0.95 success (v50) |
| **Hindsight self-imitation makes composition learnable** where sparse-reward RL sat at 0.003 | v53-v55 |
| An agent **discovers hidden recipes by failing** — no oracle, rules never shown | mastery 1.00 in 2 rounds (v55 gate) |
| Removing the recipe oracle **made the agent goal-directed**, where before it ignored the goal entirely | goal-swap 0.30 / 0.42 / 0.75 vs ~0, on 3/3 worlds (v55) |
| Accumulated memory makes new goals **far cheaper** — cheaper, not *possible* | v55/v57, with the mechanism measured |
| A **persistent per-world evidence store** lets a policy holding no item identities solve deep goals | ARC 2 gate: 4/4 mastered; zeroing the store drops the same weights 0.98 → 0.00 |
| **Frozen weights transfer to a world never seen**: a policy trained in one world, dropped into another with an empty store, masters goals a random-weight policy cannot | ARC 2 probe: 3/6 vs 0/6 (v59) |
| A family of worlds sharing a **hidden chemistry** (ARC 3, v61) makes shared knowledge *expensive to reacquire* — the property both earlier arcs lacked | law-knower 1.00 on 12/12 goals; without the law, tier 4 is out of reach in 4 rounds for 11/12 fresh learners; the law is worth 0.65–0.72 of the mastery area on tier 3 and ~1.0 on tier 4 |
| A CPU model of the store sweeper **predicts the GPU reference's cost to within a few percent** on every gate goal | ARC 3 gate, CONSISTENT check |

## What is refuted or null (published, not hidden)

- **v53 NULL · v54 PARTIAL · v57 NULL** — the "is accumulated memory *necessary* at depth?" arc, closed
  for good under its own pre-committed kill criteria.
- **Four instrument defects**, each of which had silently invalidated earlier measurements: a recipe
  oracle leaking the answer into the observation; a cost metric that saturated at zero once the agent
  got good, making compounding unobservable by construction; an evaluation that trapped the policy in an
  absorbing state; and a credit leak under which **99.69 % of gradient steps at depth concerned
  something the agent was never asked to do** — meaning no "from-scratch" control in the project's
  history was ever actually knowledge-free.

- **ARC 2 — partial negative (2026-08).** Transferred weights help an agent *start* in a new world, but
  once both an experienced and a fresh agent are allowed to learn, the fresh one catches up in 1–3
  rounds: Δ = +0.069 against a run-to-run noise of 0.076. Diagnosis: what those worlds shared was cheap
  to reacquire. `ARC2_PLAN.md` section 15.
- **ARC 3 — closed at its gate (2026-09-06).** The substrate fixed ARC 2's wall (see above), but the
  pre-registered feasibility gate failed on its fresh-learnability clause: fresh learners are *bimodal*
  on the position where "learns faster" would be read — half reach it in 2–4 rounds, half stall — so the
  frozen median criterion was missed by one round, twice by two envs of 64. No experienced arm was run;
  this is a feasibility result, not a verdict. `ARC3_PLAN.md` sections 10–11.

That map of *why this question is so hard to measure* is the most transferable thing here.

## What is open

**ARC 3 — worlds that share a hidden law.** Every world of the family obeys the same secret bonding table
between 14 elements; a world exposes ~38 % of it; an agent that has lived in four worlds knows ~85 %. The
substrate, the store (per-goal working memory), the identity-free policy with an outcome head, the
hand-coded ceilings, the gate, a frozen monotone scorer and every runner are built, tested and
pre-registered (`preregistration.md`, v61). The gate says the law is worth more than any effect this
project has had in front of it, and that the fresh baseline does not reliably learn the comparison
position at a four-round budget. The next step — a re-gated v62 with a feasibility criterion set from the
twelve measured curves, on new worlds — is the owner's call.

---

## Repo map

Most of this tree is **historical record**, kept because the negative results are part of the evidence.

```
ARC3_PLAN.md            ← the live plan (ARC 3): substrate, unit, gate, verdict rule, audit ledger, gate results
ARC2_PLAN.md            ← ARC 2: design, the two voided designs, the pilot that closed it (section 15)
preregistration.md      ← every frozen hypothesis, append-only, in chronological order
RESEARCH_DIRECTION.md   ← the running lab notebook: what was tried, what it returned, what killed it
README.md               ← you are here

ragnarok/environments/law_world.py   ← ARC 3 substrate: hidden laws, per-world skin, the env
scripts/
  *_v61.py              ← ARC 3: pair_store, pair_net, hand (L/G'), sweeper, mc, roster, gate, score,
                           pretrain, probe, confirm, figure, test_law
  *_v58..v60.py         ← ARC 2, closed: evidence_store, evidence_net, probes, pilot
  *_v49..v57.py         ← ARC 1, closed: the necessity arc that returned three NULLs
  *_v10..v48.py         ← historical: pixels, world models, notion libraries, arcade games
  play_*.py, demo.py    ← watchable demos
scripts/INDEX.md        ← what every script is, and where the archived ones went
ragnarok/               ← the reusable library: environments, learning, infrastructure
craft_v6_out/           ← results of record. JSONs are tracked; *.pt checkpoints are gitignored
                          (1.7 GB on disk, none of it in the repo)
```

**Yes, there is Tetris in here** — and Pong, Snake, Flappy, Breakout. That is the v15-v44 era, when the
question was whether skills transfer *across games* from raw pixels. Those files are kept because their
results are cited in the current reasoning, not because they are live.

## Watch something run

```bash
python -m scripts.play_v55 --seed 0 --goal 1
```

Side by side on one hidden-recipe world: an agent that has lived there, versus an amnesic control
trained from scratch on that exact goal for the full budget. The memory agent walks a five-step
prerequisite chain with zero wasted attempts; the amnesic one attacks the goal item 26 times and never
gets it. `--list` prints every goal and what each arm actually achieved in the measured run.

## Method

- Hypotheses, thresholds, strata and the decision rule are **frozen before the run**, and the scorer is
  committed before the confirmatory arms exist. It refuses to score incomplete or pre-repair data.
- Thresholds are **fitted from measured noise**, never chosen: two independent control runs define what
  "no effect" looks like on the exact statistic used, and the bar is placed beyond that null's tail.
  Three verdicts in this project were lost to thresholds picked from a pencil; that is the repair.
- Every milestone is audited by adversarial agents whose job is to kill the result. Several did — one
  caught a censoring asymmetry that would have published a fourth NULL on an extraordinary result.
- **Kill criteria are written before the run** and end a line of inquiry rather than spawning another
  iteration of it.

**Repositories** — primary: [gitlab.com/mortier.jeremie/ragnarok](https://gitlab.com/mortier.jeremie/ragnarok)
· mirror: [github.com/tynitoon/ragnarok](https://github.com/tynitoon/ragnarok)
