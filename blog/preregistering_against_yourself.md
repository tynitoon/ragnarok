# Preregistering Against Yourself

*How a six-day-old solo RL research project's preregistered kill criteria actually fired, and what that bought me.*

**2026-04-18** — Jérémie Mortier

---

## The punch line

Six days into my reinforcement-learning research project, the primary hypothesis is dead.

I preregistered specific numerical kill criteria before collecting the data. This week, the N=10 analysis came in — RMST ratio 1.036, log-rank p=0.510, leave-one-out minimum 0.871. All three kill criteria triggered simultaneously. Rather than defend the result or quietly move the threshold, the preregistered decision tree takes over: pivot to the broader research program, publish the null, write this post.

This is a post-mortem written while the data is still fresh, for other solo researchers wondering whether preregistration actually works when it's your own project on the line.

Short answer: yes — and it hurts in exactly the way it's supposed to.

---

## What Ragnarok tests

[Ragnarok](https://gitlab.com/mortier.jeremie/ragnarok) is a modular RL agent with skill crystallization and cross-action-space transfer. The short version of the central hypothesis: take a Dreamer-style RSSM world model trained on a source task (e.g. CartPole, discrete actions), extract a **transferable subset** — GRU core + prior + posterior distributions, which operate on a fixed-size latent `(h, z)` independent of the task's observation and action dimensions — and load it via shape-checked `load_state_dict` into the world model of a new task with a *different action space type* (e.g. MountainCarContinuous). Then switch the policy to operate on latent features rather than raw observations.

The question: **does this cross-action-type latent-trunk transfer actually carry dynamics knowledge, measurably reducing episodes-to-mastery on the new task?**

This isn't a random question. Existing work mostly studies transfer within fixed action-space types. Gato (Reed 2022) and RT-X (2024) handle mixed action spaces but via single-model tokenization, not subset transfer. Options-Critic, Progressive Networks, SPiRL — all fix the action space. The specific mechanism I tested was unoccupied in the published record.

The **primary pair** I preregistered for this test: CartPole-v1 (discrete, obs 4D) → MountainCarContinuous-v0 (continuous, obs 2D). Both are pendular-class systems with similar dynamics. In retrospect, this is the most favorable cross-action-type pair imaginable — which matters for how we read the kill.

---

## The pilot journey

**Pilot #2 (N=5 seeds, 3 pairs).** Primary pair landed at RMST ratio 1.238. Looks positive. But leave-one-out was brutal: dropping seed 46 collapsed the ratio to 1.049. The entire signal was one lucky seed. Fragile.

**Band B rescue (N=5 fresh seeds on primary, corrected warmup).** Ratio 1.605, LOO minimum 1.435. **Not outlier-driven** this time — 3 of 5 seeds positive, 2 neutral, no dominant single seed. Looks robust. But log-rank p = 0.259 — statistically underpowered. A reviewer could not accept this as a positive result.

At this point I faced the classic underpowered-positive trap: ratio looks great, significance doesn't hold. The instinct is to "extend to N=10 and let the variance drop." Which is exactly what optional-stopping looks like if you're not careful.

I handled it by writing a v3.7 amendment to the preregistration document, **before launching seeds 52–56**, specifying numerical kill criteria for the N=10 extension:

```
Pass:   ratio ≥ 1.30 AND log-rank p < 0.10 AND LOO min ≥ 1.15
Kill:   ratio < 1.20 OR  log-rank p ≥ 0.20 OR LOO min < 1.00
```

The kill thresholds sit just below the pass thresholds. The idea is deliberate: if the N=10 result lands in the ambiguous middle, we flag it as underpowered-intermediate; if it lands clearly below, we accept that the underlying effect isn't there. The thresholds were committed at SHA `a0c1140`, timestamped, before the new seeds ran.

**Band C N=10 (pooled seeds 47–56).**

- RMST ratio: **1.036**
- Log-rank p (one-sided): **0.510** asymptotic, **0.516** permutation N=10,000
- LOO minimum ratio: **0.871**
- Per-seed ratios: 4 positive (seeds 48, 49, 51, 54 at ratios 1.50, 3.29, 1.57, 1.91), 5 neutral (ratios between 0.97 and 1.02), and 1 actively anti-transfer (seed 55 at ratio 0.33 — the transfer arm was three times slower than scratch).

All three preregistered kill criteria triggered simultaneously. The null hypothesis cannot be rejected. The specific mechanism — shape-checked RSSM-subset loading across the discrete↔continuous action-space-type boundary — does not produce reliable transfer at N=10 on the most favorable pair imaginable.

---

## The kill moment

There is a specific feeling to watching a kill criterion fire on your own work. It is not shock — the trendline was visible three seeds into Band C. It is not defeat — the methodology is working exactly as designed. The closest word is *recalibration*: a quick collapse of what you thought was probably true into what the data says is probably not.

What matters is what you do in the next fifteen minutes. The tempting moves are:

- "Let me check if it's just seed 55 — maybe with N=20 it evens out." (Extension after the kill is exactly the garden-of-forking-paths the preregistration was designed to prevent.)
- "Maybe the kill thresholds were too strict — 1.20 is arbitrary." (Threshold revision after the data is observed is p-hacking with extra steps.)
- "The mechanism check still passed. That's something." (Mechanism check verifies plumbing, not causality.)
- "I could publish Band B at N=5 separately." (Publishing the underpowered positive while hiding the well-powered null is fraud.)

I took none of these. Instead I ran the analyzer, computed the LOO table, wrote the v3.8 amendment that records the kill, updated the research proposal to reflect the pivot, and committed within twenty minutes. The preregistered decision tree activates branch C: workshop-paper-on-primary-pair abandoned, research program pivots to the three open questions the project had already identified as the broader research agenda.

The commit history shows this happening in real time. A reviewer can verify it. That verifiability is the whole point.

---

## What preregistration actually bought me

Consider the counterfactual.

Without preregistration, at Band B's N=5 ratio 1.605, I could have written a workshop paper titled something like *"Cross-Action-Type Skill Transfer via Shared RSSM Latent Trunks"* with that number as the headline result. The paper would have been submittable. It might have been accepted at a workshop — reviewers frequently approve N=5 RL results if the ratio looks strong enough and the mechanism is plausible.

Three to six months later, someone else would have tried to replicate. Either they would have reproduced Band B's 1.605 (unlikely given the N=10 regression to 1.036), or they would have reported a null and my published result would have been the one looking suspicious.

The retrospective verdict from the community would have been one of:

1. *"Interesting but irreproducible. Probably seed lottery."* (Henderson et al. 2018 was cited on this exact failure mode. I would have been the case study.)
2. *"The author extended to N=10 and didn't report it because it killed the result."* (Worse.)
3. *"The author couldn't afford more seeds. Reasonable."* (Most charitable, but still damaging to the claim.)

None of those are the outcomes I want attached to my name for the next decade.

Instead, what actually exists now is a preregistered study that:

- Committed its primary threshold before any data was collected (§8, SHA `3cf847d` territory)
- Amended its extension thresholds before the extension's data was collected (v3.7, SHA `a0c1140`)
- Honored the kill when it fired (v3.8, SHA `6760f52`)
- Published the null data publicly and immediately
- Did a self-initiated chronology audit of a separate amendment whose phrasing was imprecise, corrected it before any external reviewer saw it (v3.5 → v3.6)

That dossier, with or without a positive scientific result, is a methodology artifact I can put in front of a grant reviewer, a collaborator, or a future paper's Reviewer 2 and defend without flinching. That is what preregistration bought me.

---

## What this does NOT mean

I want to be careful about claims here:

- **The broader research question is not dead.** "Can skills be transferred across action-space types in RL agents?" remains open. My specific mechanism (naive subset loading from a reconstruction-based RSSM) didn't work. Other mechanisms — contrastive RSSM, EWC-protected loading, kickstarting distillation, multi-skill composition — remain to be tested. The null on the most favorable pair is evidence that *easy* approaches won't work, not that the question is unanswerable.

- **The methodology is not a contribution on its own.** Preregistration + chronology audit + multi-agent adversarial reviews is a workflow, not a discovery. It's useful and I'd like to see it adopted more widely in solo-dev RL research, but the scientific value of this project still hinges on whether the Q1/Q2/Q3 research program that takes over from here produces something.

- **The specific negative result is narrow.** One task pair, N=10. I am not claiming anything about transfer in general. I am claiming that *this mechanism* on *this pair* at *this sample size* did not detect an effect. Generalizing requires more pairs — which the next phase is designed to provide.

---

## Five lessons for solo RL dev

1. **Write your kill criteria before you have the data, in the same file as your pass criteria.** Not in your head, not in a Slack DM — in git, with a commit message, before the run executes. The pass and kill thresholds should leave no ambiguous middle where you can rescue the result. Ambiguous middles get rescued. I know this about myself and I designed around it.

2. **LOO minimum is the single most informative line in any small-N RL table.** A positive mean with a bad LOO minimum is a seed lottery. A positive mean with a stable LOO minimum is a real candidate. The Band B result "looked" better than pilot #2 because its LOO minimum was well above the floor; the Band C result "looked" worse because its LOO minimum collapsed below 1.0. You can read a lot from that one number.

3. **Your mechanism check should not be your plumbing check.** "Was `acting_policy_mode` set to `latent`?" verifies that the code ran correctly. It does not verify that the transferred weights carry task-useful information. The real mechanism test is a controlled ablation: run with shuffled weights, with frozen weights, with fresh weights, and compare. I had this preregistered as A11 (GRU-shuffled ablation) and hadn't run it before the kill fired. If Band C had passed, the paper would have been weaker because of this. Run the mechanism ablations *before* you write the paper, not after.

4. **Adversarial review before submission beats adversarial review after.** I ran four multi-agent LLM reviews at different project gates — pre-pilot, mid-pilot, post-pilot-verdict, pre-compute-grant-submission. Each one flagged issues I had missed. None of the reviews was a substitute for external human peer review, but they caught the defects that would have made external review painful. The last one, run the evening the Band C kill fired, is the reason the pivot narrative in the current grant application is calibrated instead of defensive.

5. **Publish the null fast and publicly.** Two days after the kill is faster than any institutional paper track. The repository is public. The seed-level data is in git. Anyone with a Python interpreter can verify the RMST ratio in thirty seconds. That transparency does more for credibility than another month of polish would have.

---

## What comes next

The research program pivots to three questions the project had already identified as more fundamental than the specific subset-transfer mechanism:

- **Q1 — Physics-grounded world model learning.** Replace the reconstruction loss with contrastive latent prediction + disagreement-weighted ensemble, on the existing `EnsembleRSSMCore` infrastructure. The hypothesis: reconstruction-based RSSMs allocate capacity to pixel fidelity, not causal dynamics, which is why the transferable subset didn't carry enough physics.

- **Q2 — Contextual skill selection.** The current nearest-centroid selector is static and mono-skill. A PEARL-style context encoder that infers task embedding from early trajectory would be a better baseline for the follow-up.

- **Q3 — Transfer acceleration beyond subset loading.** Kickstarting (decaying-coefficient distillation), EWC-protected subset, imagination-priming through a transferred world model — each addresses the now-empirically-visible fact that raw subset loading is insufficient.

Each sprint has pre-registered pass/kill criteria already committed (v3.9). A meta-kill criterion activates if all three paths null: in that case, the thesis that "skills can be transferred across tasks via shared neural modules in the Dreamer-RSSM family" is considered empirically unsupported as a research direction, and I pivot or wind down rather than persist.

A [TPU Research Cloud](https://sites.research.google/trc/) application was submitted on 2026-04-18 to support the exploration. If granted, it accelerates the sprints; if refused, I run the same plan on a single RTX 4080 more slowly.

---

## Invitation

If you are a reinforcement-learning researcher and you see a hole in this analysis — a paper I should have cited, a confound in my design, a category of mechanism I'm ignoring — please open an issue on the [repository](https://gitlab.com/mortier.jeremie/ragnarok) or reach me at `mortier.jeremie@gmail.com`. The methodology I'm defending here only works if adversarial review is real, and right now the adversaries are all LLMs. Human critique is not a substitute for preregistration, but it is a substitute for the peer review I don't have access to as a solo researcher.

If you are a solo-dev ML researcher considering preregistration for your own work: the infrastructure to do this is lighter than you think. The commit discipline is the hard part, not the tooling. I'm happy to share what worked and what didn't for setting it up.

And if you are a reviewer at Google TRC or a similar compute-grant program: the entire trace of what I just described is [public and commit-SHA-anchored](https://gitlab.com/mortier.jeremie/ragnarok). Every claim in this post can be verified in the repository in under five minutes. The methodology is the artifact. The science continues.

---

*Ragnarok is open-source under Apache License 2.0. Repository: [gitlab.com/mortier.jeremie/ragnarok](https://gitlab.com/mortier.jeremie/ragnarok) (GitHub mirror: [github.com/tynitoon/ragnarok](https://github.com/tynitoon/ragnarok)). Jérémie Mortier is an independent RL researcher based in France; background in software engineering at Stormshield (Airbus Defence and Space subsidiary) and indie/studio game development.*
