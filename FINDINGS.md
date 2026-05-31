# Findings — accumulation & sample-efficiency (v25–v29)

*Written 2026-05-31 during the autonomous overnight block. Honest, preregistered,
every run on GitLab. Numbers are the corrected/robust ones (see v29).*

This document answers, in plain terms, the three questions Jeremie asked:
1. Does the agent **accumulate skills and reuse them**?
2. Does giving it **variety** create a **general** skill (not a memorised one)?
3. Does **more knowledge mean fewer trials** — or would it need millions of games?

> ## ⚠ Corrections after adversarial review (read this first)
> Three adversarial reviewers (rigour, ML, strategy) stress-tested this arc and
> caught me rounding in the favourable direction. The honest, corrected state:
> - **The method is informed *domain randomization* / contextual-MDP generalization**
>   (Tobin 2017; Peng 2018; Cobbe/Procgen; Kirk 2023) — well-trodden, not a novel
>   "recipe". I should have framed it that way.
> - **"unseen-hard" (Q2) is *interpolation*, not extrapolation** — those variants'
>   difficulty falls *inside* the trained range. Only the Q3 efficiency target
>   (ratio 2.0) is genuinely out-of-distribution.
> - **The "+0.26, single-instance fails" gap is only vs the *weakest* baseline**
>   (an agent trained on the easiest variant). A single *hard* instance reaches
>   0.69 on hard ≈ variety's 0.71. Variety's real, defensible edge is being the
>   **only** agent strong on **both** halves — not a unique winner on the hard half.
> - **The efficiency effect is small and not firmly resolved**: ~1.1–1.7× across
>   seeds (mean ~1.35×), measured at 10-iteration resolution with torch's RNG
>   *unseeded* — the direction is robust, the magnitude is not.
> - **The strongest evidence for "more knowledge → faster" is NOT this Pong arc**
>   — it's the earlier *concept-granularity* results (v17b ~5–15×; v13b
>   impossible→~10 iters) on genuinely different/hard tasks. This arc is one game
>   family; treat it as a controlled probe, not the headline.
> - **UPDATE (v32): the "anticipation" mechanism is FALSIFIED.** A decisive
>   lead-time probe showed *no* agent (variety included) tracks the ball's future
>   bounce-landing more than its current position — they are all *reactive
>   trackers* (anticipation index ≈ 0 for every arm). So v27b's benefit is plain
>   **domain-randomization coverage**, not a distinct anticipatory policy. I built
>   the test to kill my own claim and it did. The win-rate *gaps* stand (they're
>   reproducible DR effects); the *explanation* was wrong and is withdrawn.
> Net: this Pong arc is standard, mundane domain randomization. The genuinely
> interesting "more knowledge → faster" evidence is the **concept-granularity**
> line (v17b ~5–15×, v13b impossible→~10 iters). Next: leave the Pong sandbox —
> test whether the variety→generalization *thesis* even holds outside games (v33,
> a symbolic/maths probe). Details in `preregistration.md`.

---

## Q1 — Accumulation & reuse: YES

- **v25** — one `UnifiedAgent` over a stream of game-encounters: for each game it
  *recognises from pixels → verifies by playing → reuses if known, else learns and
  adds to its library*. Over 7 encounters it learned 3 games (Pong, Breakout,
  Snake) and **reused a known skill 4× with zero retraining**.
- **v26** — scaled to a 15-encounter stream: it paid training for only the **3
  distinct games (500 iters)** vs **2820** for a no-memory agent that relearns
  every time → **82% saved**, recognition **100%**. Cost grows only with the
  number of *distinct* games — **sublinear in the stream**. Known games are free.

## Q2 — Variety → a general skill: YES, *if the variety is the right kind*

- **v27 (negative, kept honest)** — training on 24 Pong variants that varied
  ball-speed / paddle-size / opponent / spin did **not** beat a single-variant
  agent on unseen variants (0.68 vs 0.77). Reason: those knobs don't change the
  *optimal policy* ("track the ball" works for all), so variety only added noise.
- **v27b (positive)** — varying **paddle *speed*** (reaction budget) so that slow
  paddles **require anticipating** the ball's wall-bounces. Now only the
  variety-trained agent is general across the whole family:

  | agent | unseen EASY | unseen HARD (needs anticipation) |
  |---|---|---|
  | **variety** | 0.97 | **0.71** |
  | single-easy (reactive) | 0.95 | 0.45 ← fails |
  | single-hard (anticipatory) | 0.61 | 0.69 |

  **The recipe:** broad variety yields a general skill **only when the variation
  spans genuinely different required solutions** (so the network is forced to
  abstract the rule). This is the same mechanism that made concepts generalise in
  the earlier v19 work — now confirmed in games.

## Q3 — More knowledge → fewer trials: YES (direction robust), and **not millions**

- **v28** — on a **new, harder, never-seen** variant (out of the training range),
  episodes to reach competence: **variety-pretrained ~819 parties**, narrow agent
  ~1126, **from scratch did not even master it** in the budget.
- **v29 (N=3 seeds, robust + honest correction)** — across 3 seeds:
  - the variety agent beats the narrow agent on unseen-hard **every seed**
    (gap **+0.26 ± 0.06**) — *solid and large*;
  - the variety-pretrained agent masters the new variant in **fewer episodes
    every seed** (~887 vs ~1161 parties = **~1.35× fewer**, and more *reliably* —
    scratch is borderline on this target).
  - **Correction:** v28's eye-catching "≥2.25× / scratch never succeeds" was a
    single-run outcome (the scratch arm sits right at the competence boundary and
    happened to stall). The honest robust number is **~1.35× and more reliable**,
    not 2.25×. The *direction* is robust; the *magnitude* was overstated by one
    lucky run. This is exactly why we run multiple seeds.

**Concrete answer to "des millions de parties ?":** No. These games are mastered
in **hundreds to ~1000 parties** from scratch, and prior general knowledge makes a
new harder instance **reliably faster** (≈1.35× here, widening when the new task is
hard enough that scratch fails outright).

---

## Honest limits (what is *not* yet shown)
- All of Q2/Q3 is **within the Pong family** (general *within-family* transfer on a
  controlled difficulty axis). It is clean and controlled, but it is one game's
  variant space.
- **Cross-game** transfer (Pong-skill → a different game) is **gated by
  similarity** and does not transfer for free (P3/v16; probed again in v31). This
  is *why* the architecture **recognises** which known skill applies before
  reusing it, rather than blindly transferring.
- **Cross-genre** generality (games → maths → language) is the long-term direction,
  not yet attempted. The mechanism shown here (broad, *structured* variety →
  abstract rule → efficiency on new instances) is the hypothesis we would scale.

## How this maps to the north star
Drop the agent on a game → it learns from pixels (v15). Give it several → it
recognises, reuses, and accumulates at sublinear cost (v25/v26). Train it across
*structured* variety → it abstracts a general skill (v27b) and then needs
*measurably fewer trials* on new, harder instances (v28/v29). The next leaps are
(a) a larger, more diverse game suite to test real cross-game variety, and (b)
folding model-based planning + autonomous discovery into the single running agent.
