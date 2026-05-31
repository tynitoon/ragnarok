# Night report — 2026-05-31 (autonomous block)

*For Jeremie, on waking. Written by Claude. Honest over flattering — this night
deflated some of my own earlier claims, and that's the point.*

## TL;DR
I pushed the sample-efficiency / "scale" investigation (v26–v30), then ran a
3-agent **adversarial review** of the whole recent arc. The reviewers were right
about a lot. I **corrected the record, falsified one of my own mechanistic claims
(v32), and tested the core thesis in a new domain (v33).** Net: the Pong
sample-efficiency story turned out to be **standard domain randomization** — real
but mundane, and I had over-framed it. The *general* "variety → generalization"
thesis, however, **does hold across domains** (including symbolic maths), which is
genuinely encouraging. Everything is committed and pushed.

## What I ran (all preregistered before the run, results recorded honestly)
| # | question | honest result |
|---|---|---|
| v26 | does a skill library make cost grow sublinearly? | **YES** — 15-game stream cost only the 3 distinct games (82% saved), recognition 100%. Solid. |
| v27 | variety over an "irrelevant" axis → general skill? | **NO** (kept as honest negative). |
| v27b | variety over paddle-*speed* → general skill? | win-rate gaps real, BUT see v32 — the *mechanism* I claimed was wrong. |
| v28 | more knowledge → fewer trials on a new variant? | directionally yes, but **magnitude overstated** (see v29). |
| v29 | …robust over 3 seeds? | direction robust; **magnitude corrected down 2.25× → ~1.35×** (a single lucky run had inflated it). |
| v30 | "variety scaling law"? | **PARTIAL** — trend present but RL-noisy and it failed its own preregistered bar; I kept it as partial. |
| **REVIEW** | 3 adversarial agents (rigour / ML / strategy) | found real holes — see below. |
| v32 | is v27b's benefit *anticipation* or just coverage? | **FALSIFIED my claim** — anticipation index ≈0 for *all* agents; it's domain-randomization **coverage**, not a distinct policy. |
| v33 | does variety→generalization hold *outside* games? | **YES, clean** — in-context maths regression, held-out MSE 3.47 (R=1) → 0.19 (continuous). Domain-general. |
| v31 | cross-game transfer (Pong→Breakout vs Snake)? | **INCONCLUSIVE** — Breakout (similar) −0.29, Snake (dissimilar) +8.27 (opposite of, and not robust to, my hypothesis; single unseeded run). Anecdote, not evidence — reinforces "build a real substrate". |

## What the reviewers caught (and I accepted)
1. **torch RNG was never seeded** — so "N=3 seeds" didn't control the dominant
   noise; the efficiency magnitude lives inside measurement quantization. Fixed in
   v32 (seeded); flagged everywhere else.
2. **"unseen-hard" was interpolation, not extrapolation** — those variants sit
   inside the trained difficulty range. Only the v28 target was truly OOD.
3. **The "+0.26, single-instance fails" gap was vs the weakest baseline.** A single
   *hard* instance reaches 0.69 ≈ variety's 0.71 on the hard set. Variety's real
   (modest) edge is covering *both* halves.
4. **It's domain randomization / contextual-MDP generalization** (Tobin 2017; Peng
   2018; Cobbe/Procgen; Kirk 2023) — not a novel "recipe". I should have framed it
   that way from the start.
5. **Strategy: substrate monoculture.** The arc is ~7 experiments inside *one* game
   family. The strongest "more knowledge → faster" evidence is the older
   **concept-granularity** work (v17b ~5–15×; v13b impossible→~10 iters), not the
   Pong ~1.35×.

## The honest bottom line
- **The Pong sample-efficiency arc = standard domain randomization.** Real,
  reproducible, but mundane, and the "anticipation" mechanism I attached to it was
  **wrong** (v32 killed it). The win-rate numbers stand; the story doesn't.
- **The variety→generalization *thesis* is real and domain-general** (concepts v19,
  games v27b-as-DR, **maths v33**). That's encouraging for the long-term vision —
  but it's *classic meta-learning*, not a new mechanism. Its value is showing the
  idea isn't game-specific.
- **The project's distinctive contributions are unchanged and remain the good
  stuff:** the developmental **loop** (recognize / learn / reuse + library growth,
  v25/v26), **autonomous discovery** (v7), and **concept-granularity compounding**
  (v17b). Those — not the variety effect — are where "more knowledge → faster"
  is *large* (10×, not 1.35×).

## Recommended next moves (reviewers + me, ranked)
1. **Leave the Pong sandbox. Build a genuinely diverse game suite** (MinAtar-style,
   ≥10 structurally-different games, or several new native games beyond paddle-ball)
   and re-run the v25 unified-agent loop over it. *Decisive north-star question:*
   does a library built on games 1..N make game N+1 cheaper **in episodes**, when
   N+1 is *not* visually similar to any prior game? That has never been tested with
   real diversity (our 3 games are 2 paddle-ball + 1 grid).
2. **Re-center the concept-granularity line** (CraftWorld multi-tech-tree transfer)
   — that's where compounding is large and where "discovers its own curriculum"
   (v7) lives, the most north-star-relevant capability we have.
3. **Extend the symbolic direction (v33)** toward real maths/language meta-learning
   — it's the first concrete evidence the recipe spans domains, and it's clean.

## On your specific question ("would it need millions of games?")
Still **no** — but the honest support is the **concept-granularity** result (learn
the right reusable concept → 5–15× fewer games on a genuinely hard task), not the
Pong ~1.35×. Hundreds-to-~1000 games to master these arcade games from scratch;
much less when a *relevant* prior concept exists; roughly no help when it doesn't
(transfer is similarity-gated — which is exactly why the agent *recognizes* which
skill applies before reusing it).

*All commits on GitLab. Preregistration.md has the full, timestamped, honest log
including every correction and the falsification.*
