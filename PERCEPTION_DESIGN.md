# From-scratch developmental model — Lock #1: object-centric PERCEPTION

This is the start of the **from-scratch model** the owner asked for: build our own architecture
brick by brick, starting with the perception core, and **build nothing on top until it is solid.**

## Why perception first

The whole-session map concluded the grail is gated by two open problems: (1) learned **perception**
of clean quantities/objects from pixels, and (2) **induction** of expensive reusable structure. Every
reuse mechanism we tried died for the same reason class — a monolithic CNN encoder learns whatever
the task needs from raw pixels anyway, so a handed-over "notion" as an input feature is redundant, and
blind encoder transfer is appearance-similarity that fades. The bet of the from-scratch model is that
**object-centric + relational** structure is a different substrate where reuse can finally pay,
because objects and their interaction laws are what actually transport across games. But that bet is
only testable if we can first get **clean object representations from pixels, unsupervised**. That is
lock #1.

## What failed: slot-attention (percept v0.1)

A clean slot-attention autoencoder (Locatello 2020-style: slots compete via attention, spatial-
broadcast decoder, mask competition). On our **sparse sprite scenes** (mostly-black frames, a few
small bright objects) it did NOT bind objects: reconstruction improved (0.011) but masks bloated to
~half the frame and the ball-tracking error WORSENED over training (0.21 -> 0.26). Foreground
weighting + mask-entropy + LR warmup did not rescue it. Diagnosis: slot-attention's soft masks have
no pressure to *localise* a small object when the background is trivially predictable; the labour
division it needs (many similar objects, textured background) is absent in our scenes.

## What works: explicit sprite autoencoder (percept v0.2)

Make the object-ness **explicit in the architecture** instead of hoping it emerges from soft masks.
Each of K slots is literally an object:
- a **position** (x, y) in [0,1]^2,
- a **presence** in [0,1],
- a small **appearance patch** (P x P x 3).

A differentiable **write spatial-transformer** (affine_grid + grid_sample) pastes each patch onto the
canvas at its position; the composite (sum_k presence_k * placed_k) must reconstruct the frame.
Trained on **reconstruction only** — no labels, no position supervision. The **position bottleneck**
forces each slot to commit to one localised region, i.e. one object. (This is the AIR / SQAIR /
spatial-transformer family; we use it as the perception brick because it is the right inductive bias
for sprite scenes.)

### Validation (unsupervised; the env's true positions are NEVER used in training)
- **Primary (load-bearing):** a FIXED slot's position tracks the **2D-moving ball** with mean error
  below a FAIR min-over-K random baseline and small in absolute terms, and that slot is the per-frame
  closest one in the large majority of frames (STABLE binding, not a per-frame min-trick).
- **Secondary:** greedy distinct-slot assignment maps ball + the two paddles to THREE DISTINCT slots
  -> genuine scene decomposition, not mere ball-saliency.

Smoke (1 seed, 2000 steps) was strongly positive: ball -> a fixed slot at ~0.05 err (fair-random
~0.23), stable ~90-95%; ball+padL+padR -> distinct slots. The frozen 3-seed run is the real test.

## Honest scope — what this is NOT

- **Perception is NECESSARY, NOT SUFFICIENT.** Clean object slots do not by themselves give reuse. The
  load-bearing grail question — does object-centric+relational structure enable reliable reuse where a
  monolithic encoder failed (our own prior nulls)? — is the NEXT experiment (relational world-model
  over slots, then fair cross-game transfer). Nothing here claims reuse.
- **Pong paddles are an easier object** (their x is constant at 0.06/0.94, only y varies). The BALL,
  which moves in 2D across the whole field, is the load-bearing claim; paddles are a bonus. A second
  game with multiple 2D-moving objects is the generality check before building upward.

## RESULT percept v0.2 (3 seeds, frozen run) — NEGATIVE at the bar

| seed | ball-slot err | stability | padL | padR |
|------|--------------|-----------|------|------|
| 0 | 0.076 | 0.47 | 0.017 | 0.018 |
| 1 | 0.214 | 0.57 | 0.042 | 0.042 |
| 2 | 0.204 | 0.48 | 0.075 | 0.008 |

Fair-random baseline ~0.25. **The load-bearing ball claim FAILS the frozen criterion** (err<0.06 AND
stability>0.8): on 2/3 seeds the ball is essentially UNBOUND (err ~0.21 ~ random 0.25); the best seed
(0.076) still misses; stability ~0.47-0.57 on all seeds. The PADDLES bind reliably (0.008-0.075) — but
they are the EASY object (constant x, pre-flagged). The 1-seed/2000-step smoke (0.048, 0.95) was a
small-sample fluke; the frozen 3-seed run is the honest test. **percept v0.2 = NEGATIVE.**

Diagnosis: the recurring project obstacle. The ball is ~1-2 px in a mostly-static frame; single-frame
reconstruction (even foreground-weighted) is dominated by the larger paddles, leaving no pressure to
isolate the tiny ball. The ball's one distinguishing feature is that **it MOVES every frame**. ->
percept v0.3 makes object discovery driven by MOTION (the moving thing is the object).

## RESULT percept v0.3 (motion weighting) + cheap sweep — STILL NEGATIVE on the ball

Single change vs v0.2: reconstruction weighted by motion |frame_t - frame_{t-1}| (the ball moves every
frame). Cheap smoke sweep (the ball is the load-bearing object):
- motion-w 3.0, K=4: ball err 0.205 ~ random (0.233), stability 46%. No help.
- + K=3 (one slot/object): ball err 0.132, stability 77%. Stability up (no spare slot), precision stuck.
- + patch=6: err 0.104, stability 35%. + patch=8: err 0.128, stability 66%. No clean win.

Refined diagnosis: per-frame err is fine (~0.066) — SOME slot covers the ball each frame — but no
FIXED slot OWNS it. Feed-forward slots **partition the image by REGION**, so position-stable objects
(paddles) get stable slots while the ROAMING ball is handed between slots frame-to-frame. The blocker
is not gradient pressure (motion didn't fix it) but **object identity/permanence**. Tweaking
slots/patch/motion cannot fix a missing temporal-consistency mechanism.

-> percept v0.4: **temporal object permanence** — carry slot state across frames (SAVi/SQAIR-style) so
the slot that binds the ball keeps tracking it. Still learned, no labels (self-supervised over video).

## RESULT percept v0.4 (temporal recurrent tracker) — finicky, did NOT converge

Carried slot-attention + GRU transition + sprite renderer, trained on video. recon barely moved
(0.76 -> 0.74 over 1500 steps; feed-forward v0.2 dropped 15x in 2000), slots COLLAPSED (final per-slot
all ~0.32 from ball — every slot near the center, none binding), ball err 0.32 > fair 0.26. Recurrent
slot models (SAVi/SQAIR-class) are known to need ~100k steps + careful init/warmup; not crackable in a
quick single-GPU budget. NEGATIVE here (a tuning problem, not disproof of permanence — but not worth
grinding finicky recurrence).

## DECISION (ownership) — the root cause + the right tool

Across v0.1-v0.4 the consistent root cause is **object IDENTITY / permanence**: permutation-free slots
(attention OR explicit sprites) partition by region, so a roaming object has no stable slot; recurrence
would fix it but is finicky. The structural fix that AVOIDS both pitfalls: **channel-indexed
keypoints** — a CNN outputs K heatmaps, one per FIXED output channel, position = spatial soft-argmax.
Keypoint k is the SAME channel every frame, so identity is stable BY CONSTRUCTION (no assignment, no
recurrence). Trained unsupervised by CROSS-FRAME reconstruction (Jakab 2018 / Transporter, Kulkarni
2019): reconstruct frame x' using APPEARANCE from a different frame x but GEOMETRY (keypoints) from x',
which forces keypoints onto what MOVES (the ball, paddles). This is the standard, feed-forward (fast)
tool for unsupervised object keypoints on moving scenes (validated on Atari in the literature). ->
percept v0.5. A SOLID multi-game unsupervised keypoint result is the perception foundation the owner
asked for, and the first reviewable positive on this from-scratch path.

## Roadmap (build upward only after each lock holds + adversarial review)
1. [in test] Perception: stable object slots from pixels, Pong, 3 seeds. -> then 2nd game.
2. Relational dynamics: predict next slots from (slots, action) with per-object + pairwise-relational
   structure; validate prediction. The substrate where reuse might pay.
3. Reuse test: does the relational model transfer across games FAIRLY (counting all compute), beating
   a monolithic baseline — the thing every prior mechanism failed. Only this would touch the grail.
