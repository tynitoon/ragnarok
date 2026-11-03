# ARC 4 v85 — I-stage, first pass: result and independent verification (3 November 2026)

**Status: no verdict (Interruption).** Kept as the first pass. A dated restore fix and a full I-stage rerun follow
(§10 allowed change 5); the rerun's verdict is the binding one. Outputs: `craft_v6_out/v85/istage/` (commit `980a211`),
verdict `istage_verdict.json`. Verification workflow `wf_7b462b2f-1b5` (independent recomputation without the verdict code,
audit of the verdict code, restore-artifact investigation, adversarial synthesis); scratch scripts and results in the
session scratchpad `review/`.

**Résumé (FR).** L'étape I n'a pas de verdict : un run de yars_revenge ne se rejoue pas, à cause d'un défaut d'ALE
(la restauration ne remet pas les « masques d'objets » de la puce graphique TIA ; une collision mortelle est manquée et le
run garde une vie que le jeu continu n'a pas). Le contrôle D est réussi. La ligne instrument (R 0,809, IC99 [0,682 ; 0,964])
échouerait de justesse ; aucune erreur de code ; la règle figée donne en réalité ~3 % de fausses alertes au lieu de 1 %.
Aucun bras de savoir n'a tourné : rien n'a été mesuré sur la place, le transfert ou l'apprentissage accéléré.

---

# v85 I-stage: adversarial synthesis under the frozen rules

## 1. Stage label

**Interruption is the correct label.** A.5 says any replay mismatch is an Interruption, and A.13 says an Interruption gives no verdict. Build note §1.2 states that V2s mismatches are Interruptions. The code also sets `K_gs = null`, so nothing can be calibrated or frozen from this pass. The INSTRUMENT-FAIL figure (R = 0.809, CI99 [0.682, 0.964]) is therefore a would-be label, not a verdict. G1 cannot run under either reading: the Interruption blocks F1, and INSTRUMENT-FAIL would stop before G1.

**No code bug was found.** Verifiers A and B reproduced every number independently: T, the cross-fit, ℓ_g, the clipping, the frozen bootstrap stream `[9805, crc32("I:linea:linec")]` and the stage order. Three real problems were found, none of them in the code:
- **Emulator defect (verifier C).** In ale_py 0.12.1, restoreState does not restore the TIA object-mask pointers, so the result of a restore depends on what the emulator ran before it. On yars at link 290, a stale M0 mask made a lethal collision miss. Continuous play reaches game over at link 292; the live branch keeps its last life and gains 46,924 points that continuous play never produces. The evidence is empirical (M0 is drawn at the previous emulator's POSM0). I did not re-check the source-level claim about `TIA::save`/`load`.
- **Replay check gap in the frozen A.5 rule.** Only the records at k_first and at the highest reached rung of K_g are replayed. On yars/A/9807 the mismatch was found on the rung-5 record. That rung does not even enter λ at start A, where the cross-fitted K is {3, 4]. The inflated time that does enter λ, rung 4 at 169,720, was never checked. Overall, 104 of the 275 reached T values that enter λ (38%) were never replay-confirmed; 68 of 183 (37%) among the line's four seeds. That last count is my own (`s3_unconfirmed.py`).
- **The frozen 99% interval is miscalibrated.** With one run per cell, the two-level bootstrap reduces to a percentile bootstrap over 16 unit values. On exchangeable seeds it gives a false INSTRUMENT-FAIL 2.3 to 4.0% of the time, not 1% (A and B). This is a property of the frozen rule and cannot be changed now.

**Would it still be INSTRUMENT-FAIL once yars is handled? Probably, but at a knife edge.** "Without yars, it still fails" is the wrong counterfactual. The artifact sped up seed 9807, which sits in the comparator group {9807, 9810}, so it pushed the line in the failing direction. Correcting it moves yars's log r_g upward. I recomputed the line on the exact frozen bootstrap stream (`s2_yars_frozen.py`), holding the other 15 units fixed:
- If the corrected 9807 gets the rung times of any of the five sibling yars runs, the line still fails, with upper bounds from 0.963 to 0.999.
- It flips to PASS only if the corrected 9807 needs more than about **1.36M billed frames for rung 4** (when it still reaches rung 5) or about **2.07M** (when it misses rung 5, which also shrinks yars/B's cross-fit to {3, 4}). Its two siblings at start A took 1.20M and 1.40M.
- The corrected run is effectively a new draw: the first history-sensitive restore comes at about 49k billed frames, before its rung-3 time of 96,176.
- If 9806's rung-4 record (also unchecked) was inflated, correcting it can only deepen the failure.
- The other 95 runs should rerun byte-identically: C found no history-sensitive restore outside yars in 6,051 early-search restores and 1,291 random states. This has not been checked on the full I-stage runs.

## 2. What the I-stage establishes about the yardstick

**Established:**
- **D control passed.** r_g(D:V2s) = 8 (clipped) on 16 of 16 units, and every D confirmation is exact. The smallest unclipped ℓ is 2.64 (about 14×). D makes no restores inside the run, so the artifact cannot reach it.
- **Regime ratio.** Natural V2 from its own M1 against V2s from s1: R = 1.015, 90% CI [0.855, 1.208].
- **The noise is large and roughly as v84 predicted.** The pooled within-cell SD of λ is 0.467 (median 0.30, largest 1.30), against v84's raw variance of 0.200.
- **The frozen line is marginal:**
  - r_g < 1 on 13 of 16 units, and leaving out any one unit still fails;
  - t-test p = 0.0101, and the t-based 99% interval [0.655, 1.0002] just includes 1;
  - exact sign-flip p = 0.0096; Wilcoxon p = 0.016;
  - p = 0.068 when seeds are relabelled within each (unit, start) cell, which is the null as stated.
- **No shared seed effect is detectable.** F test p = 0.54 at A and 0.11 at B; combined permutation p = 0.22. The other pairings: R(9806/9809 : 9808/9811) = 1.03 and R(9807/9810 : 9808/9811) = 1.27, and both 99% intervals include 1. Neither start alone excludes 1 (A: 0.895 [0.644, 1.260]; B: 0.761 [0.599, 1.051]). No mechanism in the random streams links 9807 and 9810.

**Not established:**
- Whether the alarm reflects a real seed-group effect or chance. With 3 seeds per cell the two cannot be separated. Measured against the third seed's noise, the observed |log R| = 0.212 is about 1.8 times its expected spread.
- Whether the line fails on artifact-free data (see section 1).
- Whether this yardstick could separate 2× from 1.5×. The calibration never ran.

**My reading.** The noise is real and as expected. The alarm is at the boundary of a rule whose real false-alarm rate is about 3%, on data that include one artifact-inflated comparator run. The alarm did coincide with a real defect of the yardstick: restores are not reproducible on yars. None of this can be used to set the frozen rule aside.

## 3. What the restore artifact means for v83 to v85 results

**Cannot be affected:**
- **v83 A1 KC-room (6/35, threshold 13).** It is a negative result. Removing an artifact that the explorer's selection mostly turns to its advantage cannot raise it to 13, and yars was not among its room games.
- **v84 G1b PROCEED, 22/35.** Criterion (b) rests on records at M_top ≥ M3, replay-confirmed on both seeds: 198 of 200 checks exact after one NOOP, and 2 Q\*bert checks exact in the post-reset context, which is the context the live run itself used. Each of these is a genuine sequence of continuous play, so (b), (c) and (d) stand, including yars's room status and k\*.
- **v84 G0 and G0-S (RECIPE-INSENSITIVE).** Both were decided by planted controls.
- **v85 S0 starts.** The 32 starts and their lineages were validated by continuous replay (A.3.1) and by the identity check in the exact `start_from` context (A.3.4). Every arm uses the same reset + NOOP + restore, so the hidden state at s1 is identical across arms.
- **v85 D control.**

**Could be affected (design inputs and captions, not claims):**
- v84 times at intermediate rungs were never confirmed (only M1, M2 and M_top were). They feed v84's σ̂² and se_proj (0.142 against a threshold of 0.23, so the margin is large) and v85's counted-rung gaps (A.4) and natural spike-ins (U = 16). Any effect there is a choice of rungs applied to every arm alike, not a bias between arms.
- The v84 caption "climbs on most" (21/35 with M3 at depth ≤ 7,000; threshold 18) uses the first M3 record, which was never confirmed. At most yars could drop out on present evidence (21 → 20, still ≥ 18).
- The v83/v84 explanation of the Q\*bert mismatches ("a post-reset context outside saved states") is very likely a special case of this mechanism, which depends on any earlier emulation, not just a reset. The verifier rule "exact in either context" remains valid. The documented lesson should be corrected.

**Is affected:**
- **v85 I-stage V2s data.** yars/A/9807 is invalid (rungs 4 and 5 inflated). The other five yars runs, and every other run's intermediate rungs, are unaudited.
- **Anything still to come that uses restores:** the O_4 and K\* sandbox look-aheads (a possible source of DiceMismatch aborts) and the v86 stall rescue. Some v84 stall states may sit on tainted lineages and must be validated by continuous replay. The explorer selects for the rare states the bug produces: 53 of 4,037 restores (1.3%) were history-sensitive in 9807, yet 221 of its 284 records descend from them.

## 4. Result statement

> v85's I-stage completed all 128 runs (96 V2s, 32 D) on 16 development units. It has no verdict: one V2s run (yars_revenge, start A, seed 9807) failed its replay check, which the frozen protocol treats as an Interruption. The cause is an emulator defect, not agent code. ALE 0.12.1 does not save the TIA's object-mask pointers, so a restored emulator can compute collisions with masks left from its earlier history. On this run, a lethal collision was missed and the climb to rungs 4 and 5 came from a life that continuous play does not have. The D control passed (r_g = 8, clipped, on 16 of 16 units; every confirmation exact). The frozen seed-split line gave R = 0.809, 99% CI [0.682, 0.964]. On a valid stage this would be INSTRUMENT-FAIL and would stop v85 before G1. Two independent recomputations reproduce it and found no code bug. The failure is marginal (t-test p = 0.010, sign-flip p = 0.0096, p = 0.07 against the third seed's noise). No seed effect shared across games is detectable, and the frozen interval fires about 3% of the time on exchangeable seeds, not 1%. Once the inflated run is corrected, the line still fails unless that run needs more than about 1.36M billed frames for rung 4; its two sibling seeds took 1.20M and 1.40M. No knowledge arm ran: nothing about room, transfer or faster learning was measured.

En français : l'étape I n'a pas de verdict (Interruption). Un défaut d'ALE rend une restauration dépendante de l'historique de l'émulateur, et un run yars a monté grâce à une vie que le jeu continu n'a pas. Le contrôle D est réussi. La ligne instrument (R 0,809, IC99 [0,682 ; 0,964]) échouerait de justesse ; elle échouerait probablement encore une fois yars corrigé, mais à la limite. Aucun savoir n'a été testé.

## 5. What the frozen protocol says happens next, and the options

**Frozen path:**
1. **Now.** No verdict; no calibration, no F1 and no G1 (A.13, A.10.6).
2. **Handling the Interruption.** §10 allowed change 5 permits a dated amendment for an Interruption, committed before the data it affects, with first-pass outputs kept. For G1, Part C words this as "one dated fix; rerun labelled".
3. **What a fix forces.** Fixing `senses_v85` changes the V2s and D closures. §13.4 allows one closure per arm type within a gate, so all 96 V2s and all 32 D runs must be redone. Rerunning only 9807 creates mixed closures, which is itself an Interruption. A patched ale_py would escape the closure hash, so its build hash must be recorded and treated as a closure change.
4. **After a rerun:**
   - If the line holds: A.10.5 calibration (UNDERPOWERED after one top-up means Jérémie decides), then F1, then G1.
   - If INSTRUMENT-FAIL: G1 does not run and v85 ends. §11's default v86 is written for NO-ROOM or INCONCLUSIVE only, so the continuation after INSTRUMENT-FAIL is Jérémie's decision.
5. **Not allowed:** running G1 now; dropping yars; swapping the percentile interval for a t or sign-flip test; re-pairing the seeds; treating the without-yars figure as a verdict.

**Options, ranked:**
1. **Dated restore fix and full I-stage rerun, with a binding verdict.** The fix is needed for every arc that uses savestates. Because runs are deterministic, the rerun is not a second roll of the dice: only runs the artifact touched change. It needs:
   - A dated build note covering the mechanism, the fix and its scope (explorer handle restores and the sandbox; `start_from` and `replay85` left alone, so S0 stays valid) and billing (I suggest keeping the fixed 60 frames per restore, which preserves byte-identity on untouched runs).
   - A fix chosen on exactness only: state equality against a continuous twin on the 53 known sensitive restores, link 290 and the 1,291-state sweep. C's warm-up option (c) is the cheapest; pre-roll (b) is closer to continuous play; an ALE patch (a) fixes the root cause.
   - The binding rule committed before any corrected yars run. Nobody has yet run the 9807 explorer under a fix, so the choice can still be made blind to R.
   - Two report-only additions: replay every counted rung, and byte-compare each rerun against its first pass.
   - Cost: about 8 h on 4 workers, or about 10 h on 3 after Tavern returns (5 Nov, 06:00).
   - My rough odds that G1 then runs: 0.2 to 0.3. The line probably still fails, and calibration can still fail.
2. **If INSTRUMENT-FAIL stands: a v86 that keeps G1's question with a properly calibrated design.** Fixed emulator; at least 6 V2s seeds per start; an instrument check whose false-alarm rate is simulated on these I-stage data before freezing; every counted rung confirmed by replay; S0 starts, entry caches and D reused under hash checks. It costs about 0.3 to 0.6G frames before G1.
3. **§11's stall rescue** (P(one rung within 2M)). Its binary endpoint is less exposed to the heavy-tailed λ noise. It needs the fix, stall states validated by continuous replay, its demonstration and H-teleport controls, and a new frozen protocol. Options 2 and 3 are close; 2 reuses more of what has been built and verified.
4. **The rewind-price dial toward live play.** Fewer restores means less exposure to the artifact and a step closer to the ARC 4 vision, but it is the furthest from a near-term result.

Scratch scripts I wrote (no worktree edits, no commits):
- C:\Users\jeremie\AppData\Local\Temp\claude\F--dev-ragnarok\826ba484-cdf9-442a-89b7-2049c18871fc\scratchpad\review\s1_yars_sens.py
- C:\Users\jeremie\AppData\Local\Temp\claude\F--dev-ragnarok\826ba484-cdf9-442a-89b7-2049c18871fc\scratchpad\review\s2_yars_frozen.py
- C:\Users\jeremie\AppData\Local\Temp\claude\F--dev-ragnarok\826ba484-cdf9-442a-89b7-2049c18871fc\scratchpad\review\s3_unconfirmed.py