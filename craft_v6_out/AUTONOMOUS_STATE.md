# AUTONOMOUS NIGHT STATE (started 2026-06-03, owner asleep ~7h)

Owner: "je te laisse gerer entierement pendant 7h, fais attention à ne pas te bloquer, fais-toi
reveiller par tes runs quand ils finissent (ou wake toutes les ~15min pour check)." Deliver a CLEAR
report when results land. Fallback idea to remember: brain-inspired modular zones + router (see
memory/decision_option2_and_brain_fallback.md).

## WAKE MECHANISM
- PRIMARY: background-run completion notifications re-invoke me (reliable all session).
- BACKSTOP: a ~25min watchdog (background Start-Sleep) re-invokes me to check for STALLS. Re-arm it
  each wake while a run is going. NEVER do a long foreground blocking monitor (would block the turn).

## CURRENT RUN
- v50 childhood-amortisation confirmatory: depth 7, 8 train trees, 5 held-out, WARM vs SCRATCH,
  --no-flat, seed 0. Background id (latest) = bmvpa3kgs. ~2h. Output: tasks/bmvpa3kgs.output ;
  JSON: craft_v6_out/v50_amortise_s0.json.
- Prereg frozen (preregistration.md, v50 section). Mechanism de-risked: childhood skill GENERALISES to
  held-out trees (nav 0.94 with 8 train trees). Exploratory 2-tree: WARM 3.32M masters held-out (0.86)
  vs SCRATCH 4.46M -> warm cheaper/task, break-even ~3.

## DECISION TREE ON v50 COMPLETION
- POSITIVE (warm-master-rate>=0.8 AND break-even reached AND mean warm<scratch):
  1. Spawn 2-3 ADVERSARIAL reviews (no GPU) to attack it (the discipline: dissent>consent).
  2. Launch 3-seed firm-up (seeds 1,2; same frozen config) for robustness (~4h). Also consider a FLAT
     baseline run (drop --no-flat on a subset) for the honest "vs end-to-end" reference.
  3. Record result honestly in RESEARCH_DIRECTION.md + commit.
- PARTIAL/NEG: diagnose (did warm fail to master? break-even>stream? skill too weak on some trees?).
  Retune (more childhood diversity / manager budget) with a FRESH prereg, OR if fundamentally stuck,
  pivot to the brain-inspired modular-router fallback (build it).
- ALWAYS keep a run going + watchdog armed. Commit/push often. No delete without asking.

## REPORT TO OWNER (when results solid)
Clear structure: per-tree WARM vs SCRATCH costs, cumulative curve, break-even, warm mastery rate,
honest caveats (within-family amortisation, NOT cross-game grail; reused skill slightly weaker than
fresh). Lead with the honest bottom line.
