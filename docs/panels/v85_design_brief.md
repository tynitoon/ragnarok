# v85 design brief (for the design panel, 24 October 2026)

## Goal (owner, unchanged)
Show that the agent learns a new game FASTER thanks to what it learned on earlier games, honestly and
presentably (Atari now; NES, 16-bit, PC later). A head start is not faster learning. Innovation wanted;
criteria fixed before results; negatives kept; no oracle information in the agent's inputs (pixels,
buttons, score rises and lives as senses; RAM evaluator-only); no downloaded weights; reserved games
(boxing, frostbite, riverraid, kangaroo) never opened; enduro excluded.

## Where we stand (read these files)
- v84 ended at feasibility (F:\dev\ragnarok_v84, docs/ARC4_V84_PROTOCOL_EN.md, docs/ARC4_V84_BUILD_NOTES.md
  notes 1-19, craft_v6_out/v84/{room_verdict_v84.json, g1a_report.json, g0/g0_verdict.json,
  g0s/g0s_verdict.json}; handoff F:\dev\ragnarok\docs\CODEX_HANDOFF.md, last sections).
- **Asset: V2**, a fresh savestate explorer (frame-code cells with adaptive resolution, four button
  processes with a Thompson bandit, plays on after a life loss, no probes) that CLIMBS: it reaches M3
  (about 5x the best simple-policy bar) on 22/35 development games within 4M billed frames (two seeds),
  where v83's explorer managed 6/35. It logs promise rows and one 44-feature expansion row per rollout.
- **Instrument lesson**: the offline "knowledge clock" (how fast a learner reaches fractions of its own
  end-of-stream AUC) cannot show 2x even with near-oracle priors: planted positive controls gave
  same-game ceilings of 1.62, 1.44 (consequence heads) and 1.16 (expansion critic). Both carried-knowledge
  routes were closed by the instrument, not by a test of the mechanism. No carried object was ever run live.
- **Content lesson**: heads predicting score rises / life losses learned on 40 games predict ~nothing on
  held-out games (AUC 0.52; v83 exploratory). Earlier: distillation of branch values into a reactive
  policy failed (v72-v74); a readable "theory" did not beat climatology (v80); generic score reading
  failed (v81).
- **Process lessons**: simulate an instrument's achievable ceiling with an oracle prior before freezing
  thresholds; ALE keeps a post-reset context outside saved states (replay checkers must reproduce it);
  supervisor kills are routine (Tavern needs 12 GB): jobs must be resumable and deterministic.
- Pools: 35 dev + 6 calibration games are burned (A1, G1b logs exist); 24 sealed + 9 second-round games
  remain UNOPENED (7 sealed games are hard exploration: montezuma_revenge, pitfall, venture, private_eye,
  solaris, gravitar, skiing). Seeds 9700-9749 are v84's; choose a fresh block.
- Machine: 12 threads, RTX 4080 16 GB (shared), ~3-4 GB RAM for Ragnarok; a 4M V2 run takes ~30-50 min
  on one core; G1b-scale (82 runs) takes ~15-24 h on 3 workers.

## The question for the panel
Design v85 so that its FIRST decisive gate answers, cheaply and with an instrument whose ceiling has been
verified: **does ANY knowledge (even same-game, oracle-ish) make V2 climb at least ~2x faster LIVE,
measured from matched rungs (not a head start)?** If yes, which carried, game-agnostic object could
transfer it across games, and how to test that against tuned defaults and placebos on the sealed games
once. If no, what that means for the project and what substrate or learner would make knowledge matter
(e.g. hard-exploration games where fresh search fails, as for humans). Include: mechanism, live
yardstick (with a simulated/oracle ceiling check before freezing), arms and controls, gates and kill
criteria, pools and seeds, compute/RAM estimates, honest odds, and novelty versus the literature
(Go-Explore and its learned-policy variants, Agent57/NGU, meta-learned exploration, options/skills).
