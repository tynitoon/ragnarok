# scripts/ — index

145 scripts across five research eras. This index exists because older documents
(`preregistration.md`, `RESEARCH_DIRECTION.md`) cite paths like `scripts/foo_v13.py` that now
live under `scripts/archive/<era>/`. **Those documents are append-only records and were NOT
rewritten** — retro-editing a frozen preregistration would be a discipline violation. Use the
table below to resolve any older path.

Archived scripts still run: `python -m scripts.archive.<era>.<name>`. All 88 were verified to
compile after the move, and the live pipeline was verified to import.

## Top level — live and shared

Anything imported by another module stays here (moving it would break imports), plus the live
ARC-2 pipeline, the demos, and the test/score/run entry points.

| script | role |
|---|---|
| `evidence_store_v58.py` | ARC 2 — the per-world evidence store + hand-coded reference policy |
| `evidence_net_v58.py` | ARC 2 — the identity-free policy, env, buffer, scored cost |
| `calibrate2_v58.py` | ARC 2 — fits the thresholds from measured noise |
| `run_confirm_v58.py` | ARC 2 — the confirmatory runner (pretrain + test phases) |
| `score_v58.py` | ARC 2 — the frozen scorer |
| `gate_k1_v58.py`, `test_*_v58.py` | ARC 2 — gate and unit tests |
| `hidden_recipe_v55.py`, `credit_fix_v57.py` | ARC 1 — frozen mechanism, imported by ARC 2 |
| `meta_manager_v51.py`, `childhood_v50.py`, `depth_scaling_v49.py` | shared env/skill infrastructure |
| `play_v55.py`, `play_world.py`, `demo.py` | watchable demos |

## `archive/arc1_v49_v57/` — ARC 1 (v49-v57) — the necessity arc, closed with three NULLs

  amortise_v52 goal_ablation_v54 transfer_experiment_v310 transfer_experiment_v311 
  transfer_experiment_v312 transfer_experiment_v313 transfer_experiment_v314 transfer_experiment_v315 
  transfer_experiment_v316 transfer_experiment_v318 transfer_experiment_v319 transfer_experiment_v320 
  transfer_experiment_v321 transfer_experiment_v322 transfer_experiment_v323 transfer_experiment_v324 
  transfer_experiment_v325 transfer_experiment_v326 v53_review_probe 

## `archive/notions_v30_v48/` — Notions & model-based (v30-v48) — gravity, notion libraries, planning

  anticipation_probe_v32 compose_hard_v48 concept_gravity_v36 concept_mastery_v37 
  crossgame_accumulation_v35b diag_v45_masterable model_based_learned_v44 model_based_v43 
  notion_amortizes_search_v42 notion_library_v39 pixel_notion_v40 pixel_notion_v41 
  symbolic_variety_v33 variety_scaling_v30 verify_v45_critique 

## `archive/pixels_games_v10_v29/` — Pixels & arcade games (v10-v29) — Pong, Tetris, Snake, Flappy, Breakout

  accumulate_v16 concept_generalize_v19 concept_ingame_v20 devloop_pixels_v13b devreuse_v13 
  dreamer_v12 flat_shaped_v13c integrate_v21 integrate_v23 latent_mpc_v12 learned_nav_v11 
  modelbased_tetris_v17 modelbased_tetris_v17b perception_v12 robustness_v29 scale_unified_v26 

## `archive/early_v1_v9/` — Early (v1-v9) — transfer, composition, the first developmental loop

  compose_v4 craft_devloop_v6 craft_sanity_v6 craft_skill_v6 hrl_ordered_visit_v4 learned_gate_v5 
  mbrl_compounding_v4 mbrl_pointmass_v4 ng_v01_predict ng_v02_reuse ng_v03_control percept_v01_slots 
  percept_v03_motion percept_v05_keypoints percept_v05b_game2 rel_v01_bounce 

## `archive/misc/` — Misc — benchmarks, probes, one-off analyses

  bench_device_env bench_device_ppo bench_vec_collection compute_budget_extrapolation 
  device_pilot_run device_recalibration migrate_checkpoints ng_probe ng_probe_reuse 
  ng_probe_specialization pilot_analysis ree_r01_catcher ree_r02_gravity ree_r02c_reuse 
  ree_r03b_robust ree_r04_reliability smoke_benchmark smoke_verdict validate_device_agent 
  validate_device_ppo validate_device_sac validate_device_wm 

