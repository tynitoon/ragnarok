# Ragnarok

**Modular reinforcement-learning agent with skill crystallization and cross-action-space transfer.**

Ragnarok is a research project exploring whether an RL agent can crystallize a learned skill (policy + world-model subset) from one task, then transfer a latent core of that skill to a new task with a *different action space* (discrete ↔ continuous) and learn faster than training from scratch.

The codebase is a solo-dev research prototype run with preregistration-grade methodology: every hypothesis, threshold, and analysis choice is committed to `preregistration.md` **before** data is collected, with a public chronology audit for any amendments.

---

## Status

**As of 2026-04-17**, the project is in Phase 3 (pilot #2 validation) with a preregistered extension to N=10 on the primary pair in progress.

| Milestone | State |
|---|---|
| Phase 1 (architecture) | ✅ Complete |
| Phase 2 (single-skill learning) | ✅ Complete |
| Phase 3 pilot #2 (3 pairs × 5 seeds) | ✅ Complete (40 runs, 12.65 GPU-hr) |
| Band B rescue (5 seeds on primary) | ✅ Complete (2026-04-17) |
| Band C N=10 extension (5 additional seeds) | 🟡 In progress |
| Phase 3 analysis + decision (workshop paper vs pivot) | ⏳ Pending Band C |
| Phase 5 (post-workshop scale, Post-1 horizontal) | ⏳ Planned |

**Current primary-pair result** (seeds 47–51, Band B rescue, cartpole→mountaincar-continuous):

- RMST ratio (scratch/transfer) = **1.605**
- Log-rank p (one-sided, permutation N=10k) = 0.259
- Leave-one-out minimum ratio = 1.435 (robust, not outlier-driven)
- Mechanism: 5/5 transfer runs on `latent` acting mode, 5/5 loaded a crystallized skill

The ratio is above the Band A threshold (1.30) but p-value is underpowered at N=5. Band C (N=10 pre-registered) is running to stabilize statistical significance.

---

## The research claim

1. **Skills can be crystallized** from a trained Dreamer-style agent as a tuple `(RSSM_core + prior + posterior + policy_trunk + latent_centroid)`.
2. **A subset of the RSSM** (GRU core + prior + posterior, excluding encoder/decoder) is transferable across tasks with *different observation and action dimensions*, via `load_state_dict` with strict shape compatibility on the transferable subset only.
3. **The transferred latent trunk accelerates new-task learning** compared to scratch, measured via restricted mean survival time (RMST) on a mastery threshold.
4. **The transfer works even across action-space types** (e.g., discrete CartPole → continuous MountainCar), because the latent trunk operates on `cat(h, z)` features upstream of the action head.

Claim 4 is the novel contribution. Cross-action-space transfer with a shared latent trunk is not published in the mainstream RL transfer-learning literature to our knowledge.

---

## Repository layout

```
ragnarok/
├── core/               # RSSM world model, encoder, policy head
│   ├── rssm.py         # Recurrent state-space model + transferable subset
│   ├── agent.py        # RagnarokAgent orchestrator, try_transfer logic
│   ├── policy.py       # Actor-critic policy head
│   └── obs_encoder.py  # Observation encoders (MLP + CNN)
├── learning/           # Training algorithms
│   ├── sac.py          # Soft Actor-Critic
│   ├── dreamer.py      # Dream-based policy training
│   ├── world_model_trainer.py
│   ├── curiosity.py    # Intrinsic motivation (latent KL surprise)
│   └── ewc.py          # Elastic Weight Consolidation (defined, not yet wired)
├── memory/             # Replay + episodic buffers
├── skills/             # Skill crystallization + library
│   ├── library.py      # SkillLibrary, save/load, latent-centroid indexing
│   ├── selector.py     # Nearest-neighbor skill selection (warmup-based)
│   ├── router.py       # CentroidRouter + LearnedRouter (latter unused)
│   └── multi_agent.py  # Multi-skill execution-time routing
└── environments/       # Env wrappers, normalizers
scripts/
├── pilot_run.py        # Phase 3 pilot pipeline (smoke + N-seed runs)
├── pilot_analysis.py   # §8 preregistered verdict analyzer (RMST, log-rank)
└── smoke_verdict.py    # Pre-pilot smoke abort logic
tests/                  # 444 tests (pytest); run with: ./venv310/Scripts/python.exe -m pytest
preregistration.md      # Preregistered study protocol + all amendments (§13)
reviews/                # Multi-agent reviews, chronology audit, research directions
pilot_results.json      # Pilot #2 seed-level data (primary + 2 secondaries)
pilot_bandb_results.json # Band B rescue seed-level data (N=5)
pilot_bandc_results.json # Band C N=10 extension (in progress)
```

---

## Running it yourself

**Environment**: Python 3.10 (Python 3.11+ has issues with `mujoco` wheels on some platforms; `venv310` is the tested path).

```bash
git clone https://gitlab.com/mortier.jeremie/ragnarok.git
cd ragnarok
python3.10 -m venv venv310
./venv310/Scripts/python.exe -m pip install -r requirements.txt
./venv310/Scripts/python.exe -m pip install -e .
```

**Test suite** (444 tests, ~3 min):
```bash
./venv310/Scripts/python.exe -m pytest tests/ -x
```

**Reproduce pilot #2 analysis**:
```bash
./venv310/Scripts/python.exe -m scripts.pilot_analysis pilot_results.json
```

**Reproduce Band B rescue analysis**:
```bash
./venv310/Scripts/python.exe -m scripts.pilot_analysis pilot_bandb_results.json
```

**Run a smoke training** (~5 min CPU):
```bash
./venv310/Scripts/python.exe -m scripts.pilot_run --smoke --output smoke_results.json
```

---

## Methodology notes

Ragnarok is developed under a **preregistration-grade protocol**. This means:

1. **All hypotheses, thresholds, and analysis choices are committed to `preregistration.md` before data is collected.** Amendments are timestamped in §13 with a full rationale, and the git history lets any reviewer verify the chronology.
2. **Multi-agent review gates.** At every milestone (pre-pilot launch, post-pilot verdict, research directions), 3–6 specialized LLM agents (RL methodology, code review, strategy, devil's advocate, architecture) review the plan independently. Dissent is logged and resolved before proceeding. See `reviews/`.
3. **Chronology audits for any post-hoc claim.** The B0 fallback plan underwent a self-initiated audit (`reviews/chronology_audit.md`) that found and corrected an integrity defect in the preregistration text — see §13 v3.6 amendment.
4. **Falsifiable kill criteria at every decision gate.** `preregistration.md` §11 lists conditions under which the project is explicitly abandoned, no redefinition.

This methodology is arguably the most valuable artifact of the project even before considering the scientific results — it is the blueprint for how solo-dev RL research can be made reviewable at the rigor level of academic preregistration.

---

## Citing

If you use this code or reference the methodology, please cite:

```bibtex
@misc{mortier2026ragnarok,
  author = {Mortier, Jérémie},
  title  = {Ragnarok: Modular RL with Skill Crystallization and Cross-Action-Space Transfer},
  year   = {2026},
  url    = {https://gitlab.com/mortier.jeremie/ragnarok}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

## Contact

Jérémie Mortier — `mortier.jeremie@gmail.com`

Independent researcher, based in France. Contract work at Stormshield (Airbus Defence and Space subsidiary). MSc in IT Engineering, Epitech.

For substantive research collaboration, technical questions on the RSSM transferable-subset design, or reviews of the methodology: email welcome.
