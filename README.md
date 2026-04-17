# Ragnarok

**Modular reinforcement-learning agent with skill crystallization and cross-action-space transfer.**

Ragnarok is a research project exploring whether an RL agent can crystallize a learned skill (policy + world-model subset) from one task, then transfer a latent core of that skill to a new task with a *different action space* (discrete ↔ continuous) and learn faster than training from scratch.

The codebase is a solo-dev research prototype run with preregistration-grade methodology: every hypothesis, threshold, and analysis choice is committed to `preregistration.md` **before** data is collected, with a public chronology audit for any amendments.

**Repositories:**
- Primary (source of truth): [gitlab.com/mortier.jeremie/ragnarok](https://gitlab.com/mortier.jeremie/ragnarok)
- Mirror (auto-synced from GitLab, ~30 min delay): [github.com/tynitoon/ragnarok](https://github.com/tynitoon/ragnarok)

---

## Status

**As of 2026-04-18**, the preregistered primary hypothesis has been **falsified at N=10**. The project has activated branch C of its pre-committed decision tree and pivoted from hypothesis-confirmation to a broader research program exploring three open questions (Q1/Q2/Q3, see `reviews/research_directions.md`).

| Milestone | State |
|---|---|
| Phase 1 (architecture) | ✅ Complete |
| Phase 2 (single-skill learning) | ✅ Complete |
| Phase 3 pilot #2 (3 pairs × 5 seeds) | ✅ Complete (40 runs, 12.65 GPU-hr) |
| Band B rescue (5 seeds on primary) | ✅ Complete (2026-04-17) |
| Band C N=10 extension (seeds 47–56 pooled) | ✅ Complete (2026-04-18) |
| Phase 3 analysis + decision | ✅ **Branch C activated** — workshop paper via primary pair abandoned per pre-registered kill clause |
| Phase 4+ — research program (Q1/Q2/Q3 exploration) | 🟡 Beginning |

**Primary-pair final result** (N=10 pooled, seeds 47–56, cartpole→mountaincar-continuous):

- RMST ratio (scratch/transfer) = **1.036**
- Log-rank p (one-sided) = **0.510** asymptotic / **0.516** permutation N=10,000
- Leave-one-out minimum ratio = **0.871** (dropping seed 51)
- Mechanism: 10/10 transfer runs on `latent` acting mode, 10/10 loaded a crystallized skill ✅
- Per-seed ratios: 4 positive (seeds 48, 49, 51, 54), 5 neutral (47, 50, 52, 53, 56), **1 actively anti-transfer** (seed 55, ratio 0.33 — transfer arm 3× slower than scratch)

**All three pre-registered Band C kill criteria triggered** (thresholds committed at SHA `a0c1140`, 2026-04-17, before seeds 52–56 launched):
- Ratio < 1.20 → observed 1.036 ✅ triggered
- Log-rank p ≥ 0.20 → observed 0.510 ✅ triggered
- LOO minimum < 1.00 → observed 0.871 ✅ triggered

**Scientific reading:** the specific mechanism tested — shape-checked transferable-subset loading of a Dreamer-RSSM's dynamics modules across the discrete↔continuous action-space-type boundary with the policy switched to latent mode — does not produce a reliable transfer benefit on the primary pair at N=10. Band B's N=5 signal (ratio 1.605) was high-variance seed lottery. The hypothesis is falsified on the most favorable pair in the preregistered matrix (both pendular-class, similar obs dim, action semantics close), which strengthens rather than weakens the motivation for the Q1/Q2/Q3 research program that now takes over.

For full details see `preregistration.md` §13 v3.8 (kill amendment) and `reviews/research_directions.md` §6 (branch C operational roadmap).

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

## Repository history

This Git repository was originally initialized in January 2023 for an unrelated game-development project (multiplayer C/C++ with raylib, networking, SQLite persistence). That project was archived in March 2025 and the repository remained dormant for 13 months.

On **2026-04-12**, the repository was repurposed from scratch for the Ragnarok RL research project via commit [`3cf847d`](https://gitlab.com/mortier.jeremie/ragnarok/-/commit/3cf847d) ("new projet"). The tag [`rl-project-start`](https://gitlab.com/mortier.jeremie/ragnarok/-/tags/rl-project-start) marks this pivot so reviewers can isolate the RL-era commits:

```bash
# Show only the RL research commits (April 2026 onward):
git log rl-project-start..HEAD
```

The older game-era commits are preserved unchanged for transparency — rewriting history to hide them would be inconsistent with this project's stated methodological rigor.

---

## LLM-assisted development

Ragnarok is developed using LLM-assisted workflows with Anthropic's Claude (code generation, documentation drafting, and the multi-agent reviews). This is declared openly.

**All scientific decisions** — the research question, hypothesis choices, preregistration thresholds, kill criteria, result interpretation, chronology audit initiation, and final arbitration — **are made and validated by the human author, who retains sole scientific and ethical responsibility** for the work. The multi-agent reviews are a tool for approximating peer review at solo-dev scale, not a substitute for external human peer review (which workshop submission itself will provide).

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
