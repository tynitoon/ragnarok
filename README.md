# Ragnarok

**A self-teaching developmental agent: learns basic skills, reuses them to learn complex ones just as cheaply, and discovers its own curriculum.**

Ragnarok is a solo-dev research project building a *childlike developmental learner*: an agent that (1) learns basic notions, (2) **reuses** them to learn deeper notions without paying more for the extra depth (compounding), (3) **discovers on its own** what to learn next, and (4) learns from scratch when nothing transfers. The current substrate is a depth-6 crafting tech-tree (and procedurally-generated random tech-trees); recent work lifts the same ideas onto **raw pixels** (CNN perception + a learned world model).

The project began as a narrower study (cross-action-space skill transfer) which was **falsified at N=10** and honestly recorded; it then pivoted to the broader developmental program above. See *Origin* under Status below — the falsification record is preserved, not hidden.

The codebase is run with preregistration-grade methodology: every hypothesis, threshold, and analysis choice is committed to `preregistration.md` **before** data is collected, with a public chronology audit for any amendments, and **honest negatives are recorded** alongside positives.

---

## Watch it learn (runnable demo)

After install (see *Running it yourself*), drop the agent into a world it knows nothing about and watch it teach **itself** the whole tech-tree — no goals, no recipes, no curriculum given:

```bash
python -m scripts.demo            # ~5-8 min on a GPU; narrates as it learns
python -m scripts.demo --fast     # ~2-3 min, smaller but still real
python -m scripts.ragnarok --target iron_pickaxe   # watch it PLAN + BUILD, live ASCII
```

`demo.py` shows the heart of the project: with no goals given, the agent explores, notices what it can newly make, learns that skill, and — because it can now produce everything it has mastered — climbs the tree bottom-up, **choosing the order itself**, with deep skills costing no more to learn than shallow ones. `ragnarok.py` lets you ask for any target and watch it infer the recipes, plan, and craft it step by step.

**Repositories:**
- Primary (source of truth): [gitlab.com/mortier.jeremie/ragnarok](https://gitlab.com/mortier.jeremie/ragnarok)
- Mirror (auto-synced from GitLab, ~30 min delay): [github.com/tynitoon/ragnarok](https://github.com/tynitoon/ragnarok)

---

## Status

**As of 2026-05-30**, the project runs the **developmental program**. The original transfer hypothesis was falsified at N=10 on 2026-04-18 (preserved below under *Origin*); the project then pivoted to the childlike-developmental learner described at the top. The results below are validated and preregistered; honest negatives are listed alongside.

**Validated (preregistered, on GitLab — see `preregistration.md`, `ROADMAP.md`):**

| Result | What it shows | State |
|---|---|---|
| v6 / M3 — developmental loop | Reuse ⇒ **flat** per-skill learning cost vs depth; a no-reuse baseline fails past depth 2 (deep mastery 1.00 vs 0.00) — the compounding claim | ✅ |
| v7.0 — autonomous discovery (N=5) | No goals given: frontier-novelty + reuse discovers its own curriculum and masters the full 9-skill tree via a DAG-valid order, 5/5 seeds; curiosity-flat baseline stalls at depth ≤4 | ✅ |
| v9 — model-based | Learns the recipe DAG from interaction (precision/recall 1.00), plans to any target zero-shot | ✅ |
| v10 — generality | 10/10 **random** unseen tech-trees: rule recovery 1.00, plans + executes (not memorising one tree) | ✅ |
| v8 — robustness | Discovery still reaches 9/9 across world sizes (grid 13, 17 / very sparse) | ✅ |
| v12-A — perception | Learns a skill from **raw pixels** (CNN, no cell-types given), matching the symbolic skill (1.00) | ✅ |
| v12-B — world model | RSSM world model from pixels predicts the future (beats a persistence baseline, open-loop k-step) | ✅ |
| v13 — reuse from pixels | Does reuse⇒faster survive on raw pixels? | 🟡 running |

**Honest negatives (recorded, not hidden):**
- **M6/M7** end-to-end execution plateaued ~0.77 (manager under-collects resource *quantity*); parked.
- **v11** universal goal-conditioned navigation didn't train (scripted nav is 1.00, so the env is correct — it's a hard exploration problem); parked.
- **v12-C** acting via the learned *pixel* world model did **not** crack in budget — Dreamer-in-imagination was degenerate, and random-shooting planning did not reliably beat random. Perception (A) + world model (B) stand; control-from-pixels is future work.

### Origin: the original transfer hypothesis (falsified 2026-04-18, preserved)

The project began as a narrower study — cross-action-space skill transfer — that was **honestly falsified** per its preregistered kill criteria. The record is kept for transparency; it is the integrity backbone the developmental program inherited.

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

## The original research claim (transfer hypothesis — falsified, kept for context)

*The project's current claims are the developmental ones at the top of this file. The four below are the original hypothesis that was falsified at N=10 (see Origin above); they are preserved because the RSSM + skill machinery they motivated is reused throughout the developmental program.*

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
├── environments/       # Env wrappers + DeviceVecCraftWorld (crafting tech-tree)
│   ├── craft_world.py   # batched tech-tree env (symbolic OR pixel obs)
│   └── tech_tree.py     # procedural random recipe-DAG generator (v10)
└── learning/ppo_discrete.py  # batched discrete PPO (MLP + CNN actor-critic)
scripts/                # developmental program (current)
├── demo.py             # << watch the agent teach itself (narrated)
├── ragnarok.py         # << ask for a target; watch it plan + build (ASCII)
├── craft_devloop_v6.py # v6/M3 developmental loop (compounding)
├── discover_v7.py      # v7.0 autonomous discovery (no goals)
├── model_based_v9.py   # v9 learn recipe DAG + plan
├── techtree_agent_v10.py # v10 generality on random trees
├── perception_v12.py / worldmodel_v12.py / dreamer_v12.py / latent_mpc_v12.py  # pixels
├── devreuse_v13.py     # v13 reuse from pixels
└── (transfer-era: pilot_run.py, pilot_analysis.py, smoke_verdict.py)
tests/                  # test suite (pytest); run with: ./venv310/Scripts/python.exe -m pytest
preregistration.md      # Preregistered protocol + ALL amendments (developmental + origin)
ROADMAP.md              # living roadmap: done / tried / now / future
reviews/                # Multi-agent reviews, chronology audit, research directions
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

**Watch the agent teach itself** (the headline demo — GPU recommended for the full run):
```bash
python -m scripts.demo            # ~5-8 min: narrates its own discovery + compounding
python -m scripts.demo --fast     # ~2-3 min, smaller but still real
```

**Watch it plan and build any target** (live ASCII view):
```bash
python -m scripts.ragnarok --target iron_pickaxe
python -m scripts.ragnarok --target furnace
```

**Re-run the core developmental experiments** (preregistered; JSON lands in `craft_v6_out/`):
```bash
python -m scripts.discover_v7   --seeds 5   # v7.0 autonomous discovery (no goals given)
python -m scripts.perception_v12            # v12-A learn a skill from raw pixels
python -m scripts.techtree_agent_v10        # v10 generality on random tech-trees
```
Most experiment scripts accept `--smoke` for a fast sanity run. (On the tested setup substitute `./venv310/Scripts/python.exe` for `python`.)

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
