# Reproducibility — What a Reviewer Can Verify in 5 Minutes

**Purpose:** enable any compute-grant reviewer, workshop reviewer, or curious peer to independently verify every empirical claim in the Ragnarok application with nothing more than a Python environment and a clone of the public repo.

---

## Setup (~2 minutes)

```bash
git clone https://gitlab.com/mortier.jeremie/ragnarok
cd ragnarok
python3.10 -m venv venv310
./venv310/Scripts/python.exe -m pip install -r requirements.txt  # Windows
# Linux/Mac: ./venv310/bin/pip install -r requirements.txt
./venv310/Scripts/python.exe -m pip install -e .
```

Python 3.10 is required because `mujoco` and `dm_control` wheels are fragile on 3.11+. The project is tested on Windows 10 + CUDA 12.6 + RTX 4080, but the analysis pipeline is CPU-only and runs on any OS.

## Claim 1 — Preregistration timestamps pre-date data

**The claim:** §8 primary threshold (RMST ratio ≥ 1.30, log-rank p < 0.10) was committed before any pilot #2 data existed.

**Reviewer verification (~30 s):**
```bash
# The §8 v3 amendment is in commit 3cf847d and descendants.
# Pilot #2 data files first appear in the repo at a later commit.
git log --all --oneline -- preregistration.md | head -20
git log --all --oneline -- pilot_results.json | head -10
# The prereg v3 commits pre-date the pilot_results.json commits by days.
```

For a deeper audit: `reviews/chronology_audit.md` reconstructs the full pilot #2 timeline from `pilot_run.log` wall-clock fields and verifies the chronology commit-by-commit.

## Claim 2 — Band B rescue RMST ratio 1.605, p=0.259, LOO min 1.435

**The claim:** the headline empirical result in the proposal.

**Reviewer verification (~1 minute):**
```bash
./venv310/Scripts/python.exe -m scripts.pilot_analysis pilot_bandb_results.json
```

Expected output (printed verbatim in the proposal):
```
cartpole_mcc  (role=primary, τ=200,000 env-steps)
  scratch : n= 5  events= 5/5   RMST=   18,047
  transfer: n= 5  events= 5/5   RMST=   11,241
  RMST ratio (scratch/transfer): 1.605
  Log-rank p (one-sided): 0.2402
  Log-rank p (permutation, N=10,000): 0.2585
  Status: PRIMARY FAIL  |  mechanism OK
  Mechanism: 5/5 transfer runs on 'latent' mode; 5/5 loaded a skill
```

LOO minimum is computed on-the-fly from the per-seed data in the JSON:
```python
import json, numpy as np
runs = json.load(open('pilot_bandb_results.json'))['runs']
by_seed = {}
for r in runs:
    if r.get('pair_alias') != 'cartpole_mcc': continue
    by_seed.setdefault(r['seed'], {})[r['arm']] = r.get('steps_to_mastery')
seeds = sorted(by_seed.keys())
all_s = np.array([by_seed[s]['scratch'] for s in seeds])
all_t = np.array([by_seed[s]['transfer'] for s in seeds])
for i, seed in enumerate(seeds):
    s = np.concatenate([all_s[:i], all_s[i+1:]])
    t = np.concatenate([all_t[:i], all_t[i+1:]])
    print(f"Drop seed={seed}: N=4 ratio = {s.mean()/t.mean():.3f}")
# Min should be 1.435 (when dropping seed 49).
```

## Claim 3 — Mechanism check: 5/5 transfer runs on latent mode with skill loaded

**The claim:** the mechanism filter preregistered in §8 passed.

**Reviewer verification (~30 s):**
```python
import json
runs = json.load(open('pilot_bandb_results.json'))['runs']
transfer_runs = [r for r in runs if r['arm'] == 'transfer' and r.get('pair_alias') == 'cartpole_mcc']
for r in transfer_runs:
    print(f"seed={r['seed']}  mode={r.get('acting_policy_mode')}  skill={r.get('transfer_skill_name')}")
```

All 5 rows must show `mode=latent` and a non-null `skill` value (e.g., `CartPole-v1_320ep`). Any deviation would invalidate the §8 mechanism check.

## Claim 4 — 444 tests passing

**The claim:** the test suite is not theater.

**Reviewer verification (~3 minutes):**
```bash
./venv310/Scripts/python.exe -m pytest tests/ -x --tb=short
```

Expected output ends with `... passed in NNNs`. If any test fails on a clean clone at HEAD, the reviewer is encouraged to open an issue — this would be a project-damaging defect worth surfacing.

## Claim 5 — Preregistration contains N amendments with commit SHAs

**The claim:** every amendment is timestamped and anchored.

**Reviewer verification (~20 s):**
```bash
grep -E "^### v3\.|^- \*\*2026-04" preregistration.md
```

The output lists every amendment with its date. Cross-reference with `git log preregistration.md` to verify each amendment date matches a commit timestamp.

## Claim 6 — Seed-level data contains full provenance

**The claim:** every run records git SHA, Python version, PyTorch version, CUDA version, GPU model, hostname.

**Reviewer verification (~10 s):**
```python
import json
print(json.dumps(json.load(open('pilot_results.json'))['provenance'], indent=2))
```

## Ongoing commitments

1. **Any future experiment's seed-level JSON is committed to the public repo** within 48 h of run completion.
2. **Any preregistration amendment is committed before the data it affects is collected** (the one exception, the v3.5 → v3.6 B0 chronology correction, is documented in `reviews/chronology_audit.md`).
3. **Any compute-grant monthly report** is published in `docs/compute_application/reports/month_NN_report.md` with the same transparency level as this document.
4. **Reproducibility is a first-class engineering concern**, not a post-hoc add-on. The POST-006 item in `reviews/post_pilot_backlog.md` commits to producing `scripts/reproduce_headline.py` before workshop submission if the paper is written.

## What this document does *not* promise

- Bit-identical reproduction across different hardware or library versions. RL results are sensitive to seed scheduling in CUDA, and the project does not freeze a container for exact reproducibility — that is a stretch goal for a later phase.
- Reproduction of the workshop paper figures prior to the paper being written. Currently the analysis script (`pilot_analysis.py`) produces tabular output; figure code will be committed alongside the paper draft.

---

*This document is a public, commit-anchored verification contract. Any of the claims above can be challenged by opening an issue on the GitLab repository; verification failures are project-damaging defects and will be treated as such.*
