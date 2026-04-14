# Ragnarok — Environment Setup

This project uses **two Python environments**:

| Env | Python | Purpose | Where |
|-----|--------|---------|-------|
| Main | 3.14.x | Gymnasium envs (CartPole, MountainCar, Acrobot, Pendulum, MountainCarContinuous) + survival analysis (`lifelines`) + all unit tests + preregistration §12 primary H1 run | System / existing project env |
| DMC | 3.10.x | DeepMind Control Suite envs (walker-walk, cheetah-run, cartpole-swingup, hopper-stand, finger-spin) — `dm_control` / `mujoco` do not yet ship wheels for Python 3.14 | `./venv310/` |

The two envs share the **same `ragnarok` source tree** (installed with `pip install -e .`), so code changes apply to both without copy.

Preregistration §6.2 originally targeted Python 3.11 for DMC. It was substituted with **Python 3.10** (see prereg §13 v3.3) because 3.11 is not available on this workstation and 3.10 is officially supported by both `dm_control` (1.0.38) and `ragnarok` (`pyproject.toml: requires-python = ">=3.10"`). No methodology change.

---

## 1. Main env (Python 3.14)

Assumes you already have Python 3.14 with CUDA-enabled PyTorch (RTX 4080, CUDA 12.6 on this workstation).

```powershell
# From F:\dev\ragnarok
py -3.14 -m pip install -e .
py -3.14 -m pip install lifelines         # survival analysis (KM, log-rank, RMST)
py -3.14 -m pytest tests/ -x              # full unit+integration test tier
```

Sanity check:

```powershell
py -3.14 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → 2.11.0+cu126 True NVIDIA GeForce RTX 4080

py -3.14 -c "from lifelines import KaplanMeierFitter; from lifelines.statistics import logrank_test; print('OK')"
# → OK
```

## 2. DMC env (Python 3.10 in `./venv310/`)

```powershell
# From F:\dev\ragnarok
py -3.10 -m venv venv310
./venv310/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
./venv310/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
./venv310/Scripts/python.exe -m pip install -e .
./venv310/Scripts/python.exe -m pip install dm_control mujoco lifelines
```

Sanity check:

```powershell
./venv310/Scripts/python.exe -c "import ragnarok, dm_control, lifelines; from dm_control import suite; print('OK')"
# → OK

./venv310/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.11.0+cu126 True
```

## 3. Running tests

| Test tier | Env | Command | Purpose |
|-----------|-----|---------|---------|
| Unit + Gym integration | Main 3.14 | `py -3.14 -m pytest tests/ -x --ignore=tests/test_dmcontrol.py` | Fast — 100+ tests, pins all H1 code paths |
| DMC integration | venv310 | `./venv310/Scripts/python.exe -m pytest tests/test_dmcontrol.py -v` | Verifies DMC wrappers + one-episode agent run |
| Smoke benchmark | Main 3.14 | `py -3.14 -m scripts.smoke_benchmark --envs cartpole mountaincar acrobot pendulum mountaincar-continuous --seeds 3 --steps 50000 --output compute_budget.json` | ~90 min, feeds compute extrapolation for prereg §13 |

## 4. Why two envs?

- `dm_control` and `mujoco` release wheels on a lag behind CPython major versions. As of 2026-04, the latest `dm_control==1.0.38` targets 3.8-3.12 only. Python 3.14 is too new.
- The main Gymnasium envs cover H1 (preregistration primary endpoint). DMC envs only enter at H2 (secondary endpoint: continuous control transfer — see prereg §7).
- Splitting keeps the fast path (3.14) free of legacy deps and allows CUDA PyTorch to remain on the wheel cadence set by the main dev env.

## 5. Troubleshooting

**"Cannot import 'setuptools.backends._legacy'"** — legacy build-backend string; `pyproject.toml:build-backend` must be `"setuptools.build_meta"` (fixed in project).

**"Multiple top-level packages discovered"** — setuptools is picking up `logs/`, `checkpoints/`, `skills_data/`, `venv310/`. `pyproject.toml` now pins `[tool.setuptools.packages.find] include = ["ragnarok*"]` with an exclude list. If adding new top-level dirs, update the exclude list.

**venv310 falling back to CPU torch** — by default `pip install torch` picks the CPU wheel on Windows. Must pass `--index-url https://download.pytorch.org/whl/cu126` explicitly.

**DMC test collection failures in main 3.14 env** — expected; `tests/test_dmcontrol.py` gates integration via `@pytest.mark.skipif(not DMC_AVAILABLE, ...)` so the file still imports. Pass `--ignore=tests/test_dmcontrol.py` to skip the unit-tier DMC registry tests too if you don't care about them in 3.14.
