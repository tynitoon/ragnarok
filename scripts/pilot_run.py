"""Phase 3 pilot experiment runner (preregistration §8).

Runs the small-N rehearsal of the H1 headline test:

  5 seeds x 3 source->target pairs x {scratch, transfer} = 30 target-env runs
  + 15 source pre-training runs (one per pair/seed) to crystallize skills

| Pair                                  | Role                            |
|---------------------------------------|---------------------------------|
| CartPole -> MountainCarContinuous     | Primary-endpoint rehearsal      |
| CartPole -> Acrobot                   | Secondary-endpoint rehearsal    |
| Pendulum -> DMC-cartpole-swingup      | Secondary-endpoint rehearsal    |

Per run: up to `--max-steps` env-steps (default 200k per §8); eval every
`--eval-every` env-steps (default 5000 per §4.5); 10 eval episodes per
checkpoint.

Output JSON schema (see PilotRun.to_dict) carries everything needed for
downstream RMST analysis via `scripts/pilot_analysis.py`:
  - eval_curve: [{"step": int, "return": float}, ...]
  - steps_to_mastery: int | None  (None = censored at truncation horizon)
  - mastery_threshold: float  (env.spec default or --mastery-thresholds file)
  - acting_policy_mode: "obs" | "latent"  (mechanism check, §8)
  - transfer_skill_name: str | None
  - source_crystallized: bool  (source runs only; lets the analyzer flag
    transfer arms whose source never produced a real skill)

Top-level JSON also carries `provenance` (git SHA, ragnarok version,
python/torch/cuda/gpu/lifelines, hostname) so a reviewer can replay.

Pass criteria are checked offline; this script only produces the data.

**τ sensitivity scope (§4.6)**: the pilot's truncation horizon is
`--max-steps`. Downstream sensitivity sweeps at τ' <= max_env_steps are
supported from the eval_curve. Upward sweeps (τ' > max_env_steps) are
*not* supported by pilot data — training stops at max_env_steps. Phase 5
headline runs (which go to 500k) are where the full §4.6 sweep
{300k, 500k, 750k, 1M} applies.

Usage:
    # Main env (Python 3.14) — pairs 1 & 2 only (no DMC)
    python -m scripts.pilot_run --pairs cartpole_mcc cartpole_acrobot

    # venv310 (Python 3.10) — all 3 pairs including DMC
    venv310/Scripts/python -m scripts.pilot_run

    # Quick smoke (2 seeds, 40k steps, telemetry enforceable per prereg v3)
    python -m scripts.pilot_run --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Force UTF-8 stdio so §/τ/≥/— in progress-print strings don't crash on
# Windows cp1252 (preflight smoke crashed exactly this way on first run).
# The preregistration document uses these glyphs in §8 language; keeping
# the code aligned beats scrubbing every print site — `.reconfigure()`
# is a standard Python ≥3.7 idiom and no-ops cleanly if stdio was already
# UTF-8 (e.g. linux terminal, piped output with PYTHONIOENCODING set).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

from ragnarok.infrastructure.config import RagnarokConfig
from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.registry import get_env_spec, make_env
from ragnarok.core.agent import RagnarokAgent


# ── Pilot matrix ────────────────────────────────────────────────────

# Pre-declared per preregistration §8. Deviation requires a §13 amendment.
PILOT_PAIRS = [
    # (alias, source env, target env, role)
    ("cartpole_mcc", "cartpole", "mountaincar-continuous", "primary"),
    ("cartpole_acrobot", "cartpole", "acrobot", "secondary"),
    ("pendulum_dmc_cartpole", "pendulum", "cartpole-swingup", "secondary"),
]

# §7 A10 adversarial-negative pair (v3.5 amendment). Same action-type
# change as the primary (Discrete → Box) but non-pendular physics class
# on the target (DMC finger-spin is rotational-forced; no gravity well).
# Predicted under H1: no transfer or anti-transfer. Run separately via
# `--run-adversarial` so it doesn't bloat the main pilot_results.json;
# consumed by the paper's headline table regardless of direction.
#
# Not in `PILOT_PAIRS` because:
#   (a) different default output path (`pilot_adversarial_results.json`),
#   (b) different DEFAULT_N (5 adversarial seeds, same as pilot seeds),
#   (c) DMC target needs the venv310 env.
ADVERSARIAL_PAIRS = [
    # (alias, source env, target env, role)
    ("cartpole_fingerspin", "cartpole", "finger-spin", "adversarial"),
]

# §7 A11 GRU-shuffled ablation (v3.5 amendment). Runs the primary pair
# with a modified transfer path: after `try_transfer()` loads the
# transferable RSSM subset, the GRU weight tensors are row-column
# permuted (preserves singular-value spectrum and Frobenius norm, destroys
# learned temporal structure). If A11 ≈ real transfer, the "learned GRU
# dynamics transfer" claim dies.
#
# Supported ablations (map of name -> help text for --ablation):
SUPPORTED_ABLATIONS = {
    "none": "No ablation (stock transfer); default.",
    "shuffled-gru": (
        "A11: row-column-permute transferable GRU weights after "
        "try_transfer(). Preserves spectral norm and parameter count, "
        "destroys learned recurrent structure. 2 seeds on cartpole_mcc "
        "per prereg v3.5 §7."
    ),
}

PILOT_SEEDS_DEFAULT = 5           # §8: 5 seeds × 3 pairs × 2 arms = 30
MAX_ENV_STEPS_DEFAULT = 200_000   # §8: 200k env-steps per run
EVAL_EVERY_STEPS_DEFAULT = 5_000  # §4.5: eval every 5000 env-steps
EVAL_EPISODES_DEFAULT = 10        # §4.5: 10 deterministic eval episodes

# Source pre-training cap. Source only needs to be "good enough" to produce a
# transferable skill; we don't need to exhaust the same 200k budget there.
# Cartpole @ ~144 steps/s -> 150k steps = ~17 min; pendulum (SAC) slower.
#
# v5.2 (post devil's-advocate v5 review): bumped 100k -> 150k to add ~1σ
# safety margin on top of the observed cartpole crystallization
# distribution (mean ≈ 44k, std ≈ 14k from n=3 prior runs at seeds
# 42/43/44 — mean+4σ ≈ 100k, i.e. the previous default left near-zero
# headroom). Pilot #2 introduces seeds 44/45/46 that have NOT been
# observed on this codepath; if any of them sits in the heavy right
# tail of cartpole's training-time distribution, hitting the cap
# without crystallizing degrades that pair's transfer arm to scratch
# (one wasted data point out of 30). The 150k cap eats ~50% more
# wall-clock (negligible vs the 200k pilot-arm budget) in exchange
# for a more robust mean+5σ safety margin.
#
# `--smoke` overrides this default to 100k (see args.source_max_steps
# in main()) — smoke validates seeds 42/43 which we've already observed
# crystallize at <60k, so the smoke cap doesn't need the extra margin
# and we save ~2 min × 2 seeds in smoke iteration time.
SOURCE_MAX_ENV_STEPS_DEFAULT = 150_000


# Sentinel string written to `kl_probe_error` when the replay buffer is empty
# (no episodes yet). Public constant so `scripts/smoke_verdict.py` can import
# it instead of duplicating the literal — v5 architecture review caught the
# brittle silent-drift coupling between producer and consumer.
TELEMETRY_BUFFER_EMPTY_SENTINEL = "buffer empty (num_episodes=0)"


# ── Result dataclass ────────────────────────────────────────────────

@dataclass
class EvalPoint:
    step: int
    eval_return: float


@dataclass
class PilotRun:
    """One pilot run: (pair, seed, arm). Serializes to JSON."""
    pair_alias: str
    pair_role: str                    # "primary" | "secondary"
    src_env: str
    tgt_env: str
    seed: int
    arm: str                          # "scratch" | "transfer"
    mastery_threshold: float
    max_env_steps: int
    total_env_steps: int
    total_episodes: int
    final_eval_return: float
    best_eval_return: float
    steps_to_mastery: int | None      # None == right-censored at max_env_steps
    eval_curve: list[EvalPoint] = field(default_factory=list)
    acting_policy_mode: str = "obs"   # Must be "latent" for cross-dim transfer
    transfer_skill_name: str | None = None
    wall_clock_sec: float = 0.0
    # Source-only flag: True iff the source pre-training produced a
    # crystallized skill. Transfer arms whose (pair, seed) source did NOT
    # crystallize should be treated as "scratch-masquerading-as-transfer"
    # by the analyzer — a silent degradation that bit us in review.
    source_crystallized: bool | None = None
    # Vec flag actually exercised (may differ from --vec requested because
    # vectorized collection is only supported for discrete A2C paths).
    used_vec: bool = False
    # Per-checkpoint diagnostic series for the transfer arm only (Bug E v3,
    # 2026-04-15, devil's-advocate review #2 BLOCKER; v4 adds kl_probe_error
    # per testing review MINOR). Each entry:
    #   {"step": int, "episode": int,
    #    "transferable_drift_max":         float in [0, ∞),
    #    "transferable_drift_per_param":  {param_name: float},
    #    "kl_posterior_prior":             float | None,
    #    "kl_probe_error":                 str | None}
    # The prereg amendment "Bug E v2" commits to logging these for the
    # smoke pre-check abort criterion (||Δθ|| > 50% by ep 100 → abort).
    # Without a side-car series in the run output, the prereg commitment
    # is unenforceable. Empty list for scratch / source / non-transfer arms
    # and for transfer arms where try_transfer returned None.
    telemetry: list[dict] = field(default_factory=list)
    # v3.5 §7 A11: ablation tag + permutation metadata. For runs with
    # `ablation == "none"` this is the stable default; when an A11
    # shuffled-gru run is executed, `ablation_info` records the
    # permutation sizes applied so the paper can verify the ablation
    # was structurally complete and not a partial shuffle.
    ablation: str = "none"
    ablation_info: dict | None = None

    def to_dict(self) -> dict:
        return {
            "pair_alias": self.pair_alias,
            "pair_role": self.pair_role,
            "src_env": self.src_env,
            "tgt_env": self.tgt_env,
            "seed": self.seed,
            "arm": self.arm,
            "mastery_threshold": self.mastery_threshold,
            "max_env_steps": self.max_env_steps,
            "total_env_steps": self.total_env_steps,
            "total_episodes": self.total_episodes,
            "final_eval_return": self.final_eval_return,
            "best_eval_return": self.best_eval_return,
            "steps_to_mastery": self.steps_to_mastery,
            "censored": self.steps_to_mastery is None,
            "eval_curve": [asdict(p) for p in self.eval_curve],
            "acting_policy_mode": self.acting_policy_mode,
            "transfer_skill_name": self.transfer_skill_name,
            "source_crystallized": self.source_crystallized,
            "used_vec": self.used_vec,
            "wall_clock_sec": self.wall_clock_sec,
            "telemetry": self.telemetry,
            # v3.5 §7 A11 ablation tag. "none" for all pre-v3.5 runs (the
            # reader defaults this on deserialize so old pilot_results.json
            # still loads cleanly). `ablation_info` carries the permutation
            # sizes when shuffled-gru is applied, and is None otherwise.
            "ablation": self.ablation,
            "ablation_info": self.ablation_info,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    """Seed torch (cpu+cuda), numpy, AND stdlib random.

    Prior version missed stdlib `random` and `torch.cuda.manual_seed_all`,
    which let residual RNG state leak across seeds (devil's-advocate
    review, Phase 3 pre-commit audit). For the pilot's 5-seed N the
    difference is small but reviewers will flag it.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic write so a Ctrl-C mid-flush cannot corrupt pilot_results.json.

    Without this, a crash during the 8-hour pilot corrupts the output and
    the resume logic's `except Exception` fall-through silently starts
    fresh — wiping up to 8 hours of completed work. Devil's-advocate
    review flagged this as the single highest-severity risk.

    Strategy: write to <path>.tmp, fsync, then os.replace (atomic on
    both POSIX and Windows per Python docs). Keep a single .bak of the
    last good file so even a bizarre two-way corruption leaves one
    recoverable snapshot.

    fsync() upgrades the guarantee from "Ctrl-C safe" to "power-loss
    safe": without it, the .tmp replace can land empty content if the
    machine loses power between write() and replace() (the write was
    buffered, not on disk).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            if bak.exists():
                bak.unlink()
            os.replace(path, bak)
        except OSError:
            # Non-fatal: the new file still lands via the replace below.
            pass
    os.replace(tmp, path)


def _collect_provenance() -> dict:
    """Collect reviewer-replay metadata. Every field is best-effort —
    missing values (e.g. no git, no GPU) become nulls rather than crashing
    the pilot launch."""
    prov: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "device": str(DEVICE),
    }
    try:
        import lifelines
        prov["lifelines"] = lifelines.__version__
    except Exception:
        prov["lifelines"] = None
    if torch.cuda.is_available():
        prov["gpu_name"] = torch.cuda.get_device_name(0)
        prov["cuda_capability"] = str(torch.cuda.get_device_capability(0))
        prov["cuda_runtime"] = getattr(torch.version, "cuda", None)
    else:
        prov["gpu_name"] = None

    # Git SHA — only read, no writes.
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        prov["git_sha"] = sha.stdout.strip() if sha.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        prov["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        prov["git_sha"] = None
        prov["git_dirty"] = None
    return prov


def _build_agent(env_name: str, seed: int, skills_dir: str) -> tuple[RagnarokAgent, object]:
    """Build an agent + env pair for training. Returns (agent, env)."""
    spec = get_env_spec(env_name)
    config = RagnarokConfig(seed=seed, checkpoint_dir="checkpoints")
    config.skill.skills_dir = skills_dir
    config.world_model.obs_dim = spec.obs_dim
    config.world_model.action_dim = spec.action_dim
    # Pilot runs must reflect the default (benchmark-clean) code path —
    # no env_overrides, no reward shaping (preregistration §6.1 fix #3).
    config.reward_shaping.enabled = False
    config.env_overrides.enabled = False

    env = make_env(env_name, seed=seed)
    agent = RagnarokAgent(config, env)

    # Devil's-advocate review (Phase 3 pre-launch, smoke #2):
    # SkillSelector's distance-threshold gate (default 50.0 in
    # skills/selector.py) is designed for continual-learning use
    # cases where "is this skill relevant enough?" is an open
    # question. Pilot pairs PIN source→target explicitly; every
    # transfer arm is *meant* to attempt transfer. The default gate
    # silently rejects the legitimate skill because source centroids
    # (trained-RSSM latent) and target-env warmup encodings
    # (fresh-RSSM latent) live in uncorrelated latent spaces, so the
    # distance is essentially noise. Without this override, smoke #2
    # saw BOTH transfer arms load `null` skill → acting_policy_mode
    # stayed "obs" → the §8 mechanism gate would trivially fail for
    # every transfer run. Bypass the gate for pilot arms.
    agent.skill_selector.distance_threshold = float("inf")

    return agent, env


def _evaluate(agent: RagnarokAgent, env, episodes: int) -> float:
    """Unified eval dispatch (SAC vs A2C vs pixel)."""
    if agent.pixel_ppo is not None:
        return agent._evaluate_pixel(episodes=episodes)
    if agent.sac_trainer is not None:
        return agent.sac_trainer.evaluate(env, episodes=episodes)
    return agent.real_trainer.evaluate(env, episodes=episodes)


def _has_vec_path(agent: RagnarokAgent, spec) -> bool:
    """Vectorized collection only supported for discrete, vector-obs A2C."""
    return (not spec.pixel_obs) and (agent.sac_trainer is None)


def _compute_transfer_telemetry(
    agent: RagnarokAgent,
    baseline: dict[str, torch.Tensor] | None,
    baseline_norms: dict[str, float] | None,
) -> dict | None:
    """Compute one telemetry record for a transfer-arm pilot run.

    Module-level (not a closure inside `_train_to_step_budget`) so the
    function is unit-testable in isolation — devil's-advocate v3 testing
    review pointed out that a closure-only implementation has zero
    regression coverage and the very BLOCKER #1 fix it implements
    (telemetry actually being emitted) could silently regress without
    any unit-test signal.

    Args:
        agent: live RagnarokAgent. We read
            `agent.wm_trainer.rssm.transferable_state_dict()` and probe
            KL on a small batch from `agent.replay_buffer`.
        baseline: snapshot of the transferable state_dict captured
            immediately after `agent.try_transfer()` returned a non-None
            skill. Tensors must already be on CPU. None for non-transfer
            arms or transfer arms where try_transfer returned None.
        baseline_norms: per-key Frobenius norm of `baseline` values
            (precomputed once at snapshot time so we don't re-compute on
            every checkpoint).

    Returns:
        dict | None: telemetry record with keys
            {step, episode, transferable_drift_max,
             transferable_drift_per_param, kl_posterior_prior,
             kl_probe_error},
        or None if `baseline` is None (signals "no telemetry for this
        run", consumed by the caller and skipped from the series).
        `kl_probe_error` is None on success or a short repr of the
        exception when the KL probe failed (silent-swallow protection
        per testing review v3 MINOR — a probe that always returns None
        without diagnostics is indistinguishable from a probe that
        works on data we never collect).
    """
    if baseline is None or baseline_norms is None:
        return None
    sd = agent.wm_trainer.rssm.transferable_state_dict()
    drift_per_param: dict[str, float] = {}
    drift_max = 0.0
    for k, v0 in baseline.items():
        v_now = sd[k].detach().cpu()
        denom = baseline_norms[k] or 1.0
        d = float((v_now - v0).norm().item()) / denom
        drift_per_param[k] = d
        drift_max = max(drift_max, d)

    # KL(posterior‖prior) probe — single small batch from replay buffer,
    # no_grad, ~few-ms cost. Reports whether the loaded prior is actually
    # being used or has been crushed by the posterior.
    #
    # CRITICAL (architecture review v3, 2026-04-15, MAJOR concern):
    # this CANNOT use `rssm.loss(...)["kl_loss"]` — that field is
    # `clamp(min=free_nats/stoch_dim).sum(-1).mean()`, i.e. the
    # free-nats-CLAMPED training objective, floored at 1.0/stoch_dim
    # per latent dim. Default `free_nats=1.0` plus stoch_dim ∈ {8,16,32}
    # means the floor IS the expected value early in training — so the
    # probe would be *unable* to detect the very failure mode the comment
    # claims to detect (prior crushed by posterior). We compute raw KL
    # directly via `observe()` + `kl_divergence(Normal, Normal)` with NO
    # clamping. This gives the actual KL(post‖prior) per step, summed
    # over latent dim, averaged over batch & time — the diagnostic
    # signal we need.
    kl_probe: float | None = None
    kl_probe_error: str | None = None
    try:
        buf = agent.replay_buffer
        if buf.num_episodes >= 1:
            trainer = agent.wm_trainer
            bs = min(8, trainer.batch_size)
            sl = min(50, trainer.seq_length)
            obs, actions, rewards, dones = buf.sample_sequences(bs, sl)
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=DEVICE)
                act_t = torch.tensor(actions, device=DEVICE)
                outputs = trainer.rssm.observe(obs_t, act_t)
                from torch.distributions import Normal, kl_divergence
                post = Normal(outputs["post_mean"],
                              outputs["post_logstd"].exp())
                prior = Normal(outputs["prior_mean"],
                               outputs["prior_logstd"].exp())
                # Raw KL — NO clamp, NO weight. Sum over latent dim,
                # mean over (batch, time).
                raw_kl = kl_divergence(post, prior).sum(dim=-1).mean()
                kl_probe = float(raw_kl.item())
        else:
            # Empty buffer — not an error, just no data yet. Distinguish
            # from a real failure so the analyzer can tell apart "probe
            # never ran" from "probe ran and crashed".
            kl_probe_error = TELEMETRY_BUFFER_EMPTY_SENTINEL
    except Exception as e:
        # Telemetry must never break training — buffer might be
        # mid-mutation, batch might be malformed in edge cases. Drop
        # to None and continue; the analyzer treats None as "not
        # measured at this checkpoint" rather than as a real signal.
        # NOTE: silently swallowing exceptions here is a calculated
        # trade-off — we never want telemetry to crash a 20-GPU-hour
        # pilot. The unit test in test_pilot_telemetry.py catches the
        # common cases at PR time so we don't rely on smoke runs to
        # detect telemetry bugs.
        #
        # Capture the repr (truncated) so post-hoc debugging knows WHY
        # the probe was None — testing review v3 MINOR: a permanently-
        # silent probe is indistinguishable from a working probe that
        # never gets called.
        kl_probe = None
        kl_probe_error = repr(e)[:200]

    return {
        "step": agent.total_steps,
        "episode": agent.total_episodes,
        "transferable_drift_max": drift_max,
        "transferable_drift_per_param": drift_per_param,
        "kl_posterior_prior": kl_probe,
        "kl_probe_error": kl_probe_error,
    }


def _apply_gru_shuffle(rssm, rng_seed: int) -> dict[str, int]:
    """§7 A11: row-column-permute the GRU weight tensors in the RSSM
    transferable subset, in-place.

    PyTorch `nn.GRU` layout (single layer, no bidir):
      - `weight_ih_l0` : (3 * hidden_size, input_size)
      - `weight_hh_l0` : (3 * hidden_size, hidden_size)
      - `bias_ih_l0`   : (3 * hidden_size,)
      - `bias_hh_l0`   : (3 * hidden_size,)

    The 3-way row split corresponds to the [reset, update, new] gates.
    We use the SAME row permutation for all 4 tensors so the gate
    association is preserved (permuting reset/update/new independently
    would inject a different mutilation — the A11 claim is "destroy the
    learned structure *within* each gate", not "scramble gate roles").
    We use DIFFERENT column permutations for weight_ih and weight_hh
    because their columns index disjoint spaces (input vs. hidden).

    Preserved:
      - Frobenius norm (row-column perm is an orthogonal transformation
        composed on each side)
      - Full singular-value spectrum (same reason)
      - Total parameter count
      - Per-tensor weight-magnitude distribution (just reshuffled)
    Destroyed:
      - Learned correlations between specific input dims and gate
        activations
      - Learned recurrent patterns (the column permutation on weight_hh
        breaks the identity-mapping-diagonal structure Xavier init
        would have given)

    Returns a dict with the permutation sizes applied, for the record
    payload.
    """
    # gru may live in different attribute paths depending on RSSMCore
    # implementation. Probe common ones.
    gru = None
    for path in ("core.gru", "gru"):
        obj = rssm
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            gru = obj
            break
    if gru is None:
        raise RuntimeError(
            "A11 shuffle: could not locate GRU on RSSM "
            "(probed core.gru, gru)"
        )
    if not hasattr(gru, "weight_ih_l0") or not hasattr(gru, "weight_hh_l0"):
        raise RuntimeError(
            "A11 shuffle: GRU does not expose "
            "weight_ih_l0/weight_hh_l0 — is it a multi-layer or "
            "bidirectional GRU? Only single-layer unidir supported."
        )

    # Deterministic: derive from the caller's seed so two runs of the
    # same pilot with the same seed produce the same shuffle.
    g = torch.Generator(device="cpu").manual_seed(rng_seed)

    with torch.no_grad():
        w_ih = gru.weight_ih_l0
        w_hh = gru.weight_hh_l0
        rows = w_ih.shape[0]            # 3 * hidden_size
        in_dim = w_ih.shape[1]          # input_size
        hidden_size = w_hh.shape[1]

        row_perm = torch.randperm(rows, generator=g)
        col_ih_perm = torch.randperm(in_dim, generator=g)
        col_hh_perm = torch.randperm(hidden_size, generator=g)

        # Apply permutations in-place via index_copy through a fresh
        # tensor (can't do it fully in-place safely because index_copy
        # on a self-overlapping view is UB).
        new_w_ih = w_ih.detach().clone()
        new_w_ih = new_w_ih[row_perm][:, col_ih_perm]
        w_ih.copy_(new_w_ih)

        new_w_hh = w_hh.detach().clone()
        new_w_hh = new_w_hh[row_perm][:, col_hh_perm]
        w_hh.copy_(new_w_hh)

        # Biases: apply the same row permutation.
        for bname in ("bias_ih_l0", "bias_hh_l0"):
            if hasattr(gru, bname):
                b = getattr(gru, bname)
                new_b = b.detach().clone()[row_perm]
                b.copy_(new_b)

    return {
        "rows_permuted": int(rows),
        "ih_cols_permuted": int(in_dim),
        "hh_cols_permuted": int(hidden_size),
    }


def _train_to_step_budget(
    env_name: str,
    seed: int,
    arm: str,
    skills_dir: str,
    max_env_steps: int,
    eval_every_steps: int,
    eval_episodes: int,
    mastery_threshold: float,
    pair_alias: str | None = None,
    pair_role: str | None = None,
    src_env: str | None = None,
    num_envs: int = 1,
    ablation: str = "none",
) -> PilotRun:
    """Train one (env, seed, arm) combination to a step budget; return a PilotRun."""
    _seed_everything(seed)
    spec = get_env_spec(env_name)
    agent, env = _build_agent(env_name, seed, skills_dir)

    # Vec env path for discrete A2C; skipped for SAC and pixel envs.
    vec_env = None
    use_vec = (num_envs > 1) and _has_vec_path(agent, spec)
    if use_vec:
        from ragnarok.environments.vec_wrapper import VecRagnarokEnv
        vec_env = VecRagnarokEnv(
            spec.gym_name, num_envs=num_envs, seed=seed,
            normalizer=env.normalizer, normalize=env.normalize,
        )

    # Transfer attempt (runs before the first training iter so it can alter
    # acting_policy_mode that propagates into the very first action).
    transfer_skill_name: str | None = None
    transferable_baseline: dict | None = None  # {param_name: cpu tensor}
    transferable_baseline_norms: dict | None = None  # {param_name: float}
    ablation_info: dict | None = None  # Non-None only if A11 shuffle applied.
    if arm == "transfer":
        loaded = agent.try_transfer()
        if loaded is not None:
            transfer_skill_name = loaded.name

            # §7 A11 GRU-shuffle ablation (v3.5, post-load mutation).
            # Apply BEFORE capturing the baseline so ||Δθ|| on the
            # transferable params is measured relative to the shuffled
            # init — otherwise the first delta would measure "shuffle
            # distance" and mask actual training drift. Only applies to
            # transfer arms with a loaded skill; scratch arms and
            # no-skill transfers silently pass through with ablation="none".
            if ablation == "shuffled-gru":
                try:
                    ablation_info = _apply_gru_shuffle(
                        agent.wm_trainer.rssm, rng_seed=seed,
                    )
                    # Annotate the skill name so the analyzer can
                    # distinguish shuffled from real transfer at a
                    # glance. Suffix style: "<name>__ablation=shuffled-gru".
                    transfer_skill_name = (
                        f"{loaded.name}__ablation=shuffled-gru"
                    )
                    print(
                        f"  [ablation=shuffled-gru] GRU permuted: "
                        f"{ablation_info}",
                        flush=True,
                    )
                except Exception as e:
                    # Shuffle failure must not silently pass — the whole
                    # point of A11 is to compare to real transfer. A
                    # failed shuffle would produce a result indistinguishable
                    # from real transfer and invalidate the ablation.
                    raise RuntimeError(
                        f"A11 shuffle failed (seed={seed}, "
                        f"env={env_name}): {e!r}. Abort run; the "
                        f"ablation would be invalid with a partial "
                        f"or skipped shuffle."
                    ) from e
            elif ablation != "none":
                raise ValueError(
                    f"Unknown ablation {ablation!r}; supported: "
                    f"{list(SUPPORTED_ABLATIONS.keys())}"
                )

            # Bug E v3 (2026-04-15, devil's-advocate review #2 BLOCKER):
            # snapshot transferable subset RIGHT AFTER load so the smoke can
            # measure ||θ_t - θ_loaded|| over training. The prereg amendment
            # commits to "abort if drift > 50% by ep 100"; without a baseline
            # captured here, that criterion is unenforceable. Cloning to CPU
            # avoids holding a duplicate of the GPU weights for a 200k-step
            # run — this is single-shot, ~100 KB total.
            # v3.5 note: when A11 is active, the baseline reflects the
            # shuffled state, so drift measures "shuffled→final" not
            # "real-init→final". That's the honest baseline for A11.
            try:
                _baseline_sd = agent.wm_trainer.rssm.transferable_state_dict()
                transferable_baseline = {
                    k: v.detach().clone().cpu()
                    for k, v in _baseline_sd.items()
                }
                transferable_baseline_norms = {
                    k: float(v.norm().item())
                    for k, v in transferable_baseline.items()
                }
            except Exception as e:
                # Defensive: if rssm/transferable_state_dict ever changes
                # signature, telemetry should degrade to empty rather than
                # crashing the pilot. Real failures will surface in tests.
                print(f"  [WARN] telemetry baseline capture failed: {e}",
                      flush=True)
                transferable_baseline = None
                transferable_baseline_norms = None

            # Devil's-advocate review (Phase 3 pre-launch): the §8 mechanism
            # gate requires acting_policy_mode == "latent" for every
            # cross-dim transfer run. agent.try_transfer() is expected to
            # flip the mode when a latent-trunk load happens, but nothing
            # in the pilot path verifies it. A silent regression in the
            # SAC rewrite (Phase 2.3) could leave the mode on "obs" and
            # turn every transfer run into a trivial mechanism failure at
            # analysis time — detecting it 8h later instead of immediately.
            # Assert the invariant here; fail the run loudly if it breaks.
            if src_env is not None and _cross_dim(src_env, env_name):
                if agent.acting_policy_mode != "latent":
                    raise RuntimeError(
                        f"Phase 3 mechanism regression: cross-dim transfer "
                        f"({src_env} -> {env_name}, seed={seed}) loaded skill "
                        f"{loaded.name!r} but acting_policy_mode="
                        f"{agent.acting_policy_mode!r} (expected 'latent'). "
                        f"This is a silent correctness bug -- check "
                        f"agent.try_transfer / latent_policy trunk load."
                    )

            if vec_env is not None:
                vec_env.normalizer = env.normalizer
                for v in vec_env.envs:
                    v.normalizer = env.normalizer
        else:
            # D1 failure mode (devil's-advocate review): for cross-dim pairs
            # the transfer arm is supposed to load a latent-trunk skill. If
            # `try_transfer()` returns None here, the "transfer" arm is
            # silently degraded to "scratch-on-a-hot-library" — which would
            # bias the RMST ratio toward 1.0 and mask true transfer effects.
            # We surface it loudly so the operator sees it in real time AND
            # the downstream analyzer can flag it from transfer_skill_name.
            if src_env is not None and _cross_dim(src_env, env_name):
                print(
                    f"  [WARN] cross-dim transfer arm "
                    f"({src_env} -> {env_name}, seed={seed}) loaded NO skill. "
                    f"Check source crystallization for this (pair, seed).",
                    flush=True,
                )

    # Main training loop — step-budget gated.
    eval_curve: list[EvalPoint] = []
    telemetry: list[dict] = []
    steps_to_mastery: int | None = None
    last_eval_step = 0
    best_eval = -float("inf")
    iteration = 0
    drift_alert_emitted = False  # Print the >50% drift alert at most once

    def _capture_telemetry() -> dict | None:
        """Thin wrapper: delegate to module-level _compute_transfer_telemetry.

        Kept as a closure so the eval-checkpoint call site stays terse;
        the actual logic lives at module level so it's unit-testable
        (Bug E v4, testing review v3 MAJOR concern #5).
        """
        return _compute_transfer_telemetry(
            agent, transferable_baseline, transferable_baseline_norms)

    start_time = time.time()
    while agent.total_steps < max_env_steps:
        iteration += 1
        # 1. Policy + replay collection
        if use_vec:
            agent.train_policy_real_vec(vec_env)
        else:
            agent.train_policy_real()

        # 2. World model training — same cadence as smoke_benchmark so the
        #    per-env throughput projection transfers cleanly.
        if (iteration % 10 == 0
                and agent.replay_buffer.num_episodes >= 10):
            agent.train_world_model(steps=2)

        # 3. Eval checkpoint — first time we cross an `eval_every_steps`
        #    boundary AND at the final step.
        if (agent.total_steps - last_eval_step) >= eval_every_steps:
            eval_r = _evaluate(agent, env, episodes=eval_episodes)
            eval_curve.append(EvalPoint(step=agent.total_steps,
                                        eval_return=eval_r))
            last_eval_step = agent.total_steps
            best_eval = max(best_eval, eval_r)
            if (steps_to_mastery is None) and (eval_r >= mastery_threshold):
                steps_to_mastery = agent.total_steps

            # Capture telemetry at the same cadence as eval. For 5k-step
            # eval intervals on CartPole/MCC (~50-200 steps/ep), this
            # gives 25-100 episodes between checkpoints — fine resolution
            # to bracket the prereg's "by ep 100" abort threshold.
            tele = _capture_telemetry()
            if tele is not None:
                telemetry.append(tele)
                # Real-time alert: print loudly the FIRST time drift
                # exceeds the prereg's 50% smoke-abort threshold so the
                # operator sees it without scraping JSON.
                #
                # v5.3: gated on episode <= 100 (the actual prereg abort
                # window). Without this gate, the alert fires spuriously
                # at late episodes (e.g. ep 1218 on a seed 42 MCC transfer
                # run) where drift > 50% is expected post-warmup and is
                # NOT an abort signal. The latching flag then burns for
                # the run, silencing any *real* early-window warning on
                # later drift re-spikes. Live-detected during pilot #2
                # (seed 42 emitted an alarming "ABORT smoke" message at
                # ep 1218 step 95,912, alarmingly framed as if it were
                # a real abort; the smoke_verdict tool is the authoritative
                # enforcer and it correctly ignores post-warmup drift).
                if (not drift_alert_emitted
                        and tele["transferable_drift_max"] > 0.50
                        and tele["episode"] <= 100):
                    drift_alert_emitted = True
                    print(
                        f"  [TELEMETRY ALERT] transferable ||Δθ||/||θ_init||"
                        f" = {tele['transferable_drift_max']:.2%} at "
                        f"ep {tele['episode']} (step {tele['step']:,}) — "
                        f"IN THE ABORT WINDOW (ep<=100). Per prereg "
                        f"amendment 'Bug E v2', this should have been "
                        f"caught by the smoke pre-check. Investigate.",
                        flush=True,
                    )

    # Final eval at the truncation horizon so every run has a last point.
    final_eval = _evaluate(agent, env, episodes=eval_episodes)
    if not eval_curve or eval_curve[-1].step < agent.total_steps:
        eval_curve.append(EvalPoint(step=agent.total_steps,
                                    eval_return=final_eval))
    best_eval = max(best_eval, final_eval)
    if (steps_to_mastery is None) and (final_eval >= mastery_threshold):
        steps_to_mastery = agent.total_steps

    wall = time.time() - start_time

    if vec_env is not None:
        vec_env.close()
    env.close()

    # One last telemetry capture at the truncation horizon so the analyzer
    # has a final reading even if the last eval checkpoint landed earlier.
    final_tele = _capture_telemetry()
    if final_tele is not None and (not telemetry
                                   or telemetry[-1]["step"] != final_tele["step"]):
        telemetry.append(final_tele)

    return PilotRun(
        pair_alias=pair_alias or "",
        pair_role=pair_role or "",
        src_env=src_env or "",
        tgt_env=env_name,
        seed=seed,
        arm=arm,
        mastery_threshold=mastery_threshold,
        max_env_steps=max_env_steps,
        total_env_steps=agent.total_steps,
        total_episodes=agent.total_episodes,
        final_eval_return=final_eval,
        best_eval_return=best_eval,
        steps_to_mastery=steps_to_mastery,
        eval_curve=eval_curve,
        acting_policy_mode=agent.acting_policy_mode,
        transfer_skill_name=transfer_skill_name,
        wall_clock_sec=wall,
        used_vec=use_vec,
        telemetry=telemetry,
        ablation=ablation,
        ablation_info=ablation_info,
    )


def _pretrain_source(
    src_env: str,
    seed: int,
    skills_dir: str,
    max_env_steps: int,
    eval_every_steps: int,
    eval_episodes: int,
    mastery_threshold: float,
) -> PilotRun:
    """Pre-train source task and force crystallization.

    Source runs are recorded as PilotRun (arm="source") so they're auditable
    alongside the pilot arms but excluded from RMST analysis.
    """
    _seed_everything(seed)
    spec = get_env_spec(src_env)
    agent, env = _build_agent(src_env, seed, skills_dir)

    # Source lowers the crystallization threshold so the skill lands in the
    # library even if the source doesn't fully master the env — the pilot
    # cares about *having* a transferable skill, not about source mastery.
    agent.config.skill.thresholds[env.env_name] = mastery_threshold

    eval_curve: list[EvalPoint] = []
    steps_to_mastery: int | None = None
    last_eval_step = 0
    best_eval = -float("inf")
    iteration = 0
    crystallized = False

    start_time = time.time()
    while agent.total_steps < max_env_steps:
        iteration += 1
        agent.train_policy_real()

        if (iteration % 10 == 0
                and agent.replay_buffer.num_episodes >= 10):
            agent.train_world_model(steps=2)

        # Try crystallization every 10 iters
        if iteration % 10 == 0 and not crystallized:
            skill = agent.check_crystallization()
            if skill is not None:
                crystallized = True
                break  # Source done — skill is saved to skills_dir

        if (agent.total_steps - last_eval_step) >= eval_every_steps:
            eval_r = _evaluate(agent, env, episodes=eval_episodes)
            eval_curve.append(EvalPoint(step=agent.total_steps,
                                        eval_return=eval_r))
            last_eval_step = agent.total_steps
            best_eval = max(best_eval, eval_r)
            if (steps_to_mastery is None) and (eval_r >= mastery_threshold):
                steps_to_mastery = agent.total_steps

    # Always emit a final eval — whether we crystallized early or hit the cap.
    # This lets reviewers compare crystallized-source quality vs capped-source
    # quality on identical axes.
    final_eval = _evaluate(agent, env, episodes=eval_episodes)
    if not eval_curve or eval_curve[-1].step < agent.total_steps:
        eval_curve.append(EvalPoint(step=agent.total_steps,
                                    eval_return=final_eval))
    best_eval = max(best_eval, final_eval)
    if (steps_to_mastery is None) and (final_eval >= mastery_threshold):
        steps_to_mastery = agent.total_steps
    wall = time.time() - start_time
    env.close()

    return PilotRun(
        pair_alias="",
        pair_role="",
        src_env=src_env,
        tgt_env=src_env,
        seed=seed,
        arm="source",
        mastery_threshold=mastery_threshold,
        max_env_steps=max_env_steps,
        total_env_steps=agent.total_steps,
        total_episodes=agent.total_episodes,
        final_eval_return=final_eval,
        best_eval_return=best_eval,
        steps_to_mastery=steps_to_mastery,
        eval_curve=eval_curve,
        acting_policy_mode=agent.acting_policy_mode,
        transfer_skill_name=None,
        source_crystallized=crystallized,
        wall_clock_sec=wall,
    )


# ── Threshold resolution ────────────────────────────────────────────

def resolve_mastery_thresholds(
    overrides_path: Path | None,
    pairs: list[tuple[str, str, str, str]] | None = None,
) -> dict[str, float]:
    """Resolve mastery thresholds for pilot target envs.

    Priority:
      1. --mastery-thresholds JSON override (e.g. SB3-derived 80% values)
      2. env registry reward_threshold (Gymnasium default)

    `pairs`: which matrix to resolve envs from. Defaults to `PILOT_PAIRS`
    for backward-compat with all existing callers; `--run-adversarial`
    passes `ADVERSARIAL_PAIRS` so `finger-spin` resolves instead of
    `mountaincar-continuous` / `acrobot` / `cartpole-swingup`.

    Returns a dict keyed by env registry name. Every pair target env MUST
    have a resolved threshold; otherwise we fail loudly rather than silently
    using a bogus proxy.
    """
    thresholds: dict[str, float] = {}
    pair_list = pairs if pairs is not None else PILOT_PAIRS
    tgt_envs = {tgt for (_, _, tgt, _) in pair_list}
    src_envs = {src for (_, src, _, _) in pair_list}

    if overrides_path is not None and overrides_path.exists():
        data = json.loads(overrides_path.read_text())
        overrides = data.get("pilot_mastery_thresholds", data)  # flexible schema
    else:
        overrides = {}

    for env_name in tgt_envs | src_envs:
        if env_name in overrides:
            thresholds[env_name] = float(overrides[env_name])
            continue
        spec = get_env_spec(env_name)
        thresholds[env_name] = float(spec.reward_threshold)

    return thresholds


# ── Pilot orchestration ─────────────────────────────────────────────

def run_pilot(
    seeds: int = PILOT_SEEDS_DEFAULT,
    max_env_steps: int = MAX_ENV_STEPS_DEFAULT,
    source_max_env_steps: int = SOURCE_MAX_ENV_STEPS_DEFAULT,
    eval_every_steps: int = EVAL_EVERY_STEPS_DEFAULT,
    eval_episodes: int = EVAL_EPISODES_DEFAULT,
    skills_root: Path = Path("pilot_skills"),
    output_path: Path = Path("pilot_results.json"),
    pair_filter: list[str] | None = None,
    mastery_thresholds: dict[str, float] | None = None,
    base_seed: int = 42,
    num_envs: int = 1,
    ablation: str = "none",
    pairs_override: list[tuple[str, str, str, str]] | None = None,
) -> list[PilotRun]:
    """Run the Phase 3 pilot matrix. Writes output_path incrementally.

    `pairs_override`: when provided, replaces `PILOT_PAIRS` as the source
    matrix. Used by `--run-adversarial` to run the A10 adversarial-negative
    pair (cartpole → finger-spin) into a separate results file without
    polluting the primary pilot JSON. `pair_filter` still applies on top.

    `ablation`: routed through to `_train_to_step_budget` for the transfer
    arm only. Supported: "none" (default) | "shuffled-gru" (A11).
    """
    if ablation not in SUPPORTED_ABLATIONS:
        raise ValueError(
            f"Unknown ablation {ablation!r}; supported: "
            f"{list(SUPPORTED_ABLATIONS.keys())}"
        )

    if mastery_thresholds is None:
        mastery_thresholds = resolve_mastery_thresholds(None)

    base_pairs = pairs_override if pairs_override is not None else PILOT_PAIRS
    pairs = base_pairs
    if pair_filter:
        pairs = [p for p in base_pairs if p[0] in set(pair_filter)]
        if not pairs:
            raise ValueError(
                f"--pairs filter {pair_filter!r} matched no pilot pairs. "
                f"Available aliases: {[p[0] for p in base_pairs]}"
            )

    # Skills root per-pilot. Each (pair, seed) gets its own subdir so transfer
    # arms only see the source skill from the matching seed — eliminates
    # cross-seed leakage that would make "transfer" look artificially good.
    skills_root.mkdir(parents=True, exist_ok=True)

    all_runs: list[PilotRun] = []

    # Resume support: if output_path exists, load prior runs and skip any
    # (pair, seed, arm) triple already present. Useful when pilot is
    # interrupted at ~hour N of an 8-hour run.
    completed_keys: set[tuple[str, int, str]] = set()
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            for r in existing.get("runs", []):
                key = (r.get("pair_alias", ""), r.get("seed", -1), r.get("arm", ""))
                # "source" runs are keyed by (src_env, seed, "source")
                if r.get("arm") == "source":
                    key = (r.get("src_env", ""), r.get("seed", -1), "source")
                completed_keys.add(key)
                all_runs.append(_run_from_dict(r))
            print(f"[pilot] Resumed: {len(all_runs)} runs already in "
                  f"{output_path}", flush=True)
        except Exception as e:
            # Engineering-architect review (Phase 3 pre-launch): if the
            # results file is corrupt (truncated write, power-loss, manual
            # edit), silently wiping all_runs and starting fresh would cause
            # the very next _flush() to overwrite the .bak with the empty
            # state — losing hours of completed runs. Abort loudly instead
            # and force manual recovery from the .bak sibling.
            bak = output_path.with_suffix(output_path.suffix + ".bak")
            hint = (f" A .bak sibling exists at {bak} — inspect it manually "
                    f"and if it looks good, copy it over {output_path.name} "
                    f"before re-running.") if bak.exists() else ""
            raise SystemExit(
                f"[pilot] ABORT: could not resume from {output_path}: {e}."
                f"{hint}\n"
                f"Fix the file (or delete it to start fresh) and re-run. "
                f"The pilot will NOT auto-wipe to prevent silent data loss."
            )

    def _already_done(key: tuple[str, int, str]) -> bool:
        return key in completed_keys

    # Provenance collected once at launch so reviewer-replay metadata doesn't
    # churn across incremental flushes (a dirty-git flip mid-pilot would
    # otherwise look alarming in the output).
    provenance = _collect_provenance()

    def _flush() -> None:
        payload = {
            "prereg_section": "§8 (pilot)",
            # Report the actual pair matrix executed — previously this was
            # hard-coded to PILOT_PAIRS, which gave the adversarial results
            # file a misleading "pairs" header. Now reviewers see exactly
            # which matrix produced the runs in this file.
            "pairs": [
                {"alias": a, "src": s, "tgt": t, "role": r}
                for (a, s, t, r) in pairs
            ],
            "seeds_N": seeds,
            "base_seed": base_seed,
            "max_env_steps": max_env_steps,
            "source_max_env_steps": source_max_env_steps,
            "eval_every_steps": eval_every_steps,
            "eval_episodes": eval_episodes,
            "mastery_thresholds": mastery_thresholds,
            # v3.5 §7 A11: top-level ablation tag so downstream consumers
            # (pilot_analysis.py, manual inspection) can distinguish A11
            # result files from stock pilot files without scanning every
            # run. Per-run `ablation` still carries the ground truth.
            "ablation": ablation,
            "provenance": provenance,
            "runs": [r.to_dict() for r in all_runs],
        }
        _atomic_write_json(output_path, payload)

    t_outer = time.time()
    for (alias, src_env, tgt_env, role) in pairs:
        print(f"\n{'#'*70}", flush=True)
        print(f"  PAIR: {alias}  ({src_env} -> {tgt_env}, role={role})",
              flush=True)
        print(f"{'#'*70}", flush=True)

        tgt_threshold = mastery_thresholds[tgt_env]
        src_threshold = mastery_thresholds[src_env]

        for s in range(seeds):
            seed = base_seed + s
            # Devil's-advocate review (Phase 3 pre-launch, smoke #2):
            # Key the source-skills dir by (src_env, seed), NOT by
            # (pair, seed). Without this, `src_key` dedup correctly
            # avoids redundant source training when pair 2 shares the
            # same src_env as pair 1 — but pair 2's per-pair skills
            # dir stays EMPTY, and its transfer arm loads nothing.
            # Share the dir across pairs with matching src_env+seed.
            per_seed_skills = skills_root / f"source_{src_env}_seed{seed}"

            # --- 1. Source pre-training (for transfer arm only) ---
            src_key = (src_env, seed, "source")
            if not _already_done(src_key):
                # Clean per-seed skills dir to guarantee isolation
                if per_seed_skills.exists():
                    shutil.rmtree(per_seed_skills)
                per_seed_skills.mkdir(parents=True)

                print(f"  [source seed={seed}] {src_env} (cap "
                      f"{source_max_env_steps:,} steps)...", flush=True)
                src_run = _pretrain_source(
                    src_env=src_env,
                    seed=seed,
                    skills_dir=str(per_seed_skills),
                    max_env_steps=source_max_env_steps,
                    eval_every_steps=eval_every_steps,
                    eval_episodes=eval_episodes,
                    mastery_threshold=src_threshold,
                )
                all_runs.append(src_run)
                completed_keys.add(src_key)
                status = ("crystallized" if src_run.total_env_steps < source_max_env_steps
                          else "reached cap without crystallizing")
                print(f"    -> {status}, eval={src_run.final_eval_return:.1f}, "
                      f"{src_run.wall_clock_sec:.0f}s", flush=True)
                _flush()
            else:
                print(f"  [source seed={seed}] {src_env}: already done (skip)",
                      flush=True)

            # --- 2. Scratch arm — isolated skills dir (empty subdir so
            #        try_transfer is even structurally unable to find a skill)
            scratch_key = (alias, seed, "scratch")
            if not _already_done(scratch_key):
                empty_skills = skills_root / f"{alias}_seed{seed}_empty"
                if empty_skills.exists():
                    shutil.rmtree(empty_skills)
                empty_skills.mkdir(parents=True)

                print(f"  [seed={seed}] scratch {tgt_env} ({max_env_steps:,} "
                      f"steps, τ={tgt_threshold:.1f})...", flush=True)
                r = _train_to_step_budget(
                    env_name=tgt_env,
                    seed=seed,
                    arm="scratch",
                    skills_dir=str(empty_skills),
                    max_env_steps=max_env_steps,
                    eval_every_steps=eval_every_steps,
                    eval_episodes=eval_episodes,
                    mastery_threshold=tgt_threshold,
                    pair_alias=alias,
                    pair_role=role,
                    src_env=src_env,
                    num_envs=num_envs,
                )
                all_runs.append(r)
                completed_keys.add(scratch_key)
                mastery_str = (f"{r.steps_to_mastery:,}" if r.steps_to_mastery
                               else "censored")
                print(f"    -> final={r.final_eval_return:.1f}, best="
                      f"{r.best_eval_return:.1f}, mastery@{mastery_str}, "
                      f"{r.wall_clock_sec:.0f}s", flush=True)
                _flush()
            else:
                print(f"  [seed={seed}] scratch {tgt_env}: already done (skip)",
                      flush=True)

            # --- 3. Transfer arm — uses per-seed skills dir with source skill
            transfer_key = (alias, seed, "transfer")
            if not _already_done(transfer_key):
                ablation_note = (f" [ablation={ablation}]"
                                 if ablation != "none" else "")
                print(f"  [seed={seed}] transfer {tgt_env} ({max_env_steps:,} "
                      f"steps, τ={tgt_threshold:.1f}){ablation_note}...",
                      flush=True)
                r = _train_to_step_budget(
                    env_name=tgt_env,
                    seed=seed,
                    arm="transfer",
                    skills_dir=str(per_seed_skills),
                    max_env_steps=max_env_steps,
                    eval_every_steps=eval_every_steps,
                    eval_episodes=eval_episodes,
                    mastery_threshold=tgt_threshold,
                    pair_alias=alias,
                    pair_role=role,
                    src_env=src_env,
                    num_envs=num_envs,
                    ablation=ablation,
                )
                all_runs.append(r)
                completed_keys.add(transfer_key)
                mastery_str = (f"{r.steps_to_mastery:,}" if r.steps_to_mastery
                               else "censored")
                src_str = r.transfer_skill_name or "NO SKILL LOADED"
                print(f"    -> final={r.final_eval_return:.1f}, best="
                      f"{r.best_eval_return:.1f}, mastery@{mastery_str}, "
                      f"mode={r.acting_policy_mode}, src={src_str}, "
                      f"{r.wall_clock_sec:.0f}s", flush=True)
                _flush()
            else:
                print(f"  [seed={seed}] transfer {tgt_env}: already done (skip)",
                      flush=True)

    total_wall = time.time() - t_outer
    print(f"\n{'='*70}", flush=True)
    print(f"  PILOT COMPLETE: {len(all_runs)} runs, "
          f"{total_wall/3600:.2f} GPU-hr wall", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Results written to: {output_path}", flush=True)

    return all_runs


def _run_from_dict(d: dict) -> PilotRun:
    """Reconstruct a PilotRun from its serialized dict (for resume)."""
    curve = [EvalPoint(step=p["step"], eval_return=p["eval_return"])
             for p in d.get("eval_curve", [])]
    return PilotRun(
        pair_alias=d.get("pair_alias", ""),
        pair_role=d.get("pair_role", ""),
        src_env=d.get("src_env", ""),
        tgt_env=d.get("tgt_env", ""),
        seed=d.get("seed", -1),
        arm=d.get("arm", ""),
        mastery_threshold=float(d.get("mastery_threshold", 0.0)),
        max_env_steps=int(d.get("max_env_steps", 0)),
        total_env_steps=int(d.get("total_env_steps", 0)),
        total_episodes=int(d.get("total_episodes", 0)),
        final_eval_return=float(d.get("final_eval_return", 0.0)),
        best_eval_return=float(d.get("best_eval_return", 0.0)),
        steps_to_mastery=d.get("steps_to_mastery"),
        eval_curve=curve,
        acting_policy_mode=d.get("acting_policy_mode", "obs"),
        transfer_skill_name=d.get("transfer_skill_name"),
        source_crystallized=d.get("source_crystallized"),
        used_vec=bool(d.get("used_vec", False)),
        wall_clock_sec=float(d.get("wall_clock_sec", 0.0)),
        telemetry=list(d.get("telemetry", [])),
        # v3.5 §7 A11 ablation metadata. Pre-v3.5 pilot_results.json
        # files don't carry these keys; default to "none"/None so the
        # resume path stays backward-compatible with pilot #2 runs that
        # landed before this amendment.
        ablation=d.get("ablation", "none"),
        ablation_info=d.get("ablation_info"),
    )


def _cross_dim(src_env: str, tgt_env: str) -> bool:
    """Return True iff the (src, tgt) pair requires the latent-policy path.

    Shared predicate between pilot_run (asserts loudly at transfer time)
    and pilot_analysis (expects mode=='latent' for cross-dim). Keeping one
    canonical definition prevents drift.
    """
    src_spec = get_env_spec(src_env)
    tgt_spec = get_env_spec(tgt_env)
    return (src_spec.obs_dim != tgt_spec.obs_dim
            or src_spec.action_dim != tgt_spec.action_dim
            or src_spec.is_discrete != tgt_spec.is_discrete)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=PILOT_SEEDS_DEFAULT,
                        help=f"Seeds per (pair, arm) (default {PILOT_SEEDS_DEFAULT})")
    parser.add_argument("--max-steps", type=int, default=MAX_ENV_STEPS_DEFAULT,
                        help=f"Env-steps per target run (default "
                             f"{MAX_ENV_STEPS_DEFAULT:,}, per §8)")
    parser.add_argument("--source-max-steps", type=int,
                        default=SOURCE_MAX_ENV_STEPS_DEFAULT,
                        help=f"Env-steps cap for source pre-training "
                             f"(default {SOURCE_MAX_ENV_STEPS_DEFAULT:,})")
    parser.add_argument("--eval-every", type=int,
                        default=EVAL_EVERY_STEPS_DEFAULT,
                        help=f"Eval frequency in env-steps (default "
                             f"{EVAL_EVERY_STEPS_DEFAULT}, per §4.5)")
    parser.add_argument("--eval-episodes", type=int,
                        default=EVAL_EPISODES_DEFAULT,
                        help=f"Eval episodes per checkpoint (default "
                             f"{EVAL_EPISODES_DEFAULT}, per §4.5)")
    parser.add_argument("--skills-root", type=Path, default=Path("pilot_skills"),
                        help="Root for per-(pair,seed) skill dirs")
    parser.add_argument("--output", type=Path, default=Path("pilot_results.json"),
                        help="Output JSON path (incrementally updated)")
    parser.add_argument("--pairs", type=str, nargs="+", default=None,
                        help="Filter to specific pair aliases "
                             "(e.g. --pairs cartpole_mcc)")
    parser.add_argument("--mastery-thresholds", type=Path, default=None,
                        help="Optional JSON with per-env mastery thresholds "
                             "(overrides registry defaults)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="First seed; runs use base_seed..base_seed+N-1")
    parser.add_argument("--vec", type=int, default=1,
                        help="Parallel envs for A2C collection (discrete only)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick-smoke mode (prereg v3, v5 source-cap "
                             "fix): seeds=2, max-steps=40k, "
                             "source-max=100k, pairs limited to "
                             "cartpole_mcc. 100k source cap matches the "
                             "v2 smoke that successfully crystallized "
                             "(354s + 256s on the two seeds). The earlier "
                             "10k value was too short for cartpole to "
                             "crystallize (eval=19/500), causing the "
                             "transfer arm to silently fall back to "
                             "scratch and emit zero telemetry.")
    # v3.5 §7 A11: GRU-shuffle ablation on the transfer path. "none" keeps
    # stock behavior; "shuffled-gru" row-column-permutes the transferable
    # GRU weights after try_transfer() loads the skill. Intended for a
    # small companion run (2 seeds on cartpole_mcc) that falsifies the
    # "learned recurrent dynamics transfer" claim if A11 ≈ real transfer.
    parser.add_argument("--ablation", type=str, default="none",
                        choices=list(SUPPORTED_ABLATIONS.keys()),
                        help="Ablation to apply on the transfer arm "
                             "(default 'none'). See SUPPORTED_ABLATIONS "
                             "for semantics; 'shuffled-gru' is A11.")
    # v3.5 §7 A10: adversarial-negative pair (cartpole → DMC finger-spin).
    # Swaps PILOT_PAIRS for ADVERSARIAL_PAIRS and changes the default
    # output path so the adversarial results sit in their own JSON file
    # (the primary pilot_results.json stays clean).
    parser.add_argument("--run-adversarial", action="store_true",
                        help="Run the A10 adversarial-negative pair "
                             "(cartpole → finger-spin) instead of "
                             "PILOT_PAIRS. Requires venv310 (DMC). "
                             "Default output: pilot_adversarial_results.json.")
    args = parser.parse_args(argv)

    if args.smoke:
        # Prereg v3 commits to 2-seed smoke with telemetry enforceable
        # at ep 100 (||Δθ|| > 50% triggers abort). Bumping max_steps to
        # 40k ensures both seeds clear ep 100 with margin even on slow
        # mastery curves; eval_every=5k gives 8 telemetry checkpoints.
        #
        # source_max_steps=100k (v5 fix): the v4 default of 10k was too
        # short for cartpole to crystallize. Without crystallization, the
        # transfer arm falls back to scratch and the smoke's whole point
        # (validating the telemetry-emitting transfer code path) is
        # defeated. 100k matches the v2 smoke that successfully
        # crystallized both seeds.
        args.seeds = 2
        args.max_steps = 40_000
        args.source_max_steps = 100_000
        args.eval_every = min(args.eval_every, 5_000)
        if args.pairs is None:
            args.pairs = ["cartpole_mcc"]

    # §7 A10: adversarial pair gets its own default output path so the
    # primary pilot_results.json stays untouched. If the user explicitly
    # passed --output, respect it (detected via the parser-default
    # sentinel); otherwise, swap in the adversarial default.
    pairs_override: list[tuple[str, str, str, str]] | None = None
    if args.run_adversarial:
        pairs_override = ADVERSARIAL_PAIRS
        # Only override --output if the user left it at the default. A
        # literal-equality check against the default Path() avoids
        # stomping on an explicit override.
        if args.output == Path("pilot_results.json"):
            args.output = Path("pilot_adversarial_results.json")
        print(f"[pilot] A10 adversarial pair mode: "
              f"pairs={[p[0] for p in ADVERSARIAL_PAIRS]}", flush=True)

    mastery = resolve_mastery_thresholds(
        args.mastery_thresholds,
        pairs=pairs_override if pairs_override is not None else PILOT_PAIRS,
    )

    print(f"[pilot] device={DEVICE} seeds={args.seeds} "
          f"max_steps={args.max_steps:,} "
          f"source_cap={args.source_max_steps:,}", flush=True)
    print(f"[pilot] mastery thresholds: {mastery}", flush=True)
    if args.pairs:
        print(f"[pilot] pair filter: {args.pairs}", flush=True)
    if args.ablation != "none":
        print(f"[pilot] A11 ablation mode: {args.ablation}", flush=True)

    run_pilot(
        seeds=args.seeds,
        max_env_steps=args.max_steps,
        source_max_env_steps=args.source_max_steps,
        eval_every_steps=args.eval_every,
        eval_episodes=args.eval_episodes,
        skills_root=args.skills_root,
        output_path=args.output,
        pair_filter=args.pairs,
        mastery_thresholds=mastery,
        base_seed=args.base_seed,
        num_envs=args.vec,
        ablation=args.ablation,
        pairs_override=pairs_override,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
