"""v4.0 Phase 3 — the developmental loop: a growing skill library + a
relevance gate that REUSES a known skill when it already solves a task,
or LEARNS a new one when nothing applies ("si pas de lien, apprend une
nouvelle notion"). Measures the learning-to-learn curve: marginal cost
to master a new task falls as the library of mastered notions grows.

Preregistered as preregistration.md amendment v4.0 Phase 3 (committed
before this script, the `regime` env change, and any run).

Substrate: DeviceVecPointMass2D with a `regime` parameter — distinct
DYNAMICS (free / drift / ice / reverse / rot90 / rot270) = distinct
sensorimotor "notions", same obs(4)/action(2)/reward. A goal-conditioned
skill pi(obs4, goal2) mastered in one regime generalizes across GOALS in
that regime but is expected NOT to transfer across regimes.

A skill         : goal-conditioned SAC policy (obs 6 = [state4, goal2]).
The gate        : probe every library skill on the task (a few eval
                  rollouts; cost counted). Reuse iff the best already
                  masters it (success >= mastery); else learn + add.
Arms:
  reuse_gated        : the developmental loop (library + empirical gate).
  no_reuse           : ablation — every task learns a fresh skill.
  always_reuse_first : control — force-reuse skill #1 for all tasks (no
                       gate); novel regimes stay UNMASTERED.

Usage:
  python -m scripts.devloop_v4 --validate   # skill x regime probe matrix
  python -m scripts.devloop_v4 [--smoke]    # the 3-arm curriculum
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.sac import SACTrainer, SACPolicy, DeviceSACBuffer
from ragnarok.learning.rollout import RolloutBatch
from ragnarok.environments.device_env import DeviceVecPointMass2D

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# --------------------------------------------------------------------------
# A skill: goal-conditioned "reach a point under THIS regime's dynamics".
# --------------------------------------------------------------------------
def _collect_goal_cond(env, sac, horizon):
    """Rollout on the regime env; obs fed to SAC is [state(4), goal(2)]."""
    n = env.num_envs
    obs_l, act_l, rew_l, done_l = [], [], [], []
    for _ in range(horizon):
        lo_obs = torch.cat([env.state, env.goal], dim=-1)
        a, _, _ = sac.device_policy_fn(lo_obs)
        _, r, _t, _tr, done = env.step(a)
        obs_l.append(lo_obs); act_l.append(a); rew_l.append(r)
        done_l.append(done.float())
    last = torch.cat([env.state, env.goal], dim=-1)
    z = torch.zeros(n, horizon, device=DEVICE)
    return RolloutBatch(obs=torch.stack(obs_l, 1), raw_obs=torch.stack(obs_l, 1),
                        actions=torch.stack(act_l, 1), rewards=torch.stack(rew_l, 1),
                        dones=torch.stack(done_l, 1), logp=z, values=z,
                        last_obs=last, last_value=torch.zeros(n, device=DEVICE))


@torch.no_grad()
def _skill_action(pi, obs4, g):
    mean, _ = pi.forward(torch.cat([obs4, g], dim=-1))
    return pi._rescale(torch.tanh(mean))


@torch.no_grad()
def _skill_success(pi, regime, cfg, n_trials=None):
    """Reach-success of skill `pi` on `regime` (random starts+goals).
    This is BOTH the mastery eval and the gate's probe; the probe's
    env-steps (n_trials * eval_steps) are counted in the loop's budget."""
    n = n_trials or cfg["probe_trials"]
    env = DeviceVecPointMass2D(n, regime=regime)
    reached = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["eval_steps"]):
        a = _skill_action(pi, env.state, env.goal)
        _, _, term, _, _ = env.step(a)
        reached = reached | term
    return float(reached.float().mean().item())


def _fresh_skill():
    pol = SACPolicy(6, 2, action_low=np.full(2, -1.0, dtype=np.float32),
                    action_high=np.full(2, 1.0, dtype=np.float32)).to(DEVICE)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


def _learn_skill(regime, cfg, verbose=False):
    """Train a goal-conditioned skill on `regime` until it is CONSOLIDATED
    (eval success >= cfg['consolidate'], default 0.95) or the rollout cap.

    Why consolidate above the mastery bar: a skill that *barely* crosses
    0.80 re-probes around 0.80 +/- noise, so the gate can later fail to
    recognise its own skill and duplicate it (observed in the v1 strict
    run: library bloated to 6). Practising to fluency (>=0.95) makes a
    same-notion re-probe reliably clear the preregistered 0.80 reuse bar.
    The reuse bar and the "mastered" definition stay at cfg['mastery'].

    Returns (frozen_policy, env_steps_spent, mastered_bool, final_success)."""
    env = DeviceVecPointMass2D(cfg["num_envs"], regime=regime)
    sac = SACTrainer(obs_dim=6, action_dim=2,
                     action_low=np.full(2, -1.0, dtype=np.float32),
                     action_high=np.full(2, 1.0, dtype=np.float32),
                     warmup_steps=cfg["num_envs"] * cfg["horizon"],
                     buffer=DeviceSACBuffer(capacity=200_000))
    total, last = 0, 0.0
    for it in range(1, cfg["skill_rollouts"] + 1):
        batch = _collect_goal_cond(env, sac, cfg["horizon"])
        sac.train_on_rollout(batch, n_updates=cfg["skill_updates"])
        total += batch.total_steps
        if it % cfg["eval_every"] == 0:
            last = _skill_success(sac.policy, regime, cfg)
            if verbose:
                print(f"      [learn {regime}] it {it:>2} | succ {last:.2f} "
                      f"| steps {total:,}", flush=True)
            if last >= cfg["consolidate"]:
                break
    pi = sac.policy
    pi.eval()
    for p in pi.parameters():
        p.requires_grad_(False)
    return pi, total, last >= cfg["mastery"], last


# --------------------------------------------------------------------------
# The relevance gate over a growing library.
# --------------------------------------------------------------------------
def _probe_library(library, regime, cfg):
    """Run every library skill on the task; return (best_idx, best_succ,
    probe_steps). Empirical — the agent is never told the regime."""
    best_idx, best_succ = None, -1.0
    probe_steps = 0
    for idx, pi in enumerate(library):
        s = _skill_success(pi, regime, cfg, n_trials=cfg["probe_trials"])
        probe_steps += cfg["probe_trials"] * cfg["eval_steps"]
        if s > best_succ:
            best_succ, best_idx = s, idx
    return best_idx, best_succ, probe_steps


def _run_reuse_gated(curriculum, cfg):
    """For each task: probe the library; REUSE if a known skill already
    masters it, else LEARN a new skill and add it."""
    library, log = [], []
    for ti, regime in enumerate(curriculum):
        best_idx, best_succ, probe_steps = _probe_library(library, regime, cfg)
        if best_succ >= cfg["mastery"]:
            decision, cost, mastered = "reuse", probe_steps, True
            note = f"skill#{best_idx} succ {best_succ:.2f}"
        else:
            pi, learn_steps, mastered, fin = _learn_skill(regime, cfg)
            library.append(pi)
            decision, cost = "learn", probe_steps + learn_steps
            note = f"new skill#{len(library)-1} succ {fin:.2f} (best prior {best_succ:.2f})"
        log.append(dict(task=ti, regime=regime, decision=decision, cost=cost,
                        mastered=mastered, note=note))
        print(f"    [reuse_gated] task {ti:>2} {regime:8s} -> {decision:5s} "
              f"| +{cost:,} steps | {note}", flush=True)
    return log, len(library)


def _run_no_reuse(curriculum, cfg):
    """Ablation: no library, no probe — learn a fresh skill every task."""
    log = []
    for ti, regime in enumerate(curriculum):
        pi, learn_steps, mastered, fin = _learn_skill(regime, cfg)
        log.append(dict(task=ti, regime=regime, decision="learn",
                        cost=learn_steps, mastered=mastered,
                        note=f"succ {fin:.2f}"))
        print(f"    [no_reuse]    task {ti:>2} {regime:8s} -> learn  "
              f"| +{learn_steps:,} steps | succ {fin:.2f}", flush=True)
    return log


def _run_always_reuse_first(curriculum, cfg):
    """Control: learn ONE skill (task #1's regime), force-reuse it for all
    tasks (no gate). Novel regimes stay unmastered."""
    log = []
    pi0, cost0, m0, fin0 = _learn_skill(curriculum[0], cfg)
    for ti, regime in enumerate(curriculum):
        if ti == 0:
            decision, cost, mastered, succ = "learn", cost0, m0, fin0
        else:
            succ = _skill_success(pi0, regime, cfg, n_trials=cfg["probe_trials"])
            decision, cost = "reuse", cfg["probe_trials"] * cfg["eval_steps"]
            mastered = succ >= cfg["mastery"]
        log.append(dict(task=ti, regime=regime, decision=decision, cost=cost,
                        mastered=mastered, note=f"succ {succ:.2f}"))
        print(f"    [always_1st]  task {ti:>2} {regime:8s} -> {decision:5s} "
              f"| +{cost:,} steps | succ {succ:.2f} | mastered={mastered}",
              flush=True)
    return log


# --------------------------------------------------------------------------
# --validate: the skill x regime transfer matrix (the diagonal property).
# --------------------------------------------------------------------------
def _validate(regimes, cfg):
    print(f"[validate] training one skill per regime, then the full "
          f"skill x regime probe matrix\n", flush=True)
    skills, diag = {}, {}
    for r in regimes:
        pi, steps, ok, fin = _learn_skill(r, cfg, verbose=True)
        skills[r] = pi; diag[r] = (steps, ok, fin)
        print(f"  [{r}] mastered={ok} @ {steps:,} steps | self-success {fin:.2f}",
              flush=True)
    print(f"\n  transfer matrix  rows=skill, cols=test-regime "
          f"(diag should be high, off-diag low):")
    header = "        " + "".join(f"{c:>9}" for c in regimes)
    print(header, flush=True)
    M = {}
    for rs in regimes:
        row = {}
        cells = ""
        for rt in regimes:
            s = (diag[rs][2] if rs == rt
                 else _skill_success(skills[rs], rt, cfg))
            row[rt] = s
            cells += f"{s:>9.2f}"
        M[rs] = row
        print(f"  {rs:>6}{cells}", flush=True)
    # Diagonal/off-diagonal separation.
    diag_vals = [M[r][r] for r in regimes]
    off_vals = [M[rs][rt] for rs in regimes for rt in regimes if rs != rt]
    print(f"\n  diag mean {np.mean(diag_vals):.2f} (min {min(diag_vals):.2f}) "
          f"| off-diag mean {np.mean(off_vals):.2f} (max {max(off_vals):.2f})",
          flush=True)
    clean = min(diag_vals) >= cfg["mastery"] and max(off_vals) < cfg["mastery"]
    msg = ("CLEAN diagonal: regimes are distinct, masterable notions" if clean
           else "NOT clean — some regime transfers in or is unmasterable")
    print(f"  -> {msg}", flush=True)
    return clean, M


# --------------------------------------------------------------------------
def _make_curriculum(regimes, n_blocks, seed):
    """n_blocks epochs, each a random permutation of all regimes — so every
    notion appears once per block; the library should saturate after block 1."""
    rng = np.random.default_rng(seed)
    cur = []
    for _ in range(n_blocks):
        perm = list(regimes)
        rng.shuffle(perm)
        cur.extend(perm)
    return cur


def _summarize_curve(log, n_per_block):
    """Marginal cost per task + early-vs-late trend (the L2L signature)."""
    costs = [e["cost"] for e in log]
    cum = np.cumsum(costs).tolist()
    first = float(np.mean(costs[:n_per_block])) if costs else 0.0
    last = float(np.mean(costs[-n_per_block:])) if costs else 0.0
    # slope of cost vs task index (negative == getting cheaper).
    x = np.arange(len(costs))
    slope = float(np.polyfit(x, costs, 1)[0]) if len(costs) > 1 else 0.0
    return dict(costs=costs, cumulative=cum, total=int(sum(costs)),
                first_block_mean=first, last_block_mean=last, slope=slope,
                all_mastered=all(e["mastered"] for e in log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regimes", nargs="+",
                   default=["free", "rot90", "reverse", "rot270"])
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--skill-rollouts", type=int, default=80)
    p.add_argument("--skill-updates", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--probe-trials", type=int, default=64)
    p.add_argument("--mastery", type=float, default=0.8,
                   help="reuse bar + 'mastered' definition (preregistered)")
    p.add_argument("--consolidate", type=float, default=0.95,
                   help="train a new skill to this fluency before shelving it "
                        "(hysteresis above the reuse bar; see v1-strict finding)")
    p.add_argument("--out-dir", default="devloop_v4_out")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.num_envs, args.horizon = 1, 64, 32
        args.skill_rollouts, args.skill_updates = 12, 32
        args.eval_every, args.eval_steps, args.probe_trials = 3, 40, 32
        args.blocks = 2
        args.regimes = ["free", "reverse"]
        args.consolidate = 0.3

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "horizon", "skill_rollouts", "skill_updates",
            "eval_every", "eval_steps", "probe_trials", "mastery",
            "consolidate")}

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[devloop-v4] device={DEVICE} | regimes={args.regimes} | "
          f"blocks={args.blocks} seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    if args.validate:
        torch.manual_seed(0); np.random.seed(0)
        clean, M = _validate(args.regimes, cfg)
        with open(os.path.join(args.out_dir, "validate.json"), "w") as f:
            json.dump({"matrix": M, "clean": clean, "regimes": args.regimes}, f,
                      indent=2)
        print(f"\n  {time.perf_counter()-t0:.0f}s", flush=True)
        return

    n_reg = len(args.regimes)
    results_path = os.path.join(args.out_dir, "results.json")
    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = json.load(f).get("seeds", {})
        print(f"[resume] {len(done)} seed(s) already done: {list(done)}",
              flush=True)

    for seed in range(args.seeds):
        if str(seed) in done:
            print(f"[seed {seed}] cached — skipping", flush=True)
            continue
        print(f"\n[seed {seed}] curriculum = "
              f"{args.blocks} blocks x {n_reg} regimes", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        curriculum = _make_curriculum(args.regimes, args.blocks, seed)

        log_rg, lib_size = _run_reuse_gated(curriculum, cfg)
        log_nr = _run_no_reuse(curriculum, cfg)
        log_ar = _run_always_reuse_first(curriculum, cfg)

        sm_rg = _summarize_curve(log_rg, n_reg)
        sm_nr = _summarize_curve(log_nr, n_reg)
        sm_ar = _summarize_curve(log_ar, n_reg)
        sm_rg["library_size"] = lib_size

        done[str(seed)] = dict(curriculum=curriculum,
                               reuse_gated=dict(log=log_rg, **sm_rg),
                               no_reuse=dict(log=log_nr, **sm_nr),
                               always_reuse_first=dict(log=log_ar, **sm_ar))
        with open(results_path, "w") as f:
            json.dump({"regimes": args.regimes, "blocks": args.blocks,
                       "n_regimes": n_reg, "seeds": done}, f, indent=2)
        print(f"  [seed {seed}] reuse_gated total {sm_rg['total']:,} "
              f"(lib {lib_size}, first-block {int(sm_rg['first_block_mean']):,} "
              f"-> last-block {int(sm_rg['last_block_mean']):,}) | "
              f"no_reuse total {sm_nr['total']:,} | results -> {results_path}",
              flush=True)

    # ---- aggregate ----
    seeds = [done[str(s)] for s in range(args.seeds) if str(s) in done]
    if not seeds:
        return
    N = len(seeds)
    tval = _T95.get(N - 1, 2.0)

    def _ci(xs):
        m = float(np.mean(xs))
        if len(xs) < 2:
            return m, 0.0
        se = float(np.std(xs, ddof=1)) / (len(xs) ** 0.5)
        return m, tval * se

    rg_tot = [s["reuse_gated"]["total"] for s in seeds]
    nr_tot = [s["no_reuse"]["total"] for s in seeds]
    rg_first = [s["reuse_gated"]["first_block_mean"] for s in seeds]
    rg_last = [s["reuse_gated"]["last_block_mean"] for s in seeds]
    rg_slope = [s["reuse_gated"]["slope"] for s in seeds]
    libs = [s["reuse_gated"]["library_size"] for s in seeds]
    rg_master = [s["reuse_gated"]["all_mastered"] for s in seeds]
    ar_master_frac = [np.mean([e["mastered"] for e in s["always_reuse_first"]["log"]])
                      for s in seeds]

    print(f"\n{'=' * 74}")
    print(f"  v4.0 PHASE 3 — developmental loop | N={N} | regimes={args.regimes}")
    print(f"{'=' * 74}")
    m, h = _ci(rg_tot); print(f"  reuse_gated  total env-steps : {int(m):>12,} +/- {int(h):,}")
    m2, h2 = _ci(nr_tot); print(f"  no_reuse     total env-steps : {int(m2):>12,} +/- {int(h2):,}")
    print(f"  savings factor (no_reuse / reuse_gated)      : {m2/m:.2f}x")
    mf, _ = _ci(rg_first); ml, _ = _ci(rg_last)
    print(f"  reuse_gated marginal cost  first block {int(mf):,} -> "
          f"last block {int(ml):,}  ({'DOWN' if ml < mf else 'flat/up'})")
    ms, hs = _ci(rg_slope)
    print(f"  reuse_gated cost-vs-task slope : {ms:,.0f} +/- {int(hs):,} "
          f"({'negative=cheaper over time' if ms < 0 else 'not negative'})")
    print(f"  library size recovered : {libs}  (true notion count = {n_reg})")
    print(f"  reuse_gated all-tasks-mastered : {rg_master}")
    print(f"  always_reuse_first mastered fraction : "
          f"{[round(float(x),2) for x in ar_master_frac]} "
          f"(novel regimes fail without the gate)")

    decisive = (m2 > 1.3 * m and ml < mf and all(np.array(rg_slope) < 0)
                and all(np.array(libs) == n_reg) and all(rg_master)
                and np.mean(ar_master_frac) < 0.6)
    if decisive:
        verdict = ("DEVELOPMENTAL LOOP WORKS — marginal cost-to-master FALLS as "
                   "the library grows; the gate recovers exactly the true notion "
                   "count, reuses what applies and learns what is novel; total "
                   "cost << no-reuse. Gating (not mere reuse) is necessary.")
    elif m2 > 1.15 * m and ml < mf and all(np.array(libs) == n_reg):
        verdict = ("PARTIAL — reuse compounds and the gate is correct, but check "
                   "margins / trend strength.")
    else:
        verdict = ("CHECK — the loop did not clearly compound (trend, library "
                   "size, or savings off).")
    print(f"\n  -> {verdict}")
    print(f"  {time.perf_counter()-t0:.0f}s", flush=True)

    with open(results_path, "w") as f:
        json.dump({"regimes": args.regimes, "blocks": args.blocks,
                   "n_regimes": n_reg, "seeds": done, "verdict": verdict}, f,
                  indent=2)
    print(f"  results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
