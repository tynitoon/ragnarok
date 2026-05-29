"""v4.0 Phase 4 — COMPOSITIONAL reuse: a NOVEL composite route is solved by
assembling KNOWN motor primitives, with combinatorial leverage; the
"complex notion = composition of known basics, learned faster" claim.

Preregistered as preregistration.md amendment v4.0 Phase 4 (committed
before this script, the DeviceVecRelay env, and any run).

Substrate: DeviceVecRelay — an episode is L legs, each a (hidden rotation
REGIME, target ZONE). obs=[x,y,vx,vy,gx,gy]; the regime is hidden (the
agent is told WHERE to go, not which motor skill the leg needs). Completing
a route requires the right per-leg primitive -> composing a sequence of
skills. With R regimes and Z zones there are (R*Z)^L routes, so a few
primitives compose into exponentially many NOVEL routes (no route is ever
trained at the composite level) -> it cannot be a cache.

A primitive = a goal-conditioned reach skill under one regime (the
validated Phase-3 skill). The agent assigns a primitive per leg via the
Phase-3 gate (probe library; reuse if one masters the leg, else LEARN a new
primitive and add it -- the "if no link, learn a notion" branch), then
EXECUTES the route by running the chosen primitive per leg. Composite-level
learning is ZERO for known-regime routes.

Arms:
  compose_reuse      : pre-built library of the known regimes + gate + execute.
  compose_no_library : same agent, EMPTY library (must learn each primitive).
  flat_scratch       : flat SAC over [fx,fy] on the full relay, from scratch
                       (same obs + same sparse leg rewards; not hobbled).

Usage:
  python -m scripts.compose_v4 --validate   # learned primitives compose?
  python -m scripts.compose_v4 [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.sac import SACTrainer, DeviceSACBuffer
from ragnarok.learning.rollout import collect_rollout
from ragnarok.environments.device_env import DeviceVecRelay
from scripts.devloop_v4 import (_learn_skill, _skill_success, _skill_action,
                                 _fresh_skill)

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

KNOWN_REGIMES = ["free", "rot90", "reverse", "rot270"]
NOVEL_REGIME = "rot45"
ZONES = [(-0.6, 0.6), (0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)]


# --------------------------------------------------------------------------
# Curriculum: novel routes (never trained as wholes).
# --------------------------------------------------------------------------
def _make_route(rng, regimes, L):
    legs = []
    last_zone = None
    for _ in range(L):
        r = regimes[rng.integers(len(regimes))]
        zi = rng.integers(len(ZONES))
        while zi == last_zone:
            zi = rng.integers(len(ZONES))
        last_zone = zi
        legs.append((r, ZONES[zi]))
    return legs


def _make_curriculum(seed, n_known, n_novel, L):
    rng = np.random.default_rng(1000 + seed)
    known = [_make_route(rng, KNOWN_REGIMES, L) for _ in range(n_known)]
    novel = []
    for _ in range(n_novel):
        route = _make_route(rng, KNOWN_REGIMES, L)
        # force >=1 leg to the novel regime
        j = rng.integers(L)
        route[j] = (NOVEL_REGIME, route[j][1])
        novel.append(route)
    return known + novel        # known block first, then the novelty block


# --------------------------------------------------------------------------
# The gate over a (list) library + compose-execute.
# --------------------------------------------------------------------------
def _assign_primitives(library, route, cfg, allow_learn=True):
    """For each distinct regime in `route`, pick the library primitive that
    masters it (probe); if none does and allow_learn, LEARN+add one. Returns
    (regime->primitive dict, probe_steps, learn_steps, learns:list)."""
    regime2pi, probe_steps, learn_steps, learns = {}, 0, 0, []
    for regime in dict.fromkeys(r for r, _ in route):     # distinct, ordered
        best_pi, best = None, -1.0
        for pi in library:
            s = _skill_success(pi, regime, cfg)
            probe_steps += cfg["probe_trials"] * cfg["eval_steps"]
            if s > best:
                best, best_pi = s, pi
        if best >= cfg["mastery"]:
            regime2pi[regime] = best_pi
        elif allow_learn:
            pi, steps, ok, fin = _learn_skill(regime, cfg)
            library.append(pi)
            regime2pi[regime] = pi
            learn_steps += steps
            learns.append((regime, steps, fin))
        else:
            regime2pi[regime] = best_pi          # forced to use best (may fail)
    return regime2pi, probe_steps, learn_steps, learns


@torch.no_grad()
def _compose_eval(regime2pi, route, cfg, n_trials):
    """Execute `route` by applying the assigned primitive on each leg.
    Composite-level learning is ZERO — only per-leg primitive selection."""
    env = DeviceVecRelay(n_trials, route)
    completed = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(env.max_steps):
        a = torch.zeros(n_trials, 2, device=DEVICE)
        for j in range(env.L):
            mask = env.leg == j
            if bool(mask.any()):
                pi = regime2pi[route[j][0]]
                a[mask] = _skill_action(pi, env.state[mask, :4], env.state[mask, 4:6])
        _, _, term, _, _ = env.step(a)
        completed = completed | term
    return float(completed.float().mean().item())


# --------------------------------------------------------------------------
# Flat from-scratch baseline on the relay (fair: same obs + sparse rewards).
# --------------------------------------------------------------------------
@torch.no_grad()
def _flat_eval(sac, route, cfg, n_trials=128):
    env = DeviceVecRelay(n_trials, route)
    completed = torch.zeros(n_trials, dtype=torch.bool, device=DEVICE)
    for _ in range(env.max_steps):
        mean, _ = sac.policy.forward(env.state)
        a = sac.policy._rescale(torch.tanh(mean))
        _, _, term, _, _ = env.step(a)
        completed = completed | term
    return float(completed.float().mean().item())


def _train_flat(route, cfg):
    env = DeviceVecRelay(cfg["num_envs"], route)
    sac = SACTrainer(obs_dim=6, action_dim=2,
                     action_low=np.full(2, -1.0, dtype=np.float32),
                     action_high=np.full(2, 1.0, dtype=np.float32),
                     warmup_steps=cfg["num_envs"] * cfg["horizon"],
                     buffer=DeviceSACBuffer(capacity=200_000))
    total, mastered_at, last = 0, None, 0.0
    for it in range(1, cfg["flat_rollouts"] + 1):
        batch = collect_rollout(env, sac.device_policy_fn, cfg["horizon"])
        sac.train_on_rollout(batch, n_updates=cfg["flat_updates"])
        total += batch.total_steps
        if it % cfg["eval_every"] == 0:
            last = _flat_eval(sac, route, cfg)
            if last >= cfg["mastery"] and mastered_at is None:
                mastered_at = total
                break
    return (mastered_at if mastered_at is not None else total), \
        mastered_at is not None, last


# --------------------------------------------------------------------------
def _build_known_library(cfg, cache_dir=None):
    """Pre-learn the known-regime primitives (the agent's prior knowledge).
    These are seed-invariant prior knowledge, so they are cached and reused
    across seeds (only compose_reuse uses this; compose_no_library learns
    fresh). Returns (library:list, dev_steps)."""
    lib, dev = [], 0
    for r in KNOWN_REGIMES:
        path = os.path.join(cache_dir, f"prim_{r}.pt") if cache_dir else None
        if path and os.path.exists(path):
            pi = _fresh_skill()
            pi.load_state_dict(torch.load(path, weights_only=False))
            lib.append(pi)
            continue
        pi, steps, ok, fin = _learn_skill(r, cfg)
        if path:
            torch.save({k: v.detach().cpu() for k, v in pi.state_dict().items()},
                       path)
        lib.append(pi); dev += steps
    return lib, dev


def _run_compose(curriculum, cfg, prebuilt, cache_dir=None):
    """Walk the curriculum. prebuilt=True -> start with known primitives;
    else start empty (must learn). Returns log + library size."""
    if prebuilt:
        library, dev_steps = _build_known_library(cfg, cache_dir)
    else:
        library, dev_steps = [], 0
    log = []
    for ti, route in enumerate(curriculum):
        regime2pi, probe_steps, learn_steps, learns = _assign_primitives(
            library, route, cfg, allow_learn=True)
        succ = _compose_eval(regime2pi, route, cfg, cfg["n_trials"])
        cost = probe_steps + learn_steps          # composite-level training = 0
        log.append(dict(task=ti, regimes=[r for r, _ in route],
                        decision=("learn" if learns else "reuse"),
                        n_learned=len(learns), probe_steps=probe_steps,
                        learn_steps=learn_steps, cost=cost,
                        success=succ, mastered=succ >= cfg["mastery"]))
        tag = "learn" if learns else "reuse"
        print(f"    [{'compose_reuse' if prebuilt else 'no_library':>13}] "
              f"task {ti:>2} {('+'.join(r[:4] for r, _ in route)):<16} -> {tag:5s}"
              f" | +{cost:>8,} | compl {succ:.2f} | lib {len(library)}", flush=True)
    return log, len(library), dev_steps


def _summ(log, n_known):
    costs = [e["cost"] for e in log]
    cum = np.cumsum(costs).tolist()
    solved = sum(1 for e in log if e["mastered"])
    learns = sum(e["n_learned"] for e in log)
    known_costs = costs[:n_known]
    return dict(costs=costs, cumulative=cum, total=int(sum(costs)),
                solved=solved, n_tasks=len(log), primitives_learned=learns,
                known_block_mean=float(np.mean(known_costs)) if known_costs else 0.0,
                all_solved=all(e["mastered"] for e in log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--leg-len", type=int, default=3)
    p.add_argument("--n-known", type=int, default=12)
    p.add_argument("--n-novel", type=int, default=6)
    p.add_argument("--n-flat-routes", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--skill-rollouts", type=int, default=80)
    p.add_argument("--skill-updates", type=int, default=128)
    p.add_argument("--flat-rollouts", type=int, default=60)
    p.add_argument("--flat-updates", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--probe-trials", type=int, default=64)
    p.add_argument("--n-trials", type=int, default=128)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--consolidate", type=float, default=0.95)
    p.add_argument("--out-dir", default="compose_v4_out")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.num_envs, args.horizon = 1, 64, 32
        args.skill_rollouts, args.skill_updates = 14, 32
        args.flat_rollouts, args.flat_updates = 6, 32
        args.eval_every, args.eval_steps, args.probe_trials = 3, 40, 32
        args.n_known, args.n_novel, args.n_flat_routes, args.n_trials = 3, 2, 1, 32
        args.consolidate = 0.5

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "horizon", "skill_rollouts", "skill_updates",
            "flat_rollouts", "flat_updates", "eval_every", "eval_steps",
            "probe_trials", "n_trials", "mastery", "consolidate")}

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.json")
    print(f"[compose-v4] device={DEVICE} | known={KNOWN_REGIMES} novel={NOVEL_REGIME} "
          f"| L={args.leg_len} known/novel routes={args.n_known}/{args.n_novel} "
          f"| seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    if args.validate:
        torch.manual_seed(0); np.random.seed(0)
        lib, dev = _build_known_library(cfg)
        rng = np.random.default_rng(7)
        routes = [_make_route(rng, KNOWN_REGIMES, args.leg_len) for _ in range(6)]
        print(f"  built {len(lib)} primitives ({dev:,} env-steps). "
              f"composing on 6 novel routes:", flush=True)
        succs = []
        for r in routes:
            regime2pi, ps, ls, _ = _assign_primitives(lib, r, cfg, allow_learn=False)
            s = _compose_eval(regime2pi, r, cfg, 128)
            succs.append(s)
            print(f"    {'+'.join(x[:4] for x, _ in r):<18} compose-completion {s:.2f}",
                  flush=True)
        ok = np.mean(succs) >= args.mastery
        print(f"  mean compose-completion {np.mean(succs):.2f} -> "
              f"{'PRIMITIVES COMPOSE' if ok else 'CHECK: composition weak'}",
              flush=True)
        print(f"  {time.perf_counter()-t0:.0f}s", flush=True)
        return

    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = json.load(f).get("seeds", {})
        print(f"[resume] {len(done)} seed(s) done: {list(done)}", flush=True)

    for seed in range(args.seeds):
        if str(seed) in done:
            print(f"[seed {seed}] cached — skipping", flush=True); continue
        print(f"\n[seed {seed}]", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        curriculum = _make_curriculum(seed, args.n_known, args.n_novel, args.leg_len)

        prim_cache = os.path.join(args.out_dir, "prims")
        os.makedirs(prim_cache, exist_ok=True)
        log_cr, lib_cr, dev_cr = _run_compose(curriculum, cfg, prebuilt=True,
                                              cache_dir=prim_cache)
        log_nl, lib_nl, _ = _run_compose(curriculum, cfg, prebuilt=False)

        # flat baseline on a few of the (known-regime) routes, from scratch
        flat = []
        for ri in range(args.n_flat_routes):
            c, ok, fin = _train_flat(curriculum[ri], cfg)
            flat.append((c, ok, fin))
            print(f"    [   flat_scratch] route {ri} "
                  f"{'+'.join(x[:4] for x, _ in curriculum[ri]):<16} -> "
                  f"master={ok}@{c:,} compl {fin:.2f}", flush=True)

        sm_cr = _summ(log_cr, args.n_known); sm_cr["library_size"] = lib_cr
        sm_cr["dev_steps"] = dev_cr
        sm_nl = _summ(log_nl, args.n_known); sm_nl["library_size"] = lib_nl
        done[str(seed)] = dict(
            compose_reuse=dict(log=log_cr, **sm_cr),
            compose_no_library=dict(log=log_nl, **sm_nl),
            flat_scratch=[[int(c), bool(ok), float(fin)] for c, ok, fin in flat])
        with open(results_path, "w") as f:
            json.dump({"known": KNOWN_REGIMES, "novel": NOVEL_REGIME,
                       "L": args.leg_len, "n_known": args.n_known,
                       "n_novel": args.n_novel, "seeds": done}, f, indent=2)
        print(f"  [seed {seed}] compose_reuse solved {sm_cr['solved']}/{sm_cr['n_tasks']}"
              f" (prims learned {sm_cr['primitives_learned']}, lib {lib_cr}) "
              f"total {sm_cr['total']:,} | flat mastered "
              f"{sum(1 for _,ok,_ in flat if ok)}/{len(flat)}", flush=True)

    # ---- aggregate ----
    seeds = [done[str(s)] for s in range(args.seeds) if str(s) in done]
    if not seeds:
        return
    N = len(seeds); tval = _T95.get(N - 1, 2.0)

    def _ci(xs):
        m = float(np.mean(xs))
        if len(xs) < 2:
            return m, 0.0
        return m, tval * float(np.std(xs, ddof=1)) / (len(xs) ** 0.5)

    cr_solved = [s["compose_reuse"]["solved"] for s in seeds]
    cr_ntasks = seeds[0]["compose_reuse"]["n_tasks"]
    cr_prims = [s["compose_reuse"]["primitives_learned"] for s in seeds]
    cr_total = [s["compose_reuse"]["total"] for s in seeds]
    nl_total = [s["compose_no_library"]["total"] for s in seeds]
    cr_allsolved = [s["compose_reuse"]["all_solved"] for s in seeds]
    flat_master = [sum(1 for c in s["flat_scratch"] if c[1]) for s in seeds]
    flat_routes = len(seeds[0]["flat_scratch"])
    flat_compl = [np.mean([c[2] for c in s["flat_scratch"]]) for s in seeds]

    print(f"\n{'=' * 76}")
    print(f"  v4.0 PHASE 4 — compositional reuse | N={N} | L={seeds[0]['compose_reuse']['n_tasks']} tasks/seed")
    print(f"{'=' * 76}")
    ms, hs = _ci(cr_solved)
    print(f"  compose_reuse novel composites solved : {ms:.1f}/{cr_ntasks} +/- {hs:.1f}")
    mp, _ = _ci(cr_prims)
    print(f"  primitives LEARNED on the curriculum  : {cr_prims} (mean {mp:.1f}) "
          f"-- vs {cr_ntasks} composites solved => combinatorial leverage")
    print(f"  compose_reuse all-solved : {cr_allsolved}")
    mt, ht = _ci(cr_total); mn, hn = _ci(nl_total)
    print(f"  curriculum env-steps  compose_reuse {int(mt):,} +/- {int(ht):,} "
          f"| no_library {int(mn):,} +/- {int(hn):,}")
    print(f"  flat_scratch mastered {flat_master} of {flat_routes} routes/seed "
          f"| mean completion {[round(float(x),2) for x in flat_compl]}")

    leverage = mp > 0 and ms >= 0.8 * cr_ntasks and ms / max(mp, 1) >= 3.0
    flat_fails = np.mean(flat_master) < 0.5 * flat_routes
    decisive = leverage and flat_fails and all(cr_allsolved)
    if decisive:
        verdict = ("COMPOSITIONAL REUSE WORKS — a handful of learned primitives "
                   "compose into many NOVEL routes solved zero-shot at the "
                   "composite level; flat-scratch cannot. Few parts -> many "
                   "wholes (combinatorial), with a learn-the-missing-part branch.")
    elif leverage and flat_fails:
        verdict = "PARTIAL — leverage + flat fails, but some composites unmastered."
    elif leverage:
        verdict = "PARTIAL — combinatorial leverage holds but flat is competitive."
    else:
        verdict = "CHECK — no clear combinatorial composition advantage."
    print(f"\n  -> {verdict}")
    print(f"  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(results_path, "w") as f:
        json.dump({"known": KNOWN_REGIMES, "novel": NOVEL_REGIME,
                   "L": args.leg_len, "n_known": args.n_known,
                   "n_novel": args.n_novel, "seeds": done, "verdict": verdict}, f,
                  indent=2)
    print(f"  results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
