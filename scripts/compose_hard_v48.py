"""v48 — Compositional reuse reaches a HARD target where flat RL cannot, FAIRLY.
FROZEN per preregistration.md entry 2026-06-01.

Closes the M4/M5/v7 rigour gaps the v45 reckoning exposed:
 - composition ORDER is LEARNED (manager PPO over run-until-achieved options), not scripted;
 - flat baseline is RE-RUN at MATCHED total primitive-step budget (not cited from M2);
 - >=3 seeds; no eval-granting (end-to-end from EMPTY inventory).

H: at matched compute + matched reward signal, the hierarchical (learn-skills + learn-manager)
agent completes depth-6 make_iron_pickaxe end-to-end (>=0.8) while flat PPO+achievement-rewards
given the SAME total budget stays <=0.2, every seed.

Honest scope: tests TEMPORAL ABSTRACTION + skill reuse. The option set = the env's achievements,
given to BOTH arms via the reward signal; LEARNED = navigation skills + composition order; craft
actions are env primitives. Autonomous decomposition (v7) and novel-composite generalisation are
v48b/c follow-ups IF this holds.

Usage: python -m scripts.compose_hard_v48 [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ACH, A_IPICK, ACH_NAMES, TREE, STONE, COAL, IRON)
from scripts.craft_endtoend_v6 import _train_collect, _skill_success, COLLECT, _grant
from scripts.craft_manager_v6 import ManagerEnv, _eval_manager

NAMES = {TREE: "wood", STONE: "stone", COAL: "coal", IRON: "iron"}


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---- reuse arm (counted) ---------------------------------------------------
def train_collect_counted(cell, cfg):
    """Train one goal-conditioned collect skill; count TRAINING primitive steps
    (periodic success-eval steps excluded, as for the flat arm's eval)."""
    goal, grant_idx = COLLECT[cell]
    grant = _grant(grant_idx)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant)
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["entropy"])
    steps, succ = 0, 0.0
    for it in range(1, cfg["skill_cap"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        steps += cfg["rollout"] * cfg["num_envs"]
        if it % 10 == 0:
            succ = _skill_success(ppo, goal, grant, cfg)
            if succ >= 0.9:
                break
    return ppo, max(succ, _skill_success(ppo, goal, grant, cfg)), steps


def train_manager_counted(skills, cfg):
    """Train the manager (PPO over run-until-achieved options); count the actual
    PRIMITIVE env steps consumed (options early-stop, so it is data-dependent)."""
    env = ManagerEnv(cfg["num_envs"], skills, K=cfg["K"], macro_budget=cfg["macro_budget"],
                     option_timeout=cfg["option_timeout"], collect_target=cfg["collect_target"],
                     grid=cfg["grid"], view=cfg["view"])
    env._prim = 0
    _orig = env.base.step
    def _cstep(a):                                   # count every primitive base step
        env._prim += env.base.num_envs
        return _orig(a)
    env.base.step = _cstep
    mgr = DiscretePPO(env.obs_dim, N_ACH, hidden=128, entropy=cfg["mgr_entropy"],
                      gamma=0.99, lam=0.95)
    for _ in range(cfg["mgr_iters"]):
        mgr.train_iter(env, cfg["macro_budget"])
    return mgr, env._prim


# ---- flat baseline (counted, matched budget) -------------------------------
def train_flat_counted(cfg, budget_steps):
    """Flat PPO on the FULL env (goal=None -> +1 per first-time achievement = the
    per-item novelty/curiosity signal), high entropy, run to the matched budget."""
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["flat_max_steps"])
    ppo = DiscretePPO(env.obs_dim, 10, hidden=cfg["hidden"], entropy=cfg["flat_entropy"])
    steps = 0
    while steps < budget_steps:
        ppo.train_iter(env, cfg["rollout"])
        steps += cfg["rollout"] * cfg["num_envs"]
    return ppo, steps


@torch.no_grad()
def eval_flat(ppo, cfg, n=512):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["eval_budget"] + 1)
    unlocked = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["eval_budget"]):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
        unlocked |= env.unlocked
    return float(unlocked.float().mean(0)[A_IPICK]), unlocked.float().mean(0).cpu()


# ---- one seed --------------------------------------------------------------
def run_seed(seed, cfg):
    seed_all(seed)
    skills, succ, rsteps = {}, {}, 0
    for cell in (TREE, STONE, COAL, IRON):
        ppo, s, st = train_collect_counted(cell, cfg)
        skills[cell] = ppo
        succ[NAMES[cell]] = round(s, 2)
        rsteps += st
    mgr, msteps = train_manager_counted(skills, cfg)
    rsteps += msteps
    prof, _ = _eval_manager(mgr, skills, cfg, n=512, random_nav=False)
    reuse_ipick = float(prof[A_IPICK])

    flat_ppo, fsteps = train_flat_counted(cfg, rsteps)          # MATCHED budget
    flat_ipick, flat_prof = eval_flat(flat_ppo, cfg)

    return dict(seed=seed, skill_success=succ,
                reuse_steps=rsteps, reuse_ipick=round(reuse_ipick, 3),
                reuse_profile=[round(float(x), 2) for x in prof],
                flat_steps=fsteps, flat_ipick=round(flat_ipick, 3),
                flat_profile=[round(float(x), 2) for x in flat_prof])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)       # skill episode length
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)      # skill PPO
    p.add_argument("--skill-cap", type=int, default=70)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--option-timeout", type=int, default=40)   # run-until-achieved (M6/M7)
    p.add_argument("--collect-target", type=int, default=2)    # quantity-aware (M7)
    p.add_argument("--macro-budget", type=int, default=24)
    p.add_argument("--mgr-iters", type=int, default=150)
    p.add_argument("--mgr-entropy", type=float, default=0.03)
    p.add_argument("--flat-max-steps", type=int, default=400)  # generous (flat's best shot)
    p.add_argument("--flat-entropy", type=float, default=0.05) # flat's best shot
    p.add_argument("--eval-budget", type=int, default=1000)     # >= reuse eval horizon (24*40)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        (args.seeds, args.num_envs, args.skill_cap, args.mgr_iters,
         args.macro_budget, args.option_timeout) = [0], 64, 10, 12, 12, 25

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden", "entropy",
            "skill_cap", "K", "option_timeout", "collect_target", "macro_budget",
            "mgr_iters", "mgr_entropy", "flat_max_steps", "flat_entropy", "eval_budget")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v48] device={DEVICE} | compositional reuse vs flat @ MATCHED primitive-step "
          f"budget | end-to-end iron_pickaxe from EMPTY inv | seeds={args.seeds}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        r = run_seed(s, cfg)
        rows.append(r)
        print(f"  seed {s}: skills {r['skill_success']} | REUSE iron_pickaxe "
              f"{r['reuse_ipick']} @ {r['reuse_steps']:,} steps | FLAT {r['flat_ipick']} "
              f"@ {r['flat_steps']:,} steps | {time.perf_counter()-t0:.0f}s", flush=True)

    positive = (len(rows) >= 3 and all(r["reuse_ipick"] >= 0.8 for r in rows)
                and all(r["flat_ipick"] <= 0.2 for r in rows))
    verdict = (
        "COMPOSITIONAL REUSE REACHES THE HARD TARGET (fair) — at MATCHED compute, the agent "
        "composes its LEARNED skills to build depth-6 iron_pickaxe end-to-end from empty "
        f"inventory (reuse {[r['reuse_ipick'] for r in rows]}) while flat PPO+curiosity at the "
        f"same budget cannot ({[r['flat_ipick'] for r in rows]}), every seed. REVIEW before reporting."
        if positive else
        f"NULL/CHECK — reuse {[r['reuse_ipick'] for r in rows]} vs flat {[r['flat_ipick'] for r in rows]}. "
        f"If flat cracks it or reuse is unreliable, hierarchy gives no fair advantage here -> honest.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v48_compose_hard.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, cfg=cfg, rows=rows, positive=positive,
                       verdict=verdict, ach_names=ACH_NAMES), f, indent=2)


if __name__ == "__main__":
    main()
