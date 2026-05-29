"""v6.0 M5 — LEARNED composition: a high-level manager discovers the order.

Closes M4's "scripted plan" gap. The manager (PPO) acts over MACRO-steps:
its action picks which of the 9 achievement nodes to pursue; a macro-step
runs that node's behaviour for K low-level steps (a learned collect skill
for collect nodes; the craft action for craft nodes). Manager obs is the
symbolic state [inventory, unlocked-achievements] (18-d) — it never sees the
grid; the skills handle navigation. Macro-horizon ~24 << ~450 primitive
steps, so the manager learns the dependency ORDER where flat PPO cannot.

Arms: manager+learned-skills, manager+random-nav (ablation), flat (M2 ref
0.11). Decisive: the learned manager autonomously masters make_iron_pickaxe
(>> 0.11) via a DISCOVERED order that respects the DAG.

Usage: python -m scripts.craft_manager_v6 [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, N_ACH, N_ITEMS, TREE, STONE, COAL, IRON,
    A_WOOD, A_TABLE, A_WPICK, A_STONE, A_COAL, A_SPICK, A_FURNACE, A_IRON, A_IPICK)
from scripts.craft_endtoend_v6 import _train_collect, _goal_onehot, COLLECT

# manager action (achievement index) -> behaviour
ACH_BEHAVIOR = {
    A_WOOD: ("collect", TREE), A_TABLE: ("craft", 5), A_WPICK: ("craft", 6),
    A_STONE: ("collect", STONE), A_COAL: ("collect", COAL),
    A_SPICK: ("craft", 7), A_FURNACE: ("craft", 8),
    A_IRON: ("collect", IRON), A_IPICK: ("craft", 9),
}
COLLECT_GOAL = {TREE: A_WOOD, STONE: A_STONE, COAL: A_COAL, IRON: A_IRON}


def _low_action(base, g_vec, skills, random_nav=False):
    """Per-env low-level action given each env's pursued node g_vec."""
    N = base.num_envs
    a = torch.zeros(N, dtype=torch.long, device=DEVICE)
    obs = base.state
    for ach, (kind, val) in ACH_BEHAVIOR.items():
        mask = g_vec == ach
        if not bool(mask.any()):
            continue
        if kind == "craft":
            a[mask] = val
        elif random_nav:
            a[mask] = torch.randint(0, 5, (int(mask.sum()),), device=DEVICE)
        else:
            goal = COLLECT_GOAL[val]
            obs_g = torch.cat([obs[mask], _goal_onehot(goal, int(mask.sum()))], -1)
            a[mask] = skills[val].act(obs_g, deterministic=True)
    return a


class ManagerEnv:
    """Wraps a CraftWorld as a semi-MDP for the manager: obs = symbolic
    [inv, unlocked]; step(macro_action) runs K low-level steps."""
    obs_dim = 2 * N_ITEMS if N_ITEMS == N_ACH else N_ITEMS + N_ACH
    action_dim = N_ACH

    def __init__(self, num_envs, skills, K=20, macro_budget=24, random_nav=False,
                 grid=9, view=5):
        self.base = DeviceVecCraftWorld(num_envs, grid=grid, view=view,
                                        max_steps=10 ** 9)   # no auto-truncation
        self.num_envs = num_envs
        self.skills = skills
        self.K = K
        self.macro_budget = macro_budget
        self.random_nav = random_nav
        self.obs_dim = N_ITEMS + N_ACH
        self.reset()

    def _set_state(self):
        inv = self.base.inv.float().clamp(max=5.0) / 5.0
        self.state = torch.cat([inv, self.base.unlocked.float()], dim=-1)

    def reset(self):
        self.base.reset()
        self.msteps = torch.zeros(self.num_envs, device=DEVICE)
        self._set_state()
        return self.state

    def step(self, g):
        g = g.reshape(self.num_envs)
        rew = torch.zeros(self.num_envs, device=DEVICE)
        for _ in range(self.K):
            a = _low_action(self.base, g, self.skills, self.random_nav)
            _, r, _, _, _ = self.base.step(a)
            rew += r
        self.msteps += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        truncated = self.msteps >= self.macro_budget
        done = truncated
        if bool(done.any()):
            self.base._reset_done(done)
            self.msteps = torch.where(done, torch.zeros_like(self.msteps), self.msteps)
        self._set_state()
        return self.state, rew, terminated, truncated, done


@torch.no_grad()
def _eval_manager(mgr, skills, cfg, n=512, random_nav=False):
    """Fraction unlocking each achievement within one macro-episode; also the
    modal discovered macro-action sequence."""
    env = ManagerEnv(n, skills, K=cfg["K"], macro_budget=cfg["macro_budget"],
                     random_nav=random_nav)
    unlocked = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
    seq = []
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        g = mgr.act(obs, deterministic=True)
        seq.append(int(torch.mode(g).values.item()))
        obs, _, _, _, _ = env.step(g)
        unlocked |= env.base.unlocked
    return unlocked.float().mean(0).cpu(), seq


def _train_manager(skills, cfg, random_nav=False):
    env = ManagerEnv(cfg["num_envs"], skills, K=cfg["K"],
                     macro_budget=cfg["macro_budget"], random_nav=random_nav)
    mgr = DiscretePPO(env.obs_dim, N_ACH, hidden=128, entropy=cfg["entropy"],
                      gamma=0.99, lam=0.95)
    for it in range(1, cfg["mgr_iters"] + 1):
        mgr.train_iter(env, cfg["macro_budget"])
        if it % cfg["eval_every"] == 0:
            prof, _ = _eval_manager(mgr, skills, cfg, n=256, random_nav=random_nav)
            print(f"    [mgr{'(rnd)' if random_nav else ''}] it {it:>3} | "
                  f"iron_pickaxe {prof[A_IPICK]:.2f} | total-ach "
                  f"{prof.sum():.2f}", flush=True)
    return mgr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.03)
    p.add_argument("--skill-cap", type=int, default=70)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--macro-budget", type=int, default=24)
    p.add_argument("--mgr-iters", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.skill_cap, args.mgr_iters = 64, 10, 12
        args.eval_every, args.macro_budget, args.K = 4, 12, 10

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap", "K", "macro_budget", "eval_every", "mgr_iters")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[craft-manager-v6] device={DEVICE}", flush=True)
    t0 = time.perf_counter()

    print("\n[skills] training the 4 reusable collect skills...", flush=True)
    skills = {}
    for cell in (TREE, STONE, COAL, IRON):
        ppo, s = _train_collect(cell, cfg)
        skills[cell] = ppo
        print(f"  collect skill (cell {cell}) success {s:.2f}", flush=True)

    print("\n[M5a] manager + LEARNED skills (learns the order)...", flush=True)
    mgr = _train_manager(skills, cfg, random_nav=False)
    prof, seq = _eval_manager(mgr, skills, cfg, n=512, random_nav=False)

    print("\n[M5b] ablation: manager + RANDOM-nav skills...", flush=True)
    mgr_r = _train_manager(skills, cfg, random_nav=True)
    prof_r, seq_r = _eval_manager(mgr_r, skills, cfg, n=512, random_nav=True)

    print(f"\n  {'achievement':20s} {'mgr+skills':>12} {'mgr+random':>12}")
    for i, nm in enumerate(ACH_NAMES):
        print(f"  {nm:20s} {prof[i]:>12.2f} {prof_r[i]:>12.2f}", flush=True)
    e2e = float(prof[A_IPICK]); e2e_r = float(prof_r[A_IPICK])
    seq_names = [ACH_NAMES[s] for s in seq]
    # is the discovered order DAG-valid? (iron before iron_pickaxe in the seq)
    def _first(seq, ach):
        return seq.index(ach) if ach in seq else 1e9
    dag_ok = (_first(seq, A_WOOD) < _first(seq, A_TABLE) < _first(seq, A_WPICK)
              and _first(seq, A_IRON) < _first(seq, A_IPICK)
              and _first(seq, A_FURNACE) < _first(seq, A_IPICK))
    print(f"\n  discovered macro-order (modal): {seq_names}")
    print(f"  order respects DAG (wood<table<wpick, iron&furnace<ipick): {dag_ok}")
    print(f"  END-TO-END iron_pickaxe: mgr+LEARNED {e2e:.2f} | mgr+RANDOM {e2e_r:.2f}"
          f" | flat PPO 0.11")
    ok = e2e >= 0.8 and dag_ok
    verdict = ("LEARNED COMPOSITION WORKS — the manager DISCOVERS a DAG-valid "
               "order over its skills and autonomously masters iron_pickaxe, "
               "far beyond flat PPO. The plan is learned, not scripted."
               if ok else "CHECK — see table / discovered order.")
    print(f"  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "m5.json"), "w") as f:
        json.dump(dict(profile_learned=prof.tolist(), profile_random=prof_r.tolist(),
                       iron_pickaxe=e2e, iron_pickaxe_random=e2e_r,
                       discovered_order=seq_names, dag_ok=bool(dag_ok),
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
