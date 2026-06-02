"""v49 — DEPTH-SCALING crossover: where does a FAIR strong-flat break but composition hold?

The reuse boundary: reuse pays only when knowledge is EXPENSIVE-TO-REDERIVE and the task is TOO HARD
to learn flat. v48 tested ONE depth (6) on the FIXED craft_world and found flat=0.54 (not collapsed),
amortised over ONE target -> only a modest edge. v49 is the untested corner (NOT a v48 rehash):
  - PROCEDURAL tech-trees, SWEEP target depth (n_items controls it).
  - Library = ONE tree-agnostic nav-collect skill, AMORTISED over MANY targets (every item is a target).
  - Manager DISCOVERS the composition order via PPO (NOT handed the DAG).
  - FLAT = strong PPO + per-achievement shaping + high-entropy exploration, MATCHED primitive-step compute.
Hypothesis: there is a depth D* where fair flat collapses (<=0.2) while composition holds (>=0.8). If
flat keeps pace at all reachable depths -> clean NULL, the reuse boundary is complete.

Usage: python -m scripts.depth_scaling_v49 [--depths 3 5 7] [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.tech_tree import gen_tree, DeviceVecTechTree


class TechTreeConvNet(nn.Module):
    """Actor-critic treating the egocentric grid as a SPATIAL map (CNN). obs = ego(P*P*C) + tail.
    broadcast_tail=True (nav skill): the target-type one-hot is broadcast as C' constant channels and
    stacked onto the ego, so the conv can MATCH each cell against the target spatially (a plain FC
    concat lets the conv features drown out the target -> the skill ignores it and reaches ~1/n_types)."""
    def __init__(self, P, C, tail_dim, action_dim, hidden=128, broadcast_tail=False):
        super().__init__()
        self.P, self.C, self.split, self.tail_dim = P, C, P * P * C, tail_dim
        self.broadcast = broadcast_tail
        in_ch = C + (tail_dim if broadcast_tail else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        with torch.no_grad():
            d = self.conv(torch.zeros(1, in_ch, P, P)).reshape(1, -1).shape[1]
        fc_in = d + (0 if broadcast_tail else tail_dim)
        self.fc = nn.Sequential(nn.Linear(fc_in, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs):
        B = obs.shape[0]
        ego = obs[:, :self.split].reshape(B, self.P, self.P, self.C).permute(0, 3, 1, 2)
        tail = obs[:, self.split:]
        if self.broadcast:
            tch = tail.reshape(B, self.tail_dim, 1, 1).expand(-1, -1, self.P, self.P)
            h = self.conv(torch.cat([ego, tch], 1)).reshape(B, -1)
        else:
            h = torch.cat([self.conv(ego).reshape(B, -1), tail], -1)
        z = self.fc(h)
        return self.actor(z), self.critic(z).squeeze(-1)

N_ITEMS_FOR_DEPTH = {3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16, 9: 18, 10: 20, 12: 26}
MAX_CELLS = 24                      # fixed cell-type width so the nav skill obs dim is constant
# NOTE: grid=view (full observability) so the difficulty is COMPOSITION DEPTH, not partial-obs
# navigation — the nav skill can always SEE its target cell-type and learn to reach it.


def make_spec(seed, depth_level, n_base_res=2):
    """Generate a tree with the requested n_items; return spec + its true target depth."""
    spec = gen_tree(seed, n_items=N_ITEMS_FOR_DEPTH[depth_level], n_base_res=n_base_res)
    return spec, spec["depth"][spec["target"]]


# ---------------------------------------------------------------- nav-collect skill (the library)
def train_nav_skill(spec, cfg, seed, log=False):
    """ONE tree-agnostic skill: navigate to & collect ANY requested resource cell-type.
    Trained with dense distance shaping (nav_goal='random'). Returns (policy, primitive_steps)."""
    torch.manual_seed(seed)
    grant = [1] * spec["n_items"]          # grant all items so tool-gating never blocks navigation training
    env = DeviceVecTechTree(cfg["num_envs"], spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["nav_max_steps"], n_resource=cfg["n_resource"],
                            nav_goal="random", max_cells=MAX_CELLS, grant=grant, seed=seed)
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, env.action_dim, broadcast_tail=True)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, net=net, entropy=cfg["entropy"],
                      gamma=0.99, lam=0.95)
    steps = 0
    for it in range(1, cfg["skill_iters"] + 1):
        ppo.train_iter(env, cfg["rollout"])
        steps += cfg["num_envs"] * cfg["rollout"]
        if log and (it % max(1, cfg["skill_iters"] // 8) == 0 or it == cfg["skill_iters"]):
            print(f"    [nav skill] it {it:>4} | success {nav_success(ppo, spec, cfg, seed):.2f} "
                  f"| {steps/1e6:.2f}M steps", flush=True)
    return ppo, steps


@torch.no_grad()
def nav_success(ppo, spec, cfg, seed, n=256):
    env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"], max_steps=cfg["nav_max_steps"],
                            n_resource=cfg["n_resource"], nav_goal="random", max_cells=MAX_CELLS,
                            grant=[1] * spec["n_items"], seed=seed + 1)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["nav_max_steps"]):
        obs, r, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        got |= term
    return float(got.float().mean())


# ---------------------------------------------------------------- manager over the tree (compose arm)
class TreeManagerEnv:
    """Semi-MDP: manager picks an ITEM to pursue. Resource item -> run the nav-collect skill until
    collected; craft item -> emit its craft action. Obs = [inv, unlocked] (symbolic). The manager is
    NOT given the DAG; it must discover the order from the per-achievement reward."""
    def __init__(self, num_envs, spec, skill, cfg, seed=0):
        self.base = DeviceVecTechTree(num_envs, spec, grid=cfg["grid"], view=cfg["view"],
                                      max_steps=10 ** 9, n_resource=cfg["n_resource"],
                                      max_cells=MAX_CELLS, seed=seed)
        self.spec, self.skill, self.cfg = spec, skill, cfg
        self.num_envs = num_envs
        self.n_items = spec["n_items"]
        self.P, self.ego_dim = cfg["view"], cfg["view"] * cfg["view"] * MAX_CELLS
        self.action_dim = self.n_items                      # pursue item i
        self.obs_dim = 2 * self.n_items
        self.option_timeout = cfg["option_timeout"]
        self.macro_budget = cfg["macro_budget"]
        self.target = spec["target"]
        # precompute per-item lookup tensors (vectorised option dispatch)
        craft_idx = {it: 5 + k for k, it in enumerate(spec["craft_actions"])}
        self.item_is_craft = torch.tensor([spec["kind"][i] == "C" for i in range(self.n_items)],
                                          device=DEVICE)
        self.item_cell = torch.tensor([spec["cell"][i] if spec["kind"][i] == "R" else 0
                                       for i in range(self.n_items)], device=DEVICE)
        self.item_craft_act = torch.tensor([craft_idx.get(i, 0) for i in range(self.n_items)],
                                           device=DEVICE)
        self._prim = 0
        self.reset()

    def _set_state(self):
        inv = self.base.inv.float().clamp(max=5.0) / 5.0
        self.state = torch.cat([inv, self.base.unlocked.float()], -1)

    def reset(self):
        self.base.reset()
        self.msteps = torch.zeros(self.num_envs, device=DEVICE)
        self._set_state()
        return self.state

    def _skill_obs(self, target_cell):
        """Build the nav-skill obs from the base: egocentric(max_cells) + target-cell-type one-hot."""
        ego = self.base.state[:, :self.ego_dim]
        g = F.one_hot(torch.full((self.num_envs,), target_cell, device=DEVICE), MAX_CELLS).float()
        return torch.cat([ego, g], -1)

    def step(self, g):
        g = g.reshape(self.num_envs)
        N = self.num_envs
        ar = torch.arange(N, device=DEVICE)
        rew = torch.zeros(N, device=DEVICE)
        is_craft = self.item_is_craft[g]                    # (N,) bool — vectorised dispatch
        cell_of = self.item_cell[g]                         # (N,) target cell-type (0 for crafts)
        craft_act = self.item_craft_act[g]                  # (N,) craft action (0 for resources)
        watched = g                                         # collect raises inv[item]; craft -> unlocked
        start = self.base.inv[ar, watched].float()
        done_opt = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        for t in range(self.option_timeout):
            ego = self.base.state[:, :self.ego_dim]
            goh = F.one_hot(cell_of, MAX_CELLS).float()
            a_skill = self.skill.act(torch.cat([ego, goh], -1), deterministic=True)   # nav skill, all envs
            a = torch.where(is_craft, craft_act, a_skill)   # craft envs override with craft action
            _, r, _, _, _ = self.base.step(a)
            self._prim += N
            rew += r * (~done_opt).float()
            cur = self.base.inv[ar, watched].float()
            got_craft = is_craft & self.base.unlocked[ar, watched]
            done_opt = done_opt | got_craft | (~is_craft & (cur >= start + 1))
            if t % 8 == 7 and bool(done_opt.all()):         # sync only every 8 steps (perf)
                break
        self.msteps += 1
        truncated = self.msteps >= self.macro_budget
        if bool(truncated.any()):
            self.base._reset_done(truncated)
            self.msteps = torch.where(truncated, torch.zeros_like(self.msteps), self.msteps)
        self._set_state()
        return self.state, rew, torch.zeros_like(truncated), truncated, truncated


def train_manager(spec, skill, cfg, seed):
    torch.manual_seed(seed + 100)
    env = TreeManagerEnv(cfg["num_envs"], spec, skill, cfg, seed=seed + 100)
    mgr = DiscretePPO(env.obs_dim, env.action_dim, hidden=cfg["mgr_hidden"], entropy=cfg["mgr_entropy"],
                      gamma=0.99, lam=0.95)
    env._prim = 0
    for _ in range(cfg["mgr_iters"]):
        mgr.train_iter(env, cfg["macro_budget"])
    return mgr, env._prim


@torch.no_grad()
def eval_target(policy, spec, cfg, seed, kind, skill=None, n=512):
    """Fraction of envs that UNLOCK the deepest target item. kind='flat' or 'compose'."""
    target = spec["target"]
    if kind == "flat":
        env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"], max_steps=cfg["flat_eval_steps"],
                                n_resource=cfg["n_resource"], max_cells=MAX_CELLS, seed=seed + 5)
        unlocked = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        obs = env.state
        for _ in range(cfg["flat_eval_steps"]):
            obs, _, _, _, _ = env.step(policy.act(obs, deterministic=True))
            unlocked |= env.base.unlocked[:, target] if hasattr(env, "base") else env.unlocked[:, target]
        return float(unlocked.float().mean())
    else:
        env = TreeManagerEnv(n, spec, skill, cfg, seed=seed + 5)
        unlocked = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        obs = env.state
        for _ in range(cfg["macro_budget"]):
            obs, _, _, _, _ = env.step(policy.act(obs, deterministic=True))
            unlocked |= env.base.unlocked[:, target]
        return float(unlocked.float().mean())


def train_flat(spec, cfg, budget_steps, seed):
    """Strong flat PPO: per-achievement novelty reward (normal mode) + high entropy, to matched budget."""
    torch.manual_seed(seed + 200)
    env = DeviceVecTechTree(cfg["num_envs"], spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["flat_max_steps"], n_resource=cfg["n_resource"],
                            max_cells=MAX_CELLS, seed=seed + 200)
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, spec["n_items"], env.action_dim)   # same CNN class (fair)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, net=net, entropy=cfg["flat_entropy"],
                      gamma=0.99, lam=0.95)
    steps = 0
    while steps < budget_steps:
        ppo.train_iter(env, cfg["rollout"])
        steps += cfg["num_envs"] * cfg["rollout"]
    return ppo, steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)      # 2*grid-1 => FULL observability from any cell
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--skill-iters", type=int, default=300)
    p.add_argument("--mgr-hidden", type=int, default=128)
    p.add_argument("--mgr-entropy", type=float, default=0.03)
    p.add_argument("--mgr-iters", type=int, default=200)
    p.add_argument("--macro-budget", type=int, default=30)
    p.add_argument("--option-timeout", type=int, default=25)
    p.add_argument("--flat-entropy", type=float, default=0.03)
    p.add_argument("--flat-max-steps", type=int, default=200)
    p.add_argument("--flat-eval-steps", type=int, default=200)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--nav-only", action="store_true", help="train+report the nav skill only (viability)")
    args = p.parse_args()
    if args.smoke:
        args.depths, args.seeds, args.num_envs = [3], [0], 64
        args.skill_iters, args.mgr_iters = 150, 60
        args.option_timeout, args.macro_budget = 18, 14

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "n_resource", "rollout", "hidden", "entropy",
            "nav_max_steps", "skill_iters", "mgr_hidden", "mgr_entropy", "mgr_iters",
            "macro_budget", "option_timeout", "flat_entropy", "flat_max_steps", "flat_eval_steps")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v49 depth-scaling] device={DEVICE} | depths {args.depths} seeds {args.seeds} | "
          f"compose (nav-skill+manager, order DISCOVERED) vs STRONG flat, MATCHED primitive steps", flush=True)
    t0 = time.perf_counter()
    if args.nav_only:
        for depth in args.depths:
            spec, true_depth = make_spec(0, depth)
            print(f"  depth~{depth} (true {true_depth}, n_items {spec['n_items']}, "
                  f"n_cells {spec['n_cells']}): training nav skill...", flush=True)
            train_nav_skill(spec, cfg, 0, log=True)
        return
    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            spec, true_depth = make_spec(seed, depth)
            skill, sk_steps = train_nav_skill(spec, cfg, seed)
            nav_s = nav_success(skill, spec, cfg, seed)
            mgr, mgr_steps = train_manager(spec, skill, cfg, seed)
            compose_total = sk_steps + mgr_steps
            flat, flat_steps = train_flat(spec, cfg, compose_total, seed)
            comp = eval_target(mgr, spec, cfg, seed, "compose", skill=skill)
            flt = eval_target(flat, spec, cfg, seed, "flat")
            row = dict(depth_level=depth, true_depth=int(true_depth), seed=seed,
                       n_items=spec["n_items"], nav_skill_success=round(nav_s, 3),
                       skill_steps=sk_steps, mgr_steps=mgr_steps, compose_steps=compose_total,
                       flat_steps=flat_steps, compose=round(comp, 3), flat=round(flt, 3))
            rows.append(row)
            print(f"  depth~{depth} (true {true_depth}) seed {seed} | nav-skill {nav_s:.2f} | "
                  f"COMPOSE {comp:.2f} vs FLAT {flt:.2f} | steps {compose_total/1e6:.1f}M | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    # crossover: a depth where flat<=0.2 AND compose>=0.8 across all its seeds
    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth_level"], []).append(r)
    crossover = None
    for d in sorted(by_depth):
        rs = by_depth[d]
        if len(rs) >= 1 and all(x["flat"] <= 0.2 for x in rs) and all(x["compose"] >= 0.8 for x in rs):
            crossover = d
            break
    positive = crossover is not None
    verdict = (
        f"DEPTH-SCALING CROSSOVER FOUND at depth~{crossover} — a FAIR strong flat collapses (<=0.2) "
        f"while composition (amortised library + DISCOVERED order) holds (>=0.8). Reuse is NECESSARY, "
        f"not merely modestly better, for deep targets. REVIEW before reporting."
        if positive else
        f"NO CROSSOVER (clean NULL) — flat tracks composition at all tested depths {args.depths}; even "
        f"deep procedural targets are within reach of a fair, shaped, well-explored flat. The reuse "
        f"boundary is complete on this substrate.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v49_depth_scaling.json"), "w") as f:
        json.dump(dict(rows=rows, crossover=crossover, positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
