"""v51 — TRANSFERABLE ROUTER (meta-manager): a learned controller that picks WHICH item/skill to
pursue, from TREE-AGNOSTIC observable features, trained across the childhood distribution so it
TRANSFERS to held-out trees. ("une zone qui commande quelle zone utiliser", learned + transferable.)

v50 showed the childhood SKILL transfers (nav 0.94) but the per-tree MANAGER is unreliable (masters
held-out trees only ~1/3 — PPO-over-options on a single tree gets stuck) AND tree-specific (not
reusable). v51 fixes both: the router observes, per item, ONLY observable affordances — [in_inv,
unlocked, craftable_now, collectable_now, is_goal, is_resource, is_valid] — and scores each item with a
SHARED per-item MLP (permutation-invariant). No tree-specific indices, no granted DAG (craftable_now is
the current action affordance, derivable by trying). Trained on a DISTRIBUTION of trees -> learns a
GENERAL compose strategy that (a) is robust (more data escapes local optima) and (b) TRANSFERS zero-shot.

If a router trained on childhood trees masters HELD-OUT trees zero-shot (reusing the nav skill), the
agent's WHOLE policy (perception skill + composition) transfers -> adulthood is near-free -> dramatic
amortisation. That is the developmental thesis with real teeth.

Usage: python -m scripts.meta_manager_v51 [--smoke]
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
from scripts.depth_scaling_v49 import MAX_CELLS, N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood, nav_success_on, NAV_ACTIONS

MAX_ITEMS = 28
N_FEAT = 7        # [in_inv, unlocked, craftable_now, collectable_now, is_goal, is_resource, is_valid]


class PerItemRouter(nn.Module):
    """Shared per-item scorer -> masked logits over items + pooled value. Permutation-invariant,
    tree-agnostic: the SAME net applies to any tree (items described by observable features only)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.score.weight, gain=0.01)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.reshape(B, MAX_ITEMS, N_FEAT)
        emb = self.enc(x)
        logits = self.score(emb).squeeze(-1)            # (B, MAX_ITEMS)
        logits = logits.masked_fill(x[..., 6] < 0.5, -1e9)   # mask invalid items (is_valid feature)
        value = self.value(emb.mean(1)).squeeze(-1)
        return logits, value


class RouterEnv:
    """Semi-MDP for the router: per-item observable features -> pick an item to pursue (resource -> run
    the nav skill until collected; craft -> emit the craft action). Reuses TreeManagerEnv's options."""
    def __init__(self, num_envs, spec, skill, cfg, seed=0):
        self.base = DeviceVecTechTree(num_envs, spec, grid=cfg["grid"], view=cfg["view"],
                                      max_steps=10 ** 9, n_resource=cfg["n_resource"],
                                      max_cells=MAX_CELLS, seed=seed)
        self.spec, self.skill, self.cfg, self.num_envs = spec, skill, cfg, num_envs
        self.n_items = spec["n_items"]
        self.P, self.ego_dim = cfg["view"], cfg["view"] * cfg["view"] * MAX_CELLS
        self.action_dim, self.obs_dim = MAX_ITEMS, MAX_ITEMS * N_FEAT
        self.option_timeout, self.macro_budget = cfg["option_timeout"], cfg["macro_budget"]
        self.target = spec["target"]
        # per-item static tensors
        self.item_is_craft = torch.tensor([spec["kind"][i] == "C" for i in range(self.n_items)], device=DEVICE)
        self.item_cell = torch.tensor([spec["cell"][i] if spec["kind"][i] == "R" else 0
                                       for i in range(self.n_items)], device=DEVICE)
        craft_idx = {it: 5 + k for k, it in enumerate(spec["craft_actions"])}
        self.item_craft_act = torch.tensor([craft_idx.get(i, 0) for i in range(self.n_items)], device=DEVICE)
        self.craft_out_idx = torch.tensor(spec["craft_actions"], device=DEVICE)   # item idx of each craft
        self.res_tool = torch.tensor([spec["tool"][i] for i in range(self.n_items)], device=DEVICE)
        self.is_res = self.item_is_craft == False  # noqa: E712
        self._prim = 0
        self.reset()

    def _set_state(self):
        N, n = self.num_envs, self.n_items
        inv, unlocked = self.base.inv, self.base.unlocked
        f = torch.zeros(N, MAX_ITEMS, N_FEAT, device=DEVICE)
        f[:, :n, 0] = (inv > 0).float()
        f[:, :n, 1] = unlocked.float()
        # craftable_now per craft item: inputs met AND tools present
        ci, ct = self.base.craft_in, self.base.craft_tool            # (ncraft,n_items),(ncraft,n_items)
        inputs_ok = (inv.unsqueeze(1) >= ci.unsqueeze(0)).all(-1)     # (N,ncraft)
        tools_ok = ((inv.unsqueeze(1) >= 1) | ~ct.unsqueeze(0)).all(-1)
        f[:, self.craft_out_idx, 2] = (inputs_ok & tools_ok).float()
        # collectable_now per resource item: tool present (or none)
        rt = self.res_tool[:n]
        has_tool = (rt < 0) | (inv.gather(1, rt.clamp(min=0).expand(N, n)) >= 1)
        f[:, :n, 3] = torch.where(self.is_res[:n].unsqueeze(0), has_tool.float(), torch.zeros(N, n, device=DEVICE))
        f[:, self.target, 4] = 1.0
        f[:, :n, 5] = self.is_res[:n].float().unsqueeze(0)
        f[:, :n, 6] = 1.0
        self.state = f.reshape(N, -1)

    def reset(self):
        self.base.reset(); self.msteps = torch.zeros(self.num_envs, device=DEVICE); self._set_state()
        return self.state

    def step(self, g):
        g = g.reshape(self.num_envs).clamp(max=self.n_items - 1)
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        rew = torch.zeros(N, device=DEVICE)
        is_craft = self.item_is_craft[g]; cell_of = self.item_cell[g]; craft_act = self.item_craft_act[g]
        start = self.base.inv[ar, g].float()
        done_opt = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        for t in range(self.option_timeout):
            ego = self.base.state[:, :self.ego_dim]
            goh = F.one_hot(cell_of, MAX_CELLS).float()
            a_skill = self.skill.act(torch.cat([ego, goh], -1), deterministic=True)
            a = torch.where(is_craft, craft_act, a_skill)
            _, r, _, _, _ = self.base.step(a); self._prim += N
            rew += r * (~done_opt).float()
            cur = self.base.inv[ar, g].float()
            got_craft = is_craft & self.base.unlocked[ar, g]
            done_opt = done_opt | got_craft | (~is_craft & (cur >= start + 1))
            if t % 8 == 7 and bool(done_opt.all()):
                break
        self.msteps += 1
        trunc = self.msteps >= self.macro_budget
        if bool(trunc.any()):
            self.base._reset_done(trunc)
            self.msteps = torch.where(trunc, torch.zeros_like(self.msteps), self.msteps)
        self._set_state()
        return self.state, rew, torch.zeros_like(trunc), trunc, trunc


def train_router(specs, skill, cfg, seed, router=None, iters=None):
    """Train a router over a DISTRIBUTION of trees (rotated). If router given, continue training it."""
    torch.manual_seed(seed + 300)
    envs = [RouterEnv(cfg["num_envs"], s, skill, cfg, seed=seed + 300 + i) for i, s in enumerate(specs)]
    net = PerItemRouter()
    ppo = DiscretePPO(envs[0].obs_dim, MAX_ITEMS, net=net, entropy=cfg["mgr_entropy"], gamma=0.99, lam=0.95)
    if router is not None:
        ppo.net.load_state_dict(router.net.state_dict())
    steps = 0
    for it in range(1, (iters or cfg["router_iters"]) + 1):
        env = envs[it % len(envs)]
        ppo.train_iter(env, cfg["macro_budget"]); steps += cfg["num_envs"] * cfg["macro_budget"]
    return ppo, steps


@torch.no_grad()
def master_rate(ppo, spec, skill, cfg, seed, n=256):
    env = RouterEnv(n, spec, skill, cfg, seed=seed + 9)
    unlocked = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
        unlocked |= env.base.unlocked[:, spec["target"]]
    return float(unlocked.float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--n-train-trees", type=int, default=8)
    p.add_argument("--n-heldout", type=int, default=6)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--view", type=int, default=13)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--nav-max-steps", type=int, default=40)
    p.add_argument("--skill-iters", type=int, default=350)
    p.add_argument("--router-iters", type=int, default=400)
    p.add_argument("--mgr-entropy", type=float, default=0.03)
    p.add_argument("--macro-budget", type=int, default=18)
    p.add_argument("--option-timeout", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n_train_trees, args.n_heldout, args.skill_iters, args.router_iters = 4, 3, 150, 150

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "n_resource", "rollout", "entropy", "nav_max_steps",
            "skill_iters", "router_iters", "mgr_entropy", "macro_budget", "option_timeout")}
    os.makedirs(args.out_dir, exist_ok=True)
    ni = N_ITEMS_FOR_DEPTH[args.depth]
    train_specs = [gen_tree(1000 + i, n_items=ni) for i in range(args.n_train_trees)]
    heldout = [gen_tree(5000 + i, n_items=ni) for i in range(args.n_heldout)]
    print(f"[v51 router] device={DEVICE} | depth~{args.depth} | {args.n_train_trees} train / "
          f"{args.n_heldout} held-out | childhood: nav skill + TRANSFERABLE router", flush=True)
    t0 = time.perf_counter()
    skill, c_skill = train_childhood(train_specs, cfg, args.seed)
    router, c_router = train_router(train_specs, skill, cfg, args.seed)
    c_lib = c_skill + c_router
    tr_master = [round(master_rate(router, s, skill, cfg, args.seed), 3) for s in train_specs]
    ho_master = [round(master_rate(router, s, skill, cfg, args.seed), 3) for s in heldout]
    print(f"  childhood lib {c_lib/1e6:.2f}M (skill {c_skill/1e6:.2f} + router {c_router/1e6:.2f}) | "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  router master on TRAIN trees:    {tr_master} (mean {sum(tr_master)/len(tr_master):.2f})", flush=True)
    print(f"  router master on HELD-OUT trees: {ho_master} (mean {sum(ho_master)/len(ho_master):.2f}) "
          f"[ZERO-SHOT]", flush=True)
    zsh = sum(ho_master) / len(ho_master)
    ok = zsh >= 0.6
    verdict = (
        f"TRANSFERABLE ROUTER (v51) — a router trained on {args.n_train_trees} childhood trees masters "
        f"HELD-OUT trees ZERO-SHOT at {zsh:.0%} (reusing the nav skill). The agent's WHOLE policy "
        f"(skill+composition) transfers -> adulthood near-free -> dramatic amortisation. REVIEW + firm up."
        if ok else
        f"PARTIAL — router zero-shot held-out master {zsh:.0%} (train {sum(tr_master)/len(tr_master):.0%}). "
        f"Composition does not transfer cleanly; tune router_iters/features/diversity.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"v51_router_s{args.seed}.json"), "w") as f:
        json.dump(dict(depth=args.depth, c_lib=c_lib, train_master=tr_master, heldout_master=ho_master,
                       zeroshot=zsh, ok=ok, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
