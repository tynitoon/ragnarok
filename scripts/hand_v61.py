"""ARC 3 — the two hand-coded references (ARC3_PLAN.md 2.7). Ceilings, never arms of the claim.

    L   LawKnower   attributes + the TRUE tables (experimenter-side): executes the plan of
                    law_world.plan_for. The ceiling WITH the law. REACHABLE in the gate = L >= 0.85.
    G'  StoreSweeper the store, the tier law and the gate law, NO Bond/Out: builds what it knows, sweeps
                    untried equal-tier pairs cheapest-first, collects to refill. The ceiling WITHOUT the law.

Both expose act(obs, env, ...) / train_steps (no-op) / reset_optimizer (no-op) so run_goal_v61 scores
them on exactly the same criterion and schedule as the learned arms (the ARC 2 section-14 lesson).
"""

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import T_MAX
from scripts.pair_store_v61 import (MAX_ITEMS, N_PAIRS, PAIR_I, PAIR_J, PAIR_INDEX, action_valid)


class _Ref:
    def train_steps(self, *a, **k):
        return float("nan")

    def reset_optimizer(self):
        pass


class LawKnower(_Ref):
    """Stateful per-env executor of the scripted plan for the commanded goal."""

    def __init__(self, spec):
        self.spec = spec
        self.plans = {}
        for g, p in spec["plans"].items():
            if not p["ok"]:
                continue
            seq = ([("c", s) for s in p["first_free"]] + [("x", i, j) for (i, j, _) in p["first_pairs"]]
                   + [("c", s) for s in p["collect_gated"]] + [("c", s) for s in p["collect_free"]]
                   + [("x", i, j) for (i, j, _) in p["pairs"]])
            acts = torch.tensor([a[1] if a[0] == "c" else MAX_ITEMS + int(PAIR_INDEX[a[1], a[2]])
                                 for a in seq], dtype=torch.long, device=DEVICE)
            watch = torch.tensor([a[1] for a in seq], dtype=torch.long, device=DEVICE)   # slot whose inv changes
            kind = torch.tensor([a[0] == "c" for a in seq], dtype=torch.bool, device=DEVICE)
            self.plans[g] = (acts, watch, kind)
        self.ptr = self.prev_inv = None

    @torch.no_grad()
    def act(self, obs, env=None, **k):
        N, ar = env.num_envs, torch.arange(env.num_envs, device=DEVICE)
        inv = env.base.inv
        if self.ptr is None or self.ptr.shape[0] != N:
            self.ptr = torch.zeros(N, dtype=torch.long, device=DEVICE)
            self.prev_inv = inv.clone()
            self.prev_goal = env.goal.clone()
        fresh = (env.msteps == 0) | (env.goal != self.prev_goal)
        self.ptr[fresh] = 0
        self.prev_inv[fresh] = 0
        out = torch.zeros(N, dtype=torch.long, device=DEVICE)
        for g in env.goal.unique().tolist():
            rows = (env.goal == g).nonzero(as_tuple=True)[0]
            if g not in self.plans:
                out[rows] = 0
                continue
            acts, watch, kind = self.plans[g]
            L = acts.shape[0]
            p = self.ptr[rows].clamp(max=L - 1)
            # advance where the previous action's effect is visible: collect -> inv[watch] rose;
            # combine -> inv[watch] fell (consumed)
            w = watch[p]
            rose = inv[rows, w] > self.prev_inv[rows, w]
            fell = inv[rows, w] < self.prev_inv[rows, w]
            adv = torch.where(kind[p], rose, fell) & ~fresh[rows]
            p = (p + adv.long()).clamp(max=L - 1)
            self.ptr[rows] = p
            out[rows] = acts[p]
        self.prev_inv = inv.clone()
        self.prev_goal = env.goal.clone()
        return out


class StoreSweeper(_Ref):
    """Stateless greedy policy over the store: goal if known and buildable > known recipe of a needed item
    > needed raw > untried equal-tier pair among held items, lowest tier first > collect to enable pairs."""

    def __init__(self, spec):
        self.tier = torch.zeros(MAX_ITEMS, dtype=torch.long, device=DEVICE)
        self.tier[:spec["n_items"]] = torch.tensor(spec["tier"], device=DEVICE)
        self.gate = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.gate[:spec["n_items"]] = torch.tensor([bool(g) for g in spec["gate"]], device=DEVICE)

    @torch.no_grad()
    def act(self, obs, env=None, **k):
        st, N = env.store, env.num_envs
        held = env.held()
        valid = action_valid(obs)
        known_recipe, need, yields = st.derived(held, env.goal)
        rnd = torch.rand(N, MAX_ITEMS + N_PAIRS, device=DEVICE)
        score = torch.where(valid, rnd, torch.full_like(rnd, -1e9))
        # diagonal: raws
        has_tool = (held & (self.tier >= 2).unsqueeze(0)).any(-1, keepdim=True)
        gate_ok = ~self.gate.unsqueeze(0) | has_tool
        d_valid = valid[:, :MAX_ITEMS] & gate_ok
        sd = score[:, :MAX_ITEMS]
        sd = torch.where(d_valid & held, 20 + rnd[:, :MAX_ITEMS], sd)
        sd = torch.where(d_valid & ~held, 30 + rnd[:, :MAX_ITEMS], sd)
        sd = torch.where(d_valid & need & ~held, 50 + rnd[:, :MAX_ITEMS], sd)
        # pairs
        pv = valid[:, MAX_ITEMS:]
        untried = (st.pair_out < 0) & (st.pair_fail == 0) & (st.pair_inert == 0) & ~st.pair_tried_ep
        ti, tj = self.tier[PAIR_I], self.tier[PAIR_J]
        equal = (ti == tj) & (ti < T_MAX)
        sp = score[:, MAX_ITEMS:]
        sp = torch.where(pv & untried & equal.unsqueeze(0), 40 - ti.float().unsqueeze(0) + rnd[:, MAX_ITEMS:], sp)
        sp = torch.where(pv & yields, 60 + rnd[:, MAX_ITEMS:], sp)
        goal_pair = st.pair_out == env.goal.unsqueeze(1)
        sp = torch.where(pv & goal_pair, torch.full_like(sp, 100.0), sp)
        return torch.cat([sd, sp], -1).argmax(-1)
