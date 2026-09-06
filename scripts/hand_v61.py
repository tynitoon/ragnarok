"""ARC 3 — the two hand-coded references (ARC3_PLAN.md 2.7). Ceilings, never arms of the claim.

    L   LawKnower   attributes + the TRUE tables (experimenter-side): executes the plan of
                    law_world.plan_for. The ceiling WITH the law. REACHABLE in the gate = L >= 0.85.
    G'  StoreSweeper the store, the tier law and the gate law, NO Bond/Out: builds what it knows, sweeps
                    untried equal-tier pairs cheapest-first, collects to refill. The ceiling WITHOUT the law.

Both expose act(obs, env, ...) / train_steps (no-op) / reset_optimizer (no-op) so run_goal_v61 scores
them on exactly the same criterion and schedule as the learned arms (the ARC 2 section-14 lesson).
"""

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from scripts.pair_store_v61 import MAX_ITEMS, PAIR_I, PAIR_J, PAIR_INDEX
from scripts.sweeper_v61 import sweeper_action


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
            # a collect entry is SATISFIED by state, not by delta: the plan still needs `need_after[k]` units of
            # that slot from entry k on; an incidental off-target collect must not stall the pointer (review)
            need_after = []
            for k, a in enumerate(seq):
                need_after.append(sum(1 for b in seq[k:] if b[0] == "c" and b[1] == a[1]) if a[0] == "c" else 0)
            need_after = torch.tensor(need_after, dtype=torch.long, device=DEVICE)
            self.plans[g] = (acts, watch, kind, need_after)
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
            acts, watch, kind, need_after = self.plans[g]
            L = acts.shape[0]
            p = self.ptr[rows].clamp(max=L - 1)
            # combine entries advance on the observed consumption of their first input; collect entries
            # advance while the inventory already holds what the rest of the plan needs of that slot
            w = watch[p]
            fell = inv[rows, w] < self.prev_inv[rows, w]
            p = (p + (~kind[p] & fell & ~fresh[rows]).long()).clamp(max=L - 1)
            for _ in range(L):
                w = watch[p]
                sat = kind[p] & (inv[rows, w] >= need_after[p]) & (p < L - 1)
                if not bool(sat.any()):
                    break
                p = (p + sat.long()).clamp(max=L - 1)
            self.ptr[rows] = p
            out[rows] = acts[p]
        self.prev_inv = inv.clone()
        self.prev_goal = env.goal.clone()
        return out


class StoreSweeper(_Ref):
    """G': the sweeper rule of scripts/sweeper_v61.py applied per env to the env's own store (pulled to
    the CPU each macro-step; the nav skill dominates the cost anyway). Same rule as the CPU model, so the
    gate's CONSISTENT check compares one policy under two executors."""

    def __init__(self, spec, seed=0):
        self.spec = spec
        self.n = spec["n_items"]
        self.tier = np.array(spec["tier"]); self.gate = np.array(spec["gate"], dtype=bool)
        self.raw_cell = np.array([c if c > 0 else 0 for c in spec["cell"]])
        self.rng = np.random.default_rng(seed)
        pi, pj = PAIR_I.cpu().numpy(), PAIR_J.cpu().numpy()
        self.pairs = list(zip(pi.tolist(), pj.tolist()))
        self.pidx = PAIR_INDEX.cpu().numpy()

    @torch.no_grad()
    def act(self, obs, env=None, **k):
        st, N, n = env.store, env.num_envs, self.n
        inv = env.base.inv.cpu().numpy()
        qleft = env.base.quota_left.cpu().numpy()                               # (N, n_cells)
        po = st.pair_out.cpu().numpy(); nonp = ((st.pair_fail + st.pair_inert) > 0).cpu().numpy()
        tried = st.pair_tried_ep.cpu().numpy()
        goal = env.goal.cpu().numpy()
        out = np.zeros(N, dtype=np.int64)
        for e in range(N):
            quota = np.where(self.tier == 1, qleft[e][self.raw_cell], 99)
            known, nonprod, tried_ep = {}, set(), set()
            for p, (i, j) in enumerate(self.pairs):
                if i >= n or j >= n:
                    continue
                if po[e, p] >= 0:
                    known[(i, j)] = int(po[e, p])
                if nonp[e, p]:
                    nonprod.add((i, j))
                if tried[e, p]:
                    tried_ep.add((i, j))
            a = sweeper_action(inv[e], quota, known, nonprod, tried_ep, self.tier, self.gate, int(goal[e]), self.rng)
            if a is None:
                out[e] = 0
            elif a[0] == "collect":
                out[e] = a[1]
            else:
                out[e] = MAX_ITEMS + int(self.pidx[a[1], a[2]])
        return torch.tensor(out, dtype=torch.long, device=DEVICE)
