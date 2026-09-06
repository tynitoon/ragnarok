"""ARC 3 — PairStore + PairEnv: the agent's per-world memory over raws and PAIRS, and the option executor
over the ACTION MATRIX (ARC3_PLAN.md 2.4-2.5).

Observation row (uint8-quantised in the buffers, single scale):
    slot block  28 slots x 32 floats
        base 8   in_inv, unlocked, tried_ep, succ_last, is_goal, is_resource, is_valid, quota_frac
        attrs 16 element one-hot (14), tier/4, gate/2            <- disclosed labels, never the law
        store 8  n_succ_log, n_fail_log, obtained_ever, since_succ_log, frac_minctx_held, ctx_new,
                 known_recipe, needed
    pair block  378 unordered pairs (i<j over 28 slots) x 5 floats
                 pair_succ_log, pair_nonprod_log (botch+inert), pair_tried_ep, pair_succ_ep,
                 pair_yields_needed_unheld
Actions: a < 28 -> "pursue raw a" (the frozen nav skill runs, one unit per option); a >= 28 -> combine the
pair PAIR_I[a-28], PAIR_J[a-28] once.

Nothing here reads spec["pair_out"] except LawVecTechTree.combine (the physics), and the store is written
ONLY from the outcome the env returns — the same bit the agent observes. test_law_v61.py (1)-(4) assert
that an unexercised edit of the true tables leaves store and observation bit-identical.
"""

import torch
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import (LawVecTechTree, NOOP, OUT_BOTCH, OUT_INERT, OUT_PRODUCT,
                                              OUT_INVALID, T_MAX)
from scripts.depth_scaling_v49 import MAX_CELLS

MAX_ITEMS = 28
K_EL = 14
N_BASE, N_ATTR, N_STORE = 8, 16, 8
N_SLOT = N_BASE + N_ATTR + N_STORE                    # 32
N_PAIRS = MAX_ITEMS * (MAX_ITEMS - 1) // 2            # 378
N_PF = 5
ROW = MAX_ITEMS * N_SLOT + N_PAIRS * N_PF             # 2786
N_ACT = MAX_ITEMS + N_PAIRS                           # 406
LOG_CAP = 6.0
GOAL_COL = 4
# slot feature indices
F_INV, F_UNL, F_TRIED, F_SUCC, F_GOAL, F_RES, F_VALID, F_QUOTA = range(8)
F_EL, F_TIER, F_GATE = 8, 22, 23
F_NSUCC, F_NFAIL, F_OBT, F_SINCE, F_MINCTX, F_CTXNEW, F_KNOWN, F_NEED = range(24, 32)
# pair feature indices
PF_SUCC, PF_NONPROD, PF_TRIED, PF_SUCCEP, PF_YIELDS = range(5)

_I, _J = zip(*[(i, j) for i in range(MAX_ITEMS) for j in range(i + 1, MAX_ITEMS)])
PAIR_I = torch.tensor(_I, dtype=torch.long, device=DEVICE)
PAIR_J = torch.tensor(_J, dtype=torch.long, device=DEVICE)
PAIR_INDEX = torch.full((MAX_ITEMS, MAX_ITEMS), -1, dtype=torch.long, device=DEVICE)
PAIR_INDEX[PAIR_I, PAIR_J] = torch.arange(N_PAIRS, device=DEVICE)
PAIR_INDEX[PAIR_J, PAIR_I] = torch.arange(N_PAIRS, device=DEVICE)


def _log(x):
    return (x.clamp(min=0).add(1).log() / LOG_CAP).clamp(max=1)


class PairStore:
    """Per-parallel-env memory. Raw part = the ARC 2 rule (successes intersect the gate-context
    candidates, failures only count). Pair part = exact three-outcome facts."""

    def __init__(self, num_envs, n_items):
        self.N, self.n = num_envs, n_items
        z = lambda *s: torch.zeros(*s, device=DEVICE)                            # noqa: E731
        self.n_succ, self.n_fail, self.since_succ, self.obtained_ever = (z(self.N, MAX_ITEMS) for _ in range(4))
        self.C = torch.zeros(self.N, MAX_ITEMS, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.C[:, :n_items, :n_items] = True
        self.C[:, torch.arange(n_items), torch.arange(n_items)] = False
        self.last_ctx = torch.zeros(self.N, MAX_ITEMS, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.ctx_new = torch.ones(self.N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.pair_succ, self.pair_inert, self.pair_fail = (z(self.N, N_PAIRS) for _ in range(3))
        self.pair_out = torch.full((self.N, N_PAIRS), -1, dtype=torch.long, device=DEVICE)
        self.pair_tried_ep = torch.zeros(self.N, N_PAIRS, dtype=torch.bool, device=DEVICE)
        self.pair_succ_ep = torch.zeros(self.N, N_PAIRS, dtype=torch.bool, device=DEVICE)

    KEYS = ("n_succ", "n_fail", "since_succ", "obtained_ever", "C", "last_ctx", "ctx_new",
            "pair_succ", "pair_inert", "pair_fail", "pair_out", "pair_tried_ep", "pair_succ_ep")

    def write_raw(self, g, held, obtained, mask):
        """g (N,) slot | held (N,28) bool before the option | obtained (N,) bool | mask (N,) which envs."""
        rows = mask.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        g, held, ob = g[rows], held[rows], obtained[rows]
        self.n_succ[rows, g] += ob.float()
        self.n_fail[rows, g] += (~ob).float()
        self.since_succ[rows, g] = torch.where(ob, torch.zeros_like(g, dtype=torch.float),
                                               self.since_succ[rows, g] + 1)
        self.obtained_ever[rows, g] = torch.maximum(self.obtained_ever[rows, g], ob.float())
        ok = ob.nonzero(as_tuple=True)[0]
        if ok.numel():
            self.C[rows[ok], g[ok]] &= held[ok]
        self.last_ctx[rows, g] = held
        self.ctx_new[rows, g] = False

    def write_pair(self, p, outcome, product, mask):
        """p (N,) pair index | outcome (N,) code | product (N,) product slot or -1 | mask (N,)."""
        rows = (mask & (outcome != OUT_INVALID)).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        p, o, pr = p[rows], outcome[rows], product[rows]
        self.pair_succ[rows, p] += (o == OUT_PRODUCT).float()
        self.pair_inert[rows, p] += (o == OUT_INERT).float()
        self.pair_fail[rows, p] += (o == OUT_BOTCH).float()
        prod = o == OUT_PRODUCT
        self.pair_out[rows, p] = torch.where(prod, pr, self.pair_out[rows, p])
        self.pair_tried_ep[rows, p] = True
        self.pair_succ_ep[rows, p] |= prod

    def note_inventory_change(self, held):
        self.ctx_new |= (self.last_ctx != held.unsqueeze(1)).any(-1)

    def reset_episode(self, mask):
        self.pair_tried_ep[mask] = False
        self.pair_succ_ep[mask] = False

    def derived(self, held, goal):
        """known_recipe (N,28), needed (N,28), yields_needed_unheld (N,378) — closure over KNOWN recipes."""
        N = self.N
        known = self.pair_out >= 0                                            # (N,378)
        M = torch.zeros(N, MAX_ITEMS, MAX_ITEMS, dtype=torch.bool, device=DEVICE)   # M[n,k,i]: k needs i
        nn_, pp = known.nonzero(as_tuple=True)
        if nn_.numel():
            k = self.pair_out[nn_, pp]
            M[nn_, k, PAIR_I[pp]] = True
            M[nn_, k, PAIR_J[pp]] = True
        known_recipe = M.any(-1)                                              # (N,28)
        need = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        need[torch.arange(N, device=DEVICE), goal] = True
        for _ in range(T_MAX):
            frontier = need & ~held
            need = need | (frontier.unsqueeze(-1) & M).any(1)
        po = self.pair_out.clamp(min=0)
        yields = known & need.gather(1, po) & ~held.gather(1, po)
        return known_recipe, need, yields

    def features(self, held, goal):
        """store block (N,28,8) and pair block (N,378,5), all in [0,1]."""
        c_missing = (self.C & ~held.unsqueeze(1)).sum(-1).float()
        c_size = self.C.sum(-1).float().clamp(min=1)
        known_recipe, need, yields = self.derived(held, goal)
        slot = torch.stack([_log(self.n_succ), _log(self.n_fail), self.obtained_ever, _log(self.since_succ),
                            1.0 - c_missing / c_size, self.ctx_new.float(), known_recipe.float(),
                            need.float()], -1)
        pair = torch.stack([_log(self.pair_succ), _log(self.pair_inert + self.pair_fail),
                            self.pair_tried_ep.float(), self.pair_succ_ep.float(), yields.float()], -1)
        return slot, pair

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in self.KEYS}

    def load_state_dict(self, st):
        for k, v in st.items():
            getattr(self, k).copy_(v)


class PairEnv:
    """Semi-MDP over the action matrix on a LawVecTechTree, with a PairStore. One env per WORLD: the store
    persists across episodes and across the goals of the stream; buffers and weights are the caller's."""

    def __init__(self, num_envs, spec, skill, cfg, seed=0, goal=None):
        self.spec, self.skill, self.cfg, self.num_envs = spec, skill, cfg, num_envs
        self.n_items, n = spec["n_items"], spec["n_items"]
        self.base = LawVecTechTree(num_envs, spec, grid=cfg["grid"], view=cfg["view"],
                                   n_resource=cfg["n_resource"], max_cells=MAX_CELLS, seed=seed)
        self.ego_dim = cfg["view"] * cfg["view"] * MAX_CELLS
        self.option_timeout, self.macro_budget = cfg["option_timeout"], cfg["macro_budget"]
        self.is_res = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.is_res[:n] = torch.tensor([k == "R" for k in spec["kind"]], device=DEVICE)
        self.valid = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE); self.valid[:n] = True
        self.item_cell = torch.zeros(MAX_ITEMS, dtype=torch.long, device=DEVICE)
        self.item_cell[:n] = torch.tensor([c if c > 0 else 0 for c in spec["cell"]], device=DEVICE)
        attrs = torch.zeros(MAX_ITEMS, N_ATTR, device=DEVICE)
        for i in range(n):
            attrs[i, spec["el"][i]] = 1.0
            attrs[i, F_TIER - F_EL] = spec["tier"][i] / T_MAX
            attrs[i, F_GATE - F_EL] = spec["gate"][i] / 2.0
        self.attrs = attrs
        self.store = PairStore(num_envs, n)
        self.tried = torch.zeros(num_envs, MAX_ITEMS, device=DEVICE)
        self.succ = torch.zeros(num_envs, MAX_ITEMS, device=DEVICE)
        self.msteps = torch.zeros(num_envs, device=DEVICE)
        self.last_trunc = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
        self.goal = torch.full((num_envs,), int(spec["target"] if goal is None else goal),
                               dtype=torch.long, device=DEVICE)
        self.msteps_total = self._prim = self._att = 0
        self.post_unlocked = self.base.unlocked.clone()

    # ---- state -----------------------------------------------------------------------------------
    def reset(self):
        self.base.reset()
        self.msteps.zero_(); self.tried.zero_(); self.succ.zero_()
        self.store.reset_episode(torch.ones(self.num_envs, dtype=torch.bool, device=DEVICE))
        self.last_trunc.zero_()
        self.post_unlocked = self.base.unlocked.clone()
        return self.obs()

    def set_goal(self, g):
        self.goal = (torch.full((self.num_envs,), int(g), dtype=torch.long, device=DEVICE)
                     if isinstance(g, int) else g.to(DEVICE).long())
        return self.obs()

    def held(self):
        h = torch.zeros(self.num_envs, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        h[:, :self.n_items] = self.base.inv > 0
        return h

    def obs(self):
        N, n, ar = self.num_envs, self.n_items, torch.arange(self.num_envs, device=DEVICE)
        held = self.held()
        x = torch.zeros(N, MAX_ITEMS, N_SLOT, device=DEVICE)
        x[:, :, F_INV] = held.float()
        x[:, :n, F_UNL] = self.base.unlocked.float()
        x[:, :, F_TRIED] = self.tried
        x[:, :, F_SUCC] = self.succ
        x[ar, self.goal, F_GOAL] = 1.0
        x[:, :, F_RES] = self.is_res.float()
        x[:, :, F_VALID] = self.valid.float()
        q = self.base.quota_left.float() / self.base.quota                  # (N, n_cells)
        x[:, :, F_QUOTA] = torch.where(self.is_res.unsqueeze(0), q[:, self.item_cell], torch.zeros(N, MAX_ITEMS, device=DEVICE))
        x[:, :, F_EL:F_EL + N_ATTR] = self.attrs.unsqueeze(0)
        slot, pair = self.store.features(held, self.goal)
        x[:, :, F_NSUCC:] = slot
        x = x * self.valid.view(1, MAX_ITEMS, 1).float()
        return torch.cat([x.reshape(N, -1), pair.reshape(N, -1)], -1)

    # ---- one macro-step --------------------------------------------------------------------------
    def step(self, a):
        """a (N,) long in [0, 406). Returns (obs, info) with info = dict(is_pair, outcome, product_el)."""
        a = a.reshape(self.num_envs).long()
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        is_pair = a >= MAX_ITEMS
        g = torch.where(is_pair, torch.zeros_like(a), a).clamp(max=MAX_ITEMS - 1)
        p = torch.where(is_pair, a - MAX_ITEMS, torch.zeros_like(a)).clamp(max=N_PAIRS - 1)
        held_before = self.held()
        gi = g.clamp(max=self.n_items - 1)
        start = self.base.inv[ar, gi].float()
        pursue = ~is_pair & self.is_res[g]
        # combine executes once, before the batched option loop (invalid for non-pair envs: i == j)
        ci = torch.where(is_pair, PAIR_I[p], torch.zeros_like(p))
        cj = torch.where(is_pair, PAIR_J[p], torch.zeros_like(p))
        outcome, product = self.base.combine(ci, cj)
        outcome = torch.where(is_pair, outcome, torch.full_like(outcome, OUT_INVALID))
        self._prim += N
        # the nav option for pursuing envs; NOOP once the unit is in hand, NOOP throughout for the others
        done_opt = ~pursue
        det = not self.cfg.get("skill_stochastic", False)
        goh = F.one_hot(self.item_cell[g], MAX_CELLS).float()
        if bool(pursue.any()):
            for t in range(self.option_timeout):
                ego = self.base.state[:, :self.ego_dim]
                a_skill = self.skill.act(torch.cat([ego, goh], -1), deterministic=det)
                prim = torch.where(done_opt, torch.full_like(a_skill, NOOP), a_skill)
                self.base.step(prim)
                self._prim += N
                done_opt = done_opt | (self.base.inv[ar, gi].float() >= start + 1)
                if t % 8 == 7 and bool(done_opt.all()):
                    break
        obtained = pursue & (self.base.inv[ar, gi].float() >= start + 1)
        # ---- own-history and store writes, BEFORE any truncation reset -----------------------------
        self.tried[ar, g] = torch.where(~is_pair, torch.ones_like(self.tried[ar, g]), self.tried[ar, g])
        self.succ[ar, g] = torch.where(~is_pair, obtained.float(), self.succ[ar, g])
        self.store.write_raw(g, held_before, obtained, pursue)
        self.store.write_pair(p, outcome, product, is_pair)
        self.store.note_inventory_change(self.held())
        self._att += N
        self.post_unlocked = self.base.unlocked.clone()
        self.msteps += 1
        self.msteps_total += 1
        trunc = self.msteps >= self.macro_budget
        self.last_trunc = trunc
        if bool(trunc.any()):
            self.base._reset_done(trunc)
            self.msteps = torch.where(trunc, torch.zeros_like(self.msteps), self.msteps)
            self.tried[trunc] = 0.0
            self.succ[trunc] = 0.0
            self.store.reset_episode(trunc)
        prod_el = torch.where(product >= 0, self.base.el_t[product.clamp(min=0)], torch.full_like(product, -1))
        return self.obs(), dict(is_pair=is_pair, outcome=outcome, product_el=prod_el, trunc=trunc)


# ---------------------------------------------------------------- masks computed from the OBSERVATION only

def action_valid(obs):
    """(B, 406) bool. Diagonal: a valid resource with quota. Pair: both held, both valid."""
    B = obs.shape[0]
    x = obs[:, :MAX_ITEMS * N_SLOT].reshape(B, MAX_ITEMS, N_SLOT)
    diag = (x[..., F_RES] > 0.5) & (x[..., F_VALID] > 0.5) & (x[..., F_QUOTA] > 0)
    held = x[..., F_INV] > 0.5
    valid = x[..., F_VALID] > 0.5
    pair = held[:, PAIR_I] & held[:, PAIR_J] & valid[:, PAIR_I] & valid[:, PAIR_J]
    return torch.cat([diag, pair], -1)


def eval_mask(obs):
    """The context-aware instrument mask for deterministic action (ARC3_PLAN 2.6), identical for every arm:
    True = allowed. Diagonal masked iff tried this episode, failed, and the context has not changed since.
    Pair masked iff it botched or was inert before, or was tried this episode without producing."""
    B = obs.shape[0]
    x = obs[:, :MAX_ITEMS * N_SLOT].reshape(B, MAX_ITEMS, N_SLOT)
    pf = obs[:, MAX_ITEMS * N_SLOT:].reshape(B, N_PAIRS, N_PF)
    diag_bad = (x[..., F_TRIED] > 0.5) & (x[..., F_SUCC] < 0.5) & (x[..., F_CTXNEW] < 0.5)
    pair_bad = (pf[..., PF_NONPROD] > 0) | ((pf[..., PF_TRIED] > 0.5) & (pf[..., PF_SUCCEP] < 0.5))
    return action_valid(obs) & ~torch.cat([diag_bad, pair_bad], -1)
