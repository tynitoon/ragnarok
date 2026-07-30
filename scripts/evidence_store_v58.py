"""ARC 2, Step 1 — EvidenceStore: the portable agent's per-world fast memory.

THE ARCHITECTURE (arbitrated by the arc-2 design workflow, before any code):
  slow WEIGHTS  = the skill of learning a world — identity-free, trained across worlds, FROZEN at test;
  fast STORE    = everything specific to THIS world — written by a FIXED rule from the agent's OWN
                  attempt outcomes, wiped on world entry. No item-identity parameter ever crosses a
                  world boundary, so the policy is portable BY CONSTRUCTION.

The store's write rule exploits one family invariant (disclosed in the prereg as an absorbable prior):
every recipe input has multiplicity exactly 1 (tech_tree.py gen_tree: ins = {x: 1, ...}). Hence, when a
craft of item i SUCCEEDS while the agent held exactly the item-set H, the true requirement set of i is a
subset of H — so intersecting success contexts, S[i] &= H, converges monotonically onto a superset of
the true recipe that shrinks with every diverse success. Once every member of S[i] is held, crafting i
is GUARANTEED to succeed (S[i] ⊇ true_req and multiplicity-1). The same intersection on successful
collects narrows a resource's tool-gate candidates C[i].

DELIBERATE DEVIATION from the design draft, for soundness: the draft eliminated gate candidates on
FAILED collects (C[i] &= ~H). That rule is unsound here — a collect can fail because the STOCHASTIC nav
skill timed out (~10% per attempt) even when the true gate tool IS held, and a candidate mask only
shrinks, so one unlucky nav timeout would permanently eliminate the true gate. Failures therefore only
increment counters (which the policy reads as "probably gated / try after inventory changes"); only
SUCCESSES — which are exact evidence — shrink candidate sets.

Everything here is unit-tested by scripts/test_evidence_v58.py:
  leak test      — editing a recipe the agent never observed leaves the store BIT-IDENTICAL
                   (the mechanical exclusion of the affordance-oracle defect class);
  equivariance   — permuting slots permutes the features identically (no hidden index preference);
  persistence    — the store survives episode truncation, resets only on world entry.
"""

import torch
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from scripts.depth_scaling_v49 import MAX_CELLS
from scripts.meta_manager_v51 import MAX_ITEMS, N_FEAT
from scripts.hidden_recipe_v55 import HiddenEnv

N_EVID = 10          # evidence features per slot (frozen list below)
LOG_CAP = 6.0        # log-bucket cap for counters


class EvidenceStore:
    """Per-parallel-env, slot-indexed evidence over one world. O(1) write per macro-attempt."""

    def __init__(self, num_envs, n_items):
        self.N, self.n = num_envs, n_items
        z = lambda *s: torch.zeros(*s, device=DEVICE)                    # noqa: E731
        self.n_succ = z(self.N, MAX_ITEMS)
        self.n_fail = z(self.N, MAX_ITEMS)
        self.since_succ = z(self.N, MAX_ITEMS)          # attempts since last success of this slot
        self.obtained_ever = z(self.N, MAX_ITEMS)
        # candidate masks: True = "still possibly required". Valid slots only; never self.
        self.S = torch.zeros(self.N, MAX_ITEMS, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.S[:, :n_items, :n_items] = True
        self.S[:, torch.arange(n_items), torch.arange(n_items)] = False
        self.C = self.S.clone()                          # resource tool-gate candidates
        self.last_ctx = torch.zeros(self.N, MAX_ITEMS, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        self.ctx_new = torch.ones(self.N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)

    def write(self, g, held, obtained, is_res):
        """g (N,) attempted slot | held (N,MAX_ITEMS) bool inv>0 BEFORE the attempt | obtained (N,) bool
        | is_res (N,) bool. Successes intersect candidate sets (exact); failures only count."""
        ar = torch.arange(self.N, device=DEVICE)
        self.n_succ[ar, g] += obtained.float()
        self.n_fail[ar, g] += (~obtained).float()
        self.since_succ[ar, g] = torch.where(obtained, torch.zeros_like(g, dtype=torch.float),
                                             self.since_succ[ar, g] + 1)
        self.obtained_ever[ar, g] = torch.maximum(self.obtained_ever[ar, g], obtained.float())
        okc = obtained & ~is_res
        okr = obtained & is_res
        if bool(okc.any()):
            rows = okc.nonzero(as_tuple=True)[0]
            self.S[rows, g[rows]] &= held[rows]
        if bool(okr.any()):
            rows = okr.nonzero(as_tuple=True)[0]
            self.C[rows, g[rows]] &= held[rows]
        self.last_ctx[ar, g] = held
        self.ctx_new[ar, g] = False

    def note_inventory_change(self, held):
        """Mark slots whose attempt context has changed (the policy's cue to re-experiment)."""
        self.ctx_new |= (self.last_ctx != held.unsqueeze(1)).any(-1)

    def features(self, held, is_res_row):
        """(N, MAX_ITEMS, N_EVID) float in [0,1]. Frozen list — the prereg pins these 10:
        [n_succ_log, n_fail_log, has_succ, S_satisfied_now, frac_S_held, S_shrunk,
         C_shrunk, C_satisfied_now, obtained_ever, since_succ_log]"""
        H = held.unsqueeze(1)                                            # (N,1,I)
        s_missing = (self.S & ~H).sum(-1).float()
        s_size = self.S.sum(-1).float()
        c_missing = (self.C & ~H).sum(-1).float()
        c_size = self.C.sum(-1).float()
        n = max(1, self.n - 1)
        f = torch.stack([
            (self.n_succ.clamp(min=0).add(1).log() / LOG_CAP).clamp(max=1),
            (self.n_fail.clamp(min=0).add(1).log() / LOG_CAP).clamp(max=1),
            (self.n_succ > 0).float(),
            (s_missing == 0).float(),
            1.0 - (s_missing / n).clamp(max=1),
            1.0 - (s_size / n).clamp(max=1),
            1.0 - (c_size / n).clamp(max=1),
            (c_missing == 0).float(),
            self.obtained_ever,
            (self.since_succ.add(1).log() / LOG_CAP).clamp(max=1),
        ], dim=-1)
        return f

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in
                ("n_succ", "n_fail", "since_succ", "obtained_ever", "S", "C", "last_ctx", "ctx_new")}

    def load_state_dict(self, st):
        for k, v in st.items():
            getattr(self, k).copy_(v)


class StoreEnv(HiddenEnv):
    """HiddenEnv + an EvidenceStore written from the honest outcome signal. The store PERSISTS across
    episode truncations (repairing v55's within-world amnesia) and resets only with the world."""

    def __init__(self, num_envs, spec, skill, cfg, seed=0, goal=None, hidden=True):
        super().__init__(num_envs, spec, skill, cfg, seed=seed, goal=goal, hidden=hidden)
        self.store = EvidenceStore(num_envs, self.n_items)

    def step(self, g):
        g = g.reshape(self.num_envs).clamp(max=self.n_items - 1)
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        held_before = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        held_before[:, :self.n_items] = self.base.inv > 0
        is_craft, cell_of = self.item_is_craft[g], self.item_cell[g]
        craft_act = self.item_craft_act[g]
        start = self.base.inv[ar, g].float()
        done_opt = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        det = not self.cfg.get("skill_stochastic", False)
        for t in range(self.option_timeout):
            ego = self.base.state[:, :self.ego_dim]
            goh = F.one_hot(cell_of, MAX_CELLS).float()
            a_skill = self.skill.act(torch.cat([ego, goh], -1), deterministic=det)
            a = torch.where(is_craft, craft_act, a_skill)
            self.base.step(a)
            self._prim += N
            cur = self.base.inv[ar, g].float()
            got_craft = is_craft & self.base.unlocked[ar, g]
            done_opt = done_opt | got_craft | (~is_craft & (cur >= start + 1))
            if t % 8 == 7 and bool(done_opt.all()):
                break
        obtained = (self.base.inv[ar, g].float() >= start + 1)
        prev_tried, prev_succ = self.tried[ar, g], self.succ[ar, g]
        self._att += N
        self._fail += int((~obtained).sum())
        self._repeat_prev_succ += int((prev_succ > 0.5).sum())
        self._first_try_ok += int((obtained & (prev_tried < 0.5)).sum())
        self.tried[ar, g] = 1.0
        self.succ[ar, g] = obtained.float()
        # ---- the store: written from the same honest signal, BEFORE any truncation reset ----------
        self.store.write(g, held_before, obtained, ~is_craft)
        held_after = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
        held_after[:, :self.n_items] = self.base.inv > 0
        self.store.note_inventory_change(held_after)
        self.post_unlocked = self.base.unlocked.clone()
        self.msteps += 1
        trunc = self.msteps >= self.macro_budget
        if bool(trunc.any()):
            self.base._reset_done(trunc)
            self.msteps = torch.where(trunc, torch.zeros_like(self.msteps), self.msteps)
            self.tried[trunc] = 0.0                      # per-episode; the STORE deliberately persists
            self.succ[trunc] = 0.0
        self._set_state()
        return self.state, torch.zeros(N, device=DEVICE), torch.zeros_like(trunc), trunc, trunc


@torch.no_grad()
def evidence_policy(env, goal):
    """Hand-coded reference policy over the store — NO learning, no recipe access. Establishes what the
    representation supports (arm G of the design; also the Step-1 feasibility demonstration).

    Priorities per env: certified goal > certified needed craft > plausible needed resource >
    experiment on a needed craft in a NEW inventory context > experiment anywhere > random valid."""
    st, N, n = env.store, env.num_envs, env.n_items
    held = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    held[:, :n] = env.base.inv > 0
    is_res = ~env.item_is_craft[:n]
    valid = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    valid[:, :n] = True
    # need-set: closure from the goal through current candidate sets (crafts) and gate candidates (res)
    need = torch.zeros(N, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    need[:, goal] = True
    cand = st.S | st.C
    for _ in range(5):
        frontier = need & ~held
        need = need | (frontier.unsqueeze(-1) & cand).any(1)
    s_sat = (st.S & ~held.unsqueeze(1)).sum(-1) == 0
    c_sat = (st.C & ~held.unsqueeze(1)).sum(-1) == 0
    never_ok = st.obtained_ever < 0.5
    res_row = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    res_row[:n] = is_res
    res_plausible = res_row.unsqueeze(0) & (c_sat | (st.n_fail < 2) | ~never_ok.bool())
    craft_row = ~res_row & valid

    score = torch.full((N, MAX_ITEMS), -1e9, device=DEVICE)
    rnd = torch.rand(N, MAX_ITEMS, device=DEVICE)
    score = torch.where(valid, rnd, score)                                        # 0: random valid
    score = torch.where(craft_row & st.ctx_new & never_ok.bool(), 10 + rnd, score)  # 1: experiment
    score = torch.where(need & craft_row & st.ctx_new & never_ok.bool(), 20 + rnd, score)
    score = torch.where(need & ~held & res_plausible, 30 + rnd, score)            # 3: needed resource
    score = torch.where(need & ~held & craft_row & s_sat, 40 + rnd, score)        # 4: certified craft
    goal_col = torch.zeros(MAX_ITEMS, dtype=torch.bool, device=DEVICE)
    goal_col[goal] = True
    goal_ready = goal_col.unsqueeze(0) & torch.where(res_row.unsqueeze(0), c_sat | (st.n_fail < 2), s_sat)
    score = torch.where(goal_ready, torch.full_like(score, 100.0), score)         # 5: the goal itself
    return score.argmax(-1)
