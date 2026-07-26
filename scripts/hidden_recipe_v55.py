"""v55 core — HIDDEN-RECIPE persistent world (prereg FROZEN before this file, preregistration.md).

The v49-v54 arc measured "does memory help?" in a world where meta_manager_v51.py:96-102 handed the
agent the recipe DAG (craftable_now/collectable_now, computed from the true recipe tensors) at EVERY
step, and where the goal was always argmax(depth). So a goal-blind "craft whatever is craftable" reflex
won BY DEFINITION: nothing was expensive-to-rederive (memory could not pay) and the goal was never
load-bearing (the composer went goal-agnostic 3/3).

v55 removes the oracle. The agent observes only its OWN attempt history [tried, last_succeeded] and must
DISCOVER the recipe DAG by paying for failed option attempts and REMEMBER it across a stream of goals in
ONE persistent world. Mechanism changes M1-M9 are frozen in the prereg; this file implements them.

Library only — the driver is scripts/run_v55.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import DeviceVecTechTree
from scripts.depth_scaling_v49 import MAX_CELLS
from scripts.meta_manager_v51 import RouterEnv, PerItemRouter, MAX_ITEMS, N_FEAT

GOAL_COL = 4
# feature layout: [in_inv, unlocked, F2, F3, is_goal, is_resource, is_valid]
#   hidden=True  (M1): F2 = tried_this_episode, F3 = last_attempt_succeeded   <- the agent's OWN history
#   hidden=False (D)  : F2 = craftable_now,     F3 = collectable_now          <- the v54 affordance oracle


# ---------------------------------------------------------------- M4: index permutation

def permute_spec(spec, seed):
    """M4 — randomly permute item indices. gen_tree emits items in TOPOLOGICAL order (tech_tree.py:33-45),
    so 'attempt items in index order' would solve any tree with zero knowledge and masquerade as memory."""
    n = spec["n_items"]
    perm = np.random.default_rng(seed).permutation(n)          # new i  <- old perm[i]
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)                                   # old j  -> new inv[j]
    m = lambda j: int(inv[j])                                  # noqa: E731
    kind = [spec["kind"][perm[i]] for i in range(n)]
    out = dict(spec)
    out.update(
        kind=kind,
        cell=[spec["cell"][perm[i]] for i in range(n)],
        tool=[m(spec["tool"][perm[i]]) if spec["tool"][perm[i]] >= 0 else -1 for i in range(n)],
        inputs=[{m(j): c for j, c in spec["inputs"][perm[i]].items()} for i in range(n)],
        tools=[[m(t) for t in spec["tools"][perm[i]]] for i in range(n)],
        depth=[spec["depth"][perm[i]] for i in range(n)],
        target=m(spec["target"]),
        true_pre=[{m(j) for j in spec["true_pre"][perm[i]]} for i in range(n)],
        craft_actions=[i for i in range(n) if kind[i] == "C"],
        _perm=perm.tolist(),
    )
    return out


# ---------------------------------------------------------------- exact symbolic difficulty

def prod_count(spec, g):
    """Productions to obtain item g once, inputs CONSUMED (tree-expanded), tools counted once.
    Permutation-invariant structural difficulty (= v54's production_count, parameterised by goal)."""
    def tree(i, d=0):
        if d > 40 or spec["kind"][i] == "R":
            return 1
        return 1 + sum(c * tree(j, d + 1) for j, c in spec["inputs"][i].items())

    def tools_of(i, acc, d=0):
        if d > 40:
            return
        if spec["kind"][i] == "R" and spec["tool"][i] >= 0:
            acc.add(spec["tool"][i]); tools_of(spec["tool"][i], acc, d + 1)
        for t in spec["tools"][i]:
            acc.add(t); tools_of(t, acc, d + 1)
        for j in spec["inputs"][i]:
            tools_of(j, acc, d + 1)

    tl = set(); tools_of(g, tl)
    return tree(g) + sum(tree(t) for t in tl)


def blind_cost(spec, g, n_mc=200, cap=300, seed=0):
    """Expected macro-steps for the REFERENCE GOAL-BLIND policy (v51 greedy_act with the is_goal rule
    removed) to first obtain g, under RANDOM tie-breaking. Random ties keep this permutation-invariant
    and deny the index-order shortcut. Symbolic + perfect execution => a LOWER bound on real cost, so
    the GOAL-NECESSARY stratum (blind > macro_budget) is conservative."""
    n = spec["n_items"]
    rng = np.random.default_rng(seed + 7 * g)
    tot = 0.0
    for _ in range(n_mc):
        inv = [0] * n
        steps = 0
        while steps < cap:
            cand, best = [], 0
            for i in range(n):
                if inv[i] > 0:
                    continue                                   # greedy: only pursue what is not held
                if spec["kind"][i] == "R":
                    ok = spec["tool"][i] < 0 or inv[spec["tool"][i]] >= 1
                    s = 1
                else:
                    ok = (all(inv[j] >= c for j, c in spec["inputs"][i].items())
                          and all(inv[t] >= 1 for t in spec["tools"][i]))
                    s = 2
                if ok:
                    if s > best:
                        cand, best = [i], s
                    elif s == best:
                        cand.append(i)
            if not cand:
                steps = cap; break
            i = int(rng.choice(cand))
            if spec["kind"][i] == "C":
                for j, c in spec["inputs"][i].items():
                    inv[j] -= c
            inv[i] += 1
            steps += 1
            if i == g:
                break
        tot += min(steps, cap)
    return tot / n_mc


def admitted_goals(spec, pc_max=20, depth_min=2):
    """Frozen admission rule: depth(g) >= 2 AND pc(g) <= 20. Returns [(g, pc, blind)] ascending by pc."""
    out = []
    for g in range(spec["n_items"]):
        if spec["depth"][g] < depth_min:
            continue
        pc = prod_count(spec, g)
        if pc > pc_max:
            continue
        out.append((g, pc, blind_cost(spec, g)))
    out.sort(key=lambda r: (r[1], r[2]))
    return out


# ---------------------------------------------------------------- env (M1 + M5)

class HiddenEnv(RouterEnv):
    """RouterEnv with (M1) the affordance oracle replaced by the agent's OWN attempt history, and
    (M5) a per-env COMMANDED goal instead of the hardcoded f[:, target, 4] = 1.

    hidden=False restores the exact v54 oracle features — that is arm D, the strong baseline."""

    def __init__(self, num_envs, spec, skill, cfg, seed=0, goal=None, hidden=True):
        self._goal0, self.hidden = goal, hidden
        super().__init__(num_envs, spec, skill, cfg, seed=seed)

    def reset(self):
        N = self.num_envs
        self.base.reset()
        self.msteps = torch.zeros(N, device=DEVICE)
        self.tried = torch.zeros(N, MAX_ITEMS, device=DEVICE)
        self.succ = torch.zeros(N, MAX_ITEMS, device=DEVICE)
        g0 = self.target if self._goal0 is None else self._goal0
        self.goal = torch.full((N,), int(g0), dtype=torch.long, device=DEVICE)
        self._set_state()
        return self.state

    def set_goal(self, g):
        """Command a goal: int (all envs) or LongTensor (N,). Does not reset the world."""
        self.goal = (torch.full((self.num_envs,), int(g), dtype=torch.long, device=DEVICE)
                     if isinstance(g, int) else g.to(DEVICE).long())
        self._set_state()
        return self.state

    def _set_state(self):
        N, n = self.num_envs, self.n_items
        inv, unlocked = self.base.inv, self.base.unlocked
        f = torch.zeros(N, MAX_ITEMS, N_FEAT, device=DEVICE)
        f[:, :n, 0] = (inv > 0).float()
        f[:, :n, 1] = unlocked.float()
        if self.hidden:
            f[:, :, 2] = self.tried                       # M1: own within-episode attempt history
            f[:, :, 3] = self.succ                        #     (NO recipe information)
        else:
            ci, ct = self.base.craft_in, self.base.craft_tool           # v54 oracle (arm D only)
            inputs_ok = (inv.unsqueeze(1) >= ci.unsqueeze(0)).all(-1)
            tools_ok = ((inv.unsqueeze(1) >= 1) | ~ct.unsqueeze(0)).all(-1)
            f[:, self.craft_out_idx, 2] = (inputs_ok & tools_ok).float()
            rt = self.res_tool[:n]
            has_tool = (rt < 0) | (inv.gather(1, rt.clamp(min=0).expand(N, n)) >= 1)
            f[:, :n, 3] = torch.where(self.is_res[:n].unsqueeze(0), has_tool.float(),
                                      torch.zeros(N, n, device=DEVICE))
        f[torch.arange(N, device=DEVICE), self.goal, GOAL_COL] = 1.0    # M5: commanded goal
        f[:, :n, 5] = self.is_res[:n].float().unsqueeze(0)
        f[:, :n, 6] = 1.0
        self.state = f.reshape(N, -1)

    def step(self, g):
        g = g.reshape(self.num_envs).clamp(max=self.n_items - 1)
        N, ar = self.num_envs, torch.arange(self.num_envs, device=DEVICE)
        rew = torch.zeros(N, device=DEVICE)
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
        # the honest outcome signal the agent gets: did I actually OBTAIN one more of item g?
        obtained = (self.base.inv[ar, g].float() >= start + 1)
        self.tried[ar, g] = 1.0
        self.succ[ar, g] = obtained.float()
        self.post_unlocked = self.base.unlocked.clone()        # BEFORE any truncation reset
        self.msteps += 1
        trunc = self.msteps >= self.macro_budget
        if bool(trunc.any()):
            self.base._reset_done(trunc)
            self.msteps = torch.where(trunc, torch.zeros_like(self.msteps), self.msteps)
            self.tried[trunc] = 0.0                            # attempt history is WITHIN-episode
            self.succ[trunc] = 0.0
        self._set_state()
        return self.state, rew, torch.zeros_like(trunc), trunc, trunc


# ---------------------------------------------------------------- M2: somewhere to put the memory

class MemoryNet(nn.Module):
    """M2 — PerItemRouter scores item i from item i's OWN features alone, so with the oracle gone it is
    architecturally incapable of representing "craft 7 because I hold 3 and 5". Adds a per-item identity
    embedding, an inventory-weighted pooled context, and the commanded goal's embedding. The embedding
    table IS the remembered world-knowledge (and is directly inspectable)."""

    def __init__(self, hidden=128, emb=16):
        super().__init__()
        self.emb = nn.Embedding(MAX_ITEMS, emb)
        self.enc = nn.Sequential(nn.Linear(N_FEAT + 3 * emb, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.score.weight, gain=0.01)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.reshape(B, MAX_ITEMS, N_FEAT)
        e = self.emb.weight.unsqueeze(0).expand(B, -1, -1)              # (B,I,E)
        w = x[..., 0:1]                                                # in_inv
        ctx = (e * w).sum(1) / w.sum(1).clamp(min=1.0)                 # inventory-weighted context
        gem = (e * x[..., GOAL_COL:GOAL_COL + 1]).sum(1)               # commanded goal's embedding
        h = torch.cat([x, e,
                       ctx.unsqueeze(1).expand(-1, MAX_ITEMS, -1),
                       gem.unsqueeze(1).expand(-1, MAX_ITEMS, -1)], -1)
        z = self.enc(h)
        logits = self.score(z).squeeze(-1).masked_fill(x[..., 6] < 0.5, -1e9)
        return logits, self.value(z.mean(1)).squeeze(-1)


class Composer:
    """Hindsight self-imitation composer. net='memory' (A/B/C/E) or 'router' (D = unmodified v54)."""

    def __init__(self, net="memory", lr=3e-4):
        self.net = (MemoryNet() if net == "memory" else PerItemRouter()).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.kind = net

    @torch.no_grad()
    def act(self, state, epsilon=0.0, temp=1.0, deterministic=False):
        logits, _ = self.net(state)
        if deterministic:
            return logits.argmax(-1)
        a = torch.multinomial(F.softmax(logits / temp, -1), 1).squeeze(-1)
        if epsilon > 0:
            valid = state.reshape(-1, MAX_ITEMS, N_FEAT)[..., 6] > 0.5
            rnd = torch.multinomial(valid.float(), 1).squeeze(-1)
            a = torch.where(torch.rand(a.shape[0], device=DEVICE) < epsilon, rnd, a)
        return a

    def train_steps(self, buf, n_steps, bs=512):
        if buf.n == 0:
            return float("nan")
        tot = 0.0
        for _ in range(n_steps):
            s, a = buf.sample(bs)
            logits, _ = self.net(s)
            loss = F.cross_entropy(logits, a)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            tot += float(loss)
        return tot / n_steps


class Buffer:
    """M9 — lifelong FIFO, cap 2M (at 32k samples/round a 400k buffer becomes a sliding window after 12
    rounds). States are BINARY, so uint8 storage is exact and keeps 2M samples at ~400MB."""

    def __init__(self, cap=2_000_000):
        self.s = torch.zeros(cap, MAX_ITEMS * N_FEAT, dtype=torch.uint8, device=DEVICE)
        self.a = torch.zeros(cap, dtype=torch.long, device=DEVICE)
        self.cap, self.n, self.ptr = cap, 0, 0

    def add(self, s, a):
        k = s.shape[0]
        if k == 0:
            return
        if k >= self.cap:
            s, a, k = s[-self.cap:], a[-self.cap:], self.cap
        s = s.to(torch.uint8)
        end = self.ptr + k
        if end <= self.cap:
            self.s[self.ptr:end], self.a[self.ptr:end] = s, a
        else:
            r = self.cap - self.ptr
            self.s[self.ptr:], self.a[self.ptr:] = s[:r], a[:r]
            self.s[:k - r], self.a[:k - r] = s[r:], a[r:]
        self.ptr = end % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, bs):
        idx = torch.randint(0, self.n, (min(bs, self.n),), device=DEVICE)
        return self.s[idx].float(), self.a[idx]

    def state_dict(self):
        return dict(s=self.s[:self.n].clone(), a=self.a[:self.n].clone(), ptr=self.ptr, n=self.n)

    def load_state_dict(self, st):
        n = int(st["n"]); self.s[:n], self.a[:n] = st["s"], st["a"]
        self.n, self.ptr = n, int(st["ptr"])                   # M9: preserve the TRUE FIFO write pointer


# ---------------------------------------------------------------- collection / hindsight

def collect_episode(env, composer, epsilon, temp, goal):
    """One synchronised macro-episode under a commanded goal. States are stored GOAL-FREE (the goal
    column is rewritten at relabel time)."""
    N, T = env.num_envs, env.macro_budget
    states = torch.zeros(T, N, MAX_ITEMS * N_FEAT, device=DEVICE)
    actions = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
    unlockstep = torch.full((N, env.n_items), -1, dtype=torch.long, device=DEVICE)
    env.reset(); env.set_goal(goal)
    prev = env.base.unlocked.clone()
    obs = env.state
    for t in range(T):
        a = composer.act(obs, epsilon=epsilon, temp=temp)
        s = obs.clone().reshape(N, MAX_ITEMS, N_FEAT)
        s[..., GOAL_COL] = 0.0
        states[t] = s.reshape(N, -1)
        actions[t] = a
        obs, _, _, _, _ = env.step(a)
        newly = env.post_unlocked & ~prev
        first = (unlockstep == -1) & newly
        unlockstep[first] = t
        prev = env.post_unlocked.clone()
    return states, actions, unlockstep


def relabel(states, actions, unlockstep, max_samples, gamma=0.7):
    """M6 — goal-DISCRIMINATIVE hindsight. v53/v54 kept (s_t, a_t, X) for EVERY X unlocked at or after t
    with equal weight, so an identical state paired with different goals carried the SAME label: goal
    INVARIANCE was the exact cross-entropy optimum and the loss itself taught goal-blindness. Here a
    sample survives with probability gamma^(unlockstep[X] - t), so the label distribution actually
    depends on the goal. gamma=1.0 reproduces the v54 rule exactly (used by arm D)."""
    T, N, D = states.shape
    t_idx = torch.arange(T, device=DEVICE).view(T, 1, 1)
    lag = unlockstep.unsqueeze(0) - t_idx
    valid = (lag >= 0) & (unlockstep.unsqueeze(0) >= 0)
    if gamma < 1.0:
        p = torch.pow(torch.tensor(gamma, device=DEVICE), lag.clamp(min=0).float())
        valid = valid & (torch.rand(valid.shape, device=DEVICE) < p)
    idx = valid.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None, None
    if idx.shape[0] > max_samples:
        idx = idx[torch.randperm(idx.shape[0], device=DEVICE)[:max_samples]]
    t, nn_, X = idx[:, 0], idx[:, 1], idx[:, 2]
    s = states[t, nn_].clone().reshape(-1, MAX_ITEMS, N_FEAT)
    s[torch.arange(s.shape[0], device=DEVICE), X, GOAL_COL] = 1.0
    return s.reshape(s.shape[0], -1), actions[t, nn_]


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def eval_goal(spec, skill, composer, cfg, seed, goal, hidden=True, n=256, command=None):
    """P(obtain `goal` within macro_budget) while COMMANDING `command` (default: the goal itself)."""
    env = HiddenEnv(n, spec, skill, cfg, seed=seed + 9,
                    goal=goal if command is None else command, hidden=hidden)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        obs, _, _, _, _ = env.step(composer.act(obs, deterministic=True))
        got |= env.post_unlocked[:, goal]
    return float(got.float().mean())


@torch.no_grad()
def goal_swap_separation(spec, skill, composer, cfg, seed, goals, hidden=True, n=256):
    """P2 — the sharp test v54 structurally could not fire.
    S = mean over ordered pairs (X != Y) of [ P(obtain Y | commanded Y) - P(obtain Y | commanded X) ].
    Goal-BLIND policy => S ~ 0 (what it does never depends on what it was asked for)."""
    own = {Y: eval_goal(spec, skill, composer, cfg, seed, Y, hidden, n) for Y in goals}
    diffs = []
    for Y in goals:
        for X in goals:
            if X != Y:
                diffs.append(own[Y] - eval_goal(spec, skill, composer, cfg, seed, Y, hidden, n, command=X))
    return (sum(diffs) / len(diffs) if diffs else 0.0), own


@torch.no_grad()
def nav_gate(skill, spec, cfg, seed, n=256):
    """Pre-run gate: per-cell-type nav success under the frozen childhood skill. v54's only two
    both-arms-fail tasks were exactly the trees with the most resource cell-types, and held-out trees
    swung 0.027 -> 0.973 on the skill seed alone. Returns {cell_type: success}."""
    out = {}
    for i in range(spec["n_items"]):
        if spec["kind"][i] != "R":
            continue
        c = spec["cell"][i]
        env = DeviceVecTechTree(n, spec, grid=cfg["grid"], view=cfg["view"], max_steps=cfg["nav_max_steps"],
                               n_resource=cfg["n_resource"], nav_goal=int(c), max_cells=MAX_CELLS,
                               grant=[1] * spec["n_items"], seed=seed + 777 + i)
        got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        obs = env.state
        for _ in range(cfg["nav_max_steps"]):
            obs, _, term, _, _ = env.step(skill.act(obs, deterministic=True))
            got |= term
        out[int(c)] = round(float(got.float().mean()), 3)
    return out


# ---------------------------------------------------------------- one goal-task

def run_goal(spec, skill, composer, buf, cfg, seed, goal, hidden=True, gamma=0.7):
    """Train on ONE commanded goal until mastered or R_max rounds.

    M7 — v53/v54 returned cost=0 on zero-shot success WITHOUT collecting anything, so the lifelong
    buffer silently stopped growing on 13-15 of 16 tasks and compounding became unobservable by
    construction. Here EVERY goal collects >= 1 round, charged to the arm."""
    env = HiddenEnv(cfg["num_envs"], spec, skill, cfg, seed=seed, goal=goal, hidden=hidden)
    env._prim = 0
    zs = eval_goal(spec, skill, composer, cfg, seed, goal, hidden)
    master, rounds = zs, 0
    for r in range(cfg["r_max"]):
        for _ in range(cfg["episodes_per_round"]):
            s, a, us = collect_episode(env, composer, cfg["epsilon"], cfg["temp"], goal)
            ss, aa = relabel(s, a, us, cfg["max_samples_per_ep"], gamma=gamma)
            if ss is not None:
                buf.add(ss, aa)
        composer.train_steps(buf, cfg["train_steps_per_round"])
        rounds = r + 1
        master = eval_goal(spec, skill, composer, cfg, seed, goal, hidden)
        if master >= cfg["thresh"]:
            break
    return dict(goal=goal, zero_shot=round(zs, 3), rounds=rounds, prim=env._prim,
                master=round(master, 3), mastered=bool(master >= cfg["thresh"]))
