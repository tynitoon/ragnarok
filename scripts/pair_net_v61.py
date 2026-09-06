"""ARC 3 — PairNet: the identity-free policy over the action matrix, its buffers, and the fixed-budget
goal loop (ARC3_PLAN.md 2.6, 2.8).

    slow WEIGHTS  = what crosses worlds: a policy over (attributes, own evidence) and an OUTCOME head that
                    predicts what a pair will do — the place the family's law can live.
    fast STORE    = this world's facts (scripts/pair_store_v61.py), wiped at world entry for every arm.

The outcome head is supervised on what the agent OBSERVED (every training combine's outcome), never on
goal credit; the policy is trained by hindsight self-imitation toward the COMMANDED goal only
(relabel_commanded_v58's rule, ported to this row layout). Every arm trains identically; the only
difference between arms is the initial weights.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import OUT_INVALID, OUT_PRODUCT
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.pair_store_v61 import (PairEnv, PairStore, MAX_ITEMS, N_SLOT, N_PAIRS, N_PF, ROW, N_ACT, K_EL, GOAL_COL,
                                    F_GOAL, F_VALID, F_EL, N_ATTR, PAIR_I, PAIR_J, action_valid, eval_mask)

QS = 255.0
N_CELL_IN = 3 * N_SLOT + N_ATTR + N_PF + 1          # 118


# ---------------------------------------------------------------- the net

class PairNet(nn.Module):
    """Scores the 406 cells of the action matrix with ONE shared MLP. Pair cells are encoded symmetrically
    ([x_i + x_j, x_i * x_j]) so the net is permutation-equivariant over slots; no nn.Embedding, no
    parameter shaped to MAX_ITEMS. The element one-hot inside x is the disclosed 14-symbol vocabulary."""

    def __init__(self, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_CELL_IN, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.out3 = nn.Linear(hidden, 3)                 # botch / inert / product
        self.outel = nn.Linear(hidden, K_EL)             # product element
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.score.weight, gain=0.01)

    def cells(self, obs):
        B = obs.shape[0]
        x = obs[:, :MAX_ITEMS * N_SLOT].reshape(B, MAX_ITEMS, N_SLOT)
        pf = obs[:, MAX_ITEMS * N_SLOT:].reshape(B, N_PAIRS, N_PF)
        valid = (x[..., F_VALID] > 0.5).float().unsqueeze(-1)
        ctx = (x * valid).sum(1) / valid.sum(1).clamp(min=1)                               # (B,32)
        goal = (x[..., F_GOAL:F_GOAL + 1] * x[..., F_EL:F_EL + N_ATTR]).sum(1)              # (B,16)
        xi, xj = x[:, PAIR_I], x[:, PAIR_J]
        ctx_p = ctx.unsqueeze(1).expand(-1, N_PAIRS, -1); goal_p = goal.unsqueeze(1).expand(-1, N_PAIRS, -1)
        pair = torch.cat([xi + xj, xi * xj, ctx_p, goal_p, pf, torch.zeros(B, N_PAIRS, 1, device=obs.device)], -1)
        ctx_d = ctx.unsqueeze(1).expand(-1, MAX_ITEMS, -1); goal_d = goal.unsqueeze(1).expand(-1, MAX_ITEMS, -1)
        diag = torch.cat([2 * x, x * x, ctx_d, goal_d, torch.zeros(B, MAX_ITEMS, N_PF, device=obs.device),
                          torch.ones(B, MAX_ITEMS, 1, device=obs.device)], -1)
        return torch.cat([diag, pair], 1)                                                  # (B,406,118)

    def forward(self, obs):
        h = self.enc(self.cells(obs))
        logits = self.score(h).squeeze(-1).masked_fill(~action_valid(obs), -1e9)
        hp = h[:, MAX_ITEMS:]
        return logits, self.out3(hp), self.outel(hp)


class ComposerV61:
    def __init__(self, lr=3e-4, hidden=128, init_seed=None):
        if init_seed is not None:
            torch.manual_seed(int(init_seed))
        self.net = PairNet(hidden).to(DEVICE)
        self.lr = lr
        self.reset_optimizer()

    def reset_optimizer(self):
        """Constructed fresh at every WORLD entry for every arm (weights are the only thing carried)."""
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    @torch.no_grad()
    def act(self, obs, env=None, epsilon=0.0, temp=1.0, deterministic=False):
        logits, _, _ = self.net(obs)
        if deterministic:
            m = eval_mask(obs)
            m = torch.where(m.any(-1, keepdim=True), m, action_valid(obs))   # no absorbing state
            return logits.masked_fill(~m, -1e9).argmax(-1)
        a = torch.multinomial(F.softmax(logits / temp, -1), 1).squeeze(-1)
        if epsilon > 0:
            valid = action_valid(obs).float()
            rnd = torch.multinomial(valid.clamp(min=1e-9), 1).squeeze(-1)
            a = torch.where(torch.rand(a.shape[0], device=DEVICE) < epsilon, rnd, a)
        return a

    def train_steps(self, buf, n_steps, bs=512):
        if buf.n == 0 and buf.n_out == 0:
            return float("nan")
        tot = 0.0
        for _ in range(n_steps):
            loss = torch.zeros((), device=DEVICE)
            if buf.n > 0:
                s, a = buf.sample(bs)
                logits, _, _ = self.net(s)
                loss = loss + F.cross_entropy(logits, a)
            if buf.n_out > 0:                                    # the outcome head trains even before any success
                so, p, o, e = buf.sample_out(bs)
                _, o3, oel = self.net(so)
                ar = torch.arange(so.shape[0], device=DEVICE)
                loss = loss + F.cross_entropy(o3[ar, p], o)
                prod = o == OUT_PRODUCT
                if bool(prod.any()):
                    loss = loss + F.cross_entropy(oel[ar[prod], p[prod]], e[prod])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            tot += float(loss.detach())
        return tot / n_steps


class BufferV61:
    """Per-WORLD FIFOs: policy rows (obs, action) and outcome rows (obs, pair, outcome, product element).
    uint8 with one quantisation scale on write and read. Eval episodes never enter either (test (13))."""

    def __init__(self, cap=200_000, cap_out=100_000):
        self.s = torch.zeros(cap, ROW, dtype=torch.uint8, device=DEVICE)
        self.a = torch.zeros(cap, dtype=torch.long, device=DEVICE)
        self.cap, self.n, self.ptr = cap, 0, 0
        self.so = torch.zeros(cap_out, ROW, dtype=torch.uint8, device=DEVICE)
        self.po = torch.zeros(cap_out, dtype=torch.long, device=DEVICE)
        self.oo = torch.zeros(cap_out, dtype=torch.long, device=DEVICE)
        self.eo = torch.zeros(cap_out, dtype=torch.long, device=DEVICE)
        self.cap_out, self.n_out, self.ptr_out = cap_out, 0, 0

    @staticmethod
    def _put(store_list, ptr, cap, n, rows):
        k = rows[0].shape[0]
        end = ptr + k
        if end <= cap:
            for st, r in zip(store_list, rows):
                st[ptr:end] = r
        else:
            r0 = cap - ptr
            for st, r in zip(store_list, rows):
                st[ptr:], st[:k - r0] = r[:r0], r[r0:]
        return end % cap, min(n + k, cap)

    def add(self, s, a):
        k = s.shape[0]
        if k == 0:
            return
        if k > self.cap:
            s, a = s[-self.cap:], a[-self.cap:]
        q = (s.clamp(0, 1) * QS).round().to(torch.uint8)
        self.ptr, self.n = self._put([self.s, self.a], self.ptr, self.cap, self.n, [q, a])

    def add_out(self, s, p, o, e):
        k = s.shape[0]
        if k == 0:
            return
        if k > self.cap_out:
            s, p, o, e = s[-self.cap_out:], p[-self.cap_out:], o[-self.cap_out:], e[-self.cap_out:]
        q = (s.clamp(0, 1) * QS).round().to(torch.uint8)
        self.ptr_out, self.n_out = self._put([self.so, self.po, self.oo, self.eo], self.ptr_out, self.cap_out,
                                             self.n_out, [q, p, o, e])

    def sample(self, bs):
        idx = torch.randint(0, self.n, (min(bs, self.n),), device=DEVICE)
        return self.s[idx].float() / QS, self.a[idx]

    def sample_out(self, bs):
        idx = torch.randint(0, self.n_out, (min(bs, self.n_out),), device=DEVICE)
        return self.so[idx].float() / QS, self.po[idx], self.oo[idx], self.eo[idx]


# ---------------------------------------------------------------- collection / hindsight / eval

def collect_episode_v61(env, composer, cfg, goal):
    """One macro-episode under a commanded goal. Policy rows stored GOAL-FREE (rewritten at relabel);
    every training combine's (obs at decision, pair, outcome, product element) goes to the outcome rows."""
    N, T = env.num_envs, env.macro_budget
    states = torch.zeros(T, N, ROW, device=DEVICE)
    actions = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
    unlockstep = torch.full((N, env.n_items), -1, dtype=torch.long, device=DEVICE)
    outs = []
    env.reset(); obs = env.set_goal(goal)
    prev = env.base.unlocked.clone()
    for t in range(T):
        a = composer.act(obs, env=env, epsilon=cfg["epsilon"], temp=cfg["temp"])
        s = obs.clone()
        s.view(N, -1)[:, :MAX_ITEMS * N_SLOT].view(N, MAX_ITEMS, N_SLOT)[..., GOAL_COL] = 0.0
        states[t] = s
        actions[t] = a
        obs, info = env.step(a)
        keep = info["is_pair"] & (info["outcome"] != OUT_INVALID)
        if bool(keep.any()):
            outs.append((s[keep], (a[keep] - MAX_ITEMS), info["outcome"][keep], info["product_el"][keep]))
        newly = env.post_unlocked & ~prev
        first = (unlockstep == -1) & newly
        unlockstep[first] = t
        prev = env.post_unlocked.clone()
    if outs:
        outs = tuple(torch.cat([o[k] for o in outs]) for k in range(4))
    else:
        outs = None
    return states, actions, unlockstep, outs


def relabel_commanded_v61(states, actions, unlockstep, max_samples, goal, gamma=0.7):
    """relabel_commanded_v58's rule on the v61 row: hindsight toward ONLY the commanded goal, each step
    t <= unlockstep kept with probability gamma^lag. Incidental unlocks are discarded (D4 guard)."""
    T, N, _ = states.shape
    u = unlockstep[:, goal]
    lag = u.view(1, N) - torch.arange(T, device=DEVICE).view(T, 1)
    valid = (lag >= 0) & (u.view(1, N) >= 0)
    if gamma < 1.0:
        p = torch.pow(torch.tensor(gamma, device=DEVICE), lag.clamp(min=0).float())
        valid = valid & (torch.rand(valid.shape, device=DEVICE) < p)
    idx = valid.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None, None, 0
    if idx.shape[0] > max_samples:
        idx = idx[torch.randperm(idx.shape[0], device=DEVICE)[:max_samples]]
    t, n = idx[:, 0], idx[:, 1]
    s = states[t, n].clone()
    s[:, goal * N_SLOT + GOAL_COL] = 1.0
    return s, actions[t, n], int(idx.shape[0])


@torch.no_grad()
def eval_goal_v61(spec, skill, composer, cfg, seed, goal, store_state=None):
    """Deterministic mastery on a FRESH grid with a COPY of the agent's current store (mutated during eval,
    never the training store; nothing here reaches any buffer)."""
    env = PairEnv(cfg["num_envs"], spec, skill, cfg, seed=seed + 9, goal=goal)
    if store_state is not None:
        env.store.load_state_dict(store_state)
    got = torch.zeros(cfg["num_envs"], dtype=torch.bool, device=DEVICE)
    obs = env.obs()
    for _ in range(cfg["macro_budget"]):
        obs, _ = env.step(composer.act(obs, env=env, deterministic=True))
        got |= env.post_unlocked[:, goal]
    return float(got.float().mean())


def run_goal_v61(env, spec, skill, composer, buf, cfg, seed, goal, r_max, train=True):
    """One commanded goal at a FIXED budget of r_max rounds, no early break; master_per_round[0..r_max]."""
    m0 = env.msteps_total
    zs = eval_goal_v61(spec, skill, composer, cfg, seed, goal, env.store.state_dict())
    master_per_round, demos, samples = [round(zs, 4)], [], []
    ep_count = 0
    first_env = torch.full((env.num_envs,), -1, dtype=torch.long, device=DEVICE)   # per-env attempts to first obtain
    for r in range(r_max):
        d = k = 0
        for _ in range(cfg["episodes_per_round"]):
            s, a, us, outs = collect_episode_v61(env, composer, cfg, goal)
            hit = int((us[:, goal] >= 0).sum())
            got = (us[:, goal] >= 0) & (first_env < 0)
            first_env = torch.where(got, ep_count * env.macro_budget + us[:, goal] + 1, first_env)
            ep_count += 1
            d += hit
            ss, aa, kept = relabel_commanded_v61(s, a, us, cfg["max_samples_per_ep"], goal)
            k += kept
            if ss is not None:
                buf.add(ss, aa)
            if outs is not None:
                buf.add_out(*outs)
        demos.append(d); samples.append(k)
        if train:
            composer.train_steps(buf, cfg["train_steps_per_round"])
        master_per_round.append(round(eval_goal_v61(spec, skill, composer, cfg, seed, goal,
                                                    env.store.state_dict()), 4))
    cap = r_max * cfg["episodes_per_round"] * env.macro_budget
    fe = torch.where(first_env < 0, torch.full_like(first_env, cap), first_env)      # censored at the budget
    first_median = int(fe.float().median()) if r_max > 0 else None                   # the CPU model's statistic
    first_any = int(first_env[first_env >= 0].min()) if bool((first_env >= 0).any()) else None
    return dict(goal=goal, master_per_round=master_per_round, demos_per_round=demos,
                samples_per_round=samples, attempts=env.msteps_total - m0, buf_n=buf.n, buf_out=buf.n_out,
                first_demo_attempt=first_median, first_demo_any=first_any, first_censored=float((first_env < 0).float().mean()),
                mastered=bool(master_per_round[-1] >= cfg["thresh"]))


def run_unit_v61(spec, skill, composer, cfg, world_seed, goals, r_max, train=True, log=None):
    """A UNIT (ARC3_PLAN 2.8): one world entered with store, buffers and optimizer EMPTY; the stream of goals
    at a fixed budget each; the store persists across the stream. Returns one row per goal."""
    env = PairEnv(cfg["num_envs"], spec, skill, cfg, seed=world_seed + 7, goal=goals[0])
    buf = BufferV61(cap=cfg.get("buf_cap", 200_000))
    composer.reset_optimizer()
    rows = []
    for p, g in enumerate(goals):
        if not cfg.get("store_persists", False):
            env.store = PairStore(env.num_envs, env.n_items)     # ARC3_PLAN 2.5: per-GOAL working memory
        r = run_goal_v61(env, spec, skill, composer, buf, cfg, world_seed + 11 * p, g, r_max, train=train)
        r["tier"] = int(spec["tier"][g]); r["pos"] = p
        rows.append(r)
        if log is not None:
            log(r)
    rows_store = env.store.state_dict()
    return rows, rows_store


def area(row, b_from=1, b_to=None):
    m = row["master_per_round"][b_from:(None if b_to is None else b_to + 1)]
    return sum(m) / len(m)


# ---------------------------------------------------------------- config and the frozen nav skill

def cfg_v61(num_envs=64, r_max=3):
    """cfg_v58's values (ARC3_PLAN 4.0): 64 envs, macro_budget 48, 4 episodes per round, 300 train steps,
    thresh 0.6, option_timeout 16."""
    return dict(num_envs=num_envs, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
                nav_max_steps=40, skill_iters=400, option_timeout=16, macro_budget=48,
                episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
                epsilon=0.05, temp=1.0, thresh=0.6, r_max=r_max, skill_stochastic=True,
                buf_cap=200_000)


def load_skill(cfg, seed=0, out_dir="craft_v6_out"):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"), map_location=DEVICE))
    return ppo


def init_seed(lineage, world, arm):
    """The seed table of ARC3_PLAN 4.7: fresh nets are a fresh draw in every unit; distinct across the
    gate / pretrain / test roles because (world - 8000) differs by role. arm 1-3 fresh, 4 = a lineage's
    own initial net, lineage 9 = the gate."""
    return 100_000 + 1000 * lineage + 10 * (world - 8000) + arm
