"""ARC 2, Step 2 — EvidenceNet: the portable (identity-free) policy over the per-world evidence store.

  slow WEIGHTS = the skill of learning a world. Identity-free by construction: the policy scores every
                 slot with the SAME MLP from observable + evidence features only. There is no
                 nn.Embedding, so no item-identity parameter can cross a world boundary.
  fast STORE   = everything specific to THIS world (scripts/evidence_store_v58.py), written by a fixed
                 rule from the agent's own attempt outcomes, wiped on world entry.

At test on a held-out world the weights are FROZEN — every bit of adaptation is a store write.

OBSERVATION LAYOUT (per slot, 17 dims): [7 base observable feats || 10 evidence feats].
Base feats are HiddenEnv's [in_inv, unlocked, tried, succ, is_goal, is_resource, is_valid]; index 4 is
still the goal column and indices 2/3 still drive the v56 instrument mask, so v57's machinery ports
unchanged. Buffer rows are uint8 with a single consistent quantisation (x*255 on write, /255 on read)
covering BOTH halves, so the binary base feats round-trip exactly.

DELIBERATE DEVIATION FROM ARC2_PLAN.md section 3, measured not assumed: the plan says to permute cell
IDs into 1..MAX_CELLS-2. The frozen childhood nav skill has only ever seen cell IDs 1..9 (measured over
its 8 training trees), and it consumes the target cell type as a one-hot; remapping into 1..22 would
feed it positions it never trained on and collapse navigation for EVERY arm. permute_spec_v58 therefore
permutes cell IDs WITHIN the set each world actually uses, which still decorrelates cell ID from
gen_tree's creation order (the stated purpose) while keeping the skill in distribution. The composer
never observes cell IDs at all, so this shortcut was dormant at policy level regardless.
scripts/test_net_v58.py measures the nav gate with and without the permutation to keep this honest.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from scripts.meta_manager_v51 import MAX_ITEMS, N_FEAT
from scripts.hidden_recipe_v55 import permute_spec, GOAL_COL
from scripts.evidence_store_v58 import StoreEnv, EvidenceStore, N_EVID

N_TOTAL = N_FEAT + N_EVID          # 17 dims per slot
QS = 255.0                         # uint8 quantisation scale, identical on write and read
INSTR_MASK = True                  # v56 fix: never argmax into an item already tried-and-failed


# ---------------------------------------------------------------- world construction

def permute_spec_v58(spec, seed):
    """v55 item-index permutation + a cell-ID permutation WITHIN the world's used set (see docstring)."""
    s = dict(permute_spec(spec, seed))
    used = sorted({s["cell"][i] for i in range(s["n_items"]) if s["kind"][i] == "R"})
    if len(used) > 1:
        perm = np.random.default_rng(seed + 7717).permutation(len(used))
        m = {c: used[perm[k]] for k, c in enumerate(used)}
        s["cell"] = [m[c] if s["kind"][i] == "R" else c for i, c in enumerate(s["cell"])]
    s["_cellperm"] = True
    return s


# ---------------------------------------------------------------- the portable policy

class EvidenceNet(nn.Module):
    """Shared per-slot MLP over [own 17 feats || mean-pooled context over valid slots]. NO embeddings.
    Permutation-equivariant by construction: mean-pooling is order-invariant and the scorer is shared."""

    def __init__(self, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * N_TOTAL, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.score.weight, gain=0.01)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.reshape(B, MAX_ITEMS, N_TOTAL)
        valid = x[..., 6] > 0.5
        ctx = (x * valid.unsqueeze(-1).float()).sum(1) / valid.sum(1, keepdim=True).clamp(min=1).float()
        h = torch.cat([x, ctx.unsqueeze(1).expand(-1, MAX_ITEMS, -1)], -1)
        z = self.enc(h)
        logits = self.score(z).squeeze(-1).masked_fill(~valid, -1e9)
        return logits, self.value(z.mean(1)).squeeze(-1)


class ComposerV58:
    """EvidenceNet + Adam + hindsight-CE training, mirroring v55's Composer API."""

    def __init__(self, lr=3e-4, hidden=128):
        self.net = EvidenceNet(hidden).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, epsilon=0.0, temp=1.0, deterministic=False):
        logits, _ = self.net(obs)
        if deterministic:
            if INSTR_MASK:
                f = obs.reshape(-1, MAX_ITEMS, N_TOTAL)
                logits = logits.masked_fill((f[..., 2] > 0.5) & (f[..., 3] < 0.5), -1e9)
            return logits.argmax(-1)
        a = torch.multinomial(F.softmax(logits / temp, -1), 1).squeeze(-1)
        if epsilon > 0:
            valid = obs.reshape(-1, MAX_ITEMS, N_TOTAL)[..., 6] > 0.5
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
            tot += float(loss.detach())
        return tot / n_steps


class BufferV58:
    """Per-WORLD FIFO of (obs, action). uint8 with a single quantisation used on write AND read."""

    def __init__(self, cap=1_500_000):
        self.s = torch.zeros(cap, MAX_ITEMS * N_TOTAL, dtype=torch.uint8, device=DEVICE)
        self.a = torch.zeros(cap, dtype=torch.long, device=DEVICE)
        self.cap, self.n, self.ptr = cap, 0, 0

    def add(self, s, a):
        k = s.shape[0]
        if k == 0:
            return
        if k >= self.cap:
            s, a, k = s[-self.cap:], a[-self.cap:], self.cap
        q = (s.clamp(0, 1) * QS).round().to(torch.uint8)
        end = self.ptr + k
        if end <= self.cap:
            self.s[self.ptr:end], self.a[self.ptr:end] = q, a
        else:
            r = self.cap - self.ptr
            self.s[self.ptr:], self.a[self.ptr:] = q[:r], a[:r]
            self.s[:k - r], self.a[:k - r] = q[r:], a[r:]
        self.ptr = end % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, bs):
        idx = torch.randint(0, self.n, (min(bs, self.n),), device=DEVICE)
        return self.s[idx].float() / QS, self.a[idx]


# ---------------------------------------------------------------- env with the widened observation

class StoreEnvV58(StoreEnv):
    """StoreEnv + the 17-dim observation and a macro-attempt counter. The store is per-WORLD: build one
    env per world and reuse it across that world's goal stream (set_goal between goals)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.msteps_total = 0

    def step(self, g):
        out = super().step(g)
        self.msteps_total += 1
        return out

    def obs(self, zero_store=False):
        base = self.state.reshape(self.num_envs, MAX_ITEMS, N_FEAT)
        if zero_store:
            ev = torch.zeros(self.num_envs, MAX_ITEMS, N_EVID, device=DEVICE)
        else:
            held = torch.zeros(self.num_envs, MAX_ITEMS, dtype=torch.bool, device=DEVICE)
            held[:, :self.n_items] = self.base.inv > 0
            ev = self.store.features(held, None)
        return torch.cat([base, ev], -1).reshape(self.num_envs, -1)


def make_world_env(spec, skill, cfg, seed, goal=None):
    return StoreEnvV58(cfg["num_envs"], spec, skill, cfg, seed=seed,
                       goal=goal if goal is not None else spec["target"], hidden=True)


# ---------------------------------------------------------------- collection / hindsight / eval

def collect_episode_v58(env, composer, cfg, goal, zero_store=False):
    """One macro-episode under a commanded goal. Observations stored GOAL-FREE (rewritten at relabel).
    The store is NOT reset here — it is per-world and persists across episodes and across goals."""
    N, T = env.num_envs, env.macro_budget
    states = torch.zeros(T, N, MAX_ITEMS * N_TOTAL, device=DEVICE)
    actions = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
    unlockstep = torch.full((N, env.n_items), -1, dtype=torch.long, device=DEVICE)
    env.reset(); env.set_goal(goal)
    prev = env.base.unlocked.clone()
    obs = env.obs(zero_store)
    for t in range(T):
        a = composer.act(obs, epsilon=cfg["epsilon"], temp=cfg["temp"])
        s = obs.clone().reshape(N, MAX_ITEMS, N_TOTAL)
        s[..., GOAL_COL] = 0.0
        states[t] = s.reshape(N, -1)
        actions[t] = a
        env.step(a)
        obs = env.obs(zero_store)
        newly = env.post_unlocked & ~prev
        first = (unlockstep == -1) & newly
        unlockstep[first] = t
        prev = env.post_unlocked.clone()
    return states, actions, unlockstep


def relabel_commanded_v58(states, actions, unlockstep, max_samples, goal, gamma=0.7):
    """D4 guard: hindsight toward ONLY the commanded goal. Every incidentally-unlocked item discarded."""
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
    s = states[t, n].clone().reshape(-1, MAX_ITEMS, N_TOTAL)
    s[:, goal, GOAL_COL] = 1.0
    return s.reshape(s.shape[0], -1), actions[t, n], int(idx.shape[0])


@torch.no_grad()
def eval_goal_v58(spec, skill, composer, cfg, seed, goal, store_state=None, zero_store=False):
    """Deterministic mastery on a FRESH grid, but with the agent's CURRENT world knowledge: the store is
    copied in (recipes are world-level, grids are not). The copy is mutated during eval, never the
    training store. zero_store=True is arm Z: same weights, no world knowledge."""
    env = StoreEnvV58(cfg["num_envs"], spec, skill, cfg, seed=seed + 9, goal=goal, hidden=True)
    if store_state is not None and not zero_store:
        env.store.load_state_dict(store_state)
    got = torch.zeros(cfg["num_envs"], dtype=torch.bool, device=DEVICE)
    obs = env.obs(zero_store)
    for _ in range(cfg["macro_budget"]):
        env.step(composer.act(obs, deterministic=True))
        obs = env.obs(zero_store)
        got |= env.post_unlocked[:, goal]
    return float(got.float().mean()), env._prim


def run_goal_v58(env, spec, skill, composer, buf, cfg, seed, goal, r_max=None, train=True,
                 zero_store_eval=False, fixed_budget=False):
    """One commanded goal inside an already-built world env. The store persists; the buffer is the
    world's. train=False (arm M at test) still writes the store — only gradients are switched off."""
    a0, p0, m0 = env._att, env._prim, env.msteps_total
    # zero_store_eval is arm Z: the SAME frozen weights judged without any world knowledge. It changes
    # only what the eval sees — collection still writes the store normally.
    zs, ev0 = eval_goal_v58(spec, skill, composer, cfg, seed, goal, env.store.state_dict(),
                            zero_store=zero_store_eval)
    # fixed_budget (design v3): run EXACTLY r_max rounds on every goal for every arm, and KEEP THE WHOLE
    # LEARNING CURVE. Two defects die together. (1) The early break made `rounds` a FIRST-PASSAGE index,
    # so a goal already mastered at budget 0 was re-evaluated at budget 1 and could fall back under the
    # threshold — a loss that grows with the number of early masteries, i.e. LARGEST FOR THE BETTER ARM.
    # That is the third non-monotonicity of the family that voided designs v1 and v2, caught before
    # freezing this time. (2) With the break, a more competent arm collected FEWER attempts, so its store
    # was smaller at the next goal's entry; at a fixed budget the store SIZE is bit-matched across arms
    # and only its CONTENT differs — which is the treatment, not an artifact.
    # master_per_round[b] = mastery after b rounds of practice on this goal, b = 0..r_max.
    master, rounds, n_eval, ev_prim = zs, 0, 1, ev0
    master_per_round = [round(zs, 4)]
    first_mastered_at = 0 if zs >= cfg["thresh"] else None
    demos, samples, first_demo_att, discovery = [], [], None, None
    for r in range(r_max if r_max is not None else cfg["r_max"]):
        d = k = 0
        for _ in range(cfg["episodes_per_round"]):
            s, a, us = collect_episode_v58(env, composer, cfg, goal)
            hit = int((us[:, goal] >= 0).sum())
            if discovery is None:                # FIRST exposure to this goal, fine-grained
                u = us[:, goal]
                reach = u[u >= 0]
                discovery = dict(
                    frac=round(float((u >= 0).float().mean()), 4),          # resolution 1/num_envs
                    min_step=(int(reach.min()) + 1) if reach.numel() else None,
                    median_step=(int(reach.float().median()) + 1) if reach.numel() else None)
            if hit > 0 and first_demo_att is None:
                # within-episode resolution: macro-steps before this episode + step inside it
                u = us[:, goal]
                within = int(u[u >= 0].min()) + 1
                first_demo_att = (env.msteps_total - m0) - env.macro_budget + within
            d += hit
            ss, aa, kept = relabel_commanded_v58(s, a, us, cfg["max_samples_per_ep"], goal)
            k += kept
            if ss is not None:
                buf.add(ss, aa)
        demos.append(d); samples.append(k)
        if train:
            composer.train_steps(buf, cfg["train_steps_per_round"])
        rounds = r + 1
        master, evp = eval_goal_v58(spec, skill, composer, cfg, seed, goal, env.store.state_dict(),
                                    zero_store=zero_store_eval)
        n_eval += 1; ev_prim += evp
        master_per_round.append(round(master, 4))
        if master >= cfg["thresh"] and first_mastered_at is None:
            first_mastered_at = rounds
        if master >= cfg["thresh"] and not fixed_budget:
            break                                   # legacy path only; design v3 always runs the budget
    # SCORE vs SPEND (calibration-v2 redesign, ARC2_PLAN section 9). The verification audit proved two
    # defects in using raw attempts as the cost metric: (a) 26/54 calibration goal-runs were mastered ON
    # ARRIVAL yet still charged a full mandatory round — dead weight on which no arm can win; (b) a
    # censored goal charged r_max full rounds, which exceeded a frozen-weight arm's entire above-floor
    # allowance, so ONE miss manufactured a REFUTED. `cost` is what the scorer reads: 0 if mastered on
    # arrival, rounds*per_round if mastered, censor_cap*per_round if censored. SPENDING is unchanged —
    # at least one round is always collected and charged to `attempts` (the v53 lesson: the buffer must
    # never silently stop growing).
    per_round = cfg["episodes_per_round"] * cfg["macro_budget"]
    if zs >= cfg["thresh"]:
        cost = 0
    elif master >= cfg["thresh"]:
        cost = rounds * per_round
    else:
        cost = min(rounds, cfg.get("censor_cap", rounds)) * per_round
    return dict(goal=goal, zero_shot=round(zs, 3), rounds=rounds, discovery=discovery, cost=cost,
                master_per_round=master_per_round, first_mastered_at=first_mastered_at,
                mastered_on_arrival=bool(zs >= cfg["thresh"]),
                attempts=env.msteps_total - m0, first_demo_attempt=first_demo_att,
                collect_prim=env._prim - p0, eval_prim=ev_prim, n_eval=n_eval,
                demos_per_round=demos, samples_per_round=samples,
                att=env._att - a0, master=round(master, 3),
                mastered=bool(master >= cfg["thresh"]), buf_n=buf.n)


def cfg_v58(num_envs=256, r_max=10, censor_cap=3):
    """Frozen ARC-2 config. macro_budget 48 (ARC2_PLAN section 3), option_timeout 16 as in v57.
    censor_cap: a censored goal contributes at most this many rounds to the SCORED cost (the run still
    trains for r_max — only the score is capped; ARC2_PLAN section 9)."""
    return dict(num_envs=num_envs, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
                nav_max_steps=40, skill_iters=400, option_timeout=16, macro_budget=48,
                episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
                epsilon=0.05, temp=1.0, thresh=0.6, r_max=r_max, censor_cap=censor_cap,
                skill_stochastic=True, mgr_entropy=0.03, router_iters=0)
