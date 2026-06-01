"""v45 — ALGORITHM DISTILLATION on PROCEDURAL tech-trees (in-context reuse, FAIR).
FROZEN per preregistration.md entry 2026-06-01.

The literature survey's #1 missed angle: in-context RL. Distil from-scratch PPO
LEARNING-HISTORIES (across-episode streams where competence visibly improves) on
a DISTRIBUTION of procedural tech-trees into a causal transformer; then test
gradient-free IN-CONTEXT mastery on HELD-OUT trees (unseen generator seeds) vs
from-scratch PPO. The reusable object is the LEARNING ALGORITHM ITSELF, amortised
over the task distribution — NOT a notion-as-feature (v36-42 nulls) nor a
model-to-plan (v43/44).

FAIR accounting (frozen): unit = ENVIRONMENT EPISODES, identical num_envs in both
arms (no parallel-actor discount); distillation = a one-time cost amortised over
the held-out set, disclosed. Decision: POSITIVE iff in-context episodes-to-mastery
<= 0.5x from-scratch PPO, every seed. NULL otherwise — honest either way.

Task family (v7-style, navigation-forced so there is something to amortise): per
tree pick the DEEPEST node that requires collecting >=1 raw resource; grant its
other prerequisites; the agent must navigate to the right cell-type(s) and
collect/craft. The cell-type<->item mapping is tree-specific, so a held-out tree
must be solved from in-context experience, not from the goal index.

Usage: python -m scripts.ad_techtree_v45 [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from ragnarok.environments.tech_tree import DeviceVecTechTree, gen_tree


# ----------------------------------------------------------------------------- #
#  Task construction (tree-agnostic, navigation-forced)                         #
# ----------------------------------------------------------------------------- #
def _transitive_prereqs(spec, i, seen=None):
    if seen is None:
        seen = set()
    for j in spec["true_pre"][i]:
        if j not in seen:
            seen.add(j)
            _transitive_prereqs(spec, j, seen)
    return seen


def choose_task(spec):
    """Return (goal_idx, grant_vec, must_collect). FROZEN-at-pilot task family
    (disclosed): the DEEPEST RESOURCE node, with its prerequisite tool-chain
    granted. The agent must navigate to the goal's cell-type and collect it. The
    cell-type<->item mapping is tree-specific, so a held-out tree must be solved
    from IN-CONTEXT reward, not the goal index. Craft goals were DROPPED at pilot
    (diag_v45_masterable showed many are unlearnable by from-scratch PPO in budget,
    so they yield no learning-progress to distil and no baseline to beat)."""
    n, kind, depth = spec["n_items"], spec["kind"], spec["depth"]
    res = [i for i in range(n) if kind[i] == "R"]
    goal = max(res, key=lambda i: depth[i])              # deepest resource
    grant = [0] * n
    for j in _transitive_prereqs(spec, goal):            # grant the gating tool-chain
        if j != goal:
            grant[j] = 5
    return goal, grant, {goal}


def make_env(seed, cfg, num_envs, env_seed=None):
    spec = gen_tree(seed, n_items=cfg["n_items"], n_base_res=cfg["n_base_res"])
    goal, grant, _ = choose_task(spec)
    env = DeviceVecTechTree(num_envs, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=cfg["max_steps"], n_resource=cfg["n_resource"],
                            goal=goal, grant=grant,
                            seed=seed if env_seed is None else env_seed,
                            max_cells=cfg["max_cells"])
    return env, spec, goal, grant


# ----------------------------------------------------------------------------- #
#  The AD causal transformer                                                    #
# ----------------------------------------------------------------------------- #
class ADTransformer(nn.Module):
    """Token_t = obs_emb(o_t) + act_emb(a_{t-1}) + rew_emb(r_{t-1}) + pos; causal
    self-attention; predict a_t. A length-L window spans several episodes, so
    in-context improvement is visible."""
    def __init__(self, obs_dim, n_actions, d_model=128, n_layers=4, n_heads=4,
                 max_len=256):
        super().__init__()
        self.n_actions = n_actions
        self.max_len = max_len
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.act_emb = nn.Embedding(n_actions + 1, d_model)   # +1 = START
        self.rew_emb = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_actions)

    def forward(self, obs, prev_act, prev_rew):
        # obs (B,L,obs_dim) float; prev_act (B,L) long; prev_rew (B,L) float
        B, L, _ = obs.shape
        tok = (self.obs_emb(obs) + self.act_emb(prev_act)
               + self.rew_emb(prev_rew.unsqueeze(-1)) + self.pos[:, :L])
        mask = torch.triu(torch.ones(L, L, device=obs.device, dtype=torch.bool), 1)
        return self.head(self.tr(tok, mask=mask))             # (B,L,n_actions)


# ----------------------------------------------------------------------------- #
#  Source data: log from-scratch PPO learning-histories                         #
# ----------------------------------------------------------------------------- #
def run_source(env, max_actions, cfg):
    """From-scratch PPO on `env`; return per-env time-ordered streams
    O (E,T,obs), A (E,T), R (E,T) for the first cfg['log_envs'] envs, plus the
    total environment episodes consumed."""
    ppo = DiscretePPO(env.obs_dim, max_actions, hidden=cfg["hidden"],
                      entropy=cfg["entropy"])
    Os, As, Rs = [], [], []
    le = cfg["log_envs"]
    cum_ep = 0
    for _ in range(cfg["src_iters"]):
        roll = ppo.collect(env, cfg["n_steps"])
        ppo.update(roll)
        cum_ep += int(roll["done"].sum().item())
        Os.append(roll["obs"][:le].to(torch.float16).cpu())
        As.append(roll["act"][:le].cpu())
        Rs.append(roll["rew"][:le].to(torch.float16).cpu())
    O = torch.cat(Os, 1)                                       # (le, iters*T, obs)
    A = torch.cat(As, 1)
    R = torch.cat(Rs, 1)
    return O, A, R, cum_ep


def build_dataset(streams_O, streams_A, streams_R, n_actions):
    """Stack per-tree streams into (S,T,*) and precompute prev-action / prev-rew
    (START / 0 prepended) for windowed sampling."""
    O = torch.cat(streams_O, 0)                               # (S,T,obs) fp16
    A = torch.cat(streams_A, 0).long()                        # (S,T)
    R = torch.cat(streams_R, 0).float()                       # (S,T)
    S, T = A.shape
    start = torch.full((S, 1), n_actions, dtype=torch.long)   # START token
    A_prev = torch.cat([start, A[:, :-1]], 1)                 # a_{t-1}
    R_prev = torch.cat([torch.zeros(S, 1), R[:, :-1]], 1)     # r_{t-1}
    return dict(O=O, A=A, A_prev=A_prev, R_prev=R_prev, S=S, T=T)


def distill(ds, obs_dim, n_actions, cfg):
    model = ADTransformer(obs_dim, n_actions, d_model=cfg["d_model"],
                          n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                          max_len=cfg["ctx"] + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["distill_lr"],
                            weight_decay=1e-2)
    S, T, L = ds["S"], ds["T"], cfg["ctx"]
    B = cfg["distill_batch"]
    model.train()
    last = 0.0
    for step in range(cfg["distill_steps"]):
        si = torch.randint(0, S, (B,))
        st = torch.randint(0, T - L, (B,))
        gi = (st.unsqueeze(1) + torch.arange(L)).clamp(max=T - 1)   # (B,L) positions
        o = ds["O"][si.unsqueeze(1), gi].to(DEVICE).float()
        ap = ds["A_prev"][si.unsqueeze(1), gi].to(DEVICE)
        rp = ds["R_prev"][si.unsqueeze(1), gi].to(DEVICE)
        tgt = ds["A"][si.unsqueeze(1), gi].to(DEVICE)
        logits = model(o, ap, rp)
        loss = F.cross_entropy(logits.reshape(-1, n_actions), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    return model, last


# ----------------------------------------------------------------------------- #
#  Evaluation: in-context (frozen model) and from-scratch PPO baseline          #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def eval_incontext(model, env, n_actions, cfg):
    """Roll the frozen model on `env`, each env accumulating its OWN across-episode
    history (gradient-free). Return per-episode success fraction + episodes/env run."""
    model.eval()
    N, L = env.num_envs, cfg["ctx"]
    env.reset()
    obs = env.state
    prev_a = torch.full((N,), n_actions, dtype=torch.long, device=DEVICE)
    prev_r = torch.zeros(N, device=DEVICE)
    bo, ba, br = deque(maxlen=L), deque(maxlen=L), deque(maxlen=L)
    ep = torch.zeros(N, dtype=torch.long, device=DEVICE)
    maxE = cfg["eval_episodes"]
    succ = torch.zeros(N, maxE + 1, dtype=torch.bool, device=DEVICE)
    ar = torch.arange(N, device=DEVICE)
    steps, cap = 0, maxE * cfg["max_steps"] * 3
    while int(ep.min()) < maxE and steps < cap:
        bo.append(obs); ba.append(prev_a); br.append(prev_r)
        o = torch.stack(tuple(bo), 1)
        ap = torch.stack(tuple(ba), 1)
        rp = torch.stack(tuple(br), 1)
        logits = model(o, ap, rp)[:, -1]
        a = torch.distributions.Categorical(logits=logits).sample()   # SAMPLE: AD
        obs, r, term, trunc, done = env.step(a)                       # explores in-context
        succ[ar, ep.clamp(max=maxE)] |= term
        ep += done.long()
        prev_a, prev_r = a, r
        steps += 1
    frac = [float(succ[:, e].float().mean()) for e in range(maxE)]
    return frac, steps


@torch.no_grad()
def ppo_success(ppo, env_eval):
    env_eval.reset()
    ever = torch.zeros(env_eval.num_envs, dtype=torch.bool, device=DEVICE)
    obs = env_eval.state
    for _ in range(env_eval.max_steps):
        obs, _, term, _, _ = env_eval.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean())


def baseline_ppo(env, env_eval, max_actions, cfg):
    """From-scratch PPO on a held-out tree; return episodes-to-mastery (or None)
    and final success."""
    ppo = DiscretePPO(env.obs_dim, max_actions, hidden=cfg["hidden"],
                      entropy=cfg["entropy"])
    cum_ep, mastered_at, final = 0, None, 0.0
    for it in range(1, cfg["base_iters"] + 1):
        roll = ppo.collect(env, cfg["n_steps"])
        ppo.update(roll)
        cum_ep += int(roll["done"].sum().item())
        if it % cfg["eval_every"] == 0:
            final = ppo_success(ppo, env_eval)
            if mastered_at is None and final >= cfg["mastery"]:
                mastered_at = cum_ep
                break
    return mastered_at, final


def mastery_episode(frac, thr):
    for e, f in enumerate(frac):
        if f >= thr:
            return e + 1                                       # episodes/env to mastery
    return None


# ----------------------------------------------------------------------------- #
#  Main                                                                          #
# ----------------------------------------------------------------------------- #
def default_cfg(args):
    return dict(
        n_items=args.n_items, n_base_res=2, grid=args.grid, view=5,
        max_steps=args.max_steps, n_resource=args.n_resource, max_cells=args.max_cells,
        num_envs=args.num_envs, hidden=256, entropy=0.02,
        src_iters=args.src_iters, n_steps=args.n_steps, log_envs=args.log_envs,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=4, ctx=args.ctx,
        distill_steps=args.distill_steps, distill_batch=256, distill_lr=3e-4,
        eval_episodes=args.eval_episodes, base_iters=args.base_iters,
        eval_every=args.eval_every, mastery=0.8)


def compute_pads(seeds_trees, cfg):
    """Max cell-count and action-count over all trees used (train+test)."""
    mc, ma = 0, 0
    for s in seeds_trees:
        sp = gen_tree(s, n_items=cfg["n_items"], n_base_res=cfg["n_base_res"])
        mc = max(mc, sp["n_cells"])
        ma = max(ma, 5 + len(sp["craft_actions"]))
    return mc + 1, ma                                          # +1 so WALL=n_cells fits


def run_seed(seed, cfg, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_seeds = list(range(args.train_trees))
    test_seeds = list(range(1000, 1000 + args.test_trees))     # disjoint seed band

    # ---- distillation: log learning-histories on TRAIN trees ----
    sO, sA, sR, src_eps = [], [], [], 0
    for ts in train_seeds:
        env, *_ = make_env(ts, cfg, cfg["num_envs"])
        O, A, R, ep = run_source(env, cfg["max_actions"], cfg)
        sO.append(O); sA.append(A); sR.append(R); src_eps += ep
    ds = build_dataset(sO, sA, sR, cfg["max_actions"])
    obs_dim = ds["O"].shape[-1]
    model, dloss = distill(ds, obs_dim, cfg["max_actions"], cfg)

    # ---- evaluation on HELD-OUT trees ----
    rows = []
    for hs in test_seeds:
        env_ic, spec, goal, grant = make_env(hs, cfg, cfg["num_envs"])
        frac, _ = eval_incontext(model, env_ic, cfg["max_actions"], cfg)
        ic_ep_per_env = mastery_episode(frac, cfg["mastery"])
        ic_total = None if ic_ep_per_env is None else ic_ep_per_env * cfg["num_envs"]

        env_tr, *_ = make_env(hs, cfg, cfg["num_envs"])
        env_ev, *_ = make_env(hs, cfg, 256, env_seed=hs + 555)
        base_ep, base_final = baseline_ppo(env_tr, env_ev, cfg["max_actions"], cfg)

        rows.append(dict(tree=hs, ic_best=round(max(frac), 3),
                         ic_ep_per_env=ic_ep_per_env, ic_total_episodes=ic_total,
                         base_episodes=base_ep, base_final=round(base_final, 3),
                         ic_curve=[round(f, 2) for f in frac]))   # per-episode (diagnose)
    return dict(seed=seed, distill_loss=round(dloss, 4),
                distill_source_episodes=src_eps, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-items", type=int, default=12)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--max-cells", type=int, default=0)          # 0 => auto
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--train-trees", type=int, default=24)
    p.add_argument("--test-trees", type=int, default=8)
    p.add_argument("--src-iters", type=int, default=80)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--log-envs", type=int, default=16)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--distill-steps", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--base-iters", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        (args.seeds, args.n_items, args.grid, args.max_steps, args.num_envs,
         args.train_trees, args.test_trees, args.src_iters, args.log_envs,
         args.d_model, args.n_layers, args.ctx, args.distill_steps,
         args.eval_episodes, args.base_iters) = (
            [0], 8, 5, 25, 64, 4, 2, 12, 8, 64, 2, 48, 300, 10, 16)

    cfg = default_cfg(args)
    all_seeds = list(range(args.train_trees)) + list(range(1000, 1000 + args.test_trees))
    auto_mc, ma = compute_pads(all_seeds, cfg)
    cfg["max_cells"] = args.max_cells if args.max_cells else auto_mc
    cfg["max_actions"] = ma

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v45-AD] device={DEVICE} | Algorithm Distillation on procedural tech-trees "
          f"| train={args.train_trees} test={args.test_trees} trees | max_cells="
          f"{cfg['max_cells']} max_actions={ma} ctx={cfg['ctx']} | FAIR: episodes, "
          f"matched num_envs={cfg['num_envs']}, distill amortised | seeds={args.seeds}",
          flush=True)
    t0 = time.perf_counter()

    runs = []
    for s in args.seeds:
        r = run_seed(s, cfg, args)
        runs.append(r)
        amort = r["distill_source_episodes"] / max(1, args.test_trees)
        for row in r["rows"]:
            print(f"  s{s} tree{row['tree']}: in-context best={row['ic_best']} "
                  f"@ {row['ic_ep_per_env']} ep/env (total {row['ic_total_episodes']}) "
                  f"| PPO scratch {row['base_episodes']} ep (final {row['base_final']})",
                  flush=True)
        print(f"  s{s}: distill_loss={r['distill_loss']} source_eps={r['distill_source_episodes']} "
              f"(amortised/tree {amort:.0f}) | {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- verdict (frozen decision rule) ----
    def seed_win(r):
        ok = []
        for row in r["rows"]:
            if row["ic_total_episodes"] is None:
                return False
            if row["base_episodes"] is None:                   # scratch failed -> IC wins iff it mastered
                ok.append(True)
            else:
                ok.append(row["ic_total_episodes"] <= 0.5 * row["base_episodes"])
        return all(ok)

    wins = [seed_win(r) for r in runs]
    positive = all(wins) and len(wins) == len(args.seeds)

    # ---- honest amortisation accounting (the review's main attack surface) ----
    base_eps = [row["base_episodes"] for r in runs for row in r["rows"] if row["base_episodes"]]
    ic_eps = [row["ic_total_episodes"] for r in runs for row in r["rows"] if row["ic_total_episodes"]]
    n_trees = sum(len(r["rows"]) for r in runs)
    D = sum(r["distill_source_episodes"] for r in runs) / max(1, len(runs))
    avg_base = (sum(base_eps) / len(base_eps)) if base_eps else None
    avg_ic = (sum(ic_eps) / len(ic_eps)) if ic_eps else None
    breakeven = (round(D / (avg_base - avg_ic)) if avg_ic and avg_base and avg_base > avg_ic
                 else None)
    amort = dict(in_context_mastered=len(ic_eps), held_out_total=n_trees,
                 avg_in_context_episodes=avg_ic, avg_scratch_episodes=avg_base,
                 one_time_distill_episodes=round(D), breakeven_tasks=breakeven)
    print(f"  AMORTISE: in-context mastered {len(ic_eps)}/{n_trees} held-out | avg in-context "
          f"{None if avg_ic is None else round(avg_ic)} vs scratch "
          f"{None if avg_base is None else round(avg_base)} ep/task | one-time distill "
          f"{round(D)} ep -> TOTAL-compute break-even ~{breakeven} held-out tasks", flush=True)

    verdict = (
        "AD IN-CONTEXT REUSE POSITIVE — on held-out procedural tech-trees the distilled "
        "transformer masters in <= half the ENVIRONMENT EPISODES of from-scratch PPO, "
        "every seed, at fair accounting (matched num_envs; distillation amortised). The "
        "reusable object is the learning algorithm itself. REVIEW (3-4 adversarial agents) "
        "BEFORE reporting." if positive else
        f"NULL/PARTIAL — seed wins {wins}. In-context reuse does not beat from-scratch PPO "
        f"at fair accounting on this distribution. Honest + important: bounds where AD pays.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v45_ad_techtree.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, cfg={k: cfg[k] for k in
                       ("n_items", "grid", "max_steps", "num_envs", "max_cells",
                        "max_actions", "ctx", "src_iters", "distill_steps")},
                       train_trees=args.train_trees, test_trees=args.test_trees,
                       runs=runs, positive=positive, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
