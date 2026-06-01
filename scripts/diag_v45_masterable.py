"""Diagnostic for v45: is the chosen task masterable by from-scratch PPO at the
intended scale, and how many ENVIRONMENT EPISODES does it take? This sets whether
there is any headroom for in-context AD to beat, and confirms the AD source data
will contain genuine learning-progress. Not a scored experiment.
"""
import time
import torch
from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.ad_techtree_v45 import make_env, ppo_success, default_cfg, compute_pads


class A:  # mimic argparse namespace with the default (non-smoke) scale
    n_items, grid, max_steps, n_resource, max_cells, num_envs = 12, 7, 40, 4, 0, 256
    train_trees, test_trees, src_iters, n_steps, log_envs = 24, 8, 80, 20, 16
    d_model, n_layers, ctx, distill_steps, eval_episodes = 128, 4, 128, 3000, 16
    base_iters, eval_every = 120, 5


cfg = default_cfg(A)
seeds = list(range(A.train_trees)) + list(range(1000, 1000 + A.test_trees))
mc, ma = compute_pads(seeds, cfg)
cfg["max_cells"], cfg["max_actions"] = mc, ma
print(f"device={DEVICE} max_cells={mc} max_actions={ma}", flush=True)

torch.manual_seed(0)
for hs in [1000, 1001, 1002, 0, 5]:
    env, spec, goal, grant = make_env(hs, cfg, cfg["num_envs"])
    env_ev, *_ = make_env(hs, cfg, 256, env_seed=hs + 555)
    ppo = DiscretePPO(env.obs_dim, ma, hidden=cfg["hidden"], entropy=cfg["entropy"])
    t0 = time.perf_counter()
    cum_ep = 0
    curve = []
    for it in range(1, A.base_iters + 1):
        roll = ppo.collect(env, cfg["n_steps"]); ppo.update(roll)
        cum_ep += int(roll["done"].sum().item())
        if it % 10 == 0:
            s = ppo_success(ppo, env_ev)
            curve.append((it, cum_ep, round(s, 2)))
    gname = spec["kind"][goal]
    print(f"tree {hs} (goal kind={gname}, depth={spec['depth'][goal]}): "
          f"{curve} | {time.perf_counter()-t0:.0f}s", flush=True)
