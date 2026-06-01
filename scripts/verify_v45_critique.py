"""Independently verify the two most damaging review claims about v45, so the
retraction is evidence-backed not relayed:
 (1) BLIND FLOOR: a non-learning random policy already scores ~0.45 on the held-out
     tasks -> the mastery=0.8 bar sits just above 'do nothing intelligent'.
 (2) MEMORISATION LEAK: corr(goal_item_index, goal_cell_type) across train trees,
     and how many train trees share each held-out tree's exact (item->cell) target
     (tree1007 is claimed to be the IDENTICAL fixed target in all train trees).
Not a scored experiment — a diagnostic audit.
"""
import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.ad_techtree_v45 import make_env, choose_task, default_cfg, compute_pads


class A:
    n_items, grid, max_steps, n_resource, max_cells, num_envs = 12, 7, 40, 4, 0, 256
    train_trees, test_trees, src_iters, n_steps, log_envs = 96, 8, 50, 20, 16
    d_model, n_layers, ctx, distill_steps, eval_episodes = 160, 5, 256, 10000, 32
    base_iters, eval_every = 120, 5


cfg = default_cfg(A)
allseeds = list(range(96)) + list(range(1000, 1008))
mc, ma = compute_pads(allseeds, cfg)
cfg["max_cells"], cfg["max_actions"] = mc, ma

# ---- (2) memorisation leak -------------------------------------------------
train_map = []
for s in range(96):
    sp = gen_tree(s, n_items=cfg["n_items"], n_base_res=cfg["n_base_res"])
    g, _, _ = choose_task(sp)
    train_map.append((g, sp["cell"][g]))
gi = [m[0] for m in train_map]
gc = [m[1] for m in train_map]
print(f"LEAK corr(goal_item_idx, goal_cell_type) over 96 train trees = "
      f"{np.corrcoef(gi, gc)[0, 1]:+.3f}", flush=True)
print("held-out (item->cell) target & how many TRAIN trees share it exactly:", flush=True)
for hs in range(1000, 1008):
    sp = gen_tree(hs, n_items=cfg["n_items"], n_base_res=cfg["n_base_res"])
    g, _, _ = choose_task(sp)
    cell = sp["cell"][g]
    same = sum(1 for m in train_map if m == (g, cell))
    print(f"  tree{hs}: item{g}->cell{cell} | identical target in {same}/96 train trees", flush=True)

# ---- (1) blind (non-learning) floor ---------------------------------------
print("BLIND random-policy single-episode success on held-out trees:", flush=True)
torch.manual_seed(0)
floors = []
for hs in range(1000, 1008):
    env, _, _, _ = make_env(hs, cfg, 256, env_seed=hs + 555)
    env.reset()
    ever = torch.zeros(256, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["max_steps"]):
        a = torch.randint(0, cfg["max_actions"], (256,), device=DEVICE)
        _, _, term, _, _ = env.step(a)
        ever |= term
    f = float(ever.float().mean())
    floors.append(f)
    print(f"  tree{hs}: blind success {f:.3f}", flush=True)
print(f"  -> uniform-random(15 actions) floor mean {np.mean(floors):.3f}", flush=True)

print("SMART-BLIND random-policy (move/collect only, NO craft no-ops) success:", flush=True)
sfloors = []
for hs in range(1000, 1008):
    env, _, _, _ = make_env(hs, cfg, 256, env_seed=hs + 555)
    env.reset()
    ever = torch.zeros(256, dtype=torch.bool, device=DEVICE)
    for _ in range(cfg["max_steps"]):
        a = torch.randint(0, 5, (256,), device=DEVICE)        # actions 0-4 only
        _, _, term, _, _ = env.step(a)
        ever |= term
    f = float(ever.float().mean())
    sfloors.append(f)
    print(f"  tree{hs}: smart-blind success {f:.3f}", flush=True)
print(f"  -> SMART-blind (move/collect) floor mean {np.mean(sfloors):.3f} "
      f"(mastery bar is 0.80; non-identifying policy)", flush=True)
