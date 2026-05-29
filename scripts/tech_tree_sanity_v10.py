"""v10 sanity — DeviceVecTechTree: random trees are completable (scripted
oracle reaches the deepest item) and depth-gated (random policy stalls)."""

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import DeviceVecTechTree, gen_tree


def _needed(spec):
    """Total quantity of each item consumed to build 1 target (recursive)."""
    need = [0] * spec["n_items"]
    need[spec["target"]] = 1
    for i in range(spec["n_items"] - 1, -1, -1):
        if need[i] == 0:
            continue
        for j, c in spec["inputs"][i].items():
            need[j] += need[i] * c
        for t in spec["tools"][i]:
            need[t] = max(need[t], 1)
        if spec["tool"][i] >= 0:
            need[spec["tool"][i]] = max(need[spec["tool"][i]], 1)
    return need


def _bfs_order(spec):
    """Topological order to build the target (items by increasing depth)."""
    order = sorted(range(spec["n_items"]), key=lambda i: spec["depth"][i])
    return [i for i in order if spec["depth"][i] <= spec["depth"][spec["target"]]]


def _nearest_move(env, target_cell):
    """Move toward nearest cell of `target_cell` type, or collect (4) if on it."""
    N, G = env.num_envs, env.G
    mask = env.grid == target_cell.view(N, 1, 1)
    rows = torch.arange(G, device=DEVICE).view(1, G, 1)
    cols = torch.arange(G, device=DEVICE).view(1, 1, G)
    pr, pc = env.pos[:, 0].view(N, 1, 1), env.pos[:, 1].view(N, 1, 1)
    dist = torch.where(mask, (rows - pr).abs() + (cols - pc).abs(),
                       torch.full((N, G, G), G * G * 9, device=DEVICE))
    flat = dist.view(N, -1).argmin(-1)
    tr, tc = flat // G, flat % G
    on = (tr == env.pos[:, 0]) & (tc == env.pos[:, 1])
    dr, dc = torch.sign(tr - env.pos[:, 0]), torch.sign(tc - env.pos[:, 1])
    z = torch.zeros(N, dtype=torch.long, device=DEVICE)
    move = torch.where(dr < 0, z, torch.where(dr > 0, z + 1,
                       torch.where(dc < 0, z + 2, z + 3)))
    return torch.where(on, z + 4, move)


def _oracle(env, spec, need, craft_action_of):
    """Greedy: build items in dependency order to their needed counts."""
    N = env.num_envs
    a = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
    for it in _bfs_order(spec):
        if need[it] == 0:
            continue
        short = (a < 0) & (env.inv[:, it] < need[it])
        if not bool(short.any()):
            continue
        if spec["kind"][it] == "R":
            tgt = torch.full((N,), spec["cell"][it], device=DEVICE)
            mv = _nearest_move(env, tgt)
            a = torch.where(short, mv, a)
        else:
            a = torch.where(short, torch.full((N,), craft_action_of[it],
                                              device=DEVICE), a)
    a[a < 0] = 4
    return a


def main():
    for seed in range(3):
        spec = gen_tree(seed, n_items=14, n_base_res=2)
        tgt, tdep = spec["target"], spec["depth"][spec["target"]]
        nR = sum(1 for k in spec["kind"] if k == "R")
        print(f"\n[tree {seed}] {spec['n_items']} items "
              f"({nR} resources, {spec['n_items']-nR} crafts) | target item "
              f"{tgt} depth {tdep} | max depth {max(spec['depth'])}", flush=True)
        need = _needed(spec)
        craft_action_of = {it: 5 + k for k, it in enumerate(spec["craft_actions"])}

        # random policy
        e = DeviceVecTechTree(256, spec, grid=11, max_steps=300, seed=seed)
        unlocked = torch.zeros(256, spec["n_items"], dtype=torch.bool, device=DEVICE)
        for _ in range(300):
            e.step(torch.randint(0, e.action_dim, (256,), device=DEVICE))
            unlocked |= e.unlocked
        rand_reach = unlocked.float().mean(0)
        rand_target = float(rand_reach[tgt])

        # scripted oracle
        e2 = DeviceVecTechTree(256, spec, grid=11, max_steps=400, seed=seed + 100)
        got = torch.zeros(256, dtype=torch.bool, device=DEVICE)
        for _ in range(400):
            e2.step(_oracle(e2, spec, need, craft_action_of))
            got |= e2.inv[:, tgt] > 0
        orac_target = float(got.float().mean())
        print(f"  oracle reaches target: {orac_target:.2f} | "
              f"random reaches target: {rand_target:.2f} | obs_dim {e.obs_dim} "
              f"actions {e.action_dim}", flush=True)
    print("\n  -> if oracle>0.8 and random<0.3 across trees: substrate OK", flush=True)


if __name__ == "__main__":
    main()
