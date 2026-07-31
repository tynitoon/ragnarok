"""v6.0 M1 sanity — DeviceVecCraftWorld dependency structure.

Shows (a) a RANDOM policy unlocks only shallow achievements (deep nodes
near-impossible by chance), and (b) a scripted greedy ORACLE reliably
reaches the deepest node (make_iron_pickaxe) — confirming the tech tree is
completable and the depth gate is real, not a bug.

Usage: python -m scripts.craft_sanity_v6
"""

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, ACH_NAMES, ACH_DEPTH, N_ACH,
    TREE, STONE, COAL, IRON, WOOD, STONE_I, COAL_I, IRON_I,
    WPICK, SPICK, IPICK, TABLE, FURNACE)


def _nearest_action(env, target):
    """Greedy move toward the nearest cell of per-env type `target`
    (full-grid knowledge — an oracle), or collect (4) if standing on it."""
    N, G = env.num_envs, env.G
    mask = env.grid == target.view(N, 1, 1)
    rows = torch.arange(G, device=DEVICE).view(1, G, 1)
    cols = torch.arange(G, device=DEVICE).view(1, 1, G)
    pr, pc = env.pos[:, 0].view(N, 1, 1), env.pos[:, 1].view(N, 1, 1)
    dist = (rows - pr).abs() + (cols - pc).abs()
    dist = torch.where(mask, dist, torch.full_like(dist, G * G * 10))
    flat = dist.view(N, -1).argmin(-1)
    tr, tc = flat // G, flat % G
    on = (tr == env.pos[:, 0]) & (tc == env.pos[:, 1])
    dr, dc = torch.sign(tr - env.pos[:, 0]), torch.sign(tc - env.pos[:, 1])
    z = torch.zeros(N, dtype=torch.long, device=DEVICE)
    move = torch.where(dr < 0, z + 0, torch.where(dr > 0, z + 1,
            torch.where(dc < 0, z + 2, z + 3)))
    return torch.where(on, z + 4, move)


def _oracle_action(env):
    inv = env.inv
    N = env.num_envs
    a = torch.full((N,), -1, dtype=torch.long, device=DEVICE)

    def setif(mask, act):
        m = mask & (a < 0)
        a[m] = act

    # craft priority (deepest first), only if not already owned
    setif((inv[:, WOOD] >= 1) & (inv[:, COAL_I] >= 1) & (inv[:, IRON_I] >= 1)
          & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] >= 1) & (inv[:, IPICK] == 0), 9)
    setif((inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, FURNACE] == 0), 8)
    setif((inv[:, WOOD] >= 1) & (inv[:, STONE_I] >= 1) & (inv[:, TABLE] >= 1)
          & (inv[:, SPICK] == 0), 7)
    setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] >= 1) & (inv[:, WPICK] == 0), 6)
    setif((inv[:, WOOD] >= 1) & (inv[:, TABLE] == 0), 5)

    # otherwise navigate to the next needed resource
    target = torch.full((N,), TREE, dtype=torch.long, device=DEVICE)  # default: wood
    has_w = inv[:, WPICK] >= 1
    has_s = inv[:, SPICK] >= 1
    target[has_w & (~has_s) & (inv[:, STONE_I] < 2)] = STONE
    target[has_s & (inv[:, COAL_I] < 1) & (inv[:, IRON_I] >= 1)] = COAL
    target[has_s & (inv[:, IRON_I] < 1)] = IRON
    nav = _nearest_action(env, target)
    a = torch.where(a < 0, nav, a)
    return a


def _run(policy, n=256, steps=400, episodes=3):
    """Mean fraction of envs that unlock each achievement within an episode,
    averaged over `episodes` resets."""
    env = DeviceVecCraftWorld(n, grid=9, view=5, max_steps=steps, seed=0)
    acc = torch.zeros(N_ACH, device=DEVICE)
    for _ in range(episodes):
        env.reset()
        ep = torch.zeros(n, N_ACH, dtype=torch.bool, device=DEVICE)
        for _ in range(steps):
            a = policy(env)
            env.step(a)
            ep |= env.unlocked
        acc += ep.float().mean(0)
    return (acc / episodes).cpu()


def main():
    print(f"[craft-sanity] device={DEVICE}", flush=True)
    rand = _run(lambda e: torch.randint(0, 10, (e.num_envs,), device=DEVICE))
    orac = _run(_oracle_action)
    print(f"\n  {'achievement':22s} {'depth':>5} {'random':>8} {'oracle':>8}")
    for i, nm in enumerate(ACH_NAMES):
        print(f"  {nm:22s} {ACH_DEPTH[i]:>5} {rand[i]:>8.2f} {orac[i]:>8.2f}",
              flush=True)
    deep = orac[N_ACH - 1].item()       # make_iron_pickaxe
    print(f"\n  oracle reaches deepest node (iron_pickaxe): {deep:.2f} | "
          f"random: {rand[N_ACH-1]:.2f}")
    ok = deep > 0.8 and rand[N_ACH - 1] < 0.1
    print(f"  -> {'SUBSTRATE OK: deep tree completable, depth-gated (random fails deep)' if ok else 'CHECK substrate'}",
          flush=True)


if __name__ == "__main__":
    main()
