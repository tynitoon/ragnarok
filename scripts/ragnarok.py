"""Ragnarok — a runnable, watchable developmental agent.

Drop the agent into a small crafting world it knows NOTHING about. It:
  1. LEARNS basic skills (how to collect each resource), caching them.
  2. FIGURES OUT THE RECIPES by experimenting (which items need which).
  3. For any target you ask for, PLANS the steps and BUILDS it — live, with
     an ASCII view of the world so you can watch.

First run trains + caches (a few minutes). After that it loads instantly and
you can ask it to build anything and watch it plan + craft.

Usage:
  python -m scripts.ragnarok                      # build iron_pickaxe, watch
  python -m scripts.ragnarok --target furnace     # build something else
  python -m scripts.ragnarok --retrain            # relearn from scratch
  python -m scripts.ragnarok --no-render          # quiet (no ASCII frames)
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ITEMS, EMPTY, TREE, STONE, COAL, IRON,
    WOOD, STONE_I, COAL_I, IRON_I, A_WOOD)
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.craft_endtoend_v6 import _train_collect, _goal_onehot
from scripts.model_based_v9 import (
    _learn_preconditions, _plan, ITEM_INFO, NAME, ALL, COLLECT_CELL, _goal_of)

CELL_SYM = {EMPTY: ".", TREE: "T", STONE: "S", COAL: "c", IRON: "i"}
GATHER = 4              # how many of a resource to gather per collect step


# --------------------------------------------------------------------------
def _skill_path(d, item):
    return os.path.join(d, f"skill_{NAME[item]}.pt")


def _save_skill(ppo, path):
    torch.save(ppo.net.state_dict(), path)


def _load_skill(path, obs_dim):
    ppo = DiscretePPO(obs_dim, 10, hidden=256)
    ppo.net.load_state_dict(torch.load(path, weights_only=False))
    ppo.net.eval()
    return ppo


def learn_or_load(cfg, out_dir, retrain):
    os.makedirs(out_dir, exist_ok=True)
    goal_obs_dim = DeviceVecCraftWorld(1, grid=cfg["grid"], view=cfg["view"],
                                       goal=A_WOOD).obs_dim
    skills = {}
    have_cache = all(os.path.exists(_skill_path(out_dir, it))
                     for it in (WOOD, STONE_I, COAL_I, IRON_I)) \
        and os.path.exists(os.path.join(out_dir, "rules.json"))
    if have_cache and not retrain:
        print("[ragnarok] loading what I already learned...", flush=True)
        for it in (WOOD, STONE_I, COAL_I, IRON_I):
            skills[it] = _load_skill(_skill_path(out_dir, it), goal_obs_dim)
        with open(os.path.join(out_dir, "rules.json")) as f:
            raw = json.load(f)
        rules = {int(k): set(v) for k, v in raw.items()}
        print("  ...loaded 4 skills + the recipe book.\n", flush=True)
        return skills, rules

    print("[ragnarok] I know nothing yet. Learning the basics by doing...",
          flush=True)
    for it in (WOOD, STONE_I, COAL_I, IRON_I):
        print(f"  learning to collect {NAME[it]} ...", flush=True)
        ppo, s = _train_collect(COLLECT_CELL[it], cfg)
        _save_skill(ppo, _skill_path(out_dir, it))
        skills[it] = ppo
        print(f"    -> can now get {NAME[it]} reliably ({s:.2f}).", flush=True)

    print("\n[ragnarok] now experimenting to figure out the RECIPES "
          "(what needs what)...", flush=True)
    rules = _learn_preconditions(skills, cfg)
    with open(os.path.join(out_dir, "rules.json"), "w") as f:
        json.dump({str(k): sorted(v) for k, v in rules.items()}, f, indent=2)
    print("  -> recipe book learned.\n", flush=True)
    return skills, rules


def _render(env, doing):
    g = env.grid[0].tolist()
    pr, pc = env.pos[0].tolist()
    G = env.G
    rows = []
    for r in range(G):
        line = "".join("@" if (r == pr and c == pc)
                       else CELL_SYM.get(g[r][c], "?") for c in range(G))
        rows.append("   " + line)
    inv = env.inv[0].tolist()
    held = ", ".join(f"{NAME[i]}x{inv[i]}" for i in ALL if inv[i] > 0) or "(nothing)"
    print("\n".join(rows), flush=True)
    print(f"   holding: {held}\n   doing: get {NAME[doing]}\n", flush=True)


def play(target, skills, rules, cfg, render=True):
    plan = _plan(target, rules)
    if plan is None:
        print(f"[ragnarok] I can't see how to make {NAME[target]} yet.", flush=True)
        return False
    print(f"[ragnarok] you asked for: {NAME[target]}")
    print(f"  my plan: {' -> '.join(NAME[i] for i in plan)}\n", flush=True)

    env = DeviceVecCraftWorld(1, grid=cfg["grid"], view=cfg["view"],
                              max_steps=10 ** 9)
    for item in plan:
        goal = _goal_of(item)
        _, is_collect, craft_a, _ = ITEM_INFO[item]
        if is_collect:
            steps = 0
            while int(env.inv[0, item].item()) < GATHER and steps < 140:
                obs = torch.cat([env.state, _goal_onehot(goal, 1)], -1)
                env.step(skills[item].act(obs, deterministic=True))
                steps += 1
                if render and steps % 8 == 0:
                    _render(env, item)
            print(f"  [+] gathered {NAME[item]} "
                  f"(have {int(env.inv[0, item].item())})", flush=True)
        else:
            before = int(env.inv[0, item].item())
            for _ in range(4):
                env.step(torch.full((1,), craft_a, dtype=torch.long, device=DEVICE))
            ok = int(env.inv[0, item].item()) > before
            print(f"  [+] crafted {NAME[item]}" if ok
                  else f"  [!] could not craft {NAME[item]} (missing inputs)",
                  flush=True)
    built = int(env.inv[0, target].item()) > 0
    msg = f"BUILT it! ({NAME[target]})" if built else f"did not finish {NAME[target]}"
    print(f"\n[ragnarok] {msg}.", flush=True)
    return built


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="iron_pickaxe",
                   help="item to build (e.g. iron_pickaxe, furnace, stone_pickaxe)")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--skill-cap", type=int, default=70)
    p.add_argument("--out-dir", default="ragnarok_agent")
    args = p.parse_args()

    name_to_item = {NAME[i]: i for i in ALL}
    if args.target not in name_to_item:
        print(f"unknown target '{args.target}'. choose from: "
              f"{sorted(name_to_item)}")
        return
    target = name_to_item[args.target]

    cfg = {k: getattr(args, k) for k in
           ("num_envs", "grid", "view", "max_steps", "rollout", "hidden",
            "entropy", "skill_cap")}
    print(f"[ragnarok] device={DEVICE}\n", flush=True)
    t0 = time.perf_counter()
    skills, rules = learn_or_load(cfg, args.out_dir, args.retrain)
    play(target, skills, rules, cfg, render=not args.no_render)
    print(f"\n  ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
