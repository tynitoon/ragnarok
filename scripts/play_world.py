"""Watch Ragnarok solve a RANDOM world it has never seen.

Generates a procedurally-random tech-tree world, shows you its (hidden-to-the-
agent) recipe structure, then the agent: EXPERIMENTS to learn the recipes,
PLANS the path to the hardest item, and BUILDS it — live, with an ASCII view.

Each run = a different random world (unless you pass --seed). This is the
generality result (v10) made watchable: the agent develops in worlds nobody
designed. (Navigation uses a scripted primitive so one agent runs on any
world; the learning is the rule-discovery + planning.)

Usage:
  python -m scripts.play_world                 # a fresh random world
  python -m scripts.play_world --seed 7        # a specific world
  python -m scripts.play_world --items 18      # bigger world
"""

import argparse
import time

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import DeviceVecTechTree, gen_tree
from scripts.tech_tree_sanity_v10 import _nearest_move, _needed, _bfs_order
from scripts.techtree_agent_v10 import (_attempt, _plan, _craft_action_of)


def iname(spec, i):
    k = "resource" if spec["kind"][i] == "R" else "craft"
    return f"item{i}[{k}]"


def describe(spec):
    print("  the world's recipes (the agent does NOT get told these):", flush=True)
    for i in range(spec["n_items"]):
        if spec["kind"][i] == "R":
            tool = spec["tool"][i]
            req = f" — needs tool {iname(spec, tool)}" if tool >= 0 else " — free"
            print(f"    {iname(spec,i)}: gather from the ground{req}", flush=True)
        else:
            ins = ", ".join(iname(spec, j) for j in spec["inputs"][i])
            tls = spec["tools"][i]
            tl = f" + tool {','.join(iname(spec,t) for t in tls)}" if tls else ""
            print(f"    {iname(spec,i)}: craft from {ins}{tl}", flush=True)
    print(f"  hardest item to make: {iname(spec, spec['target'])} "
          f"(depth {spec['depth'][spec['target']]})\n", flush=True)


def learn_rules(spec, cfg):
    print("[ragnarok] experimenting to figure out the recipes...", flush=True)
    learned = {}
    for I in range(spec["n_items"]):
        cand = [x for x in range(spec["n_items"]) if x != I]
        base = _attempt(spec, I, set(cand), cfg)
        pre = set()
        if base > 0.5:
            for X in cand:
                if _attempt(spec, I, set(cand) - {X}, cfg) < 0.15:
                    pre.add(X)
        learned[I] = pre
        if pre:
            print(f"    figured out: {iname(spec,I)} needs "
                  f"{{{', '.join(iname(spec,x) for x in sorted(pre))}}}", flush=True)
        else:
            print(f"    figured out: {iname(spec,I)} needs nothing", flush=True)
    return learned


SYMS = "ABCDEFGHIJ"


def render(env, spec, doing):
    g = env.grid[0].tolist()
    pr, pc = env.pos[0].tolist()
    rows = []
    for r in range(env.G):
        line = "".join("@" if (r == pr and c == pc)
                       else ("." if g[r][c] == 0 else SYMS[(g[r][c] - 1) % len(SYMS)])
                       for c in range(env.G))
        rows.append("   " + line)
    inv = env.inv[0].tolist()
    held = ", ".join(f"{iname(spec,i)}x{inv[i]}" for i in range(spec["n_items"])
                     if inv[i] > 0) or "(nothing)"
    print("\n".join(rows), flush=True)
    print(f"   holding: {held}\n   doing: get {iname(spec, doing)}\n", flush=True)


def play(spec, learned, cfg, render_on=True):
    plan = _plan(spec["target"], learned, spec["n_items"])
    if plan is None:
        print("[ragnarok] I couldn't find a plan from what I learned.", flush=True)
        return False
    print(f"\n[ragnarok] my plan to build {iname(spec, spec['target'])}:")
    print(f"    {' -> '.join(iname(spec,i) for i in plan)}\n", flush=True)

    need = _needed(spec)
    ca = _craft_action_of(spec)
    env = DeviceVecTechTree(1, spec, grid=cfg["grid"], view=cfg["view"],
                            max_steps=10 ** 9)
    for it in plan:
        is_res = spec["kind"][it] == "R"
        want = max(1, need[it])                  # make ENOUGH (covers all uses)
        last_frame = -99
        for step in range(260):
            if int(env.inv[0, it].item()) >= want:
                break
            if is_res:
                a = _nearest_move(env, torch.full((1,), spec["cell"][it], device=DEVICE))
            else:                                # craft: emit once per step, up to `want`
                a = torch.full((1,), ca[it], dtype=torch.long, device=DEVICE)
            env.step(a)
            if render_on and is_res and step - last_frame >= 10:
                render(env, spec, it); last_frame = step
        have = int(env.inv[0, it].item())
        tag = f"made {have}" if have > 0 else "FAILED"
        print(f"  [{tag}] {iname(spec,it)}", flush=True)
    built = int(env.inv[0, spec["target"]].item()) > 0
    print(f"\n[ragnarok] {'BUILT it: ' + iname(spec, spec['target']) if built else 'did not finish'}.",
          flush=True)
    return built


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=-1, help="-1 = random world")
    p.add_argument("--items", type=int, default=14)
    p.add_argument("--grid", type=int, default=11)
    p.add_argument("--view", type=int, default=5)
    p.add_argument("--attempt-steps", type=int, default=60)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    seed = args.seed if args.seed >= 0 else int(time.time()) % 100000
    cfg = {"grid": args.grid, "view": args.view, "attempt_steps": args.attempt_steps}
    print(f"[ragnarok] device={DEVICE}\n[ragnarok] NEW RANDOM WORLD (seed {seed})\n",
          flush=True)
    spec = gen_tree(seed, n_items=args.items, n_base_res=2)
    describe(spec)
    t0 = time.perf_counter()
    learned = learn_rules(spec, cfg)
    built = play(spec, learned, cfg, render_on=not args.no_render)
    print(f"\n  (world seed {seed}, {time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
