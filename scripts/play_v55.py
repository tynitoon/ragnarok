"""Watch the v55 result: an agent that REMEMBERS a hidden-recipe world vs one that does not.

Both agents face the SAME world, the SAME goal, and the SAME hidden recipes — nobody is ever told what
crafts into what; recipes must be DISCOVERED by attempting things and paying for the failures.

  MEMORY  : the arm-A composer from the v55 run — it has lived in this world and learned its rules.
  AMNESIC : a fresh composer trained from scratch ON THIS EXACT GOAL for the full v55 budget
            (10 rounds x 4 episodes x 256 envs). This is the real control arm B, not a blank net.

In the v55 measurement the amnesic arm, after that full budget, still failed 3 of world 3016's 4
GOAL-NECESSARY goals — while the memory arm solved them at the first attempt, cost zero.

Usage:
  python -m scripts.play_v55                    # seed 2 (world 3016), default goal
  python -m scripts.play_v55 --seed 0 --goal 1  # world 3002, a goal the amnesic arm never solved
  python -m scripts.play_v55 --list             # show the world's goals and what each arm did in v55
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from ragnarok.learning.ppo_discrete import DiscretePPO
from scripts.depth_scaling_v49 import MAX_CELLS, TechTreeConvNet
from scripts.childhood_v50 import nav_env, NAV_ACTIONS
from scripts.hidden_recipe_v55 import (permute_spec, admitted_goals, HiddenEnv, Composer, Buffer,
                                       collect_episode, relabel)

WORLDS = {0: 3002, 1: 3003, 2: 3016}
SYMS = "ABCDEFGHIJ"


def load_cfg(n=1):
    return dict(num_envs=n, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02, nav_max_steps=40,
                skill_iters=400, option_timeout=16, macro_budget=26, episodes_per_round=4,
                train_steps_per_round=300, max_samples_per_ep=8192, epsilon=0.05, temp=1.0,
                thresh=0.6, r_max=10, skill_stochastic=True, mgr_entropy=0.03, router_iters=0)


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def name(spec, i):
    return f"item{i}" + ("(gather)" if spec["kind"][i] == "R" else "(craft)")


def recipe_lines(spec, g, seen=None, depth=0):
    """The hidden rules the agent had to discover for goal g (shown to YOU, never to the agent)."""
    seen = seen if seen is not None else set()
    if g in seen or depth > 6:
        return []
    seen.add(g)
    out = []
    if spec["kind"][g] == "R":
        t = spec["tool"][g]
        out.append(f"    {name(spec,g)}: gather from the ground" +
                   (f" — but only while holding {name(spec,t)}" if t >= 0 else " — freely"))
        if t >= 0:
            out += recipe_lines(spec, t, seen, depth + 1)
    else:
        ins = ", ".join(name(spec, j) for j in spec["inputs"][g])
        tl = spec["tools"][g]
        out.append(f"    {name(spec,g)}: craft from {ins}" +
                   (f" + tool {', '.join(name(spec,t) for t in tl)}" if tl else ""))
        for j in list(spec["inputs"][g]) + list(tl):
            out += recipe_lines(spec, j, seen, depth + 1)
    return out


def render(env, spec, pick, ok, step, tag):
    g = env.base.grid[0].tolist()
    pr, pc = env.base.pos[0].tolist()
    board = ["   " + "".join("@" if (r == pr and c == pc) else
                             ("." if g[r][c] == 0 else SYMS[(g[r][c] - 1) % len(SYMS)])
                             for c in range(env.base.G)) for r in range(env.base.G)]
    inv = env.base.inv[0].tolist()
    held = ", ".join(f"{name(spec,i)}x{inv[i]}" for i in range(spec["n_items"]) if inv[i] > 0)
    print(f"  [{tag}] macro-step {step:>2} -> tries {name(spec,pick)}  "
          f"{'OK' if ok else 'failed (learns: not yet possible)'}")
    print("\n".join(board))
    print(f"   holding: {held or '(nothing)'}\n")


def play(spec, skill, composer, cfg, seed, goal, tag, show=True, pause=0.35):
    env = HiddenEnv(1, spec, skill, cfg, seed=seed, goal=goal, hidden=True)
    got, seq = False, []
    for t in range(cfg["macro_budget"]):
        pick = int(composer.act(env.state, deterministic=True)[0])
        before = env.base.inv[0, pick].item()
        env.step(torch.tensor([pick], device=DEVICE))
        ok = env.base.inv[0, pick].item() > before
        seq.append((pick, ok))
        if show:
            render(env, spec, pick, ok, t, tag)
            time.sleep(pause)
        if bool(env.post_unlocked[0, goal]):
            got = True
            break
    return got, seq


def train_amnesic(spec, skill, cfg, seed, goal, out_dir):
    """The REAL control: a fresh composer given v55's full per-goal budget on this exact goal."""
    p = os.path.join(out_dir, f"v55_demo_amnesic_s{seed}_g{goal}.pt")
    comp = Composer("memory")
    if os.path.exists(p):
        comp.net.load_state_dict(torch.load(p, map_location=DEVICE))
        return comp, True
    print(f"  training the AMNESIC control from scratch on goal {goal} "
          f"({cfg['r_max']} rounds, v55's full per-goal budget) — one-off, then cached...", flush=True)
    big = dict(cfg, num_envs=256)
    env = HiddenEnv(256, spec, skill, big, seed=seed, goal=goal, hidden=True)
    buf = Buffer(cap=400_000)
    for r in range(big["r_max"]):
        for _ in range(big["episodes_per_round"]):
            s, a, us = collect_episode(env, comp, big["epsilon"], big["temp"], goal)
            ss, aa = relabel(s, a, us, big["max_samples_per_ep"], gamma=0.7)
            if ss is not None:
                buf.add(ss, aa)
        comp.train_steps(buf, big["train_steps_per_round"])
        print(f"    round {r+1}/{big['r_max']} | buffer {buf.n}", flush=True)
    torch.save(comp.net.state_dict(), p)
    return comp, False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--goal", type=int, default=None)
    p.add_argument("--pause", type=float, default=0.35)
    p.add_argument("--list", action="store_true")
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = load_cfg(1)
    w = WORLDS[a.seed]
    spec = permute_spec(gen_tree(w, n_items=14), w)
    adm = admitted_goals(spec)
    nec = [g for g, _, b in adm if b > cfg["macro_budget"]]
    res = json.load(open(os.path.join(a.out_dir, f"v55_s{a.seed}.json")))
    rowA = {r["goal"]: r for r in res["arms"]["A"]["rows"]}
    rowB = {r["goal"]: r for r in res["arms"]["B"]["rows"]}

    if a.list:
        print(f"world {w} (seed {a.seed}) — * = GOAL-NECESSARY (a blind sweep cannot afford it)\n")
        print(f"  {'goal':>5} {'pc':>3}  {'MEMORY (arm A)':<22} {'AMNESIC (arm B)':<22}")
        for g, pc, b in adm:
            A, B = rowA.get(g, {}), rowB.get(g, {})
            f = lambda r: (f"solved in {r['rounds']}r" if r.get("mastered")            # noqa: E731
                           else "NEVER (10 rounds)")
            print(f"  {g:>4}{'*' if g in nec else ' '} {pc:>3}  {f(A):<22} {f(B):<22}")
        return

    goal = a.goal if a.goal is not None else next(
        (g for g in nec if rowA.get(g, {}).get("mastered") and not rowB.get(g, {}).get("mastered")),
        nec[0] if nec else adm[0][0])
    skill = load_skill(cfg, a.seed, a.out_dir)
    cA = Composer("memory")
    cA.net.load_state_dict(torch.load(os.path.join(a.out_dir, f"v55_ckpt_s{a.seed}.pt"),
                                      map_location=DEVICE)["net"])

    print(f"\n{'='*78}\nWORLD {w} — goal {name(spec,goal)}"
          f"{'  [GOAL-NECESSARY]' if goal in nec else ''}")
    print(f"{'='*78}\nThe hidden rules (the agents are NEVER told these — they must be discovered by")
    print("attempting things and paying for the failures):")
    for line in recipe_lines(spec, goal):
        print(line)
    print(f"\nIn the v55 measurement:  MEMORY {'solved it in ' + str(rowA[goal]['rounds']) + ' round(s)' if rowA.get(goal,{}).get('mastered') else 'failed'}"
          f"  |  AMNESIC {'solved it in ' + str(rowB[goal]['rounds']) + ' round(s)' if rowB.get(goal,{}).get('mastered') else 'NEVER solved it, after the full 10-round budget'}")
    print(f"{'='*78}\n")

    cB, cached = train_amnesic(spec, skill, cfg, a.seed, goal, a.out_dir)
    print(f"  (amnesic control {'loaded from cache' if cached else 'trained'})\n")

    print(f"{'-'*78}\n>>> MEMORY agent — has lived in this world\n{'-'*78}")
    okA, seqA = play(spec, skill, cA, cfg, a.seed + 9, goal, "MEMORY", pause=a.pause)
    print(f"{'-'*78}\n>>> AMNESIC agent — trained from scratch on THIS goal for the full budget\n{'-'*78}")
    okB, seqB = play(spec, skill, cB, cfg, a.seed + 9, goal, "AMNESIC", pause=a.pause)

    fa = sum(1 for _, o in seqA if not o); fb = sum(1 for _, o in seqB if not o)
    print(f"{'='*78}")
    print(f"  MEMORY : {'REACHED the goal' if okA else 'did not reach it'} in {len(seqA)} macro-steps "
          f"({fa} wasted attempts)")
    print(f"  AMNESIC: {'REACHED the goal' if okB else 'did not reach it'} in {len(seqB)} macro-steps "
          f"({fb} wasted attempts)")
    print(f"{'='*78}")
    print("Both saw the same world, the same goal and the same hidden recipes. The difference is")
    print("only what one of them remembers about how this world works.")
    print("\nHonest footnote: a SINGLE episode is noisy — the v55 numbers above are the measured")
    print("result over 256 parallel worlds. Re-run with --seed/--goal to see other cases, and")
    print("`--list` to see every goal and how each arm actually did.")


if __name__ == "__main__":
    main()
