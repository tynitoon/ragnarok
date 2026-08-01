"""ARC 2 — the last cheap gate: does a FROZEN policy transfer to a world it has never seen?

probe_v59 established that random frozen weights score 0/12 while the worlds stay solvable (fresh
in-world training 10/12, hand-coded-over-store 9/12). So the M-vs-R comparison has dynamic range.

But R is frozen, and M will be frozen too. If FROZENNESS is what condemns an arm — rather than
randomness — then M scores 0 like R and the ~15-20 GPU-hour confirmatory is a guaranteed null. Nothing
measured so far speaks to this: gate K1 tested a policy TRAINED IN THE WORLD it was evaluated on, never
a frozen one dropped into a new world.

This runs the actual question in miniature, in about an hour instead of twenty:
    train a policy on world A  ->  FREEZE it  ->  evaluate on world B (never seen), store fresh
and compares against the two references already measured on B: random-frozen (0/12) and in-world-trained.

  T >> R   the weights carry something portable; the confirmatory is worth running.
  T ~= R   frozenness itself is the wall, not randomness. The confirmatory would be a guaranteed null
           and must NOT be run; the honest finding is that this architecture's portable knowledge lives
           in the store, and the weights do not survive a world change.

Usage: python -m scripts.probe_transfer_v59
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
from scripts.hidden_recipe_v55 import admitted_goals, nav_gate
from scripts.evidence_net_v58 import (ComposerV58, BufferV58, permute_spec_v58, cfg_v58,
                                      make_world_env, run_goal_v58)

HARD = dict(n_items=20, p_resource=0.15)


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-world", type=int, default=6101)
    p.add_argument("--test-world", type=int, default=6100)
    p.add_argument("--n-train-goals", type=int, default=8)
    p.add_argument("--n-test-goals", type=int, default=6)
    p.add_argument("--r-max", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = cfg_v58(num_envs=64, r_max=a.r_max)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()
    skill = load_skill(cfg, a.seed, a.out_dir)

    print("=" * 96)
    print(f"ARC 2 — FROZEN TRANSFER PROBE: train on {a.train_world}, freeze, test on {a.test_world}")
    print("=" * 96, flush=True)

    # ---- train on world A --------------------------------------------------------------------
    sA = permute_spec_v58(gen_tree(a.train_world, **HARD), a.train_world)
    gA = [g for g, _, _ in admitted_goals(sA)][:a.n_train_goals]
    envA = make_world_env(sA, skill, cfg, seed=a.train_world + 7, goal=gA[0])
    comp, buf = ComposerV58(), BufferV58(cap=800_000)
    for i, g in enumerate(gA):
        r = run_goal_v58(envA, sA, skill, comp, buf, cfg, a.train_world + 11 * i, g, train=True)
        print(f"  [train {a.train_world}] goal {g:>2}: master {r['master']:.2f} in {r['rounds']}r "
              f"({'M' if r['mastered'] else 'x'}) | {time.perf_counter()-t0:.0f}s", flush=True)
    torch.save(comp.net.state_dict(), os.path.join(a.out_dir, f"v59_T_{a.train_world}.pt"))

    # ---- FREEZE and evaluate on world B ------------------------------------------------------
    sB = permute_spec_v58(gen_tree(a.test_world, **HARD), a.test_world)
    nav = nav_gate(skill, sB, cfg, a.seed)
    admB = admitted_goals(sB)
    gB = [g for g, _, _ in admB][:a.n_test_goals]
    pcs = {g: pc for g, pc, _ in admB}
    print(f"\n  test world {a.test_world} | nav min {min(nav.values()):.3f} | goals {gB} "
          f"(pc {[pcs[g] for g in gB]}) | weights FROZEN, store fresh", flush=True)
    envB = make_world_env(sB, skill, cfg, seed=a.test_world + 7, goal=gB[0])
    bufB = BufferV58(cap=200_000)
    rows = []
    for i, g in enumerate(gB):
        r = run_goal_v58(envB, sB, skill, comp, bufB, cfg, a.test_world + 11 * i, g, train=False)
        rows.append(r)
        print(f"  [T frozen on {a.test_world}] goal {g:>2}: master {r['master']:.2f} "
              f"({'M' if r['mastered'] else 'x'}) | disc {(r['discovery'] or {}).get('frac')} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    n = len(rows); m = sum(r["mastered"] for r in rows)
    ref = json.load(open(os.path.join(a.out_dir, "v59_probe.json")))["worlds"].get(str(a.test_world), {})
    json.dump(dict(train_world=a.train_world, test_world=a.test_world, goals=gB,
                   T=[r["master"] for r in rows], T_mastered=m, n=n,
                   R_mastered=ref.get("R_mastered"), F_mastered=ref.get("F_mastered"),
                   G_mastered=ref.get("G_mastered")),
              open(os.path.join(a.out_dir, "v59_transfer.json"), "w"), indent=2)

    print(f"\n{'='*96}")
    print(f"  T (trained on {a.train_world}, FROZEN on {a.test_world}) : {m}/{n}")
    print(f"  R (random frozen, same world)                 : {ref.get('R_mastered','?')}/{ref.get('n','?')}")
    print(f"  F (trained in-world, same world)              : {ref.get('F_mastered','?')}/{ref.get('n','?')}")
    if m >= 2 and (ref.get("R_mastered") or 0) == 0:
        print("\n  => FROZEN WEIGHTS TRANSFER. A policy that never saw this world beats random-frozen")
        print("     without a single gradient step here. Design v3 is worth its confirmatory run.")
    elif m == 0:
        print("\n  => FROZENNESS IS THE WALL, not randomness. A frozen policy scores like a random one")
        print("     in an unseen world. The confirmatory would be a guaranteed null and must NOT run:")
        print("     this architecture's portable knowledge lives in the STORE, not in the weights.")
        print("     That is the finding — publish it.")
    else:
        print("\n  => Weak or ambiguous. Report the numbers; do not spend 15-20 GPU-hours on this.")
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
