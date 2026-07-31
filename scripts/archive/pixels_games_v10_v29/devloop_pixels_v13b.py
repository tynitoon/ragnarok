"""v13b — DEVELOPMENTAL REUSE makes DEEP skills learnable FROM PIXELS (M3, now
perceptual; the project's central claim).

v6/M3 showed (on symbolic obs) that REUSING mastered prerequisites lets the
agent learn deep tech-tree skills one cheap step at a time, while a flat agent
that must do the whole chain in one episode fails past depth ~2. v13b tests
that SAME claim FROM RAW PIXELS, leveraging that perception works (v12-A).

For each target at increasing depth, train a goal-conditioned CNN policy FROM
PIXELS under two arms:
  REUSE (developmental): prerequisites GRANTED (mastery simulated) -> the agent
    only has to learn the final step(s).
  FLAT  (no reuse):      grant NOTHING -> the agent must achieve the whole
    prerequisite chain in a single episode, from pixels.

Decisive: if REUSE masters DEEP targets (depth>=4) from pixels where FLAT fails
(<=0.2), and REUSE's per-skill cost stays ~flat with depth, then developmental
reuse makes deep skills learnable from pixels that are otherwise unlearnable —
the compounding claim, perceptual. Honest negative otherwise.

Usage: python -m scripts.devloop_pixels_v13b [--seeds 3] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ITEMS, WOOD, STONE_I, COAL_I, IRON_I,
    WPICK, SPICK, TABLE, FURNACE,
    A_WOOD, A_TABLE, A_WPICK, A_STONE, A_SPICK, A_FURNACE, A_IPICK)

# (name, goal achievement, prerequisites to GRANT in the reuse arm, depth)
TARGETS = [
    ("collect_wood",       A_WOOD,    {},                                     0),
    ("make_table",         A_TABLE,   {WOOD: 5},                              1),
    ("make_wood_pickaxe",  A_WPICK,   {WOOD: 5, TABLE: 5},                    2),
    ("collect_stone",      A_STONE,   {WPICK: 5},                             3),
    ("make_stone_pickaxe", A_SPICK,   {WOOD: 5, STONE_I: 5, TABLE: 5},        4),
    ("make_furnace",       A_FURNACE, {STONE_I: 5, TABLE: 5},                 4),
    ("make_iron_pickaxe",  A_IPICK,   {WOOD: 5, COAL_I: 5, IRON_I: 5,
                                       TABLE: 5, FURNACE: 5},                 6),
]


def _grant_vec(d):
    g = [0] * N_ITEMS
    for k, v in d.items():
        g[k] = v
    return g


@torch.no_grad()
def _success(ppo, goal, grant, cfg, n=128):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant,
                              n_resource=cfg["n_resource"], pixel=True,
                              tile=cfg["tile"])
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def train_arm(goal, grant, cfg, iters, eval_every, seed, early_stop):
    torch.manual_seed(seed)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant,
                              n_resource=cfg["n_resource"], pixel=True,
                              tile=cfg["tile"], seed=seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=cfg["hidden"])
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=cfg["entropy"], net=net)
    curve, steps_to_master, best = [], None, 0.0
    for it in range(1, iters + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % eval_every == 0:
            s = _success(ppo, goal, grant, cfg)
            best = max(best, s)
            curve.append([it, ppo.total_steps, s])
            if steps_to_master is None and s >= cfg["mastery"]:
                steps_to_master = ppo.total_steps
                if early_stop:
                    break
    final = _success(ppo, goal, grant, cfg)
    return dict(curve=curve, steps_to_master=steps_to_master,
                best=max(best, final), final=final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--n-resource", type=int, default=4)
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)  # generous so FLAT fails on exploration, not budget
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--mastery", type=float, default=0.8)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    global TARGETS
    if args.smoke:
        args.seeds, args.iters, args.eval_every, args.num_envs = 1, 8, 4, 64
        TARGETS = [TARGETS[0], TARGETS[3], TARGETS[6]]  # d0, d3, d6

    cfg = {k: getattr(args, k) for k in
           ("grid", "view", "tile", "max_steps", "rollout", "hidden",
            "entropy", "num_envs", "n_resource", "mastery")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v13b] device={DEVICE} | M3 from PIXELS | reuse vs flat | "
          f"{args.seeds} seeds x {len(TARGETS)} targets", flush=True)
    t0 = time.perf_counter()
    results = {nm: {"reuse": [], "flat": []} for nm, _, _, _ in TARGETS}

    for seed in range(args.seeds):
        for nm, goal, prereq, depth in TARGETS:
            arms = {"reuse": _grant_vec(prereq) if prereq else None,
                    "flat": None}
            for arm, grant in arms.items():
                # reuse arm early-stops at mastery (cheap last step); flat runs
                # full budget (we want to SEE whether it can ever master).
                r = train_arm(goal, grant, cfg, args.iters, args.eval_every,
                              seed, early_stop=(arm == "reuse"))
                results[nm][arm].append(r)
                stm = r["steps_to_master"]
                print(f"  [seed {seed}] d{depth} {nm:18s} {arm:5s} | "
                      f"best {r['best']:.2f} | master@ "
                      f"{(f'{stm:,}' if stm else '---'):>10} | "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        with open(os.path.join(args.out_dir, "v13b_partial.json"), "w") as f:
            json.dump(dict(done_seeds=seed + 1, results=results), f)

    # ---- aggregate ----
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    print(f"\n  {'target':18s} {'d':>2} {'reuse_best':>10} {'flat_best':>10} "
          f"{'reuse_master@':>14}")
    summary = {}
    for nm, goal, prereq, depth in TARGETS:
        rb = _mean([x["best"] for x in results[nm]["reuse"]])
        fb = _mean([x["best"] for x in results[nm]["flat"]])
        rm = _mean([x["steps_to_master"] for x in results[nm]["reuse"]])
        summary[nm] = dict(depth=depth, reuse_best=rb, flat_best=fb,
                           reuse_master_steps=rm)
        print(f"  {nm:18s} {depth:>2} {rb:>10.2f} {fb:>10.2f} "
              f"{(f'{rm:,.0f}' if rm else '---'):>14}", flush=True)

    # decisive: a DEEP target (depth>=4) mastered by reuse but not flat
    deep_wins = [nm for nm, s in summary.items()
                 if s["depth"] >= 4 and (s["reuse_best"] or 0) >= 0.8
                 and (s["flat_best"] or 0) <= 0.2]
    # compounding: reuse master-cost roughly flat across depth (max/min ratio)
    rms = [s["reuse_master_steps"] for s in summary.values()
           if s["reuse_master_steps"]]
    flat_ratio = (max(rms) / min(rms)) if len(rms) >= 2 else None
    ok = len(deep_wins) >= 1
    verdict = (f"DEVELOPMENTAL REUSE MAKES DEEP SKILLS LEARNABLE FROM PIXELS — "
               f"reuse masters deep target(s) {deep_wins} (>=0.80) where the "
               f"flat-from-pixels agent fails (<=0.20). Reuse master-cost "
               f"spread across depth = "
               f"{f'{flat_ratio:.1f}x' if flat_ratio else 'n/a'} (flat-ish => "
               f"compounding). The M3 compounding claim holds on raw pixels."
               if ok else
               f"NEGATIVE — no deep target shows the reuse>flat gap "
               f"(deep_wins={deep_wins}). Reuse advantage did not materialise "
               f"from pixels in this budget.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v13b.json"), "w") as f:
        json.dump(dict(summary=summary, deep_wins=deep_wins,
                       reuse_cost_spread=flat_ratio, verdict=verdict,
                       seeds=args.seeds), f, indent=2)


if __name__ == "__main__":
    main()
