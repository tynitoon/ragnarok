"""v13 — DEVELOPMENTAL REUSE FROM PIXELS (the project's heart, perceptual).

v6/M3 showed (on symbolic obs) that reusing learned skills makes new skills
learn faster — the compounding claim. v13 tests whether that reuse advantage
SURVIVES on raw pixels: the agent first learns `collect_wood` from pixels
(a CNN perception "notion"), then learns NEW collect-skills (stone/coal/iron —
same task structure, different target colour) under three arms:

  SCRATCH        fresh CNN, learn everything from pixels cold.
  REUSE-FINETUNE warm-start the conv+fc encoder from the wood-skill, train all.
  REUSE-FROZEN   warm-start AND FREEZE the encoder; train only a small head.

Decisive: if REUSE reaches the skill threshold in FEWER env-steps than SCRATCH
(and FROZEN works at all), then the learned perceptual notion is genuinely
REUSED to learn new notions faster — the developmental claim holds from pixels,
not just from symbols. Honest negative if reuse is no faster.

Usage: python -m scripts.devreuse_v13 [--seeds 3] [--smoke]
"""

import argparse
import json
import os
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from ragnarok.environments.craft_world import (
    DeviceVecCraftWorld, N_ITEMS, A_WOOD, A_STONE, A_COAL, A_IRON,
    WPICK, SPICK)

THRESH = 0.5


def _grant(idx, count=1):
    g = [0] * N_ITEMS
    g[idx] = count
    return g


# downstream skills: (name, goal achievement, granted prerequisites)
SKILLS = [
    ("collect_stone", A_STONE, _grant(WPICK)),   # stone needs a wood pickaxe
    ("collect_coal", A_COAL, _grant(WPICK)),     # coal  needs a wood pickaxe
    ("collect_iron", A_IRON, _grant(SPICK)),     # iron  needs a stone pickaxe
]


@torch.no_grad()
def _success(ppo, goal, grant, cfg, n=128):
    env = DeviceVecCraftWorld(n, grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal,
                              grant=grant, n_resource=cfg["n_resource"],
                              pixel=True, tile=cfg["tile"])
    ever = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["max_steps"]):
        obs, _, term, _, _ = env.step(ppo.act(obs, deterministic=True))
        ever |= term
    return float(ever.float().mean().item())


def _copy_encoder(dst_net, src_state):
    """Copy conv+fc (perception) weights from a trained net into dst_net,
    leaving actor/critic heads freshly initialised."""
    sd = dst_net.state_dict()
    for k in sd:
        if k.startswith("conv.") or k.startswith("fc."):
            sd[k] = src_state[k].clone()
    dst_net.load_state_dict(sd)


def train_skill(goal, grant, cfg, iters, eval_every, seed,
                init_state=None, freeze_encoder=False):
    """Train one goal-conditioned skill from pixels; return success curve,
    env-steps-to-threshold, final success, and the net's state_dict."""
    torch.manual_seed(seed)
    env = DeviceVecCraftWorld(cfg["num_envs"], grid=cfg["grid"], view=cfg["view"],
                              max_steps=cfg["max_steps"], goal=goal, grant=grant,
                              n_resource=cfg["n_resource"], pixel=True,
                              tile=cfg["tile"], seed=seed)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=cfg["hidden"])
    if init_state is not None:
        _copy_encoder(net, init_state)
    if freeze_encoder:
        for nm, p in net.named_parameters():
            if nm.startswith("conv.") or nm.startswith("fc."):
                p.requires_grad_(False)
    ppo = DiscretePPO(env.obs_dim, env.action_dim, entropy=cfg["entropy"], net=net)
    if freeze_encoder:                      # optimiser over trainable params only
        ppo.opt = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad], lr=3e-4, eps=1e-5)

    curve, steps_to_thresh, best = [], None, 0.0
    for it in range(1, iters + 1):
        ppo.train_iter(env, cfg["rollout"])
        if it % eval_every == 0:
            s = _success(ppo, goal, grant, cfg)
            best = max(best, s)
            curve.append([it, ppo.total_steps, s])
            if steps_to_thresh is None and s >= THRESH:
                steps_to_thresh = ppo.total_steps
    final = _success(ppo, goal, grant, cfg)
    best = max(best, final)
    auc = sum(c[2] for c in curve) / max(1, len(curve))     # mean success = sample-eff
    return dict(curve=curve, steps_to_thresh=steps_to_thresh, final=final,
                best=best, auc=auc, state=net.state_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--base-iters", type=int, default=100)
    p.add_argument("--skill-iters", type=int, default=140)
    p.add_argument("--eval-every", type=int, default=3)  # fine: skills learn fast
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--grid", type=int, default=13)       # harder nav: perception
    p.add_argument("--n-resource", type=int, default=3)  # sparser: is the bottleneck
    p.add_argument("--view", type=int, default=7)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=130)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.base_iters, args.skill_iters = 1, 8, 8
        args.eval_every, args.num_envs = 4, 64
        global SKILLS
        SKILLS = SKILLS[:1]

    cfg = {k: getattr(args, k) for k in
           ("grid", "view", "tile", "max_steps", "rollout", "hidden",
            "entropy", "num_envs", "n_resource")}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v13] device={DEVICE} | developmental REUSE from pixels | "
          f"{args.seeds} seeds x {len(SKILLS)} skills x 3 arms", flush=True)
    t0 = time.perf_counter()
    results = {nm: {"scratch": [], "reuse_ft": [], "reuse_frozen": []}
               for nm, _, _ in SKILLS}

    for seed in range(args.seeds):
        base = train_skill(A_WOOD, None, cfg, args.base_iters,
                           args.eval_every, seed)
        print(f"  [seed {seed}] base collect_wood-from-pixels: "
              f"best {base['best']:.2f} | {time.perf_counter()-t0:.0f}s", flush=True)
        for nm, goal, grant in SKILLS:
            arms = {
                "scratch": dict(init_state=None, freeze_encoder=False),
                "reuse_ft": dict(init_state=base["state"], freeze_encoder=False),
                "reuse_frozen": dict(init_state=base["state"], freeze_encoder=True),
            }
            for arm, kw in arms.items():
                r = train_skill(goal, grant, cfg, args.skill_iters,
                                args.eval_every, seed, **kw)
                results[nm][arm].append(
                    dict(steps_to_thresh=r["steps_to_thresh"], final=r["final"],
                         best=r["best"], auc=r["auc"], curve=r["curve"]))
                st = r["steps_to_thresh"]
                print(f"    [seed {seed}] {nm:14s} {arm:13s} | "
                      f"best {r['best']:.2f} | auc {r['auc']:.2f} | "
                      f"steps->0.5 {st if st else '---':>10} | "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        # checkpoint after each seed so a late crash keeps completed seeds
        with open(os.path.join(args.out_dir, "v13_partial.json"), "w") as f:
            json.dump(dict(done_seeds=seed + 1, results=results), f)

    # ---- aggregate ----
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    print(f"\n  {'skill':14s} {'arm':13s} {'mean_best':>9} {'mean_auc':>9} "
          f"{'mean_steps->0.5':>16} {'reached':>8}")
    summary = {}
    for nm, _, _ in SKILLS:
        summary[nm] = {}
        for arm in ("scratch", "reuse_ft", "reuse_frozen"):
            rs = results[nm][arm]
            reached = [x["steps_to_thresh"] for x in rs if x["steps_to_thresh"]]
            mb, ma = _mean([x["best"] for x in rs]), _mean([x["auc"] for x in rs])
            ms = _mean(reached)
            summary[nm][arm] = dict(mean_best=mb, mean_auc=ma, mean_steps=ms,
                                    reached=len(reached), n=len(rs))
            print(f"  {nm:14s} {arm:13s} {mb:>9.2f} {ma:>9.2f} "
                  f"{(f'{ms:,.0f}' if ms else '---'):>16} {len(reached)}/{len(rs)}",
                  flush=True)

    # verdict: reuse faster (fewer steps to threshold) and/or higher AUC than scratch
    speedups, auc_gains = [], []
    for nm in summary:
        sc = summary[nm]["scratch"]
        for arm in ("reuse_ft", "reuse_frozen"):
            ru = summary[nm][arm]
            if sc["mean_steps"] and ru["mean_steps"]:
                speedups.append(sc["mean_steps"] / ru["mean_steps"])
            if sc["mean_auc"] is not None and ru["mean_auc"] is not None:
                auc_gains.append(ru["mean_auc"] - sc["mean_auc"])
    mean_speedup = _mean(speedups)
    mean_auc_gain = _mean(auc_gains)
    ok = (mean_speedup is not None and mean_speedup > 1.15) or \
         (mean_auc_gain is not None and mean_auc_gain > 0.08)
    verdict = (f"REUSE ACCELERATES NEW SKILLS FROM PIXELS — mean steps-to-"
               f"threshold speedup {mean_speedup:.2f}x, mean AUC gain "
               f"{mean_auc_gain:+.2f}. The learned perceptual notion is reused "
               f"to learn new notions faster: the developmental compounding "
               f"claim holds on raw pixels, not just symbols."
               if ok else
               f"NEGATIVE — reuse not faster from pixels (speedup "
               f"{mean_speedup}, AUC gain {mean_auc_gain}). Perceptual reuse "
               f"advantage did not materialise in this budget.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v13.json"), "w") as f:
        json.dump(dict(summary=summary, mean_speedup=mean_speedup,
                       mean_auc_gain=mean_auc_gain, verdict=verdict,
                       seeds=args.seeds), f, indent=2)


if __name__ == "__main__":
    main()
