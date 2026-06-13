"""v54 POST-HOC diagnostic — goal-ablation (does NOT touch the frozen v54 run).

The interpretation lock (preregistration.md, post-audit) hinges on ONE fact: is the composer
goal-agnostic? If yes, any positive Delta_k is a goal-blind unlock-everything warm-start, NOT
experience-transfer. This loads each seed's FINAL Arm-A composer from the v54 checkpoint and
compares deterministic master-rate WITH the goal feature vs goal-ABLATED, on held-out + stream
specs. Ablated >= conditioned (as v53: 0.605 vs 0.121) confirms the goal-blind reflex.

Read-only on craft_v6_out/v54_ckpt_d7_s{seed}.pt. Run AFTER the seeds finish (GPU contention otherwise).

Usage: python -m scripts.goal_ablation_v54 [--seeds 0 1 2]
"""

import argparse
import json
import os

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.tech_tree import gen_tree
from scripts.depth_scaling_v49 import N_ITEMS_FOR_DEPTH
from scripts.childhood_v50 import train_childhood
from scripts.flywheel_v53 import Composer, eval_master
from scripts.clean_compound_v54 import build_pool


def _cfg():
    """The exact v54 cfg (defaults) — eval_master must behave identically to the run."""
    return dict(num_envs=256, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02,
                nav_max_steps=40, skill_iters=400, option_timeout=40, macro_budget=40,
                episodes_per_round=4, train_steps_per_round=300, max_samples_per_ep=8192,
                epsilon=0.05, temp=1.0, thresh=0.6, task_budget=3e6, skill_stochastic=True,
                mgr_entropy=0.03, router_iters=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--out-dir", default="craft_v6_out")
    args = p.parse_args()
    cfg = _cfg()
    ni = N_ITEMS_FOR_DEPTH[args.depth]
    skill_specs = [gen_tree(1000 + i, n_items=ni) for i in range(8)]
    stream, held, _ = build_pool(ni, 10, 16, 16, 4, 5000)
    out = dict(depth=args.depth, seeds={})

    for seed in args.seeds:
        ppath = os.path.join(args.out_dir, f"v54_ckpt_d{args.depth}_s{seed}.pt")
        if not os.path.exists(ppath):
            print(f"[seed {seed}] no checkpoint {ppath} — skip", flush=True)
            continue
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        skill, _ = train_childhood(skill_specs, cfg, seed)        # deterministic; same skill as the run
        composer = Composer()
        composer.net.load_state_dict(torch.load(ppath, map_location=DEVICE)["net"])

        hc = [round(eval_master(s, skill, composer, cfg, seed + 777, ablate_goal=False), 3) for _, s, _ in held]
        ha = [round(eval_master(s, skill, composer, cfg, seed + 777, ablate_goal=True), 3) for _, s, _ in held]
        sc = [round(eval_master(s, skill, composer, cfg, seed + 555, ablate_goal=False), 3) for _, s, _ in stream]
        sa = [round(eval_master(s, skill, composer, cfg, seed + 555, ablate_goal=True), 3) for _, s, _ in stream]
        mhc, mha = sum(hc) / len(hc), sum(ha) / len(ha)
        msc, msa = sum(sc) / len(sc), sum(sa) / len(sa)
        goal_agnostic = (mha >= mhc - 0.05) and (msa >= msc - 0.05)
        out["seeds"][seed] = dict(held_cond=hc, held_abl=ha, stream_cond=sc, stream_abl=sa,
                                  mean_held_cond=mhc, mean_held_abl=mha, mean_stream_cond=msc,
                                  mean_stream_abl=msa, goal_agnostic=goal_agnostic)
        print(f"[seed {seed}] held cond {mhc:.3f} vs ABL {mha:.3f} | stream cond {msc:.3f} vs ABL "
              f"{msa:.3f} | goal_agnostic={goal_agnostic} "
              f"({'goal head HURTS/ignored -> warm-start interpretation locked' if goal_agnostic else 'goal head HELPS -> stronger claim possible'})",
              flush=True)
        json.dump(out, open(os.path.join(args.out_dir, "v54_goal_ablation.json"), "w"), indent=2)

    n = len(out["seeds"])
    if n:
        agn = sum(v["goal_agnostic"] for v in out["seeds"].values())
        print(f"\n=> goal-agnostic on {agn}/{n} seeds. "
              f"{'CONFIRMS interpretation lock #1: any positive Delta = goal-blind warm-start, NOT experience-transfer.' if agn >= n - 0 else 'goal head helps on some seeds — revisit interpretation #1.'}",
              flush=True)


if __name__ == "__main__":
    main()
