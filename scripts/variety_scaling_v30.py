"""v30 — the VARIETY SCALING LAW: does MORE experience-breadth monotonically buy
more generality AND fewer trials on new instances? (The user's thesis as a curve.)

v27b/v28 showed broad variety -> a general skill -> sample efficiency. v30 asks
HOW MUCH variety, holding COMPUTE FIXED. For K in {1,2,4,8,16,24}, pretrain a Pong
agent on K paddle-speed variants for the SAME total iterations (so the only thing
that changes is the BREADTH of experience, not the amount of training). Then for
each K measure: (a) GENERALISATION = win-rate on a FIXED held-out set of unseen
HARD variants; (b) EFFICIENCY = iters/episodes-to-threshold mastering a NEW
out-of-distribution HARD target (clone + fine-tune). Nested subsets (pool[:K]) so
K is a clean dose. FIXED test set across all K for comparability.

Hypothesis: generalisation RISES and episodes-to-master FALLS as K grows (more
breadth -> more general, fewer trials), likely saturating. That is 'plus elle
connait de choses, plus/mieux elle resout' drawn as a scaling curve.

Usage: python -m scripts.variety_scaling_v30 [--ks 1 2 4 8 16 24] [--smoke]
"""

import argparse
import json
import os
import random
import time

from ragnarok.infrastructure.device import DEVICE
from scripts.variety_efficiency_v27 import winrate, new_ppo
from scripts.variety_policyaxis_v27b import train, gen, difficulty, mean_wr
from scripts.fewshot_efficiency_v28 import adapt, clone_for_finetune


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    p.add_argument("--pre-iters", type=int, default=200)
    p.add_argument("--max-iters", type=int, default=140)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--pool", type=int, default=24)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.ks, args.pre_iters, args.max_iters = [1, 4], 10, 20
        args.eval_every, args.pool, args.num_envs = 5, 4, 64

    rng = random.Random(args.seed)
    pool = gen(args.pool, rng)                           # random order -> nested subsets unbiased
    test_v = sorted(gen(args.n_test, rng), key=difficulty)
    h = args.n_test // 2
    easy_unseen, hard_unseen = test_v[:h], test_v[h:]    # FIXED across all K
    target = dict(paddle_speed=0.020, ball_speed=0.040, paddle_half=0.11,
                  opp_speed=0.018, spin=0.5)
    epi_per_iter = args.num_envs * 32 / 800.0
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v30] device={DEVICE} | VARIETY SCALING LAW | K={args.ks} variants at "
          f"EQUAL compute ({args.pre_iters} iters each) | generalisation + "
          f"episodes-to-master(OOD ratio {target['ball_speed']/target['paddle_speed']:.1f})",
          flush=True)
    t0 = time.perf_counter()

    rows = []
    for K in args.ks:
        agent = train(pool[:K], args.pre_iters, rng, args.num_envs, args.img)
        gen_hard = mean_wr(agent, hard_unseen)
        gen_easy = mean_wr(agent, easy_unseen)
        hit, _ = adapt(clone_for_finetune(agent, args.img), target, args.threshold,
                       args.max_iters, args.eval_every, args.num_envs, args.img)
        it = hit if hit is not None else args.max_iters
        row = dict(K=K, gen_hard=round(gen_hard, 3), gen_easy=round(gen_easy, 3),
                   master_iters=it, master_reached=hit is not None,
                   master_episodes=round(it * epi_per_iter))
        rows.append(row)
        print(f"  K={K:>2}: unseen-hard {gen_hard:.2f} | unseen-easy {gen_easy:.2f} | "
              f"master OOD in {it}it (~{row['master_episodes']}p) reached="
              f"{row['master_reached']} | {time.perf_counter()-t0:.0f}s", flush=True)

    lo, hi = rows[0], rows[-1]
    gen_gain = round(hi["gen_hard"] - lo["gen_hard"], 3)
    epi_drop = lo["master_episodes"] - hi["master_episodes"]
    gen_monotone = all(rows[i]["gen_hard"] <= rows[i + 1]["gen_hard"] + 0.08
                       for i in range(len(rows) - 1))      # non-decreasing (tolerance)
    ok = gen_gain >= 0.15 and epi_drop > 0
    verdict = (
        f"VARIETY SCALING LAW — at EQUAL compute, growing experience-breadth from "
        f"K={lo['K']} to K={hi['K']} variants raised unseen-hard generalisation "
        f"{lo['gen_hard']:.2f}->{hi['gen_hard']:.2f} (+{gen_gain}) and cut episodes "
        f"to master a NEW OOD-hard variant {lo['master_episodes']}->{hi['master_episodes']} "
        f"(-{epi_drop}). More breadth -> more general AND fewer trials on new "
        f"instances, even with the SAME training budget. {'Generalisation is ~monotone in K. ' if gen_monotone else ''}"
        f"This is 'plus elle connait, mieux/plus vite elle resout' as a scaling curve."
        if ok else
        f"PARTIAL — gen {lo['gen_hard']:.2f}->{hi['gen_hard']:.2f} (+{gen_gain}), "
        f"episodes {lo['master_episodes']}->{hi['master_episodes']} (-{epi_drop}), "
        f"monotone={gen_monotone}. See per-K rows.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v30_scaling.json"), "w") as f:
        json.dump(dict(ks=args.ks, pre_iters=args.pre_iters, target=target,
                       threshold=args.threshold, episodes_per_iter=epi_per_iter,
                       rows=rows, gen_gain=gen_gain, episode_drop=epi_drop,
                       verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
