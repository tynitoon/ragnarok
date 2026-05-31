"""v29 — ROBUSTNESS (N seeds): are v27b (general skill) and v28 (efficiency) real?

Single-seed results can be flukes. v29 re-runs the two NOVEL positive claims
across N independent seeds and aggregates mean +/- std:
  (A) GENERAL-SKILL GAP (v27b): on unseen HARD (slow-paddle) variants, does the
      VARIETY agent beat the SINGLE-EASY (reactive) agent — consistently?
  (B) SAMPLE EFFICIENCY (v28): on a NEW out-of-distribution HARD target, does the
      VARIETY-pretrained agent reach competence in fewer iters/episodes than
      SCRATCH — consistently?

Decisive: across ALL seeds, gap (variety_hard - single_easy_hard) > 0 with mean
>= 0.15, AND variety reaches the efficiency threshold in fewer iters than scratch
every seed (mean speed-up reported). Honest: per-seed values are printed so a
single bad seed is visible.

Usage: python -m scripts.robustness_v29 [--seeds 0 1 2] [--smoke]
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pre-iters", type=int, default=220)
    p.add_argument("--max-iters", type=int, default=140)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.pre_iters, args.max_iters = [0, 1], 10, 20
        args.eval_every, args.n_train, args.num_envs = 5, 6, 64

    target = dict(paddle_speed=0.020, ball_speed=0.040, paddle_half=0.11,
                  opp_speed=0.018, spin=0.5)
    epi_per_iter = args.num_envs * 32 / 800.0
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v29] device={DEVICE} | ROBUSTNESS over seeds {args.seeds} | "
          f"(A) general-skill gap (v27b) + (B) efficiency vs scratch (v28) | "
          f"OOD target ratio {target['ball_speed']/target['paddle_speed']:.1f}", flush=True)
    t0 = time.perf_counter()

    rows = []
    for s in args.seeds:
        rng = random.Random(s)
        train_v = sorted(gen(args.n_train, rng), key=difficulty)
        test_v = sorted(gen(args.n_test, rng), key=difficulty)
        h = args.n_test // 2
        easy_unseen, hard_unseen = test_v[:h], test_v[h:]
        variety = train(train_v, args.pre_iters, rng, args.num_envs, args.img)
        single_easy = train([train_v[0]], args.pre_iters, rng, args.num_envs, args.img)
        vuh, seuh = mean_wr(variety, hard_unseen), mean_wr(single_easy, hard_unseen)
        vue, seue = mean_wr(variety, easy_unseen), mean_wr(single_easy, easy_unseen)
        v_hit, _ = adapt(clone_for_finetune(variety, args.img), target, args.threshold,
                         args.max_iters, args.eval_every, args.num_envs, args.img)
        sc_hit, sc_curve = adapt(new_ppo(args.img), target, args.threshold,
                                 args.max_iters, args.eval_every, args.num_envs, args.img)
        v_it = v_hit if v_hit is not None else args.max_iters
        sc_it = sc_hit if sc_hit is not None else args.max_iters
        row = dict(seed=s, variety_hard=round(vuh, 3), single_easy_hard=round(seuh, 3),
                   gap=round(vuh - seuh, 3), variety_easy=round(vue, 3),
                   single_easy_easy=round(seue, 3),
                   variety_iters=v_it, variety_reached=v_hit is not None,
                   scratch_iters=sc_it, scratch_reached=sc_hit is not None,
                   variety_episodes=round(v_it * epi_per_iter),
                   scratch_episodes=round(sc_it * epi_per_iter),
                   scratch_final=sc_curve[-1][1])
        rows.append(row)
        print(f"  seed {s}: gap {row['gap']:+.2f} (variety-hard {vuh:.2f} vs single-easy "
              f"{seuh:.2f}) | eff variety {v_it}it/{row['variety_episodes']}p "
              f"reached={row['variety_reached']} vs scratch {sc_it}it reached="
              f"{row['scratch_reached']} | {time.perf_counter()-t0:.0f}s", flush=True)

    def agg(key):
        xs = [r[key] for r in rows]
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return round(m, 3), round(sd, 3)

    gap_m, gap_sd = agg("gap")
    veps_m, _ = agg("variety_episodes")
    sceps_m, _ = agg("scratch_episodes")
    all_gap_pos = all(r["gap"] > 0 for r in rows)
    all_faster = all(r["variety_iters"] < r["scratch_iters"] for r in rows)
    speedups = [r["scratch_iters"] / r["variety_iters"] for r in rows if r["variety_iters"] > 0]
    speedup_m = round(sum(speedups) / len(speedups), 2) if speedups else None
    ok = all_gap_pos and gap_m >= 0.15 and all_faster
    verdict = (
        f"ROBUST across {len(args.seeds)} seeds — (A) the variety agent beats the "
        f"reactive single-easy agent on unseen HARD variants every seed (gap "
        f"{gap_m:+.2f} +/- {gap_sd:.2f}); (B) the variety-pretrained agent masters a "
        f"NEW OOD-hard variant in ~{veps_m:.0f} parties vs ~{sceps_m:.0f} for scratch "
        f"every seed (~{speedup_m}x fewer episodes). The general-skill and "
        f"sample-efficiency claims are not single-seed flukes."
        if ok else
        f"PARTIAL — gap {gap_m:+.2f}+/-{gap_sd:.2f} all_pos={all_gap_pos}; "
        f"efficiency all_faster={all_faster} (variety ~{veps_m:.0f}p vs scratch "
        f"~{sceps_m:.0f}p, ~{speedup_m}x). See per-seed rows.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v29_robustness.json"), "w") as f:
        json.dump(dict(seeds=args.seeds, target=target, threshold=args.threshold,
                       episodes_per_iter=epi_per_iter, rows=rows, gap_mean=gap_m,
                       gap_std=gap_sd, speedup_mean=speedup_m, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
