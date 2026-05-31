"""v27b — does broad variety help when the variation is POLICY-RELEVANT?

v27 was a negative: varying ball-speed / paddle-SIZE / opponent / spin doesn't
change Pong's optimal policy ('track the ball'), so a single-variant agent
already transfers and variety only added noise. THE RECIPE (v19) needs the
variation to span genuinely DIFFERENT required solutions. Here we pick such an
axis: paddle_SPEED (reaction budget). A FAST paddle lets you react late (reactive
tracking); a SLOW paddle forces you to ANTICIPATE the ball's bounce and head to
the landing point early. The ball reflects off the top/bottom walls, so
anticipation is non-trivial -> reactive and anticipatory are genuinely different
policies.

Three agents, equal budget: VARIETY (24 variants spanning slow..fast paddle),
SINGLE-EASY (trained only on the fastest-paddle / most reactive variant),
SINGLE-HARD (trained only on the slowest-paddle / anticipatory variant). Test all
on UNSEEN variants split into a HARD half (slow paddle) and an EASY half.

Hypotheses: (a) hard variants ARE solvable (variety wins them on train); (b)
SINGLE-EASY fails the hard unseen variants (reactive policy can't anticipate),
exposing that one easy instance is a NARROW skill; (c) VARIETY covers BOTH halves
-> broad variety finds the general skill the family needs. If single-HARD also
covers both, the lesson is 'the general/anticipatory policy is what transfers,
and variety reliably finds it' (still the recipe's point).

Usage: python -m scripts.variety_policyaxis_v27b [--iters 300] [--smoke]
"""

import argparse
import json
import os
import random
import time

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from scripts.variety_efficiency_v27 import winrate, new_ppo


def gen(n, rng):
    # vary ONLY the policy-relevant axis: paddle_speed (reaction budget) + ball_speed.
    # hold size / opponent / spin fixed so difficulty is driven by reaction budget.
    return [dict(paddle_speed=rng.uniform(0.022, 0.048),
                 ball_speed=rng.uniform(0.026, 0.040),
                 paddle_half=0.11, opp_speed=0.018, spin=0.5) for _ in range(n)]


def difficulty(v):                       # fast ball vs slow paddle = must anticipate = hard
    return v["ball_speed"] / v["paddle_speed"]


def mean_wr(ppo, variants):
    return sum(winrate(ppo, v) for v in variants) / len(variants)


def train(variants, iters, rng, num_envs, img):
    envs = [DeviceVecPong(num_envs, img=img, max_steps=800, **v) for v in variants]
    ppo = new_ppo(img)
    for _ in range(iters):
        ppo.train_iter(envs[rng.randrange(len(envs))], 32)
    return ppo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.n_train, args.num_envs = 12, 6, 64

    rng = random.Random(args.seed)
    train_v = sorted(gen(args.n_train, rng), key=difficulty)     # easy..hard
    test_v = sorted(gen(args.n_test, rng), key=difficulty)
    h = args.n_test // 2
    easy_unseen, hard_unseen = test_v[:h], test_v[h:]            # hard = slow paddle
    easiest, hardest = train_v[0], train_v[-1]
    train_hard = train_v[len(train_v) // 2:]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v27b] device={DEVICE} | POLICY-RELEVANT variety (paddle_speed) | "
          f"variety vs single-easy vs single-hard | unseen split hard/easy", flush=True)
    print(f"  difficulty(ball/paddle) train {difficulty(train_v[0]):.1f}..{difficulty(train_v[-1]):.1f}"
          f" | unseen-hard {[round(difficulty(v),1) for v in hard_unseen]}", flush=True)
    t0 = time.perf_counter()

    variety = train(train_v, args.iters, rng, args.num_envs, args.img)
    single_easy = train([easiest], args.iters, rng, args.num_envs, args.img)
    single_hard = train([hardest], args.iters, rng, args.num_envs, args.img)
    print(f"  trained 3 agents | {time.perf_counter()-t0:.0f}s", flush=True)

    res = dict(
        variety_train_hard=mean_wr(variety, train_hard),         # solvability check
        variety_unseen_hard=mean_wr(variety, hard_unseen),
        variety_unseen_easy=mean_wr(variety, easy_unseen),
        single_easy_unseen_hard=mean_wr(single_easy, hard_unseen),
        single_easy_unseen_easy=mean_wr(single_easy, easy_unseen),
        single_hard_unseen_hard=mean_wr(single_hard, hard_unseen),
        single_hard_unseen_easy=mean_wr(single_hard, easy_unseen),
    )
    for k, v in res.items():
        print(f"    {k:28s} {v:.2f}", flush=True)
    print(f"  {time.perf_counter()-t0:.0f}s", flush=True)

    solvable = res["variety_train_hard"] >= 0.6
    variety_beats_easy_on_hard = res["variety_unseen_hard"] >= res["single_easy_unseen_hard"] + 0.15
    no_harm_easy = res["variety_unseen_easy"] >= res["single_easy_unseen_easy"] - 0.05
    ok = solvable and variety_beats_easy_on_hard and no_harm_easy
    verdict = (
        f"POLICY-RELEVANT VARIETY HELPS — hard variants are solvable "
        f"(variety wins {res['variety_train_hard']:.0%} on hard train). On UNSEEN "
        f"HARD (slow-paddle, needs anticipation) variety wins {res['variety_unseen_hard']:.0%} "
        f"vs the single-EASY (reactive) agent's {res['single_easy_unseen_hard']:.0%} — a "
        f"reactive skill learned on one easy instance FAILS where the policy differs, "
        f"while variety covers it (and stays ~equal on easy: {res['variety_unseen_easy']:.0%} "
        f"vs {res['single_easy_unseen_easy']:.0%}). single-HARD on hard "
        f"{res['single_hard_unseen_hard']:.0%}. CONFIRMS the sharpened recipe: broad "
        f"variety yields the general skill when the variation spans different required "
        f"solutions (where v27's invariant-policy family showed no benefit)."
        if ok else
        f"PARTIAL/NEG — solvable={solvable} (variety hard-train {res['variety_train_hard']:.2f}); "
        f"variety unseen-hard {res['variety_unseen_hard']:.2f} vs single-easy "
        f"{res['single_easy_unseen_hard']:.2f} vs single-hard {res['single_hard_unseen_hard']:.2f}."
    )
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v27b_policyaxis.json"), "w") as f:
        json.dump(dict(results=res, verdict=verdict,
                       difficulty_train=[round(difficulty(v), 2) for v in train_v],
                       difficulty_unseen=[round(difficulty(v), 2) for v in test_v]), f, indent=2)


if __name__ == "__main__":
    main()
