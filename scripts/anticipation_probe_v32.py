"""v32 — does the variety agent ANTICIPATE, or just cover more difficulty?
(Decisive behavioral test of the v27b mechanism, per the phase-gate review.)

The review's strongest objection: v27b's "policy-relevant axis (paddle speed) ->
requires ANTICIPATION" is asserted geometrically, never measured; the gap might be
pure domain-randomisation COVERAGE. This probe settles it with a BEHAVIORAL metric.

The env is fully known, so we compute the ball's TRUE future interception y at the
agent plane, INCLUDING wall reflections (period-2 triangular fold). Then, while the
ball approaches, we measure how EARLY the paddle is already parked within a paddle-
half of that true landing — the LEAD TIME. A reactive controller only gets there
near contact; an anticipatory one is ready early, and the difference is largest on
trajectories that will BOUNCE (a reactive agent cannot see past the reflection).

Agents (all seeded, equal compute): VARIETY (24 paddle-speed variants), SINGLE-EASY
(fastest paddle / most reactive), SINGLE-HARD (slowest / most anticipatory), and a
FAIR SINGLE-MEDIAN baseline (the review's missing control). Metric: early-readiness
(paddle within paddle_half of true landing at 8-16 steps before contact), reported
overall AND on bounce trajectories.

If VARIETY's early-readiness (esp. on bounce trajectories) exceeds the single-easy/
median agents -> anticipation is real. If all are similar -> it's coverage, and we
say so. torch + env seeded (fixing the review's #1 issue).

Usage: python -m scripts.anticipation_probe_v32 [--iters 240] [--smoke]
"""

import argparse
import json
import os
import random
import time

import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from scripts.variety_efficiency_v27 import new_ppo
from scripts.variety_policyaxis_v27b import train, gen, difficulty


@torch.no_grad()
def predict_landing(env):
    """True interception y at the agent plane incl. wall reflections; time-to-contact;
    approaching mask; will-bounce mask. Walls at y in [0,1] -> period-2 fold."""
    approaching = env.vx < 0
    t = (env.bx - env.x_a) / (-env.vx).clamp(min=1e-6)
    yraw = env.by + env.vy * t
    m = torch.remainder(yraw, 2.0)
    landing = torch.where(m <= 1.0, m, 2.0 - m)
    will_bounce = (yraw < 0.0) | (yraw > 1.0)          # at least one reflection before contact
    return landing, t, approaching, will_bounce


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def leadtime(ppo, variant, n=256, steps=799, early=(8, 16), late=(1, 4)):
    env = DeviceVecPong(n, max_steps=800, seed=0, **variant)
    obs = env.state
    eh = et = ehb = etb = lh = lt = 0.0
    for _ in range(steps):
        landing, t, appr, bounce = predict_landing(env)
        ready = (env.pad_a - landing).abs() <= env.PH
        e = appr & (t >= early[0]) & (t < early[1])
        l = appr & (t >= late[0]) & (t < late[1])
        eb = e & bounce
        eh += float((ready & e).sum()); et += float(e.sum())
        ehb += float((ready & eb).sum()); etb += float(eb.sum())
        lh += float((ready & l).sum()); lt += float(l.sum())
        obs, _, _, _, _ = env.step(ppo.act(obs, deterministic=True))
    return (eh / max(et, 1.0), ehb / max(etb, 1.0), lh / max(lt, 1.0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=240)
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

    seed_all(args.seed)
    rng = random.Random(args.seed)
    train_v = sorted(gen(args.n_train, rng), key=difficulty)
    test_v = sorted(gen(args.n_test, rng), key=difficulty)
    easiest, hardest = train_v[0], train_v[-1]
    median = train_v[len(train_v) // 2]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v32] device={DEVICE} | ANTICIPATION probe (seeded) | variety vs "
          f"single-easy/median/hard | early-readiness vs true bounce-aware landing",
          flush=True)
    t0 = time.perf_counter()

    agents = {}
    for name, variants in [("variety", train_v), ("single_easy", [easiest]),
                           ("single_median", [median]), ("single_hard", [hardest])]:
        seed_all(args.seed + len(name))                # reproducible, distinct per agent
        agents[name] = train(variants, args.iters, rng, args.num_envs, args.img)
        print(f"  trained {name} | {time.perf_counter()-t0:.0f}s", flush=True)

    # evaluate lead-time on the UNSEEN test variants (mean over variants)
    out = {}
    for name, ppo in agents.items():
        es, ebs, ls = [], [], []
        for v in test_v:
            e, eb, l = leadtime(ppo, v)
            es.append(e); ebs.append(eb); ls.append(l)
        out[name] = dict(early_ready=round(sum(es) / len(es), 3),
                         early_ready_bounce=round(sum(ebs) / len(ebs), 3),
                         late_ready=round(sum(ls) / len(ls), 3))
        print(f"  {name:14s} early {out[name]['early_ready']:.2f} | early-bounce "
              f"{out[name]['early_ready_bounce']:.2f} | late {out[name]['late_ready']:.2f}"
              f" | {time.perf_counter()-t0:.0f}s", flush=True)

    v = out["variety"]; se = out["single_easy"]; sm = out["single_median"]
    # anticipation = variety parks at the TRUE landing earlier than reactive baselines,
    # especially on bounce trajectories
    adv_all = round(v["early_ready"] - max(se["early_ready"], sm["early_ready"]), 3)
    adv_bounce = round(v["early_ready_bounce"] - max(se["early_ready_bounce"],
                                                     sm["early_ready_bounce"]), 3)
    anticipates = adv_bounce >= 0.10 and adv_all >= 0.05
    verdict = (
        f"ANTICIPATION IS REAL — the variety agent is parked at the ball's TRUE "
        f"(bounce-aware) landing EARLY far more often than the reactive single-easy/"
        f"median agents: early-readiness {v['early_ready']:.2f} vs "
        f"{max(se['early_ready'], sm['early_ready']):.2f} (+{adv_all}), and on BOUNCE "
        f"trajectories {v['early_ready_bounce']:.2f} vs "
        f"{max(se['early_ready_bounce'], sm['early_ready_bounce']):.2f} (+{adv_bounce}) "
        f"— exactly where a reactive policy is blind. So v27b's variety benefit is a "
        f"genuinely different (anticipatory) policy, not only difficulty coverage."
        if anticipates else
        f"NOT ANTICIPATION (review was right) — variety's early-readiness "
        f"{v['early_ready']:.2f} (bounce {v['early_ready_bounce']:.2f}) is NOT clearly "
        f"above the reactive single-easy/median baselines (all/bounce advantage "
        f"+{adv_all}/+{adv_bounce}). The v27b gap is better explained by "
        f"domain-randomisation COVERAGE than by a distinct anticipatory policy. "
        f"Recorded honestly; the mechanistic 'anticipation' framing is retracted.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v32_anticipation.json"), "w") as f:
        json.dump(dict(results=out, adv_all=adv_all, adv_bounce=adv_bounce,
                       anticipates=anticipates, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
