"""v28 — SAMPLE EFFICIENCY: does prior knowledge mean FEWER episodes on a new,
harder variant? (The user's literal question, measured in 'parties'.)

v27b established that BROAD VARIETY over a policy-relevant axis yields a GENERAL
Pong skill (covers slow..fast paddle). The pay-off the developmental vision
predicts: when a genuinely NEW, HARDER variant appears, an agent that already
holds the general skill should MASTER it in far fewer episodes than one starting
from scratch — and fewer than a NARROW single-instance agent that must un-learn
its bias. We measure iters/episodes-to-threshold on an OUT-OF-DISTRIBUTION hard
target (slower paddle + faster ball than anything trained), fine-tuning from
three starting points: VARIETY-pretrained, SINGLE-EASY-pretrained (reactive),
and SCRATCH.

Decisive: variety-pretrained reaches competence in MUCH fewer episodes than
scratch (a large speed-up), and <= the narrow single-instance agent. That is
'more knowledge -> fewer trials', quantified.

Usage: python -m scripts.fewshot_efficiency_v28 [--pre-iters 260] [--smoke]
"""

import argparse
import json
import os
import random
import time

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.pong import DeviceVecPong
from ragnarok.learning.ppo_discrete import DiscretePPO, ConvPPONet
from scripts.variety_efficiency_v27 import winrate, new_ppo
from scripts.variety_policyaxis_v27b import train, gen


def clone_for_finetune(src, img):
    """Fresh DiscretePPO (new optimizer) starting from src's weights."""
    env = DeviceVecPong(2, img=img)
    net = ConvPPONet(env.img_hw, env.action_dim, hidden=256)
    net.load_state_dict(src.net.state_dict())
    return DiscretePPO(env.obs_dim, env.action_dim, entropy=0.01, net=net)


def adapt(ppo, target, thr, max_iters, eval_every, num_envs, img):
    """Fine-tune on the target variant; return (iters_to_threshold|None, curve)."""
    env = DeviceVecPong(num_envs, img=img, max_steps=800, **target)
    wr0 = winrate(ppo, target)
    curve, hit = [(0, round(wr0, 3))], (0 if wr0 >= thr else None)
    for it in range(1, max_iters + 1):
        ppo.train_iter(env, 32)
        if it % eval_every == 0:
            wr = winrate(ppo, target)
            curve.append((it, round(wr, 3)))
            if hit is None and wr >= thr:
                hit = it
    return hit, curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pre-iters", type=int, default=260)
    p.add_argument("--max-iters", type=int, default=180)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--img", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.pre_iters, args.max_iters, args.eval_every = 10, 20, 5
        args.n_train, args.num_envs = 6, 64

    rng = random.Random(args.seed)
    train_v = sorted(gen(args.n_train, rng), key=lambda v: v["ball_speed"] / v["paddle_speed"])
    easiest = train_v[0]                                  # fastest paddle (reactive)
    # OUT-OF-DISTRIBUTION HARD target: slower paddle + faster ball than trained.
    target = dict(paddle_speed=0.020, ball_speed=0.040, paddle_half=0.11,
                  opp_speed=0.018, spin=0.5)
    epi_per_iter = args.num_envs * 32 / 800.0
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[v28] device={DEVICE} | SAMPLE EFFICIENCY | OOD-hard target ratio "
          f"{target['ball_speed']/target['paddle_speed']:.1f} (train max "
          f"{train_v[-1]['ball_speed']/train_v[-1]['paddle_speed']:.1f}) | "
          f"iters->wr>={args.threshold} from variety / single-easy / scratch | "
          f"{epi_per_iter:.1f} episodes/iter", flush=True)
    t0 = time.perf_counter()

    variety = train(train_v, args.pre_iters, rng, args.num_envs, args.img)
    single_easy = train([easiest], args.pre_iters, rng, args.num_envs, args.img)
    print(f"  pretrained variety + single-easy | {time.perf_counter()-t0:.0f}s", flush=True)

    starts = {"variety": clone_for_finetune(variety, args.img),
              "single_easy": clone_for_finetune(single_easy, args.img),
              "scratch": new_ppo(args.img)}
    out = {}
    for name, ppo in starts.items():
        hit, curve = adapt(ppo, target, args.threshold, args.max_iters,
                           args.eval_every, args.num_envs, args.img)
        iters = hit if hit is not None else args.max_iters
        out[name] = dict(hit=hit, iters_used=iters, episodes=round(iters * epi_per_iter),
                         zero_shot=curve[0][1], final=curve[-1][1],
                         reached=hit is not None, curve=curve)
        tag = (f"{hit} iters (~{round(hit*epi_per_iter)} parties)" if hit is not None
               else f">{args.max_iters} iters (NOT reached, final wr {curve[-1][1]})")
        print(f"  {name:12s} zero-shot {curve[0][1]:.2f} -> threshold in {tag}", flush=True)

    v, s, sc = out["variety"], out["single_easy"], out["scratch"]
    speedup = (sc["iters_used"] / v["iters_used"]) if v["reached"] else None
    ok = v["reached"] and v["iters_used"] <= s["iters_used"] and (
        not sc["reached"] or v["iters_used"] * 2 <= sc["iters_used"])
    verdict = (
        f"MORE KNOWLEDGE -> FEWER TRIALS — on a NEW out-of-distribution HARD variant, "
        f"the VARIETY-pretrained general agent reached competence in {v['iters_used']} "
        f"iters (~{v['episodes']} parties, zero-shot already {v['zero_shot']:.2f}), vs "
        f"{'scratch ' + str(sc['iters_used']) + ' iters (~' + str(sc['episodes']) + ' parties)' if sc['reached'] else 'scratch NEVER reached it in ' + str(sc['iters_used']) + ' iters (final ' + str(sc['final']) + ')'}"
        f" and single-easy {s['iters_used']} iters" +
        (f" — a ~{speedup:.0f}x speed-up vs scratch. " if speedup else " — scratch censored. ") +
        f"Prior general knowledge converts directly into sample-efficiency: the "
        f"agent that already abstracted the skill needs far fewer episodes on a "
        f"novel harder instance. (Naive single-instance knowledge helps less — it "
        f"must un-learn its bias.)"
        if ok else
        f"PARTIAL — variety {v['iters_used']}it/{v['episodes']}p reached={v['reached']}, "
        f"single-easy {s['iters_used']}it reached={s['reached']}, scratch "
        f"{sc['iters_used']}it reached={sc['reached']}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v28_fewshot.json"), "w") as f:
        json.dump(dict(target=target, threshold=args.threshold,
                       episodes_per_iter=epi_per_iter, results=out,
                       speedup_vs_scratch=speedup, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
