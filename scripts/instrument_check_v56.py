"""INSTRUMENT CHECK on the v55 outcome variable — NOT a re-score of v55.

v55's committed verdict (NULL, commit c9ca97f, scored by the pre-committed scripts/score_v55.py) STANDS
and is not revisited here. This script asks a separate, narrower question: was the measuring instrument
sound?

THE DEFECT (found by a read-only design audit, provable from the code, not from the results):
eval_goal (hidden_recipe_v55.py) reads the policy with composer.act(deterministic=True) = argmax.
HiddenEnv._set_state builds the ENTIRE observation from (inv>0), unlocked, tried, succ, goal, is_res,
is_valid — there is no grid position, no macro-step counter, no option-internal state. So after a SECOND
consecutive failed attempt on the same item, tried[g] is already 1, succ[g] already 0, and inv/unlocked
are unchanged: the observation is BIT-IDENTICAL to the previous one and argmax necessarily repeats the
same action forever. Deterministic evaluation therefore has an ABSORBING STATE that training (which
samples at temp=1.0 with epsilon=0.05) can never enter. The reported mastery is a lower bound on the
policy's competence, and the size of the gap is unknown a priori.

THE REPAIR (evaluation only; identical for every arm; changes no training, no data, no threshold):
mask out of the argmax any item already attempted-and-failed this episode (tried=1, succ=0).

KILL-A (pre-committed, from the same audit): if the repaired metric changes >= 10 of arm A's mastery
cells, v55's outcome variable was unsound, no v49-v55 mastery comparison may be cited again, and the
follow-up experiment does NOT happen — we publish the instrument defect and close the arc instead.

Also re-measures the nav gate at the horizon that actually binds: v55 certified navigation over
nav_max_steps=40 primitive steps, but HiddenEnv.step gives every option only option_timeout=16.

Usage: python -m scripts.instrument_check_v56
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
from scripts.meta_manager_v51 import MAX_ITEMS, N_FEAT
from scripts.hidden_recipe_v55 import (permute_spec, admitted_goals, nav_gate, HiddenEnv, Composer)

WORLDS = {0: 3002, 1: 3003, 2: 3016}
GOAL_COL = 4


@torch.no_grad()
def eval_goal_repaired(spec, skill, composer, cfg, seed, goal, n=256, repair=True):
    """Identical to hidden_recipe_v55.eval_goal except that, when repair=True, items already
    attempted-and-failed THIS episode are masked out of the argmax (removing the absorbing state)."""
    env = HiddenEnv(n, spec, skill, cfg, seed=seed + 9, goal=goal, hidden=True)
    got = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    obs = env.state
    for _ in range(cfg["macro_budget"]):
        logits, _ = composer.net(obs)
        if repair:
            f = obs.reshape(n, MAX_ITEMS, N_FEAT)
            dead = (f[..., 2] > 0.5) & (f[..., 3] < 0.5)      # tried this episode, and it failed
            logits = logits.masked_fill(dead, -1e9)
        obs, _, _, _, _ = env.step(logits.argmax(-1))
        got |= env.post_unlocked[:, goal]
    return float(got.float().mean())


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
    p.add_argument("--out-dir", default="craft_v6_out")
    a = p.parse_args()
    cfg = dict(num_envs=256, grid=7, view=13, n_resource=4, rollout=32, entropy=0.02, nav_max_steps=40,
               skill_iters=400, option_timeout=16, macro_budget=26, episodes_per_round=4,
               train_steps_per_round=300, max_samples_per_ep=8192, epsilon=0.05, temp=1.0,
               thresh=0.6, r_max=10, skill_stochastic=True, mgr_entropy=0.03, router_iters=0)
    t0 = time.perf_counter()
    out = dict(note="INSTRUMENT CHECK only — v55's committed NULL verdict stands and is not re-scored",
               thresh=cfg["thresh"], seeds={})
    total_flips = 0

    print("=" * 96)
    print("INSTRUMENT CHECK — does removing the argmax absorbing state change v55's mastery readout?")
    print("v55's committed verdict (NULL) is NOT revisited. KILL-A fires at >= 10 flipped arm-A cells.")
    print("=" * 96, flush=True)

    for seed in (0, 1, 2):
        w = WORLDS[seed]
        spec = permute_spec(gen_tree(w, n_items=14), w)
        adm = admitted_goals(spec)
        nec = [g for g, _, b in adm if b > cfg["macro_budget"]]
        skill = load_skill(cfg, seed, a.out_dir)
        cA = Composer("memory")
        cA.net.load_state_dict(torch.load(os.path.join(a.out_dir, f"v55_ckpt_s{seed}.pt"),
                                          map_location=DEVICE)["net"])
        res = json.load(open(os.path.join(a.out_dir, f"v55_s{seed}.json")))
        rowA = {r["goal"]: r for r in res["arms"]["A"]["rows"]}

        # nav gate at the horizon that actually binds every option
        nav16 = nav_gate(skill, spec, dict(cfg, nav_max_steps=cfg["option_timeout"]), seed)
        print(f"\nSEED {seed} | world {w}")
        print(f"  nav gate: 40-step (v55 certified) min {min(res['nav'].values()):.3f} | "
              f"16-step (actually binding) min {min(nav16.values()):.3f}  {nav16}")

        rows, flips = [], 0
        for g, pc, b in adm:
            old = rowA[g]["master"]
            new = eval_goal_repaired(spec, skill, cA, cfg, seed, g, repair=True)
            fl = (new >= cfg["thresh"]) != (old >= cfg["thresh"])
            flips += fl
            rows.append(dict(goal=g, pc=pc, blind=round(b, 1), necessary=g in nec,
                             committed=old, repaired=round(new, 3), delta=round(new - old, 3),
                             flipped=bool(fl)))
            print(f"    goal {g:>2} (pc {pc:>2}{', NEC' if g in nec else '    '}): committed {old:.3f}"
                  f" -> repaired {new:.3f}  ({new-old:+.3f}){'   *** FLIPS MASTERY ***' if fl else ''}",
                  flush=True)
        total_flips += flips
        mo = sum(r["committed"] >= cfg["thresh"] for r in rows)
        mn = sum(r["repaired"] >= cfg["thresh"] for r in rows)
        print(f"  arm A mastery: committed {mo}/{len(rows)} -> repaired {mn}/{len(rows)} | "
              f"{flips} cells flipped | {time.perf_counter()-t0:.0f}s")
        out["seeds"][seed] = dict(world=w, nav40=res["nav"], nav16=nav16, rows=rows, flips=flips,
                                  mastered_committed=mo, mastered_repaired=mn)
        json.dump(out, open(os.path.join(a.out_dir, "v56_instrument_check.json"), "w"), indent=2)

    out["total_flips"] = total_flips
    out["kill_a"] = bool(total_flips >= 10)
    print(f"\n{'='*96}")
    print(f"TOTAL arm-A cells whose MASTERY VERDICT flips: {total_flips} / 31")
    if total_flips >= 10:
        print("!! KILL-A FIRES: v55's outcome variable was UNSOUND. No v49-v55 mastery comparison may be")
        print("   cited again. Publish the instrument defect and close the arc — do NOT run a follow-up.")
    else:
        print("KILL-A does not fire: the committed v55 mastery readout is sound enough to stand.")
        print("The per-cell deltas above are still reported as a known lower-bound bias of the metric.")
    print("=" * 96, flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "v56_instrument_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
