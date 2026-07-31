"""ARC 2, Step 5 — the CONFIRMATORY runner. Built per ARC2_PLAN section 10 step 3.

DOES NOT LAUNCH ITSELF INTO THE CONFIRMATORY. Building and smoking it is the stop point; the run starts
only after the verification pass signs off on the re-frozen scorer (ARC2_PLAN section 10 step 5).

  phase pretrain : train M across the 4 pretrain worlds (weights carry over, store+buffer are per-world),
                   then Mdeg on the degenerate family to MATCHED sample volume. Saves both checkpoints.
  phase test     : for each held-out world, run M (weights FROZEN), F (fresh, trained in-world, equal
                   budget), Z (M's weights, every eval with the store zeroed), G (hand-coded, no
                   learning), F6 (fresh at 6x on <=2 cells F failed) and optionally D (oracle ceiling).
                   Writes craft_v6_out/v58_test_<world>.json in exactly the shape score_v58.py reads.

Every world is built with permute_spec_v58; every arm uses num_envs 64, commanded-only credit, the v56
instrument mask, per-env stores, per-world buffers and the SCORED cost (0 on arrival-mastery, censor cap
on failure). D is a ceiling reference only and never enters a meta arm.

Usage:
  python -m scripts.run_confirm_v58 --phase pretrain [--resume]
  python -m scripts.run_confirm_v58 --phase test     [--resume]
  python -m scripts.run_confirm_v58 --phase test --smoke      # 1/10 scale, code path only
"""

import argparse
import copy
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
from scripts.evidence_store_v58 import evidence_policy
from scripts.evidence_net_v58 import (ComposerV58, BufferV58, StoreEnvV58, permute_spec_v58,
                                      cfg_v58, make_world_env, run_goal_v58, eval_goal_v58)

PRETRAIN_WORLDS = [4000, 4001, 4002, 4003]
DEFAULT_TEST = [6000, 6001, 6002]
SHIFTED_TEST = {7000: dict(n_items=20), 7001: dict(p_resource=0.15)}   # p_tool is not exposed by gen_tree
DEGENERATE = dict(n_items=8, max_inputs=1)
NUM_ENVS = 64
CENSOR_CAP = 3


def load_skill(cfg, seed, out_dir):
    specs = [gen_tree(1000 + i, n_items=14) for i in range(8)]
    net = TechTreeConvNet(cfg["view"], MAX_CELLS, MAX_CELLS, NAV_ACTIONS, broadcast_tail=True)
    ppo = DiscretePPO(nav_env(specs[0], cfg, seed, 2).obs_dim, NAV_ACTIONS, net=net,
                      entropy=cfg["entropy"], gamma=0.99, lam=0.95)
    ppo.net.load_state_dict(torch.load(os.path.join(out_dir, f"v55_skill_s{seed}.pt"),
                                       map_location=DEVICE))
    return ppo


def build(world, **kw):
    """Every world in ARC 2 is built exactly here — permute_spec_v58 always applied."""
    return permute_spec_v58(gen_tree(world, **{**dict(n_items=14), **kw}), world)


def stream_of(spec):
    return [g for g, _, _ in admitted_goals(spec)]


# ---------------------------------------------------------------- phase: pretrain

def pretrain(composer, worlds, skill, cfg, tag, t0, world_kw=None, budget_samples=None):
    """Weights carry across worlds; store and buffer are per-world. Stops early once budget_samples of
    hindsight data have been consumed (used to volume-match Mdeg to M)."""
    total, log = 0, []
    for w in worlds:
        spec = build(w, **(world_kw or {}))
        goals = stream_of(spec)
        buf = BufferV58(cap=1_200_000)                       # per-world, never shared
        env = make_world_env(spec, skill, cfg, seed=w, goal=goals[0])
        for g in goals:
            r = run_goal_v58(env, spec, skill, composer, buf, cfg, w + 11 * goals.index(g), g,
                             train=True)
            total += sum(r["samples_per_round"])
            log.append(dict(world=w, **{k: r[k] for k in
                                        ("goal", "rounds", "cost", "mastered", "master")}))
            print(f"    [{tag}] world {w} goal {g:>2}: master {r['master']:.2f} in {r['rounds']}r "
                  f"({'M' if r['mastered'] else 'x'}) | samples {total} | "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            if budget_samples and total >= budget_samples:
                print(f"    [{tag}] matched sample volume {total} >= {budget_samples} — stop", flush=True)
                return total, log
    return total, log


# ---------------------------------------------------------------- phase: test

@torch.no_grad()
def run_G(spec, skill, cfg, goals, seed):
    """Hand-coded evidence policy — no learning. Headroom reference: what the representation supports."""
    rows = []
    for g in goals:
        env = StoreEnvV58(cfg["num_envs"], spec, skill, cfg, seed=seed + 9, goal=g, hidden=True)
        got = torch.zeros(cfg["num_envs"], dtype=torch.bool, device=DEVICE)
        for ep in range(cfg["r_max"]):
            env.reset(); env.set_goal(g)
            for _ in range(cfg["macro_budget"]):
                env.step(evidence_policy(env, g))
                got |= env.post_unlocked[:, g]
        rows.append(dict(goal=g, master=round(float(got.float().mean()), 3),
                         mastered=bool(float(got.float().mean()) >= cfg["thresh"])))
    return rows


def test_world(w, spec, skill, cfg, cM, cMdeg, out_dir, t0, res, arms, seed, max_goals=0, sfx=""):
    """sfx is '_smoke' for smoke runs so a smoke artifact can NEVER be globbed by score_v58 as if it
    were a confirmatory result — the v53 lesson (a smoke's JSON polluted a confirmatory resume)."""
    jf = os.path.join(out_dir, f"v58_test_{w}{sfx}.json")
    goals = stream_of(spec)
    if max_goals:                       # smoke only: cap the stream so the code path runs quickly
        goals = goals[:max_goals]
    nav = nav_gate(skill, spec, cfg, seed)
    res.setdefault("world", w); res["nav"] = nav; res["goals"] = goals
    res.setdefault("arms", {})
    print(f"\n  WORLD {w} | nav min {min(nav.values()):.3f} | {len(goals)} goals", flush=True)
    if min(nav.values()) < 0.85:
        print("    NAV GATE FAILS — world excluded, recorded as invalid", flush=True)
        res["nav_failed"] = True
        return res

    def arm(tag, composer, train, zero_store_eval=False, r_max=None, goal_list=None):
        if res["arms"].get(tag, {}).get("complete"):
            return res["arms"][tag]["rows"]
        gl = goal_list or goals
        buf = BufferV58(cap=1_200_000)                       # per-world
        env = make_world_env(spec, skill, cfg, seed=w + 7, goal=gl[0])
        rows = []
        for g in gl:
            r = run_goal_v58(env, spec, skill, composer, buf, cfg, w + 11 * goals.index(g), g,
                             r_max=r_max, train=train, zero_store_eval=zero_store_eval)
            rows.append(r)
            res["arms"][tag] = dict(rows=rows, complete=(g == gl[-1]))
            json.dump(res, open(jf, "w"), indent=2)
            print(f"    [{tag}] goal {g:>2}: master {r['master']:.2f} in {r['rounds']}r "
                  f"({'M' if r['mastered'] else 'x'}) cost {r['cost']} "
                  f"disc {(r['discovery'] or {}).get('frac')} | {time.perf_counter()-t0:.0f}s",
                  flush=True)
        return rows

    if "M" in arms:
        arm("M", copy.deepcopy(cM), train=False)                      # FROZEN weights at test
    if "F" in arms:
        arm("F", ComposerV58(), train=True)                           # fresh, equal budget
    if "Z" in arms:
        arm("Z", copy.deepcopy(cM), train=False, zero_store_eval=True)  # M without world knowledge
    if "Mdeg" in arms and cMdeg is not None:
        arm("Mdeg", copy.deepcopy(cMdeg), train=False)                # cheap-transfer isolator
    if "G" in arms and not res["arms"].get("G", {}).get("complete"):
        rows = run_G(spec, skill, cfg, goals, w)
        res["arms"]["G"] = dict(rows=rows, complete=True)
        print(f"    [G] hand-coded: mastered {sum(r['mastered'] for r in rows)}/{len(rows)}", flush=True)
    if "F6" in arms and "F" in res["arms"] and not res["arms"].get("F6", {}).get("complete"):
        failed = [r["goal"] for r in res["arms"]["F"]["rows"] if not r["mastered"]][:2]
        if failed:
            rows = arm("F6", ComposerV58(), train=True, r_max=cfg["r_max"] * 6, goal_list=failed)
            res["arms"]["F6"] = dict(rows=rows, complete=True)
        else:
            res["arms"]["F6"] = dict(rows=[], complete=True, note="F failed no goal")
    json.dump(res, open(jf, "w"), indent=2)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pretrain", "test"), required=True)
    p.add_argument("--seed", type=int, default=0, help="skill seed; also the run seed")
    p.add_argument("--r-max", type=int, default=8)
    p.add_argument("--pretrain-r-max", type=int, default=4)
    p.add_argument("--arms", default="M,F,Z,Mdeg,G,F6")
    p.add_argument("--max-hours", type=float, default=12.0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.r_max, a.pretrain_r_max, a.max_hours = 1, 1, 0.5
    # SMOKE ONLY: shrink the env so the code path runs in seconds. This never touches the frozen
    # cfg_v58 defaults used by calibration and the confirmatory run — a smoke validates plumbing,
    # never numbers, and its outputs are written to *_smoke.* paths.
    n_envs, mb = (16, 12) if a.smoke else (NUM_ENVS, None)
    cfg = cfg_v58(num_envs=n_envs, r_max=a.r_max, censor_cap=CENSOR_CAP)
    if mb:
        cfg["macro_budget"] = mb
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.perf_counter()
    skill = load_skill(cfg, a.seed, a.out_dir)
    sfx = "_smoke" if a.smoke else ""
    mp = os.path.join(a.out_dir, f"v58_M_s{a.seed}{sfx}.pt")
    dp = os.path.join(a.out_dir, f"v58_Mdeg_s{a.seed}{sfx}.pt")
    jp = os.path.join(a.out_dir, f"v58_pretrain_s{a.seed}{sfx}.json")

    if a.phase == "pretrain":
        pcfg = cfg_v58(num_envs=n_envs, r_max=a.pretrain_r_max, censor_cap=CENSOR_CAP)
        if mb:
            pcfg["macro_budget"] = mb
        worlds = PRETRAIN_WORLDS[:1] if a.smoke else PRETRAIN_WORLDS
        print(f"=== PRETRAIN M on {worlds} (weights carry over, store+buffer per world) ===", flush=True)
        cM = ComposerV58()
        nM, logM = pretrain(cM, worlds, skill, pcfg, "M", t0)
        torch.save(cM.net.state_dict(), mp)
        print(f"\n=== PRETRAIN Mdeg on the degenerate family {DEGENERATE}, matched to {nM} samples ===",
              flush=True)
        cD = ComposerV58()
        degw = [9000 + i for i in range(12)]
        nD, logD = pretrain(cD, degw, skill, pcfg, "Mdeg", t0, world_kw=DEGENERATE,
                            budget_samples=nM)
        torch.save(cD.net.state_dict(), dp)
        json.dump(dict(seed=a.seed, M_samples=nM, Mdeg_samples=nD, M_log=logM, Mdeg_log=logD,
                       elapsed_s=round(time.perf_counter() - t0)), open(jp, "w"), indent=2)
        print(f"\nPRETRAIN DONE — M {nM} samples, Mdeg {nD} samples -> {mp}, {dp}", flush=True)
        return

    # ---- test phase --------------------------------------------------------------------------------
    if not os.path.exists(mp):
        print(f"missing {mp} — run --phase pretrain first"); return
    cM = ComposerV58(); cM.net.load_state_dict(torch.load(mp, map_location=DEVICE))
    cMdeg = None
    if os.path.exists(dp):
        cMdeg = ComposerV58(); cMdeg.net.load_state_dict(torch.load(dp, map_location=DEVICE))
    arms = a.arms.split(",")
    todo = (DEFAULT_TEST[:1] if a.smoke else DEFAULT_TEST + list(SHIFTED_TEST))
    for w in todo:
        f = os.path.join(a.out_dir, f"v58_test_{w}{sfx}.json")
        res = json.load(open(f)) if (a.resume and os.path.exists(f)) else {}
        spec = build(w, **SHIFTED_TEST.get(w, {}))
        test_world(w, spec, skill, cfg, cM, cMdeg, a.out_dir, t0, res, arms, a.seed,
                   max_goals=2 if a.smoke else 0, sfx=sfx)
        if time.perf_counter() - t0 > a.max_hours * 3600:
            print(f"  !! {a.max_hours}h cap — state saved, relaunch with --resume", flush=True)
            return
    print(f"\nTEST DONE in {time.perf_counter()-t0:.0f}s — score with: python -m scripts.score_v58",
          flush=True)


if __name__ == "__main__":
    main()
