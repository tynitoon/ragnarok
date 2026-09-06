"""ARC 3 — the MECHANISM PROBE (ARC3_PLAN.md 4.3), world 8150 (burned), after pretraining, before the
confirmatory. Frozen checkpoints M1..M4 of lineage 0 and a random-init R; weights frozen; store empty.

    (a) FIRST-PROPOSAL LAWFULNESS: over 256 episodes (4 eval episodes x 64 envs per stream goal), the
        fraction of each arm's FIRST combine per episode that is Bond-true (equal tier). PROCEED iff
        r(M4) - r(R) >= 3 * sqrt(r(1-r)/256) pooled.
    (b) reported: outcome-head AUC on 8150's true Bond table, split into cells EXERCISED in the lineage's
        pretraining vs unexercised, beside the roster ceiling (craft_v6_out/v61_roster.json).

Usage: python -m scripts.probe_v61 [--lineage 0] [--world 8150]
"""

import argparse
import json
import os

import numpy as np
import torch

from ragnarok.infrastructure.device import DEVICE
from ragnarok.environments.law_world import sample_laws, make_world, goal_stream, T_MAX
from scripts.pair_store_v61 import PairEnv, MAX_ITEMS, PAIR_I, PAIR_J, PAIR_INDEX
from scripts.pair_net_v61 import ComposerV61, cfg_v61, load_skill, init_seed
from scripts.gate_v61 import law_auc

LAWS_SEED = 31337


@torch.no_grad()
def first_proposals(comp, spec, skill, cfg, laws, goals, n_eps=4):
    """Deterministic acting on fresh grids with an EMPTY store; the first combine of each env-episode."""
    el, tier = spec["el"], spec["tier"]
    lawful, total = 0, 0
    for p, g in enumerate(goals):
        for e in range(n_eps):
            env = PairEnv(cfg["num_envs"], spec, skill, cfg, seed=spec["seed"] + 11 * p + 9 + 100 * e, goal=g)
            obs = env.obs()
            first = torch.full((cfg["num_envs"],), -1, dtype=torch.long, device=DEVICE)
            for _ in range(cfg["macro_budget"]):
                a = comp.act(obs, env=env, deterministic=True)
                is_pair = a >= MAX_ITEMS
                first = torch.where((first < 0) & is_pair, a, first)
                obs, _ = env.step(a)
                if bool((first >= 0).all()):
                    break
            for a_ in first.tolist():
                if a_ < 0:
                    continue
                i, j = int(PAIR_I[a_ - MAX_ITEMS]), int(PAIR_J[a_ - MAX_ITEMS])
                if i >= spec["n_items"] or j >= spec["n_items"]:
                    continue
                total += 1
                lawful += int(tier[i] == tier[j] < T_MAX and laws["Bond"][el[i], el[j]])
    return (lawful / total if total else float("nan")), total


@torch.no_grad()
def auc_split(comp, spec, skill, cfg, laws, exercised):
    """Outcome-head AUC over equal-tier real pairs, split by whether the ELEMENT pair was exercised in
    pretraining."""
    n, el, tier, decoy = spec["n_items"], spec["el"], spec["tier"], spec["decoy"]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if not decoy[i] and not decoy[j] and tier[i] == tier[j] < T_MAX]
    env = PairEnv(len(pairs), spec, skill, cfg, seed=1, goal=spec["target"])
    env.reset(); env.base.inv[:] = 0
    for k, (i, j) in enumerate(pairs):
        env.base.inv[k, i] = 1; env.base.inv[k, j] = 1
    env.base._set_state()
    _, o3, _ = comp.net(env.obs())
    idx = torch.tensor([int(PAIR_INDEX[i, j]) for (i, j) in pairs], device=DEVICE)
    p_law = (1 - torch.softmax(o3[torch.arange(len(pairs), device=DEVICE), idx], -1)[:, 0]).cpu().numpy()
    y = np.array([laws["Bond"][el[i], el[j]] for (i, j) in pairs], dtype=bool)
    ex = np.array([(min(el[i], el[j]), max(el[i], el[j])) in exercised for (i, j) in pairs])

    def auc(m):
        pos, neg = p_law[m & y], p_law[m & ~y]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return dict(all=auc(np.ones_like(y)), exercised=auc(ex), unexercised=auc(~ex),
                n_ex=int(ex.sum()), n_unex=int((~ex).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", type=int, default=0)
    ap.add_argument("--world", type=int, default=8150)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--out-dir", default="craft_v6_out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    cfg = cfg_v61(num_envs=a.num_envs)
    laws = sample_laws(LAWS_SEED)
    skill = load_skill(cfg, 0, a.out_dir)
    spec = make_world(a.world, laws)
    goals = goal_stream(spec)
    pre = json.load(open(os.path.join(a.out_dir, f"v61_pre_s{a.lineage}{sfx}.json")))
    exercised = {tuple(c) for c in pre["exercised"]}
    roster = json.load(open(os.path.join(a.out_dir, "v61_roster.json")))
    print(f"PROBE world {a.world} | stream {goals} | lineage {a.lineage} exercised cells {len(exercised)}")
    arms = {}
    for k in range(1, len(pre["worlds"]) + 1):
        c = ComposerV61(init_seed=0)
        c.net.load_state_dict(torch.load(os.path.join(a.out_dir, f"v61_pre_s{a.lineage}{sfx}_M{k}.pt"), map_location=DEVICE))
        arms[f"M{k}"] = c
    arms["R"] = ComposerV61(init_seed=init_seed(a.lineage, a.world, 1))
    out = dict(world=a.world, lineage=a.lineage, arms={})
    for name, c in arms.items():
        r, n = first_proposals(c, spec, skill, cfg, laws, goals)
        au = auc_split(c, spec, skill, cfg, laws, exercised)
        out["arms"][name] = dict(lawful=r, n=n, auc=au)
        print(f"  {name}: first-proposal lawfulness {r:.3f} (n {n}) | head AUC all {au['all']:.3f} exercised "
              f"{au['exercised']:.3f} ({au['n_ex']}) unexercised {au['unexercised']:.3f} ({au['n_unex']})", flush=True)
    rM, rR = out["arms"][f"M{len(pre['worlds'])}"]["lawful"], out["arms"]["R"]["lawful"]
    nM = out["arms"][f"M{len(pre['worlds'])}"]["n"]
    pooled = 0.5 * (rM + rR)
    se = (pooled * (1 - pooled) / max(nM, 1)) ** 0.5
    passed = (rM - rR) >= 3 * se
    out["summary"] = dict(r_M=rM, r_R=rR, se=se, passed=bool(passed),
                          roster_ceiling=roster, rule="r(M_last) - r(R) >= 3 se pooled")
    print(f"\n  r(M) - r(R) = {rM - rR:+.3f} vs 3 se = {3*se:.3f} -> {'PROCEED' if passed else 'STOP (weights carry no law)'}")
    print(f"  accumulation of the law in the weights: " +
          " ".join(f"{k}={v['lawful']:.3f}" for k, v in out['arms'].items()))
    json.dump(out, open(os.path.join(a.out_dir, f"v61_probe{sfx}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
