"""Milestone C PILOT (exploratory, no claim): does experience on other shooters make a new shooter
learned faster?

Leave-one-out over the vertical-shooter family. For held-out game h:
  experienced arm E: one policy trained in sequence on the 3 other shooters (weights carried; every
      per-game table rebuilt at entry), then learning h with exploration guided by the policy;
  fresh arms Fa, Fb: new policies (two seeds) learning h with random-repeat exploration.
Every arm learns h at budgets B in BUDGETS (explore iterations, learning rounds scale with it);
each budget is a separate run from the same starting weights. Live score (seed 7, sticky 0.25,
5 games) per (arm, budget); budget 0 = zero-shot play after entry only. Frames counted per run.

Usage: C:/Python314/python.exe -m scripts.slice_transfer_v71 [--workers 3]
"""
import argparse
import copy
import json
import os
from multiprocessing import Pool

FAMILY = ["assault", "carnival", "demon_attack", "name_this_game"]
BUDGETS = [0, 100, 300, 1000]
OUT = "craft_v6_out/v71_transfer"


def pretrain(held):
    import torch
    from ragnarok.agent_v71.learner import run_game
    from ragnarok.agent_v71.policy import Policy
    from ragnarok.agent_v71.me_buttons import EFFECTS
    path = f"{OUT}/E_without_{held}.pt"
    if os.path.exists(path):
        return path
    pol = Policy(len(EFFECTS))
    torch.manual_seed(100)
    logs = []
    for g in [x for x in FAMILY if x != held]:
        res, ag = run_game(g, policy=pol, explore_iters=1500, guided=0.5, rounds=30, eval_games=3)
        pol = ag.policy
        logs.append(res)
    torch.save(pol.net.state_dict(), path)
    json.dump(logs, open(path.replace(".pt", "_pretrain.json"), "w"), indent=1)
    return path


def unit(args):
    held, arm, budget = args
    import torch
    from ragnarok.agent_v71.learner import run_game
    from ragnarok.agent_v71.policy import Policy
    from ragnarok.agent_v71.me_buttons import EFFECTS
    out = f"{OUT}/{held}_{arm}_{budget}.json"
    if os.path.exists(out):
        return json.load(open(out))
    seed = {"E": 0, "Fa": 0, "Fb": 1}[arm]
    torch.manual_seed({"E": 200, "Fa": 201, "Fb": 202}[arm])
    pol = Policy(len(EFFECTS))
    if arm == "E":
        pol.net.load_state_dict(torch.load(f"{OUT}/E_without_{held}.pt"))
    try:
        res, _ = run_game(held, policy=pol, seed=seed, explore_iters=budget, guided=0.5 if arm == "E" else 0.0,
                          rounds=max(0, budget // 33), eval_games=5)
    except Exception as e:  # noqa: BLE001
        import traceback
        res = dict(game=held, error=traceback.format_exc()[-800:])
    res.update(held=held, arm=arm, budget=budget)
    json.dump(res, open(out, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.workers) as p:
        p.map(pretrain, FAMILY)
        res = p.map(unit, [(h, arm, b) for h in FAMILY for arm in ("E", "Fa", "Fb") for b in BUDGETS])
    json.dump(res, open(f"{OUT}/all.json", "w"), indent=1)
    for h in FAMILY:
        print(h)
        for arm in ("E", "Fa", "Fb"):
            row = [r for r in res if r.get("held") == h and r.get("arm") == arm]
            row.sort(key=lambda r: r["budget"])
            print(f"   {arm:3s} " + "  ".join(f"B{r['budget']}: {r.get('live_mean', float('nan')):.0f}"
                                           f" ({(r.get('frames_explore', 0) + r.get('frames_learn', 0)) / 1e6:.2f}M)"
                                           for r in row))


if __name__ == "__main__":
    main()
