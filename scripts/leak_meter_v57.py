"""v57 STEP 1 — LEAK METER. How much of what our agents learned was ever about the goal we asked for?

THE DEFECT (found by the v56 audit, confirmed in code): relabel (hidden_recipe_v55.py) keeps a sample
(s_t, a_t, X) for EVERY item X unlocked at or after t, not just for the goal that was COMMANDED. So every
"from-scratch" control this project has run was fed a free ascending curriculum by its own loss function.
That includes v55's arm E — the sole basis of v55's "memory buys speed, not possibility" deflation.

THIS SCRIPT SPENDS NO GPU. It is pure accounting over the committed v55 result JSONs.

  commanded-goal samples per goal  ~=  S * sum(demos_per_round)
      where a demo is an env-episode that actually unlocked the COMMANDED goal (run_goal records
      int((us[:, goal] >= 0).sum()) per round), and S = sum_{t<=u} gamma^(u-t) -> 1/(1-gamma) = 3.33 at
      gamma=0.7. S is the LIMIT, so S*demos OVERSTATES the commanded share and the leak fraction printed
      below is a LOWER BOUND on the real leak.
  total samples per goal = the buffer growth attributable to that goal (exact for the fresh-per-goal arms
      B and D, whose buffer starts empty at every goal; arm A's buffer is cumulative and is differenced).
  max_samples_per_ep subsampling is uniform over items, so it scales numerator and denominator alike and
      leaves the fraction unbiased.

KILL-2 PRECURSOR (pre-committed in the v57 plan): if the commanded-goal share stays below 5% at depth,
there is no channel through which goal-conditioned recipe knowledge could be learned there, and the
exchange rate has nothing to measure -> END THE LINE.

Usage: python -m scripts.leak_meter_v57
"""

import argparse
import json
import os

GAMMA = 0.7
S = 1.0 / (1.0 - GAMMA)          # 3.333 — the LIMIT, so the leak below is a lower bound


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="craft_v6_out")
    a = ap.parse_args()

    print("=" * 104)
    print("v57 LEAK METER — fraction of collected training samples that was ever about the COMMANDED goal")
    print(f"(gamma={GAMMA}, S={S:.2f} samples per demo — the LIMIT, so every leak figure here is a LOWER bound)")
    print("=" * 104)

    rows_all, out = [], dict(gamma=GAMMA, seeds={})
    for seed in (0, 1, 2):
        p = os.path.join(a.out_dir, f"v55_s{seed}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        pc = {g: c for g, c, _ in d["admitted"]}
        nec = set(d["necessary"])
        print(f"\nSEED {seed} | world {d['world']}")
        print(f"  {'arm':>3} {'goal':>5} {'pc':>3} {'rounds':>7} {'demos':>7} {'cmd-samples':>12} "
              f"{'total':>9} {'COMMANDED share':>16}")
        seed_rows = []
        for arm in ("A", "B", "D"):
            rows = d["arms"].get(arm, {}).get("rows", [])
            prev = 0
            for r in rows:
                dem = sum(r.get("demos_per_round", []))
                tot = (r.get("buf_n", 0) - prev) if arm == "A" else r.get("buf_n", 0)
                prev = r.get("buf_n", 0)
                if tot <= 0:
                    continue
                cmd = S * dem
                share = min(1.0, cmd / tot)
                rec = dict(seed=seed, arm=arm, goal=r["goal"], pc=pc[r["goal"]],
                           necessary=r["goal"] in nec, rounds=r["rounds"], demos=dem,
                           cmd_samples=round(cmd), total=tot, share=round(share, 4))
                seed_rows.append(rec); rows_all.append(rec)
                print(f"  {arm:>3} {r['goal']:>5} {pc[r['goal']]:>3} {r['rounds']:>7} {dem:>7} "
                      f"{round(cmd):>12} {tot:>9} {share*100:>15.2f}%")
        out["seeds"][seed] = seed_rows

    if not rows_all:
        print("no v55 result files found"); return

    print(f"\n{'='*104}\nCOMMANDED-GOAL SHARE BY TASK DEPTH (all seeds, all arms pooled)\n{'='*104}")
    print(f"  {'depth band':>14} {'cells':>6} {'median share':>14} {'max share':>11} {'zero-demo cells':>17}")
    bands = [("pc <= 5", lambda x: x <= 5), ("pc 6-9", lambda x: 6 <= x <= 9),
             ("pc 10-13", lambda x: 10 <= x <= 13), ("pc >= 14", lambda x: x >= 14)]
    band_out = {}
    for label, f in bands:
        sel = [r for r in rows_all if f(r["pc"])]
        if not sel:
            continue
        sh = sorted(r["share"] for r in sel)
        med = sh[len(sh) // 2]
        zero = sum(1 for r in sel if r["demos"] == 0)
        band_out[label] = dict(cells=len(sel), median=med, max=max(sh), zero_demo=zero)
        print(f"  {label:>14} {len(sel):>6} {med*100:>13.2f}% {max(sh)*100:>10.2f}% "
              f"{zero:>13}/{len(sel)}")

    deep = [r for r in rows_all if r["pc"] >= 10]
    deep_med = sorted(r["share"] for r in deep)[len(deep) // 2] if deep else 0.0
    leak_med = 1.0 - deep_med
    out["bands"] = band_out
    out["deep_median_commanded_share"] = round(deep_med, 4)
    out["deep_median_leak"] = round(leak_med, 4)
    out["kill2_precursor"] = bool(deep_med < 0.05)

    print(f"\n{'='*104}")
    print(f"AT DEPTH (pc >= 10, {len(deep)} arm-goal cells): median commanded-goal share "
          f"{deep_med*100:.2f}%  =>  {leak_med*100:.2f}% OF EVERY GRADIENT STEP WAS ABOUT SOMETHING")
    print(f"                                                      ELSE THAN WHAT WE ASKED FOR.")
    if deep_med < 0.05:
        print("\n!! KILL-2 PRECURSOR IS LIVE: below the 5% floor. Under the CURRENT rule there is almost no")
        print("   goal-conditioned channel at depth at all. The v57 gate must show the FIXED rule opens one;")
        print("   if it cannot, the line ends there.")
    else:
        print("\nAbove the 5% floor: a goal-conditioned channel exists at depth under the current rule.")
    print("\nEvery from-scratch control this project ran was therefore trained mostly on incidental")
    print("achievements it happened to stumble into — i.e. on a curriculum it was handed for free.")
    print("v55's arm E, which produced the published 'memory buys speed, not possibility' deflation,")
    print("was never a knowledge-free control.")
    print("=" * 104)
    json.dump(out, open(os.path.join(a.out_dir, "v57_leak_meter.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
