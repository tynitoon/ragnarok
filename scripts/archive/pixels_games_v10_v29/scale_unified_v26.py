"""v26 — SCALE the unified agent: accumulation makes cost grow SUBLINEARLY.

The value of a growing library compounds with scale: over a long stream of game-
encounters, every game already in the library is FREE (recognise -> reuse, no
training). So the unified agent's cumulative learning cost grows only with the
number of DISTINCT games, while a no-memory agent (relearns every encounter)
grows LINEARLY with the stream. We quantify that on a long randomised stream.

Honest scope: with only 3 games this shows the accumulation/cost benefit at
scale + robust recognition over a long stream. Cross-game LEARNING-efficiency
(making a NEW game faster via the library) needs BROAD game variety (the v19
recipe) — a many-game suite — and is the further frontier.

Usage: python -m scripts.scale_unified_v26 [--length 15] [--smoke]
"""

import argparse
import json
import os
import random
import time

from scripts.unified_agent_v25 import UnifiedAgent, ITERS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="craft_v6_out")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        for k in ITERS:
            ITERS[k] = 6
        args.length = 8

    games = list(ITERS.keys())
    rng = random.Random(args.seed)
    stream = [rng.choice(games) for _ in range(args.length)]
    os.makedirs(args.out_dir, exist_ok=True)
    agent = UnifiedAgent()
    print(f"[v26] SCALE | unified agent over a {len(stream)}-game stream "
          f"{stream}", flush=True)
    t0 = time.perf_counter()

    seen, uni_cost, nomem_cost, recog_ok, reuses, curve = set(), 0, 0, 0, 0, []
    for i, name in enumerate(stream):
        nomem_cost += ITERS[name]                    # a no-memory agent relearns each time
        ev = agent.encounter(name)
        if "LEARNED" in ev["action"]:
            uni_cost += ITERS[name]; seen.add(name)
        else:                                        # RECOGNISED -> REUSED
            reuses += 1
            recog_ok += int(ev["action"].split("as ")[1].split(" ")[0] == name)
        curve.append(dict(i=i + 1, game=name, uni_cost=uni_cost, nomem_cost=nomem_cost,
                          lib=ev["lib"]))
        print(f"  [{i+1:>2}/{len(stream)}] {name:9s} -> {ev['action']:32s} | "
              f"unified-cost {uni_cost:>4} vs no-memory {nomem_cost:>4} iters | "
              f"lib={ev['lib']}", flush=True)

    n_distinct = len(set(stream))
    recog_acc = recog_ok / max(1, reuses)
    savings = 1 - uni_cost / max(1, nomem_cost)
    ok = uni_cost == sum(ITERS[g] for g in set(stream)) and recog_acc >= 0.9
    verdict = (f"ACCUMULATION SCALES SUBLINEARLY — over {len(stream)} encounters "
               f"the unified agent paid for only {n_distinct} DISTINCT games "
               f"({uni_cost} iters), vs {nomem_cost} for a no-memory agent "
               f"({savings:.0%} saved); recognition on reuses {recog_acc:.0%}. "
               f"The library's value compounds with scale — known games are free."
               if ok else
               f"PARTIAL — unified {uni_cost}, no-memory {nomem_cost}, recog "
               f"{recog_acc:.0%}, library {sorted(seen)}.")
    print(f"\n  -> {verdict}\n  {time.perf_counter()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, "v26_scale.json"), "w") as f:
        json.dump(dict(stream=stream, distinct=n_distinct, unified_cost=uni_cost,
                       nomemory_cost=nomem_cost, savings=savings,
                       recog_acc=recog_acc, curve=curve, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
