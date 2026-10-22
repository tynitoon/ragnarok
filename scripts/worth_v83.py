"""v83 A3, WORTH (docs/ARC4_V83_PROTOCOL.md §4 A3; synthesis §2.7), live, development.

Units: the room games of A1 (development and calibration, build notes §15); if more than 12, the
first 12 of a fixed shuffle (seed 9149). Arms V, F, K* (W_same), paired seeds 9135-9138, cap 4M,
early stop at M*_g (A1). PASS if R(K* vs the better of V and F, per game) >= 2.0 with lower bound
>= 1.25 and r_g >= 2 on >= 1/3 of the units; else KC-worth. Flag KC-harm: F more than 1.25x slower
than V (pooled ratio of climbs). Censoring against K*.

  python scripts/worth_v83.py run k/n        python scripts/worth_v83.py verdict
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.climb_stats_v83 import pooled, run_climbs, summary   # noqa: E402
from ragnarok.evaluation.milestones_v83 import ladder                        # noqa: E402
from ragnarok.evaluation.pools_v83 import SEEDS                              # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                         # noqa: E402
from scripts.room_v83 import gate                                            # noqa: E402

OUT = ROOT / "craft_v6_out" / "v83"
CAP = 4_000_000
ARMS = ("V", "F", "K")
SEEDS_A3 = SEEDS["worth_freeze1"]


def units():
    rv = json.load(open(OUT / "room_verdict.json", encoding="utf-8"))
    assert rv["G_room"] == "PASS" and not rv["interruptions"], rv["G_room"]
    gp = json.load(open(OUT / "gpred_verdict.json", encoding="utf-8"))
    assert gp["G_pred"] == "PASS", gp["G_pred"]
    rows = {r["game"]: r for r in rv["rows"]}
    games = rv["room_dev"] + rv["room_cal"]
    if len(games) > 12:
        order = np.random.default_rng(SEEDS["placebo_shift"]).permutation(len(games))
        games = [games[i] for i in order[:12]]
    out = {}
    for g in games:
        r = rows[g]
        lad = ladder(r["B"], r["u"])
        out[g] = dict(B=r["B"], u=r["u"], k_star=r["k_star"], M_star=lad[r["k_star"]])
    return out


def queue():
    U = units()
    jobs = []
    for s in SEEDS_A3:
        for g, u in U.items():
            for a in ARMS:
                stem = OUT / "worth" / f"{a}_{g}_{s}"
                jobs.append((str(stem) + ".json", ["scripts/arm_job_v83.py", a, g, str(s), str(CAP),
                                                    repr(u["M_star"]), str(stem)]))
    return jobs


def run_all(shard, passes=3):
    k, n = (int(x) for x in shard.split("/"))
    os.makedirs(OUT / "worth" / "logs", exist_ok=True)
    for _ in range(passes):
        todo = [(o, a) for i, (o, a) in enumerate(queue()) if i % n == k and not os.path.exists(o)]
        if not todo:
            break
        for out, args in todo:
            if os.path.exists(out):
                continue
            gate()
            name = pathlib.Path(out).stem
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            with open(OUT / "worth" / "logs" / f"{name}.log", "w") as f:
                r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)


def verdict():
    U = units()
    data, code, bad = {}, set(), []
    for g, u in U.items():
        lad = ladder(u["B"], u["u"])
        for a in ARMS:
            C = []
            for s in SEEDS_A3:
                p = OUT / "worth" / f"{a}_{g}_{s}.json"
                if not p.exists():
                    bad.append(str(p.name))
                    continue
                j = json.load(open(p, encoding="utf-8"))
                code.add((j["provenance"]["code_sha"], j["provenance"]["dirty"]))
                if j["arm"] != a or j["seed"] != s or j["cap"] != CAP or j["stop_at"] != u["M_star"]:
                    bad.append(str(p.name))
                C.append(run_climbs([j["records"]], lad, u["k_star"], CAP, carried=(a == "K"))[0])
            data[(g, a)] = C
    complete = not bad
    res = dict(rule="docs/ARC4_V83_PROTOCOL.md §4 A3", units=U, complete=complete, missing_or_off=bad,
               code=sorted(code), senses="score rises and lives given as senses (owner decision 2026-10-22)")
    if complete:
        k_units = {g: (data[(g, "K")], [data[(g, "V")], data[(g, "F")]]) for g in U}
        s = summary(k_units)
        frac = float(np.mean([r >= 2 for r in s["r_g"].values()]))
        passed = s["R"] >= 2.0 and s["lo"] >= 1.25 and frac >= 1 / 3
        harm = pooled({g: (data[(g, "F")], [data[(g, "V")]]) for g in U})[0]
        res.update(R=s["R"], lo=s["lo"], hi=s["hi"], r_g=s["r_g"], frac_r2=frac,
                   KC_harm=bool(harm < 1 / 1.25), R_V_over_F=harm, climbs={f"{g}|{a}": c for (g, a), c in data.items()})
        mixed = len(code) > 1 or any(d for _, d in code)
        res["WORTH"] = "Interruption" if mixed else ("PASS" if passed else "KC-worth")
    else:
        res["WORTH"] = "incomplete"
    res["provenance"] = stamp()
    with open(OUT / "worth_verdict.json.tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    os.replace(OUT / "worth_verdict.json.tmp", OUT / "worth_verdict.json")
    print(json.dumps({k: res.get(k) for k in ("WORTH", "R", "lo", "frac_r2", "KC_harm")}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_all(sys.argv[2])
    elif sys.argv[1] == "verdict":
        verdict()
