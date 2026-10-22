"""v83 A4, development rehearsal, freeze 1 (docs/ARC4_V83_PROTOCOL.md §4 A4; synthesis §2.8).

Runs only after A3 WORTH = PASS. Units = A3's. X (W_-g) and P (W_perm,-g) on seeds 9135-9138, paired
with A3's V and F; X0 (X frozen) on 9135-9136 (attribution, reported). Cap 4M, early stop at M*_g.
GO if R(X:F) >= 1.5 with lower bound > 1.1, R(X:V) lower bound > 1.0, R(X:P) lower bound > 1.0 and
R_late(X:F) lower bound > 1.0 (late climb from M2). Censoring against X. Otherwise one declared
freeze 2 (written before its runs) or KC-dev.

  python scripts/rehearsal_v83.py run k/n        python scripts/rehearsal_v83.py verdict
"""
import json
import os
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.climb_stats_v83 import head_start, run_climbs, summary, T1s   # noqa: E402
from ragnarok.evaluation.milestones_v83 import ladder                                 # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp                                  # noqa: E402
from scripts.room_v83 import gate                                                     # noqa: E402
from scripts.worth_v83 import CAP, OUT, SEEDS_A3, units                               # noqa: E402

RUNS = [("X", s) for s in SEEDS_A3] + [("P", s) for s in SEEDS_A3] + [("X0", s) for s in SEEDS_A3[:2]]


def queue():
    wv = json.load(open(OUT / "worth_verdict.json", encoding="utf-8"))
    assert wv["WORTH"] == "PASS", wv["WORTH"]
    U = units()
    jobs = []
    for a, s in RUNS:
        for g, u in U.items():
            stem = OUT / "rehearsal" / f"{a}_{g}_{s}"
            jobs.append((str(stem) + ".json", ["scripts/arm_job_v83.py", a, g, str(s), str(CAP), repr(u["M_star"]), str(stem)]))
    return jobs


def run_all(shard, passes=3):
    k, n = (int(x) for x in shard.split("/"))
    os.makedirs(OUT / "rehearsal" / "logs", exist_ok=True)
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
            with open(OUT / "rehearsal" / "logs" / f"{name}.log", "w") as f:
                r = subprocess.run([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(time.strftime("%H:%M:%S"), "end", name, "exit", r.returncode, flush=True)


def _load(dirname, a, g, seeds, u, code, bad):
    out = []
    for s in seeds:
        p = OUT / dirname / f"{a}_{g}_{s}.json"
        if not p.exists():
            bad.append(p.name)
            continue
        j = json.load(open(p, encoding="utf-8"))
        code.add((j["provenance"]["code_sha"], j["provenance"]["dirty"]))
        if j["arm"] != a or j["seed"] != s or j["cap"] != CAP or j["stop_at"] != u["M_star"]:
            bad.append(p.name)
        out.append(j["records"])
    return out


def verdict():
    U = units()
    code, bad, rec = set(), [], {}
    for g, u in U.items():
        for a in ("V", "F"):
            rec[(g, a)] = _load("worth", a, g, SEEDS_A3, u, code, bad)
        for a in ("X", "P"):
            rec[(g, a)] = _load("rehearsal", a, g, SEEDS_A3, u, code, bad)
        rec[(g, "X0")] = _load("rehearsal", "X0", g, SEEDS_A3[:2], u, code, bad)
    res = dict(rule="docs/ARC4_V83_PROTOCOL.md §4 A4 (freeze 1)", complete=not bad, missing_or_off=bad,
               code=sorted(code), senses="score rises and lives given as senses (owner decision 2026-10-22)")
    if not bad:
        def C(g, a, carried, k_from=1):
            u = U[g]
            return run_climbs(rec[(g, a)], ladder(u["B"], u["u"]), u["k_star"], CAP, carried, k_from)
        xf = summary({g: (C(g, "X", True), [C(g, "F", False)]) for g in U})
        xv = summary({g: (C(g, "X", True), [C(g, "V", False)]) for g in U})
        xp = summary({g: (C(g, "X", True), [C(g, "P", False)]) for g in U})
        late = summary({g: (C(g, "X", True, 2), [C(g, "F", False, 2)]) for g in U})
        x0f = summary({g: (C(g, "X0", True), [C(g, "F", False)]) for g in U})
        go = xf["R"] >= 1.5 and xf["lo"] > 1.1 and xv["lo"] > 1.0 and xp["lo"] > 1.0 and late["lo"] > 1.0
        hs = {g: head_start(T1s(rec[(g, "X")], ladder(U[g]["B"], U[g]["u"])),
                            T1s(rec[(g, "F")], ladder(U[g]["B"], U[g]["u"]))) for g in U}
        mixed = len(code) > 1 or any(d for _, d in code)
        res.update(R_XF=xf, R_XV=xv, R_XP=xp, R_late_XF=late, R_X0F=x0f, head_start=hs,
                   GO="Interruption" if mixed else ("GO" if go else "freeze-2-or-KC-dev"))
    else:
        res["GO"] = "incomplete"
    res["provenance"] = stamp()
    with open(OUT / "rehearsal_verdict.json.tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(OUT / "rehearsal_verdict.json.tmp", OUT / "rehearsal_verdict.json")
    print(json.dumps({k: (res[k]["R"], res[k]["lo"]) if isinstance(res.get(k), dict) and "R" in res[k] else res.get(k)
                      for k in ("GO", "R_XF", "R_XV", "R_XP", "R_late_XF", "R_X0F")}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_all(sys.argv[2])
    elif sys.argv[1] == "verdict":
        verdict()
