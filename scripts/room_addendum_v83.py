"""v83 A1 addendum: lifting the Q*bert Interruption (build notes 21).

The recorded verdict (craft_v6_out/v83/room_verdict.json) is Interruption: the replay check started from
the reset base, a state Q*bert does not restore faithfully. Both Q*bert runs are re-executed with the same
agent code (V is unchanged since the A1 commit) and the anchored replay. The Interruption is lifted only
if (1) the re-executed records equal the A1 records exactly (the runs are deterministic, so the A1 numbers
are reproduced) and (2) every replay check at M1, M2 and M* passes from the anchors. The G-room rule is
then read on the A1 numbers, unchanged, and also without Q*bert (robustness).

  python scripts/room_addendum_v83.py
"""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.milestones_v83 import confirm, first_index, ladder   # noqa: E402
from ragnarok.evaluation.pools_v83 import DEV, SEEDS                         # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp              # noqa: E402

OUT = ROOT / "craft_v6_out" / "v83"


def se_proj(rows):
    room = [r for r in rows if r.get("room")]
    if not room:
        return 0, None, None
    s2 = float(np.mean([(np.log(r["C"][0]) - np.log(r["C"][1])) ** 2 / 2 for r in room]))
    return len(room), s2, float(np.sqrt((s2 / 2 + 0.09) / (17 * len(room) / 35)))


def main():
    v = json.load(open(OUT / "room_verdict.json", encoding="utf-8"))
    row = next(r for r in v["rows"] if r["game"] == "qbert")
    lad = ladder(row["B"], row["u"])
    checks, same = [], {}
    for s in SEEDS["room"]:
        a1 = json.load(open(OUT / "room" / f"room_qbert_{s}.json", encoding="utf-8"))
        stem = OUT / "qbert_check" / f"room_qbert_{s}"
        re = json.load(open(str(stem) + ".json", encoding="utf-8"))
        same[s] = dict(records_equal=a1["records"] == re["records"], billed_equal=a1["summary"]["billed"] == re["summary"]["billed"],
                       code_a1=a1["provenance"]["code_sha"], code_rerun=re["provenance"]["code_sha"], rerun_dirty=re["provenance"]["dirty"])
        npz = np.load(str(stem) + ".npz")
        for k in sorted({1, 2, row["k_star"]}):
            i = first_index(re["records"], lad[k])
            got, want, ok = confirm("qbert", s, npz, re["records"], i)
            checks.append(dict(seed=s, k=k, replayed=got, recorded=want, ok=ok))
    lifted = all(x["records_equal"] and x["billed_equal"] and not x["rerun_dirty"] for x in same.values()) \
        and all(c["ok"] for c in checks) and v["interruptions"] == ["qbert"]
    dev = [r for r in v["rows"] if r["game"] in DEV]
    n, s2, se = se_proj(dev)
    n_wo, s2_wo, se_wo = se_proj([r for r in dev if r["game"] != "qbert"])
    passed = n >= 13 and se is not None and se <= 0.23
    out = dict(rule="docs/ARC4_V83_PROTOCOL.md §4 A1; build notes 21",
               qbert_rerun=same, qbert_replay_checks=checks, interruption_lifted=lifted,
               n_room_dev=n, sigma=None if s2 is None else float(np.sqrt(s2)), se_proj=se,
               without_qbert=dict(n_room_dev=n_wo, se_proj=se_wo),
               G_room=("PASS" if passed else "KC-room") if lifted else "Interruption",
               verdict_sha256=full_hash(OUT / "room_verdict.json"),
               senses="score rises and lives given as senses (owner decision 2026-10-22)", provenance=stamp())
    with open(OUT / "room_verdict_addendum.json.tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT / "room_verdict_addendum.json.tmp", OUT / "room_verdict_addendum.json")
    print(json.dumps({k: out[k] for k in ("interruption_lifted", "n_room_dev", "se_proj", "without_qbert", "G_room")}))


if __name__ == "__main__":
    main()
