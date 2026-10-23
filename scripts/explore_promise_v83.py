"""v83 exploratory analysis after KC-room (docs/ARC4_V83_EXPLORATORY.md; no claim): do promise heads
trained on other games predict GOOD/BAD on a held-out A1 room game better than the placebo and the
per-button base rate? Reuses the train_promise_v83 / gpred_v83 functions with cheaper settings.

  python scripts/explore_promise_v83.py
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.gpred_v83 as gp                       # noqa: E402
import scripts.train_promise_v83 as tp               # noqa: E402
from ragnarok.agent_v83.promise import HEADS         # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp   # noqa: E402

EXP = ROOT / "craft_v6_out" / "v83" / "explore"
HELD = ("alien", "amidar", "bank_heist", "ice_hockey", "jamesbond", "qbert", "breakout", "ms_pacman")


def main():
    prov = stamp()
    EXP.mkdir(parents=True, exist_ok=True)
    tp.PER_RUN = 5000
    tp.STORE = str(EXP / "promise_store")
    if not (EXP / "promise_store" / "meta.npz").exists():
        tp.build_store()                              # reads OUT/room (the imported A1 runs)
    out_w = tp.OUT
    tp.OUT = str(EXP)
    tp.fit_all(list(HELD))                            # writes EXP/promise_W.npz and .json
    tp.OUT = out_w
    X, m = tp.load_store()
    import numpy as np
    W = np.load(EXP / "promise_W.npz")
    res = {h: [gp.game_test(X, m, W, g, h) for g in HELD] for h in HEADS}
    summary = {h: dict(would_pass=sum(r.get("passed", False) for r in res[h]), of=len(HELD),
                       median_auc=float(np.median([r["auc"] for r in res[h] if "auc" in r])) if any("auc" in r for r in res[h]) else None)
               for h in HEADS}
    rep = dict(note="EXPLORATORY, no claim (docs/ARC4_V83_EXPLORATORY.md)", held_out=HELD, per_run=tp.PER_RUN,
               summary=summary, per_game=res, W_sha256=full_hash(EXP / "promise_W.npz"),
               senses="score rises and lives given as senses (owner decision 2026-10-22)", provenance=prov)
    with open(EXP / "explore_promise.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
