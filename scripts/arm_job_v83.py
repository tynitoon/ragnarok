"""One v83 arm run (A3, A4, B1): room_job_v83.run with the arm's steering and carried prior.

  python scripts/arm_job_v83.py ARM GAME SEED CAP STOP_AT OUT_STEM [--sealed]

Arms (docs/ARC4_V83_PROTOCOL.md §1): V (no promise model); F (w0 = 0); X (W_-g in development, W_dev
on the sealed pool); P (placebo W_perm,-g / W_perm); X0 (X's W frozen: intercepts only); K (K*, W_same(g),
development only). Only the heads that passed G-pred are carried (gpred_verdict.json); the other head's
w0 is 0. STOP_AT is the evaluator's early stop at M*_g (true cumulative score), set after entry.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.promise import HEADS, Prior, Promise   # noqa: E402
from scripts.room_job_v83 import run                            # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash          # noqa: E402

OUT = ROOT / "craft_v6_out" / "v83"
ARMS = ("V", "F", "X", "P", "X0", "K")


def prior_key(arm, game, sealed):
    if arm in ("X", "X0"):
        return "dev" if sealed else "minus_" + game
    if arm == "P":
        return "perm" if sealed else "perm_minus_" + game
    if arm == "K":
        assert not sealed, "K* is a development arm only"
        return "same_" + game
    return None


def load_prior(key, heads):
    W = np.load(OUT / "promise_W.npz")
    w = {h: W[f"{key}|{h}|w"] for h in HEADS if h in heads}
    return Prior(w=w, mean=W[f"{key}|{HEADS[0]}|mean"], scale=W[f"{key}|{HEADS[0]}|scale"], heads=heads)


def steer_for(arm, game, seed, sealed):
    if arm == "V":
        return None, None, {}
    if arm == "F":
        return Promise(Prior(), seed=seed), None, {}
    heads = tuple(json.load(open(OUT / "gpred_verdict.json", encoding="utf-8"))["carried_heads"])
    key = prior_key(arm, game, sealed)
    info = dict(carried_heads=list(heads), W_sha256=full_hash(OUT / "promise_W.npz"),
                gpred_sha256=full_hash(OUT / "gpred_verdict.json"))
    return Promise(load_prior(key, heads), frozen=(arm == "X0"), seed=seed), key, info


if __name__ == "__main__":
    arm, game, seed, cap, stop_at, stem = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), \
        float(sys.argv[5]), sys.argv[6]
    sealed = "--sealed" in sys.argv
    assert arm in ARMS
    steer, key, info = steer_for(arm, game, seed, sealed)
    o = run(game, seed, cap, stem, stop_at=stop_at, steer=steer,
            extra=dict(arm=arm, prior_key=key, sealed=sealed, **info))
    print(json.dumps(dict(arm=arm, game=game, seed=seed, best=o["records"][-1] if o["records"] else None,
                          billed=o["summary"]["billed"], wall_s=round(o["wall_s"]))))
