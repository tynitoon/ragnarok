"""G0 feature store (docs/ARC4_V84_PROTOCOL_EN.md A.2-A.4), zero emulator frames.

For each of the 82 A1 streams (41 games x seeds 9131, 9132): check the file against the A1 manifest,
keep rows with row_lost == False and row_it < len(iter_billed) (rows of the rollout cut by the billed cap
have no clock), draw at most 24,000 of them with default_rng([9701, game_index, seed]) keeping the order,
compute the v83 features (float64, stored float16) and labels, the clock c = iter_billed[row_it] - entry
frames (non-decreasing, else Interruption), blocks row_it // 50 and the deduplication hash
blake2b(view, screen, slot, lives). One file per stream: store/<game>_<seed>.npz.

  python scripts/g0_store_v84.py [workers]
"""
import hashlib
import json
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.agent_v83.logs import labels                          # noqa: E402
from ragnarok.agent_v83.promise import feats                        # noqa: E402
from ragnarok.evaluation.pools_v83 import CALIBRATION, DEV          # noqa: E402
from ragnarok.evaluation.provenance_v81 import full_hash, stamp     # noqa: E402

A1 = pathlib.Path(r"F:\dev\ragnarok_v83\craft_v6_out\v83")
OUT = ROOT / "craft_v6_out" / "v84" / "g0"
STORE = OUT / "store"
GAMES = DEV + CALIBRATION
SEEDS = (9131, 9132)
CAP = 24000
SHAPE = (210, 160)
SEALED = "explore"


def guard(path):
    assert SEALED not in str(path).replace("\\", "/").split("/"), f"sealed path: {path}"
    return path


def row_hash(view, screen, slot, lives):
    h = hashlib.blake2b(digest_size=8)
    h.update(view.tobytes())
    h.update(screen.tobytes())
    h.update(np.int16(slot).tobytes())
    h.update(np.int16(lives).tobytes())
    return np.frombuffer(h.digest(), np.uint64)[0]


def stream_features(z, idx):
    """float64 features of rows idx (the v83 promise.feats call, A.3)."""
    V, S, MX, MY, LV, SL = (z[k] for k in ("row_view", "row_screen", "row_mx", "row_my", "row_lives", "row_slot"))
    X = np.zeros((len(idx), 427), np.float64)
    for j, i in enumerate(idx):
        X[j] = feats(V[i], S[i], False, (float(MX[i]), float(MY[i])), int(LV[i]), int(SL[i]), SHAPE)
    return X


def build_one(args):
    gi, game, seed, manifest = args
    npz = guard(A1 / "room" / f"room_{game}_{seed}.npz")
    js = guard(A1 / "room" / f"room_{game}_{seed}.json")
    for p in (npz, js):
        key = f"room/{p.name}"
        if full_hash(p) != manifest[key]:
            raise RuntimeError(f"Interruption: manifest mismatch for {key}")
    run = json.load(open(js, encoding="utf-8"))
    entry = int(run["entry"]["entry_frames"])
    z = np.load(npz)
    it_b = z["iter_billed"]
    lost, it = z["row_lost"], z["row_it"]
    cut = (~lost) & (it >= len(it_b))
    keep = np.nonzero((~lost) & (it < len(it_b)))[0]
    n_kept0 = len(keep)
    if len(keep) > CAP:
        keep = np.sort(np.random.default_rng([9701, gi, seed]).choice(keep, CAP, replace=False))
    X = stream_features(z, keep)
    clock = it_b[it[keep]].astype(np.int64) - entry
    if np.any(np.diff(clock) < 0):
        raise RuntimeError(f"Interruption: clock decreases in {game} {seed}")
    t, avail = z["row_t"][keep], z["row_avail"][keep]
    yg = labels(z["row_rise_lag"][keep], avail, t, 24).astype(np.int8)
    yb = labels(z["row_drop_lag"][keep], avail, t, 40, stops_on_event=True).astype(np.int8)
    V, S, SL, LV = z["row_view"], z["row_screen"], z["row_slot"], z["row_lives"]
    hashes = np.array([row_hash(V[i], S[i], SL[i], LV[i]) for i in keep], np.uint64)
    out = STORE / f"{game}_{seed}.npz"
    np.savez(out, X=X.astype(np.float16), y_good=yg, y_bad=yb, clock=clock, block=(it[keep] // 50).astype(np.int32),
             it=it[keep].astype(np.int32), t=t, avail=avail, slot=z["row_slot"][keep], lives=z["row_lives"][keep],
             hash=hashes, rows_index=keep.astype(np.int64))
    return dict(game=game, seed=seed, n_rows=int(len(lost)), n_lost=int(lost.sum()), n_cut=int(cut.sum()),
                n_kept_before_cap=int(n_kept0), n_kept=int(len(keep)), entry_frames=entry,
                c_end=int(clock[-1]) if len(clock) else 0, file_sha256=full_hash(out))


def main(workers=2):
    prov = stamp()
    STORE.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(guard(A1 / "a1_manifest.json"), encoding="utf-8"))["files"]
    jobs = [(gi, g, s, manifest) for gi, g in enumerate(GAMES) for s in SEEDS]
    with Pool(workers) as pool:
        info = []
        for r in pool.imap(build_one, jobs):
            info.append(r)
            print(r["game"], r["seed"], r["n_kept"], r["n_cut"], flush=True)
    out = dict(provenance=prov, streams=info, games=list(GAMES), seeds=list(SEEDS), cap=CAP,
               a1_manifest_sha256=full_hash(A1 / "a1_manifest.json"))
    with open(STORE / "store.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("rows", sum(r["n_kept"] for r in info))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
