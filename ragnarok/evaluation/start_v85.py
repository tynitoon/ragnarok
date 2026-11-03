"""v85 S0 (docs/ARC4_V85_PROTOCOL_EN.md A.2-A.4), evaluator side, 0 billed frames of any arm.

For unit g and reference run r (A = 9712, B = 9713, v84 G1b logs):
- `top_lineage`: the first record with cum >= M_{k*}, replayed from the run's reset base in the after-step
  context, else post-reset (v84 build note 16); neither matches -> invalid. During that replay, s1 = the
  state right after the first lineage step with cum >= M1 (blob, S(s1), lives, step index, actions and
  per-step cum above s1).
- `entry_cache`: Body.enter on SenseEnv85(g, seed=E), E = 9819 (A) / 9820 (B), rng [E, 0]; the pickled
  Body, its SHA-256 and the entry frames (billed to the start, off every clock).
- `start_identity`: fresh SenseEnv85(g, seed=E), start_from(s1) (reset, NOOP, restore), the lineage
  above s1; every rung M2..M_{k*} must be reached at its recorded step with its recorded cum.
- `counted_rungs`: K_g of A.4 from the reference records only.
Ladder: v84 bars (seed 9706), M_k = B' + 2^k u'.
"""
import hashlib
import json
import pathlib
import pickle
import zlib

import numpy as np

from ragnarok.agent_v71.env import AtariEnv
from ragnarok.agent_v83.entry import Body
from ragnarok.evaluation.senses_v85 import SenseEnv85, make_blob

V84 = pathlib.Path(r"F:\dev\ragnarok_v84\craft_v6_out\v84")
REF_SEEDS = {"A": 9712, "B": 9713}
ENTRY_SEEDS = {"A": 9819, "B": 9820}
MIN_GAP = 50_000


def eligible_units(v84=V84):
    """The 16 room development games with k* >= 4 (§12), as {game: k*}."""
    v = json.load(open(v84 / "room_verdict_v84.json", encoding="utf-8"))
    return {r["game"]: r["k_star"] for r in v["rows"] if r["game"] in v["room_dev"] and r["k_star"] >= 4}


def unit_order(games):
    """A.2: default_rng([9801, 1]).permutation of the eligible units (sorted by name first)."""
    return [str(g) for g in np.random.default_rng([9801, 1]).permutation(sorted(games))]


def ladder(game, v84=V84):
    return json.load(open(v84 / "bars" / f"{game}_9706.json", encoding="utf-8"))["ladder"]


def ref_run(game, seed, v84=V84):
    stem = v84 / "g1b" / f"{game}_{seed}"
    return json.load(open(str(stem) + ".json", encoding="utf-8"))["records"], np.load(str(stem) + ".npz")


def _replay(game, seed, npz, idx, pre_step, m1):
    off = npz["rec_offsets"]
    acts = npz["rec_actions"][off[idx]:off[idx + 1]]
    env = AtariEnv(game, seed=seed)
    if pre_step:
        env.step(0)
    env.ale.restoreState(pickle.loads(zlib.decompress(bytes(npz["base_state"]))))
    env._true_score, cum = 0.0, 0.0
    cums, s1 = np.zeros(len(acts)), None
    for i, a in enumerate(acts):
        before = env._true_score
        env.step(int(a))
        cum += max(0.0, env._true_score - before)
        cums[i] = cum
        if s1 is None and cum >= m1:
            s1 = dict(blob=make_blob(env), S=float(cum), lives=int(env.ale.lives()), step=i)
    return acts, cums, s1


def top_lineage(game, seed, lad, k_star, v84=V84):
    """A.3 steps 1-2. Returns None (invalid start) or the s1 dict with the lineage above it."""
    recs, npz = ref_run(game, seed, v84)
    idx = next(j for j, r in enumerate(recs) if r[1] >= lad[k_star])
    for ctx, pre in (("after_step", True), ("post_reset", False)):
        acts, cums, s1 = _replay(game, seed, npz, idx, pre, lad[1])
        if len(cums) and cums[-1] == recs[idx][1] and s1 is not None:
            i = s1["step"]
            s1.update(context=ctx, record=idx, actions=acts[i + 1:].astype(np.uint8), cum=cums[i + 1:])
            return s1
    return None


def rung_steps(cums, lad, ks):
    """{k: (step index above s1, cum)} of the first step with cum >= M_k, for k in ks."""
    out = {}
    for k in ks:
        j = np.flatnonzero(cums >= lad[k])
        out[k] = (int(j[0]), float(cums[j[0]])) if len(j) else None
    return out


def entry_cache(game, E, path):
    env = SenseEnv85(game, seed=E)
    body = Body()
    entry = body.enter(env, np.random.default_rng([E, 0]), E)
    blob = pickle.dumps(body)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), entry_frames=int(env.billed), entry=entry)


def load_body(path, sha256):
    blob = pathlib.Path(path).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == sha256, "cached Body hash mismatch"
    return pickle.loads(blob)


def start_identity(game, E, s1, lad, k_star):
    """A.3 step 4 on a fresh SenseEnv85: True iff every rung M2..M_{k*} is reached at its recorded step and cum."""
    env = SenseEnv85(game, seed=E)
    env.start_from(s1["blob"], s1["S"])
    cums = np.zeros(len(s1["actions"]))
    for i, a in enumerate(s1["actions"]):
        env.step(int(a))
        cums[i] = env._cum
    ks = range(2, k_star + 1)
    want, got = rung_steps(s1["cum"], lad, ks), rung_steps(cums, lad, ks)
    return dict(ok=bool(want == got), all_steps_equal=bool(np.array_equal(cums, s1["cum"])),
                rungs={int(k): [want[k], got[k]] for k in ks})


def _first(records, m):
    return next(((j, r[0]) for j, r in enumerate(records) if r[1] >= m), (None, None))


def counted_rungs(lad, k_star, S1s, refs):
    """A.4. S1s: S(s1) at every valid start; refs: the reference runs' record lists [[billed, cum, ...]]."""
    keep = []
    for k in range(2, k_star + 1):
        if not all(lad[k] > s for s in S1s):
            continue
        gaps = [_first(rs, lad[k])[1] - _first(rs, lad[1])[1] for rs in refs]
        if float(np.median(gaps)) < MIN_GAP:
            continue
        if keep and all(_first(rs, lad[k])[0] == _first(rs, lad[keep[-1]])[0] for rs in refs):
            continue                                   # same record on both runs: merged, the lowest kept
        keep.append(k)
    return keep
