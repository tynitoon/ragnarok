"""v84 bars B' (docs/ARC4_V84_PROTOCOL_EN.md §7.1), evaluator side, unbilled.

Modes: each constant button (10 episodes per button; the best is the mode value), uniform per step,
held 0.95, held 0.75; 10 episodes each; all lives, sticky 0, 0-30 NOOPs, 7,000 steps, minimal action set,
fresh emulator per episode. Recorded per episode in cumulative sensed rises (sum of max(0, r_step)) and
in true score. B' = max over modes of the mean episode rises; u' = max(0.5 B', median positive per-step
rise); rungs M_k = B' + 2^k u', k = 0..10.

  python scripts/bars_v84.py SEED OUT_DIR GAME [GAME ...]      (one JSON per game)
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.pools_v83 import EXCLUDED, RESERVED   # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp           # noqa: E402
from scripts.bars_v83 import _ale                              # noqa: E402

PATHS = ("ragnarok", "scripts", "tests")


def episode(game, seed, policy, rng, max_steps=7000, frameskip=4):
    ale = _ale(game, seed)
    acts = ale.getMinimalActionSet()
    for _ in range(int(rng.integers(0, 31))):
        for _ in range(frameskip):
            ale.act(acts[0])
    st = {"a": 0}
    true, cum, rises = 0.0, 0.0, []
    for _ in range(max_steps):
        a = policy(rng, st, len(acts))
        r = 0.0
        for _ in range(frameskip):
            r += ale.act(acts[a])
        true += r
        if r > 0:
            cum += r
            rises.append(r)
        if ale.game_over():
            break
    return cum, true, rises


def held(rep):
    def pol(r, st, k):
        if r.random() > rep:
            st["a"] = int(r.integers(k))
        return st["a"]
    return pol


def bar(game, seed, n=10):
    assert game not in RESERVED and game not in EXCLUDED
    rng = np.random.default_rng(seed)
    n_act = len(_ale(game, seed).getMinimalActionSet())
    rises, out = [], {}

    def run(policy):
        c, t = [], []
        for _ in range(n):
            ci, ti, ri = episode(game, seed, policy, rng)
            c.append(ci)
            t.append(ti)
            rises.extend(ri)
        return dict(mean_rise=float(np.mean(c)), mean_true=float(np.mean(t)), eps_rise=c, eps_true=t)

    out["constant"] = [run(lambda r, st, k, b=b: b) for b in range(n_act)]
    best = int(np.argmax([m["mean_rise"] for m in out["constant"]]))
    out["best_constant"] = dict(button=best, **{k: out["constant"][best][k] for k in ("mean_rise", "mean_true")})
    out["uniform"] = run(lambda r, st, k: int(r.integers(k)))
    out["held_095"] = run(held(0.95))
    out["held_075"] = run(held(0.75))
    modes = dict(constant=out["best_constant"]["mean_rise"], uniform=out["uniform"]["mean_rise"],
                 held_095=out["held_095"]["mean_rise"], held_075=out["held_075"]["mean_rise"])
    Bp = max(modes.values())
    med = float(np.median(rises)) if rises else None
    up = max(0.5 * Bp, med) if med is not None else (0.5 * Bp if Bp > 0 else None)
    out.update(modes=modes, B_prime=Bp, median_rise=med, n_rises=len(rises), u_prime=up,
               ladder=None if up is None else [Bp + (2 ** k) * up for k in range(11)])
    return out


if __name__ == "__main__":
    seed, out_dir, games = int(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = stamp((), cwd=str(ROOT), paths=PATHS)
    for g in games:
        p = out_dir / f"{g}_{seed}.json"
        if p.exists():
            continue
        t0 = time.time()
        b = bar(g, seed)
        b.update(game=g, seed=seed, wall_s=time.time() - t0, provenance=prov,
                 provenance_end=stamp((), cwd=str(ROOT), paths=PATHS))
        with open(str(p) + ".tmp", "w", encoding="utf-8") as f:
            json.dump(b, f, indent=1)
        os.replace(str(p) + ".tmp", p)
        print(g, round(b["B_prime"], 2), b["u_prime"], round(b["wall_s"]), flush=True)
