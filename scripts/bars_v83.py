"""v83 bars (docs/ARC4_V83_PROTOCOL.md §3): B_g = max(best constant button, uniform random, random with
repeat 0.95), 10 episodes each, practice conditions (sticky 0, 0-30 NOOPs, 7000 steps, frameskip 4,
a fresh emulator on the bar seed per episode, as eligibility_v81.episode); u_g = max(0.5*|B_g|, median
positive true increment per step over all those episodes). Evaluator side, unbilled.

  python scripts/bars_v83.py SEED OUT.json GAME [GAME ...]
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragnarok.evaluation.pools_v83 import EXCLUDED, RESERVED   # noqa: E402
from ragnarok.evaluation.provenance_v81 import stamp           # noqa: E402

ALEInterface.setLoggerMode(LoggerMode.Error)


def _ale(game, seed):
    ale = ALEInterface()
    ale.setInt("random_seed", int(seed))
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", 1)
    ale.loadROM(roms.get_rom_path(game))
    ale.reset_game()
    return ale


def episode(game, seed, policy, rng, max_steps=7000, frameskip=4):
    """True score and the per-step positive increments of one episode."""
    ale = _ale(game, seed)
    acts = ale.getMinimalActionSet()
    for _ in range(int(rng.integers(0, 31))):
        for _ in range(frameskip):
            ale.act(acts[0])
    st = {"a": 0}
    total, rises = 0.0, []
    for _ in range(max_steps):
        a = policy(rng, st, len(acts))
        r = 0.0
        for _ in range(frameskip):
            r += ale.act(acts[a])
        total += r
        if r > 0:
            rises.append(r)
        if ale.game_over():
            break
    return total, rises


def bar(game, seed, n=10):
    assert game not in RESERVED and game not in EXCLUDED
    rng = np.random.default_rng(seed)
    n_act = len(_ale(game, seed).getMinimalActionSet())
    rises, out = [], {"constant": []}

    def run(policy):
        s = []
        for _ in range(n):
            t, r = episode(game, seed, policy, rng)
            s.append(t)
            rises.extend(r)
        return float(np.mean(s)), [float(x) for x in s]

    for b in range(n_act):
        out["constant"].append(run(lambda r, st, k, b=b: b)[0])

    def rep(r, st, k):
        if r.random() > 0.95:
            st["a"] = int(r.integers(k))
        return st["a"]
    out["uniform"], out["uniform_eps"] = run(lambda r, st, k: int(r.integers(k)))
    out["random_repeat"], out["repeat_eps"] = run(rep)
    out["best_constant"] = dict(button=int(np.argmax(out["constant"])), mean=float(max(out["constant"])))
    B = max(out["best_constant"]["mean"], out["uniform"], out["random_repeat"])
    med = float(np.median(rises)) if rises else None
    out.update(B=B, median_rise=med, n_rises=len(rises),
               u=max(0.5 * abs(B), med) if med is not None else (0.5 * abs(B) if B else None))
    return out


if __name__ == "__main__":
    seed, path, games = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    res = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    for g in games:
        if g in res.get("bars", {}):
            continue
        t0 = time.time()
        b = bar(g, seed)
        b["wall_s"] = time.time() - t0
        res.setdefault("bars", {})[g] = b
        res["seed"], res["provenance"] = seed, stamp()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, path)
        print(g, round(b["B"], 2), b["u"], round(b["wall_s"]), flush=True)
