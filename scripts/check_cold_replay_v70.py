"""Scorer-only adversarial replay after moving a nonterminal snapshot to a new ALE."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS, _ColdPixelEngine


def check_game(game):
    engine = _ColdPixelEngine(EmulatorConfig(game=game, seed=22011, frame_skip=1))
    for step in range(1280):
        engine.step((step // 4) % 18)
    if engine._ale.game_over():
        engine.step(0)
    assert not engine._ale.game_over(), "The tested anchor must be nonterminal"
    anchor = engine.observe(raw=True)
    anchor_ram = engine._ale.getRAM().copy()
    token = engine.clone()
    before = engine.accounting()
    baseline = []
    for step in range(32):
        action = (step + 7) % 18
        pooled = engine.step(action)
        baseline.append((action, pooled, engine.observe(raw=True), engine._ale.getRAM().copy()))
        if engine._ale.game_over():
            break
    assert engine.accounting()["cold_loads"] == before["cold_loads"], "No replacement before the witness"
    previous_native = engine._ale
    engine._cold_start(experimental=True)
    assert engine._ale is not previous_native
    engine.restore(token)
    restored_pixels = np.array_equal(engine.observe(raw=True), anchor)
    restored_ram = np.array_equal(engine._ale.getRAM(), anchor_ram)
    assert not engine._ale.game_over()
    mismatches = []
    baseline_digest, replay_digest = hashlib.sha256(), hashlib.sha256()
    for step, (action, pooled, raw, ram) in enumerate(baseline):
        actual_pooled = engine.step(action)
        actual_raw = engine.observe(raw=True)
        actual_ram = engine._ale.getRAM()
        equal = dict(pooled=np.array_equal(pooled, actual_pooled),
                     raw=np.array_equal(raw, actual_raw), ram=np.array_equal(ram, actual_ram))
        if not all(equal.values()):
            mismatches.append(dict(step=step, **equal))
        for arr in (pooled, raw, ram):
            baseline_digest.update(arr.tobytes())
        for arr in (actual_pooled, actual_raw, actual_ram):
            replay_digest.update(arr.tobytes())
    after = engine.accounting()
    expected = 2 * len(baseline) + ROM_SPECS[game][1] + 60
    exact_cost = after["charged_frames"] - before["charged_frames"] == expected
    no_extra_reset = after["cold_loads"] == before["cold_loads"] + 1
    engine.release(token)
    return dict(game=game, nonterminal_anchor=True, witness_frames=len(baseline),
                immediate_pixels_equal=bool(restored_pixels), immediate_ram_equal=bool(restored_ram),
                mismatches=mismatches, exact_cost=exact_cost, no_extra_reset=no_extra_reset,
                baseline_sha256=baseline_digest.hexdigest(), replay_sha256=replay_digest.hexdigest(),
                passed=bool(restored_pixels and restored_ram and not mismatches and exact_cost and no_extra_reset),
                before=before, after=after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep earlier checks; choose a new path")
    rows = [check_game(game) for game in ROM_SPECS]
    root = Path(__file__).resolve().parents[1]
    sources = ("ragnarok/environments/cold_emu_pool_v70.py", "ragnarok/environments/emu_pool.py",
               "ragnarok/environments/frame_budget_v70.py", "scripts/check_cold_replay_v70.py")
    result = dict(kind="development_nonterminal_cross_instance_replay", records=rows,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  passed=all(r["passed"] for r in rows), gate_result=None,
                  total_charged_frames=sum(r["after"]["charged_frames"] for r in rows),
                  source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                  limitation="One nonterminal anchor per development ROM; does not prove every possible TIA state.")
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed":result["passed"], "games":[{"game":r["game"],"passed":r["passed"],"mismatches":r["mismatches"]} for r in rows]},indent=2))
    if not result["passed"]:
        raise SystemExit("Cross-instance replay failed")
