"""Development certification of six fixed-hardware cold-start configurations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import BINARY_SHA256, CONVENTION, ROM_SPECS, _ColdPixelEngine


def check():
    rows = []
    for game in ROM_SPECS:
        for seed in (22000, 22001):
            for replicate in range(3):
                engine = _ColdPixelEngine(EmulatorConfig(game=game, seed=seed, frame_skip=1,
                                                       mode="live", max_episode_frames=64))
                initial_rgb = hashlib.sha256(engine.observe().tobytes()).hexdigest()
                initial_ram = hashlib.sha256(engine._ale.getRAM().tobytes()).hexdigest()
                digest = hashlib.sha256()
                for action in range(64):
                    digest.update(engine.step(action % 18).tobytes())
                rows.append(dict(game=game, seed=seed, replicate=replicate,
                                 initial_rgb_sha256=initial_rgb, initial_ram_sha256=initial_ram,
                                 trajectory_rgb_sha256=digest.hexdigest(), accounting=engine.accounting()))
                del engine
    tests = {}
    for game in ROM_SPECS:
        group = [r for r in rows if r["game"] == game]
        tests[game] = dict(initial_rgb_equal=len({r["initial_rgb_sha256"] for r in group}) == 1,
                           initial_ram_equal=len({r["initial_ram_sha256"] for r in group}) == 1,
                           repeated_trajectories_equal=all(len({r["trajectory_rgb_sha256"] for r in group if r["seed"] == s}) == 1
                                                           for s in (22000,22001)),
                           exact_cost=all(r["accounting"]["charged_frames"] == ROM_SPECS[game][1] + 64 for r in group))
    root = Path(__file__).resolve().parents[1]
    sources = ("ragnarok/environments/cold_emu_pool_v70.py", "ragnarok/environments/emu_pool.py",
               "ragnarok/environments/frame_budget_v70.py", "scripts/check_cold_start_v70.py",
               "docs/ARC4_COLD_START_PROTOCOL.json")
    return dict(kind="development_fixed_hardware_construction_check", convention=CONVENTION,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ale_binary_sha256=BINARY_SHA256, checks=tests, records=rows,
                passed=all(all(checks.values()) for checks in tests.values()),
                total_charged_frames=sum(r["accounting"]["charged_frames"] for r in rows),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                gate_result=None, agent_learning=False,
                limitation="Fixed hardware starts; different episode distribution from ALE natural reset. No scorer geometry calibration outside Pong.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new path; preserve earlier results")
    result = check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ("passed", "checks", "total_charged_frames")}, indent=2))
    if not result["passed"]:
        raise SystemExit("Cold-start certification failed")
