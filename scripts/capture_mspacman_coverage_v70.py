"""Committed prospective RGB-only Ms. Pac-Man coverage; development only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION
from scripts.mspacman_coverage_controller_v70 import CoverageController

PROTOCOL = "docs/ARC4_MSPACMAN_COVERAGE_PROTOCOL.json"
SOURCE_PATHS = (
    "scripts/capture_mspacman_coverage_v70.py", "scripts/mspacman_coverage_controller_v70.py", PROTOCOL,
    "ragnarok/evaluation/mspacman_pixels_v70.py", "ragnarok/evaluation/mspacman_ghosts_v70.py",
    "ragnarok/evaluation/mspacman_v70.py", "ragnarok/evaluation/pong_v70.py",
    "ragnarok/environments/emu_pool.py", "ragnarok/environments/frame_budget_v70.py",
    "ragnarok/environments/cold_emu_pool_v70.py",
)


def capture(destination):
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/PROTOCOL).read_text(encoding="utf-8"))
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    commit=git("rev-parse","HEAD")
    for relative in SOURCE_PATHS:
        if git("hash-object",f"--path={relative}",str(root/relative)) != git("rev-parse",f"HEAD:{relative}"):
            raise ValueError(f"Commit the prospective capture sources first: {relative}")
    if destination.exists():
        raise ValueError("Preserve every earlier attempt; existing destination is forbidden")
    destination.mkdir(parents=True)
    # Durable attempt record precedes the first emulator construction.
    attempt=dict(kind="mspacman_coverage_development_attempt",source_commit=commit,source_sha256=hashes,
                 seed=protocol["development_seed"],status="started",
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (destination/"attempt.json").write_text(json.dumps(attempt,indent=2),encoding="utf-8")
    frames=protocol["frames"]
    engine=_ColdPixelEngine(EmulatorConfig(game="ms_pacman",seed=protocol["development_seed"],frame_skip=1,
                mode="live",max_episode_frames=protocol["episode_limit"],max_charged_frames=protocol["max_charged_frames"]))
    controller=CoverageController()
    images,ram,episodes,actions,charges,shards=[],[],[],[],[],[]
    episode=0
    reset_spent=engine.accounting()["reset_emulated_frames"]
    started=time.perf_counter()
    for index in range(1,frames+1):
        if (index-1)%protocol["action_repeat"]==0:
            action=controller.action(engine.observe(raw=True))
        engine.step(action)
        report=engine.accounting()
        if report["reset_emulated_frames"]!=reset_spent:
            episode+=1;reset_spent=report["reset_emulated_frames"]
        images.append(engine.observe(raw=True))
        ram.append(engine._ale.getRAM().copy())  # Recorder only; controller receives RGB.
        episodes.append(episode);actions.append(action);charges.append(report["charged_frames"])
        if len(images)==256 or index==frames:
            path=destination/f"frames_{index-len(images)+1:05d}_{index:05d}.npz"
            np.savez_compressed(path,rgb=np.stack(images),ram=np.stack(ram),
                        episode=np.array(episodes,dtype=np.int32),action=np.array(actions,dtype=np.uint8),
                        charged_frame=np.array(charges,dtype=np.int64))
            shards.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=len(images)))
            images,ram,episodes,actions,charges=[],[],[],[],[]
        if index%4096==0:
            print(f"coverage {index}/{frames}; charged={report['charged_frames']}; episodes={episode+1}",flush=True)
    if hashes!={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Sources changed during capture; preserve the incomplete attempt")
    final=engine.accounting()
    if (final["charged_frames"]!=frames+132*final["cold_loads"] or final["cold_loads"]!=episode+1
            or final["restores"] or final["experiment_frames"]):
        raise RuntimeError("Capture cost identity differs; preserve the incomplete attempt")
    result=dict(kind="scorer_only_cold_raw_development_fixture",game="ms_pacman",mode=0,
                reset_convention=CONVENTION,seed=protocol["development_seed"],frames=frames,
                action_repeat=protocol["action_repeat"],action_schedule="instrumentation_rgb_maze_walker_v1",
                raw_every_real_frame=True,episode_limit=protocol["episode_limit"],sticky_probability=.25,
                sticky_implementation="worker-python-random-per-frame",source_commit=commit,source_sha256=hashes,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),elapsed_seconds=time.perf_counter()-started,
                accounting=final,shards=shards,versions={p:importlib.metadata.version(p) for p in ("ale-py","numpy","scipy")},
                agent_training_allowed=False,reference_training_allowed=False,agent_entry_states_allowed=False,gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=capture(args.out)
    print(json.dumps({k:result[k] for k in ("seed","frames","elapsed_seconds","accounting")},indent=2))
