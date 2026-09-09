"""Prospective RGB-only instrumentation controller; never an agent component.

The executor alone can access RAM to write evaluator fixtures. Its controller
receives RGB only. No reward, RAM, calibration score or snapshot drives actions.
See docs/ARC4_SEAQUEST_COVERAGE_V4_PROTOCOL.json, fixed before the first capture.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig
from ragnarok.environments.cold_emu_pool_v70 import _ColdPixelEngine, CONVENTION

PROTOCOL = "docs/ARC4_SEAQUEST_COVERAGE_V4_PROTOCOL.json"
SOURCE_PATHS = ("scripts/capture_seaquest_coverage_v4.py", "scripts/seaquest_coverage_controller_v4.py", PROTOCOL,
                "ragnarok/environments/emu_pool.py", "ragnarok/environments/frame_budget_v70.py",
                "ragnarok/environments/cold_emu_pool_v70.py")
from scripts.seaquest_coverage_controller_v4 import CoverageController


def capture(destination, *, partition):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/PROTOCOL).read_text(encoding="utf-8"))
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    seed = protocol[f"{partition}_seed"]
    if partition != "development":
        raise ValueError("Reserved coverage disabled until the full geometry/checker/protocol seal is implemented and committed")
    if destination.exists():
        raise ValueError("Keep previous fixtures; choose a new destination")
    destination.mkdir(parents=True)
    frames = protocol["frames"]
    engine = _ColdPixelEngine(EmulatorConfig(game="seaquest", seed=seed, frame_skip=1, mode="live",
                max_episode_frames=protocol["episode_limit"], max_charged_frames=protocol["max_charged_frames"]))
    controller = CoverageController()
    images, ram, episodes, actions, charges, shards, previews = [], [], [], [], [], [], []
    episode = 0
    reset_spent = engine.accounting()["reset_emulated_frames"]
    started = time.perf_counter()
    for index in range(1, frames+1):
        if (index-1) % protocol["action_repeat"] == 0:
            action = controller.action(engine.observe(raw=True))
        engine.step(action)
        report = engine.accounting()
        if report["reset_emulated_frames"] != reset_spent:
            episode += 1
            reset_spent = report["reset_emulated_frames"]
        raw = engine.observe(raw=True)
        images.append(raw)
        ram.append(engine._ale.getRAM().copy())  # recorder only; never passed to controller
        episodes.append(episode)
        actions.append(action)
        charges.append(report["emulated_frames"])
        if index in (1, 128, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, frames):
            previews.append((index, raw.copy()))
        if len(images) == 256 or index == frames:
            path = destination/f"frames_{index-len(images)+1:05d}_{index:05d}.npz"
            np.savez_compressed(path, rgb=np.stack(images), ram=np.stack(ram),
                                episode=np.array(episodes, dtype=np.int32), action=np.array(actions, dtype=np.uint8),
                                charged_frame=np.array(charges, dtype=np.int64))
            shards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(images)))
            images, ram, episodes, actions, charges = [], [], [], [], []
        if index % 4096 == 0:
            print(f"coverage {index}/{frames}; charged={report['charged_frames']}; episodes={episode+1}", flush=True)
    if hashes != {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}:
        raise RuntimeError("Capture sources changed during the run; preserve this incomplete capture")
    final = engine.accounting()
    if (final["charged_frames"]-128*final["cold_loads"] != frames or
            final["cold_loads"] != episode+1 or final["restores"] or final["experiment_frames"]):
        raise RuntimeError("Ordinary-frame/cold-load accounting identity failed; preserve the incomplete capture")
    sheet = Image.new("RGB", (704, 720), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (index, pixels) in enumerate(previews):
        x, y = i % 4*176, i//4*240
        draw.text((x+8, y+5), f"frame {index}", fill="white")
        sheet.paste(Image.fromarray(pixels), (x+8, y+24))
    sheet.save(destination/"fixed_frames.png")
    manifest = dict(kind="scorer_only_cold_raw_development_fixture", game="seaquest", mode=0,
            reset_convention=CONVENTION, seed=seed, frames=frames, action_repeat=protocol["action_repeat"],
            action_schedule="instrumentation_rgb_controller_v4", raw_every_real_frame=True,
            episode_limit=protocol["episode_limit"], sticky_probability=.25,
            sticky_implementation="worker-python-random-per-frame",
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), elapsed_seconds=time.perf_counter()-started,
            accounting=engine.accounting(), source_sha256=hashes, shards=shards,
            versions={p: importlib.metadata.version(p) for p in ("ale-py", "numpy", "scipy")},
            agent_training_allowed=False, reference_training_allowed=False, agent_entry_states_allowed=False, gate_result=None)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", choices=("development",), default="development")
    args = parser.parse_args()
    result = capture(args.out, partition=args.partition)
    print(json.dumps({k: result[k] for k in ("seed", "frames", "elapsed_seconds", "accounting")}))
