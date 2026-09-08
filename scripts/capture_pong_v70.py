"""Bounded, scorer-only Pong fixtures; actions are fixed without consulting RAM.

This executable owns its emulator; it has no agent endpoint. Every real frame and
reset is charged by the shared engine. Captures are paired at the same raw frame.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw

from ragnarok.environments.emu_pool import EmulatorConfig, _PixelEngine


def capture(destination, *, seed, frames, episode_limit=108_000):
    if destination.exists():
        raise ValueError("Keep previous fixtures; choose a new destination")
    if not 256 <= frames <= 16_384:
        raise ValueError("Fixture length must be in 256..16384")
    destination.mkdir(parents=True)
    engine = _PixelEngine(EmulatorConfig(seed=seed, frame_skip=1, mode="live",
                           max_episode_frames=episode_limit,
                           max_charged_frames=frames * 73 + 72))
    rng = random.Random(seed + 100_000)
    images, ram, episodes, actions, counts = [], [], [], [], []
    selected = {1, 16, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, frames}
    previews, shards = [], []
    episode = 0
    reset_spent = engine.accounting()["reset_emulated_frames"]
    started = time.perf_counter()
    for index in range(1, frames + 1):
        if (index - 1) % 16 == 0:
            action = rng.randrange(18)  # fixed button stream; neither labels nor reward influence it
        pixels = engine.step(action)
        report = engine.accounting()
        if report["reset_emulated_frames"] != reset_spent:
            episode += 1
            reset_spent = report["reset_emulated_frames"]
        raw = engine.observe(raw=True)
        images.append(raw)
        ram.append(engine._ale.getRAM().copy())  # scorer-only fixture executable
        episodes.append(episode)
        actions.append(action)
        counts.append(report["emulated_frames"])
        if index in selected:
            previews.append((index, raw.copy()))
        if len(images) == 256 or index == frames:
            path = destination / f"frames_{index - len(images) + 1:05d}_{index:05d}.npz"
            np.savez_compressed(path, rgb=np.stack(images), ram=np.stack(ram),
                                episode=np.array(episodes, dtype=np.int32),
                                action=np.array(actions, dtype=np.uint8),
                                charged_frame=np.array(counts, dtype=np.int64))
            shards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               frames=len(images)))
            images, ram, episodes, actions, counts = [], [], [], [], []
    sheet = Image.new("RGB", (4 * 176, ((len(previews) + 3) // 4) * 240), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (index, pixels) in enumerate(previews):
        x, y = (i % 4) * 176, (i // 4) * 240
        draw.text((x + 8, y + 5), f"frame {index}", fill="white")
        sheet.paste(Image.fromarray(pixels), (x + 8, y + 24))
    sheet.save(destination / "fixed_frames.png")
    root = Path(__file__).resolve().parents[1]
    source_paths = ["scripts/capture_pong_v70.py", "ragnarok/environments/emu_pool.py",
                    "ragnarok/environments/frame_budget_v70.py"]
    manifest = dict(kind="scorer_only_development_fixture", game="pong", mode=0,
                    seed=seed, action_seed=seed + 100_000, frames=frames,
                    action_repeat=16, raw_every_real_frame=True, episode_limit=episode_limit,
                    sticky_probability=.25, sticky_implementation="worker-python-random-per-frame",
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    elapsed_seconds=time.perf_counter() - started, accounting=engine.accounting(),
                    versions={x: importlib.metadata.version(x) for x in ("ale-py", "numpy")},
                    source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths},
                    shards=shards, agent_training_allowed=False, gate_result=None)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--episode-limit", type=int, default=108_000)
    args = parser.parse_args()
    result = capture(args.out, seed=args.seed, frames=args.frames, episode_limit=args.episode_limit)
    print(json.dumps({k: result[k] for k in ("kind", "seed", "frames", "elapsed_seconds", "accounting")}))
