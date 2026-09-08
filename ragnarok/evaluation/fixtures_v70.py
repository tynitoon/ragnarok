"""Strict reader of scorer-only frame/RAM fixtures, never imported by agent code."""
import hashlib
import json
from pathlib import Path

import numpy as np


def fixture_frames(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "scorer_only_development_fixture" or manifest.get("agent_training_allowed") is not False:
        raise ValueError("Not a declared scorer-only development fixture")
    if manifest.get("game") != "pong" or manifest.get("mode") != 0 or manifest.get("raw_every_real_frame") is not True:
        raise ValueError("Expected Pong mode-0 raw frame fixtures")
    total, previous_count, previous_episode = 0, None, None
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("Shard must be a file in the fixture directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Fixture hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            # NpzFile inflates on each lookup: materialise each array once per shard.
            data = {key: archive[key] for key in ("rgb", "ram", "episode", "action", "charged_frame")}
            n = shard["frames"]
            if (data["rgb"].shape != (n, 210, 160, 3) or data["rgb"].dtype != np.uint8 or
                data["ram"].shape != (n, 128) or data["ram"].dtype != np.uint8 or
                any(data[k].shape != (n,) for k in ("episode", "action", "charged_frame"))):
                raise ValueError("Malformed fixture shard")
            for i in range(n):
                episode, count = int(data["episode"][i]), int(data["charged_frame"][i])
                if previous_count is not None:
                    expected = 1 if episode == previous_episode else 73
                    if episode not in (previous_episode, previous_episode + 1) or count - previous_count != expected:
                        raise ValueError("Fixture is not a contiguous raw trajectory with charged resets")
                total += 1
                yield dict(index=total, rgb=data["rgb"][i], ram=data["ram"][i], episode=episode,
                           action=int(data["action"][i]), charged_frame=count)
                previous_count, previous_episode = count, episode
    if total != manifest["frames"]:
        raise ValueError("Fixture manifest frame count mismatch")
