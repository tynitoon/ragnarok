"""Strict scorer-only reader for the actual two-frame observation windows."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ragnarok.environments.cold_emu_pool_v70 import ROM_SPECS, CONVENTION


def cold_pooled_fixture_frames(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != "scorer_only_cold_two_raw_frame_fixture" or
            manifest.get("agent_training_allowed") is not False or manifest.get("game") not in ROM_SPECS or manifest.get("reset_convention") != CONVENTION or
            manifest.get("mode") != 0 or manifest.get("action_repeat") != 4):
        raise ValueError("Expected declared scorer-only cold repeat4 window fixtures")
    total = 0
    bootstrap=ROM_SPECS[manifest["game"]][1]
    last_episode, last_local, last_count = 0, 0, bootstrap
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("Shard must remain in the fixture directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Pooled fixture hash mismatch")
        keys = ("raw_previous", "raw_current", "pooled", "ram_previous", "ram_current",
                "episode", "frame_in_episode", "charged_frame", "action")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in keys}
        n = shard["decisions"]
        for key in keys[:3]:
            if data[key].shape != (n, 210, 160, 3) or data[key].dtype != np.uint8:
                raise ValueError("Malformed RGB windows")
        for key in keys[3:5]:
            if data[key].shape != (n, 128) or data[key].dtype != np.uint8:
                raise ValueError("Malformed paired RAM")
        for key in keys[5:]:
            if data[key].shape != (n,) or data[key].dtype.kind not in "iu":
                raise ValueError("Malformed frame/window indices")
        for i in range(n):
            episode, local, count = (int(data[key][i]) for key in ("episode", "frame_in_episode", "charged_frame"))
            if episode not in (last_episode, last_episode + 1) or (total==0 and episode!=0):
                raise ValueError("Non-monotone episode sequence")
            reset_cost = bootstrap if episode != last_episode else 0
            advanced = local if reset_cost else local - last_local
            if not 1 <= advanced <= 4 or count - last_count != advanced + reset_cost:
                raise ValueError("Discontinuous frames or uncharged reset")
            if not np.array_equal(data["pooled"][i], np.maximum(data["raw_previous"][i], data["raw_current"][i])):
                raise ValueError("Saved window is not the actual two-frame max")
            total += 1
            yield dict(index=total, **{key: data[key][i] for key in keys})
            last_episode, last_local, last_count = episode, local, count
    if total != manifest["decisions"]:
        raise ValueError("Pooled manifest decision count mismatch")
    if last_count != manifest["accounting"]["charged_frames"]:
        raise ValueError("Final ledger differs from fixture count")

    if manifest["accounting"]["cold_loads"]!=last_episode+1 or manifest["episode_count"]!=last_episode+1:
        raise ValueError("Cold load count differs from episodes")
